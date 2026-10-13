import ast
import json
import logging
import re
import traceback as tb
from typing import Annotated, Any, NotRequired, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from agents.review.rev_agent_utils import (
    make_reviewer_system_prompt,
    parse_review_text,
    review_citation_schema_nudge,
    validate_review_citation_schema,
)
from env_setup.config_factory import ToolInfo
from system.llm_retry import invoke_with_content_filter_retry
from system.schema import AgentNetworkMetadata, AgentProfile
from system.trackers import attach_logs, get_logs

log = logging.getLogger(__name__)


class ReviewerState(TypedDict):
    """State container for the reviewer agent graph.

    Attributes:
        messages: Conversation history tracked by LangGraph.
        step: Current iteration counter.
        last_ai_response: Raw text of the most recent model output.
        parsed_review: Structured review fields if parsing succeeded.
        valid_review: Flag indicating whether the parsed review is complete.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    step: int
    last_ai_response: str
    parsed_review: dict[str, Any] | None
    valid_review: bool | None
    metadata: NotRequired[dict[str, Any]]


ACTION_HDR_RE = re.compile(r"(?i)\baction\s*:\s*", re.DOTALL)


def _msg_text(msg: Any) -> str:
    """Best-effort string extraction for message objects."""
    try:
        c = getattr(msg, "content", msg)
        return c if isinstance(c, str) else str(c)
    except Exception:
        return repr(msg)


def _accumulate_usage(log_obj: Any, ai: Any) -> None:
    """Defensively accumulate usage_metadata across provider-specific response shapes."""
    if not log_obj:
        return
    usage = getattr(ai, "usage_metadata", {}) or {}
    log_obj.input_tokens += int(usage.get("input_tokens", 0) or 0)
    log_obj.output_tokens += int(usage.get("output_tokens", 0) or 0)
    log_obj.total_tokens += int(usage.get("total_tokens", 0) or 0)


def _extract_json_like_block(text: str) -> str | None:
    """Extract the first balanced JSON-like `{...}` block following an `Action:` header."""
    text = (text or "").strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1 :]
        if text.endswith("```"):
            text = text[:-3].rstrip()

    m = ACTION_HDR_RE.search(text)
    if not m:
        return None
    rest = text[m.end() :].lstrip()

    brace = 0
    start = None
    in_single = False
    in_double = False
    escape = False
    for i, ch in enumerate(rest):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            continue
        if in_single or in_double:
            continue
        if ch == "{":
            if brace == 0:
                start = i
            brace += 1
        elif ch == "}":
            brace -= 1
            if brace == 0 and start is not None:
                return rest[start : i + 1]
    return None


def parse_action(text: str) -> dict[str, Any] | None:
    """Extract an action dict from a tool call string embedded in the model output.

    Args:
        text: Raw model response text expected to contain an `Action: { ... }` block.

    Returns:
        Parsed JSON/dict action if found and valid (a dict whose ``args``, if any, is a dict or a
        string); otherwise None. A set from mismatched quotes or a bare value counts as invalid:
        the act node reads the action with ``.get``.
    """
    block = _extract_json_like_block(text)
    if not block:
        return None
    try:
        parsed = json.loads(block)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(block)
        except Exception as e:
            err = "".join(tb.format_exception_only(type(e), e)).strip()
            log.debug(f"parse_action: literal_eval failed: {err}")
            return None
    if (
        isinstance(parsed, dict)
        and isinstance(parsed.get("tool", ""), str)
        and isinstance(parsed.get("args", {}), (dict, str))
    ):
        return parsed
    log.debug(f"parse_action: parsed into a {type(parsed).__name__}, not an action dict")
    return None


def get_reviewer_agent(
    llm: BaseChatModel,
    reviewer_agent_profile: AgentProfile,
    agent_network_metadata: AgentNetworkMetadata,
    reviewer_tools: dict[str, ToolInfo] | None = None,
    require_assertion_justification: bool = False,
    metrics_allowlist: list[str] | None = None,
    community_norms: str | None = None,
    include_metrics_guidance: bool = True,
) -> Any:
    """Build the reviewer agent LangGraph with thinking/acting/review validation.

    Args:
        llm: Chat model used for reasoning and tool selection.
        reviewer_agent_profile: Metadata describing the reviewer agent persona.
        agent_network_metadata: Aggregate network stats surfaced in the prompt.
        reviewer_tools: Optional mapping of tool name to ToolInfo objects.
        require_assertion_justification: If true, require every assertion to be justified with archive/arxiv/code evidence.
        metrics_allowlist: Metrics the prompt's guidance lists; None lists every metric.
        community_norms: Reviewer-norm text for the prompt (experiment 09); None for none.
        include_metrics_guidance: Whether the prompt describes the metrics; false when
            ``review.metrics_pipeline`` is off (``system.main``).

    Returns:
        Compiled LangGraph agent ready to be invoked.
    """
    if reviewer_tools is None:
        reviewer_tools = {}

    system_prompt = make_reviewer_system_prompt(
        agent_profile=reviewer_agent_profile,
        agent_network_metadata=agent_network_metadata,
        tools=reviewer_tools,
        require_assertion_justification=require_assertion_justification,
        metrics_allowlist=metrics_allowlist,
        community_norms=community_norms,
        include_metrics_guidance=include_metrics_guidance,
    )

    prompt_structure = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),
            MessagesPlaceholder("messages"),
        ]
    )

    def think_node(state: ReviewerState) -> dict[str, Any]:
        """Generate the next model message given conversation state.

        Args:
            state: Current reviewer agent state.

        Returns:
            Updated state fragment containing the new AI message and response text.
        """
        m = get_logs(agent)
        prompt_val = prompt_structure.invoke({"messages": state["messages"]})
        ai = invoke_with_content_filter_retry(
            llm,
            prompt_val,
            stage_label=f"reviewer_agent.think.agent={getattr(reviewer_agent_profile, 'agent_id', 'unknown')}",
            logger=log,
        )

        if m:
            _accumulate_usage(m, ai)
            m.turns += 1
            meta = state.get("metadata", {}) or {}
            m.add_event(
                kind="think",
                reviewer_id=meta.get("reviewer_id", "unknown"),
                paper_id=meta.get("paper_id", "unknown"),
                iteration=meta.get("iteration"),
                step=state.get("step", 1),
                messages=[_msg_text(msg) for msg in state.get("messages", [])],
                response=_msg_text(ai),
            )

        return {"messages": [ai], "last_ai_response": ai.content}

    def act_node(state: ReviewerState) -> dict[str, Any]:
        """Execute the chosen tool action and surface the observation.

        Args:
            state: Current reviewer agent state.

        Returns:
            State fragment with observation message and incremented step.
        """
        act = parse_action(state["last_ai_response"])  # parse here
        m = get_logs(agent)

        if not act:
            if m:
                m.invalid_actions += 1
            return {"messages": [HumanMessage(content="Observation: Failed action")]}

        tool_name = act.get("tool")
        if not tool_name:
            if m:
                m.invalid_actions += 1
            return {"messages": [HumanMessage(content="Observation: Failed tool")]}

        tool_args = act.get("args", {})
        tool_info = reviewer_tools.get(tool_name)

        if not tool_info:
            if m:
                m.invalid_actions += 1
            return {"messages": [HumanMessage(content="Observation: Failed tool")]}

        tool = tool_info.object  # get tool object

        if m:
            m.tool_calls[tool_name] += 1

        try:
            result = tool.invoke(tool_args)
        except Exception as e:
            result = f"Error: {e!s}"

        return {
            "messages": [HumanMessage(content=f"Observation: {result}")],
            "step": state["step"] + 1,
        }

    def check_final_review_node(state: ReviewerState) -> dict[str, Any]:
        """Validate and, if needed, nudge the model to produce a complete review.

        Args:
            state: Current reviewer agent state.

        Returns:
            State fragment containing parsed review data and validity flag.
        """
        m = get_logs(agent)
        review = state["last_ai_response"] or ""
        parsed_review, is_valid = parse_review_text(review)
        citation_valid, citation_errors = validate_review_citation_schema(
            review,
            require_assertion_justification=require_assertion_justification,
        )

        if not is_valid or not citation_valid:
            # one retry with a targeted nudge
            nudge_lines = []
            if not is_valid:
                nudge_lines.append(
                    "Return the required sections. Your previous message was missing one of the required sections of the review."
                )
            if not citation_valid:
                nudge_lines.append(review_citation_schema_nudge(citation_errors))
            nudge_lines.extend(
                [
                    "Rewrite your review exactly, but include the mandatory sections:",
                    "# Review Text",
                    "(Concise, structured text under sections 1-7)",
                    "# Quantitative Ratings",
                    "- Support: <number>",
                    "- Soundness: <number>",
                    "- Significance: <number>",
                    "- Originality: <number>",
                    "- Overall Recommendation: <number>",
                ]
            )
            nudge = HumanMessage(content="\n".join(nudge_lines).strip())

            prompt_val = prompt_structure.invoke({"messages": state["messages"] + [nudge]})
            retry = invoke_with_content_filter_retry(
                llm,
                prompt_val,
                stage_label=f"reviewer_agent.nudge.agent={getattr(reviewer_agent_profile, 'agent_id', 'unknown')}",
                logger=log,
            )
            _accumulate_usage(m, retry)

            review = retry.content or ""
            parsed_review, is_valid = parse_review_text(review)
            citation_valid, _ = validate_review_citation_schema(
                review,
                require_assertion_justification=require_assertion_justification,
            )

            return {
                "messages": [retry],
                "last_ai_response": retry.content,
                "step": state["step"] + 1,
                "parsed_review": parsed_review,
                "valid_review": bool(is_valid and citation_valid),
            }

        return {
            "parsed_review": parsed_review,
            "valid_review": bool(is_valid and citation_valid),
        }

    graph = StateGraph(ReviewerState)
    graph.add_node("think", think_node)
    graph.add_node("act", act_node)
    graph.add_node("check_final_review", check_final_review_node)
    graph.set_entry_point("think")

    def route_from_think(state: ReviewerState) -> str:
        """Branch to act if an action is present, else validate the review.

        Args:
            state: Current reviewer agent state.

        Returns:
            Next node name.
        """
        if parse_action(state["last_ai_response"]):
            return "act"
        return "check_final_review"

    graph.add_conditional_edges(
        "think",
        route_from_think,
        {"act": "act", "check_final_review": "check_final_review"},
    )

    graph.add_edge("act", "think")

    def route_from_check(state: ReviewerState) -> str:
        """End on a valid review; the step limit is the graph's recursion limit.

        The caller sets it with ``system.graph_limits.recursion_limit_for``.

        Args:
            state: Current reviewer agent state.

        Returns:
            Next node name or END marker.
        """
        return END if state.get("valid_review") else "think"

    graph.add_conditional_edges(
        "check_final_review",
        route_from_check,
        {"think": "think", END: END},
    )
    agent = graph.compile()
    attach_logs(agent, type="review")
    return agent
