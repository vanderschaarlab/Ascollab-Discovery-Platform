"""LangGraph agent definition for the LLM-judge ranker.

Graph structure:
    [START] -> think -> route_from_think -> act -> think (loop)
                              |
                              +-> check_verdict -> route_from_check -> END
                                                         |
                                                         +-> think (retry)
"""

import logging
import time
from collections.abc import Callable
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .ranker_schemas import RankerState
from .ranker_utils import (
    NUDGE_MESSAGE,
    parse_action,
    parse_verdict,
)

log = logging.getLogger(__name__)

_TRUNCATE_LEN = 200


def _truncate(text: str, max_len: int = _TRUNCATE_LEN) -> str:
    """Truncate text for log messages."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def _serialize_messages(messages) -> list[dict[str, str]]:
    """Convert LangChain message objects to simple dicts for the trace."""
    result = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            role = "system"
        elif isinstance(msg, HumanMessage):
            role = "human"
        else:
            role = "ai"
        result.append({"role": role, "content": getattr(msg, "content", str(msg))})
    return result


# ---------------------------------------------------------------------------
# LLM retry with exponential backoff
# ---------------------------------------------------------------------------


def _invoke_with_retry(
    llm: BaseChatModel,
    prompt_val: Any,
    retry_cfg: dict[str, Any],
) -> Any:
    """Invoke LLM with exponential backoff on transient errors.

    Args:
        llm: The chat model to invoke.
        prompt_val: The rendered prompt (list of messages).
        retry_cfg: Dict with ``max_retries``, ``base_delay_seconds``,
            ``max_delay_seconds``.

    Returns:
        The AI message response.

    Raises:
        openai.APIError: The last error (rate limits included) once the retries are used up.
    """
    from openai import APIError, RateLimitError

    max_retries = retry_cfg.get("max_retries", 3)
    base_delay = retry_cfg.get("base_delay_seconds", 2.0)
    max_delay = retry_cfg.get("max_delay_seconds", 30.0)

    for attempt in range(max_retries + 1):
        try:
            return llm.invoke(prompt_val)
        except (RateLimitError, APIError) as exc:
            if attempt == max_retries:
                raise
            delay = min(base_delay * (2**attempt), max_delay)
            log.warning(
                "[retry] LLM call failed (%s), retrying in %.1fs (attempt %d/%d)",
                type(exc).__name__,
                delay,
                attempt + 1,
                max_retries,
            )
            time.sleep(delay)


# ---------------------------------------------------------------------------
# Agent graph builder
# ---------------------------------------------------------------------------


def build_ranker_agent(
    llm: BaseChatModel,
    system_prompt: str,
    max_iterations: int = 10,
    tools: dict[str, Any] | None = None,
    llm_retry_cfg: dict[str, Any] | None = None,
    max_parse_retries: int = 2,
    parse_verdict_fn: Callable | None = None,
    nudge_message: str | None = None,
    score_label_fn: Callable | None = None,
) -> CompiledStateGraph:
    """Build the compiled ranker LangGraph agent.

    Args:
        llm: Chat model for the judge.
        system_prompt: Pre-built system prompt string.
        max_iterations: Maximum steps (tool calls and failed verdict checks). Once reached, the
            next reply is checked for a verdict instead of running a tool, and a reply without
            one ends the comparison with ``valid_result=False``.
        tools: Dict mapping tool name to dict with "object" (StructuredTool)
            and "tool_signature" (str). None or empty for no tools.
        llm_retry_cfg: Retry config for LLM calls.
        max_parse_retries: How many nudge retries on parse failure.
        parse_verdict_fn: Verdict parser function. Defaults to
            ``parse_verdict`` (comparison mode).
        nudge_message: Nudge message for parse retries. Defaults to
            ``NUDGE_MESSAGE``.
        score_label_fn: Maps score int to outcome label string. Defaults to
            ``lambda s: "A" if s == 1 else "B"``.

    Returns:
        Compiled LangGraph ready to invoke.
    """
    if tools is None:
        tools = {}
    if llm_retry_cfg is None:
        llm_retry_cfg = {}
    if parse_verdict_fn is None:
        parse_verdict_fn = parse_verdict
    if nudge_message is None:
        nudge_message = NUDGE_MESSAGE
    if score_label_fn is None:
        score_label_fn = lambda s: "A" if s == 1 else "B"  # noqa: E731

    # Tracking counters (mutated by nodes, read by caller)
    tracker = {
        "input_tokens": 0,
        "output_tokens": 0,
        "turns": 0,
        "tool_calls": {},
        "events": [],
    }

    _seq_counter = [0]  # mutable counter for event sequence numbers

    def _next_seq() -> int:
        seq = _seq_counter[0]
        _seq_counter[0] += 1
        return seq

    prompt_structure = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),
            MessagesPlaceholder("messages"),
        ]
    )

    # -----------------------------------------------------------------------
    # Node: think
    # -----------------------------------------------------------------------
    def think_node(state: RankerState) -> dict[str, Any]:
        """Invoke the LLM with the full message history."""
        turn = tracker["turns"] + 1
        log.debug("[think] invoking LLM (turn %d)", turn)

        prompt_val = prompt_structure.invoke({"messages": state["messages"]})
        t0 = time.time()
        ai = _invoke_with_retry(llm, prompt_val, llm_retry_cfg)
        duration = time.time() - t0

        # Track token usage
        usage = getattr(ai, "usage_metadata", None) or {}
        in_tok = usage.get("input_tokens", 0)
        out_tok = usage.get("output_tokens", 0)
        tracker["input_tokens"] += in_tok
        tracker["output_tokens"] += out_tok
        tracker["turns"] += 1

        log.debug("[think] got response (%d input tokens, %d output tokens)", in_tok, out_tok)

        # Trace event
        tracker["events"].append(
            {
                "seq": _next_seq(),
                "ts": time.time(),
                "kind": "llm_call",
                "turn": turn,
                "input_messages": _serialize_messages(prompt_val.to_messages()),
                "output_text": ai.content,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "duration_seconds": round(duration, 3),
            }
        )

        return {"messages": [ai], "last_ai_response": ai.content}

    # -----------------------------------------------------------------------
    # Node: act
    # -----------------------------------------------------------------------
    def act_node(state: RankerState) -> dict[str, Any]:
        """Parse and dispatch a tool action from the LLM response."""
        act = parse_action(state["last_ai_response"])

        if not act:
            return {
                "messages": [HumanMessage(content="Observation: Failed to parse action.")],
                "step": state["step"] + 1,
            }

        tool_name = act.get("tool")
        tool_args = act.get("args", {})
        tool_entry = tools.get(tool_name)

        if not tool_entry:
            log.warning("[act] unknown tool %r requested", tool_name)
            return {
                "messages": [
                    HumanMessage(
                        content=f"Observation: Unknown tool '{tool_name}'. "
                        f"Available tools: {list(tools.keys())}"
                    )
                ],
                "step": state["step"] + 1,
            }

        tool_obj = tool_entry["object"]

        log.info("[act] calling tool %r with args %s", tool_name, _truncate(str(tool_args)))

        # Track tool usage
        tracker["tool_calls"][tool_name] = tracker["tool_calls"].get(tool_name, 0) + 1

        error = False
        t0 = time.time()
        try:
            result = tool_obj.invoke(tool_args)
        except Exception as e:
            result = f"Error: {e}"
            error = True
            log.warning("[act] tool %r raised: %s", tool_name, e)
        duration = time.time() - t0

        if not error:
            log.debug("[act] tool %r returned (%d chars)", tool_name, len(str(result)))

        # Trace event
        tracker["events"].append(
            {
                "seq": _next_seq(),
                "ts": time.time(),
                "kind": "tool_call",
                "tool": tool_name,
                "args": tool_args,
                "result": str(result),
                "duration_seconds": round(duration, 3),
                "error": error,
            }
        )

        return {
            "messages": [HumanMessage(content=f"Observation: {result}")],
            "step": state["step"] + 1,
        }

    # -----------------------------------------------------------------------
    # Node: check_verdict
    # -----------------------------------------------------------------------
    def check_verdict_node(state: RankerState) -> dict[str, Any]:
        """Attempt to parse a final verdict; nudge on failure."""
        response_text = state["last_ai_response"] or ""
        parsed, is_valid = parse_verdict_fn(response_text)

        # Extract outcome label for trace. A parse that returns a dict with no
        # "score" key must NOT be labelled -- score_label_fn(None) would record
        # "B" next to `valid: false`, and the backfill trusts these events.
        outcome_label = None
        if parsed and parsed.get("score") is not None:
            outcome_label = score_label_fn(parsed["score"])

        tracker["events"].append(
            {
                "seq": _next_seq(),
                "ts": time.time(),
                "kind": "verdict_parse",
                "valid": is_valid,
                "outcome": outcome_label,
                "retry_index": 0,
            }
        )

        if is_valid:
            log.info("[verdict] parsed: outcome=%s", outcome_label)
            return {
                "parsed_result": parsed,
                "valid_result": True,
            }

        log.debug("[verdict] parse failed, nudging (retry %d/%d)", 1, max_parse_retries)

        # Nudge retry: send nudge message and try again.
        # `retry_messages` accumulates each (nudge, failed reply) pair so the model
        # can see why its previous attempt was rejected -- both within this loop and,
        # via the return values below, in the outer think -> check_verdict cycle.
        current_step = state.get("step", 0)
        retries_used = 0
        retry_messages: list[AnyMessage] = []

        while not is_valid and retries_used < max_parse_retries:
            nudge_msg = HumanMessage(content=nudge_message)
            prompt_val = prompt_structure.invoke(
                {"messages": state["messages"] + retry_messages + [nudge_msg]}
            )
            t0 = time.time()
            retry_ai = _invoke_with_retry(llm, prompt_val, llm_retry_cfg)
            duration = time.time() - t0

            # Track tokens
            usage = getattr(retry_ai, "usage_metadata", None) or {}
            in_tok = usage.get("input_tokens", 0)
            out_tok = usage.get("output_tokens", 0)
            tracker["input_tokens"] += in_tok
            tracker["output_tokens"] += out_tok
            tracker["turns"] += 1

            # Trace the nudge LLM call
            tracker["events"].append(
                {
                    "seq": _next_seq(),
                    "ts": time.time(),
                    "kind": "llm_call",
                    "turn": tracker["turns"],
                    "input_messages": _serialize_messages(prompt_val.to_messages()),
                    "output_text": retry_ai.content or "",
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "duration_seconds": round(duration, 3),
                }
            )

            response_text = retry_ai.content or ""
            parsed, is_valid = parse_verdict_fn(response_text)
            retries_used += 1
            retry_messages.extend([nudge_msg, retry_ai])

            # Trace verdict parse attempt
            outcome_label = None
            if parsed and parsed.get("score") is not None:
                outcome_label = score_label_fn(parsed["score"])

            tracker["events"].append(
                {
                    "seq": _next_seq(),
                    "ts": time.time(),
                    "kind": "verdict_parse",
                    "valid": is_valid,
                    "outcome": outcome_label,
                    "retry_index": retries_used,
                }
            )

            if is_valid:
                log.info("[verdict] parsed: outcome=%s", outcome_label)
                return {
                    "messages": list(retry_messages),
                    "last_ai_response": response_text,
                    "step": current_step + retries_used,
                    "parsed_result": parsed,
                    "valid_result": True,
                }

            if retries_used < max_parse_retries:
                log.debug(
                    "[verdict] parse failed, nudging (retry %d/%d)",
                    retries_used + 1,
                    max_parse_retries,
                )

        # All retries exhausted. Return the LAST failed exchange so that if
        # route_from_check sends us back to `think`, the model sees a rejected
        # attempt instead of restarting blind -- capped at one pair because this
        # return persists into graph state on every outer iteration.
        log.warning(
            "[verdict] all %d parse retries exhausted at step %d",
            max_parse_retries,
            current_step + retries_used,
        )
        # A failed check always uses up at least one step, so the step limit also ends a judge
        # that never gives a verdict when max_parse_retries is 0.
        return {
            "messages": retry_messages[-2:],
            "last_ai_response": response_text,
            "parsed_result": parsed,
            "valid_result": is_valid,
            "step": current_step + max(retries_used, 1),
        }

    # -----------------------------------------------------------------------
    # Routing
    # -----------------------------------------------------------------------
    def route_from_think(state: RankerState) -> str:
        """Route after think: to act for an action within the step limit, else check verdict."""
        if parse_action(state["last_ai_response"]) and state.get("step", 0) < max_iterations:
            return "act"
        return "check_verdict"

    def route_from_check(state: RankerState) -> str:
        """Route after check: END if valid or max iterations, else think."""
        if state.get("valid_result") or state.get("step", 0) >= max_iterations:
            return END
        return "think"

    # -----------------------------------------------------------------------
    # Build graph
    # -----------------------------------------------------------------------
    graph = StateGraph(RankerState)
    graph.add_node("think", think_node)
    graph.add_node("act", act_node)
    graph.add_node("check_verdict", check_verdict_node)
    graph.set_entry_point("think")

    graph.add_conditional_edges(
        "think",
        route_from_think,
        {"act": "act", "check_verdict": "check_verdict"},
    )

    graph.add_edge("act", "think")

    graph.add_conditional_edges(
        "check_verdict",
        route_from_check,
        {"think": "think", END: END},
    )

    compiled = graph.compile()

    # Attach tracker so caller can read it after invocation
    compiled._ranker_tracker = tracker

    return compiled
