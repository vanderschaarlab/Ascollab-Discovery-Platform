import ast
import json
import logging
import re
import traceback as tb
from typing import Annotated, Any, NotRequired, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from agents.research.collab_agent_utils import make_collab_agent_system_prompt
from env_setup.config_factory import ToolInfo
from env_setup.sandbox_utils import (
    extract_sandbox_response,
    get_session_id,
    persist_sandbox_images_to_snapshot,
)
from system.llm_retry import invoke_with_content_filter_retry
from system.schema import AgentNetworkMetadata, AgentProfile
from system.trackers import attach_logs, get_logs

log = logging.getLogger(__name__)


class CollabAgentState(TypedDict):
    """Typed state container for the collaborative agent LangGraph.

    Attributes:
        messages: The running conversation history (model-agnostic message types).
        step: The current iteration count for the agent's think–act loop.
        last_ai_response: The raw text of the most recent AI message (used to
            decide whether to act or finish).
    """

    messages: Annotated[list[AnyMessage], add_messages]
    step: int
    last_ai_response: str
    metadata: NotRequired[dict[str, Any]]
    script: NotRequired[str]


ACTION_HDR_RE = re.compile(r"(?i)\baction\s*:\s*", re.DOTALL)


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


def _normalize_action_block(block: str) -> str:
    block = (block or "").strip()
    if not block:
        return block
    block = block.translate(
        str.maketrans(
            {
                "\u2018": "'",
                "\u2019": "'",
                "\u201c": '"',
                "\u201d": '"',
            }
        )
    )
    block = re.sub(r",\s*([}\]])", r"\1", block)
    return block


def _extract_action_block_by_markers(text: str) -> str | None:
    if not text:
        return None
    # A word-bounded header, as in _extract_json_like_block: "interaction:" or "fraction:" in a
    # report is not an action.
    m = ACTION_HDR_RE.search(text)
    if not m:
        return None
    tail = text[m.end() :]
    stop_idx = len(tail)
    for marker in ("observation:", "final feedback:"):
        pos = tail.lower().find(marker)
        if pos != -1:
            stop_idx = min(stop_idx, pos)
    block = tail[:stop_idx].strip()
    if not block:
        return None
    return block


def parse_action(text: str) -> dict[str, Any] | None:
    """Extract and parse an action payload from model output text.

    The function extracts the first JSON-like block after an ``Action:`` header.
    It tries strict JSON parsing first, then falls back to ``ast.literal_eval``
    for loose quotes. A parse that is not a dict with dict or string ``args`` (a set from
    mismatched quotes, a bare value) is an invalid action: the act node reads it with
    ``.get``.

    Args:
        text: The model output text potentially containing an action block.

    Returns:
        dict | None: A dictionary containing the parsed action (e.g., with
        keys like ``tool`` and ``args``), the ``__invalid_action__`` marker when an action
        does not parse into one, or ``None`` when there is no action.
    """
    parsed = _parse_action_block(text)
    if parsed is None or (
        isinstance(parsed, dict)
        and isinstance(parsed.get("tool", ""), str)
        and isinstance(parsed.get("args", {}), (dict, str))
    ):
        return parsed
    log.warning(
        f"[collab_agent][action_parse] parsed into a {type(parsed).__name__}, not an action dict"
    )
    return {"tool": "__invalid_action__", "args": {"raw": str(text or "")[:500]}}


def _parse_action_block(text: str) -> Any:
    """Parse the action block of `text` as `parse_action` describes, without checking its type."""
    block = _extract_json_like_block(text)
    if not block:
        block = _extract_action_block_by_markers(text or "")
    if not block:
        if isinstance(text, str) and "Action:" in text:
            snippet = text.strip()[:200]
            log.warning(f"[collab_agent][action_parse] no_block snippet={snippet!r}")
            return {"tool": "__invalid_action__", "args": {"raw": snippet}}
        return None
    try:
        return json.loads(block)
    except json.JSONDecodeError:
        pass
    normalized = _normalize_action_block(block)
    if normalized != block:
        try:
            return json.loads(normalized)
        except json.JSONDecodeError:
            pass
    try:
        return ast.literal_eval(block)
    except Exception as e:
        try:
            return ast.literal_eval(normalized)
        except Exception as e2:
            err = "".join(tb.format_exception_only(type(e), e)).strip()
            err2 = "".join(tb.format_exception_only(type(e2), e2)).strip()
            snippet = block[:500].replace("\n", "\\n")
            log.warning(f"[collab_agent][action_parse] literal_eval failed: {err}")
            if normalized != block:
                log.warning(f"[collab_agent][action_parse] normalized_literal_eval failed: {err2}")
            log.warning(f"[collab_agent][action_parse] block_snippet={snippet!r}")
            return {"tool": "__invalid_action__", "args": {"raw": snippet}}


FINAL_FEEDBACK_RE = re.compile(r"Final Feedback:\s*(.*)", re.DOTALL | re.IGNORECASE)


def parse_final_feedback(text: str) -> str:
    """Extract final feedback from model output text.

    If the text contains a section starting with ``Final Feedback:``, this
    function returns the content that follows. Otherwise, it returns the
    original text unchanged.

    Args:
        text: The model output text potentially containing final feedback.

    Returns:
        str: The extracted final feedback if present; otherwise the input text.
    """
    m = FINAL_FEEDBACK_RE.search(text)
    if m:
        feedback_content = m.group(1).strip()
        return feedback_content
    return text


def get_collab_agent(
    llm: BaseChatModel,
    agent_profile: AgentProfile,
    primary_agent_profile: AgentProfile,
    agent_network_metadata: AgentNetworkMetadata,
    tools: dict[str, ToolInfo] | None = None,
    dataset_metadata: dict[str, str] | None = None,
    run_name: str | None = None,
    iteration: int | None = None,
    base_snapshots_dir: str = "snapshots",
) -> Any:
    """Create and compile a think–act LangGraph for a collaborative agent.

    This function constructs a two-node graph:
      - ``think``: Calls the language model with the running message history.
        If an ``Action: {...}`` block is present in the output, the graph
        transitions to ``act``; otherwise it ends and records final feedback.
      - ``act``: Parses the requested tool call and invokes it. The tool's
        result is appended as an observation, and the graph loops back to
        ``think`` with an incremented step counter.

    The system prompt is assembled via ``make_collab_agent_system_prompt`` and
    includes agent profiles, network metadata, available tools, and optional
    dataset metadata. The compiled graph can be invoked using the LangGraph
    runtime.

    Args:
        llm: The chat model to use for the agent's ``think`` step.
        agent_profile: The profile describing this collaborating agent.
        primary_agent_profile: The profile of the primary/orchestrating agent.
        agent_network_metadata: Metadata describing the multi-agent network.
        tools: A mapping from tool name to ``ToolInfo`` objects that can be
            invoked by the agent in the ``act`` node. Defaults to empty dict.
        dataset_metadata: Optional dictionary of dataset-related metadata to
            include in the system prompt.
        run_name: Snapshot run name used for plot persistence.
        iteration: Iteration index used for plot persistence.
        base_snapshots_dir: Snapshot root directory relative to project root.

    Returns:
        Any: A compiled LangGraph app that can be executed with an initial
        ``CollabAgentState``.
    """
    if tools is None:
        tools = {}

    collab_id = getattr(agent_profile, "agent_id", "unknown_collab")

    system_prompt = make_collab_agent_system_prompt(
        agent_profile=agent_profile,
        agent_network_metadata=agent_network_metadata,
        primary_agent=primary_agent_profile,
        tools=tools,
        dataset_metadata=dataset_metadata,
    )

    prompt_structure = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),
            MessagesPlaceholder("messages"),
        ]
    )

    def think_node(state: CollabAgentState) -> dict[str, Any]:
        """Model-thinking node: generate next message and decide to act or finish.

        Invokes the chat model with the current conversation history. If the
        response contains an actionable ``Action: {...}`` block, the raw
        content is stored as ``last_ai_response`` to be consumed by ``act``.
        Otherwise, the function extracts any final feedback and returns it,
        signaling the graph to end.

        Args:
            state: The current agent state with ``messages``, ``step``, and
                ``last_ai_response``.

        Returns:
            dict: Partial state updates including appended message and either
            the raw model output or extracted final feedback under
            ``last_ai_response``.
        """
        m = get_logs(agent)
        prompt_val = prompt_structure.invoke({"messages": state["messages"]})
        ai = invoke_with_content_filter_retry(
            llm,
            prompt_val,
            stage_label=f"collab_agent.think.agent={collab_id}",
            logger=log,
        )
        if m:
            usage = getattr(ai, "usage_metadata", {}) or {}
            m.input_tokens += int(usage.get("input_tokens", 0) or 0)
            m.output_tokens += int(usage.get("output_tokens", 0) or 0)
            m.total_tokens += int(usage.get("total_tokens", 0) or 0)
            m.turns += 1
            m.add_event(
                kind="think",
                step=state.get("step", 1),
            )
        has_action = parse_action(ai.content) is not None
        if isinstance(ai.content, str) and "Action:" in ai.content:
            first_line = ai.content.strip().splitlines()[0]
            log.debug(
                "[collab_agent][action_emit] "
                f"collab_agent_id={collab_id} step={state.get('step', 1)} line={first_line[:200]!r}"
            )

        if not has_action:
            final_feedback = parse_final_feedback(ai.content)
            return {"messages": [ai], "last_ai_response": final_feedback}
        return {"messages": [ai], "last_ai_response": ai.content}

    def act_node(state: CollabAgentState) -> dict[str, Any]:
        """Action-execution node: parse tool call and return observation.

        Parses the ``last_ai_response`` for an ``Action`` payload. If present,
        it looks up the referenced tool, invokes it with provided arguments,
        and appends the result as an ``Observation`` message. If parsing or
        tool resolution fails, an error observation is appended instead.

        Args:
            state: The current agent state including a ``last_ai_response``
                that should contain an ``Action`` block.

        Returns:
            dict: Partial state updates with an appended human ``Observation``
            message and an incremented ``step`` count.
        """
        m = get_logs(agent)
        act = parse_action(state["last_ai_response"])

        if not act:
            return {"messages": [HumanMessage(content="Observation: Failed to parse action")]}

        tool_name = act.get("tool")
        if not tool_name:
            return {"messages": [HumanMessage(content="Observation: Failed action")]}

        tool_args = act.get("args", {})
        if tool_name == "__invalid_action__":
            if m:
                m.invalid_actions += 1
            return {
                "messages": [
                    HumanMessage(
                        content="Observation: Malformed action. Please output a clean Action block only, with no extra text."
                    )
                ],
                "step": state["step"] + 1,
            }
        tool_info = tools.get(tool_name)

        if not tool_info:
            return {"messages": [HumanMessage(content="Observation: Failed tool")]}

        tool = tool_info.object
        if m:
            m.tool_calls[tool_name] += 1
            m.add_event(
                kind="tool_call",
                step=state.get("step", 1),
                tool=tool_name,
                args=tool_args,
            )
        if tool_name == "execute_code":
            code = tool_args.get("code", "")
            log.debug(
                "[collab_agent][execute_code] "
                f"code_len={len(code) if isinstance(code, str) else 'n/a'}"
            )

        try:
            result = tool.invoke(tool_args)
        except Exception as e:
            result = f"Error: {e!s}"

        observation_text = str(result)
        if tool_name == "execute_code":
            meta = state.get("metadata", {}) or {}
            sid_dbg = None
            job_id = None
            cell_no = None
            plot_paths: list[str] = []
            plot_filenames: list[str] = []
            plot_handles: list[str] = []
            plot_citations: list[str] = []
            plot_summary_paths: list[str] = []
            plot_artifacts: list[dict[str, Any]] = []
            image_count = 0
            image_warnings: list[str] = []
            plots_skipped = 0
            manifest_path = None
            summary_manifest_path = None
            plot_summary_backend = None
            plot_summary_stats_completeness = None
            updated_metadata = dict(meta)

            if m:
                m.code_executions += 1

            if isinstance(result, dict):
                sid_dbg = get_session_id(result)
                job_id = result.get("job_id")
                intr = result.get("introspection") or {}
                if not intr and isinstance(result.get("pre_install"), list):
                    intr = (result["pre_install"] or [{}])[-1].get("introspection", {}) or {}
                cell_no = intr.get("cell_no")

                observation_text, _, was_successful = extract_sandbox_response(result)
                if was_successful:
                    persist_summary = persist_sandbox_images_to_snapshot(
                        resp=result,
                        run_name=meta.get("run_name", run_name),
                        iteration=meta.get("iteration", iteration),
                        step=state.get("step"),
                        agent_id=meta.get("agent_id", collab_id),
                        session_id=sid_dbg,
                        job_id=job_id,
                        cell_no=cell_no,
                        base_snapshots_dir=str(
                            meta.get("base_snapshots_dir", base_snapshots_dir) or "snapshots"
                        ),
                        max_images_per_execution=meta.get("max_images_per_execution", None),
                        max_image_bytes=meta.get("max_image_bytes", None),
                        graph_logic_cfg=meta.get("graph_logic", None),
                        code=code,
                    )
                    plot_paths = list(persist_summary.get("plot_paths") or [])
                    plot_filenames = [
                        str(x)
                        for x in (persist_summary.get("plot_filenames") or [])
                        if str(x).strip()
                    ]
                    plot_handles = [
                        str(x)
                        for x in (persist_summary.get("plot_handles") or [])
                        if str(x).strip()
                    ]
                    plot_citations = [
                        str(x)
                        for x in (persist_summary.get("plot_citations") or [])
                        if str(x).strip()
                    ]
                    plot_summary_paths = [
                        str(x)
                        for x in (persist_summary.get("plot_summary_paths") or [])
                        if str(x).strip()
                    ]
                    plot_artifacts = [
                        dict(x)
                        for x in (persist_summary.get("plot_artifacts") or [])
                        if isinstance(x, dict)
                    ]
                    image_count = int(persist_summary.get("image_count", 0) or 0)
                    image_warnings = [str(w) for w in (persist_summary.get("warnings") or [])]
                    plots_skipped = int(persist_summary.get("skipped_count", 0) or 0)
                    manifest_path = persist_summary.get("manifest_path")
                    summary_manifest_path = persist_summary.get("summary_manifest_path")
                    plot_summary_backend = persist_summary.get("plot_summary_backend")
                    plot_summary_stats_completeness = persist_summary.get(
                        "plot_summary_stats_completeness"
                    )
                else:
                    raw_images = result.get("images") or []
                    if isinstance(raw_images, list) and raw_images:
                        plots_skipped = len(raw_images)
                        status_token = (
                            str(result.get("status", "") or "").strip().lower() or "unknown"
                        )
                        image_warnings.append(
                            f"non_success_status:{status_token}:ignored_{len(raw_images)}"
                        )
                if plot_paths:
                    observation_text = (
                        f"{observation_text}\nPLOTS = {json.dumps(plot_paths, ensure_ascii=False)}"
                    )
                if plot_artifacts:
                    observation_text = (
                        f"{observation_text}\n"
                        f"PLOT_ARTEFACTS = {json.dumps(plot_artifacts, ensure_ascii=False)}"
                    )
                    plot_citation_rows = []
                    for artifact in plot_artifacts:
                        if not isinstance(artifact, dict):
                            continue
                        citation = str(artifact.get("plot_citation", "")).strip()
                        handle = str(artifact.get("plot_handle", "")).strip()
                        filename = str(artifact.get("plot_filename", "")).strip()
                        if not citation and not handle and not filename:
                            continue
                        plot_citation_rows.append(
                            {
                                "plot_citation": citation,
                                "plot_handle": handle,
                                "plot_filename": filename,
                            }
                        )
                    if plot_citation_rows:
                        observation_text = (
                            f"{observation_text}\n"
                            f"PLOT_CITATIONS = {json.dumps(plot_citation_rows, ensure_ascii=False)}"
                        )
                    existing = list(updated_metadata.get("plot_artifacts", []) or [])
                    known_pairs = {
                        (
                            str(item.get("plot_filename", "")).strip(),
                            str(item.get("plot_path", "")).strip(),
                        )
                        for item in existing
                        if isinstance(item, dict)
                    }
                    for artifact in plot_artifacts:
                        plot_filename = str(artifact.get("plot_filename", "")).strip()
                        plot_path = str(artifact.get("plot_path", "")).strip()
                        pair = (plot_filename, plot_path)
                        if pair in known_pairs:
                            continue
                        known_pairs.add(pair)
                        existing.append(dict(artifact))
                    updated_metadata["plot_artifacts"] = existing
                    updated_metadata["known_plot_filenames"] = sorted(
                        {
                            str(item.get("plot_filename", "")).strip()
                            for item in existing
                            if isinstance(item, dict) and str(item.get("plot_filename", "")).strip()
                        }
                    )
                    updated_metadata["known_plot_handles"] = sorted(
                        {
                            str(item.get("plot_handle", "")).strip()
                            for item in existing
                            if isinstance(item, dict) and str(item.get("plot_handle", "")).strip()
                        }
                    )
                    updated_metadata["plot_handle_to_filename"] = {
                        str(item.get("plot_handle", "")).strip(): str(
                            item.get("plot_filename", "")
                        ).strip()
                        for item in existing
                        if (
                            isinstance(item, dict)
                            and str(item.get("plot_handle", "")).strip()
                            and str(item.get("plot_filename", "")).strip()
                        )
                    }
                    updated_metadata["plot_citations"] = sorted(
                        {
                            str(item.get("plot_citation", "")).strip()
                            for item in existing
                            if isinstance(item, dict) and str(item.get("plot_citation", "")).strip()
                        }
                    )
            else:
                was_successful = False

            if m:
                if was_successful:
                    m.code_executions_successful += 1
                else:
                    m.code_executions_failed += 1
                if image_count > 0:
                    m.images_created += image_count
                if image_warnings:
                    for warning in image_warnings[:10]:
                        m.add_event(
                            kind="tool_warning_execute_code_images",
                            agent_id=collab_id,
                            step=state.get("step", 1),
                            tool="execute_code",
                            warning=warning,
                        )
                m.add_event(
                    kind="tool_result_execute_code",
                    agent_id=collab_id,
                    step=state.get("step", 1),
                    tool="execute_code",
                    session_id=sid_dbg,
                    job_id=job_id,
                    cell_no=cell_no,
                    status=result.get("status") if isinstance(result, dict) else None,
                    was_successful=was_successful,
                    image_count=image_count,
                    plot_paths=plot_paths,
                    plot_filenames=plot_filenames,
                    plot_handles=plot_handles,
                    plot_citations=plot_citations,
                    plot_summary_paths=plot_summary_paths,
                    plot_artifacts=plot_artifacts,
                    plots_skipped=plots_skipped,
                    plot_manifest=manifest_path,
                    plot_summary_manifest=summary_manifest_path,
                    plot_summary_backend=plot_summary_backend,
                    plot_summary_stats_completeness=plot_summary_stats_completeness,
                    plot_warnings=image_warnings,
                    observation=observation_text,
                )

        return {
            "messages": [HumanMessage(content=f"Observation: {observation_text}")],
            "step": state["step"] + 1,
            "metadata": updated_metadata
            if tool_name == "execute_code"
            else state.get("metadata", {}),
        }

    graph = StateGraph(CollabAgentState)
    graph.add_node("think", think_node)
    graph.add_node("act", act_node)
    graph.set_entry_point("think")

    graph.add_conditional_edges(
        "think",
        lambda s: "act" if parse_action(s["last_ai_response"]) else END,
        {"act": "act", END: END},
    )
    graph.add_edge("act", "think")

    agent = graph.compile()
    attach_logs(agent, type="research")
    return agent
