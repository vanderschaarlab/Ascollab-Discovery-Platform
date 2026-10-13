import ast
import itertools
import json
import logging
import random
import re
import traceback as tb
from collections.abc import Mapping
from typing import Annotated, Any, NotRequired, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from agents.research.collab_agent import get_collab_agent
from agents.research.primary_agent_utils import make_primary_agent_system_prompt
from env_setup.config_factory import ToolInfo, tool_signature_line
from env_setup.llama_emulation import is_llama_backend, make_llama_auto_tools_prompt
from env_setup.sandbox_utils import (
    extract_sandbox_response,
    get_session_id,
    make_code_executor_tool,
    # sticky-aware tool factory (you’ll add this to sandbox_utils if not present yet)
    make_code_executor_tool_for_sticky,
    persist_sandbox_images_to_snapshot,
)
from stores import state as _global_state
from system.citation_tags import rewrite_plot_citation_tags, validate_citation_tags
from system.llm_retry import invoke_with_content_filter_retry
from system.paper_schema import make_paper_model
from system.plot_summary_artifacts import normalize_graph_logic_cfg
from system.schema import AgentNetworkMetadata, AgentProfile, PaperMetadataPrivate
from system.trackers import attach_logs, count_loc, get_logs

log = logging.getLogger(__name__)


def _normalize_fixed_workflow_runtime_context(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {"enabled": False}
    assigned = str(raw.get("assigned_collaborator_id", "") or "").strip()
    enabled = bool(raw.get("enabled", False)) and bool(assigned)
    try:
        min_collab_messages = int(raw.get("min_collab_messages", 1) or 0)
    except Exception:
        min_collab_messages = 1
    return {
        "enabled": enabled,
        "assigned_collaborator_id": assigned,
        "require_establish_before_first_code": bool(
            raw.get("require_establish_before_first_code", True)
        ),
        "force_tool_target": bool(raw.get("force_tool_target", True)),
        "disable_agent_registry_tool": bool(raw.get("disable_agent_registry_tool", True)),
        "require_collab_message": bool(raw.get("require_collab_message", True)),
        "min_collab_messages": max(0, min_collab_messages),
        "assignment_policy": str(raw.get("assignment_policy", "") or ""),
    }


def _fixed_workflow_established(state: Mapping[str, Any], ctx: Mapping[str, Any]) -> bool:
    assigned = str(ctx.get("assigned_collaborator_id", "") or "").strip()
    if not assigned:
        return False
    return assigned in {
        str(collab_id).strip()
        for collab_id in (state.get("collab_agent_ids", []) or [])
        if str(collab_id).strip()
    }


def _fixed_workflow_block_observation(tool_name: str, ctx: Mapping[str, Any]) -> str | None:
    if not bool(ctx.get("enabled", False)):
        return None
    assigned = str(ctx.get("assigned_collaborator_id", "") or "").strip()
    if tool_name == "search_agent_registry" and bool(ctx.get("disable_agent_registry_tool", True)):
        return (
            "Observation: Fixed workflow disables agent-registry search. "
            f"The assigned collaborator is {assigned}; use establish_collaboration with that agent."
        )
    return None


def _fixed_workflow_execute_code_block_observation(
    state: Mapping[str, Any],
    ctx: Mapping[str, Any],
) -> str | None:
    if not bool(ctx.get("enabled", False)):
        return None
    if not bool(ctx.get("require_establish_before_first_code", True)):
        return None
    if _fixed_workflow_established(state, ctx):
        return None
    assigned = str(ctx.get("assigned_collaborator_id", "") or "").strip()
    return (
        "Observation: Fixed workflow requires collaboration before code execution. "
        f"Establish collaboration with assigned collaborator {assigned} first."
    )


# —————————————————————————————————————————————————
# Primary Agent State Definition
# —————————————————————————————————————————————————
class PrimaryAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    script: str | None
    step: int
    last_ai_response: str
    collab_agent_ids: list[str]
    metadata: dict[str, Any]
    final_report_valid: NotRequired[bool]
    final_report_errors: NotRequired[list[str]]


# —————————————————————————————————————————————————
# Action Parsing Utilities
# —————————————————————————————————————————————————
# Accepts: Action: {...}
# Also tolerates:
#   Action: ```json { ... } ```
#   action: { ... }
#   Action :     { ... }  (spaces)
ACTION_HDR_RE = re.compile(r"(?i)\baction\s*:\s*", re.DOTALL)


def _extract_json_like_block(text: str) -> str | None:
    """Extract the first balanced JSON-like `{...}` block following an `Action:` header.

    This helper tolerates common LLM formatting variants (e.g., fenced code blocks,
    extra whitespace, lowercase `action:`), strips any surrounding triple backticks,
    locates the `Action:` prefix, and then scans forward to return the first
    balanced curly-brace section.

    Args:
        text: Raw model output that may contain an action payload.

    Returns:
        Optional[str]: The substring containing a balanced `{...}` block if found;
        otherwise `None`.
    """
    # strip code fences if present
    text = text.strip()
    if text.startswith("```"):
        # remove first fence
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1 :]
        if text.endswith("```"):
            text = text[:-3].rstrip()

    # Find the first '{' after "Action:"
    m = ACTION_HDR_RE.search(text)
    if not m:
        return None
    rest = text[m.end() :].lstrip()

    # Find a balanced {...} block, ignoring braces inside quoted strings.
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


_SMART_TO_ASCII = {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'}


def _normalize_action_block(block: str, *, ascii_quotes: bool = False) -> str:
    """Repair delimiter-level formatting noise in a raw action block.

    Trailing commas before closers are always removed. Smart quotes are rewritten to
    ASCII only with ``ascii_quotes=True``: when the model used them as delimiters
    that is what makes the block parse, but when they belong to prose ("Cohen's d")
    rewriting them would end the surrounding string early. ``parse_action`` tries
    each parser on the block as written before the normalized forms.

    Args:
        block: Raw `{...}` action block extracted from model output.
        ascii_quotes: Also rewrite smart quotes to ASCII quotes.

    Returns:
        str: The normalized block.
    """
    block = (block or "").strip()
    if not block:
        return block
    if ascii_quotes:
        block = block.translate(str.maketrans(_SMART_TO_ASCII))
    block = re.sub(r",\s*([}\]])", r"\1", block)
    return block


def _action_block_candidates(block: str) -> list[str]:
    """The block as written, then with trailing commas removed, then with ASCII quotes."""
    candidates = [block]
    for candidate in (
        _normalize_action_block(block),
        _normalize_action_block(block, ascii_quotes=True),
    ):
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


# A quote inside a string body only ends that string when the next non-space
# character can legally follow a value.
_VALUE_TERMINATORS = ",:}]"


def _repair_unescaped_quotes(block: str) -> str:
    """Escape stray quote characters that appear inside string bodies.

    Models routinely emit Python-style blocks whose prose or code contains an
    unescaped apostrophe (``'the cell's state'``), which makes the block invalid
    for both `json.loads` and `ast.literal_eval`. This walks the block tracking
    string state and escapes any delimiter-matching quote that is not followed by
    a character which could legally terminate a value.

    Args:
        block: Action block that failed strict parsing.

    Returns:
        str: The block with stray in-string quotes escaped.
    """
    out: list[str] = []
    quote: str | None = None
    i = 0
    length = len(block)
    while i < length:
        char = block[i]
        if quote is None:
            if char in "\"'":
                quote = char
            out.append(char)
            i += 1
            continue
        if char == "\\" and i + 1 < length:
            out.append(char)
            out.append(block[i + 1])
            i += 2
            continue
        if char == quote:
            lookahead = i + 1
            while lookahead < length and block[lookahead].isspace():
                lookahead += 1
            if lookahead >= length or block[lookahead] in _VALUE_TERMINATORS:
                quote = None
            else:
                out.append("\\")
            out.append(char)
            i += 1
            continue
        out.append(char)
        i += 1
    return "".join(out)


def _extract_action_block_by_markers(text: str) -> str | None:
    if not text:
        return None
    # A word-bounded header, as in _extract_json_like_block: "interaction:" or "fraction:" in a
    # report is not an action.
    m = ACTION_HDR_RE.search(text)
    if not m:
        return None
    # Try bounded extraction between Action: and Observation:/Final Report:
    tail = text[m.end() :]
    stop_idx = len(tail)
    for marker in ("observation:", "final report:"):
        pos = tail.lower().find(marker)
        if pos != -1:
            stop_idx = min(stop_idx, pos)
    block = tail[:stop_idx].strip()
    if not block:
        return None
    return block


def _is_action_dict(parsed: Any) -> bool:
    """An action is a dict whose ``tool`` is a string and ``args`` a dict or a string (if present).

    A string ``args`` is a generic tool's single input (LangChain passes it to the tool's first
    field); the tools that read named fields reject it in the act node (`_NAMED_ARGS_TOOLS`).
    """
    return (
        isinstance(parsed, dict)
        and isinstance(parsed.get("tool", ""), str)
        and isinstance(parsed.get("args", {}), (dict, str))
    )


# Tools the act node handles itself by reading named fields of `args` (`tool_args.get(...)`), so a
# string `args` cannot be used for them.
_NAMED_ARGS_TOOLS = frozenset(
    {
        "execute_code",
        "record_notepad_idea",
        "establish_collaboration",
        "communicate_with_collaborator",
        "search_agent_registry",
    }
)


def _unusable_args(tool_name: str, tool_args: Any) -> bool:
    """Whether `tool_args` is a string given to a tool that needs named arguments."""
    return isinstance(tool_args, str) and tool_name in _NAMED_ARGS_TOOLS


def parse_action(text: str) -> dict[str, Any] | None:
    """Parse an `Action` payload from model output into an action dict.

    The function extracts a JSON-like block after the `Action:` header via
    `_extract_json_like_block` (or, failing that, the text up to the next
    `Observation:`/`Final Report:`). It then tries `json.loads`, `ast.literal_eval`,
    and `ast.literal_eval` after `_repair_unescaped_quotes`, in that order, each on the
    block as written, with trailing commas removed, and with smart quotes rewritten to
    ASCII. The first parse that succeeds decides: a dict whose `args` (if any) is a dict
    or a string is the action; anything else, such as a set from mismatched quotes or a
    bare string, makes the block invalid.

    Args:
        text: Raw model output potentially containing an `Action` block.

    Returns:
        dict | None: The action dict (e.g., with `tool` and `args` keys); the
        ``__invalid_action__`` marker dict when there is an action that does not parse
        into one; ``None`` when there is no action.
    """
    block = _extract_json_like_block(text or "")
    if not block:
        block = _extract_action_block_by_markers(text or "")
    if not block:
        if isinstance(text, str) and "Action:" in text:
            snippet = text.strip()[:200]
            log.warning(f"[research_team][action_parse] no_block snippet={snippet!r}")
            return {"tool": "__invalid_action__", "args": {"raw": snippet}}
        return None
    # Strictest parser first, as in the source: json.loads on every form of the block,
    # then ast.literal_eval on every form, then the quote repair. Within each parser the
    # block as written comes first. Blocks that mix smart-quote delimiters with
    # ASCII-quoted strings only parse once the smart quotes are rewritten.
    candidates = _action_block_candidates(block)
    parsers = (
        json.loads,
        ast.literal_eval,
        # Last resort: prose or code carrying an unescaped quote.
        lambda c: ast.literal_eval(_repair_unescaped_quotes(c)),
    )
    first_error: Exception | None = None
    for parse, candidate in itertools.product(parsers, candidates):
        try:
            parsed = parse(candidate)
        except Exception as e:
            first_error = first_error or e
            continue
        if _is_action_dict(parsed):
            return parsed
        # A set (from mismatched quotes) or a bare value is not an action. Returned, it
        # made the think/act nodes fail on .get() and end the session; reject the block.
        first_error = TypeError(f"parsed into a {type(parsed).__name__}, not an action dict")
        break
    err = "".join(tb.format_exception_only(type(first_error), first_error)).strip()
    snippet = block[:500].replace("\n", "\\n")
    log.warning(f"[research_team][action_parse] no action dict parsed: {err}")
    log.warning(f"[research_team][action_parse] block_snippet={snippet!r}")
    return {"tool": "__invalid_action__", "args": {"raw": snippet}}


# ——————————————————————————————————————————————————
# code block extraction utility
# ——————————————————————————————————————————————————
CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def _extract_code_block(text: str) -> str | None:
    """Return the first fenced code block (``` or ```python) from text, if any."""
    if not text:
        return None
    m = CODE_BLOCK_RE.search(text)
    if not m:
        return None
    return m.group(1).strip() or None


# ——————————————————————————————————————————————————
# _msg_text extraction utility (for observations)
# ——————————————————————————————————————————————————
def _msg_text(msg: Any) -> str:
    """Best-effort string extraction for message objects."""
    try:
        c = getattr(msg, "content", msg)
        return c if isinstance(c, str) else str(c)
    except Exception:
        return repr(msg)


def _record_notepad_idea_passthrough(
    title: str,
    idea_text: str,
    deferred_reason: str | None = None,
    tags: list[str] | None = None,
) -> str:
    """Record a deferred research idea in your private agent notepad."""
    _ = (title, idea_text, deferred_reason, tags)
    return "notepad_idea_recorded"


def get_research_team(
    llm: BaseChatModel,
    sandbox_client: Any,
    iteration: int,
    primary_agent_profile: AgentProfile,
    agent_network_metadata: AgentNetworkMetadata | None = None,
    prior_papers: list[PaperMetadataPrivate] | None = None,
    prior_paper_summaries: list[str] | None = None,
    prior_papers_enabled: bool = True,
    prior_papers_mode: str = "summary",
    self_reflection_prompt_enabled: bool = False,
    full_citing_enabled: bool = False,
    graph_logic_cfg: dict[str, Any] | None = None,
    prior_paper_status_by_id: dict[str, str] | None = None,
    meta_research_mode: bool = False,
    meta_research_context: dict[str, Any] | None = None,
    nature_style_papers_prompt_enabled: bool = False,
    metacognition_feedback_context: dict[str, Any] | None = None,
    *,
    network_setting: str | None = None,
    primary_agent_tools: dict[str, ToolInfo] | None = None,
    collab_agent_tools: dict[str, ToolInfo] | None = None,
    python_packages: dict[str, str] | None = None,
    dataset_metadata: dict[str, str] | None = None,
    # if provided, we MUST NOT create another session
    sandbox_session_id: str | None = None,
    # sticky wrapper overrides the raw session id when present
    sandbox_sticky: Any | None = None,
    code_collaboration: bool = True,
    execute_code_retries: int = 120,
    execute_code_retry_delay: float = 5.0,
    fixed_workflow_context: dict[str, Any] | None = None,
) -> tuple[Any, str]:
    """Construct and compile the primary research agent graph, returning the app and session id.

    This function wires up a think–act loop for the primary agent. It prepares a
    code-execution tool bound either to a provided sticky session handle or to a
    newly/externally created sandbox session, augments the primary agent's toolset,
    builds a strict Pydantic paper schema for output parsing, and then assembles a
    `ChatPromptTemplate` from the primary agent system prompt.

    The compiled graph has two nodes:
      - `think`: Calls the LLM with the running message history and records the
        raw output in `last_ai_response`.
      - `act`: Parses an `Action` block from `last_ai_response`, dispatches to the
        appropriate tool (e.g., `execute_code`, `establish_collaboration`,
        `communicate_with_collaborator`, or any other registered tool), and appends
        an `Observation` back into the message stream. Certain tools update
        auxiliary state such as the running `script` or collaboration context.

    Args:
        llm: Chat model used for the agent's reasoning steps.
        sandbox_client: Client used to create/manage non-sticky sandbox sessions.
        primary_agent_profile: Profile describing the primary agent (id, expertise, behavior, etc.).
        agent_network_metadata: Metadata describing the agent network (e.g., number of agents).
        network_setting: Optional network configuration hint (e.g., homo/hetero settings).
        primary_agent_tools: Optional mapping of tool name to `ToolInfo` for the primary agent.
        collab_agent_tools: Optional mapping of tool name to `ToolInfo` exposed to collaborators.
        python_packages: Optional mapping of package name -> short description for prompt context.
        dataset_metadata: Optional mapping of dataset key -> human-readable description.
        sandbox_session_id: If provided, reuse this sandbox session id (no new session will be created).
        sandbox_sticky: Optional sticky session handle; when provided, it overrides `sandbox_session_id`
            and forces execution in that persistent environment.

    Returns:
        Tuple[Any, str]: A tuple of (compiled LangGraph app, sandbox session id used).
    """
    if primary_agent_tools is None:
        primary_agent_tools = {}
    if collab_agent_tools is None:
        collab_agent_tools = {}

    notepad_store = None
    try:
        notepad_store = _global_state.notepad_store()
    except Exception:
        notepad_store = None
    notepad_enabled = bool(notepad_store is not None and getattr(notepad_store, "enabled", False))
    if notepad_enabled and "record_notepad_idea" not in primary_agent_tools:
        notepad_tool = StructuredTool.from_function(
            func=_record_notepad_idea_passthrough,
            name="record_notepad_idea",
            description=(
                "Record a promising research idea you are deferring for now in your private notepad."
            ),
        )
        primary_agent_tools["record_notepad_idea"] = ToolInfo(
            object=notepad_tool,
            tool_signature=tool_signature_line(notepad_tool),
            code="",
            code_signature="",
        )

    def _tool_name(t):
        # Handle ToolInfo wrappers, LangChain tools, etc.
        obj = getattr(t, "object", t)
        return getattr(obj, "name", getattr(obj, "__name__", type(obj).__name__))

    if sandbox_sticky is not None:
        # Hard bind to the caller’s sticky handle — never create a new container here.
        agent_id = getattr(primary_agent_profile, "agent_id", "unknown_agent")
        code_executor = make_code_executor_tool_for_sticky(
            sandbox_sticky,
            agent_id=agent_id,
            retries=execute_code_retries,
            retry_delay=execute_code_retry_delay,
        )
        # If you want to return an id for logging/compat, prefer the current sid if available:
        sandbox_session_id = getattr(sandbox_sticky, "session_id", None) or sandbox_session_id
    else:
        if sandbox_session_id is None:
            # Back-compat: create here only if absolutely necessary.
            owner_tag = "research_team"
            agent_id = getattr(primary_agent_profile, "agent_id", "unknown_agent")
            sandbox_session_id = sandbox_client.create_session(
                session_name=f"{agent_id}_adhoc",
                extra_payload={"owner": owner_tag, "agent_id": agent_id},
            )
        code_executor = make_code_executor_tool(
            sandbox_client,
            sandbox_session_id,
            agent_id=getattr(primary_agent_profile, "agent_id", "unknown_agent"),
            retries=execute_code_retries,
            retry_delay=execute_code_retry_delay,
        )

    primary_agent_tools["execute_code"] = ToolInfo(
        object=code_executor,
        tool_signature=tool_signature_line(code_executor),
        code="",
        code_signature="",
    )
    log.debug(f"[research_team][tools] primary_tools={list(primary_agent_tools.keys())}")

    # Build the “allowed dataset keys” list for schema constraints
    allowed_dataset_keys = list(dataset_metadata.keys()) if dataset_metadata else []

    # 1) Build the Pydantic schema + parser for strict JSON
    PaperModel, _ = make_paper_model(allowed_dataset_keys=allowed_dataset_keys)
    parser = PydanticOutputParser(pydantic_object=PaperModel)
    parser.get_format_instructions()

    graph_cfg = normalize_graph_logic_cfg(graph_logic_cfg)
    graph_logic_enabled = bool(graph_cfg.get("enabled", False))
    require_plot_citations = bool(graph_cfg.get("require_plot_citations_in_research", False))
    try:
        plot_only_rewrite_cap = max(0, int(graph_cfg.get("plot_only_rewrite_cap", 2) or 0))
    except Exception:
        plot_only_rewrite_cap = 2
    allow_plot_validation_waiver_after_cap = bool(
        graph_cfg.get("allow_plot_validation_waiver_after_cap", True)
    )
    fixed_workflow_ctx = _normalize_fixed_workflow_runtime_context(fixed_workflow_context)

    # 2) Your existing system prompt (tools, code, network, etc.)
    system_prompt = make_primary_agent_system_prompt(
        agent_profile=primary_agent_profile,
        agent_network_metadata=agent_network_metadata,
        iteration=iteration,
        tools=primary_agent_tools,
        python_packages=python_packages,
        dataset_metadata=dataset_metadata,
        prior_papers=prior_papers,
        prior_paper_summaries=prior_paper_summaries,
        prior_papers_enabled=prior_papers_enabled,
        prior_papers_mode=prior_papers_mode,
        self_reflection_prompt_enabled=self_reflection_prompt_enabled,
        full_citing_enabled=full_citing_enabled,
        graph_logic_enabled=graph_logic_enabled,
        require_plot_citations_in_research=require_plot_citations,
        meta_research_mode=meta_research_mode,
        meta_research_context=meta_research_context,
        nature_style_papers_prompt_enabled=nature_style_papers_prompt_enabled,
        metacognition_feedback_context=metacognition_feedback_context,
        fixed_workflow_context=fixed_workflow_ctx,
    )
    # --- Llama “auto tools” emulation (only for the llama backend) ---
    if is_llama_backend(llm):
        # Optional: pass known ids to constrain Llama’s args
        try:
            known_ids = [a.agent_id for a in _global_state.registry().get_all_agents()]
        except Exception:
            known_ids = []
        system_prompt = make_llama_auto_tools_prompt(system_prompt, valid_agent_ids=known_ids)

    # 3) Build the chat prompt AFTER the augmented system prompt
    prompt_structure = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),
            MessagesPlaceholder("messages"),
        ]
    )

    collab_agent = None
    collab_agent_state = None
    prior_status_by_id = {
        str(pid): str(status).strip().lower()
        for pid, status in (prior_paper_status_by_id or {}).items()
    }

    def _is_plot_validation_error(err: str) -> bool:
        lowered = str(err or "").lower()
        plot_error_signals = (
            "unknown plot filename",
            "unknown plot citation token",
            "unknown plot handle",
            "at least one [plot: ...] citation tag is required",
            "plot tag missing filename",
            "plot tag must contain only filename token",
            "plot tag filename contains unsupported characters",
        )
        return any(sig in lowered for sig in plot_error_signals)

    def _is_plot_only_error_set(errors: list[str]) -> bool:
        return bool(errors) and all(_is_plot_validation_error(err) for err in errors)

    def _validate_final_report_citations(
        text: str,
        *,
        successful_code_executions_current_iteration: int,
        code_script: str,
        known_plot_filenames: list[str] | None = None,
        known_plot_handles: list[str] | None = None,
        plot_handle_to_filename: dict[str, str] | None = None,
        enforce_plot_validation: bool = True,
    ) -> tuple[bool, list[str], dict[str, Any]]:
        known_filenames = [str(x).strip() for x in (known_plot_filenames or []) if str(x).strip()]
        known_handles = [str(x).strip() for x in (known_plot_handles or []) if str(x).strip()]
        handle_to_filename = {
            str(k).strip(): str(v).strip()
            for k, v in (plot_handle_to_filename or {}).items()
            if str(k).strip() and str(v).strip()
        }
        enforce_plot_checks = bool(graph_logic_enabled and enforce_plot_validation)
        enforce_plot_tags = bool(
            enforce_plot_checks and require_plot_citations and len(known_filenames) > 0
        )
        if not full_citing_enabled and not graph_logic_enabled:
            return (
                True,
                [],
                {
                    "is_valid": True,
                    "errors": [],
                    "code_evidence": {
                        "code_tag_count": 0,
                        "successful_code_executions_current_iteration": successful_code_executions_current_iteration,
                        "code_script_chars": len(str(code_script or "").strip()),
                        "is_backed": True,
                        "errors": [],
                    },
                },
            )
        verdict = validate_citation_tags(
            text or "",
            require_at_least_one=bool(full_citing_enabled),
            known_archive_status_by_id=(prior_status_by_id if full_citing_enabled else None),
            require_accepted_archive_ids=bool(full_citing_enabled),
            require_backed_code_tags=bool(full_citing_enabled),
            successful_code_executions_current_iteration=successful_code_executions_current_iteration,
            code_script=code_script,
            known_plot_filenames=(known_filenames if enforce_plot_checks else None),
            known_plot_handles=(known_handles if enforce_plot_checks else None),
            plot_handle_to_filename=(handle_to_filename if enforce_plot_checks else None),
            require_known_plot_filenames=bool(enforce_plot_checks),
            require_at_least_one_plot_tag=enforce_plot_tags,
        )
        errors = [str(e) for e in verdict.get("errors", [])]
        return bool(verdict.get("is_valid", False)), errors, verdict

    def think_node(state: PrimaryAgentState) -> dict[str, Any]:
        """Invoke the LLM, log token usage, and record the latest AI response.

        This node feeds the running conversation (`state["messages"]`) into the
        model via the compiled prompt, prints a short preview for tracing, updates
        usage metrics (if a tracker is attached), and returns the new AI message
        while preserving pass-through fields in the state.

        Args:
            state: Current primary agent state including messages and metadata.

        Returns:
            Dict[str, Any]: Partial state update containing the appended AI message,
            updated `last_ai_response`, and preserved passthrough fields.
        """
        m = get_logs(agent)

        prompt_val = prompt_structure.invoke({"messages": state["messages"]})
        ai = invoke_with_content_filter_retry(
            llm,
            prompt_val,
            stage_label=f"research_team.think.agent={state.get('metadata', {}).get('agent_id', 'unknown')}",
            logger=log,
        )
        # show first 220 chars of raw output
        preview = (
            ai.content if isinstance(ai.content, str) else str(ai.content)
        )  # for debugging print below
        parsed_action = parse_action(preview)
        log.debug(
            f"[agent={state.get('metadata', {}).get('agent_id', 'unknown')}] think: step={state.get('step', 1)} action?={bool(parsed_action)} preview={preview[:220]!r}"
        )
        if isinstance(ai.content, str) and "Action:" in ai.content:
            aid = state.get("metadata", {}).get("agent_id", "unknown")
            first_line = ai.content.strip().splitlines()[0]
            log.debug(
                "[research_team][action_emit] "
                f"agent={aid} step={state.get('step', 1)} line={first_line[:200]!r}"
            )
            if not parsed_action:
                log.warning(
                    "[research_team][action_parse] "
                    f"agent={aid} step={state.get('step', 1)} failed_parse=True"
                )
            if parsed_action and parsed_action.get("tool") == "__invalid_action__":
                log.warning(
                    "[research_team][action_parse] "
                    f"agent={aid} step={state.get('step', 1)} invalid_action=True"
                )

        if m:
            usage = getattr(ai, "usage_metadata", {}) or {}
            m.input_tokens += int(usage.get("input_tokens", 0) or 0)
            m.output_tokens += int(usage.get("output_tokens", 0) or 0)
            m.total_tokens += int(usage.get("total_tokens", 0) or 0)
            m.turns += 1
            meta = state.get("metadata", {}) or {}
            m.add_event(
                kind="think",
                agent_id=meta.get("agent_id", "unknown"),
                iteration=meta.get("iteration"),
                step=state.get("step", 1),
                messages=[_msg_text(msg) for msg in state.get("messages", [])],
                response=_msg_text(ai),
            )

        # IMPORTANT: preserve everything we already had in the state
        return {
            "messages": [ai],
            "last_ai_response": ai.content,
            # passthroughs:
            "script": state.get("script", "") or "",
            "step": state.get("step", 1),
            "collab_agent_ids": state.get("collab_agent_ids", []),
            "metadata": state.get("metadata", {}),
        }

    def act_node(state: PrimaryAgentState) -> dict[str, Any]:
        """Parse the pending action and dispatch to the appropriate tool.

        Depending on `tool_name`, this node can:
          - Execute code in a sandbox (`execute_code`), capturing observations and
            appending code plus results to a running `script`.
          - Establish a collaborator agent (`establish_collaboration`) and cache
            its compiled graph/state.
          - Send a message to the collaborator (`communicate_with_collaborator`)
            and return their response.
          - Invoke any other registered tool and forward its stringified result.

        It also increments counters in the attached tracker, handles common error
        cases (unknown tool, parse failure), and always appends an `Observation`
        message to the conversation.

        Args:
            state: Current primary agent state which must include `last_ai_response`
                containing a candidate `Action` block.

        Returns:
            Dict[str, Any]: Partial state update including appended observation
            message(s), updated `step`, and optionally a modified `script` or
            collaboration metadata.
        """
        nonlocal collab_agent, collab_agent_state
        act = parse_action(state["last_ai_response"])
        log.debug(f"act: step={state['step']} parsed_action={bool(act)}")
        m = get_logs(agent)

        meta = state.get("metadata", {}) or {}
        aid = meta.get("agent_id", "unknown")
        if m:
            m.add_event(
                kind="action_parsed",
                agent_id=aid,
                iteration=meta.get("iteration"),
                step=state["step"],
                raw_response=str(state.get("last_ai_response", "")),
                action=act,
            )

        if not act:
            if m:
                m.invalid_actions += 1
            return {
                "messages": [HumanMessage(content="Observation: Failed to parse action")],
                "script": state.get("script", "") or "",
                "step": state["step"] + 1,
            }
        aid = state.get("metadata", {}).get("agent_id", "unknown")
        log.debug(f"[agent={aid}] action payload: {act!r}")
        tool_name = act.get("tool")
        if not tool_name:
            if m:
                m.invalid_actions += 1
            return {
                "messages": [HumanMessage(content="Observation: Unknwn tool")],
                "script": state.get("script", "") or "",
                "step": state["step"] + 1,
                "collab_agent_ids": state.get("collab_agent_ids", []),
                "metadata": state.get("metadata", {}),
            }
        tool_args = act.get("args", {})
        if _unusable_args(tool_name, tool_args):
            tool_name = "__invalid_action__"
        if tool_name == "__invalid_action__":
            if m:
                m.invalid_actions += 1
            return {
                "messages": [
                    HumanMessage(
                        content="Observation: Malformed action. Please output a clean Action block only, with no extra text."
                    )
                ],
                "script": state.get("script", "") or "",
                "step": state["step"] + 1,
                "collab_agent_ids": state.get("collab_agent_ids", []),
                "metadata": state.get("metadata", {}),
            }
        fixed_block = _fixed_workflow_block_observation(tool_name, fixed_workflow_ctx)
        if fixed_block:
            if m:
                m.invalid_actions += 1
                m.add_event(
                    kind="fixed_workflow_blocked_tool",
                    agent_id=aid,
                    iteration=meta.get("iteration"),
                    step=state["step"],
                    tool=tool_name,
                    assigned_collaborator_id=fixed_workflow_ctx.get("assigned_collaborator_id"),
                    reason="agent_registry_disabled",
                )
            return {
                "messages": [HumanMessage(content=fixed_block)],
                "script": state.get("script", "") or "",
                "step": state["step"] + 1,
                "collab_agent_ids": state.get("collab_agent_ids", []),
                "metadata": state.get("metadata", {}),
            }
        if (
            bool(fixed_workflow_ctx.get("enabled", False))
            and tool_name == "establish_collaboration"
            and bool(fixed_workflow_ctx.get("force_tool_target", True))
        ):
            assigned = str(fixed_workflow_ctx.get("assigned_collaborator_id", "") or "").strip()
            requested = str((tool_args or {}).get("agent_id", "") or "").strip()
            if assigned and requested != assigned:
                if m:
                    m.add_event(
                        kind="fixed_workflow_forced_collaboration_target",
                        agent_id=aid,
                        iteration=meta.get("iteration"),
                        step=state["step"],
                        requested_collaborator_id=requested,
                        assigned_collaborator_id=assigned,
                    )
                tool_args = dict(tool_args or {})
                tool_args["agent_id"] = assigned
        tool_info = primary_agent_tools.get(tool_name)
        if tool_name and tool_info is None:
            log.warning(
                "[research_team][tool_missing] "
                f"agent={aid} tool={tool_name} "
                f"available={list(primary_agent_tools.keys())}"
            )

        log.debug(
            "[research_team][tool_dispatch] "
            f"agent={aid} tool={tool_name} has_log={bool(m)} "
            f"log_type={type(m).__name__ if m else None}"
        )
        if tool_name == "execute_code" and not m:
            log.warning(f"[research_team][execute_code] missing stats log agent={aid}")

        if not tool_info:
            if m:
                m.invalid_actions += 1
            return {
                "messages": [HumanMessage(content="Observation: Unknown tool")],
                "script": state.get("script", "") or "",
                "step": state["step"] + 1,
            }

        tool = tool_info.object

        if tool_name == "execute_code":
            execute_block = _fixed_workflow_execute_code_block_observation(
                state,
                fixed_workflow_ctx,
            )
            if execute_block:
                if m:
                    m.invalid_actions += 1
                    m.add_event(
                        kind="fixed_workflow_blocked_tool",
                        agent_id=aid,
                        iteration=meta.get("iteration"),
                        step=state["step"],
                        tool=tool_name,
                        assigned_collaborator_id=fixed_workflow_ctx.get("assigned_collaborator_id"),
                        reason="collaboration_required_before_code",
                    )
                return {
                    "messages": [HumanMessage(content=execute_block)],
                    "script": state.get("script", "") or "",
                    "step": state["step"] + 1,
                    "collab_agent_ids": state.get("collab_agent_ids", []),
                    "metadata": state.get("metadata", {}),
                }

        if m:
            m.tool_calls[tool_name] += 1
            m.add_event(
                kind="tool_call",
                agent_id=aid,
                iteration=meta.get("iteration"),
                step=state["step"],
                tool=tool_name,
                args=tool_args,
            )

        if tool_name == "execute_code":
            code = tool_args.get("code", "")
            log.debug(f"[agent={aid}] execute_code: LOC={count_loc(code)}")

            if m:
                m.code_executions += 1
                m.lines_of_code += count_loc(code)

            tool_args = {"code": code}
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
            sid_dbg = None
            job_id = None
            cell_no = None
            updated_metadata = dict(meta)
            try:
                result = tool.invoke(tool_args)
                if not isinstance(result, dict):
                    raise RuntimeError(
                        f"execute_code returned non-dict response: {type(result).__name__}"
                    )
                # include agent_id and session id (if present) for disambiguation
                sid_dbg = get_session_id(result)
                job_id = result.get("job_id")
                intr = result.get("introspection") or {}  # now always present
                if not intr and isinstance(result.get("pre_install"), list):
                    intr = (result["pre_install"] or [{}])[-1].get("introspection", {}) or {}
                cell_no = intr.get("cell_no")
                log.debug(
                    f"[research_team][execute_code][agent={aid}][sid={sid_dbg}][job={job_id}][cell_no={cell_no}] keys={list(result.keys())}"
                )

                if result.get("status") == "error" and log.isEnabledFor(logging.DEBUG):
                    hist = result.get("history_full") or result.get("history_tail") or []
                    history_file = (result.get("introspection") or {}).get("history_file") or (
                        (result.get("pre_install") or [{}])[-1].get("introspection") or {}
                    ).get("history_file")
                    dump = [
                        f"[history_dump][sid={sid_dbg}] total_records={len(hist)} file={history_file}"
                    ]
                    # Only the *code cells*: the 'run_start' entries written at submission time
                    cells = [h for h in hist if h.get("event") == "run_start"]
                    # Ensure stable order by cell_no (or fallback)
                    cells.sort(key=lambda r: r.get("cell_no") or 0)
                    for rec in cells:
                        code = (rec.get("code") or "").rstrip("\n")
                        dump.append(f"  --- cell #{rec.get('cell_no')} [{rec.get('ts')}]")
                        dump.extend(
                            f"    {i:>3}: {line}" for i, line in enumerate(code.splitlines(), 1)
                        )
                    dump.append("[failing_cell]\n" + (tool_args.get("code") or "").strip())

                    # basic sanity—cell numbers should be monotonic ⇒ proves stickiness
                    seq = [
                        rec.get("cell_no") for rec in cells if isinstance(rec.get("cell_no"), int)
                    ]
                    if seq:
                        mono = all(b > a for a, b in itertools.pairwise(seq))
                        dump.append(
                            f"[stickiness_check] cell_no_monotonic={mono} first={seq[0]} last={seq[-1]}"
                        )
                    log.debug("\n".join(dump))

                obs_text, _, was_successful = extract_sandbox_response(result)
                if was_successful:
                    persist_summary = persist_sandbox_images_to_snapshot(
                        resp=result,
                        run_name=meta.get("run_name"),
                        iteration=meta.get("iteration"),
                        step=state.get("step"),
                        agent_id=aid,
                        session_id=sid_dbg,
                        job_id=job_id,
                        cell_no=cell_no,
                        base_snapshots_dir=str(
                            meta.get("base_snapshots_dir", "snapshots") or "snapshots"
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
                if m and image_count > 0:
                    m.images_created += image_count
                if m and image_warnings:
                    for warning in image_warnings[:10]:
                        m.add_event(
                            kind="tool_warning_execute_code_images",
                            agent_id=aid,
                            iteration=meta.get("iteration"),
                            step=state.get("step"),
                            tool="execute_code",
                            warning=warning,
                        )
                if plot_artifacts:
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
                    known_plot_filenames = sorted(
                        {
                            str(item.get("plot_filename", "")).strip()
                            for item in existing
                            if isinstance(item, dict) and str(item.get("plot_filename", "")).strip()
                        }
                    )
                    known_plot_handles = sorted(
                        {
                            str(item.get("plot_handle", "")).strip()
                            for item in existing
                            if isinstance(item, dict) and str(item.get("plot_handle", "")).strip()
                        }
                    )
                    plot_handle_to_filename = {
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
                    plot_citations = sorted(
                        {
                            str(item.get("plot_citation", "")).strip()
                            for item in existing
                            if isinstance(item, dict) and str(item.get("plot_citation", "")).strip()
                        }
                    )
                    updated_metadata["known_plot_filenames"] = known_plot_filenames
                    updated_metadata["known_plot_handles"] = known_plot_handles
                    updated_metadata["plot_handle_to_filename"] = plot_handle_to_filename
                    updated_metadata["plot_citations"] = plot_citations
            except Exception as e:
                if m:
                    m.code_executions_failed += 1
                obs_text = e
                messages = [HumanMessage(content=f"Observation: {obs_text}")]
                current_script = state.get("script", "") or ""
                script = current_script + "\n" + code + "\n" + f"# Error: {obs_text}"
                return {
                    "messages": messages,
                    "script": script,
                    "step": state["step"] + 1,
                    "metadata": state.get("metadata", {}),
                }
            aid = state.get("metadata", {}).get("agent_id", "unknown")
            if plot_paths:
                obs_text = f"{obs_text}\nPLOTS = {json.dumps(plot_paths, ensure_ascii=False)}"
            if plot_artifacts:
                obs_text = (
                    f"{obs_text}\nPLOT_ARTEFACTS = {json.dumps(plot_artifacts, ensure_ascii=False)}"
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
                    obs_text = (
                        f"{obs_text}\n"
                        f"PLOT_CITATIONS = {json.dumps(plot_citation_rows, ensure_ascii=False)}"
                    )
            log.debug(
                f"[research_team][execute_code][agent={aid}][sid={sid_dbg}] "
                f"success={was_successful} obs_len={len(obs_text)}"
            )

            if m:
                if was_successful:
                    m.code_executions_successful += 1
                else:
                    m.code_executions_failed += 1
                m.add_event(
                    kind="tool_result_execute_code",
                    agent_id=aid,
                    iteration=meta.get("iteration"),
                    step=state["step"],
                    tool="execute_code",
                    session_id=sid_dbg,
                    job_id=job_id,
                    cell_no=cell_no,
                    status=result.get("status"),
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
                    observation=obs_text,
                )

            if was_successful:
                current_script = state.get("script", "") or ""
                # record observation & append to running script
                messages = [HumanMessage(content=f"Observation: {obs_text}")]
                script = (
                    current_script
                    + "\n"
                    + code
                    + "\n"
                    + "# Observation:\n"
                    + "\n".join(f"# {line}" for line in obs_text.splitlines())
                    + "\n"
                )
                return {
                    "messages": [HumanMessage(content=f"Observation: {obs_text}")],
                    "step": state["step"] + 1,
                    "script": script,
                    "metadata": updated_metadata,
                }
            current_script = state.get("script", "") or ""
            script = (
                current_script
                + "\n"
                + code
                + "\n"
                + "# Error or timeout:\n"
                + "\n".join(f"# {line}" for line in obs_text.splitlines())
                + "\n"
            )
            return {
                "messages": [HumanMessage(content=f"Observation: {obs_text}")],
                "step": state["step"] + 1,
                "script": script,
                "metadata": updated_metadata,
            }

        if tool_name == "record_notepad_idea":
            if not notepad_enabled or notepad_store is None:
                if m:
                    m.invalid_actions += 1
                return {
                    "messages": [
                        HumanMessage(
                            content="Observation: Private notepad recording is disabled for this run."
                        )
                    ],
                    "step": state["step"] + 1,
                    "script": state.get("script", "") or "",
                    "collab_agent_ids": state.get("collab_agent_ids", []),
                    "metadata": state.get("metadata", {}),
                }
            title = str(tool_args.get("title", "") or "").strip()
            idea_text = str(tool_args.get("idea_text", "") or "").strip()
            deferred_reason = str(tool_args.get("deferred_reason", "") or "").strip() or None
            tags_raw = tool_args.get("tags", [])
            tags = (
                [str(x).strip() for x in (tags_raw or []) if str(x).strip()]
                if isinstance(tags_raw, list)
                else []
            )
            if not title or not idea_text:
                if m:
                    m.invalid_actions += 1
                return {
                    "messages": [
                        HumanMessage(
                            content="Observation: record_notepad_idea requires non-empty 'title' and 'idea_text'."
                        )
                    ],
                    "step": state["step"] + 1,
                    "script": state.get("script", "") or "",
                    "collab_agent_ids": state.get("collab_agent_ids", []),
                    "metadata": state.get("metadata", {}),
                }
            try:
                row = notepad_store.append_idea(
                    agent_id=str(aid),
                    iteration=int(meta.get("iteration", 0) or 0),
                    run_name=str(meta.get("run_name", "") or "unnamed_experiment"),
                    title=title,
                    idea_text=idea_text,
                    deferred_reason=deferred_reason,
                    tags=tags,
                )
                obs_text = (
                    f"Observation: Notepad idea recorded with idea_id={row.get('idea_id')} "
                    f"at {row.get('notepad_path')}."
                )
                if m:
                    m.add_event(
                        kind="tool_result_record_notepad_idea",
                        agent_id=aid,
                        iteration=meta.get("iteration"),
                        step=state["step"],
                        tool="record_notepad_idea",
                        idea_id=row.get("idea_id"),
                        notepad_path=row.get("notepad_path"),
                        append_status=row.get("append_status"),
                    )
            except Exception as e:
                if m:
                    m.invalid_actions += 1
                obs_text = f"Observation: Failed to record notepad idea: {e}"
            return {
                "messages": [HumanMessage(content=obs_text)],
                "step": state["step"] + 1,
                "script": state.get("script", "") or "",
                "collab_agent_ids": state.get("collab_agent_ids", []),
                "metadata": state.get("metadata", {}),
            }

        if tool_name == "establish_collaboration":
            log.debug(f"[establish_collaboration] incoming args={tool_args!r}")

            try:
                known_ids = [
                    a.agent_id
                    for a in _global_state.registry().get_all_agents()
                    if a.agent_id != state.get("metadata", {}).get("agent_id", "unknown")
                ]  # fallback candidates; an explicit request for this agent is refused below
            except Exception:
                known_ids = []

            aid_in = tool_args.get("agent_id") or ""
            if (not isinstance(aid_in, str)) or (not aid_in.startswith("agent_")):
                if known_ids:
                    tool_args["agent_id"] = random.choice(known_ids)
                else:
                    return {
                        "messages": [
                            HumanMessage(
                                content="Observation: No such agent_id and registry is empty."
                            )
                        ],
                        "step": state["step"] + 1,
                        "script": state.get("script", "") or "",
                        "collab_agent_ids": state.get("collab_agent_ids", []),
                        "metadata": state.get("metadata", {}),
                    }

            log.debug(
                f"[research_team][establish_collaboration][agent={primary_agent_profile.agent_id}] attempting to establish collaboration"
            )
            exists, collab_profile, _ = tool.invoke(tool_args)
            log.debug(
                f"[research_team][establish_collaboration] exists={exists} profile={collab_profile!r}"
            )

            if not collab_profile:
                hint = f" Valid agent_ids: {', '.join(known_ids)}" if known_ids else ""
                return {
                    "messages": [HumanMessage(content=f"Observation: No such agent.{hint}")],
                    "step": state["step"] + 1,
                    "script": state.get("script", "") or "",
                    "collab_agent_ids": state.get("collab_agent_ids", []),
                    "metadata": state.get("metadata", {}),
                }

            if collab_profile.agent_id == primary_agent_profile.agent_id:
                return {
                    "messages": [
                        HumanMessage(
                            content="Observation: You cannot collaborate with yourself. "
                            "Please choose a different agent_id."
                        )
                    ],
                    "step": state["step"] + 1,
                    "script": state.get("script", "") or "",
                    "collab_agent_ids": state.get("collab_agent_ids", []),
                    "metadata": state.get("metadata", {}),
                }

            if not exists:
                # Hint valid ids in observation to steer Llama’s next THINK
                hint = f" Valid agent_ids: {', '.join(known_ids)}" if known_ids else ""
                return {
                    "messages": [HumanMessage(content=f"Observation: No such agent.{hint}")],
                    "step": state["step"] + 1,
                    "script": state.get("script", "") or "",
                    "collab_agent_ids": state.get("collab_agent_ids", []),
                    "metadata": state.get("metadata", {}),
                }

            collab_agent = get_collab_agent(
                llm=llm,
                agent_profile=collab_profile,
                primary_agent_profile=primary_agent_profile,
                agent_network_metadata=agent_network_metadata,
                tools=collab_agent_tools,
                dataset_metadata=dataset_metadata,
                run_name=meta.get("run_name"),
                iteration=meta.get("iteration"),
                base_snapshots_dir=str(meta.get("base_snapshots_dir", "snapshots") or "snapshots"),
            )
            if m:
                collab_log = get_logs(collab_agent)
                if collab_log is not None:
                    m.collab_logs[collab_profile.agent_id] = collab_log
            current_script = state.get("script", "") or ""
            collab_agent_state = {
                "messages": [],
                "step": 1,
                "last_ai_response": "",
                "script": current_script if code_collaboration else "",
                "metadata": {
                    **(state.get("metadata", {}) or {}),
                    "agent_id": collab_profile.agent_id,
                },
            }

            collab_agent_ids = state.get("collab_agent_ids", [])
            collab_agent_ids.append(collab_profile.agent_id)

            aid = state.get("metadata", {}).get("agent_id", "unknown")
            log.debug(
                f"[research_team][establish_collaboration][agent={aid}][collab={collab_profile.agent_id}] established collaboration"
            )
            log.debug(
                f"[research_team][establish_collaboration][agent={aid}][collab={collab_profile.agent_id}] exists={exists} collab_profile={collab_profile!r} collab_agent={collab_agent!r}"
            )
            if m:
                m.collab_established += 1
                m.add_event(
                    kind="tool_result_establish_collaboration",
                    agent_id=aid,
                    iteration=meta.get("iteration"),
                    step=state["step"],
                    tool="establish_collaboration",
                    collab_agent_id=collab_profile.agent_id,
                    collab_agent_ids=list(collab_agent_ids),
                )
            return {
                "messages": [HumanMessage(content="Observation: Collaboration established.")],
                "step": state["step"] + 1,
                "script": state.get("script", "") or "",
                "collab_agent_ids": collab_agent_ids,
                "metadata": state.get("metadata", {}),
            }

        if tool_name == "communicate_with_collaborator":
            if not collab_agent or not collab_agent_state:
                return {
                    "messages": [HumanMessage(content="Observation: No collaborators.")],
                    "step": state["step"] + 1,
                    "script": state.get("script", "") or "",
                    "collab_agent_ids": state.get("collab_agent_ids", []),
                    "metadata": state.get("metadata", {}),
                }

            if code_collaboration:
                # ---- pair-programming mode ---------------------------------
                current_script = state.get("script", "") or ""
                base_message = tool_args.get("message", "")

                if current_script:
                    message_to_collab = (
                        base_message
                        + "\n\nHere is the current shared Python script; please revise it as "
                        "needed and reply with the FULL updated script inside a single "
                        "```python ... ``` block:\n" + "```python\n" + current_script + "\n```"
                    )
                else:
                    message_to_collab = (
                        base_message
                        + "\n\nThere is currently no shared script. If you propose one, reply "
                        "with the FULL script inside a single ```python ... ``` block."
                    )
            else:
                message_to_collab = tool_args.get("message", "")

            if m:
                m.collab_interactions += 1
                m.add_event(
                    kind="tool_call_communicate_with_collaborator",
                    agent_id=aid,
                    iteration=meta.get("iteration"),
                    step=state["step"],
                    tool="communicate_with_collaborator",
                    message=message_to_collab,
                )

            new_message = HumanMessage(content=message_to_collab)
            all_messages = collab_agent_state.get("messages", []) + [new_message]
            collab_input_state = {**collab_agent_state, "messages": all_messages}

            collab_response_state = collab_agent.invoke(collab_input_state)
            collab_agent_state = collab_response_state

            if collab_agent_state.get("last_ai_response"):
                response_content = collab_agent_state["last_ai_response"]

            if code_collaboration:
                # Try to read a new script from the collaborator’s reply
                new_script = _extract_code_block(response_content) or state.get("script", "") or ""
                collab_agent_state["script"] = new_script
                script_for_primary = new_script
                obs_msg = (
                    "Observation: Collaborator responded and the shared script has been "
                    f"{'updated' if new_script != (state.get('script', '') or '') else 'left unchanged'}."
                )
            else:
                # Original behaviour: don't touch script, just surface response
                script_for_primary = state.get("script", "") or ""
                obs_msg = f"Observation: {response_content}"

            if m:
                m.add_event(
                    kind="tool_result_communicate_with_collaborator",
                    agent_id=aid,
                    iteration=meta.get("iteration"),
                    step=state["step"],
                    tool="communicate_with_collaborator",
                    response=response_content,
                )
            return {
                "messages": [HumanMessage(content=obs_msg)],
                "step": state["step"] + 1,
                "script": script_for_primary,
                "collab_agent_ids": state.get("collab_agent_ids", []),
            }
        if tool_name == "search_agent_registry":
            # NOTE: this tool expects query_text/k (not agent_id).
            query_text = tool_args.get("query_text", "")
            if not isinstance(query_text, str) or not query_text.strip():
                if m:
                    m.invalid_actions += 1
                return {
                    "messages": [
                        HumanMessage(
                            content=(
                                "Observation: search_agent_registry requires a non-empty "
                                "'query_text' argument and optional 'k'."
                            )
                        )
                    ],
                    "step": state["step"] + 1,
                    "script": state.get("script", "") or "",
                    "collab_agent_ids": state.get("collab_agent_ids", []),
                }

            raw_k = tool_args.get("k", 3)
            try:
                k = int(raw_k)
            except Exception:
                k = 3
            if k <= 0:
                k = 3

            invoke_args = {"query_text": query_text.strip(), "k": k}
            try:
                obs_text = tool.invoke(invoke_args)
            except Exception as e:
                if m:
                    m.invalid_actions += 1
                obs_text = f"Error: search_agent_registry invocation failed: {e!s}"
            log.debug(f"[search_agent_registry] obs_text={obs_text}")

            obs_str = str(obs_text)
            if obs_str.lstrip().startswith("Observation:"):
                obs_msg = obs_str
            else:
                obs_msg = f"Observation: {obs_str}"

            if m:
                m.add_event(
                    kind="tool_result_search_agent_registry",
                    agent_id=aid,
                    iteration=meta.get("iteration"),
                    step=state["step"],
                    tool="search_agent_registry",
                    args=invoke_args,
                    observation=obs_msg,
                )
            return {
                "messages": [HumanMessage(content=obs_msg)],
                "step": state["step"] + 1,
                "script": state.get("script", "") or "",
                "collab_agent_ids": state.get("collab_agent_ids", []),
            }
        try:
            result = tool.invoke(tool_args)
        except Exception as e:
            # E.g. a ValidationError for a misnamed argument: the agent sees the error and can
            # retry, as a collaborator does, instead of its session ending.
            log.warning(f"[research_team][tool_error] agent={aid} tool={tool_name}: {e!r}")
            result = f"Error: {e!s}"
        obs_text = str(result)
        aid = state.get("metadata", {}).get("agent_id", "unknown")
        if m:
            m.add_event(
                kind="tool_result_other",
                agent_id=aid,
                iteration=meta.get("iteration"),
                step=state["step"],
                tool=tool_name,
                args=tool_args,
                observation=obs_text,
            )
        return {
            "messages": [HumanMessage(content=f"Observation: {obs_text}")],
            "step": state["step"] + 1,
            "script": state.get("script", "") or "",
            "collab_agent_ids": state.get("collab_agent_ids", []),
            "metadata": state.get("metadata", {}),
        }

    def check_final_report_node(state: PrimaryAgentState) -> dict[str, Any]:
        report_text = state.get("last_ai_response", "") or ""
        m = get_logs(agent)
        successful_execs = int(getattr(m, "code_executions_successful", 0) or 0) if m else 0
        code_script = state.get("script", "") or ""
        metadata = dict(state.get("metadata", {}) or {})
        known_plot_filenames = [
            str(x).strip()
            for x in (metadata.get("known_plot_filenames", []) or [])
            if str(x).strip()
        ]
        known_plot_handles = [
            str(x).strip() for x in (metadata.get("known_plot_handles", []) or []) if str(x).strip()
        ]
        plot_handle_to_filename = {
            str(k).strip(): str(v).strip()
            for k, v in (metadata.get("plot_handle_to_filename", {}) or {}).items()
            if str(k).strip() and str(v).strip()
        }
        known_plot_citations = [
            str(x).strip() for x in (metadata.get("plot_citations", []) or []) if str(x).strip()
        ]
        try:
            plot_only_rewrite_attempts = int(metadata.get("plot_only_rewrite_attempts", 0) or 0)
        except Exception:
            plot_only_rewrite_attempts = 0
        plot_only_rewrite_attempts = max(0, plot_only_rewrite_attempts)

        is_valid, errors, verdict = _validate_final_report_citations(
            report_text,
            successful_code_executions_current_iteration=successful_execs,
            code_script=code_script,
            known_plot_filenames=known_plot_filenames,
            known_plot_handles=known_plot_handles,
            plot_handle_to_filename=plot_handle_to_filename,
        )
        if (
            not is_valid
            and graph_logic_enabled
            and (known_plot_filenames or known_plot_handles or plot_handle_to_filename)
        ):
            repair = rewrite_plot_citation_tags(
                report_text,
                known_plot_filenames=known_plot_filenames,
                known_plot_handles=known_plot_handles,
                plot_handle_to_filename=plot_handle_to_filename,
                prefer_plot_handles=True,
            )
            repaired_text = str(repair.get("text", "") or "")
            if bool(repair.get("changed")) and repaired_text:
                repaired_valid, repaired_errors, repaired_verdict = (
                    _validate_final_report_citations(
                        repaired_text,
                        successful_code_executions_current_iteration=successful_execs,
                        code_script=code_script,
                        known_plot_filenames=known_plot_filenames,
                        known_plot_handles=known_plot_handles,
                        plot_handle_to_filename=plot_handle_to_filename,
                    )
                )
                if repaired_valid:
                    metadata.pop("plot_only_rewrite_attempts", None)
                    metadata["plot_citation_auto_repaired"] = True
                    if m:
                        m.add_event(
                            kind="plot_citation_auto_repaired",
                            agent_id=state.get("metadata", {}).get("agent_id", "unknown"),
                            iteration=state.get("metadata", {}).get("iteration"),
                            step=state.get("step", 1),
                            replacements=[dict(x) for x in (repair.get("replacements", []) or [])][
                                :20
                            ],
                        )
                    return {
                        "final_report_valid": True,
                        "final_report_errors": [],
                        "last_ai_response": repaired_text,
                        "metadata": metadata,
                    }
                if len(repaired_errors) < len(errors):
                    report_text = repaired_text
                    errors = repaired_errors
                    verdict = repaired_verdict
                    if m:
                        m.add_event(
                            kind="plot_citation_auto_repair_partial",
                            agent_id=state.get("metadata", {}).get("agent_id", "unknown"),
                            iteration=state.get("metadata", {}).get("iteration"),
                            step=state.get("step", 1),
                            replacements=[dict(x) for x in (repair.get("replacements", []) or [])][
                                :20
                            ],
                            remaining_errors=[str(e) for e in errors[:10]],
                        )

        if is_valid:
            metadata.pop("plot_only_rewrite_attempts", None)
            return {
                "final_report_valid": True,
                "final_report_errors": [],
                "metadata": metadata,
            }

        plot_only_errors = _is_plot_only_error_set(errors)
        if not plot_only_errors and plot_only_rewrite_attempts > 0:
            metadata["plot_only_rewrite_attempts"] = 0
        if plot_only_errors and allow_plot_validation_waiver_after_cap:
            if plot_only_rewrite_attempts >= plot_only_rewrite_cap:
                relaxed_valid, relaxed_errors, _ = _validate_final_report_citations(
                    report_text,
                    successful_code_executions_current_iteration=successful_execs,
                    code_script=code_script,
                    known_plot_filenames=known_plot_filenames,
                    known_plot_handles=known_plot_handles,
                    plot_handle_to_filename=plot_handle_to_filename,
                    enforce_plot_validation=False,
                )
                if relaxed_valid:
                    metadata["plot_validation_waived"] = True
                    metadata["plot_validation_waived_errors"] = [str(e) for e in errors[:20]]
                    if m:
                        m.add_event(
                            kind="plot_validation_waived",
                            agent_id=state.get("metadata", {}).get("agent_id", "unknown"),
                            iteration=state.get("metadata", {}).get("iteration"),
                            step=state.get("step", 1),
                            attempts=plot_only_rewrite_attempts,
                            cap=plot_only_rewrite_cap,
                            errors=[str(e) for e in errors[:5]],
                        )
                    return {
                        "final_report_valid": True,
                        "final_report_errors": [],
                        "metadata": metadata,
                    }
                if m:
                    m.add_event(
                        kind="plot_validation_waiver_rejected",
                        agent_id=state.get("metadata", {}).get("agent_id", "unknown"),
                        iteration=state.get("metadata", {}).get("iteration"),
                        step=state.get("step", 1),
                        attempts=plot_only_rewrite_attempts,
                        cap=plot_only_rewrite_cap,
                        errors=[str(e) for e in relaxed_errors[:5]],
                    )

        nudge_lines = [
            "Your final report citation tags are invalid.",
            "You must use exact in-line tags only:",
            "[Archive: paper_id]",
            "[Arxiv: paper title | author | year]",
            "[Code: brief description]",
            "[Plot: plot_token]",
            "For [Code: ...], use at most 5 words.",
        ]
        if known_plot_citations:
            nudge_lines.extend(
                [
                    "",
                    "Valid plot citation tokens from this iteration (copy exactly):",
                ]
            )
            for citation in known_plot_citations[:20]:
                nudge_lines.append(f"- {citation}")
        elif known_plot_handles:
            nudge_lines.extend(
                [
                    "",
                    "Valid plot citation tokens from this iteration (copy exactly):",
                ]
            )
            for handle in known_plot_handles[:20]:
                filename = plot_handle_to_filename.get(handle, "")
                if filename:
                    nudge_lines.append(f"- [Plot: {handle}] -> {filename}")
                else:
                    nudge_lines.append(f"- [Plot: {handle}]")
        elif known_plot_filenames:
            nudge_lines.extend(
                [
                    "",
                    "Valid plot filenames from this iteration:",
                ]
            )
            for plot_filename in known_plot_filenames[:20]:
                nudge_lines.append(f"- {plot_filename}")
        code_evidence_errors = [
            str(e) for e in (verdict.get("code_evidence", {}) or {}).get("errors", [])
        ]
        if code_evidence_errors:
            nudge_lines.extend(
                [
                    "",
                    "Code citation rule:",
                    "- You may use [Code: ...] only if you successfully executed code in this iteration and the resulting script is present.",
                    "- Prior-iteration or prior-paper code does not qualify; rerun now in this iteration.",
                ]
            )
        if errors:
            nudge_lines.append("Detected issues:")
            for err in errors[:5]:
                nudge_lines.append(f"- {err}")
        if plot_only_errors:
            metadata["plot_only_rewrite_attempts"] = plot_only_rewrite_attempts + 1
            if m:
                m.add_event(
                    kind="plot_only_rewrite_attempt",
                    agent_id=state.get("metadata", {}).get("agent_id", "unknown"),
                    iteration=state.get("metadata", {}).get("iteration"),
                    step=state.get("step", 1),
                    attempts=metadata["plot_only_rewrite_attempts"],
                    cap=plot_only_rewrite_cap,
                    errors=[str(e) for e in errors[:5]],
                )
        nudge_lines.append(
            "Rewrite the Final Report now, preserving substance but fixing citation tags exactly."
        )
        return {
            "messages": [HumanMessage(content="\n".join(nudge_lines))],
            "step": state.get("step", 1) + 1,
            "final_report_valid": False,
            "final_report_errors": errors,
            "script": state.get("script", "") or "",
            "collab_agent_ids": state.get("collab_agent_ids", []),
            "metadata": metadata,
        }

    g = StateGraph(PrimaryAgentState)
    g.add_node("think", think_node)
    g.add_node("act", act_node)
    g.add_node("check_final_report", check_final_report_node)
    g.set_entry_point("think")

    def _route_from_think(state: PrimaryAgentState) -> str:
        if parse_action(state["last_ai_response"]):
            return "act"
        return "check_final_report"

    g.add_conditional_edges(
        "think",
        _route_from_think,
        {"act": "act", "check_final_report": "check_final_report"},
    )
    g.add_edge("act", "think")

    def _route_from_report_check(state: PrimaryAgentState) -> str:
        # Only a valid report ends the session. The step limit is the graph's recursion limit
        # (system.graph_limits), set by the caller.
        return END if state.get("final_report_valid") else "think"

    g.add_conditional_edges(
        "check_final_report",
        _route_from_report_check,
        {"think": "think", END: END},
    )

    agent = g.compile()
    attach_logs(agent, type="research")
    return agent, sandbox_session_id
