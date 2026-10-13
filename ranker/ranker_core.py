"""Core ``run_ranker`` function -- the ranker's programmatic entry point."""

import logging
import time
from typing import Any
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from .ranker_agent import build_ranker_agent
from .ranker_schemas import RankerResult, RankerTrace
from .ranker_tools import build_ranker_tools
from .ranker_utils import (
    build_section_notes_block,
    get_system_prompt_builder,
    get_verdict_parser,
    make_ranker_user_message,
)

log = logging.getLogger(__name__)


def _build_trace(
    tracker: dict[str, Any],
    ranker_id: str,
    start_time: float,
    end_time: float,
    paper_a_title: str,
    paper_b_title: str,
) -> RankerTrace:
    """Build a RankerTrace from tracker data and timing info."""
    events = tracker.get("events", [])

    # Build summary
    llm_calls = sum(1 for e in events if e["kind"] == "llm_call")
    tool_calls_total = sum(1 for e in events if e["kind"] == "tool_call")
    tool_calls_by_name: dict[str, int] = {}
    for e in events:
        if e["kind"] == "tool_call":
            name = e["tool"]
            tool_calls_by_name[name] = tool_calls_by_name.get(name, 0) + 1

    nudge_retries = sum(
        1 for e in events if e["kind"] == "verdict_parse" and e.get("retry_index", 0) > 0
    )

    summary = {
        "llm_calls": llm_calls,
        "tool_calls": tool_calls_total,
        "tool_calls_by_name": tool_calls_by_name,
        "nudge_retries": nudge_retries,
        "total_messages": tracker.get("turns", 0),
        "input_tokens": tracker.get("input_tokens", 0),
        "output_tokens": tracker.get("output_tokens", 0),
    }

    return RankerTrace(
        ranker_id=ranker_id,
        started_at=start_time,
        ended_at=end_time,
        duration_seconds=round(end_time - start_time, 3),
        paper_a_title=paper_a_title,
        paper_b_title=paper_b_title,
        events=events,
        summary=summary,
    )


def _get_score_label_fn(mode: str):
    """Return a function mapping score int to outcome label string."""
    if mode == "hypothesis_similarity":
        return lambda s: "different" if s == 1 else "same"
    return lambda s: "A" if s == 1 else "B"


def run_ranker(
    paper_a: dict[str, str],
    paper_b: dict[str, str],
    llm: BaseChatModel,
    criteria_prompt: str | None = None,
    mode: str = "comparison",
    tools_cfg: dict[str, Any] | None = None,
    sandbox_client: Any = None,
    max_iterations: int = 10,
    llm_retry_cfg: dict[str, Any] | None = None,
    max_parse_retries: int = 2,
    ranker_id: str | None = None,
    include_code: bool = True,
    include_dataset_description: bool = True,
    dataset_descriptions_a: dict[str, str] | None = None,
    dataset_descriptions_b: dict[str, str] | None = None,
    include_cited_papers: bool = True,
    cited_papers_a: list[dict[str, str]] | None = None,
    cited_papers_b: list[dict[str, str]] | None = None,
    cited_papers_order: str = "newest_first",
    paper_separator: str | None = None,
    prompt_section_notes: dict[str, str] | None = None,
) -> RankerResult:
    """Run the ranker on a single pair of papers.

    This is the primary public API. It builds an agent graph, invokes it,
    and returns a structured result. Fully stateless between invocations.

    Args:
        paper_a: Dict with keys "title", "abstract", "manuscript", and
            optionally "code".
        paper_b: Same structure as paper_a.
        llm: Pre-built chat model instance for the judge.
        criteria_prompt: Evaluation criteria text. Defaults depend on mode.
        mode: Ranker mode — ``"comparison"`` (default) or
            ``"hypothesis_similarity"``.
        tools_cfg: Optional dict with "arxiv" and/or "sandbox" sub-dicts.
            Each should have an "enabled" bool. Example::

                {"arxiv": {"enabled": True, "top_k_results": 5}}

        sandbox_client: Optional SandboxClient instance. Required if
            sandbox tool is enabled. Shared across invocations (each
            invocation creates its own session).
        max_iterations: Maximum think-act loops before forcing a verdict.
        llm_retry_cfg: Retry config dict with ``max_retries``,
            ``base_delay_seconds``, ``max_delay_seconds``.
        max_parse_retries: Number of nudge retries on verdict parse failure.
        ranker_id: Optional identifier for this invocation. Defaults to
            auto-generated ``ranker_<hex>``.
        include_code: Whether to include the code section in the prompt.
        include_dataset_description: Whether to include dataset descriptions.
        dataset_descriptions_a: Dataset key -> description for Paper A.
        dataset_descriptions_b: Dataset key -> description for Paper B.
        include_cited_papers: Whether to include cited papers sections.
        cited_papers_a: Ordered ancestor paper dicts for Paper A.
        cited_papers_b: Ordered ancestor paper dicts for Paper B.
        cited_papers_order: ``"newest_first"`` (default) or ``"oldest_first"``.
        paper_separator: Visual separator between Paper A and Paper B.

    Returns:
        RankerResult with score, justification, raw_response, tool_calls,
        turns, token counts, and trace. When no verdict could be parsed the
        result carries ``valid_result=False`` and ``score=None`` rather than
        defaulting the score to 0.
    """
    if ranker_id is None:
        ranker_id = f"ranker_{uuid4().hex[:8]}"

    paper_a_title = paper_a.get("title", "N/A")
    paper_b_title = paper_b.get("title", "N/A")

    log.info("Starting ranker: paper_a=%r vs paper_b=%r", paper_a_title, paper_b_title)

    if criteria_prompt is None:
        if mode == "hypothesis_similarity":
            criteria_prompt = (
                "Determine whether the two papers test the same hypothesis or "
                "different hypotheses.\n\n"
                'Two hypotheses should be considered "the same" if they address '
                "the same core research question and test the same causal or "
                "relational claim, even if the specific methods, datasets, or "
                "experimental details differ.\n\n"
                'Two hypotheses should be considered "different" if they address '
                "fundamentally different research questions or test different "
                "causal/relational claims."
            )
        else:
            criteria_prompt = (
                "Compare the two papers on the following dimensions:\n"
                "1. Scientific rigor and methodological soundness\n"
                "2. Strength and quality of evidence\n"
                "3. Novelty and originality of contributions\n"
                "4. Significance and potential impact\n"
                "5. Clarity and quality of writing\n\n"
                "Weigh all dimensions equally unless one paper has a critical flaw\n"
                "(e.g., unsound methodology), in which case that should dominate."
            )

    if tools_cfg is None:
        tools_cfg = {}
    if llm_retry_cfg is None:
        llm_retry_cfg = {}

    # Build tools
    tools, cleanup_fn = build_ranker_tools(tools_cfg, sandbox_client)
    tool_names = list(tools.keys()) if tools else []
    log.info("Tools enabled: %s", tool_names or "none")

    # Build tool signatures for the prompt
    tool_signatures = {}
    for name, info in tools.items():
        tool_signatures[name] = info["tool_signature"]

    # Build section notes block
    section_notes_block = build_section_notes_block(
        include_code=include_code,
        include_dataset_description=include_dataset_description,
        include_cited_papers=include_cited_papers,
        section_notes=prompt_section_notes,
    )

    # Build system prompt (mode-dispatched)
    system_prompt_builder = get_system_prompt_builder(mode)
    system_prompt = system_prompt_builder(
        criteria_prompt,
        tool_signatures or None,
        section_notes_block=section_notes_block,
    )

    # Build user message
    user_message = make_ranker_user_message(
        paper_a,
        paper_b,
        include_code=include_code,
        include_dataset_description=include_dataset_description,
        dataset_descriptions_a=dataset_descriptions_a,
        dataset_descriptions_b=dataset_descriptions_b,
        include_cited_papers=include_cited_papers,
        cited_papers_a=cited_papers_a,
        cited_papers_b=cited_papers_b,
        cited_papers_order=cited_papers_order,
        paper_separator=paper_separator,
    )

    # Build the agent graph (mode-dispatched parser and nudge)
    parse_fn, nudge = get_verdict_parser(mode)
    agent = build_ranker_agent(
        llm=llm,
        system_prompt=system_prompt,
        max_iterations=max_iterations,
        tools=tools,
        llm_retry_cfg=llm_retry_cfg,
        max_parse_retries=max_parse_retries,
        parse_verdict_fn=parse_fn,
        nudge_message=nudge,
        score_label_fn=_get_score_label_fn(mode),
    )

    start_time = time.time()

    try:
        # Invoke the graph
        log.info("Invoking ranker agent (max_iterations=%d)", max_iterations)

        initial_state = {
            "messages": [HumanMessage(content=user_message)],
            "step": 0,
            "last_ai_response": "",
            "parsed_result": None,
            "valid_result": None,
        }

        final_state = agent.invoke(initial_state)

        end_time = time.time()

        # Extract results from tracker
        tracker = agent._ranker_tracker

        parsed = final_state.get("parsed_result") or {}
        valid_result = bool(final_state.get("valid_result", False))
        score = parsed.get("score") if valid_result else None
        justification = parsed.get("justification", "")
        raw_response = final_state.get("last_ai_response", "")
        if not valid_result:
            log.warning(
                "Ranker %s produced no parseable verdict after %d turns; recording score=None",
                ranker_id,
                tracker.get("turns", 0),
            )

        # Build trace
        trace = _build_trace(
            tracker=tracker,
            ranker_id=ranker_id,
            start_time=start_time,
            end_time=end_time,
            paper_a_title=paper_a_title,
            paper_b_title=paper_b_title,
        )

        outcome = _get_score_label_fn(mode)(score) if valid_result else "unparsed"
        log.info(
            "Ranker finished: outcome=%s, turns=%d, tokens=%d/%d (in/out)",
            outcome,
            tracker.get("turns", 0),
            tracker.get("input_tokens", 0),
            tracker.get("output_tokens", 0),
        )

        return RankerResult(
            score=score,
            valid_result=valid_result,
            justification=justification,
            raw_response=raw_response,
            tool_calls=dict(tracker.get("tool_calls", {})),
            turns=tracker.get("turns", 0),
            input_tokens=tracker.get("input_tokens", 0),
            output_tokens=tracker.get("output_tokens", 0),
            trace=trace,
        )

    except Exception as exc:
        log.error("Ranker failed: %s", exc)
        raise

    finally:
        # Always clean up sandbox sessions
        cleanup_fn()
        log.debug("Cleanup complete")
