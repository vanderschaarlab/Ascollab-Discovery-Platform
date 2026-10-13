import asyncio
import logging
from collections.abc import Callable, Mapping
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langsmith import traceable
from omegaconf import OmegaConf
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from agents.review.rev_agent_utils import (
    make_reviewer_user_message,
    parse_review_text,
    review_citation_schema_nudge,
    validate_review_citation_schema,
)
from agents.review.reviewer_agent import (
    get_reviewer_agent,
)
from stores.agent_registry import AgentRegistry
from stores.private_paper_archive import PrivatePaperArchive
from system.graph_limits import recursion_limit_for
from system.llm_retry import ainvoke_with_content_filter_retry
from system.plot_summary_artifacts import normalize_graph_logic_cfg
from system.reviewer_matching import assign_papers_to_reviewers
from system.schema import AgentNetworkMetadata, PaperMetadataPrivate, PaperReview
from system.trackers import _CURRENT_LOG, SessionStats, get_logs

_console = Console(stderr=True, log_path=False, record=False)
log = logging.getLogger(__name__)


def _build_assignment_metadata(
    *,
    strategy: str,
    seed: int,
    condition_label: str | None,
    num_reviewers_per_paper: int,
    allow_self_review: bool,
    excluded_reviewer_ids: set[str],
    assignments: dict[str, list[str]],
    papers_by_id: dict[str, PaperMetadataPrivate],
) -> dict[str, Any]:
    """Build a serializable reviewer-assignment record with diagnostics."""
    reviewer_load: dict[str, int] = {}
    unique_reviewers: set[str] = set()
    self_review_violations = 0
    underfilled = 0

    for paper_id, reviewer_ids in assignments.items():
        if len(reviewer_ids) < num_reviewers_per_paper:
            underfilled += 1
        paper = papers_by_id.get(paper_id)
        author_id = getattr(paper, "primary_agent_id", None)
        for reviewer_id in reviewer_ids:
            unique_reviewers.add(str(reviewer_id))
            reviewer_load[str(reviewer_id)] = reviewer_load.get(str(reviewer_id), 0) + 1
            if author_id and reviewer_id == author_id and not allow_self_review:
                self_review_violations += 1

    return {
        "strategy": str(strategy or "expertise_matching"),
        "seed": int(seed),
        "condition_label": condition_label,
        "num_reviewers_per_paper": int(num_reviewers_per_paper),
        "allow_self_review": bool(allow_self_review),
        "excluded_reviewer_ids": sorted(str(x) for x in (excluded_reviewer_ids or set())),
        "assignments": {str(pid): [str(rid) for rid in rids] for pid, rids in assignments.items()},
        "diagnostics": {
            "num_papers": len(assignments),
            "num_unique_reviewers": len(unique_reviewers),
            "reviewer_load": dict(sorted(reviewer_load.items())),
            "num_self_review_violations": int(self_review_violations),
            "num_underfilled_papers": int(underfilled),
        },
    }


def _extract_llm_runtime_metadata(llm_obj: Any) -> dict[str, Any]:
    deployment = None
    for attr in ("azure_deployment", "deployment_name", "model_name", "model", "deployment"):
        value = getattr(llm_obj, attr, None)
        if value is not None and str(value).strip():
            deployment = str(value).strip()
            break
    reasoning_effort = getattr(llm_obj, "reasoning_effort", None)
    model_kwargs = getattr(llm_obj, "model_kwargs", None)
    if isinstance(model_kwargs, Mapping):
        if reasoning_effort in (None, ""):
            reasoning_effort = model_kwargs.get("reasoning_effort", reasoning_effort)
        if deployment is None:
            model_name = model_kwargs.get("model")
            if model_name is not None and str(model_name).strip():
                deployment = str(model_name).strip()
    return {
        "deployment": (str(deployment).strip() if deployment is not None else None),
        "reasoning_effort": (
            str(reasoning_effort).strip()
            if reasoning_effort is not None and str(reasoning_effort).strip()
            else None
        ),
    }


def _build_compute_requirements(
    *,
    session_type: str,
    backend_key: str | None,
    llm_runtime_metadata: dict[str, Any] | None,
    default_llm_runtime_metadata: dict[str, Any] | None,
    session_timeout_seconds: int | float | None,
    recursion_limit: int | None,
    sandbox_required: bool,
    sandbox_requirements: dict[str, Any] | None,
) -> dict[str, Any]:
    llm_meta = dict(default_llm_runtime_metadata or {})
    for key, value in (llm_runtime_metadata or {}).items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        llm_meta[key] = value

    sandbox_meta = dict(sandbox_requirements or {})
    limits_meta = dict(sandbox_meta.get("limits", {}) or {})
    return {
        "session_type": str(session_type),
        "llm": {
            "backend_key": str(backend_key or "default"),
            "deployment": llm_meta.get("deployment"),
            "reasoning_effort": llm_meta.get("reasoning_effort"),
        },
        "runtime_limits": {
            "session_timeout_seconds": (
                int(session_timeout_seconds)
                if isinstance(session_timeout_seconds, (int, float))
                else None
            ),
            "recursion_limit": (int(recursion_limit) if recursion_limit is not None else None),
        },
        "sandbox": {
            "required": bool(sandbox_required),
            "limits": {
                "cpus": limits_meta.get("cpus"),
                "memory": limits_meta.get("memory"),
                "pids": limits_meta.get("pids"),
                "network_disabled": limits_meta.get("network_disabled"),
            },
            "max_sessions": sandbox_meta.get("max_sessions"),
            "admission_limit": sandbox_meta.get("admission_limit"),
        },
    }


def _resolve_assignment_strategy(
    *,
    matching_strategy: str,
    reviewer_assignment_strategy: str,
) -> str:
    """Resolve strategy from legacy and new config keys with safe precedence."""
    legacy = str(matching_strategy or "registry").strip().lower()
    modern = str(reviewer_assignment_strategy or "expertise_matching").strip().lower()
    if modern not in {"", "expertise_matching", "matching", "expertise", "registry"}:
        return modern
    if legacy in {"random"}:
        return "random"
    if legacy in {"registry"}:
        return "expertise_matching"
    return modern or "expertise_matching"


def _resolve_assignment_seed(
    *,
    matching_seed: int | None,
    reviewer_assignment_seed: int,
) -> int:
    """Resolve the reviewer-assignment seed with explicit, visible precedence.

    Args:
        matching_seed: Deprecated ``review.matching_seed``; wins when supplied.
        reviewer_assignment_seed: Preferred ``review.reviewer_assignment_seed``.

    Returns:
        The seed to use for reviewer assignment.
    """
    if matching_seed is None:
        return int(reviewer_assignment_seed)
    log.info(
        "[reviewer_matching] using deprecated review.matching_seed=%s "
        "(overrides review.reviewer_assignment_seed=%s)",
        matching_seed,
        reviewer_assignment_seed,
    )
    return int(matching_seed)


async def _wrap(agent_runnable, coro, progress, task_id):
    """Run a reviewer coroutine while wiring logs and advancing progress.

    Args:
        agent_runnable: The reviewer agent runnable whose logs should be surfaced.
        coro: Awaitable representing the reviewer execution.
        progress: Rich progress instance to update.
        task_id: Task identifier within the progress bar.

    Returns:
        The awaited result of `coro`.
    """
    token = None
    try:
        logs = get_logs(agent_runnable)
        if logs:
            token = _CURRENT_LOG.set(logs)
        return await coro
    finally:
        if token is not None:
            _CURRENT_LOG.reset(token)
        logs = get_logs(agent_runnable)
        if logs:
            logs.finish()
        progress.update(task_id, advance=1)


async def run_one_reviewer(
    agent: Any,
    state: Any,
    reviewer_id: str,
    paper_id: str,
    iteration: int,
    recursion_limit: int,
    run_name: str | None = None,
    total_session_timeout_seconds: int | None = None,
):
    """Invoke a single reviewer agent with tracing and optional timeout.

    Args:
        agent: Reviewer agent runnable.
        state: Initial agent state.
        reviewer_id: ID of the reviewer agent.
        paper_id: Target paper ID.
        iteration: Current iteration number.
        recursion_limit: Max recursion depth for the agent graph.
        run_name: Optional run name for tracing.
        total_session_timeout_seconds: Optional wall-clock timeout for the review.

    Returns:
        Structured review state or timeout/error payload.
    """
    configured = agent.with_config(
        {
            "run_name": f"{reviewer_id}_t{iteration}_{paper_id}_review",
            "metadata": {"reviewer_id": reviewer_id, "paper_id": paper_id, "iteration": iteration},
            "recursion_limit": recursion_limit,
        }
    )
    traced_invoke = traceable(
        name=f"review_t{iteration}_{paper_id}_{reviewer_id}_trace",
        run_type="chain",
        metadata={"reviewer_id": reviewer_id, "paper_id": paper_id, "iteration": iteration},
        tags=[f"reviewer_{reviewer_id}", f"paper_{paper_id}", f"iteration_{iteration}"],
        project_name=run_name,
    )(configured.ainvoke)

    if total_session_timeout_seconds:
        try:
            return await asyncio.wait_for(
                traced_invoke(state), timeout=total_session_timeout_seconds
            )
        except TimeoutError:
            return {
                "messages": state.get("messages", []) + [HumanMessage(content="TIMEOUT.")],
                "step": state.get("step", 1),
                "last_ai_response": "Review timeout",
                "parsed_review": {
                    "Review Text": "TIMEOUT",
                    "Overall Score": "N/A",
                    "Decision": "Timeout",
                },
                "valid_review": False,
                "route": "timeout",
            }
    else:
        try:
            return await traced_invoke(state)
        except Exception as e:
            # Convert unexpected failures (e.g., GraphRecursionError) into a structured result
            return {
                "messages": state.get("messages", []),
                "step": state.get("step", 1),
                "last_ai_response": f"ERROR: {type(e).__name__}: {e}",
                "parsed_review": None,
                "valid_review": False,
                "route": "error",
                "error": f"{type(e).__name__}: {e}",
            }


async def _retry_review_for_citation_schema(
    *,
    llm_model: BaseChatModel,
    review_text: str,
    citation_errors: list[str],
    stage_label: str,
    require_assertion_justification: bool,
) -> tuple[dict[str, Any] | None, bool]:
    """Re-prompt a reviewer whose output failed the strict citation schema.

    Args:
        llm_model: Model to re-prompt.
        review_text: The rejected review text.
        citation_errors: Validator errors, used to build the nudge.
        stage_label: Label for retry telemetry.
        require_assertion_justification: Passed through to the validator.

    Returns:
        Tuple of (parsed_review, ok). `ok` is False when the retry also failed.
    """
    retry_prompt = [
        SystemMessage(
            content=(
                "You are an expert scientific reviewer. "
                "Rewrite the review exactly in the required format."
            )
        ),
        HumanMessage(content=review_text),
        HumanMessage(
            content=(
                "Your review did not follow the strict citation schema.\n"
                f"{review_citation_schema_nudge(citation_errors)}\n"
                "Rewrite the full review now with valid citation tags only."
            )
        ),
    ]
    retry = await ainvoke_with_content_filter_retry(
        llm_model, retry_prompt, stage_label=stage_label, logger=log
    )
    retry_text = retry.content if hasattr(retry, "content") else ""
    parsed_retry, ok_retry = parse_review_text(retry_text)
    # Validate the SAME text the caller will store -- the parsed "Review Text"
    # section -- not the raw response. Validating the raw response would let a
    # retry pass by placing citation tags outside that section, which is not what
    # the main path checks.
    citation_ok_retry, _ = validate_review_citation_schema(
        (parsed_retry or {}).get("Review Text", ""),
        require_assertion_justification=require_assertion_justification,
    )
    return parsed_retry, bool(ok_retry and citation_ok_retry)


async def review_one_step(
    iteration: int,
    llm: BaseChatModel,
    llm_for: Callable[[str], BaseChatModel] | None,
    candidate_papers: list[PaperMetadataPrivate],
    network_metadata: AgentNetworkMetadata,
    private_archive: PrivatePaperArchive,
    agent_registry: AgentRegistry,
    agent_registry_search_tool: StructuredTool,
    task_config: OmegaConf,
    num_reviewers_per_paper: int,
    review_max_iterations: int,
    llm_backend_key_for: Callable[[str], str | None] | None = None,
    default_llm_runtime_metadata: dict[str, Any] | None = None,
    sandbox_requirements: dict[str, Any] | None = None,
    include_code_in_reviews: bool = True,
    total_session_timeout_seconds: int | None = None,
    run_name: str | None = None,
    paper_metrics_by_id: dict[str, dict[str, Any]] | None = None,
    metrics_allowlist: list[str] | None = None,
    include_metrics_guidance: bool = True,
    allow_self_review: bool = False,
    excluded_reviewer_ids: set[str] | None = None,
    require_assertion_justification: bool = False,
    graph_logic_cfg: dict[str, Any] | None = None,
    reviewer_norm_text: str | None = None,
    matching_strategy: str = "registry",
    matching_seed: int | None = None,
    random_matching_seed_offset: int = 0,
    reviewer_assignment_strategy: str = "expertise_matching",
    reviewer_assignment_seed: int = 0,
    reviewer_assignment_condition_label: str | None = None,
    return_assignment_metadata: bool = False,
) -> (
    tuple[dict[str, list[PaperReview]], list[SessionStats]]
    | tuple[dict[str, list[PaperReview]], list[SessionStats], dict[str, Any]]
):
    """Run the full review stage: assign reviewers and collect reviews.

    Args:
        iteration: Current iteration number.
        llm: Default LLM for reviewers.
        llm_for: Optional function returning an LLM for a reviewer_id.
        candidate_papers: Papers requiring review.
        network_metadata: Aggregate network metadata for prompting.
        private_archive: Private paper archive for persisting reviews.
        agent_registry: Registry of available reviewer agents.
        agent_registry_search_tool: Tool used for reviewer matching.
        task_config: Task configuration with tools/prompts.
        num_reviewers_per_paper: Desired number of reviewers per paper.
        review_max_iterations: Max iterations for reviewer agent graphs.
        include_code_in_reviews: Whether to include code artifacts in prompts.
        total_session_timeout_seconds: Optional wall-clock timeout per reviewer.
        run_name: Optional tracing run name.
        paper_metrics_by_id: Optional metrics to surface to reviewers keyed by paper_id.
        metrics_allowlist: Optional list of metric keys enabled for reviewer guidance/prompt.
        include_metrics_guidance: Whether the reviewer prompt describes the metrics and asks for
            ratings grounded in them. ``system.main`` passes False when
            ``review.metrics_pipeline`` is off (no metrics are computed); the review replay keeps
            the default.
        allow_self_review: Whether to allow primary authors to review their own papers.
        require_assertion_justification: If true, require every assertion to be justified with archive/arxiv/code evidence.
        matching_strategy: Legacy reviewer assignment strategy (`registry` or `random`).
        matching_seed: Optional legacy seed for random reviewer matching.
        random_matching_seed_offset: Legacy additional seed offset for random reviewer matching.
        reviewer_assignment_strategy: Reviewer assignment strategy (`expertise_matching` or `random`).
        reviewer_assignment_seed: Seed used by assignment dispatcher.
        reviewer_assignment_condition_label: Optional deterministic label for random assignment replay.
        return_assignment_metadata: When true, returns an additional assignment metadata dict.

    Returns:
        Tuple of (reviews_by_paper, review_logs), plus a third assignment-metadata
        element when ``return_assignment_metadata`` is true.
    """

    excluded = set(excluded_reviewer_ids or [])
    graph_cfg = normalize_graph_logic_cfg(graph_logic_cfg)
    graph_logic_enabled = bool(graph_cfg.get("enabled", False))
    resolved_strategy = _resolve_assignment_strategy(
        matching_strategy=matching_strategy,
        reviewer_assignment_strategy=reviewer_assignment_strategy,
    )
    resolved_seed = _resolve_assignment_seed(
        matching_seed=matching_seed,
        reviewer_assignment_seed=reviewer_assignment_seed,
    )
    resolved_condition_label = reviewer_assignment_condition_label
    if (
        resolved_condition_label is None
        and str(matching_strategy or "").strip().lower() == "random"
    ):
        resolved_condition_label = (
            f"legacy_random_iter{iteration}_offset{int(random_matching_seed_offset or 0)}"
        )

    paper_reviewer_map = assign_papers_to_reviewers(
        strategy=resolved_strategy,
        agent_search_tool=agent_registry_search_tool,
        num_reviewers_per_paper=num_reviewers_per_paper,
        papers=candidate_papers,
        agent_registry=agent_registry,
        exclude_reviewer_ids=list(excluded),
        allow_self_review=allow_self_review,
        seed=resolved_seed,
        condition_label=resolved_condition_label,
        iteration=iteration,
        seed_offset=int(random_matching_seed_offset or 0),
    )
    papers_dict = {paper.paper_id: paper for paper in candidate_papers}
    assignment_metadata = _build_assignment_metadata(
        strategy=resolved_strategy,
        seed=resolved_seed,
        condition_label=resolved_condition_label,
        num_reviewers_per_paper=num_reviewers_per_paper,
        allow_self_review=allow_self_review,
        excluded_reviewer_ids=excluded,
        assignments=paper_reviewer_map,
        papers_by_id=papers_dict,
    )

    for paper_id, reviewer_ids in paper_reviewer_map.items():
        if len(reviewer_ids) < num_reviewers_per_paper:
            log.warning(
                f"[reviewer_matching] only {len(reviewer_ids)}/{num_reviewers_per_paper} reviewers available for {paper_id}"
            )

    reviewer_tasks = []
    task_metadata = []
    llm_runtime_meta_by_reviewer: dict[str, dict[str, Any]] = {}
    review_tools_cfg = getattr(task_config, "collab_agent_tools", {}) or {}
    review_has_execute_code_tool = (
        isinstance(review_tools_cfg, Mapping) and "execute_code" in review_tools_cfg
    )
    review_sandbox_required = bool(include_code_in_reviews or review_has_execute_code_tool)

    for paper_id, reviewer_ids in paper_reviewer_map.items():
        paper = papers_dict[paper_id]

        for reviewer_id in reviewer_ids:
            if reviewer_id in excluded:
                continue
            exists, reviewer_profile = agent_registry.get_agent(reviewer_id)
            if not exists or reviewer_profile is None:
                log.warning(
                    f"[review_one_step] SKIP: reviewer_id '{reviewer_id}' not found in registry"
                )
                continue
            # Pick per-reviewer LLM if provided; fall back to default
            active_llm = llm_for(reviewer_id) if llm_for else llm
            llm_runtime_meta_by_reviewer[reviewer_id] = _extract_llm_runtime_metadata(active_llm)
            metrics_for_paper = None
            if paper_metrics_by_id is not None:
                metrics_for_paper = paper_metrics_by_id.get(paper_id)
            reviewer_agent = get_reviewer_agent(
                active_llm,
                reviewer_agent_profile=reviewer_profile,
                reviewer_tools=task_config.collab_agent_tools,
                agent_network_metadata=network_metadata,
                require_assertion_justification=require_assertion_justification,
                metrics_allowlist=metrics_allowlist,
                community_norms=reviewer_norm_text,
                include_metrics_guidance=include_metrics_guidance,
            )

            paper_kind = str(getattr(paper, "paper_kind", "research") or "research").strip().lower()
            meta_reviewer_context = None
            if paper_kind == "meta_research":
                source_ids = [
                    str(x).strip()
                    for x in (getattr(paper, "meta_research_source_paper_ids", []) or [])
                    if str(x).strip()
                ]
                meta_reviewer_context = {
                    "paper_type": "meta_research synthesis paper",
                    "source_paper_ids": source_ids,
                    "expectations": [
                        "Evaluate breadth and integration quality across source findings.",
                        "Evaluate contradiction/tension handling quality and uncertainty framing.",
                        "Evaluate evidence traceability of synthesis claims to archive citations.",
                        "Evaluate coherence of the overarching synthesis narrative.",
                    ],
                }

            reviewer_message = make_reviewer_user_message(
                report_text=paper.manuscript,
                code_report=paper.code_script if include_code_in_reviews else None,
                metrics=metrics_for_paper,
                plot_artifacts=(
                    list(getattr(paper, "plot_artifacts", []) or [])
                    if graph_logic_enabled
                    else None
                ),
                meta_research_context=meta_reviewer_context,
            )

            reviewer_state = {
                "messages": [HumanMessage(content=reviewer_message)],
                "step": 1,
                "last_ai_response": "",
                "parsed_review": None,
                "valid_review": None,
                "metadata": {
                    "reviewer_id": reviewer_id,
                    "paper_id": paper_id,
                    "iteration": iteration,
                },
            }
            try:
                llm_key = getattr(reviewer_profile, "llm_key", None)
                if llm_key:
                    # propagate to the agent runtime via config metadata as well
                    reviewer_state.setdefault("metadata", {})
                    reviewer_state["metadata"]["llm_key"] = llm_key
            except Exception:
                pass

            task = run_one_reviewer(
                reviewer_agent,
                reviewer_state,
                reviewer_id,
                paper_id,
                iteration,
                recursion_limit=recursion_limit_for(review_max_iterations),
                run_name=run_name,
                total_session_timeout_seconds=total_session_timeout_seconds,
            )

            reviewer_tasks.append((task, reviewer_agent))
            task_metadata.append((paper_id, reviewer_id, reviewer_agent, reviewer_profile))

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]reviews running...[/]"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        transient=True,
        console=_console,
    ) as progress:
        task_id = progress.add_task("reviewers", total=len(reviewer_tasks))
        results = await asyncio.gather(
            *[
                _wrap(reviewer_agent, task, progress, task_id)
                for (task, reviewer_agent) in reviewer_tasks
            ],
            return_exceptions=True,
        )

    review_logs = []
    collected_reviews: dict[str, list[dict[str, Any]]] = {}
    # (paper_id, reviewer_id, review_text, citation_errors) awaiting a schema-nudge
    # retry. Retried as one concurrent batch after the result loop so N failures cost
    # one round-trip, not N.
    pending_citation_retries: list[tuple[str, str, str, list[str]]] = []
    for result_idx, (paper_id, reviewer_id, reviewer_agent, reviewer_profile) in enumerate(
        task_metadata
    ):
        logs = get_logs(reviewer_agent)
        if logs:
            # attach llm_key if available for downstream analysis
            extra_meta = {}
            try:
                llm_key = getattr(reviewer_profile, "llm_key", None)
                if llm_key:
                    extra_meta["llm_key"] = llm_key
            except Exception:
                pass
            llm_runtime_meta = llm_runtime_meta_by_reviewer.get(reviewer_id, {})
            backend_key = (
                llm_backend_key_for(reviewer_id)
                if llm_backend_key_for
                else extra_meta.get("llm_key")
            )
            if not backend_key:
                backend_key = "default"
            logs.compute_requirements = _build_compute_requirements(
                session_type="review",
                backend_key=backend_key,
                llm_runtime_metadata=llm_runtime_meta,
                default_llm_runtime_metadata=default_llm_runtime_metadata,
                session_timeout_seconds=total_session_timeout_seconds,
                recursion_limit=recursion_limit_for(review_max_iterations),
                sandbox_required=review_sandbox_required,
                sandbox_requirements=sandbox_requirements,
            )

            review_logs.append(
                SessionStats(
                    iteration=iteration,
                    session_id=f"{reviewer_id}_{paper_id}",
                    session_agent_id=reviewer_id,
                    type="review",
                    stats_logs=logs,
                )
            )
    reviews_by_paper: dict[str, list[PaperReview]] = {}

    for result_idx, result in enumerate(results):
        paper_id, reviewer_id, _, reviewer_profile = task_metadata[result_idx]

        # If a reviewer crashed or hit recursion limit, we get an Exception object
        if isinstance(result, Exception):
            log.error(
                f"[review_one_step] reviewer '{reviewer_id}' on paper '{paper_id}' -> {result!r}"
            )
            continue

        # Timeouts or other routes should be handled gracefully
        route = result.get("route")
        if route == "timeout":
            log.warning(
                f"[review_one_step] TIMEOUT: reviewer '{reviewer_id}' on paper '{paper_id}'"
            )
            continue
        if route == "error":
            log.error(
                f"[review_one_step] ERROR route: reviewer '{reviewer_id}' on paper '{paper_id}': {result.get('error')}"
            )
            continue

        parsed_review = result.get("parsed_review")
        if not isinstance(parsed_review, dict):
            log.warning(
                f"[review_one_step] reviewer '{reviewer_id}' returned non-dict parsed_review for paper '{paper_id}': {type(parsed_review).__name__}"
            )
            continue

        # Extract fields correctly — PaperReview.review_text is a string
        review_text = parsed_review.get("Review Text", "")
        citation_ok, citation_errors = validate_review_citation_schema(
            review_text,
            require_assertion_justification=require_assertion_justification,
        )
        if not citation_ok:
            log.warning(
                f"[review_one_step] reviewer '{reviewer_id}' returned invalid citation schema "
                f"for paper '{paper_id}': {citation_errors[:3]} - queued for schema-nudge retry"
            )
            pending_citation_retries.append((paper_id, reviewer_id, review_text, citation_errors))
            continue
        # Keep only scalar-ish scores; skip the free-text
        scores = {k: v for k, v in parsed_review.items() if k != "Review Text"}

        collected_reviews.setdefault(paper_id, []).append(
            {
                "reviewer_id": reviewer_id,
                "review_text": review_text,
                "scores": scores,
            }
        )

    if pending_citation_retries:
        log.warning(
            f"[review_one_step] retrying {len(pending_citation_retries)} review(s) "
            "that failed the citation schema"
        )
        retry_outcomes = await asyncio.gather(
            *[
                _retry_review_for_citation_schema(
                    llm_model=(llm_for(reviewer_id) if llm_for else llm),
                    review_text=text,
                    citation_errors=errs,
                    stage_label=f"review_citation_retry.paper={paper_id}.reviewer={reviewer_id}",
                    require_assertion_justification=require_assertion_justification,
                )
                for (paper_id, reviewer_id, text, errs) in pending_citation_retries
            ],
            return_exceptions=True,
        )
        for (paper_id, reviewer_id, _text, _errs), outcome in zip(
            pending_citation_retries, retry_outcomes, strict=True
        ):
            if isinstance(outcome, BaseException):
                log.warning(
                    f"[review_one_step] DROP: citation retry raised for reviewer "
                    f"'{reviewer_id}' on paper '{paper_id}': {outcome!r}"
                )
                continue
            parsed_retry, retry_ok = outcome
            if not retry_ok or not isinstance(parsed_retry, dict):
                log.warning(
                    f"[review_one_step] DROP: reviewer '{reviewer_id}' failed citation schema "
                    f"after retry for paper '{paper_id}'"
                )
                continue
            collected_reviews.setdefault(paper_id, []).append(
                {
                    "reviewer_id": reviewer_id,
                    "review_text": parsed_retry.get("Review Text", ""),
                    "scores": {k: v for k, v in parsed_retry.items() if k != "Review Text"},
                }
            )

    # Persist final reviews to the PRIVATE archive
    try:
        private_archive.add_reviews_for_iteration(
            iteration=iteration,
            reviews_by_paper=collected_reviews,
        )
    except Exception as e:
        log.warning(f"[review] failed to persist reviews to private archive: {e}")

    # Convert reviews into PaperReview objects
    for paper_id, entries in collected_reviews.items():
        for entry in entries:
            review_text = entry["review_text"]
            scores = entry["scores"]

            paper_review = PaperReview(
                paper_id=paper_id,
                reviewer_agent_id=entry["reviewer_id"],
                review_text=review_text,
                score=scores,
            )
            reviews_by_paper.setdefault(paper_id, []).append(paper_review)

    log.info(f"[review] papers passed to meta-review: {len(reviews_by_paper)}")
    for pid, revs in list(reviews_by_paper.items())[:6]:
        log.debug(f"[review] {pid}: {len(revs)} reviews")

    if return_assignment_metadata:
        return reviews_by_paper, review_logs, assignment_metadata
    return reviews_by_paper, review_logs
