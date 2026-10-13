import asyncio
import logging
import random
from collections.abc import Callable, Mapping
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langsmith import traceable
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from agents.meta_review.meta_rev_agent_utils import make_metareviewer_user_message
from agents.meta_review.meta_reviewer_agent import get_metareviewer_agent
from system.schema import PaperMetadataPrivate, PaperMetaReview, PaperReview
from system.trackers import _CURRENT_LOG, SessionStats, get_logs

log = logging.getLogger(__name__)

_console = Console(stderr=True, log_path=False, record=False)


async def _wrap(agent_runnable, coro, progress, task_id):
    """Await a coroutine with tracker context and advance progress when done."""
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


def _build_metareview_session_id(*, meta_agent_id: str, iteration: int, tournament_id: int) -> str:
    safe_agent = str(meta_agent_id or "meta_reviewer").strip() or "meta_reviewer"
    return f"metareview_{safe_agent}_t{int(iteration):02d}_tour{int(tournament_id):03d}"


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
            # The meta-reviewer graph is one node, so it has no step limit.
            "recursion_limit": None,
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


async def run_one_tournament(
    agent: Any,
    state: Any,
    tournament_id: int,
    iteration: int,
    run_name: str | None = None,
    total_session_timeout_seconds: float | None = None,
):
    """Execute a single metareview tournament with tracing and timeout."""
    configured = agent.with_config(
        {
            "run_name": f"metareview_t{iteration}_tournament{tournament_id}",
            "metadata": {"tournament_id": tournament_id, "iteration": iteration},
        }
    )
    traced_invoke = traceable(
        name=f"metareview_t{iteration}_tournament{tournament_id}_trace",
        run_type="chain",
        metadata={"tournament_id": tournament_id, "iteration": iteration},
        tags=[f"tournament_{tournament_id}", f"iteration_{iteration}"],
        project_name=run_name,
    )(configured.ainvoke)

    if total_session_timeout_seconds:
        try:
            return await asyncio.wait_for(
                traced_invoke(state), timeout=total_session_timeout_seconds
            )
        except TimeoutError:
            return {
                "messages": state.get("messages", []),
                "parsed_reviews": {},
                "valid_metareview": False,
                "route": "timeout",
                "metadata": {
                    "tournament_id": tournament_id,
                    "iteration": iteration,
                    "status": "timeout",
                },
            }
    else:
        return await traced_invoke(state)


async def metareview_one_step(
    iteration: int,
    llm: BaseChatModel,
    llm_for: Callable[[str], BaseChatModel] | None,
    candidate_papers: list[PaperMetadataPrivate],
    reviews_by_paper: dict[str, list[PaperReview]],
    tournament_size: int,
    llm_backend_key_for: Callable[[str], str | None] | None = None,
    default_llm_runtime_metadata: dict[str, Any] | None = None,
    sandbox_requirements: dict[str, Any] | None = None,
    shuffle_before_tournament: bool = True,
    total_session_timeout_seconds: float | None = None,
    run_name: str | None = None,
    require_reviews: bool = True,
    review_free: bool = False,
    **_kwargs: Any,
) -> tuple[list[PaperMetaReview], list[SessionStats]]:
    """Run the metareview stage across candidate papers via tournaments.

    Args:
        iteration: Current iteration.
        llm: Default chat model for meta-reviewers.
        llm_for: Maps a meta-reviewer ID to its chat model (model mixes); ``llm`` when ``None``.
        candidate_papers: Papers submitted this iteration.
        reviews_by_paper: Peer reviews keyed by paper ID.
        tournament_size: Papers compared in each meta-review tournament.
        llm_backend_key_for: Maps a meta-reviewer ID to its backend key, for run metadata.
        default_llm_runtime_metadata: The default model's runtime metadata (deployment,
            reasoning effort): the base of each session's ``compute_requirements.llm``
            record, overridden field by field by the meta-reviewer's own model.
        sandbox_requirements: Sandbox limits copied into each session's
            ``compute_requirements`` metadata only. Meta-reviewers run no code and get no
            sandbox (``sandbox.required`` is False).
        shuffle_before_tournament: Shuffle papers before forming tournaments.
        total_session_timeout_seconds: Wall-clock cap per meta-review session.
        run_name: Tracing project name for the tournament traces (``traceable``'s
            ``project_name``); not part of session IDs.
        require_reviews: When True (default), only papers that received at least one
            review enter the tournaments. When False (the reviewerless
            ablation, experiment 07) all candidate papers enter, regardless of reviews.
        review_free: When True, build review-free meta-reviewer prompts (no mention
            of reviews). Typically set together with ``require_reviews=False``.
    """
    if require_reviews:
        papers_with_reviews = []
        for paper in candidate_papers:
            if reviews_by_paper.get(paper.paper_id):
                papers_with_reviews.append(paper)
    else:
        papers_with_reviews = list(candidate_papers)

    shuffled_papers = papers_with_reviews.copy()
    if shuffle_before_tournament:
        random.shuffle(shuffled_papers)

    tournaments = []
    for i in range(0, len(shuffled_papers), tournament_size):
        tournament_papers = shuffled_papers[i : i + tournament_size]
        tournaments.append(tournament_papers)

    tournament_tasks = []
    task_metadata = []

    for tournament_id, tournament_papers in enumerate(tournaments):
        meta_agent_id = f"meta_reviewer_t{iteration}_{tournament_id}"
        active_llm = llm_for(meta_agent_id) if llm_for else llm
        meta_agent = get_metareviewer_agent(active_llm, review_free=review_free)
        session_id = _build_metareview_session_id(
            meta_agent_id=meta_agent_id,
            iteration=iteration,
            tournament_id=tournament_id,
        )
        mlog = get_logs(meta_agent)
        if mlog is not None:
            backend_key = llm_backend_key_for(meta_agent_id) if llm_backend_key_for else None
            if not backend_key:
                backend_key = "default"
            mlog.compute_requirements = _build_compute_requirements(
                session_type="metareview",
                backend_key=backend_key,
                llm_runtime_metadata=_extract_llm_runtime_metadata(active_llm),
                default_llm_runtime_metadata=default_llm_runtime_metadata,
                session_timeout_seconds=total_session_timeout_seconds,
                sandbox_required=False,
                sandbox_requirements=sandbox_requirements,
            )
        tournament_reviews = {
            paper.paper_id: reviews_by_paper.get(paper.paper_id, []) for paper in tournament_papers
        }

        user_message = make_metareviewer_user_message(
            candidates=tournament_papers,
            reviews=tournament_reviews,
            review_free=review_free,
        )

        initial_state = {
            "messages": [HumanMessage(content=user_message)],
            "metareview_text": None,
            "parsed_reviews": None,
            "valid_metareview": None,
            "num_papers": len(tournament_papers),
            # Traceability: record the meta agent id executing this tournament
            "metadata": {
                "meta_agent_id": meta_agent_id,
                "tournament_id": tournament_id,
                "iteration": iteration,
                "session_id": session_id,
            },
        }

        task = run_one_tournament(
            meta_agent,
            initial_state,
            tournament_id,
            iteration,
            run_name=run_name,
            total_session_timeout_seconds=total_session_timeout_seconds,
        )

        tournament_tasks.append((task, meta_agent))
        task_metadata.append(
            (tournament_id, tournament_papers, meta_agent, meta_agent_id, session_id)
        )

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]meta-reviews running...[/]"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        transient=True,
        console=_console,
    ) as progress:
        task_id = progress.add_task("tournaments", total=len(tournament_tasks))
        results = await asyncio.gather(
            *[
                _wrap(meta_agent, task, progress, task_id)
                for (task, meta_agent) in tournament_tasks
            ],
            return_exceptions=True,
        )

    for result_idx, res in enumerate(results):
        if isinstance(res, Exception):
            tournament_id, tournament_papers, _, _, _ = task_metadata[result_idx]
            paper_ids = [p.paper_id for p in tournament_papers]
            log.error(
                f"[metareview] ERROR (exception) in tournament {tournament_id} for papers={paper_ids}: {res!r}"
            )

    metareview_logs: list[SessionStats] = []
    for tournament_id, tournament_papers, meta_agent, meta_agent_id, session_id in task_metadata:
        logs = get_logs(meta_agent)
        if logs is None:
            continue
        metareview_logs.append(
            SessionStats(
                iteration=iteration,
                session_id=session_id,
                session_agent_id=meta_agent_id,
                type="metareview",
                stats_logs=logs,
                metadata={
                    "tournament_id": int(tournament_id),
                    "paper_ids": [str(p.paper_id) for p in tournament_papers],
                },
            )
        )

    all_meta_reviews = []
    paper_scores = []

    for result_idx, result in enumerate(results):
        if isinstance(result, Exception):
            tournament_id, tournament_papers, _, _, _ = task_metadata[result_idx]
            paper_ids = [p.paper_id for p in tournament_papers]
            log.error(
                f"[metareview] ERROR in tournament {tournament_id} for papers={paper_ids}: {result!r}"
            )
            continue

        tournament_id, tournament_papers, _, _, _ = task_metadata[result_idx]

        if isinstance(result, dict):
            parsed_reviews = result.get("parsed_reviews", {}) or {}
        else:
            parsed_reviews = {}

        tournament_size_actual = len(tournament_papers)

        for paper in tournament_papers:
            tournament_size_actual = max(1, len(tournament_papers))

            paper_data = parsed_reviews.get(paper.paper_id, {}) or {}

            # score (robust parse)
            try:
                paper_score = float(paper_data.get("score", 0.0))
            except (TypeError, ValueError):
                paper_score = 0.0

            # rank (robust + clamped)
            raw_rank = paper_data.get("rank", None)
            try:
                paper_rank = int(raw_rank)
            except (TypeError, ValueError):
                paper_rank = 999  # fallback when LLM didn’t supply a sensible rank
            paper_rank = max(1, min(paper_rank, tournament_size_actual))
            rank_str = f"{paper_rank}/{tournament_size_actual}"

            meta_review = PaperMetaReview(
                paper_id=paper.paper_id,
                meta_review_text=paper_data.get("text", ""),
                overall_score=paper_score,
                rank=rank_str,
                justification=None,
                decision="reject",
            )

            all_meta_reviews.append(meta_review)
            paper_scores.append((paper.paper_id, paper_score, meta_review))

    if paper_scores:
        total_papers = len(paper_scores)
        num_to_accept = max(1, total_papers // tournament_size)
        # If higher score means better, sort descending (largest first)
        paper_scores.sort(key=lambda x: x[1], reverse=True)

        accepted_paper_ids = set()
        for i in range(min(num_to_accept, len(paper_scores))):
            paper_id, score, meta_review_obj = paper_scores[i]
            meta_review_obj.decision = "accept"
            accepted_paper_ids.add(paper_id)

    return all_meta_reviews, metareview_logs
