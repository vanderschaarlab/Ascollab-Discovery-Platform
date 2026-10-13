from __future__ import annotations

import hashlib
import json
import logging
import random
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from system.schema import PaperMetadataPrivate
from system.trackers import _CURRENT_LOG

log = logging.getLogger(__name__)

# Agents the matcher asks the registry search for per paper. The registry returns at most
# `memory.max_k` (default 5) of them, so that is the real size of a paper's expert pool.
_SEARCH_K = 12

if TYPE_CHECKING:
    from stores.agent_registry import AgentRegistry


def _assignment_rng(
    *,
    paper_id: str,
    seed: int | None = 0,
    iteration: int | None = None,
    seed_offset: int = 0,
    condition_label: str = "",
) -> random.Random:
    """Build the deterministic per-paper RNG for reviewer assignment.

    Every component that identifies an experimental condition is part of the key,
    so two conditions differing only in `condition_label` -- or only in
    `iteration` -- get different orderings at the same seed. Previous versions
    dropped `condition_label` whenever `iteration` was supplied, which made the
    label dead config for the main pipeline.

    Args:
        paper_id: Paper being assigned.
        seed: Run seed.
        iteration: Iteration index, or None when not iteration-scoped.
        seed_offset: Extra offset for replaying a shifted condition.
        condition_label: Experiment condition identifier.

    Returns:
        A seeded random.Random.
    """
    blob = "|".join(
        [
            str(int(seed or 0)),
            str(int(iteration) if iteration is not None else -1),
            str(int(seed_offset or 0)),
            str(condition_label or ""),
            str(paper_id),
        ]
    )
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
    return random.Random(int(digest, 16))


def _all_agent_ids(registry: AgentRegistry) -> list[str]:
    """Return all agent_ids from the registry."""
    return [a.agent_id for a in registry.get_all_agents()]


def _clip_unique(seq: Sequence[str]) -> list[str]:
    """Deduplicate a sequence while preserving order."""
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _safe_take(cands: list[str], k: int) -> list[str]:
    """Take up to k items from a candidate list without errors on short lists."""
    return cands[: max(0, k)]


def _run_search(tool, payload):
    """Invoke a tool using its run/invoke/call interface best-effort."""
    # Try .run, then .invoke, then call directly
    for attr in ("run", "invoke"):
        f = getattr(tool, attr, None)
        if callable(f):
            return f(payload)
    if callable(tool):
        return tool(payload)
    return []


def _parse_agent_ids(res: Any) -> list[str]:
    """Return the agent ids in a registry search result, in rank order.

    Args:
        res: The search tool's output: its observation string
            (``"Observation: AGENT_SEARCH_RESULTS = {json}"``, see
            ``env_setup/langchain_tools.registry_query_tool``), a dict holding a list under
            ``ids``/``result``/``results``/``data``, or a plain list. List items are agent ids
            or result dicts with an ``agent_id``.

    Returns:
        Agent ids in the order the search ranked them; empty if none can be read.
    """
    if isinstance(res, str):
        _, sep, payload = res.partition("=")
        try:
            res = json.loads(payload) if sep else []
        except json.JSONDecodeError:
            return []
    if isinstance(res, dict):
        res = next(
            (
                res[key]
                for key in ("ids", "result", "results", "data")
                if isinstance(res.get(key), list)
            ),
            [],
        )
    if not isinstance(res, list):
        return []
    ids = [item.get("agent_id") if isinstance(item, dict) else item for item in res]
    return [str(rid) for rid in ids if rid]


def _search_reviewer_ids(agent_search_tool: Any, query_text: str) -> list[str]:
    """Rank agents by expertise for a query with the registry search tool.

    The search is the matcher's, not an agent's, so the current agent log is hidden while it
    runs: the recording wrapper both callers pass (``env_setup/tools.make_tools``) then does not
    count it among the agents' tool queries.

    Args:
        agent_search_tool: The ``search_agent_registry`` tool (or a stand-in), or None.
        query_text: Text the agents' expertise is matched against.

    Returns:
        Agent ids in rank order.
    """
    token = _CURRENT_LOG.set(None)
    try:
        res = _run_search(agent_search_tool, {"query_text": query_text, "k": _SEARCH_K})
    finally:
        _CURRENT_LOG.reset(token)
    return _parse_agent_ids(res)


def match_papers_to_reviewers(
    *,
    agent_search_tool,
    num_reviewers_per_paper: int,
    papers: list[PaperMetadataPrivate],
    agent_registry: AgentRegistry,
    exclude_reviewer_ids: list[str] | None = None,
    allow_self_review: bool = False,
    seed: int | None = 0,
    iteration: int | None = None,
    condition_label: str | None = None,
) -> dict[str, list[str]]:
    """Assign reviewers to papers using a search tool and soft diversity.

    Strategy:
      1) Search the agent registry with the paper's title to rank candidate reviewers by
         expertise (at most ``memory.max_k`` of them). The search is not recorded as an agent
         tool query.
      2) Filter out the paper's primary author (unless self-review enabled) and duplicates.
      3) If empty or insufficient → fill from the rest of the registry in a seeded random
         order, behind the tool's own candidates at equal batch usage.
      4) Order by (batch usage, tool rank, jitter); select `num_reviewers_per_paper`.
      5) Soft diversity: avoid overusing the same reviewer across this batch when ties occur.

    Args:
        agent_search_tool: The ``search_agent_registry`` tool, called with
            ``{"query_text": <paper title>, "k": 12}``; None skips the search.
        num_reviewers_per_paper: Desired number of reviewers per paper.
        papers: List of papers needing reviewers.
        agent_registry: Registry of available agents.
        exclude_reviewer_ids: Reviewer ids that must never be assigned.
        allow_self_review: Whether a paper's author may review it.
        seed: Seed for the deterministic per-paper tiebreak RNG.
        iteration: Iteration index, mixed into the RNG key. Accepted so both
            assignment strategies build their key from the same components.
        condition_label: Label mixed into the RNG key so distinct experimental
            conditions (e.g. experiment 10's ``n_reviewers=k`` arms) get distinct,
            reproducible orderings at the same seed. None resolves to
            ``"expertise_matching"``, the dispatcher's default label.

    Returns:
        Mapping of paper_id -> list of assigned reviewer_ids.
    """
    exclude = set(exclude_reviewer_ids or [])
    all_ids = [aid for aid in _all_agent_ids(agent_registry) if aid not in exclude]
    all_ids_set = set(all_ids)
    usage_count = dict.fromkeys(all_ids, 0)  # soft diversity across this batch

    assignments: dict[str, list[str]] = {}
    tool_sourced = 0
    fallback_sourced = 0
    # None (e.g. a caller forwarding an unset config value) means the same thing as
    # the dispatcher default, so direct and dispatched calls key identically.
    resolved_label = condition_label if condition_label is not None else "expertise_matching"

    for paper in papers:
        paper_id = paper.paper_id
        author_id = paper.primary_agent_id
        exclude_author = not allow_self_review
        # condition_label (e.g. "n_reviewers=2" in experiment 10) is part of the RNG key together
        # with seed and iteration, so reviewer-count conditions are reproducible and
        # independently salted. A missing label resolves to the dispatcher default.
        rng = _assignment_rng(
            paper_id=paper_id,
            seed=seed,
            iteration=iteration,
            condition_label=resolved_label,
        )

        # 1) Rank agents by how well their expertise matches the paper's title. A failed
        #    search leaves the paper to the fallback below.
        tool_candidates: list[str] = []
        title = (paper.title or "").strip()
        if title:
            try:
                tool_candidates = _search_reviewer_ids(agent_search_tool, title)
            except Exception as exc:
                log.warning(
                    f"[reviewer_matching] registry search failed for {paper_id}: "
                    f"{type(exc).__name__}: {exc}"
                )

        # 2) Clean, de-dup, and remove author
        tool_candidates = _clip_unique(
            [
                rid
                for rid in tool_candidates
                if rid
                and rid in all_ids_set
                and (rid != author_id or not exclude_author)
                and rid not in exclude
            ]
        )

        # 3) Fallback if tool yields nothing or not enough. Tool candidates keep their
        #    rank; fallback entries follow them in a seeded random order.
        fallback_pool = [rid for rid in all_ids if (rid != author_id or not exclude_author)]
        tool_rank = {rid: i for i, rid in enumerate(tool_candidates)}
        if len(tool_candidates) < num_reviewers_per_paper:
            shuffled_fallback = [rid for rid in fallback_pool if rid not in tool_rank]
            rng.shuffle(shuffled_fallback)
            candidates = _clip_unique(tool_candidates + shuffled_fallback)
        else:
            candidates = list(tool_candidates)

        # 4) Order by batch usage, then by expertise rank (the rank is what sets this
        #    strategy apart from `random`); at equal usage, fallback candidates sort
        #    behind every tool candidate.
        jitter = {rid: rng.random() for rid in candidates}
        rank_floor = len(tool_candidates)
        candidates.sort(
            key=lambda rid: (
                usage_count.get(rid, 0),
                tool_rank.get(rid, rank_floor),
                jitter[rid],
            )
        )
        selected = _safe_take(candidates, num_reviewers_per_paper)
        tool_sourced += sum(1 for rid in selected if rid in tool_rank)
        fallback_sourced += sum(1 for rid in selected if rid not in tool_rank)

        # Edge case: registry too small
        if len(selected) < num_reviewers_per_paper and len(fallback_pool) > 0:
            # Add missing from fallback in diversity order
            extra_pool = [rid for rid in fallback_pool if rid not in selected]
            extra_jitter = {rid: rng.random() for rid in extra_pool}
            extra_pool.sort(key=lambda rid: (usage_count.get(rid, 0), extra_jitter[rid]))
            need = num_reviewers_per_paper - len(selected)
            extra = _safe_take(extra_pool, need)
            selected.extend(extra)
            fallback_sourced += len(extra)

        # Final guard: still could be < k if registry is tiny; that's OK—use what we have.
        assignments[paper_id] = selected

        # 5) Update batch usage counts
        for rid in selected:
            usage_count[rid] = usage_count.get(rid, 0) + 1

    # Distinctness check: surface a precise reason whenever a paper could not be
    # assigned `num_reviewers_per_paper` distinct reviewers (small eligible pool).
    eligible_pool_size = len(all_ids)
    for paper in papers:
        selected = assignments.get(paper.paper_id, [])
        distinct = len(set(selected))
        if distinct < num_reviewers_per_paper:
            author_excluded = (
                1 if (not allow_self_review and paper.primary_agent_id in all_ids_set) else 0
            )
            log.warning(
                f"[reviewer_matching] paper {paper.paper_id} got {distinct} distinct "
                f"reviewer(s) < requested {num_reviewers_per_paper}. "
                f"eligible_pool={eligible_pool_size} (registry minus {len(exclude)} excluded, "
                f"minus {author_excluded} author under allow_self_review={allow_self_review})."
            )

    # compact logging—no empty pre-filter spam
    try:
        first_pid = next(iter(assignments))
        assignments[first_pid]
        log.info("[reviewer_matching] Using registry: AgentRegistry")
        log.debug(f"[reviewer_matching] registry_ids (first 10): {all_ids[:10]} (n={len(all_ids)})")
        preview = [(p.paper_id, assignments[p.paper_id]) for p in papers[:3]]
        log.debug(f"[reviewer_matching] sample assignments (first 3): {preview}")
        # A run where tool_share is near zero is a run where the expertise arm
        # did nothing -- the check this counter exists to make possible.
        log.info(
            f"[reviewer_matching] sourcing: tool={tool_sourced} fallback={fallback_sourced} "
            f"(tool_share={tool_sourced / max(1, tool_sourced + fallback_sourced):.0%})"
        )
    except Exception:
        pass

    return assignments


def randomly_assign_papers_to_reviewers(
    *,
    num_reviewers_per_paper: int,
    papers: list[PaperMetadataPrivate],
    agent_registry: AgentRegistry,
    exclude_reviewer_ids: list[str] | None = None,
    allow_self_review: bool = False,
    seed: int | None = None,
    iteration: int | None = None,
    seed_offset: int = 0,
    condition_label: str = "random",
) -> dict[str, list[str]]:
    """Assign reviewers without querying agent-registry search.

    The eligibility rules intentionally mirror :func:`match_papers_to_reviewers`:
    excluded reviewer ids are removed, self-review is disallowed by default,
    and each paper receives unique reviewers when the eligible pool is large
    enough.

    Determinism uses a single key built from every condition-identifying
    component: (seed, iteration, seed_offset, condition_label, paper_id). Earlier
    versions branched between two key formats and silently dropped
    ``condition_label`` whenever ``iteration`` was supplied.
    """
    exclude = set(exclude_reviewer_ids or [])
    all_ids = [aid for aid in _all_agent_ids(agent_registry) if aid not in exclude]
    usage_count = dict.fromkeys(all_ids, 0)
    assignments: dict[str, list[str]] = {}

    for paper in papers:
        paper_id = str(getattr(paper, "paper_id", "") or "")
        author_id = str(getattr(paper, "primary_agent_id", "") or "")
        exclude_author = not allow_self_review
        rng = _assignment_rng(
            paper_id=paper_id,
            seed=seed,
            iteration=iteration,
            seed_offset=int(seed_offset or 0),
            condition_label=condition_label,
        )

        candidates = [
            rid
            for rid in all_ids
            if rid and (rid != author_id or not exclude_author) and rid not in exclude
        ]
        candidates = _clip_unique(candidates)
        rng.shuffle(candidates)
        order_index = {rid: i for i, rid in enumerate(candidates)}
        candidates.sort(key=lambda rid: (usage_count.get(rid, 0), order_index[rid]))
        selected = _safe_take(candidates, num_reviewers_per_paper)

        assignments[paper_id] = selected
        for rid in selected:
            usage_count[rid] = usage_count.get(rid, 0) + 1

    try:
        preview = [(p.paper_id, assignments.get(p.paper_id, [])) for p in papers[:3]]
        log.info("[reviewer_matching] Using strategy: random")
        log.debug(f"[reviewer_matching] registry_ids (first 10): {all_ids[:10]} (n={len(all_ids)})")
        log.debug(f"[reviewer_matching] sample assignments (first 3): {preview}")
    except Exception:
        pass

    return assignments


def match_papers_to_reviewers_random(
    *,
    num_reviewers_per_paper: int,
    papers: list[PaperMetadataPrivate],
    agent_registry: AgentRegistry,
    exclude_reviewer_ids: list[str] | None = None,
    allow_self_review: bool = False,
    seed: int | None = None,
    iteration: int | None = None,
    seed_offset: int = 0,
) -> dict[str, list[str]]:
    """Backward-compatible alias used by fixed_workflow tests and config."""
    return randomly_assign_papers_to_reviewers(
        num_reviewers_per_paper=num_reviewers_per_paper,
        papers=papers,
        agent_registry=agent_registry,
        exclude_reviewer_ids=exclude_reviewer_ids,
        allow_self_review=allow_self_review,
        seed=seed,
        iteration=iteration,
        seed_offset=seed_offset,
        condition_label=f"legacy:{iteration}:{seed_offset}",
    )


def assign_papers_to_reviewers(
    *,
    strategy: str,
    agent_search_tool,
    num_reviewers_per_paper: int,
    papers: list[PaperMetadataPrivate],
    agent_registry: AgentRegistry,
    exclude_reviewer_ids: list[str] | None = None,
    allow_self_review: bool = False,
    seed: int = 0,
    condition_label: str | None = None,
    iteration: int | None = None,
    seed_offset: int = 0,
) -> dict[str, list[str]]:
    """Dispatch reviewer assignment to expertise matching or random assignment."""
    normalized = str(strategy or "expertise_matching").strip().lower()
    if normalized in {"expertise_matching", "matching", "expertise", "registry"}:
        return match_papers_to_reviewers(
            agent_search_tool=agent_search_tool,
            num_reviewers_per_paper=num_reviewers_per_paper,
            papers=papers,
            agent_registry=agent_registry,
            exclude_reviewer_ids=exclude_reviewer_ids,
            allow_self_review=allow_self_review,
            seed=seed,
            iteration=iteration,
            condition_label=condition_label or normalized,
        )
    if normalized in {"random", "random_assignment", "no_matching"}:
        return randomly_assign_papers_to_reviewers(
            num_reviewers_per_paper=num_reviewers_per_paper,
            papers=papers,
            agent_registry=agent_registry,
            exclude_reviewer_ids=exclude_reviewer_ids,
            allow_self_review=allow_self_review,
            seed=seed,
            condition_label=condition_label or normalized,
            iteration=iteration,
            seed_offset=seed_offset,
        )
    raise ValueError(
        f"Unknown reviewer assignment strategy: {strategy!r}. "
        "Expected expertise_matching/registry or random."
    )
