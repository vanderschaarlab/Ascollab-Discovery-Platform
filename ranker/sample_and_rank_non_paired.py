"""Sample-and-rank (non-paired): batch pairwise comparison of papers from a single pool.

Collects papers from one or more snapshot directories into a unified pool,
generates random uniform pairs from this pool for pairwise comparison,
ranks them in parallel, and saves structured results.

Usage::

    from ranker.sample_and_rank_non_paired import run_sample_and_rank_non_paired

    result = run_sample_and_rank_non_paired(
        paper_pool_cfg={"label": "pool", "snapshot_dirs": [...]},
        sampling_cfg={"mode": "all", "n_repeats": 1, "seed": 42},
        ranker_cfg={...},
        parallelism_cfg={"max_workers": 4},
        llm=llm,
        paper_selection_cfg={"iterations": "last", "cumulative": True, ...},
    )
"""

import logging
import os
import random
import time
from datetime import UTC, datetime
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from .sample_and_rank_common import (
    _build_summary_base,
    _collect_experiment_configs,
    _collect_paper_data_group,
    _count_by_iteration,
    _count_paper_statuses,
    _print_final_summary,
    _run_comparisons_parallel,
    _save_results,
    _top_n,
    _top_n_per_iteration,
    collect_papers_for_pool,
)
from .snapshot_import import (
    build_citation_chains,
    collect_citable_papers,
    collect_dataset_descriptions,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_sample_and_rank_non_paired(
    paper_pool_cfg: dict[str, Any],
    sampling_cfg: dict[str, Any],
    ranker_cfg: dict[str, Any],
    parallelism_cfg: dict[str, Any],
    llm: BaseChatModel,
    sandbox_client: Any = None,
    run_id: str | None = None,
    output_dir: str | None = None,
    paper_selection_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute the full non-paired sample-and-rank workflow.

    Args:
        paper_pool_cfg: Dict with "label" and "snapshot_dirs" for the
            single paper pool.
        sampling_cfg: Sampling config with "mode", "n_repeats", "n_pairs",
            "with_replacement", "allow_self_comparison", "seed".
        ranker_cfg: Ranker config (llm_backend, max_iterations, criteria_prompt,
            tools, llm_retry, output_parsing).
        parallelism_cfg: Dict with "max_workers".
        llm: Pre-built LLM instance for the judge.
        sandbox_client: Optional SandboxClient for sandbox tool.
        run_id: Identifier for this batch run.
        output_dir: Directory to write output files.
        paper_selection_cfg: Dict with "iterations", "cumulative",
            "include_rejected", "exclude_timeout", and "top_n" sub-dicts
            controlling paper gathering and pre-filtering.
            No "balance" (single pool, not applicable).

    Returns:
        Dict with keys: "comparisons" (list of records), "summary" (dict),
        "output_dir" (str path where results were saved).
    """
    run_start = time.time()
    run_datetime = datetime.now(UTC).isoformat()

    if paper_selection_cfg is None:
        paper_selection_cfg = {}

    iterations = paper_selection_cfg.get("iterations", "last")
    cumulative = paper_selection_cfg.get("cumulative", True)

    pool_label = paper_pool_cfg.get("label", "paper_pool")
    snapshot_dirs = list(paper_pool_cfg["snapshot_dirs"])

    if run_id is None:
        run_id = f"sample_and_rank_non_paired_{datetime.now().strftime('%y%m%d_%H%M')}"

    if output_dir is None:
        output_dir = os.path.join("outputs", "_sample_and_rank_non_paired", run_id)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Gather papers from the single pool
    papers, collection_info = _gather_papers_non_paired(
        snapshot_dirs,
        pool_label,
        paper_selection_cfg,
    )
    _print_collection_summary_non_paired(collection_info, iterations, cumulative)

    # 1a. Apply paper selection (top-N only, no balance)
    if paper_selection_cfg:
        papers, selection_info = _apply_paper_selection_non_paired(
            papers,
            paper_selection_cfg,
        )
        selection_info["filtering"] = collection_info.get("filtering", {})
        collection_info["paper_selection"] = selection_info
        _print_selection_summary_non_paired(selection_info)

    # 1b. Collect comprehensive paper data
    paper_data = _collect_paper_data_non_paired(papers, pool_label)

    # 2. Collect experiment configs
    exp_configs = _collect_experiment_configs(snapshot_dirs)

    # 2b. Collect dataset descriptions for all papers
    dataset_desc_lookup = collect_dataset_descriptions(papers)

    # 2c. Build citation chains: each paper's ancestors come from all accepted papers of its own
    # run, not only from the papers selected for ranking.
    cited_papers_lookup = build_citation_chains(
        papers, citable=collect_citable_papers(snapshot_dirs, iterations)
    )

    # 3. Generate pairs
    tasks = _generate_pairs_non_paired(papers, sampling_cfg)
    _print_sampling_summary_non_paired(tasks, sampling_cfg, len(papers))

    # 4-5. Rank pairs in parallel, fill run_id/datetime, sort
    comparisons, traces = _run_comparisons_parallel(
        tasks,
        llm,
        ranker_cfg,
        sandbox_client,
        iterations,
        cumulative,
        parallelism_cfg,
        run_id,
        run_datetime,
        dataset_desc_lookup=dataset_desc_lookup,
        cited_papers_lookup=cited_papers_lookup,
    )

    # 6. Build summary
    run_end = time.time()
    summary = _build_summary_non_paired(
        run_id=run_id,
        run_datetime=run_datetime,
        duration_seconds=round(run_end - run_start, 1),
        collection_info=collection_info,
        iterations=iterations,
        cumulative=cumulative,
        sampling_cfg=sampling_cfg,
        papers=papers,
        comparisons=comparisons,
        parallelism_cfg=parallelism_cfg,
        paper_data=paper_data,
        pool_label=pool_label,
    )

    # 7. Save everything
    _save_results(comparisons, summary, ranker_cfg, exp_configs, traces, output_dir, paper_data)

    # 8. Print final summary
    _print_final_summary(summary)

    return {"comparisons": comparisons, "summary": summary, "output_dir": output_dir}


# ---------------------------------------------------------------------------
# Internal functions (non-paired-specific)
# ---------------------------------------------------------------------------


def _gather_papers_non_paired(
    snapshot_dirs: list[str],
    pool_label: str,
    paper_selection_cfg: dict[str, Any] | None = None,
) -> tuple[list[dict], dict]:
    """Collect papers for the single pool.

    Returns:
        (papers, collection_info)
    """
    if paper_selection_cfg is None:
        paper_selection_cfg = {}

    papers = collect_papers_for_pool(snapshot_dirs, paper_selection_cfg)

    include_rejected = paper_selection_cfg.get("include_rejected", False)
    exclude_timeout = paper_selection_cfg.get("exclude_timeout", True)
    paper_kind_filter = paper_selection_cfg.get("paper_kind_filter", "all")

    collection_info = {
        "paper_pool": {
            "label": pool_label,
            "snapshot_dirs": list(snapshot_dirs),
            "papers": papers,
        },
        "filtering": {
            "include_rejected": include_rejected,
            "exclude_timeout": exclude_timeout,
            "paper_kind_filter": paper_kind_filter,
            "counts": _count_paper_statuses(papers),
        },
    }

    return papers, collection_info


def _apply_paper_selection_non_paired(
    papers: list[dict],
    paper_selection_cfg: dict[str, Any],
) -> tuple[list[dict], dict[str, Any]]:
    """Apply top-N filtering to the paper pool (no balance for non-paired)."""
    info: dict[str, Any] = {
        "original_count": len(papers),
    }

    top_n_cfg = paper_selection_cfg.get("top_n", {})

    if top_n_cfg.get("enabled", False):
        n = top_n_cfg.get("n", 10)
        per_iteration = top_n_cfg.get("per_iteration", False)

        if per_iteration:
            papers = _top_n_per_iteration(papers, n, "pool")
        else:
            papers = _top_n(papers, n, "pool")

        top_n_info: dict[str, Any] = {
            "enabled": True,
            "n": n,
            "per_iteration": per_iteration,
            "count_after": len(papers),
        }
        if per_iteration:
            top_n_info["per_iteration_counts"] = _count_by_iteration(papers)
        info["top_n"] = top_n_info
    else:
        info["top_n"] = {"enabled": False}

    info["final_count"] = len(papers)
    return papers, info


def _collect_paper_data_non_paired(
    papers: list[dict],
    pool_label: str,
) -> list[dict[str, Any]]:
    """Collect comprehensive paper data for the single pool."""
    return _collect_paper_data_group(papers, pool_label, None)


def _generate_pairs_non_paired(
    papers: list[dict],
    sampling_cfg: dict[str, Any],
) -> list[dict]:
    """Generate comparison tasks from a single pool with randomized A/B assignment.

    For ``mode="all"``:
        Generate all C(n,2) unique unordered pairs (or C(n,2)+n if
        allow_self_comparison), each repeated ``n_repeats`` times.

    For ``mode="random"``:
        Sample ``n_pairs`` from the valid pair pool (with or without replacement).
    """
    mode = sampling_cfg.get("mode", "all")
    seed = sampling_cfg.get("seed", 42)
    allow_self = sampling_cfg.get("allow_self_comparison", False)
    n = len(papers)

    if mode == "all":
        n_repeats = sampling_cfg.get("n_repeats", 1)
        raw_pairs = []
        # Generate unordered pairs: i < j (or i <= j if self-comparison)
        for i in range(n):
            start_j = i if allow_self else i + 1
            for j in range(start_j, n):
                for r in range(n_repeats):
                    raw_pairs.append((i, j, r))

    elif mode == "random":
        n_pairs = sampling_cfg.get("n_pairs", 100)
        with_replacement = sampling_cfg.get("with_replacement", False)

        # Build pool of valid pairs
        if allow_self:
            all_valid = [(i, j) for i in range(n) for j in range(i, n)]
        else:
            all_valid = [(i, j) for i in range(n) for j in range(i + 1, n)]

        pool_size = len(all_valid)

        if not with_replacement and n_pairs > pool_size:
            raise ValueError(
                f"Cannot sample {n_pairs} pairs without replacement from "
                f"a pool of {pool_size} valid pairs "
                f"({n} papers, allow_self_comparison={allow_self}). "
                f"Maximum possible: {pool_size}."
            )

        rng = random.Random(seed)
        if with_replacement:
            sampled = rng.choices(all_valid, k=n_pairs)
        else:
            sampled = rng.sample(all_valid, k=n_pairs)

        raw_pairs = [(i, j, 0) for i, j in sampled]
    else:
        raise ValueError(f"Unknown sampling mode: {mode!r}. Expected 'all' or 'random'.")

    # Build comparison tasks with A/B randomization
    tasks = []
    for idx, (i, j, repeat_idx) in enumerate(raw_pairs):
        comparison_id = f"cmp_{idx + 1:04d}"

        paper_i = papers[i]
        paper_j = papers[j]

        # Randomize A/B position using a seeded RNG
        position_rng = random.Random(seed + idx)
        swap = position_rng.random() < 0.5

        if swap:
            paper_a_slot = paper_j
            paper_b_slot = paper_i
        else:
            paper_a_slot = paper_i
            paper_b_slot = paper_j

        tasks.append(
            {
                "comparison_id": comparison_id,
                "paper_a": paper_a_slot,
                "paper_b": paper_b_slot,
                "paper_a_experiment": None,
                "paper_b_experiment": None,
                "repeat_index": repeat_idx,
            }
        )

    return tasks


def _build_summary_non_paired(
    run_id: str,
    run_datetime: str,
    duration_seconds: float,
    collection_info: dict,
    iterations: str | list[int],
    cumulative: bool,
    sampling_cfg: dict[str, Any],
    papers: list[dict],
    comparisons: list[dict],
    parallelism_cfg: dict[str, Any],
    paper_data: list[dict[str, Any]] | None = None,
    pool_label: str = "paper_pool",
) -> dict[str, Any]:
    """Build the run_summary.json content for the non-paired pipeline."""
    n = len(papers)
    allow_self = sampling_cfg.get("allow_self_comparison", False)
    total_unique_pairs = n * (n + 1) // 2 if allow_self else n * (n - 1) // 2

    # --- Non-paired-specific: pool breakdown ---
    pool_info = collection_info.get("paper_pool", {})
    pool_papers = pool_info.get("papers", papers)
    papers_by_snapshot: dict[str, int] = {}
    for p in pool_papers:
        sd = p.get("source_snapshot_dir", "")
        papers_by_snapshot[sd] = papers_by_snapshot.get(sd, 0) + 1
    experiments_summary = {
        "paper_pool": {
            "label": pool_label,
            "snapshot_dirs": pool_info.get("snapshot_dirs", []),
            "num_papers": len(pool_papers),
            "papers_by_snapshot": papers_by_snapshot,
        },
    }

    # --- Non-paired-specific: paper_data breakdown ---
    if paper_data is None:
        paper_data = []
    with_system_metrics = sum(1 for m in paper_data if m.get("metrics") is not None)
    paper_data_summary = {
        "total": len(paper_data),
        "with_system_metrics": with_system_metrics,
        "without_system_metrics": len(paper_data) - with_system_metrics,
    }

    return _build_summary_base(
        run_id=run_id,
        run_datetime=run_datetime,
        duration_seconds=duration_seconds,
        iterations=iterations,
        cumulative=cumulative,
        sampling_cfg=sampling_cfg,
        total_unique_pairs=total_unique_pairs,
        comparisons=comparisons,
        parallelism_cfg=parallelism_cfg,
        paper_data=paper_data,
        experiments_summary=experiments_summary,
        paper_data_summary=paper_data_summary,
        extra_sampling={"allow_self_comparison": allow_self},
        collection_info=collection_info,
    )


# ---------------------------------------------------------------------------
# Printing helpers (non-paired-specific)
# ---------------------------------------------------------------------------


def _print_collection_summary_non_paired(
    collection_info: dict,
    iterations: str | list[int],
    cumulative: bool = True,
) -> None:
    """Print the paper collection summary for the non-paired pool."""
    cum_label = "cumulative" if cumulative else "per-iteration"
    lines = [
        "",
        "=" * 60,
        "PAPER COLLECTION SUMMARY (NON-PAIRED)",
        "=" * 60,
        f"Iteration selection: {iterations} ({cum_label})",
        "",
    ]

    pool_info = collection_info.get("paper_pool", {})
    label = pool_info.get("label", "paper_pool")
    papers = pool_info.get("papers", [])

    by_dir: dict[str, list[dict]] = {}
    for p in papers:
        sd = p.get("source_snapshot_dir", "?")
        by_dir.setdefault(sd, []).append(p)

    lines.append(f'Paper Pool ("{label}"):')
    for snap_dir, snap_papers in by_dir.items():
        run_name = snap_papers[0].get("source_snapshot", "?")
        iters = sorted({p["source_iteration"] for p in snap_papers})
        iter_str = ", ".join(str(i) for i in iters)
        lines.append(
            f"  {snap_dir}  ->  run_name: {run_name}  ->  "
            f"{len(snap_papers)} papers (iteration {iter_str})"
        )
    lines.append(f"  Total: {len(papers)} papers")
    lines.append("")

    n = len(papers)
    unique_pairs = n * (n - 1) // 2
    lines.append(f"Possible unique pairs (no self-comparison): C({n}, 2) = {unique_pairs}")
    lines.append("=" * 60)

    summary_text = "\n".join(lines)
    print(summary_text)
    log.info(summary_text)


def _print_selection_summary_non_paired(selection_info: dict[str, Any]) -> None:
    """Print paper selection effects for the non-paired pipeline."""
    lines = [
        "",
        "-" * 60,
        "PAPER SELECTION (NON-PAIRED)",
        "-" * 60,
    ]

    filtering = selection_info.get("filtering", {})
    include_rejected = filtering.get("include_rejected", False)
    exclude_timeout = filtering.get("exclude_timeout", True)
    paper_kind_filter = filtering.get("paper_kind_filter", "all")
    counts = filtering.get("counts", {})
    lines.append(f"Paper kind filter: {paper_kind_filter}")

    if include_rejected:
        n_acc = counts.get("accepted", 0)
        n_rej = counts.get("rejected", 0)
        lines.append("Status filter: accepted + rejected")
        lines.append(f"  Pool: {n_acc} accepted, {n_rej} rejected")
    else:
        lines.append("Status filter: accepted only")

    if exclude_timeout:
        lines.append("Timeout filter: excluded")
    else:
        n_to = counts.get("timeout", 0)
        if n_to > 0:
            lines.append(f"Timeout filter: included ({n_to} in pool)")
        else:
            lines.append("Timeout filter: included (0 found)")

    top_n = selection_info.get("top_n", {})
    if top_n.get("enabled"):
        n = top_n["n"]
        per_iter = top_n.get("per_iteration", False)
        n_label = f"n={n} per iteration" if per_iter else f"n={n}"
        lines.append(f"Top-N: enabled ({n_label})")
        lines.append(f"  Pool: {selection_info['original_count']} -> {top_n['count_after']} papers")
        if per_iter and "per_iteration_counts" in top_n:
            breakdown = ", ".join(
                f"iter {it}: {cnt}" for it, cnt in top_n["per_iteration_counts"].items()
            )
            lines.append(f"    ({breakdown})")
    else:
        lines.append("Top-N: disabled")

    lines.append(f"Final: {selection_info['final_count']} papers")
    lines.append("-" * 60)

    text = "\n".join(lines)
    print(text)
    log.info(text)


def _print_sampling_summary_non_paired(
    tasks: list[dict],
    sampling_cfg: dict[str, Any],
    n_papers: int,
) -> None:
    """Print the sampling summary for the non-paired pipeline."""
    mode = sampling_cfg.get("mode", "all")
    seed = sampling_cfg.get("seed", 42)
    allow_self = sampling_cfg.get("allow_self_comparison", False)

    if allow_self:
        unique_pairs = n_papers * (n_papers + 1) // 2
    else:
        unique_pairs = n_papers * (n_papers - 1) // 2

    lines = [
        "",
        "=" * 60,
        "SAMPLING SUMMARY (NON-PAIRED)",
        "=" * 60,
    ]

    if mode == "all":
        n_repeats = sampling_cfg.get("n_repeats", 1)
        lines.append(f"Mode: all (n_repeats={n_repeats})")
    else:
        n_pairs = sampling_cfg.get("n_pairs", 100)
        wr = sampling_cfg.get("with_replacement", False)
        lines.append(f"Mode: random (n_pairs={n_pairs}, with_replacement={wr})")

    lines.append(f"Allow self-comparison: {allow_self}")
    lines.append(f"Possible unique pairs: {unique_pairs}")
    lines.append(f"Total comparisons: {len(tasks)}")
    lines.append(f"A/B position: randomized (seed={seed})")
    lines.append("=" * 60)

    summary_text = "\n".join(lines)
    print(summary_text)
    log.info(summary_text)
