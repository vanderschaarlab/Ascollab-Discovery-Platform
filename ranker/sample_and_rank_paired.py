"""Sample-and-rank (paired): batch pairwise comparison of papers from two experiments.

Collects papers from two experiment conditions (A vs B), generates
cross-experiment pairs, ranks them in parallel via the ranker, and saves
structured results.

Usage::

    from ranker.sample_and_rank_paired import run_sample_and_rank_paired

    result = run_sample_and_rank_paired(
        experiments_cfg={"experiment_a": {...}, "experiment_b": {...}},
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
    _metareview_sort_key,
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


def run_sample_and_rank_paired(
    experiments_cfg: dict[str, Any],
    sampling_cfg: dict[str, Any],
    ranker_cfg: dict[str, Any],
    parallelism_cfg: dict[str, Any],
    llm: BaseChatModel,
    sandbox_client: Any = None,
    run_id: str | None = None,
    output_dir: str | None = None,
    paper_selection_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute the full paired sample-and-rank workflow.

    Args:
        experiments_cfg: Dict with "experiment_a" and "experiment_b" keys,
            each containing "label" and "snapshot_dirs".
        sampling_cfg: Sampling config with "mode", "n_repeats", "n_pairs",
            "with_replacement", "cap_n_pairs", "seed".
        ranker_cfg: Ranker config (llm_backend, max_iterations, criteria_prompt,
            tools, llm_retry, output_parsing).
        parallelism_cfg: Dict with "max_workers".
        llm: Pre-built LLM instance for the judge.
        sandbox_client: Optional SandboxClient for sandbox tool.
        run_id: Identifier for this batch run.
        output_dir: Directory to write output files.
        paper_selection_cfg: Dict with "iterations", "cumulative",
            "include_rejected", "exclude_timeout", "top_n", and "balance"
            sub-dicts controlling paper gathering and pre-filtering.

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

    if run_id is None:
        run_id = f"sample_and_rank_paired_{datetime.now().strftime('%y%m%d_%H%M')}"

    if output_dir is None:
        output_dir = os.path.join("outputs", "_sample_and_rank_paired", run_id)
    os.makedirs(output_dir, exist_ok=True)

    exp_a_cfg = experiments_cfg["experiment_a"]
    exp_b_cfg = experiments_cfg["experiment_b"]
    label_a = exp_a_cfg.get("label", "experiment_a")
    label_b = exp_b_cfg.get("label", "experiment_b")

    # 1. Gather papers
    papers_a, papers_b, collection_info = _gather_papers(
        experiments_cfg,
        paper_selection_cfg,
    )
    _print_collection_summary(collection_info, iterations, cumulative)

    # 1a. Apply paper selection (top-N, balance) if configured
    if paper_selection_cfg:
        papers_a, papers_b, selection_info = _apply_paper_selection(
            papers_a,
            papers_b,
            paper_selection_cfg,
        )
        selection_info["filtering"] = collection_info.get("filtering", {})
        collection_info["paper_selection"] = selection_info
        _print_selection_summary(selection_info)

    # 1b. Collect comprehensive paper data for the gathered papers
    paper_data = _collect_paper_data(papers_a, papers_b, label_a, label_b)

    # 2. Collect experiment configs (for output)
    all_snapshot_dirs = list(exp_a_cfg["snapshot_dirs"]) + list(exp_b_cfg["snapshot_dirs"])
    exp_configs = _collect_experiment_configs(all_snapshot_dirs)

    # 2b. Collect dataset descriptions for all papers
    all_papers = papers_a + papers_b
    dataset_desc_lookup = collect_dataset_descriptions(all_papers)

    # 2c. Build citation chains: each paper's ancestors come from all accepted papers of its own
    # run, not only from the papers selected for ranking.
    cited_papers_lookup = build_citation_chains(
        all_papers, citable=collect_citable_papers(all_snapshot_dirs, iterations)
    )

    # 3. Generate pairs
    tasks = _generate_pairs(papers_a, papers_b, sampling_cfg, label_a, label_b)
    _print_sampling_summary(tasks, sampling_cfg, len(papers_a), len(papers_b))

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
    summary = _build_summary(
        run_id=run_id,
        run_datetime=run_datetime,
        duration_seconds=round(run_end - run_start, 1),
        collection_info=collection_info,
        iterations=iterations,
        cumulative=cumulative,
        sampling_cfg=sampling_cfg,
        papers_a=papers_a,
        papers_b=papers_b,
        comparisons=comparisons,
        parallelism_cfg=parallelism_cfg,
        paper_data=paper_data,
        label_a=label_a,
        label_b=label_b,
    )

    # 7. Save everything
    _save_results(comparisons, summary, ranker_cfg, exp_configs, traces, output_dir, paper_data)

    # 8. Print final summary
    _print_final_summary(summary)

    return {"comparisons": comparisons, "summary": summary, "output_dir": output_dir}


# ---------------------------------------------------------------------------
# Internal functions (paired-specific)
# ---------------------------------------------------------------------------


def _gather_papers(
    experiments_cfg: dict[str, Any],
    paper_selection_cfg: dict[str, Any] | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """Collect papers for both experiments.

    Returns:
        (papers_a, papers_b, collection_info)
    """
    exp_a = experiments_cfg["experiment_a"]
    exp_b = experiments_cfg["experiment_b"]

    if paper_selection_cfg is None:
        paper_selection_cfg = {}

    paper_kind_filter_default = paper_selection_cfg.get("paper_kind_filter", "all")
    paper_kind_filter_a = paper_selection_cfg.get("paper_kind_filter_a", paper_kind_filter_default)
    paper_kind_filter_b = paper_selection_cfg.get("paper_kind_filter_b", paper_kind_filter_default)

    papers_a = collect_papers_for_pool(
        list(exp_a["snapshot_dirs"]),
        paper_selection_cfg,
        paper_kind_filter_override=paper_kind_filter_a,
    )
    papers_b = collect_papers_for_pool(
        list(exp_b["snapshot_dirs"]),
        paper_selection_cfg,
        paper_kind_filter_override=paper_kind_filter_b,
    )

    include_rejected = paper_selection_cfg.get("include_rejected", False)
    exclude_timeout = paper_selection_cfg.get("exclude_timeout", True)

    collection_info = {
        "experiment_a": {
            "label": exp_a.get("label", "experiment_a"),
            "snapshot_dirs": list(exp_a["snapshot_dirs"]),
            "papers": papers_a,
        },
        "experiment_b": {
            "label": exp_b.get("label", "experiment_b"),
            "snapshot_dirs": list(exp_b["snapshot_dirs"]),
            "papers": papers_b,
        },
        "filtering": {
            "include_rejected": include_rejected,
            "exclude_timeout": exclude_timeout,
            "paper_kind_filter_a": paper_kind_filter_a,
            "paper_kind_filter_b": paper_kind_filter_b,
            "counts_a": _count_paper_statuses(papers_a),
            "counts_b": _count_paper_statuses(papers_b),
        },
    }

    return papers_a, papers_b, collection_info


def _apply_paper_selection(
    papers_a: list[dict],
    papers_b: list[dict],
    paper_selection_cfg: dict[str, Any],
) -> tuple[list[dict], list[dict], dict[str, Any]]:
    """Apply top-N filtering and/or balance equalization to the paper sets.

    **Order:** top-N is applied first (per-experiment quality gate), then
    balance (cross-experiment equalization).
    """
    info: dict[str, Any] = {
        "original_count_a": len(papers_a),
        "original_count_b": len(papers_b),
    }

    top_n_cfg = paper_selection_cfg.get("top_n", {})
    balance_cfg = paper_selection_cfg.get("balance", {})

    # --- Top-N ---
    if top_n_cfg.get("enabled", False):
        n = top_n_cfg.get("n", 10)
        per_iteration = top_n_cfg.get("per_iteration", False)

        if per_iteration:
            papers_a = _top_n_per_iteration(papers_a, n, "A")
            papers_b = _top_n_per_iteration(papers_b, n, "B")
        else:
            papers_a = _top_n(papers_a, n, "A")
            papers_b = _top_n(papers_b, n, "B")

        top_n_info: dict[str, Any] = {
            "enabled": True,
            "n": n,
            "per_iteration": per_iteration,
            "count_a_after": len(papers_a),
            "count_b_after": len(papers_b),
        }
        if per_iteration:
            top_n_info["per_iteration_counts_a"] = _count_by_iteration(papers_a)
            top_n_info["per_iteration_counts_b"] = _count_by_iteration(papers_b)
        info["top_n"] = top_n_info
    else:
        info["top_n"] = {"enabled": False}

    # --- Balance ---
    if balance_cfg.get("enabled", False):
        method = balance_cfg.get("method", "random")
        seed = balance_cfg.get("seed", 42)
        before_a, before_b = len(papers_a), len(papers_b)

        if len(papers_a) != len(papers_b):
            target = min(len(papers_a), len(papers_b))
            if len(papers_a) > target:
                papers_a = _cut_to_size(papers_a, target, method, seed, "A")
            else:
                papers_b = _cut_to_size(papers_b, target, method, seed, "B")

        info["balance"] = {
            "enabled": True,
            "method": method,
            "seed": seed,
            "count_a_before": before_a,
            "count_b_before": before_b,
            "count_a_after": len(papers_a),
            "count_b_after": len(papers_b),
        }
    else:
        info["balance"] = {"enabled": False}

    info["final_count_a"] = len(papers_a)
    info["final_count_b"] = len(papers_b)
    return papers_a, papers_b, info


def _cut_to_size(
    papers: list[dict],
    target: int,
    method: str,
    seed: int,
    label: str,
) -> list[dict]:
    """Reduce *papers* to *target* size using *method*."""
    if method == "random":
        rng = random.Random(seed)
        return rng.sample(papers, target)
    if method == "top_metareview":
        null_count = sum(1 for p in papers if _metareview_sort_key(p) == float("-inf"))
        if null_count:
            log.warning(
                "Experiment %s: %d/%d papers lack a metareview score; "
                "they will be ranked lowest in balance selection.",
                label,
                null_count,
                len(papers),
            )
        ranked = sorted(papers, key=_metareview_sort_key, reverse=True)
        return ranked[:target]
    raise ValueError(f"Unknown balance method: {method!r}. Expected 'random' or 'top_metareview'.")


def _collect_paper_data(
    papers_a: list[dict],
    papers_b: list[dict],
    label_a: str,
    label_b: str,
) -> list[dict[str, Any]]:
    """Collect comprehensive paper data for all papers from both experiments."""
    return _collect_paper_data_group(papers_a, label_a, "A") + _collect_paper_data_group(
        papers_b, label_b, "B"
    )


def _generate_pairs(
    papers_a: list[dict],
    papers_b: list[dict],
    sampling_cfg: dict[str, Any],
    experiment_a_label: str,
    experiment_b_label: str,
) -> list[dict]:
    """Generate the list of comparison tasks with randomized A/B assignment."""
    mode = sampling_cfg.get("mode", "all")
    seed = sampling_cfg.get("seed", 42)

    product_size = len(papers_a) * len(papers_b)

    if mode == "all":
        n_repeats = sampling_cfg.get("n_repeats", 1)
        raw_pairs = []
        for i in range(len(papers_a)):
            for j in range(len(papers_b)):
                for r in range(n_repeats):
                    raw_pairs.append((i, j, r))

    elif mode == "random":
        n_pairs = sampling_cfg.get("n_pairs", 100)
        with_replacement = sampling_cfg.get("with_replacement", False)

        if not with_replacement and n_pairs > product_size:
            if not sampling_cfg.get("cap_n_pairs", False):
                raise ValueError(
                    f"Cannot sample {n_pairs} pairs without replacement from "
                    f"a Cartesian product of size {product_size} "
                    f"({len(papers_a)} x {len(papers_b)}). "
                    f"Maximum possible: {product_size}."
                )
            log.warning(
                "Only %d pairs are possible (%d x %d papers); judging all of them instead of "
                "n_pairs=%d (sampling.cap_n_pairs).",
                product_size,
                len(papers_a),
                len(papers_b),
                n_pairs,
            )
            n_pairs = product_size

        rng = random.Random(seed)
        all_indices = [(i, j) for i in range(len(papers_a)) for j in range(len(papers_b))]

        if with_replacement:
            sampled = rng.choices(all_indices, k=n_pairs)
        else:
            sampled = rng.sample(all_indices, k=n_pairs)

        raw_pairs = [(i, j, 0) for i, j in sampled]
    else:
        raise ValueError(f"Unknown sampling mode: {mode!r}. Expected 'all' or 'random'.")

    # Build comparison tasks with A/B randomization
    tasks = []
    for idx, (i, j, repeat_idx) in enumerate(raw_pairs):
        comparison_id = f"cmp_{idx + 1:04d}"

        paper_from_a = papers_a[i]
        paper_from_b = papers_b[j]

        position_rng = random.Random(seed + idx)
        swap = position_rng.random() < 0.5

        if swap:
            paper_a_slot = paper_from_b
            paper_b_slot = paper_from_a
            paper_a_experiment = experiment_b_label
            paper_b_experiment = experiment_a_label
        else:
            paper_a_slot = paper_from_a
            paper_b_slot = paper_from_b
            paper_a_experiment = experiment_a_label
            paper_b_experiment = experiment_b_label

        tasks.append(
            {
                "comparison_id": comparison_id,
                "paper_a": paper_a_slot,
                "paper_b": paper_b_slot,
                "paper_a_experiment": paper_a_experiment,
                "paper_b_experiment": paper_b_experiment,
                "repeat_index": repeat_idx,
            }
        )

    return tasks


def _build_summary(
    run_id: str,
    run_datetime: str,
    duration_seconds: float,
    collection_info: dict,
    iterations: str | list[int],
    cumulative: bool,
    sampling_cfg: dict[str, Any],
    papers_a: list[dict],
    papers_b: list[dict],
    comparisons: list[dict],
    parallelism_cfg: dict[str, Any],
    paper_data: list[dict[str, Any]] | None = None,
    label_a: str = "experiment_a",
    label_b: str = "experiment_b",
) -> dict[str, Any]:
    """Build the run_summary.json content for the paired pipeline."""
    # --- Paired-specific: experiments breakdown ---
    experiments_summary: dict[str, Any] = {}
    for key in ("experiment_a", "experiment_b"):
        info = collection_info[key]
        papers = info["papers"]
        papers_by_snapshot: dict[str, int] = {}
        for p in papers:
            sd = p.get("source_snapshot_dir", "")
            papers_by_snapshot[sd] = papers_by_snapshot.get(sd, 0) + 1
        experiments_summary[info["label"]] = {
            "label": info["label"],
            "snapshot_dirs": info["snapshot_dirs"],
            "num_papers": len(papers),
            "papers_by_snapshot": papers_by_snapshot,
        }

    # --- Paired-specific: paper_data breakdown ---
    if paper_data is None:
        paper_data = []
    count_a = sum(1 for m in paper_data if m.get("experiment") == label_a)
    count_b = sum(1 for m in paper_data if m.get("experiment") == label_b)
    with_system_metrics = sum(1 for m in paper_data if m.get("metrics") is not None)
    paper_data_summary = {
        "total": len(paper_data),
        "experiment_a": count_a,
        "experiment_b": count_b,
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
        total_unique_pairs=len(papers_a) * len(papers_b),
        comparisons=comparisons,
        parallelism_cfg=parallelism_cfg,
        paper_data=paper_data,
        experiments_summary=experiments_summary,
        paper_data_summary=paper_data_summary,
        collection_info=collection_info,
    )


def _print_collection_summary(
    collection_info: dict, iterations: str | list[int], cumulative: bool = True
) -> None:
    """Print the paper collection summary to console and log."""
    cum_label = "cumulative" if cumulative else "per-iteration"
    lines = [
        "",
        "=" * 60,
        "PAPER COLLECTION SUMMARY",
        "=" * 60,
        f"Iteration selection: {iterations} ({cum_label})",
        "",
    ]

    for key in ("experiment_a", "experiment_b"):
        info = collection_info[key]
        label = info["label"]
        papers = info["papers"]

        by_dir: dict[str, list[dict]] = {}
        for p in papers:
            sd = p.get("source_snapshot_dir", "?")
            by_dir.setdefault(sd, []).append(p)

        exp_label = "A" if key == "experiment_a" else "B"
        lines.append(f'Experiment {exp_label} ("{label}"):')
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

    product = len(collection_info["experiment_a"]["papers"]) * len(
        collection_info["experiment_b"]["papers"]
    )
    lines.append(
        f"Possible cross-experiment pairs: "
        f"{len(collection_info['experiment_a']['papers'])} x "
        f"{len(collection_info['experiment_b']['papers'])} = {product}"
    )
    lines.append("=" * 60)

    summary_text = "\n".join(lines)
    print(summary_text)
    log.info(summary_text)


def _print_selection_summary(selection_info: dict[str, Any]) -> None:
    """Print paper selection effects to console and log."""
    lines = [
        "",
        "-" * 60,
        "PAPER SELECTION",
        "-" * 60,
    ]

    filtering = selection_info.get("filtering", {})
    include_rejected = filtering.get("include_rejected", False)
    exclude_timeout = filtering.get("exclude_timeout", True)
    paper_kind_filter_a = filtering.get("paper_kind_filter_a", "all")
    paper_kind_filter_b = filtering.get("paper_kind_filter_b", "all")
    counts_a = filtering.get("counts_a", {})
    counts_b = filtering.get("counts_b", {})
    lines.append(f"Paper kind filter: A={paper_kind_filter_a}, B={paper_kind_filter_b}")

    if include_rejected:
        n_acc_a = counts_a.get("accepted", 0)
        n_rej_a = counts_a.get("rejected", 0)
        n_acc_b = counts_b.get("accepted", 0)
        n_rej_b = counts_b.get("rejected", 0)
        lines.append("Status filter: accepted + rejected")
        lines.append(f"  Experiment A: {n_acc_a} accepted, {n_rej_a} rejected")
        lines.append(f"  Experiment B: {n_acc_b} accepted, {n_rej_b} rejected")
    else:
        lines.append("Status filter: accepted only")

    if exclude_timeout:
        lines.append("Timeout filter: excluded")
    else:
        n_to_a = counts_a.get("timeout", 0)
        n_to_b = counts_b.get("timeout", 0)
        total_to = n_to_a + n_to_b
        if total_to > 0:
            lines.append(f"Timeout filter: included ({n_to_a} in A, {n_to_b} in B)")
        else:
            lines.append("Timeout filter: included (0 found)")

    top_n = selection_info.get("top_n", {})
    if top_n.get("enabled"):
        n = top_n["n"]
        per_iter = top_n.get("per_iteration", False)
        n_label = f"n={n} per iteration" if per_iter else f"n={n}"
        lines.append(f"Top-N: enabled ({n_label})")
        lines.append(
            f"  Experiment A: {selection_info['original_count_a']} -> "
            f"{top_n['count_a_after']} papers"
        )
        if per_iter and "per_iteration_counts_a" in top_n:
            breakdown = ", ".join(
                f"iter {it}: {cnt}" for it, cnt in top_n["per_iteration_counts_a"].items()
            )
            lines.append(f"    ({breakdown})")
        lines.append(
            f"  Experiment B: {selection_info['original_count_b']} -> "
            f"{top_n['count_b_after']} papers"
        )
        if per_iter and "per_iteration_counts_b" in top_n:
            breakdown = ", ".join(
                f"iter {it}: {cnt}" for it, cnt in top_n["per_iteration_counts_b"].items()
            )
            lines.append(f"    ({breakdown})")
    else:
        lines.append("Top-N: disabled")

    balance = selection_info.get("balance", {})
    if balance.get("enabled"):
        method = balance["method"]
        lines.append(f"Balance: enabled (method={method})")
        lines.append(
            f"  Experiment A: {balance['count_a_before']} -> {balance['count_a_after']} papers"
        )
        lines.append(
            f"  Experiment B: {balance['count_b_before']} -> {balance['count_b_after']} papers"
        )
    else:
        lines.append("Balance: disabled")

    lines.append(
        f"Final: {selection_info['final_count_a']} (A) x {selection_info['final_count_b']} (B)"
    )
    lines.append("-" * 60)

    text = "\n".join(lines)
    print(text)
    log.info(text)


def _print_sampling_summary(
    tasks: list[dict],
    sampling_cfg: dict[str, Any],
    n_papers_a: int,
    n_papers_b: int,
) -> None:
    """Print the sampling summary to console and log."""
    mode = sampling_cfg.get("mode", "all")
    seed = sampling_cfg.get("seed", 42)
    product = n_papers_a * n_papers_b

    lines = [
        "",
        "=" * 60,
        "SAMPLING SUMMARY",
        "=" * 60,
    ]

    if mode == "all":
        n_repeats = sampling_cfg.get("n_repeats", 1)
        lines.append(f"Mode: all (n_repeats={n_repeats})")
    else:
        n_pairs = sampling_cfg.get("n_pairs", 100)
        wr = sampling_cfg.get("with_replacement", False)
        lines.append(f"Mode: random (n_pairs={n_pairs}, with_replacement={wr})")

    lines.append(f"Possible unique pairs: {product}")
    lines.append(f"Total comparisons: {len(tasks)}")
    lines.append(f"A/B position: randomized (seed={seed})")
    lines.append("=" * 60)

    summary_text = "\n".join(lines)
    print(summary_text)
    log.info(summary_text)
