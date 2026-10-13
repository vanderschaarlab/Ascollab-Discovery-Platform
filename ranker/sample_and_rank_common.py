"""Shared helpers for sample-and-rank pipelines (paired and non-paired).

Contains utility functions used by both ``sample_and_rank_paired`` and
``sample_and_rank_non_paired``.
"""

import dataclasses
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from .ranker_core import run_ranker
from .snapshot_import import collect_metrics, collect_papers, paper_uid, to_ranker_format

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paper collection helper
# ---------------------------------------------------------------------------


def collect_papers_for_pool(
    snapshot_dirs: list[str],
    paper_selection_cfg: dict[str, Any] | None = None,
    paper_kind_filter_override: str | None = None,
) -> list[dict]:
    """Gather papers from snapshot dirs using ``collect_papers()`` with filtering flags.

    Reads ``iterations``, ``cumulative``, ``include_rejected``, and
    ``exclude_timeout`` from *paper_selection_cfg*.

    Args:
        snapshot_dirs: List of snapshot directory paths.
        paper_selection_cfg: Dict with ``iterations``, ``cumulative``,
            ``include_rejected``, ``exclude_timeout`` keys.
        paper_kind_filter_override: Optional override for
            ``paper_selection.paper_kind_filter`` when the caller needs
            per-pool filtering in paired mode.

    Returns:
        List of paper dicts as returned by :func:`collect_papers`.
    """
    if paper_selection_cfg is None:
        paper_selection_cfg = {}

    iterations = paper_selection_cfg.get("iterations", "last")
    cumulative = paper_selection_cfg.get("cumulative", True)
    include_rejected = paper_selection_cfg.get("include_rejected", False)
    exclude_timeout = paper_selection_cfg.get("exclude_timeout", True)
    paper_kind_filter = (
        paper_kind_filter_override
        if paper_kind_filter_override is not None
        else paper_selection_cfg.get("paper_kind_filter", "all")
    )

    return collect_papers(
        snapshot_dirs,
        iterations=iterations,
        cumulative=cumulative,
        include_rejected=include_rejected,
        exclude_timeout=exclude_timeout,
        paper_kind_filter=paper_kind_filter,
    )


# ---------------------------------------------------------------------------
# Paper status / sorting helpers
# ---------------------------------------------------------------------------


def _count_paper_statuses(papers: list[dict]) -> dict[str, int]:
    """Count papers by status and timeout for filtering reports."""
    accepted = 0
    rejected = 0
    timeout = 0
    for p in papers:
        status = p.get("status")
        if status == "accepted":
            accepted += 1
        elif status == "rejected":
            rejected += 1
        pid = p.get("paper_id", "")
        manuscript = p.get("manuscript", "")
        if pid.endswith("_timeout") or manuscript == "TIMEOUT":
            timeout += 1
    return {"accepted": accepted, "rejected": rejected, "timeout": timeout}


def _metareview_sort_key(paper: dict) -> float:
    """Return the metareview overall_score for sorting (higher = better).

    Papers without a metareview or without an ``overall_score`` return
    ``-inf`` so they sort to the bottom.
    """
    mr = paper.get("metareview")
    if mr is None:
        return float("-inf")
    score = mr.get("overall_score")
    if score is None:
        return float("-inf")
    return score


def _top_n(papers: list[dict], n: int, label: str) -> list[dict]:
    """Return the *n* highest-scoring papers by metareview ``overall_score``."""
    null_count = sum(1 for p in papers if _metareview_sort_key(p) == float("-inf"))
    if null_count:
        log.warning(
            "Experiment %s: %d/%d papers lack a metareview score; "
            "they will be ranked lowest in top-N selection.",
            label,
            null_count,
            len(papers),
        )
    ranked = sorted(papers, key=_metareview_sort_key, reverse=True)
    return ranked[:n]


def _top_n_per_iteration(papers: list[dict], n: int, label: str) -> list[dict]:
    """Return the top *n* papers per ``source_iteration`` group.

    Groups papers by ``source_iteration``, applies ``_top_n`` to each group,
    and returns the combined result.
    """
    from collections import defaultdict

    by_iter: dict[int, list[dict]] = defaultdict(list)
    for p in papers:
        by_iter[p["source_iteration"]].append(p)

    result: list[dict] = []
    for iteration in sorted(by_iter):
        group = by_iter[iteration]
        selected = _top_n(group, n, f"{label}/iter_{iteration}")
        result.extend(selected)
    return result


def _count_by_iteration(papers: list[dict]) -> dict[int, int]:
    """Count papers per ``source_iteration``, returned in sorted order."""
    counts: dict[int, int] = {}
    for p in papers:
        it = p["source_iteration"]
        counts[it] = counts.get(it, 0) + 1
    return dict(sorted(counts.items()))


# ---------------------------------------------------------------------------
# Single comparison execution
# ---------------------------------------------------------------------------


def _run_single_comparison(
    task: dict,
    llm: BaseChatModel,
    ranker_cfg: dict[str, Any],
    sandbox_client: Any,
    iteration_selection: str | list[int],
    cumulative: bool = True,
    dataset_desc_lookup: dict[str, dict[str, str]] | None = None,
    cited_papers_lookup: dict[str, list[dict[str, Any]]] | None = None,
) -> dict:
    """Run a single pairwise comparison.

    Strips provenance via to_ranker_format() before calling run_ranker().
    Returns a comparison record dict (with full provenance from the task).
    """
    comparison_id = task["comparison_id"]
    start_time = time.time()

    try:
        paper_a_ranker = to_ranker_format(task["paper_a"])
        paper_b_ranker = to_ranker_format(task["paper_b"])

        tools_cfg = ranker_cfg.get("tools", {})
        llm_retry_cfg = ranker_cfg.get("llm_retry", {})
        max_iterations = ranker_cfg.get("max_iterations", 10)
        criteria_prompt = ranker_cfg.get("criteria_prompt")
        max_parse_retries = 2
        output_parsing = ranker_cfg.get("output_parsing", {})
        if output_parsing:
            max_parse_retries = output_parsing.get("max_parse_retries", 2)

        # Extract new config flags
        include_code = ranker_cfg.get("include_code", True)
        include_dataset_description = ranker_cfg.get("include_dataset_description", True)
        include_cited_papers = ranker_cfg.get("include_cited_papers", True)
        cited_papers_order = ranker_cfg.get("cited_papers_order", "newest_first")
        paper_separator = ranker_cfg.get("paper_separator")
        prompt_section_notes = ranker_cfg.get("prompt_section_notes")

        # Look up dataset descriptions for each paper
        dataset_descriptions_a = None
        dataset_descriptions_b = None
        if dataset_desc_lookup:
            snap_dir_a = task["paper_a"].get("source_snapshot_dir", "")
            snap_dir_b = task["paper_b"].get("source_snapshot_dir", "")
            dataset_descriptions_a = dataset_desc_lookup.get(snap_dir_a)
            dataset_descriptions_b = dataset_desc_lookup.get(snap_dir_b)

        # Look up citation chains for each paper
        cited_papers_a = None
        cited_papers_b = None
        if cited_papers_lookup:
            cited_papers_a = cited_papers_lookup.get(paper_uid(task["paper_a"]))
            cited_papers_b = cited_papers_lookup.get(paper_uid(task["paper_b"]))

        mode = ranker_cfg.get("mode", "comparison")

        result = run_ranker(
            paper_a=paper_a_ranker,
            paper_b=paper_b_ranker,
            llm=llm,
            criteria_prompt=criteria_prompt,
            mode=mode,
            tools_cfg=tools_cfg,
            sandbox_client=sandbox_client,
            max_iterations=max_iterations,
            llm_retry_cfg=llm_retry_cfg,
            max_parse_retries=max_parse_retries,
            ranker_id=comparison_id,
            include_code=include_code,
            include_dataset_description=include_dataset_description,
            dataset_descriptions_a=dataset_descriptions_a,
            dataset_descriptions_b=dataset_descriptions_b,
            include_cited_papers=include_cited_papers,
            cited_papers_a=cited_papers_a,
            cited_papers_b=cited_papers_b,
            cited_papers_order=cited_papers_order,
            paper_separator=paper_separator,
            prompt_section_notes=prompt_section_notes,
        )

        duration = round(time.time() - start_time, 1)

        # An unparseable verdict is NOT a Paper B win: record it under its own
        # status so it is excluded from win-rate denominators instead of biasing
        # them in a fixed direction.
        if not result.valid_result or result.score is None:
            outcome = None
            status = "unparsed_verdict"
        elif mode == "hypothesis_similarity":
            outcome = "different" if result.score == 1 else "same"
            status = "completed"
        else:
            outcome = "A" if result.score == 1 else "B"
            status = "completed"

        record = _build_comparison_record(
            task=task,
            status=status,
            score=result.score,
            outcome=outcome,
            justification=result.justification,
            turns=result.turns,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            tool_calls=result.tool_calls,
            duration_seconds=duration,
            iteration_selection=iteration_selection,
            cumulative=cumulative,
            run_id=None,
            run_datetime=None,
            error=None,
        )

        if result.trace is not None:
            record["_trace"] = dataclasses.asdict(result.trace)

        return record

    except Exception as exc:
        duration = round(time.time() - start_time, 1)
        log.error("Comparison %s failed: %s", comparison_id, exc)
        record = _build_comparison_record(
            task=task,
            status="failed",
            score=None,
            outcome=None,
            justification=None,
            turns=0,
            input_tokens=0,
            output_tokens=0,
            tool_calls={},
            duration_seconds=duration,
            iteration_selection=iteration_selection,
            cumulative=cumulative,
            run_id=None,
            run_datetime=None,
            error=str(exc),
        )
        return record


# ---------------------------------------------------------------------------
# Comparison record builders
# ---------------------------------------------------------------------------


def _build_comparison_record(
    task: dict,
    status: str,
    score: int | None,
    outcome: str | None,
    justification: str | None,
    turns: int,
    input_tokens: int,
    output_tokens: int,
    tool_calls: dict,
    duration_seconds: float,
    iteration_selection: str | list[int],
    cumulative: bool,
    run_id: str | None,
    run_datetime: str | None,
    error: str | None,
) -> dict[str, Any]:
    """Build a flat comparison record dict."""
    pa = task["paper_a"]
    pb = task["paper_b"]

    return {
        "comparison_id": task["comparison_id"],
        "run_id": run_id,
        "run_datetime": run_datetime,
        "status": status,
        "score": score,
        "outcome": outcome,
        "justification": justification,
        "paper_a_id": pa.get("paper_id", ""),
        "paper_a_title": pa.get("title", ""),
        "paper_a_experiment": task["paper_a_experiment"],
        "paper_a_source_snapshot": pa.get("source_snapshot", ""),
        "paper_a_source_snapshot_dir": pa.get("source_snapshot_dir", ""),
        "paper_a_source_iteration": pa.get("source_iteration", 0),
        "paper_a_primary_agent_id": pa.get("primary_agent_id", ""),
        "paper_b_id": pb.get("paper_id", ""),
        "paper_b_title": pb.get("title", ""),
        "paper_b_experiment": task["paper_b_experiment"],
        "paper_b_source_snapshot": pb.get("source_snapshot", ""),
        "paper_b_source_snapshot_dir": pb.get("source_snapshot_dir", ""),
        "paper_b_source_iteration": pb.get("source_iteration", 0),
        "paper_b_primary_agent_id": pb.get("primary_agent_id", ""),
        "iteration_selection": iteration_selection,
        "cumulative": cumulative,
        "repeat_index": task.get("repeat_index", 0),
        "turns": turns,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tool_calls": tool_calls,
        "duration_seconds": duration_seconds,
        "ranker_config_ref": "ranker_config.json",
        "experiment_configs_ref": "experiment_configs/",
        "ranker_trace_ref": f"ranker_traces/{task['comparison_id']}.json",
        "error": error,
    }


def _make_failed_record(
    task: dict,
    exc: Exception,
    run_id: str | None = None,
    run_datetime: str | None = None,
    iteration_selection: str | list[int] = "last",
    cumulative: bool = True,
) -> dict:
    """Build a comparison record with status='failed' from a task and exception."""
    return _build_comparison_record(
        task=task,
        status="failed",
        score=None,
        outcome=None,
        justification=None,
        turns=0,
        input_tokens=0,
        output_tokens=0,
        tool_calls={},
        duration_seconds=0.0,
        iteration_selection=iteration_selection,
        cumulative=cumulative,
        run_id=run_id,
        run_datetime=run_datetime,
        error=str(exc),
    )


# ---------------------------------------------------------------------------
# Parallel execution
# ---------------------------------------------------------------------------


def _run_comparisons_parallel(
    tasks: list[dict],
    llm: "BaseChatModel",
    ranker_cfg: dict[str, Any],
    sandbox_client: Any,
    iterations: str | list[int],
    cumulative: bool,
    parallelism_cfg: dict[str, Any],
    run_id: str,
    run_datetime: str,
    dataset_desc_lookup: dict[str, dict[str, str]] | None = None,
    cited_papers_lookup: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[list[dict], dict[str, Any]]:
    """Run comparison tasks in parallel and return (comparisons, traces).

    Uses a :class:`ThreadPoolExecutor` with progress logging.  Records are
    filled with *run_id*/*run_datetime* and sorted by ``comparison_id``
    before returning.
    """
    comparisons: list[dict] = []
    traces: dict[str, Any] = {}
    completed_count = 0
    failed_count = 0
    lock = threading.Lock()
    total_tasks = len(tasks)

    def _worker(task):
        return _run_single_comparison(
            task,
            llm,
            ranker_cfg,
            sandbox_client,
            iterations,
            cumulative,
            dataset_desc_lookup=dataset_desc_lookup,
            cited_papers_lookup=cited_papers_lookup,
        )

    max_workers = parallelism_cfg.get("max_workers", 4)

    # Log ranker configuration report before starting comparisons
    mode = ranker_cfg.get("mode", "comparison")
    criteria = ranker_cfg.get("criteria_prompt", "(default)")
    log.info(
        "\nRanker configuration report:\n"
        "----------------------------\n"
        "Mode: %s\n"
        "Criteria prompt:\n```\n%s\n```\n"
        "Total comparisons: %d\n"
        "Max workers: %d",
        mode,
        criteria,
        total_tasks,
        max_workers,
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(_worker, t): t for t in tasks}
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                record = future.result()
                with lock:
                    comparisons.append(record)
                    if record.get("status") == "completed":
                        completed_count += 1
                    else:
                        failed_count += 1
                    if "_trace" in record:
                        traces[record["comparison_id"]] = record.pop("_trace")
                    total = completed_count + failed_count
                    _log_progress(total, total_tasks, record)
            except Exception as exc:
                with lock:
                    failed_record = _make_failed_record(
                        task, exc, run_id, run_datetime, iterations, cumulative
                    )
                    comparisons.append(failed_record)
                    failed_count += 1
                    total = completed_count + failed_count
                    _log_progress(total, total_tasks, failed_record)

    # Fill in run_id and run_datetime on all records, then sort
    for c in comparisons:
        c["run_id"] = run_id
        c["run_datetime"] = run_datetime
    comparisons.sort(key=lambda r: r["comparison_id"])

    return comparisons, traces


# ---------------------------------------------------------------------------
# Paper data collection
# ---------------------------------------------------------------------------


def _collect_paper_data_group(
    papers: list[dict],
    label: str,
    condition: str | None,
) -> list[dict[str, Any]]:
    """Collect comprehensive paper data for a single group of papers.

    Args:
        papers: List of paper dicts.
        label: Human-friendly label (experiment name or pool label).
        condition: Experiment condition (``"A"``, ``"B"``) or ``None``
            for non-paired.

    Returns:
        List of enriched paper records with ``experiment``,
        ``experiment_condition``, and ``metrics`` keys.
    """
    metrics_map = collect_metrics(papers)
    records: list[dict[str, Any]] = []
    for paper in papers:
        m = metrics_map.get(paper_uid(paper))
        record = dict(paper)
        record["experiment"] = label
        record["experiment_condition"] = condition
        record["metrics"] = m.get("metrics") if m else None
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------


def _build_summary_base(
    *,
    run_id: str,
    run_datetime: str,
    duration_seconds: float,
    iterations: str | list[int],
    cumulative: bool,
    sampling_cfg: dict[str, Any],
    total_unique_pairs: int,
    comparisons: list[dict],
    parallelism_cfg: dict[str, Any],
    paper_data: list[dict[str, Any]] | None,
    experiments_summary: dict[str, Any],
    paper_data_summary: dict[str, Any],
    extra_sampling: dict[str, Any] | None = None,
    collection_info: dict | None = None,
) -> dict[str, Any]:
    """Build the common ``run_summary.json`` skeleton.

    Pipeline-specific sections (experiments, paper-data breakdown, extra
    sampling fields) are passed in by the caller.

    Args:
        experiments_summary: Placed at ``summary["experiments"]``.
        paper_data_summary: Placed at ``summary["paper_data"]``.
        extra_sampling: Merged into the ``sampling`` sub-dict
            (e.g. ``{"allow_self_comparison": False}``).
        collection_info: If it contains a ``"paper_selection"`` key the
            value is copied into the summary.
    """
    completed = [c for c in comparisons if c["status"] == "completed"]
    failed = [c for c in comparisons if c["status"] == "failed"]
    # Rows where the judge ran but emitted no parseable verdict. Counted
    # explicitly so a run with a high failure rate cannot look healthy just
    # because those rows silently left the `completed` denominator.
    unparsed = [c for c in comparisons if c["status"] == "unparsed_verdict"]

    total_input_tokens = sum(c.get("input_tokens", 0) for c in completed)
    total_output_tokens = sum(c.get("output_tokens", 0) for c in completed)
    total_duration = sum(c.get("duration_seconds", 0) for c in completed)
    mean_duration = round(total_duration / len(completed), 1) if completed else 0.0

    sampling_summary: dict[str, Any] = {
        "mode": sampling_cfg.get("mode", "all"),
        "n_repeats": sampling_cfg.get("n_repeats", 1),
        "with_replacement": sampling_cfg.get("with_replacement", False),
        "total_unique_pairs": total_unique_pairs,
        "total_comparisons": len(comparisons),
        "seed": sampling_cfg.get("seed", 42),
    }
    if extra_sampling:
        sampling_summary.update(extra_sampling)

    summary: dict[str, Any] = {
        "run_id": run_id,
        "run_datetime": run_datetime,
        "duration_seconds": duration_seconds,
        "experiments": experiments_summary,
        "iteration_selection": iterations,
        "cumulative": cumulative,
        "sampling": sampling_summary,
        "results": {
            "completed": len(completed),
            "failed": len(failed),
            "unparsed_verdict": len(unparsed),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_duration_seconds": round(total_duration, 1),
            "mean_duration_seconds": mean_duration,
        },
        "paper_data": paper_data_summary,
        "parallelism": {
            "max_workers": parallelism_cfg.get("max_workers", 4),
        },
        "ranker_config_ref": "ranker_config.json",
        "experiment_configs_ref": "experiment_configs/",
    }

    if collection_info:
        paper_selection = collection_info.get("paper_selection")
        if paper_selection:
            summary["paper_selection"] = paper_selection

    return summary


# ---------------------------------------------------------------------------
# Output / save helpers
# ---------------------------------------------------------------------------


def _save_results(
    comparisons: list[dict],
    summary: dict,
    ranker_cfg: dict[str, Any],
    experiment_configs: dict[str, str],
    traces: dict[str, Any],
    output_dir: str,
    paper_data: list[dict[str, Any]] | None = None,
) -> None:
    """Write all output files to the output directory."""
    # comparisons.json
    comparisons_path = os.path.join(output_dir, "comparisons.json")
    with open(comparisons_path, "w") as f:
        json.dump(comparisons, f, indent=2, ensure_ascii=False)
    log.info("Wrote %s", comparisons_path)

    # run_summary.json
    summary_path = os.path.join(output_dir, "run_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info("Wrote %s", summary_path)

    # ranker_config.json
    config_path = os.path.join(output_dir, "ranker_config.json")
    with open(config_path, "w") as f:
        json.dump(ranker_cfg, f, indent=2, ensure_ascii=False)
    log.info("Wrote %s", config_path)

    # experiment_configs/
    exp_configs_dir = os.path.join(output_dir, "experiment_configs")
    os.makedirs(exp_configs_dir, exist_ok=True)
    for run_name, config_content in experiment_configs.items():
        safe_name = run_name.replace("/", "_").replace("\\", "_")
        config_file = os.path.join(exp_configs_dir, f"{safe_name}.yaml")
        with open(config_file, "w") as f:
            f.write(config_content)
    log.info("Wrote %d experiment configs to %s", len(experiment_configs), exp_configs_dir)

    # ranker_traces/
    if traces:
        traces_dir = os.path.join(output_dir, "ranker_traces")
        os.makedirs(traces_dir, exist_ok=True)
        for cmp_id, trace_data in traces.items():
            trace_file = os.path.join(traces_dir, f"{cmp_id}.json")
            with open(trace_file, "w") as f:
                json.dump(trace_data, f, indent=2, ensure_ascii=False)
        log.info("Wrote %d traces to %s", len(traces), traces_dir)

    # papers/paper_data.json
    if paper_data:
        papers_dir = os.path.join(output_dir, "papers")
        os.makedirs(papers_dir, exist_ok=True)
        paper_data_path = os.path.join(papers_dir, "paper_data.json")
        with open(paper_data_path, "w") as f:
            json.dump(paper_data, f, indent=2, ensure_ascii=False)
        log.info("Wrote %d paper records to %s", len(paper_data), paper_data_path)


# ---------------------------------------------------------------------------
# Experiment config collection
# ---------------------------------------------------------------------------


def _collect_experiment_configs(
    snapshot_dirs: list[str],
) -> dict[str, str]:
    """Collect exp_config.yaml content from a flat list of snapshot dir paths.

    Returns mapping from run_name to config YAML string.
    """
    from .snapshot_import import (
        _discover_iterations,
        _latest_readable_iteration,
        _read_snapshot_metadata,
    )

    configs: dict[str, str] = {}
    used_names: dict[str, int] = {}

    for snap_dir in snapshot_dirs:
        snap_path = Path(snap_dir)
        if not snap_path.is_dir():
            log.warning("Snapshot dir %s does not exist, skipping config collection", snap_dir)
            continue

        iter_nums = _discover_iterations(snap_path)
        if not iter_nums:
            log.warning("No iterations found in %s, skipping config collection", snap_dir)
            continue

        latest_iter = _latest_readable_iteration(snap_path)
        if latest_iter is None:
            log.warning("No readable iterations found in %s, skipping config collection", snap_dir)
            continue

        run_name = _read_snapshot_metadata(snap_path, latest_iter)

        config_path = snap_path / f"iteration_{latest_iter}" / "exp_config.yaml"
        if config_path.exists():
            config_content = config_path.read_text()
        else:
            config_content = f"# exp_config.yaml not found in {snap_path}/iteration_{latest_iter}\n"
            log.warning("exp_config.yaml not found: %s", config_path)

        if run_name in used_names:
            used_names[run_name] += 1
            final_name = f"{run_name}_{used_names[run_name]}"
        else:
            used_names[run_name] = 1
            final_name = run_name

        configs[final_name] = config_content

    return configs


# ---------------------------------------------------------------------------
# Logging / printing helpers
# ---------------------------------------------------------------------------


def _log_progress(current: int, total: int, record: dict) -> None:
    """Print a progress line for a completed/failed comparison."""
    cmp_id = record.get("comparison_id", "?")
    status = record.get("status", "?")

    if status == "completed":
        outcome = record.get("outcome", "?")
        turns = record.get("turns", 0)
        duration = record.get("duration_seconds", 0)
        msg = f"[{current}/{total}] {cmp_id} completed: outcome={outcome} (turns={turns}, {duration}s)"
    elif status == "unparsed_verdict":
        turns = record.get("turns", 0)
        duration = record.get("duration_seconds", 0)
        msg = (
            f"[{current}/{total}] {cmp_id} UNPARSED VERDICT: no decision recorded "
            f"(turns={turns}, {duration}s)"
        )
    else:
        error = record.get("error", "unknown error")
        msg = f"[{current}/{total}] {cmp_id} FAILED: {error}"

    print(msg)
    log.info(msg)


def _print_final_summary(summary: dict) -> None:
    """Print the final results summary."""
    results = summary.get("results", {})
    lines = [
        "",
        "=" * 60,
        "RUN COMPLETE",
        "=" * 60,
        f"Run ID: {summary.get('run_id', '?')}",
        f"Total comparisons: {results.get('completed', 0) + results.get('failed', 0) + results.get('unparsed_verdict', 0)}",
        f"  Completed: {results.get('completed', 0)}",
        f"  Failed: {results.get('failed', 0)}",
        f"  Unparsed verdict (no decision recorded): {results.get('unparsed_verdict', 0)}",
        f"Total duration: {summary.get('duration_seconds', 0)}s",
        f"Total tokens: input={results.get('total_input_tokens', 0)}, output={results.get('total_output_tokens', 0)}",
        f"Mean comparison time: {results.get('mean_duration_seconds', 0)}s",
        "=" * 60,
    ]
    summary_text = "\n".join(lines)
    print(summary_text)
    log.info(summary_text)
