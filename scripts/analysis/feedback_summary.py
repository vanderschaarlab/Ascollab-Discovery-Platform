"""Summarise feedback-experiment runs (experiment 04).

    python -m scripts.analysis.feedback_summary <run dir or iteration dir>... \
        [--ranker-output outputs/ranker/04/kirc/<a>_vs_<b>/<rubric>/seed1] [--output summary.json]

For each snapshot's last (or given) iteration: papers produced, accepted / rejected / timed out,
mean meta-review score, mean review metrics, research-session totals (tokens, turns, code runs,
collaboration, tool calls; including collaborators' work inside other agents' sessions, which
``collab_sessions`` counts apart from ``sessions``) and how many agents received feedback. With ``--ranker-output``, also
the win counts in each ranker output directory. Prints JSON, or writes it to ``--output``.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from statistics import mean
from typing import Any

_ITER_RE = re.compile(r"^iteration_(\d+)$")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize one-round metacognition feedback experiment snapshots."
    )
    parser.add_argument(
        "snapshot_dirs", nargs="+", help="Run dirs or iteration_<N> dirs to summarize."
    )
    parser.add_argument(
        "--ranker-output",
        action="append",
        default=[],
        help="Optional paired-ranker output directory containing comparisons.json.",
    )
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    summaries = [_summarize_snapshot(Path(p)) for p in args.snapshot_dirs]
    payload = {
        "snapshots": summaries,
        "ranker": [_summarize_ranker_output(Path(p)) for p in args.ranker_output],
    }

    text = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def _summarize_snapshot(path: Path) -> dict[str, Any]:
    iteration_dir = _resolve_iteration_dir(path)
    iteration = _iteration_number(iteration_dir)
    private_data = _load_json(iteration_dir / "private_archive_data.json")
    papers = [
        p
        for p in _flatten_papers(private_data.get("papers", []))
        if _as_int(p.get("publication_t"), -1) == iteration
    ]
    metrics_by_paper = _load_metrics(iteration_dir / "metrics.jsonl")
    _merge_paper_metrics(metrics_by_paper, papers)
    session_stats = _load_pickle(iteration_dir / "session_stats_data.pkl") or {}
    research_stats = _summarize_research_stats(session_stats, iteration)
    metric_means = _metric_means(metrics_by_paper)
    feedback_path = iteration_dir / "metacognition_feedback.json"
    feedback = _load_json(feedback_path) if feedback_path.exists() else {}

    metareview_scores = []
    for paper in papers:
        mr = paper.get("metareview")
        if isinstance(mr, Mapping) and isinstance(mr.get("overall_score"), (int, float)):
            metareview_scores.append(float(mr["overall_score"]))

    statuses = [str(p.get("status", "") or "").strip().lower() for p in papers]
    return {
        "snapshot_dir": str(iteration_dir.parent),
        "iteration_dir": str(iteration_dir),
        "iteration": iteration,
        "run_name": iteration_dir.parent.name,
        "metacognition_mode": feedback.get("mode"),
        "metacognition_source_iteration": feedback.get("source_iteration"),
        "agents_with_feedback": len(feedback.get("feedback_by_agent", {}) or {}) if feedback else 0,
        "papers_produced": len(papers),
        "accepted_count": sum(1 for s in statuses if s == "accepted"),
        "rejected_count": sum(1 for s in statuses if s == "rejected"),
        "timeout_or_incomplete_count": sum(
            1
            for s in statuses
            if s in {"timeout", "incomplete"} or "timeout" in s or "incomplete" in s
        ),
        "mean_metareview_score": mean(metareview_scores) if metareview_scores else None,
        "research": research_stats,
        "metric_means": metric_means,
    }


def _summarize_research_stats(session_stats: Mapping[str, Any], iteration: int) -> dict[str, Any]:
    registry = session_stats.get("registry", {}) if isinstance(session_stats, Mapping) else {}
    iter_data = registry.get(iteration) or registry.get(str(iteration)) or {}
    research = iter_data.get("research", {}) if isinstance(iter_data, Mapping) else {}
    totals = {
        "sessions": 0,
        "collab_sessions": 0,
        "total_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "turns": 0,
        "code_executions": 0,
        "code_executions_successful": 0,
        "lines_of_code": 0,
        "collab_established": 0,
        "collab_interactions": 0,
        "tool_calls": {},
    }
    for session_id, raw in research.items() if isinstance(research, Mapping) else []:
        row = _stats_row(raw)
        # A collaborator's work inside another agent's session is logged as
        # `<collab>_tNN_collab_for_<author>`: its tokens and tool calls are real and counted
        # below, but it is not a research session of its own.
        if "_collab_for_" in str(session_id):
            totals["collab_sessions"] += 1
        else:
            totals["sessions"] += 1
        for key in (
            "total_tokens",
            "input_tokens",
            "output_tokens",
            "turns",
            "code_executions",
            "code_executions_successful",
            "lines_of_code",
            "collab_established",
            "collab_interactions",
        ):
            totals[key] += _as_int(row.get(key), 0)
        tool_calls = row.get("tool_calls", {})
        if isinstance(tool_calls, Mapping):
            for name, count in tool_calls.items():
                name = str(name)
                totals["tool_calls"][name] = totals["tool_calls"].get(name, 0) + _as_int(count, 0)
    return totals


def _summarize_ranker_output(path: Path) -> dict[str, Any]:
    comparisons_path = path / "comparisons.json"
    if not comparisons_path.exists():
        return {"path": str(path), "available": False}
    rows = _load_json(comparisons_path)
    if not isinstance(rows, list):
        return {"path": str(path), "available": False}
    completed = [r for r in rows if isinstance(r, Mapping) and r.get("status") == "completed"]
    wins: dict[str, int] = {}
    for row in completed:
        outcome = str(row.get("outcome", "") or "").strip().upper()
        if outcome == "A":
            winner = row.get("paper_a_experiment")
        elif outcome == "B":
            winner = row.get("paper_b_experiment")
        else:
            winner = None
        if winner:
            key = str(winner)
            wins[key] = wins.get(key, 0) + 1
    total = sum(wins.values())
    return {
        "path": str(path),
        "available": True,
        "completed": len(completed),
        "wins": wins,
        "win_rates": {k: (v / total if total else None) for k, v in wins.items()},
    }


def _resolve_iteration_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if _ITER_RE.match(path.name):
        return path
    candidates = []
    for child in path.iterdir():
        if not child.is_dir():
            continue
        match = _ITER_RE.match(child.name)
        if match:
            candidates.append((int(match.group(1)), child))
    if not candidates:
        raise FileNotFoundError(f"No iteration_<N> directories found under {path}")
    return sorted(candidates)[-1][1]


def _iteration_number(iteration_dir: Path) -> int:
    match = _ITER_RE.match(iteration_dir.name)
    if not match:
        raise ValueError(f"Not an iteration directory: {iteration_dir}")
    return int(match.group(1))


def _load_json(path: Path) -> Any:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_pickle(path: Path) -> Any:
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return pickle.load(f)


def _flatten_papers(raw: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in raw or []:
        if isinstance(item, list):
            out.extend(dict(x) for x in item if isinstance(x, Mapping))
        elif isinstance(item, Mapping):
            out.append(dict(item))
    return out


def _load_metrics(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(row, Mapping) and isinstance(row.get("metrics"), Mapping):
                paper_id = str(row.get("paper_id", "") or "").strip()
                if paper_id:
                    out[paper_id] = dict(row["metrics"])
    return out


def _metric_means(metrics_by_paper: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    buckets: dict[str, dict[str, list[float]]] = {}
    for metrics in metrics_by_paper.values():
        if not isinstance(metrics, Mapping):
            continue
        for category, values in metrics.items():
            if not isinstance(values, Mapping):
                continue
            for key, value in values.items():
                if isinstance(value, bool):
                    number = 1.0 if value else 0.0
                elif isinstance(value, (int, float)):
                    number = float(value)
                else:
                    continue
                buckets.setdefault(str(category), {}).setdefault(str(key), []).append(number)
    return {
        category: {key: mean(values) for key, values in sorted(metrics.items()) if values}
        for category, metrics in sorted(buckets.items())
    }


def _merge_paper_metrics(
    metrics_by_paper: dict[str, dict[str, Any]], papers: Iterable[Mapping[str, Any]]
) -> None:
    for paper in papers or []:
        paper_id = str(paper.get("paper_id", "") or "").strip()
        if not paper_id or paper_id in metrics_by_paper:
            continue
        metrics = paper.get("review_metrics", {})
        if isinstance(metrics, Mapping) and metrics:
            metrics_by_paper[paper_id] = dict(metrics)


def _stats_row(raw: Any) -> Mapping[str, Any]:
    if isinstance(raw, Mapping):
        return raw
    data = getattr(raw, "__dict__", None)
    return data if isinstance(data, Mapping) else {}


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


if __name__ == "__main__":
    main()
