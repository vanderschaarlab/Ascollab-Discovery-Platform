"""Analyse the reviewer-matching replay outputs (experiment 03)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .analysis_utils import pair_condition_deltas, summarize_condition
from .io_utils import read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/reviewer_matching")
    parser.add_argument("--accepted-k", type=int, default=5)
    parser.add_argument("--output-dir")
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir) if args.output_dir else root / "analysis"
    rows = []

    for condition_summary_path in sorted((root / "reviews").glob("*/condition_summary.json")):
        condition_dir = condition_summary_path.parent
        manifest = read_json(condition_dir / "manifest.json")
        condition_summary = read_json(condition_summary_path)
        metareviews = read_json(condition_dir / "metareviews.json")
        accepted_path = condition_dir / f"accepted_top{args.accepted_k}.json"
        if not accepted_path.exists():
            continue
        accepted = read_json(accepted_path)["accepted_paper_ids"]

        auto_dir_name = f"{manifest['dataset']}_iter{manifest['target_iteration']}_cohort{manifest['cohort_index']}"
        auto_scores_path = root / "autoranker" / auto_dir_name / "autoranker_scores.json"
        if not auto_scores_path.exists():
            continue
        auto_scores = read_json(auto_scores_path)
        rows.append(
            summarize_condition(
                manifest=manifest,
                condition_summary=condition_summary,
                metareviews=metareviews,
                accepted_paper_ids=accepted,
                autoranker_scores=auto_scores,
                condition_dir=str(condition_dir),
            )
        )

    deltas = pair_condition_deltas(rows)
    _write_csv(output_dir / "per_condition_results.csv", rows)
    _write_csv(output_dir / "per_cohort_deltas.csv", deltas)

    aggregates = []
    by_group: dict[tuple, list[dict]] = {}
    for row in rows:
        by_group.setdefault(
            (row["dataset"], row["target_iteration"], row["reviewer_assignment_strategy"]), []
        ).append(row)
    for (dataset, iteration, strategy), group in sorted(by_group.items()):
        vals = [
            r["mean_accepted_autoranker_score"]
            for r in group
            if r["mean_accepted_autoranker_score"] is not None
        ]
        aggregates.append(
            {
                "dataset": dataset,
                "target_iteration": iteration,
                "reviewer_assignment_strategy": strategy,
                "n": len(vals),
                "mean_of_mean_accepted_autoranker_score": sum(vals) / len(vals) if vals else None,
            }
        )
    _write_csv(output_dir / "aggregate_results.csv", aggregates)
    _write_csv(output_dir / "condition_delta_by_dataset_iteration.csv", deltas)
    print(f"Wrote analysis outputs to {output_dir}")


if __name__ == "__main__":
    main()
