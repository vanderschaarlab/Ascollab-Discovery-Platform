"""Aggregate AutoRanker pairwise comparison outputs into per-paper scores."""

from __future__ import annotations

import argparse

from .autoranker_utils import aggregate_win_rate_scores
from .io_utils import read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparisons", required=True, help="Path to comparisons.json or autoranker_pairs.json"
    )
    parser.add_argument("--manifest", help="Optional cohort manifest for expected paper ids")
    parser.add_argument("--output", required=True, help="Output autoranker_scores.json path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparisons = read_json(args.comparisons)
    paper_ids = None
    if args.manifest:
        manifest = read_json(args.manifest)
        paper_ids = manifest.get("candidate_paper_ids", [])
    scores = aggregate_win_rate_scores(comparisons, paper_ids=paper_ids)
    write_json(args.output, scores)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
