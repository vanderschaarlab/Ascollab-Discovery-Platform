"""Prepare fixed candidate-paper cohorts for the reviewer-matching ablation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .cohort_utils import build_manifest, load_candidate_pool, select_cohort
from .io_utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--snapshot-dirs", nargs="+", required=True)
    parser.add_argument("--target-iterations", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--cohorts-per-iteration", type=int, default=5)
    parser.add_argument("--cohort-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", default="outputs/reviewer_matching/manifests")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    max_iter = max(args.target_iterations)
    papers = load_candidate_pool(args.snapshot_dirs, max_iteration=max_iter)

    for target_iteration in args.target_iterations:
        for cohort_index in range(args.cohorts_per_iteration):
            cohort_seed = int(args.seed) * 1000 + int(target_iteration) * 100 + int(cohort_index)
            selection = select_cohort(
                papers,
                target_iteration=target_iteration,
                cohort_size=args.cohort_size,
                seed=cohort_seed,
            )
            manifest = build_manifest(
                dataset=args.dataset,
                target_iteration=target_iteration,
                cohort_index=cohort_index,
                cohort_seed=cohort_seed,
                cohort_size=args.cohort_size,
                snapshot_dirs=[str(p) for p in args.snapshot_dirs],
                selection=selection,
            )
            out = output_dir / f"{args.dataset}_iter{target_iteration}_cohort{cohort_index}.json"
            write_json(out, manifest)
            print(f"Wrote {out}")


if __name__ == "__main__":
    main()
