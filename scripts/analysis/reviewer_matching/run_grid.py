"""Run the reviewer-matching replay grid (experiment 03): expertise-matched vs random reviewers.

The cohort manifests in ``<root>/manifests`` record the snapshot the cohorts were drawn from. A
later call on the same root (the AutoRanker step, ``--skip-reviews``) reuses that snapshot when
``--snapshot-dirs`` is omitted, and stops if it is given a different one, so that the reviews and
the AutoRanker scores always cover the same papers.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .io_utils import read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument(
        "--snapshot-dirs",
        nargs="+",
        default=[],
        help="Dataset=snapshot_dir entries (default: the snapshot recorded in <root>/manifests)",
    )
    parser.add_argument("--target-iterations", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--cohorts-per-iteration", type=int, default=5)
    parser.add_argument("--cohort-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--root-output-dir", default="outputs/reviewer_matching")
    parser.add_argument("--skip-autoranker", action="store_true")
    parser.add_argument("--skip-reviews", action="store_true")
    parser.add_argument("--judge", help="AutoRanker judge backend (default: the config's)")
    parser.add_argument("--max-workers", type=int, help="AutoRanker parallelism")
    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def _recorded_snapshot_dirs(manifest_dir: Path, dataset: str) -> list[str] | None:
    """The snapshot dirs that the existing cohort manifests of ``dataset`` were drawn from.

    Args:
        manifest_dir: The ``manifests`` folder of a grid root.
        dataset: Dataset label of the manifests (``<dataset>_iter<k>_cohort<i>.json``).

    Returns:
        The recorded ``snapshot_dirs``, or None when the folder has no manifest for ``dataset``.

    Raises:
        SystemExit: If the manifests were drawn from different snapshot dirs.
    """
    found = {
        tuple(read_json(path)["snapshot_dirs"])
        for path in sorted(manifest_dir.glob(f"{dataset}_iter*_cohort*.json"))
    }
    if len(found) > 1:
        raise SystemExit(
            f"{manifest_dir} holds {dataset} cohorts from different snapshots: {found}"
        )
    return list(found.pop()) if found else None


def _snapshot_dirs_for(root: Path, dataset: str, given: list[str] | None) -> list[str]:
    """The snapshot dirs to draw ``dataset``'s cohorts from: the given ones or the recorded ones."""
    recorded = _recorded_snapshot_dirs(root / "manifests", dataset)
    if not given:
        if recorded is None:
            raise SystemExit(
                f"No --snapshot-dirs for dataset {dataset!r} and no cohort manifests in "
                f"{root / 'manifests'}: run the review step first, or pass --snapshot-dirs."
            )
        return recorded
    if recorded is not None and [Path(p).resolve() for p in given] != [
        Path(p).resolve() for p in recorded
    ]:
        raise SystemExit(
            f"{root / 'manifests'} holds {dataset} cohorts drawn from {recorded}, not {given}; "
            "the reviews and the AutoRanker scores would cover different papers. Omit "
            "--snapshot-dirs to use the recorded snapshot, or use another --root-output-dir."
        )
    return given


def main() -> None:
    args = parse_args()
    root = Path(args.root_output_dir)
    snap_map: dict[str, list[str]] = {}
    for item in args.snapshot_dirs:
        if "=" not in item:
            raise ValueError("--snapshot-dirs entries must be dataset=path")
        dataset, path = item.split("=", 1)
        snap_map.setdefault(dataset, []).append(path)

    for dataset in args.datasets:
        snaps = _snapshot_dirs_for(root, dataset, snap_map.get(dataset))
        _run(
            [
                sys.executable,
                "-m",
                "scripts.analysis.reviewer_matching.prepare_cohorts",
                "--dataset",
                dataset,
                "--snapshot-dirs",
                *snaps,
                "--target-iterations",
                *[str(x) for x in args.target_iterations],
                "--cohorts-per-iteration",
                str(args.cohorts_per_iteration),
                "--cohort-size",
                str(args.cohort_size),
                "--seed",
                str(args.seed),
                "--output-dir",
                str(root / "manifests"),
            ]
        )

    ranker_args = []
    if args.judge:
        ranker_args += ["--judge", args.judge]
    if args.max_workers is not None:
        ranker_args += ["--max-workers", str(args.max_workers)]
    manifests = sorted((root / "manifests").glob("*.json"))
    for manifest in manifests:
        stem = manifest.stem
        if not args.skip_autoranker:
            _run(
                [
                    sys.executable,
                    "-m",
                    "scripts.analysis.reviewer_matching.run_autoranker_for_cohort",
                    "--manifest",
                    str(manifest),
                    "--output-dir",
                    str(root / "autoranker" / stem),
                    *ranker_args,
                ]
            )
        if not args.skip_reviews:
            for condition, strategy in (
                ("condition_a_expertise_matching", "expertise_matching"),
                ("condition_b_random", "random"),
            ):
                _run(
                    [
                        sys.executable,
                        "-m",
                        "scripts.analysis.reviewer_matching.run_review_replay",
                        "--manifest",
                        str(manifest),
                        "--condition",
                        condition,
                        "--reviewer-assignment-strategy",
                        strategy,
                        "--output-dir",
                        str(root / "reviews" / f"{stem}_{condition}"),
                    ]
                )

    # The analysis joins the reviews with the AutoRanker scores, so it needs the AutoRanker step;
    # without it (03's run.sh) it would only write empty tables.
    if not args.skip_autoranker:
        _run(
            [
                sys.executable,
                "-m",
                "scripts.analysis.reviewer_matching.analyze_results",
                "--root",
                str(root),
            ]
        )


if __name__ == "__main__":
    main()
