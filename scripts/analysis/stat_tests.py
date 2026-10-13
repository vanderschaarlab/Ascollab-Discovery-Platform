"""Statistics for the paired AutoRanker comparisons written by ``scripts/experiments/*/rank.sh``.

    python -m scripts.analysis.stat_tests --ranker-dir outputs/ranker/01 --out <output dir>

``rank.sh`` writes each comparison to
``outputs/ranker/<experiment>/<dataset>/<condition_a>_vs_<condition_b>/<rubric>/seed<k>/``
and labels the two sides ``<dataset>_<condition>``. For every (condition pair, rubric) this
reports, per dataset and as a macro average over datasets:

- ``n`` comparisons and the win rate of condition_a;
- per dataset, the exact two-sided binomial test of a 0.5 win rate, and its Holm-adjusted
  p-value over all per-dataset tests in the run;
- per dataset, a Wilson 95% interval, a rough reference only (it treats pairs as independent,
  but each paper appears in many pairs);
- a paper-clustered bootstrap 95% interval, the primary interval (it resamples papers).

Seeds of the same dataset are pooled (each run's papers are separate clusters, although paper
IDs repeat between runs); the macro average weights datasets equally and never pools
comparisons across datasets. Results are written only to ``--out`` (``stat_tests.csv`` and
``stat_tests.json``).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from fractions import Fraction
from pathlib import Path
from typing import Any

from scripts.analysis import aggregate

COLUMNS = [
    "experiment",
    "condition_a",
    "condition_b",
    "rubric",
    "dataset",
    "seeds",
    "n",
    "wins_a",
    "win_rate_a",
    "binom_p",
    "binom_p_holm",
    "wilson_low",
    "wilson_high",
    "bootstrap_low",
    "bootstrap_high",
]


def binom_two_sided_p(wins: int, n: int) -> float | None:
    """Exact two-sided binomial test of a 0.5 success rate (``None`` when n = 0).

    Sums the probabilities of every outcome at least as far from n/2 as the observed one, which
    for p = 0.5 equals the usual "outcomes no more likely than observed" definition.
    """
    if n <= 0:
        return None
    distance = abs(2 * wins - n)
    tail = sum(math.comb(n, i) for i in range(n + 1) if abs(2 * i - n) >= distance)
    return min(1.0, float(Fraction(tail, 2**n)))


def holm(p_values: list[float | None]) -> list[float | None]:
    """Holm step-down adjustment; ``None`` entries are left out of the family."""
    indexed = sorted((p, i) for i, p in enumerate(p_values) if p is not None)
    m = len(indexed)
    adjusted: list[float | None] = [None] * len(p_values)
    running = 0.0
    for rank, (p, i) in enumerate(indexed):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[i] = running
    return adjusted


def find_comparisons(ranker_dir: Path) -> Iterable[dict[str, Any]]:
    """Yield one entry per ``comparisons.json`` laid out the way rank.sh writes them."""
    for path in sorted(ranker_dir.rglob("comparisons.json")):
        parts = path.relative_to(ranker_dir).parts
        if len(parts) < 5 or "_vs_" not in parts[-4] or not parts[-2].startswith("seed"):
            print(f"[stat_tests] skipping {path}: not <dataset>/<a>_vs_<b>/<rubric>/seed<k>/")
            continue
        dataset, pair, rubric, seed_dir = parts[-5], parts[-4], parts[-3], parts[-2]
        condition_a, condition_b = pair.split("_vs_", 1)
        experiment = parts[-6] if len(parts) >= 6 else ranker_dir.name
        records = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(records, dict):
            records = records.get("comparisons", [])
        yield {
            "experiment": experiment,
            "dataset": dataset,
            "condition_a": condition_a,
            "condition_b": condition_b,
            "rubric": rubric,
            "seed": seed_dir.removeprefix("seed"),
            "records": records,
        }


def compute(ranker_dir: Path, *, n_boot: int = 2000, seed: int = 12345) -> list[dict[str, Any]]:
    """Rows of statistics for every (experiment, pair, rubric) found under ``ranker_dir``."""
    groups: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for entry in find_comparisons(ranker_dir):
        key = (entry["experiment"], entry["condition_a"], entry["condition_b"], entry["rubric"])
        ds = groups[key].setdefault(entry["dataset"], {"records": [], "seeds": set()})
        ds["records"].extend(entry["records"])
        ds["seeds"].add(entry["seed"])

    rows: list[dict[str, Any]] = []
    for (experiment, cond_a, cond_b, rubric), by_dataset in sorted(groups.items()):
        labels = {ds: (f"{ds}_{cond_a}", f"{ds}_{cond_b}") for ds in by_dataset}
        macro = aggregate.macro_average(
            {ds: v["records"] for ds, v in by_dataset.items()},
            cond_a,
            cond_b,
            n_boot=n_boot,
            seed=seed,
            labels_by_dataset=labels,
        )
        base = {
            "experiment": experiment,
            "condition_a": cond_a,
            "condition_b": cond_b,
            "rubric": rubric,
        }
        for ds, stats in sorted(macro["per_dataset"].items()):
            rows.append(
                {
                    **base,
                    "dataset": ds,
                    "seeds": " ".join(sorted(by_dataset[ds]["seeds"])),
                    "n": stats["n"],
                    "wins_a": stats["wins"],
                    "win_rate_a": stats["win_rate"],
                    "binom_p": binom_two_sided_p(stats["wins"], stats["n"]),
                    "wilson_low": stats["wilson_low"],
                    "wilson_high": stats["wilson_high"],
                    "bootstrap_low": stats["ci_low"],
                    "bootstrap_high": stats["ci_high"],
                }
            )
        if len(by_dataset) > 1:
            rows.append(
                {
                    **base,
                    "dataset": "macro",
                    "n": macro["n"],
                    "win_rate_a": macro["win_rate"],
                    "bootstrap_low": macro["ci_low"],
                    "bootstrap_high": macro["ci_high"],
                }
            )

    adjusted = holm([row.get("binom_p") for row in rows])
    for row, p in zip(rows, adjusted, strict=True):
        row["binom_p_holm"] = p
    return [{col: row.get(col) for col in COLUMNS} for row in rows]


def write(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "stat_tests.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "stat_tests.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--ranker-dir",
        required=True,
        type=Path,
        help="rank.sh output, e.g. outputs/ranker/01 (or outputs/ranker for all experiments)",
    )
    parser.add_argument("--out", required=True, type=Path, help="Directory for the results")
    parser.add_argument("--n-boot", type=int, default=2000, help="Bootstrap resamples")
    parser.add_argument("--seed", type=int, default=12345, help="Bootstrap seed")
    args = parser.parse_args(argv)

    rows = compute(args.ranker_dir, n_boot=args.n_boot, seed=args.seed)
    if not rows:
        raise SystemExit(f"No rank.sh comparisons found under {args.ranker_dir}")
    write(rows, args.out)
    print(f"[stat_tests] {len(rows)} rows -> {args.out / 'stat_tests.csv'}")


if __name__ == "__main__":
    main()
