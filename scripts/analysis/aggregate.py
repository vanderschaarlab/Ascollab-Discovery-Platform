"""Win-rate aggregation for paired AutoRanker `comparisons.json` outputs (used by stat_tests).

Responsibilities (pure, dependency-light, unit-testable):
  * Drop records whose ``status != "completed"``.
  * Resolve wins through the recorded ``paper_a_experiment`` / ``paper_b_experiment``
    labels so the A/B position randomisation is correctly inverted.
  * Report ``n`` and win rate.
  * Aggregate per dataset, then macro-average across datasets (never pool raw
    comparisons across datasets as one sample). Report the per-seed breakdown.
  * Primary uncertainty: a cluster bootstrap that resamples *papers* (not pairs),
    because each paper appears in many pairs so comparisons are not independent. A paper is
    identified by its source run and ID, so papers from different seeds stay separate.
  * A Wilson interval is offered only as a clearly-labelled rough reference.
"""

from __future__ import annotations

import math
import random
import re
from typing import Any

_SEED_RE = re.compile(r"seed[_-]?(\d+)", re.IGNORECASE)


def _slot_of_condition(record: dict[str, Any], a_label: str, b_label: str) -> str | None:
    """Return 'A' or 'B' for the slot holding ``a_label`` (condition_a), else None.

    Matches the experiment labels recorded on each side. Falls back to substring
    matching so composite labels (e.g. ``"<run>_<condition>"``) still resolve when
    the plain condition label is unambiguously contained in exactly one side.
    """
    pae = str(record.get("paper_a_experiment") or "")
    pbe = str(record.get("paper_b_experiment") or "")

    def _matches(text: str, label: str) -> bool:
        return text == label or (bool(label) and label in text)

    a_in_A = _matches(pae, a_label)
    a_in_B = _matches(pbe, a_label)
    b_in_A = _matches(pae, b_label)
    b_in_B = _matches(pbe, b_label)

    if a_in_A and b_in_B and not (a_in_B or b_in_A):
        return "A"
    if a_in_B and b_in_A and not (a_in_A or b_in_B):
        return "B"
    # Exact-match tie-break when substring logic is ambiguous.
    if pae == a_label and pbe == b_label:
        return "A"
    if pae == b_label and pbe == a_label:
        return "B"
    return None


def resolve_condition_a_win(record: dict[str, Any], a_label: str, b_label: str) -> bool | None:
    """True if condition_a won this comparison, False if condition_b won, else None.

    ``None`` when the record is not completed or the A/B slots cannot be resolved.
    Uses ``score`` (1 = the A slot won, 0 = the B slot won) inverted through the
    recorded slot membership.
    """
    if str(record.get("status")) != "completed":
        return None
    score = record.get("score")
    if score not in (0, 1):
        # Fall back to the outcome label if score is missing/odd.
        outcome = record.get("outcome")
        if outcome == "A":
            score = 1
        elif outcome == "B":
            score = 0
        else:
            return None
    slot = _slot_of_condition(record, a_label, b_label)
    if slot is None:
        return None
    a_won_slot = "A" if score == 1 else "B"
    return slot == a_won_slot


def _condition_a_paper_id(record: dict[str, Any], a_label: str, b_label: str) -> str | None:
    """Cluster key of condition_a's paper: ``<source run>::<paper_id>``.

    Paper IDs (``paper_tNN_agent_XX``) repeat in every run, so the key includes the run the
    paper came from; otherwise pooling seeds would merge unrelated papers into one cluster.
    """
    slot = _slot_of_condition(record, a_label, b_label)
    if slot not in ("A", "B"):
        return None
    side = "paper_a" if slot == "A" else "paper_b"
    paper_id = str(record.get(f"{side}_id") or "")
    if not paper_id:
        return None
    run = str(
        record.get(f"{side}_source_snapshot_dir") or record.get(f"{side}_source_snapshot") or ""
    )
    return f"{run}::{paper_id}" if run else paper_id


def _seed_of(record: dict[str, Any], a_label: str, b_label: str) -> str | None:
    slot = _slot_of_condition(record, a_label, b_label)
    key = "paper_a_source_snapshot" if slot == "A" else "paper_b_source_snapshot"
    text = str(record.get(key) or record.get(key.replace("snapshot", "snapshot_dir")) or "")
    m = _SEED_RE.search(text)
    return m.group(1) if m else None


def point_win_rate(records: list[dict[str, Any]], a_label: str, b_label: str) -> dict[str, Any]:
    """Pooled win rate of condition_a over the completed, resolvable records."""
    wins = 0
    n = 0
    for rec in records:
        w = resolve_condition_a_win(rec, a_label, b_label)
        if w is None:
            continue
        n += 1
        if w:
            wins += 1
    return {"n": n, "wins": wins, "win_rate": (wins / n) if n else None}


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — a ROUGH reference only (ignores paper clustering)."""
    if n == 0:
        return (0.0, 1.0)
    phat = wins / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def cluster_bootstrap_ci(
    records: list[dict[str, Any]],
    a_label: str,
    b_label: str,
    *,
    n_boot: int = 1000,
    seed: int = 12345,
    alpha: float = 0.05,
) -> tuple[float | None, float | None]:
    """Cluster bootstrap CI for condition_a's win rate, resampling *papers*.

    Papers (condition_a's paper in each comparison) are the clusters: every
    comparison sharing a paper is correlated, so we resample papers with replacement
    and pool their comparisons. Returns a percentile interval. Deterministic under
    ``seed``. Returns (None, None) when there is no resolvable data.
    """
    by_paper: dict[str, list[bool]] = {}
    for rec in records:
        w = resolve_condition_a_win(rec, a_label, b_label)
        if w is None:
            continue
        pid = _condition_a_paper_id(rec, a_label, b_label) or "__unknown__"
        by_paper.setdefault(pid, []).append(bool(w))
    paper_ids = list(by_paper.keys())
    if not paper_ids:
        return (None, None)

    rng = random.Random(seed)
    stats: list[float] = []
    n_papers = len(paper_ids)
    for _ in range(n_boot):
        wins = 0
        total = 0
        for _ in range(n_papers):
            pid = paper_ids[rng.randrange(n_papers)]
            outcomes = by_paper[pid]
            wins += sum(1 for o in outcomes if o)
            total += len(outcomes)
        if total:
            stats.append(wins / total)
    if not stats:
        return (None, None)
    stats.sort()
    lo_idx = int((alpha / 2) * (len(stats) - 1))
    hi_idx = int((1 - alpha / 2) * (len(stats) - 1))
    return (stats[lo_idx], stats[hi_idx])


def aggregate_dataset(
    records: list[dict[str, Any]],
    a_label: str,
    b_label: str,
    *,
    n_boot: int = 1000,
    seed: int = 12345,
) -> dict[str, Any]:
    """Per-dataset aggregation: point win rate, cluster CI, Wilson reference, per-seed."""
    pt = point_win_rate(records, a_label, b_label)
    ci_low, ci_high = cluster_bootstrap_ci(records, a_label, b_label, n_boot=n_boot, seed=seed)
    wilson_low, wilson_high = wilson_interval(pt["wins"], pt["n"]) if pt["n"] else (None, None)

    per_seed: dict[str, dict[str, Any]] = {}
    for rec in records:
        w = resolve_condition_a_win(rec, a_label, b_label)
        if w is None:
            continue
        s = _seed_of(rec, a_label, b_label) or "unknown"
        bucket = per_seed.setdefault(s, {"n": 0, "wins": 0})
        bucket["n"] += 1
        if w:
            bucket["wins"] += 1
    for bucket in per_seed.values():
        bucket["win_rate"] = bucket["wins"] / bucket["n"] if bucket["n"] else None

    return {
        "n": pt["n"],
        "wins": pt["wins"],
        "win_rate": pt["win_rate"],
        "ci_low": ci_low,
        "ci_high": ci_high,
        "wilson_low": wilson_low,
        "wilson_high": wilson_high,
        "per_seed": per_seed,
    }


def macro_average(
    records_by_dataset: dict[str, list[dict[str, Any]]],
    a_label: str,
    b_label: str,
    *,
    n_boot: int = 1000,
    seed: int = 12345,
    labels_by_dataset: dict[str, tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Macro-average condition_a's win rate across datasets (equal weight per dataset).

    Never pools raw comparisons across datasets. The macro CI is a nested cluster
    bootstrap: within each iteration, resample each dataset's papers, compute that
    dataset's win rate, then average across datasets. ``labels_by_dataset`` gives
    each dataset its own exact (condition_a, condition_b) labels, e.g. the
    ``<dataset>_<condition>`` labels rank.sh records.
    """

    def _labels(ds: str) -> tuple[str, str]:
        return (labels_by_dataset or {}).get(ds, (a_label, b_label))

    per_dataset: dict[str, dict[str, Any]] = {}
    for ds, recs in records_by_dataset.items():
        per_dataset[ds] = aggregate_dataset(recs, *_labels(ds), n_boot=n_boot, seed=seed)

    dataset_rates = [v["win_rate"] for v in per_dataset.values() if v["win_rate"] is not None]
    macro_win_rate = sum(dataset_rates) / len(dataset_rates) if dataset_rates else None
    total_n = sum(v["n"] for v in per_dataset.values())

    # Nested cluster bootstrap for the macro CI.
    by_paper_by_ds: dict[str, dict[str, list[bool]]] = {}
    for ds, recs in records_by_dataset.items():
        bucket = by_paper_by_ds.setdefault(ds, {})
        ds_a, ds_b = _labels(ds)
        for rec in recs:
            w = resolve_condition_a_win(rec, ds_a, ds_b)
            if w is None:
                continue
            pid = _condition_a_paper_id(rec, ds_a, ds_b) or "__unknown__"
            bucket.setdefault(pid, []).append(bool(w))

    rng = random.Random(seed + 7)
    macro_stats: list[float] = []
    for _ in range(n_boot):
        ds_rates: list[float] = []
        for _ds, by_paper in by_paper_by_ds.items():
            paper_ids = list(by_paper.keys())
            if not paper_ids:
                continue
            wins = 0
            total = 0
            for _ in range(len(paper_ids)):
                outcomes = by_paper[paper_ids[rng.randrange(len(paper_ids))]]
                wins += sum(1 for o in outcomes if o)
                total += len(outcomes)
            if total:
                ds_rates.append(wins / total)
        if ds_rates:
            macro_stats.append(sum(ds_rates) / len(ds_rates))
    if macro_stats:
        macro_stats.sort()
        ci_low = macro_stats[int(0.025 * (len(macro_stats) - 1))]
        ci_high = macro_stats[int(0.975 * (len(macro_stats) - 1))]
    else:
        ci_low = ci_high = None

    return {
        "condition_a": a_label,
        "condition_b": b_label,
        "n": total_n,
        "win_rate": macro_win_rate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "per_dataset": per_dataset,
    }
