"""Analysis helpers for the reviewer-matching ablation."""

from __future__ import annotations

import math
from statistics import median
from typing import Any


def dcg(relevances: list[float]) -> float:
    return sum((rel / math.log2(idx + 2)) for idx, rel in enumerate(relevances))


def ndcg_at_k(accepted_ids: list[str], score_by_paper: dict[str, float], k: int) -> float:
    """Compute NDCG@k over accepted ids using AutoRanker scores as relevance."""
    if k <= 0:
        return 0.0
    observed = [float(score_by_paper.get(pid, 0.0)) for pid in accepted_ids[:k]]
    ideal = sorted((float(v) for v in score_by_paper.values()), reverse=True)[:k]
    denom = dcg(ideal)
    return dcg(observed) / denom if denom else 0.0


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Compute Spearman rank correlation without scipy."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None

    def ranks(vals: list[float]) -> list[float]:
        ordered = sorted(enumerate(vals), key=lambda item: item[1])
        out = [0.0] * len(vals)
        i = 0
        while i < len(ordered):
            j = i
            while j + 1 < len(ordered) and ordered[j + 1][1] == ordered[i][1]:
                j += 1
            avg = (i + j + 2) / 2.0
            for idx in range(i, j + 1):
                out[ordered[idx][0]] = avg
            i = j + 1
        return out

    rx = ranks(xs)
    ry = ranks(ys)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=False))
    den_x = math.sqrt(sum((a - mx) ** 2 for a in rx))
    den_y = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (den_x * den_y) if den_x and den_y else None


def summarize_condition(
    *,
    manifest: dict[str, Any],
    condition_summary: dict[str, Any],
    metareviews: list[dict[str, Any]],
    accepted_paper_ids: list[str],
    autoranker_scores: dict[str, Any],
    condition_dir: str,
) -> dict[str, Any]:
    """Build one per-condition analysis row."""
    score_rows = autoranker_scores.get("paper_scores", [])
    score_by_paper = {
        row["paper_id"]: float(row.get("autoranker_score", 0.0)) for row in score_rows
    }
    accepted_scores = [score_by_paper[pid] for pid in accepted_paper_ids if pid in score_by_paper]
    top5 = {row["paper_id"] for row in score_rows if int(row.get("autoranker_rank", 999999)) <= 5}
    top10 = {row["paper_id"] for row in score_rows if int(row.get("autoranker_rank", 999999)) <= 10}
    mr_by_pid = {
        row["paper_id"]: float(row.get("overall_score", 0.0) or 0.0) for row in metareviews
    }
    shared_ids = sorted(set(mr_by_pid) & set(score_by_paper))
    rho = spearman(
        [mr_by_pid[pid] for pid in shared_ids], [score_by_paper[pid] for pid in shared_ids]
    )
    return {
        "dataset": manifest.get("dataset"),
        "target_iteration": manifest.get("target_iteration"),
        "cohort_index": manifest.get("cohort_index"),
        "cohort_seed": manifest.get("cohort_seed"),
        "condition": condition_summary.get("condition"),
        "reviewer_assignment_strategy": condition_summary.get("reviewer_assignment_strategy"),
        "condition_dir": condition_dir,
        "accepted_k": len(accepted_paper_ids),
        "mean_accepted_autoranker_score": sum(accepted_scores) / len(accepted_scores)
        if accepted_scores
        else None,
        "median_accepted_autoranker_score": median(accepted_scores) if accepted_scores else None,
        "best_accepted_autoranker_score": max(accepted_scores) if accepted_scores else None,
        "fraction_accepted_in_autoranker_top5": (
            sum(1 for pid in accepted_paper_ids if pid in top5) / len(accepted_paper_ids)
            if accepted_paper_ids
            else None
        ),
        "fraction_accepted_in_autoranker_top10": (
            sum(1 for pid in accepted_paper_ids if pid in top10) / len(accepted_paper_ids)
            if accepted_paper_ids
            else None
        ),
        "ndcg_at_k": ndcg_at_k(accepted_paper_ids, score_by_paper, len(accepted_paper_ids)),
        "spearman_metareview_autoranker": rho,
        "mean_candidate_autoranker_score": (
            sum(score_by_paper.values()) / len(score_by_paper) if score_by_paper else None
        ),
        "candidate_autoranker_score_variance": (
            sum(
                (v - (sum(score_by_paper.values()) / len(score_by_paper))) ** 2
                for v in score_by_paper.values()
            )
            / len(score_by_paper)
            if score_by_paper
            else None
        ),
    }


def pair_condition_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair expertise and random rows within cohort and compute deltas."""
    by_key: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (row["dataset"], row["target_iteration"], row["cohort_index"], row["cohort_seed"])
        by_key.setdefault(key, {})[str(row["reviewer_assignment_strategy"])] = row

    deltas = []
    for key, grouped in sorted(by_key.items()):
        expertise = (
            grouped.get("expertise_matching") or grouped.get("matching") or grouped.get("expertise")
        )
        random_row = (
            grouped.get("random") or grouped.get("random_assignment") or grouped.get("no_matching")
        )
        if not expertise or not random_row:
            continue
        e = expertise.get("mean_accepted_autoranker_score")
        r = random_row.get("mean_accepted_autoranker_score")
        if e is None or r is None:
            continue
        deltas.append(
            {
                "dataset": key[0],
                "target_iteration": key[1],
                "cohort_index": key[2],
                "cohort_seed": key[3],
                "delta_mean_accepted_score": e - r,
                "expertise_mean_accepted_score": e,
                "random_mean_accepted_score": r,
            }
        )
    return deltas
