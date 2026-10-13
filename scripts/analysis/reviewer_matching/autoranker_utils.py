"""AutoRanker scoring helpers for fixed candidate cohorts."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def comparison_winner(record: dict[str, Any]) -> str | None:
    """Return the winning original paper_id for one completed ranker comparison."""
    if record.get("status") != "completed":
        return None
    outcome = record.get("outcome")
    if outcome not in {"A", "B"}:
        score = record.get("score")
        if score == 1:
            outcome = "A"
        elif score == 0:
            outcome = "B"
        else:
            return None
    key = "paper_a_id" if outcome == "A" else "paper_b_id"
    winner = record.get(key)
    return str(winner) if winner else None


def aggregate_win_rate_scores(
    comparisons: list[dict[str, Any]],
    *,
    paper_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Aggregate pairwise AutoRanker comparisons into deterministic win-rate scores."""
    ids = {str(pid) for pid in (paper_ids or []) if pid}
    wins: dict[str, float] = defaultdict(float)
    losses: dict[str, float] = defaultdict(float)

    for record in comparisons:
        if record.get("status") != "completed":
            continue
        a = record.get("paper_a_id")
        b = record.get("paper_b_id")
        if not a or not b:
            continue
        a = str(a)
        b = str(b)
        ids.update([a, b])
        winner = comparison_winner(record)
        if winner == a:
            wins[a] += 1.0
            losses[b] += 1.0
        elif winner == b:
            wins[b] += 1.0
            losses[a] += 1.0
        elif record.get("outcome") == "tie":
            wins[a] += 0.5
            wins[b] += 0.5
            losses[a] += 0.5
            losses[b] += 0.5

    rows = []
    for pid in sorted(ids):
        w = float(wins.get(pid, 0.0))
        loss_count = float(losses.get(pid, 0.0))
        n = w + loss_count
        score = w / n if n else 0.0
        rows.append(
            {
                "paper_id": pid,
                "wins": w,
                "losses": loss_count,
                "comparisons": int(n),
                "win_rate": score,
                "autoranker_score": score,
            }
        )

    rows.sort(key=lambda row: (-row["autoranker_score"], row["paper_id"]))
    for rank, row in enumerate(rows, start=1):
        row["autoranker_rank"] = rank

    return {
        "scoring_method": "win_rate",
        "paper_scores": rows,
    }
