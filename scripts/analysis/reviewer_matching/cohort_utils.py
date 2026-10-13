"""Cohort selection helpers for the reviewer-matching ablation."""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ranker.snapshot_import import collect_papers

INVALID_RESEARCH_STATUSES = {
    "timeout",
    "incomplete",
    "invalid_action_report",
    "code_script_recording_anomaly",
    "insufficient_code_execution_report",
}


@dataclass(frozen=True)
class CohortSelection:
    selected: list[dict[str, Any]]
    excluded: list[dict[str, str]]
    selection_policy: str


def is_valid_candidate_paper(paper: dict[str, Any]) -> tuple[bool, str | None]:
    """Return whether a snapshot paper is eligible for fixed-cohort replay."""
    paper_id = str(paper.get("paper_id", "") or "").strip()
    manuscript = str(paper.get("manuscript", "") or "")
    status = str(paper.get("research_validation_status", "") or "").strip().lower()
    if not paper_id:
        return False, "missing_paper_id"
    if paper_id.endswith("_timeout") or manuscript == "TIMEOUT":
        return False, "timeout"
    if not manuscript.strip():
        return False, "empty_manuscript"
    if status in INVALID_RESEARCH_STATUSES:
        return False, status
    return True, None


def load_candidate_pool(
    snapshot_dirs: Iterable[str | Path],
    *,
    max_iteration: int | str = "all",
) -> list[dict[str, Any]]:
    """Load accepted and rejected non-timeout papers from snapshots."""
    iterations: str | list[int]
    cumulative = False
    if max_iteration == "all":
        iterations = "all"
    else:
        iterations = [int(max_iteration)]
        cumulative = True
    return collect_papers(
        [str(p) for p in snapshot_dirs],
        iterations=iterations,
        cumulative=cumulative,
        include_rejected=True,
        exclude_timeout=True,
    )


def select_cohort(
    papers: list[dict[str, Any]],
    *,
    target_iteration: int,
    cohort_size: int,
    seed: int,
) -> CohortSelection:
    """Select a reproducible target-iteration cohort with nearest-iteration fill."""
    valid: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for paper in papers:
        ok, reason = is_valid_candidate_paper(paper)
        if ok:
            valid.append(paper)
        else:
            excluded.append({"paper_id": str(paper.get("paper_id", "")), "reason": str(reason)})

    by_distance = sorted(
        valid,
        key=lambda p: (
            abs(int(p.get("source_iteration", p.get("publication_t", 0))) - int(target_iteration)),
            0
            if int(p.get("source_iteration", p.get("publication_t", 0))) <= int(target_iteration)
            else 1,
            int(p.get("source_iteration", p.get("publication_t", 0))),
            str(p.get("paper_id", "")),
        ),
    )

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for distance in sorted(
        {
            abs(int(p.get("source_iteration", p.get("publication_t", 0))) - int(target_iteration))
            for p in by_distance
        }
    ):
        bucket = [
            p
            for p in by_distance
            if abs(
                int(p.get("source_iteration", p.get("publication_t", 0))) - int(target_iteration)
            )
            == distance
        ]
        rng.shuffle(bucket)
        for paper in bucket:
            pid = str(paper.get("paper_id", ""))
            if pid and pid not in seen:
                selected.append(paper)
                seen.add(pid)
            if len(selected) >= cohort_size:
                break
        if len(selected) >= cohort_size:
            break

    if len(selected) < cohort_size:
        raise ValueError(
            f"Only {len(selected)} valid papers available for target_iteration={target_iteration}; "
            f"requested cohort_size={cohort_size}."
        )

    return CohortSelection(
        selected=selected[:cohort_size],
        excluded=excluded,
        selection_policy="exact_iteration_then_nearest_fill",
    )


def build_manifest(
    *,
    dataset: str,
    target_iteration: int,
    cohort_index: int,
    cohort_seed: int,
    cohort_size: int,
    snapshot_dirs: list[str],
    selection: CohortSelection,
) -> dict[str, Any]:
    """Build the persisted cohort manifest."""
    candidates = []
    for paper in selection.selected:
        candidates.append(
            {
                "paper_id": paper["paper_id"],
                "source_snapshot": paper.get("source_snapshot"),
                "source_snapshot_dir": paper.get("source_snapshot_dir"),
                "source_iteration": paper.get("source_iteration", paper.get("publication_t")),
                "primary_agent_id": paper.get("primary_agent_id"),
                "title": paper.get("title", ""),
            }
        )
    return {
        "schema_version": 1,
        "dataset": dataset,
        "target_iteration": int(target_iteration),
        "cohort_index": int(cohort_index),
        "cohort_seed": int(cohort_seed),
        "cohort_size": int(cohort_size),
        "snapshot_dirs": [str(p) for p in snapshot_dirs],
        "candidate_paper_ids": [c["paper_id"] for c in candidates],
        "candidate_papers": candidates,
        "selection_policy": selection.selection_policy,
        "excluded_paper_ids": selection.excluded,
    }
