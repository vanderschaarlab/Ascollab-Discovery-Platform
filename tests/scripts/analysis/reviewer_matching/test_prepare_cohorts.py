from scripts.analysis.reviewer_matching.cohort_utils import is_valid_candidate_paper, select_cohort


def _paper(pid, iteration, manuscript="m"):
    return {
        "paper_id": pid,
        "publication_t": iteration,
        "source_iteration": iteration,
        "manuscript": manuscript,
        "source_snapshot": "run",
        "source_snapshot_dir": "snapshots/run",
        "primary_agent_id": "agent",
        "title": pid,
    }


def test_select_cohort_prefers_target_iteration_then_nearest_fill():
    papers = [_paper(f"p5_{i}", 5) for i in range(2)]
    papers += [_paper(f"p4_{i}", 4) for i in range(2)]
    papers += [_paper("p_timeout", 5, "TIMEOUT")]

    selection = select_cohort(papers, target_iteration=5, cohort_size=3, seed=1)
    iterations = [p["source_iteration"] for p in selection.selected]

    assert iterations.count(5) == 2
    assert iterations.count(4) == 1
    assert any(
        row["paper_id"] == "p_timeout" and row["reason"] == "timeout" for row in selection.excluded
    )


def test_candidate_validity_rejects_empty_manuscript():
    ok, reason = is_valid_candidate_paper(_paper("p", 1, ""))
    assert not ok
    assert reason == "empty_manuscript"
