from scripts.analysis.reviewer_matching.autoranker_utils import aggregate_win_rate_scores


def test_aggregate_autoranker_scores_respects_ab_position():
    comparisons = [
        {"status": "completed", "paper_a_id": "p1", "paper_b_id": "p2", "outcome": "A"},
        {"status": "completed", "paper_a_id": "p1", "paper_b_id": "p3", "outcome": "B"},
        {"status": "failed", "paper_a_id": "p2", "paper_b_id": "p3", "outcome": "A"},
    ]

    result = aggregate_win_rate_scores(comparisons, paper_ids=["p1", "p2", "p3"])
    by_id = {row["paper_id"]: row for row in result["paper_scores"]}

    assert by_id["p1"]["wins"] == 1
    assert by_id["p1"]["losses"] == 1
    assert by_id["p2"]["losses"] == 1
    assert by_id["p3"]["wins"] == 1
    assert by_id["p3"]["autoranker_rank"] == 1
