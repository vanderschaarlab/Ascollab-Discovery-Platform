from scripts.analysis.reviewer_matching.analysis_utils import (
    ndcg_at_k,
    pair_condition_deltas,
    summarize_condition,
)


def test_ndcg_at_k_is_one_for_ideal_order():
    scores = {"p1": 1.0, "p2": 0.5, "p3": 0.1}
    assert ndcg_at_k(["p1", "p2"], scores, 2) == 1.0


def test_summarize_condition_and_pair_deltas():
    manifest = {"dataset": "d", "target_iteration": 5, "cohort_index": 0, "cohort_seed": 1}
    autoranker = {
        "paper_scores": [
            {"paper_id": "p1", "autoranker_score": 1.0, "autoranker_rank": 1},
            {"paper_id": "p2", "autoranker_score": 0.5, "autoranker_rank": 2},
        ]
    }
    metareviews = [
        {"paper_id": "p1", "overall_score": 0.8},
        {"paper_id": "p2", "overall_score": 0.4},
    ]
    expertise = summarize_condition(
        manifest=manifest,
        condition_summary={"condition": "a", "reviewer_assignment_strategy": "expertise_matching"},
        metareviews=metareviews,
        accepted_paper_ids=["p1"],
        autoranker_scores=autoranker,
        condition_dir="a",
    )
    random = summarize_condition(
        manifest=manifest,
        condition_summary={"condition": "b", "reviewer_assignment_strategy": "random"},
        metareviews=metareviews,
        accepted_paper_ids=["p2"],
        autoranker_scores=autoranker,
        condition_dir="b",
    )

    deltas = pair_condition_deltas([expertise, random])
    assert deltas[0]["delta_mean_accepted_score"] == 0.5
