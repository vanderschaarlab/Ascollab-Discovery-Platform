from agents.meta_review.meta_rev_agent_utils import (
    make_metareviewer_system_prompt,
    make_metareviewer_user_message,
    parse_metareview,
)
from system.schema import PaperReview


def _valid_text(rank_a: int = 1, rank_b: int = 2, score_a: float = 0.9) -> str:
    return f"""
Paper ID: paper_a
1. Brief Summary
- Strong.
2. Comparative Analysis
- Better supported.
3. Decision
- score: {score_a}
- rank: {rank_a}
- justification: Best support.

Paper ID: paper_b
1. Brief Summary
- Weaker.
2. Comparative Analysis
- Less support.
3. Decision
- score: 0.3
- rank: {rank_b}
- justification: Less complete.
"""


def test_metareviewer_prompt_and_user_message_shape(private_paper_factory):
    prompt = make_metareviewer_system_prompt()
    user = make_metareviewer_user_message(
        [private_paper_factory(paper_id="paper_a")],
        {
            "paper_a": [
                PaperReview(
                    paper_id="paper_a",
                    reviewer_agent_id="reviewer_01",
                    review_text="Detailed review.",
                    score={"Overall Recommendation": 0.8},
                )
            ]
        },
    )

    assert "score: <float between 0 and 1>" in prompt
    assert "Paper ID: paper_a" in user
    assert "Detailed review." in user


def test_parse_metareview_accepts_valid_complete_output():
    parsed, valid = parse_metareview(_valid_text(), num_papers=2)

    assert valid is True
    assert parsed["paper_a"]["score"] == 0.9
    assert parsed["paper_b"]["rank"] == 2


def test_parse_metareview_rejects_duplicate_or_missing_rank():
    _, duplicate_valid = parse_metareview(_valid_text(rank_a=1, rank_b=1), num_papers=2)
    missing_text = _valid_text().replace("- rank: 2", "")
    parsed, missing_valid = parse_metareview(missing_text, num_papers=2)

    assert duplicate_valid is False
    assert missing_valid is False
    assert "paper_b" not in parsed


def test_parse_metareview_rejects_out_of_range_score():
    _, valid = parse_metareview(_valid_text(score_a=1.2), num_papers=2)

    assert valid is False
