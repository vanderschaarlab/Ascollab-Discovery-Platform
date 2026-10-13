from agents.review.rev_agent_utils import parse_review_text


def test_parse_review_text_accepts_decimal_scores() -> None:
    review = """
# Review Text
1. Summary
Strong work.
2. Motivation
Clear.
3. Claims and Evidence
Reasonable.
4. Soundness of Methodology
Mostly sound.
5. Relation to Prior Knowledge
Adequate.
6. Novelty and Significance
Moderate novelty.
7. Other Comments
None.

# Quantitative Ratings
- Support: 3.5
- Soundness: 3
- Significance: 2.5
- Originality: 3.5
- Overall Recommendation: 3.5
""".strip()

    parsed, ok = parse_review_text(review)
    assert ok
    assert parsed["Support"] == 3.5
    assert parsed["Soundness"] == 3.0
    assert parsed["Significance"] == 2.5
    assert parsed["Originality"] == 3.5
    assert parsed["Overall Recommendation"] == 3.5
