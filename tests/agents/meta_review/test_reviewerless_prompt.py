"""Reviewer-free meta-reviewer prompt behaviour (experiment 07)."""

from agents.meta_review.meta_rev_agent_utils import (
    make_metareviewer_system_prompt,
    make_metareviewer_user_message,
    parse_metareview,
)
from system.schema import PaperMetadataPrivate, PaperReview


def _paper(pid: str) -> PaperMetadataPrivate:
    return PaperMetadataPrivate(
        paper_id=pid,
        primary_agent_id="a1",
        collab_agent_ids=[],
        title=f"Title {pid}",
        abstract="abstract",
        manuscript=f"MANUSCRIPT-{pid}",
        citation_count=0,
        publication_t=1,
        cited_paper_ids=[],
    )


def test_system_prompt_default_mentions_reviews():
    prompt = make_metareviewer_system_prompt()
    assert "expert reviews" in prompt
    for token in ("score:", "rank:", "justification:"):
        assert token in prompt
    for criterion in ("Support", "Soundness", "Significance", "Originality"):
        assert criterion in prompt


def test_system_prompt_review_free_omits_reviews_keeps_contract():
    prompt = make_metareviewer_system_prompt(review_free=True)
    assert "expert reviews" not in prompt
    assert "reviews they have received" not in prompt
    # The output contract and criteria are unchanged.
    for token in ("score:", "rank:", "justification:"):
        assert token in prompt
    for criterion in ("Support", "Soundness", "Significance", "Originality"):
        assert criterion in prompt


def test_user_message_review_free_drops_review_sections():
    papers = [_paper("p1")]
    reviews = {
        "p1": [PaperReview(paper_id="p1", reviewer_agent_id="r1", review_text="REVTEXT", score={})]
    }

    default_msg = make_metareviewer_user_message(papers, reviews)
    rf_msg = make_metareviewer_user_message(papers, reviews, review_free=True)

    assert "REVTEXT" in default_msg
    assert "Review 1" in default_msg
    assert "and their reviews" in default_msg

    assert "REVTEXT" not in rf_msg
    assert "Review 1" not in rf_msg
    assert "and their reviews" not in rf_msg
    # Paper content is still present in both.
    assert "MANUSCRIPT-p1" in rf_msg
    assert "Paper ID: p1" in rf_msg


def test_parser_accepts_review_free_style_metareview():
    text = (
        "Paper ID: p1\n1. Brief Summary\n- s\n2. Comparative Analysis\n- c\n"
        "3. Decision\n- score: 0.9\n- rank: 1\n- justification: best\n\n"
        "Paper ID: p2\n1. Brief Summary\n- s\n2. Comparative Analysis\n- c\n"
        "3. Decision\n- score: 0.3\n- rank: 2\n- justification: worse\n"
    )
    parsed, valid = parse_metareview(text, num_papers=2)
    assert valid
    assert set(parsed) == {"p1", "p2"}
