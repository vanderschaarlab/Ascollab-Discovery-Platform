"""Regression: weakest and strongest dimensions must never overlap."""

import logging

from stores.private_paper_archive import PrivatePaperArchive
from system.update_utils import _compute_agent_reputation

SCORE_KEYS = ["Overall Recommendation", "Support", "Soundness", "Significance", "Originality"]


def _archive_with_scores(private_paper_factory, scores):
    archive = PrivatePaperArchive([private_paper_factory(paper_id="p1", status="accepted")])
    archive.add_reviews_for_iteration(
        1, {"p1": [{"reviewer_id": "rev_01", "review_text": "ok", "scores": scores}]}
    )
    return archive


def _diagnostics(private_paper_factory, scores):
    reputations = _compute_agent_reputation(
        _archive_with_scores(private_paper_factory, scores),
        score_keys=SCORE_KEYS,
        primary_score_key="Overall Recommendation",
        iteration=1,
    )
    return reputations["agent_01"]["diagnostics"]


def test_three_scored_dimensions_do_not_overlap(private_paper_factory):
    diag = _diagnostics(
        private_paper_factory,
        {"Overall Recommendation": 3, "Support": 1, "Soundness": 2, "Significance": 4},
    )

    assert diag["weakest_dimensions"] == ["Support"]
    assert diag["strongest_dimensions"] == ["Significance"]
    assert not set(diag["weakest_dimensions"]) & set(diag["strongest_dimensions"])


def test_four_scored_dimensions_keep_two_at_each_end(private_paper_factory):
    diag = _diagnostics(
        private_paper_factory,
        {
            "Overall Recommendation": 3,
            "Support": 1,
            "Soundness": 2,
            "Significance": 3,
            "Originality": 4,
        },
    )

    assert diag["weakest_dimensions"] == ["Support", "Soundness"]
    assert diag["strongest_dimensions"] == ["Originality", "Significance"]
    assert not set(diag["weakest_dimensions"]) & set(diag["strongest_dimensions"])


def test_single_scored_dimension_yields_no_split(private_paper_factory, caplog):
    caplog.set_level(logging.DEBUG)
    diag = _diagnostics(private_paper_factory, {"Overall Recommendation": 3, "Support": 2})

    assert diag["weakest_dimensions"] == []
    assert diag["strongest_dimensions"] == []
    assert "policy learning has no weakest/strongest split" in caplog.text


def test_primary_score_key_is_excluded_from_the_ranking(private_paper_factory):
    """The aggregate is not a weighting dimension, so it must not take a slot."""
    diag = _diagnostics(
        private_paper_factory,
        {
            "Overall Recommendation": 1,  # lowest of all, but not a sub-dimension
            "Support": 2,
            "Soundness": 3,
            "Significance": 4,
            "Originality": 5,
        },
    )

    assert "Overall Recommendation" not in diag["weakest_dimensions"]
    assert diag["weakest_dimensions"] == ["Support", "Soundness"]
