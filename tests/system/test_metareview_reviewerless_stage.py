"""Reviewerless meta-review stage + startup validation (experiment 07)."""

import asyncio

import pytest
from langchain_core.messages import AIMessage
from omegaconf import OmegaConf

from system.main import validate_reviewerless_config
from system.metareview_one_step import metareview_one_step
from system.schema import PaperMetadataPrivate


class _FakeLLM:
    """Returns a fixed, valid two-paper metareview regardless of input."""

    def __init__(self, text: str) -> None:
        self._text = text

    def invoke(self, messages, **_):
        return AIMessage(content=self._text)


def _paper(pid: str) -> PaperMetadataPrivate:
    return PaperMetadataPrivate(
        paper_id=pid,
        primary_agent_id="a1",
        collab_agent_ids=[],
        title=f"T{pid}",
        abstract="ab",
        manuscript="MAN",
        citation_count=0,
        publication_t=1,
        cited_paper_ids=[],
    )


_VALID_METAREVIEW = (
    "Paper ID: p1\n1. Brief Summary\n- s\n2. Comparative Analysis\n- c\n"
    "3. Decision\n- score: 0.8\n- rank: 1\n- justification: best\n\n"
    "Paper ID: p2\n1. Brief Summary\n- s\n2. Comparative Analysis\n- c\n"
    "3. Decision\n- score: 0.4\n- rank: 2\n- justification: worse\n"
)


def test_reviewerless_stage_scores_all_candidates_without_reviews():
    papers = [_paper("p1"), _paper("p2")]
    reviews, _logs = asyncio.run(
        metareview_one_step(
            iteration=1,
            llm=_FakeLLM(_VALID_METAREVIEW),
            llm_for=None,
            candidate_papers=papers,
            reviews_by_paper={},
            tournament_size=4,
            require_reviews=False,
            review_free=True,
            shuffle_before_tournament=False,
        )
    )
    assert {r.paper_id for r in reviews} == {"p1", "p2"}
    # Accept quota max(1, total // tournament_size) == 1 accepted.
    assert sum(1 for r in reviews if r.decision == "accept") == 1


def test_require_reviews_true_drops_papers_without_reviews():
    papers = [_paper("p1"), _paper("p2")]
    reviews, _logs = asyncio.run(
        metareview_one_step(
            iteration=1,
            llm=_FakeLLM(_VALID_METAREVIEW),
            llm_for=None,
            candidate_papers=papers,
            reviews_by_paper={},
            tournament_size=4,
            require_reviews=True,
        )
    )
    assert reviews == []


def _baseline_reviewerless_cfg():
    return OmegaConf.create(
        {
            "metareview": {"reviewerless": {"enabled": True}},
            "review": {
                "num_reviewers_per_paper": 4,
                "norm_prompt": {"enabled": False},
            },
            "research": {"metacognition_feedback": {"enabled": False}},
            "reputation": {"enabled": False},
            "reputation_pause": {"mode": "none"},
        }
    )


def test_validate_reviewerless_passes_for_baseline():
    validate_reviewerless_config(_baseline_reviewerless_cfg())  # must not raise


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c.reputation.__setitem__("enabled", True),
        lambda c: c.research.metacognition_feedback.__setitem__("enabled", True),
        lambda c: c.reputation_pause.__setitem__("mode", "no_review"),
    ],
)
def test_validate_reviewerless_rejects_incompatible_settings(mutate):
    cfg = _baseline_reviewerless_cfg()
    mutate(cfg)
    with pytest.raises(ValueError):
        validate_reviewerless_config(cfg)


def test_validate_reviewerless_noop_when_disabled():
    cfg = _baseline_reviewerless_cfg()
    cfg.metareview.reviewerless.enabled = False
    cfg.reputation.enabled = True  # would be an offender if reviewerless were on
    validate_reviewerless_config(cfg)  # must not raise


def test_reviewerless_warns_about_reviewer_count_only_when_overridden(caplog):
    cfg = _baseline_reviewerless_cfg()
    with caplog.at_level("WARNING", logger="system.main"):
        validate_reviewerless_config(cfg, ["metareview.reviewerless.enabled=true"])
    assert "inert" not in caplog.text  # the config default alone is not a user choice

    with caplog.at_level("WARNING", logger="system.main"):
        validate_reviewerless_config(cfg, ["review.num_reviewers_per_paper=3"])
    assert "review.num_reviewers_per_paper" in caplog.text


def test_reviewerless_warns_about_an_enabled_norm(caplog):
    cfg = _baseline_reviewerless_cfg()
    cfg.review.norm_prompt.enabled = True
    with caplog.at_level("WARNING", logger="system.main"):
        validate_reviewerless_config(cfg)
    assert "review.norm_prompt" in caplog.text
