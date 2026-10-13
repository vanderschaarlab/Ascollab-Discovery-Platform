import asyncio
from types import SimpleNamespace

import pytest

from agents.review.rev_agent_utils import make_reviewer_user_message, parse_review_text
from system import review_one_step as ros
from system.main import _normalize_metrics_pipeline_mode, _reviewer_metrics_allowlist
from system.schema import AgentProfile
from tests.e2e.fakes import REVIEW_TEXT


def test_metrics_pipeline_mode_accepts_default_and_experimental():
    assert _normalize_metrics_pipeline_mode("default") == ("default", False)
    assert _normalize_metrics_pipeline_mode("experimental") == ("experimental", False)


def test_metrics_pipeline_mode_accepts_none_aliases():
    assert _normalize_metrics_pipeline_mode("none") == ("none", False)
    assert _normalize_metrics_pipeline_mode("off") == ("none", False)
    assert _normalize_metrics_pipeline_mode("disabled") == ("none", False)


def test_metrics_pipeline_mode_is_backward_compatible_for_unknown_values():
    mode, fallback = _normalize_metrics_pipeline_mode("legacy_custom_value")
    assert mode == "default"
    assert fallback is True


def test_reviewer_user_message_skips_metrics_block_when_metrics_none():
    msg = make_reviewer_user_message(report_text="Report body.", metrics=None)
    assert "precomputed quantitative METRICS" not in msg


def test_reviewer_user_message_includes_metrics_block_when_metrics_present():
    msg = make_reviewer_user_message(
        report_text="Report body.",
        metrics={"support": {"lines_of_code": 10}},
    )
    assert "precomputed quantitative METRICS" in msg


def test_reviewer_metrics_allowlist_applies_only_in_experimental_mode():
    allowlist = ("support.lines_of_code", "soundness.internal_consistency_score")
    assert _reviewer_metrics_allowlist("experimental", allowlist) == list(allowlist)
    assert _reviewer_metrics_allowlist("experimental", None) is None
    assert _reviewer_metrics_allowlist("default", allowlist) is None
    assert _reviewer_metrics_allowlist("none", allowlist) is None


class _Registry:
    def get_agent(self, agent_id):
        return True, AgentProfile(
            agent_id=agent_id,
            behavior="",
            expertise="genomics",
            expertise_topics=[],
            citation_count=0,
            num_accepted_papers=0,
        )

    def get_all_agents(self):
        return [self.get_agent("reviewer_01")[1]]


def _reviewer_agent_kwargs(monkeypatch, paper, paper_metrics_by_id, **step_kwargs):
    """Run review_one_step for one paper and one reviewer; return get_reviewer_agent's kwargs."""
    from stores.private_paper_archive import PrivatePaperArchive

    seen = []

    def fake_get_reviewer_agent(*args, **kwargs):
        seen.append(kwargs)
        return object()

    async def fake_run_one_reviewer(agent, state, reviewer_id, paper_id, iteration, **kwargs):
        return {"route": "final", "parsed_review": parse_review_text(REVIEW_TEXT)[0]}

    monkeypatch.setattr(ros, "get_reviewer_agent", fake_get_reviewer_agent)
    monkeypatch.setattr(ros, "get_logs", lambda _agent: None)
    monkeypatch.setattr(ros, "run_one_reviewer", fake_run_one_reviewer)
    asyncio.run(
        ros.review_one_step(
            iteration=1,
            llm=None,
            llm_for=None,
            candidate_papers=[paper],
            network_metadata=None,
            private_archive=PrivatePaperArchive(),
            agent_registry=_Registry(),
            agent_registry_search_tool=None,
            task_config=SimpleNamespace(collab_agent_tools={}),
            num_reviewers_per_paper=1,
            review_max_iterations=1,
            reviewer_assignment_strategy="random",
            paper_metrics_by_id=paper_metrics_by_id,
            **step_kwargs,
        )
    )
    return seen


@pytest.mark.parametrize(
    ("paper_metrics_by_id", "step_kwargs", "guidance"),
    [
        # system.main with metrics_pipeline=none: no metrics, guidance off.
        (None, {"include_metrics_guidance": False}, False),
        # The review replay (experiment 03) passes no metrics and keeps the guidance, as in the
        # paper's runs.
        (None, {}, True),
        # A paper without metrics under the default pipeline keeps the guidance, as before.
        ({}, {"include_metrics_guidance": True}, True),
        (
            {"paper_01": {"support": {"lines_of_code": 10}}},
            {"include_metrics_guidance": True},
            True,
        ),
    ],
    ids=["pipeline-off", "review-replay", "paper-without-metrics", "paper-with-metrics"],
)
def test_reviewers_get_metrics_guidance_unless_metrics_are_off(
    monkeypatch, private_paper_factory, paper_metrics_by_id, step_kwargs, guidance
):
    paper = private_paper_factory(paper_id="paper_01", primary_agent_id="author_01")
    (kwargs,) = _reviewer_agent_kwargs(monkeypatch, paper, paper_metrics_by_id, **step_kwargs)
    assert kwargs["include_metrics_guidance"] is guidance
