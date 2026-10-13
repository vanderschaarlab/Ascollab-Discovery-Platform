"""Reviewer-norm injection into the reviewer prompt (experiment 09)."""

from agents.review.rev_agent_utils import make_reviewer_system_prompt
from agents.review.reviewer_agent import get_reviewer_agent
from system.schema import AgentNetworkMetadata, AgentProfile

_NORM = "ZZ_TEST_NORM_STRING_ZZ"


def _profile():
    return AgentProfile(
        agent_id="a1",
        behavior="b",
        expertise="genomics",
        expertise_topics=["genomics"],
        citation_count=0,
        num_accepted_papers=0,
    )


def _meta():
    return AgentNetworkMetadata(num_agents=4)


def test_reviewer_system_prompt_default_has_no_norm_block():
    prompt = make_reviewer_system_prompt(_profile(), _meta())
    assert "Community reviewing norms" not in prompt


def test_reviewer_system_prompt_injects_norm_once():
    prompt = make_reviewer_system_prompt(_profile(), _meta(), community_norms=_NORM)
    assert "Community reviewing norms" in prompt
    assert prompt.count(_NORM) == 1
    # Output schema markers still present (norm is prompt-only).
    assert "Evaluation Criteria" in prompt


def test_reviewer_builder_threads_norm(monkeypatch):
    captured = {}

    import agents.review.reviewer_agent as ra

    real = ra.make_reviewer_system_prompt

    def _spy(*args, **kwargs):
        captured.setdefault("calls", []).append(kwargs.get("community_norms"))
        return real(*args, **kwargs)

    monkeypatch.setattr(ra, "make_reviewer_system_prompt", _spy)

    get_reviewer_agent(
        llm=object(),
        reviewer_agent_profile=_profile(),
        agent_network_metadata=_meta(),
        community_norms=_NORM,
    )

    assert captured["calls"] == [_NORM]
