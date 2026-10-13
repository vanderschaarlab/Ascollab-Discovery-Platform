from agents.review.rev_agent_utils import (
    make_reviewer_system_prompt,
    review_citation_schema_nudge,
    validate_review_citation_schema,
)
from system.schema import AgentNetworkMetadata, AgentProfile


def test_reviewer_system_prompt_interpolates_metrics_guidance():
    prompt = make_reviewer_system_prompt(
        agent_profile=AgentProfile(
            agent_id="reviewer_1",
            behavior="Careful and concise.",
            expertise="Computational biology",
            expertise_topics=["genomics"],
            citation_count=0,
            num_accepted_papers=0,
        ),
        agent_network_metadata=AgentNetworkMetadata(),
        metrics_allowlist=["support.external_papers_read"],
    )

    assert "{metrics_guidance_text}" not in prompt
    assert "You will ALSO be given a JSON object called METRICS" in prompt
    assert "metrics.support.external_papers_read" in prompt


def test_reviewer_system_prompt_without_metrics_does_not_ask_for_them():
    profile = AgentProfile(
        agent_id="reviewer_1",
        behavior="Careful and concise.",
        expertise="Computational biology",
        expertise_topics=["genomics"],
        citation_count=0,
        num_accepted_papers=0,
    )
    with_metrics = make_reviewer_system_prompt(profile, AgentNetworkMetadata())
    without = make_reviewer_system_prompt(
        profile, AgentNetworkMetadata(), include_metrics_guidance=False
    )

    assert "You MUST use these METRICS" in with_metrics
    for text in ("METRICS", "metrics.support", "what the metrics alone suggest", "metric values"):
        assert text not in without
    assert "justify the Overall Recommendation." in without
    assert "# Quantitative Ratings" in without and "- Overall Recommendation: <number>" in without
    # Everything before the metrics part is the same prompt.
    head = with_metrics[: with_metrics.index("You will ALSO be given")]
    assert without.startswith(head)


def test_validate_review_citation_schema_respects_feature_flag():
    ok, errors = validate_review_citation_schema(
        "Unsupported statement without tags.",
        require_assertion_justification=False,
    )

    assert ok is True
    assert errors == []


def test_validate_review_citation_schema_requires_tags_when_enabled():
    ok, errors = validate_review_citation_schema(
        "Unsupported statement without tags.",
        require_assertion_justification=True,
    )

    assert ok is False
    assert errors
    assert "citation" in review_citation_schema_nudge(errors).lower()
