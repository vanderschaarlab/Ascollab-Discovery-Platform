from agents.research.primary_agent_utils import make_primary_agent_system_prompt
from system.schema import AgentNetworkMetadata, AgentProfile


def _agent_profile() -> AgentProfile:
    return AgentProfile(
        agent_id="agent_01",
        behavior="careful",
        expertise="omics",
        expertise_topics=["omics"],
        citation_count=0,
        num_accepted_papers=0,
    )


def test_meta_prompt_includes_required_sections_when_enabled():
    prompt = make_primary_agent_system_prompt(
        agent_profile=_agent_profile(),
        agent_network_metadata=AgentNetworkMetadata(num_agents=4),
        iteration=3,
        tools={},
        meta_research_mode=True,
        meta_research_context={
            "source_cards": [
                {
                    "paper_id": "paper_a",
                    "title": "Paper A",
                    "abstract": "A",
                    "manuscript_excerpt": "A",
                    "publication_iteration": 1,
                    "citation_count": 2,
                }
            ],
            "requirements": {
                "require_contradictions_section": True,
                "require_archive_citations": True,
                "min_archive_citations": 3,
            },
            "quality_contract": {
                "require_claim_evidence_table": True,
                "require_support_checks_section": True,
                "require_contradiction_resolution_section": True,
                "require_bounded_final_claims_section": True,
                "require_executable_claim_checks_section": True,
                "require_reproducibility_notes_section": True,
                "min_supported_major_claims": 3,
                "min_code_verified_major_claims": 1,
                "max_words": 950,
            },
        },
    )
    assert "# Cross-Paper Themes" in prompt
    assert "# Claim-Evidence Table" in prompt
    assert "# Support and Validity Checks" in prompt
    assert "# Contradictions in Public Archive" in prompt
    assert "# Contradictions and Resolution" in prompt
    assert "# Coherent Synthesis" in prompt
    assert "# Evidence Map" in prompt
    assert "# Bounded Final Claims" in prompt
    assert "# Executable Claim Checks" in prompt
    assert "# Reproducibility Notes" in prompt
    assert "[paper_a]" in prompt
    assert "Include at least 3 archive citations" in prompt
    assert "Include at least 1 code-verified major claims" in prompt
    assert "at most 950 words" in prompt


def test_prompt_unchanged_for_meta_sections_when_disabled():
    prompt = make_primary_agent_system_prompt(
        agent_profile=_agent_profile(),
        agent_network_metadata=AgentNetworkMetadata(num_agents=4),
        iteration=3,
        tools={},
        meta_research_mode=False,
        meta_research_context=None,
    )
    assert "# Cross-Paper Themes" not in prompt
    assert "# Coherent Synthesis" not in prompt
    assert "# Claim-Evidence Table" not in prompt
    assert "Meta-Research Objective" not in prompt


def test_metacognition_prompt_is_absent_when_context_missing():
    base_prompt = make_primary_agent_system_prompt(
        agent_profile=_agent_profile(),
        agent_network_metadata=AgentNetworkMetadata(num_agents=4),
        iteration=3,
        tools={},
    )
    explicit_none_prompt = make_primary_agent_system_prompt(
        agent_profile=_agent_profile(),
        agent_network_metadata=AgentNetworkMetadata(num_agents=4),
        iteration=3,
        tools={},
        metacognition_feedback_context=None,
    )
    assert base_prompt == explicit_none_prompt
    assert "Metacognitive Feedback From Previous Round" not in base_prompt


def test_metacognition_prompt_includes_feedback_when_context_provided():
    prompt = make_primary_agent_system_prompt(
        agent_profile=_agent_profile(),
        agent_network_metadata=AgentNetworkMetadata(num_agents=4),
        iteration=3,
        tools={},
        self_reflection_prompt_enabled=True,
        metacognition_feedback_context={
            "source_iteration": 2,
            "mode": "metrics_only",
            "summary": "Improve support.",
            "axes": {
                "support": {
                    "strengths": ["ran code"],
                    "weaknesses": ["low evidence"],
                    "actions": ["run validation"],
                }
            },
            "raw_metrics": {"paper_a": {"support": {"successful_code_executions": 1}}},
            "metric_definitions": {
                "support": {"successful_code_executions": "Successful analysis code executions."}
            },
        },
    )
    assert "Metacognitive Feedback From Previous Round" in prompt
    assert "Improve support." in prompt
    assert "Feedback Use:" in prompt
    assert "successful_code_executions" in prompt
