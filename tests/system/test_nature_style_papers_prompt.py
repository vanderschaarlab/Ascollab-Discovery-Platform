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


def test_nature_style_prompt_includes_sections_template_and_guardrails_when_enabled():
    prompt = make_primary_agent_system_prompt(
        agent_profile=_agent_profile(),
        agent_network_metadata=AgentNetworkMetadata(num_agents=4),
        iteration=3,
        tools={},
        nature_style_papers_prompt_enabled=True,
    )
    assert "# Summary" in prompt
    assert "# Results and Analysis" in prompt
    assert "# Methods" in prompt
    assert "# Data and Code Availability" in prompt
    assert "# Formal Methods" in prompt
    assert "Example Template (structure only; fill with run-specific evidence)" in prompt
    assert "<dataset_key_1>" in prompt
    assert "[Archive: <paper_id>]" in prompt
    assert "Never output placeholder tokens" in prompt
    assert "I replaced placeholders, removed template artifacts" in prompt
    assert "target 500-900 words" in prompt


def test_nature_style_prompt_is_unchanged_when_disabled():
    prompt = make_primary_agent_system_prompt(
        agent_profile=_agent_profile(),
        agent_network_metadata=AgentNetworkMetadata(num_agents=4),
        iteration=3,
        tools={},
        nature_style_papers_prompt_enabled=False,
    )
    assert "# Summary" not in prompt
    assert "# Results and Analysis" not in prompt
    assert "# Data and Code Availability" not in prompt
    assert "Example Template (structure only; fill with run-specific evidence)" not in prompt


def test_nature_style_prompt_not_injected_for_meta_research_mode():
    prompt = make_primary_agent_system_prompt(
        agent_profile=_agent_profile(),
        agent_network_metadata=AgentNetworkMetadata(num_agents=4),
        iteration=3,
        tools={},
        meta_research_mode=True,
        nature_style_papers_prompt_enabled=True,
    )
    assert "Example Template (structure only; fill with run-specific evidence)" not in prompt
    assert "# Cross-Paper Themes" in prompt
    assert "# Summary" not in prompt


def test_nature_style_forces_formal_methods():
    prompt = make_primary_agent_system_prompt(
        agent_profile=_agent_profile(),
        agent_network_metadata=AgentNetworkMetadata(num_agents=4),
        iteration=3,
        tools={},
        nature_style_papers_prompt_enabled=True,
    )
    assert "# Formal Methods" in prompt
    assert "`# Formal Methods` is mandatory in nature-style mode." in prompt
