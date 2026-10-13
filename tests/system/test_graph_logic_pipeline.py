from agents.research.primary_agent_utils import make_primary_agent_system_prompt
from system.research_one_step import _collect_plot_artifacts_from_log
from system.schema import AgentNetworkMetadata, AgentProfile


class _DummyLogs:
    def __init__(self, timeline):
        self.timeline = timeline


def _agent_profile() -> AgentProfile:
    return AgentProfile(
        agent_id="agent_1",
        behavior="Careful and empirical",
        expertise="Bioinformatics",
        expertise_topics=["gene expression"],
        citation_count=0,
        num_accepted_papers=0,
    )


def test_collect_plot_artifacts_from_log_extracts_and_deduplicates():
    logs = _DummyLogs(
        timeline=[
            {
                "kind": "tool_result_execute_code",
                "plot_artifacts": [
                    {"plot_filename": "a.png", "plot_path": "plots/a.png"},
                    {"plot_filename": "a.png", "plot_path": "plots/a.png"},
                    {"plot_filename": "b.png", "plot_path": "plots/b.png"},
                ],
            },
            {
                "kind": "tool_result_execute_code",
                "plot_artifacts": [
                    {"plot_filename": "b.png", "plot_path": "plots/b.png"},
                    {"plot_filename": "c.png", "plot_path": "plots/c.png"},
                ],
            },
        ]
    )
    rows = _collect_plot_artifacts_from_log(logs)
    assert len(rows) == 3
    assert rows[0]["plot_filename"] == "a.png"
    assert rows[1]["plot_filename"] == "b.png"
    assert rows[2]["plot_filename"] == "c.png"


def test_primary_prompt_includes_plot_citation_guidance_when_graph_logic_enabled():
    prompt = make_primary_agent_system_prompt(
        agent_profile=_agent_profile(),
        agent_network_metadata=AgentNetworkMetadata(num_agents=3),
        iteration=2,
        tools={},
        graph_logic_enabled=True,
        require_plot_citations_in_research=True,
    )
    assert "PLOT_ARTEFACTS" in prompt
    assert "PLOT_CITATIONS" in prompt
    assert "[Plot: ...]" in prompt
