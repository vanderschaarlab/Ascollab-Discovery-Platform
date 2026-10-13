"""The step limit of research and review sessions (``system.graph_limits``).

A session that never finishes gets exactly M actions; a report or review must come within
M - 1 steps. Scripted models, no sandbox.
"""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool
from langgraph.errors import GraphRecursionError

from agents.research.research_team import get_research_team
from agents.review.reviewer_agent import get_reviewer_agent
from env_setup.config_factory import ToolInfo
from system.graph_limits import recursion_limit_for
from system.schema import AgentNetworkMetadata, AgentProfile

ACTION = 'Action: {"tool": "lookup", "args": {"query": "kidney cancer"}}'
REVIEW = """
# Review Text
1. Summary
Clear work.
2. Motivation
Clear.
3. Claims and Evidence
Reasonable.
4. Soundness of Methodology
Sound.
5. Relation to Prior Knowledge
Adequate.
6. Novelty and Significance
Moderate.
7. Other Comments
None.

# Quantitative Ratings
- Support: 3
- Soundness: 3
- Significance: 3
- Originality: 3
- Overall Recommendation: 3
""".strip()


class _ScriptedLLM:
    """Returns the scripted replies in order, then repeats the last one."""

    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.calls = 0

    def invoke(self, prompt_val):
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return AIMessage(content=reply)


def _profile() -> AgentProfile:
    return AgentProfile(
        agent_id="agent_01",
        behavior="",
        expertise="genomics",
        expertise_topics=[],
        citation_count=0,
        num_accepted_papers=0,
    )


def _lookup_tool(calls: list[str]) -> dict[str, ToolInfo]:
    def lookup(query: str) -> str:
        calls.append(query)
        return "no results"

    tool = StructuredTool.from_function(func=lookup, name="lookup", description="Look it up.")
    return {
        "lookup": ToolInfo(object=tool, tool_signature="lookup(query)", code="", code_signature="")
    }


def _run_research(llm, max_steps: int, calls: list[str]):
    agent, _ = get_research_team(
        llm=llm,
        sandbox_client=None,
        iteration=1,
        primary_agent_profile=_profile(),
        agent_network_metadata=AgentNetworkMetadata(num_agents=3),
        sandbox_sticky=SimpleNamespace(session_id="s1"),
        primary_agent_tools=_lookup_tool(calls),
    )
    state = {
        "messages": [HumanMessage(content="Start.")],
        "script": "",
        "step": 1,
        "last_ai_response": "",
        "collab_agent_ids": [],
        "metadata": {"agent_id": "agent_01", "iteration": 1},
    }
    return agent.invoke(state, {"recursion_limit": recursion_limit_for(max_steps)})


def _run_review(llm, max_steps: int, calls: list[str]):
    agent = get_reviewer_agent(
        llm,
        reviewer_agent_profile=_profile(),
        agent_network_metadata=AgentNetworkMetadata(num_agents=3),
        reviewer_tools=_lookup_tool(calls),
    )
    state = {
        "messages": [HumanMessage(content="Review the paper.")],
        "step": 1,
        "last_ai_response": "",
        "parsed_review": None,
        "valid_review": None,
        "metadata": {"reviewer_id": "agent_01", "paper_id": "p1", "iteration": 1},
    }
    return agent.invoke(state, {"recursion_limit": recursion_limit_for(max_steps)})


def test_the_paper_settings_give_the_paper_runs_limits():
    # research_max_iters 40 and review_max_iterations 6 (sys_config/ascollab.yaml) gave the
    # recursion limits of the paper's runs, 80 and 12.
    assert recursion_limit_for(40) == 80
    assert recursion_limit_for(6) == 12


@pytest.mark.parametrize("run", [_run_research, _run_review], ids=["research", "review"])
@pytest.mark.parametrize("max_steps", [1, 3, 6])
def test_a_session_that_never_finishes_gets_exactly_its_steps(run, max_steps):
    calls: list[str] = []
    llm = _ScriptedLLM(ACTION)

    with pytest.raises(GraphRecursionError):
        run(llm, max_steps, calls)

    assert len(calls) == max_steps
    assert llm.calls == max_steps


@pytest.mark.parametrize("max_steps", [2, 6])
def test_a_session_must_finish_before_its_last_step(max_steps):
    # LangGraph stops a run that reaches the limit even when its last node would have ended it.
    in_time = _ScriptedLLM(*([ACTION] * (max_steps - 2)), REVIEW)
    assert _run_review(in_time, max_steps, [])["valid_review"] is True

    too_late = _ScriptedLLM(*([ACTION] * (max_steps - 1)), REVIEW)
    with pytest.raises(GraphRecursionError):
        _run_review(too_late, max_steps, [])
