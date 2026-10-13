import json
import logging

import pytest
from langchain_core.embeddings import Embeddings

from env_setup.langchain_tools import agent_search_tool
from env_setup.tools import make_tools
from stores import state
from stores.agent_registry import AgentRegistry
from system.reviewer_matching import (
    _assignment_rng,
    _parse_agent_ids,
    assign_papers_to_reviewers,
    match_papers_to_reviewers,
    randomly_assign_papers_to_reviewers,
)
from system.schema import AgentProfile, PaperMetadataPrivate
from system.trackers import _CURRENT_LOG, ReviewStatsLog


class DummyRegistry:
    def __init__(self, ids):
        self._agents = [
            AgentProfile(
                agent_id=aid,
                behavior="",
                expertise=f"expertise {aid}",
                expertise_topics=[],
                citation_count=0,
                num_accepted_papers=0,
            )
            for aid in ids
        ]

    def get_all_agents(self):
        return list(self._agents)


def _paper(pid, author, title=None):
    return PaperMetadataPrivate(
        paper_id=pid,
        primary_agent_id=author,
        collab_agent_ids=[],
        title=title or pid,
        abstract="a",
        manuscript="m",
        citation_count=0,
        publication_t=1,
        cited_paper_ids=[],
    )


def test_random_assignment_is_deterministic_and_excludes_authors():
    registry = DummyRegistry(["a1", "a2", "a3", "a4", "a5"])
    papers = [_paper("p1", "a1"), _paper("p2", "a2")]

    first = randomly_assign_papers_to_reviewers(
        num_reviewers_per_paper=3,
        papers=papers,
        agent_registry=registry,
        seed=7,
        condition_label="b",
    )
    second = randomly_assign_papers_to_reviewers(
        num_reviewers_per_paper=3,
        papers=papers,
        agent_registry=registry,
        seed=7,
        condition_label="b",
    )

    assert first == second
    assert "a1" not in first["p1"]
    assert "a2" not in first["p2"]
    assert all(len(set(rids)) == len(rids) == 3 for rids in first.values())


def _draws(rng, n=6):
    return [rng.random() for _ in range(n)]


def test_reviewer_assignment_seed_is_used_unless_the_legacy_alias_is_set():
    """main.py used to pass matching_seed unconditionally."""
    from system.review_one_step import _resolve_assignment_seed

    assert _resolve_assignment_seed(matching_seed=None, reviewer_assignment_seed=7) == 7
    assert _resolve_assignment_seed(matching_seed=3, reviewer_assignment_seed=7) == 3
    assert _resolve_assignment_seed(matching_seed=0, reviewer_assignment_seed=7) == 0


def test_assignment_rng_keys_on_every_condition_component():
    """Seed, iteration, offset AND condition_label must all reach the key."""
    base = {"paper_id": "p1", "seed": 1, "iteration": 2, "seed_offset": 0, "condition_label": "a"}

    assert _draws(_assignment_rng(**base)) == _draws(_assignment_rng(**base))
    assert _draws(_assignment_rng(**{**base, "condition_label": "b"})) != _draws(
        _assignment_rng(**base)
    )
    assert _draws(_assignment_rng(**{**base, "iteration": 3})) != _draws(_assignment_rng(**base))
    assert _draws(_assignment_rng(**{**base, "seed": 2})) != _draws(_assignment_rng(**base))
    assert _draws(_assignment_rng(**{**base, "seed_offset": 5})) != _draws(_assignment_rng(**base))


def test_condition_label_changes_random_assignment_even_with_an_iteration():
    """The label used to be discarded whenever iteration was supplied."""
    registry = DummyRegistry([f"a{i}" for i in range(1, 9)])
    papers = [_paper("p1", "a1"), _paper("p2", "a2")]

    def assign(label):
        return randomly_assign_papers_to_reviewers(
            num_reviewers_per_paper=3,
            papers=papers,
            agent_registry=registry,
            seed=5,
            iteration=1,
            condition_label=label,
        )

    assert assign("a") == assign("a")
    assert assign("a") != assign("b")


class _RankedSearchTool:
    """Search tool returning a fixed ranked candidate list."""

    def __init__(self, ranked):
        self.ranked = list(ranked)
        self.calls = 0

    def invoke(self, payload):
        self.calls += 1
        return list(self.ranked)


def test_expertise_matching_honours_the_tool_ranking_for_the_first_paper():
    """Shuffling the tool's ranking away made this arm equal to `random`."""
    registry = DummyRegistry([f"a{i:02d}" for i in range(1, 21)])
    ranked = [f"a{i:02d}" for i in range(5, 17)]  # 12 ids, none of them the author
    tool = _RankedSearchTool(ranked)

    assignments = match_papers_to_reviewers(
        agent_search_tool=tool,
        num_reviewers_per_paper=3,
        papers=[_paper("p1", "a01"), _paper("p2", "a02")],
        agent_registry=registry,
        seed=1,
    )

    # Every usage count starts at zero, so the first paper's reviewers must be
    # exactly the tool's top three.
    assert assignments["p1"] == ranked[:3]
    # The second paper still draws from the tool's list, not the wider registry.
    assert set(assignments["p2"]).issubset(set(ranked))


def test_expertise_matching_falls_back_to_the_registry_when_the_tool_is_empty(caplog):
    caplog.set_level(logging.DEBUG)
    registry = DummyRegistry([f"a{i:02d}" for i in range(1, 9)])
    tool = _RankedSearchTool([])

    assignments = match_papers_to_reviewers(
        agent_search_tool=tool,
        num_reviewers_per_paper=3,
        papers=[_paper("p1", "a01")],
        agent_registry=registry,
        seed=1,
    )

    assert len(assignments["p1"]) == 3
    assert "a01" not in assignments["p1"]
    assert "tool=0" in caplog.text


def test_expertise_matching_is_seed_controllable():
    """The expertise arm now varies with the seed like the random arm.

    Ties are what the seed can move: the tool supplies two ranked experts and the
    remaining two slots are drawn from an otherwise-tied fallback pool.
    """
    registry = DummyRegistry([f"a{i:02d}" for i in range(1, 21)])
    ranked = ["a05", "a06"]
    papers = [_paper("p1", "a01"), _paper("p2", "a02")]

    def assign(seed):
        return assign_papers_to_reviewers(
            strategy="expertise_matching",
            agent_search_tool=_RankedSearchTool(ranked),
            num_reviewers_per_paper=4,
            papers=papers,
            agent_registry=registry,
            seed=seed,
        )

    assert assign(1) == assign(1)
    assert assign(1) != assign(2)
    # The tool's experts still lead every assignment, whatever the seed does.
    assert assign(7)["p1"][:2] == ranked


def test_expertise_matching_ranking_is_stable_when_the_tool_is_decisive():
    """A complete unique ranking leaves nothing for the seed to move."""
    registry = DummyRegistry([f"a{i:02d}" for i in range(1, 21)])
    ranked = [f"a{i:02d}" for i in range(5, 17)]
    papers = [_paper("p1", "a01"), _paper("p2", "a02"), _paper("p3", "a03")]

    def assign(seed):
        return match_papers_to_reviewers(
            agent_search_tool=_RankedSearchTool(ranked),
            num_reviewers_per_paper=4,
            papers=papers,
            agent_registry=registry,
            seed=seed,
        )

    assert assign(1) == assign(2)
    assert assign(1)["p1"] == ranked[:4]


def test_assignment_dispatcher_rejects_unknown_strategy():
    registry = DummyRegistry(["a1", "a2"])
    try:
        assign_papers_to_reviewers(
            strategy="mystery",
            agent_search_tool=None,
            num_reviewers_per_paper=1,
            papers=[_paper("p1", "a1")],
            agent_registry=registry,
        )
    except ValueError as exc:
        assert "Unknown reviewer assignment strategy" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


class _TopicEmbeddings(Embeddings):
    """One axis per topic word: texts sharing a topic word embed identically."""

    topics = ("survival", "proteomics")

    def _embed(self, text):
        hits = [1.0 if topic in text.lower() else 0.0 for topic in self.topics]
        norm = sum(hits) ** 0.5 or 1.0
        return [h / norm for h in hits] + [0.0 if any(hits) else 1.0]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


def _topic_registry():
    """a00-a03 work on survival, a04-a07 on proteomics; max_k=5 as in the default config."""
    agents = [
        AgentProfile(
            agent_id=f"a{i:02d}",
            behavior="",
            expertise=f"{'survival analysis' if i < 4 else 'proteomics assays'} (agent {i})",
            expertise_topics=[],
            citation_count=0,
            num_accepted_papers=0,
        )
        for i in range(8)
    ]
    return AgentRegistry(embedder=_TopicEmbeddings(), agents=agents, max_k=5)


def _real_registry_tools():
    """The base tool and the recording wrapper main.py and the replay script pass."""
    return {
        "base": agent_search_tool,
        "make_tools": make_tools({"search_agent_registry": {}})["search_agent_registry"],
    }


@pytest.mark.parametrize("which", ["base", "make_tools"])
def test_expertise_matching_ranks_with_the_real_registry_search_tool(monkeypatch, caplog, which):
    """With the real search tool, reviewers come from the paper's topic, all tool-ranked."""
    caplog.set_level(logging.INFO)
    registry = _topic_registry()
    monkeypatch.setattr(state, "_registry", registry)  # the tool searches state.registry()
    papers = [
        _paper("p1", "a00", "Survival in kidney cohorts"),
        _paper("p2", "a04", "Proteomics of kidney tumours"),
    ]

    assignments = match_papers_to_reviewers(
        agent_search_tool=_real_registry_tools()[which],
        num_reviewers_per_paper=2,
        papers=papers,
        agent_registry=registry,
        seed=1,
    )

    assert set(assignments["p1"]) <= {"a01", "a02", "a03"}
    assert set(assignments["p2"]) <= {"a05", "a06", "a07"}
    assert "tool=4 fallback=0 (tool_share=100%)" in caplog.text


def test_expertise_matching_does_not_record_agent_tool_queries(monkeypatch):
    """Matching searches are not an agent's tool use, even with an agent log current."""
    registry = _topic_registry()
    monkeypatch.setattr(state, "_registry", registry)
    agent_log = ReviewStatsLog()
    token = _CURRENT_LOG.set(agent_log)
    try:
        match_papers_to_reviewers(
            agent_search_tool=_real_registry_tools()["make_tools"],
            num_reviewers_per_paper=2,
            papers=[_paper("p1", "a00", "Survival in kidney cohorts")],
            agent_registry=registry,
            seed=1,
        )
        assert _CURRENT_LOG.get() is agent_log
    finally:
        _CURRENT_LOG.reset(token)

    assert dict(agent_log.tool_inputs) == {}
    assert dict(agent_log.tool_calls) == {}


def test_parse_agent_ids_reads_every_search_result_shape():
    hit = {"agent_id": "a2", "expertise": "x = y", "score": 0.5}
    observation = f"Observation: AGENT_SEARCH_RESULTS = {json.dumps({'results': [hit]})}"

    assert _parse_agent_ids(observation) == ["a2"]
    assert _parse_agent_ids("Observation: AGENT_SEARCH_RESULTS = []") == []
    assert _parse_agent_ids("There are no agents right now") == []
    assert _parse_agent_ids({"results": [{"agent_id": "a1"}, "a3", {"score": 1.0}]}) == ["a1", "a3"]
    assert _parse_agent_ids(["a4", "a5"]) == ["a4", "a5"]
    assert _parse_agent_ids(None) == []
