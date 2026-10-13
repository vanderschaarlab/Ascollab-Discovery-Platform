"""Retrieval settings (``memory.*``): config wiring, resume precedence, citation boost on."""

from pathlib import Path

from hydra import compose, initialize_config_dir

from env_setup import langchain_tools
from stores.agent_registry import AgentRegistry
from stores.state import retrieval_settings

_CFG_DIR = str(Path(__file__).resolve().parents[2] / "sys_config")
_TOPICS = [
    "survival analysis of kidney cancer cohorts",
    "proteomics and RPPA pathway activity",
    "single-cell transcriptomics clustering",
    "epidemiology of cardiovascular risk",
]


def _cfg(overrides=None):
    with initialize_config_dir(config_dir=_CFG_DIR, version_base=None):
        return compose(config_name="ascollab", overrides=overrides or [])


def _agents(agent_profile_factory, citations=(0, 0, 0, 0)):
    return [
        agent_profile_factory(agent_id=f"agent_{i + 1:02d}", expertise=topic, citation_count=c)
        for i, (topic, c) in enumerate(zip(_TOPICS, citations, strict=True))
    ]


def test_citation_boost_is_on_by_default():
    archive, registry = retrieval_settings(_cfg().memory)
    assert archive == {
        "with_citation_boost": True,
        "with_recency_boost": False,
        "w_citation": 0.1,
        "w_recency": 0.1,
        "max_k": 5,
    }
    assert registry == {"with_citation_boost": True, "w_sim": 0.8, "w_cit": 0.2, "max_k": 5}


def test_config_overrides_reach_the_settings():
    cfg = _cfg(
        ["memory.max_k=7", "memory.archive.citation_boost=false", "memory.registry.w_cit=0.5"]
    )
    archive, registry = retrieval_settings(cfg.memory)
    assert archive["with_citation_boost"] is False
    assert archive["max_k"] == 7
    assert registry["with_citation_boost"] is True
    assert registry["w_cit"] == 0.5
    assert registry["max_k"] == 7


def test_saved_settings_win_on_resume():
    archive, registry = retrieval_settings(
        _cfg().memory,
        saved_archive={"with_citation_boost": False, "max_k": 9},
        saved_registry={"with_citation_boost": False, "w_sim": 0.6},
    )
    assert archive["with_citation_boost"] is False
    assert archive["max_k"] == 9
    assert registry["with_citation_boost"] is False
    assert registry["w_sim"] == 0.6
    assert registry["w_cit"] == 0.2  # not saved: the config value stands


def test_boosted_registry_ranks_by_similarity_before_any_citation(
    fake_embeddings, agent_profile_factory
):
    plain = AgentRegistry(fake_embeddings, _agents(agent_profile_factory))
    boosted = AgentRegistry(
        fake_embeddings, _agents(agent_profile_factory), with_citation_boost=True
    )
    expected = [p.agent_id for p, _ in plain.query("kidney cancer survival", k=3)]
    assert [p.agent_id for p, _ in boosted.query("kidney cancer survival", k=3)] == expected


def test_boosted_registry_lifts_cited_agents(fake_embeddings, agent_profile_factory):
    boosted = AgentRegistry(
        fake_embeddings,
        _agents(agent_profile_factory, citations=(0, 0, 0, 50)),
        with_citation_boost=True,
        w_sim=0.0,
        w_cit=1.0,
    )
    # k=2 re-ranks 2k=4 candidates, i.e. every agent here.
    assert boosted.query("kidney cancer survival", k=2)[0][0].agent_id == "agent_04"


def test_reviewer_search_finds_agents_with_boost_on_before_any_citation(
    monkeypatch, fake_embeddings, agent_profile_factory
):
    # Reviewer matching calls this tool and falls back to random reviewers if it returns
    # nothing, so a failure here would silently turn expertise matching into random.
    registry = AgentRegistry(
        fake_embeddings, _agents(agent_profile_factory), with_citation_boost=True
    )
    monkeypatch.setattr(langchain_tools.state, "registry", lambda: registry)
    out = langchain_tools.registry_query_tool("kidney cancer survival", k=2)
    assert out.count('"agent_id"') == 2, out
