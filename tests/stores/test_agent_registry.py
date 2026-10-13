import pytest

from stores.agent_registry import AgentRegistry


def test_agent_registry_round_trips_extended_metadata(fake_embeddings, agent_profile_factory):
    registry = AgentRegistry(fake_embeddings)
    profile = agent_profile_factory(
        agent_id="agent_a",
        metric_weights={"support": 0.7},
        reputation={"score": 0.9},
        pause_state={"paused": False},
        learning_state={"lesson": "prefer validation"},
    )

    registry.insert(profile)
    found, loaded = registry.get_agent("agent_a")

    assert found is True
    assert loaded.metric_weights == {"support": 0.7}
    assert loaded.reputation == {"score": 0.9}
    assert loaded.pause_state == {"paused": False}
    assert loaded.learning_state == {"lesson": "prefer validation"}


def test_agent_registry_update_replaces_vector_store_document(
    fake_embeddings, agent_profile_factory
):
    registry = AgentRegistry(fake_embeddings, [agent_profile_factory(agent_id="agent_a")])

    registry.update(
        "agent_a",
        new_expertise="causal inference",
        new_citations=10,
        new_accepted_papers=4,
    )

    found, loaded = registry.get_agent("agent_a")
    assert found is True
    assert loaded.expertise == "causal inference"
    assert loaded.citation_count == 10
    assert loaded.num_accepted_papers == 4
    with pytest.raises(ValueError, match="cannot be less"):
        registry.update("agent_a", new_citations=9)
    with pytest.raises(KeyError):
        registry.update("missing", new_citations=1)


def test_agent_registry_citation_boost_orders_highest_score_first(
    fake_embeddings, agent_profile_factory
):
    registry = AgentRegistry(fake_embeddings, with_citation_boost=True, w_sim=0.0, w_cit=1.0)
    registry.insert(
        [
            agent_profile_factory(agent_id="low", citation_count=1),
            agent_profile_factory(agent_id="high", citation_count=100),
        ]
    )

    result = registry.query("genomics", k=1)

    assert result[0][0].agent_id == "high"


def test_agent_registry_empty_and_missing_behaviour(fake_embeddings):
    registry = AgentRegistry(fake_embeddings)

    assert registry.query("anything") == []
    assert registry.get_all_agents() == []
    with pytest.raises(KeyError):
        registry.get_agent("missing")
