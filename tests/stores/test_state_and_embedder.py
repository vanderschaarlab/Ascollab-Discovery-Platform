import numpy as np
import pytest

import stores.state as state
from stores.embedder import Embedder


def _reset_state(monkeypatch):
    monkeypatch.setattr(state, "_archive", None)
    monkeypatch.setattr(state, "_private_archive", None)
    monkeypatch.setattr(state, "_registry", None)
    monkeypatch.setattr(state, "_network_metadata", None)
    monkeypatch.setattr(state, "_notepad_store", None)
    monkeypatch.setattr(state, "_seq", 0)


def test_embedder_normalizes_document_and_query_vectors(fake_embeddings):
    embedder = Embedder(fake_embeddings)

    docs = embedder.embed_items(["alpha", "beta"])
    query = embedder.embed_query("alpha")

    assert docs.shape == (2, fake_embeddings.dim)
    assert np.allclose(np.linalg.norm(docs, axis=1), 1.0)
    assert np.isclose(np.linalg.norm(query), 1.0)


def test_state_requires_initialization(monkeypatch):
    _reset_state(monkeypatch)

    with pytest.raises(AssertionError, match="init_state"):
        state.archive()


def test_state_initializes_accessors_and_updates_network_metadata(
    monkeypatch, fake_embeddings, agent_profile_factory, paper_factory
):
    _reset_state(monkeypatch)
    state.init_state(
        embedder=fake_embeddings, notepad_enabled=True, notepad_base_snapshots_dir="tmp"
    )

    state.registry().insert(
        [agent_profile_factory(agent_id="a1"), agent_profile_factory(agent_id="a2")]
    )
    state.archive().add([paper_factory(paper_id="p1"), paper_factory(paper_id="p2")])

    assert state.bump_seq() == 1
    assert state.current_seq() == 1
    metadata = state.update_network_metadata()
    assert metadata.num_agents == 2
    assert metadata.num_papers == 2
