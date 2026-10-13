"""Experiment 08: create_snapshot writes the social_memory_pruning.json artefact."""

import json
from pathlib import Path

from omegaconf import OmegaConf

from system.snapshot_utils import create_snapshot
from system.trackers import SessionStatsRegistry


class _DummyRegistry:
    _max_citations_seen = 0
    with_citation_boost = False
    max_k = 5

    def get_all_agents(self):
        return []


class _DummyArchive:
    with_citation_boost = False
    with_recency_boost = False
    w_citation = 0.1
    w_recency = 0.1
    max_k = 5
    _entries: dict = {}


class _DummyPrivateArchive(_DummyArchive):
    reviews_by_paper: dict = {}


def test_snapshot_writes_social_memory_pruning_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("PROJECT_DIR", str(tmp_path))
    out_dir = create_snapshot(
        iteration=3,
        agent_registry=_DummyRegistry(),
        archive=_DummyArchive(),
        private_archive=_DummyPrivateArchive(),
        network_metadata=None,
        session_stats_registry=SessionStatsRegistry(),
        exp_config=OmegaConf.create({"run_name": "run_markovian"}),
        run_name="run_markovian",
        base_snapshots_dir="snapshots",
        social_memory_pruning_metadata={
            "mode": "markovian",
            "iteration": 3,
            "retained_publication_t": 2,
            "retained_count": 4,
            "dropped_paper_ids": ["p1", "p2"],
            "dropped_count": 2,
        },
    )
    path = Path(out_dir) / "social_memory_pruning.json"
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["mode"] == "markovian"
    assert payload["retained_publication_t"] == 2
    assert payload["dropped_paper_ids"] == ["p1", "p2"]
    assert payload["dropped_count"] == 2
