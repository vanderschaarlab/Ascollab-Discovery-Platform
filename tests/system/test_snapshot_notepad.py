import json
from pathlib import Path

from omegaconf import OmegaConf

from system.snapshot_utils import create_snapshot
from system.trackers import SessionStatsRegistry


class _DummyRegistry:
    _max_citations_seen = 0
    with_citation_boost = False
    w_sim = 0.8
    w_cit = 0.2
    max_k = 5

    def get_all_agents(self):
        return []


class _DummyArchive:
    with_citation_boost = False
    with_recency_boost = False
    w_citation = 0.1
    w_recency = 0.1
    max_k = 5
    _entries = {}


class _DummyPrivateArchive:
    with_citation_boost = False
    with_recency_boost = False
    w_citation = 0.1
    w_recency = 0.1
    max_k = 5
    _entries = {}
    reviews_by_paper = {}


def test_snapshot_writes_notepad_data_files(tmp_path, monkeypatch):
    monkeypatch.setenv("PROJECT_DIR", str(tmp_path))
    out_dir = create_snapshot(
        iteration=1,
        agent_registry=_DummyRegistry(),
        archive=_DummyArchive(),
        private_archive=_DummyPrivateArchive(),
        network_metadata=None,
        session_stats_registry=SessionStatsRegistry(),
        exp_config=OmegaConf.create({"run_name": "run_notepad"}),
        run_name="run_notepad",
        base_snapshots_dir="snapshots",
    )

    iteration_dir = Path(out_dir)
    assert (iteration_dir / "notepad_data.json").exists()
    assert (iteration_dir / "notepad_data.pkl").exists()
    metadata = json.loads((iteration_dir / "snapshot_metadata.json").read_text(encoding="utf-8"))
    assert "notepad_data.json" in metadata["files_saved"]
    assert "notepad_data.pkl" in metadata["files_saved"]
