"""A snapshot records its own iteration, even when the session-stats registry ends earlier."""

import json
from pathlib import Path

from omegaconf import OmegaConf

from system.snapshot_utils import create_snapshot
from system.trackers import ResearchStatsLog, SessionStats, SessionStatsRegistry
from tests.system.test_snapshot_social_memory_pruning import (
    _DummyArchive,
    _DummyPrivateArchive,
    _DummyRegistry,
)


def test_snapshot_metadata_uses_the_snapshot_iteration(tmp_path, monkeypatch):
    monkeypatch.setenv("PROJECT_DIR", str(tmp_path))
    # Round 3 added no session logs (e.g. every agent paused): the registry's last key is 2.
    registry = SessionStatsRegistry()
    registry.add_log(
        [
            SessionStats(
                iteration=2,
                session_id="agent_01_t02",
                session_agent_id="agent_01",
                type="research",
                stats_logs=ResearchStatsLog(),
            )
        ]
    )

    out_dir = create_snapshot(
        iteration=3,
        agent_registry=_DummyRegistry(),
        archive=_DummyArchive(),
        private_archive=_DummyPrivateArchive(),
        network_metadata=None,
        session_stats_registry=registry,
        exp_config=OmegaConf.create({"run_name": "run_iter"}),
        run_name="run_iter",
    )

    assert Path(out_dir).name == "iteration_3"
    meta = json.loads((Path(out_dir) / "snapshot_metadata.json").read_text(encoding="utf-8"))
    assert meta["iteration"] == 3
