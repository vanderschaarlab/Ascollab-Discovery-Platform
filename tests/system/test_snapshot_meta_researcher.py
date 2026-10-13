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


def test_snapshot_writes_meta_researcher_artifact_and_summary(tmp_path, monkeypatch):
    monkeypatch.setenv("PROJECT_DIR", str(tmp_path))
    out_dir = create_snapshot(
        iteration=3,
        agent_registry=_DummyRegistry(),
        archive=_DummyArchive(),
        private_archive=_DummyPrivateArchive(),
        network_metadata=None,
        session_stats_registry=SessionStatsRegistry(),
        exp_config=OmegaConf.create({"run_name": "run_meta_snapshot"}),
        run_name="run_meta_snapshot",
        base_snapshots_dir="snapshots",
        meta_researcher_metadata={
            "enabled": True,
            "write_iteration_file": True,
            "write_meta_claim_checks": True,
            "write_citation_lint": True,
            "write_ranker_pools": True,
            "trigger": {"triggered": True},
            "num_source_papers": 7,
            "num_meta_papers": 2,
            "num_meta_papers_generated": 3,
            "num_meta_papers_filtered_out": 1,
            "meta_quality_summary": {"passed": 2, "failed": 1, "revised_passed": 1},
            "meta_claim_checks_artifacts": [
                {"paper_id": "p_meta_1", "summary": {"code_verified_major_claim_count": 1}}
            ],
            "citation_lint_artifacts": [{"paper_id": "p_meta_1", "lint_status": "pass"}],
            "skipped_reason": None,
        },
    )

    iteration_dir = Path(out_dir)
    assert (iteration_dir / "meta_researcher.json").exists()
    assert (iteration_dir / "meta_claim_checks.json").exists()
    assert (iteration_dir / "citation_lint.json").exists()
    assert (iteration_dir / "ranker_pools" / "manifest.json").exists()
    assert (iteration_dir / "ranker_pools" / "papers_all.jsonl").exists()
    assert (iteration_dir / "ranker_pools" / "papers_standard.jsonl").exists()
    assert (iteration_dir / "ranker_pools" / "papers_meta.jsonl").exists()

    metadata = json.loads((iteration_dir / "snapshot_metadata.json").read_text(encoding="utf-8"))
    assert "meta_researcher.json" in metadata["files_saved"]
    assert "meta_claim_checks.json" in metadata["files_saved"]
    assert "citation_lint.json" in metadata["files_saved"]
    assert "ranker_pools/manifest.json" in metadata["files_saved"]
    assert metadata["meta_researcher"]["triggered"] is True
    assert metadata["meta_researcher"]["num_source_papers"] == 7
    assert metadata["meta_researcher"]["num_meta_papers"] == 2
    assert metadata["meta_researcher"]["num_meta_papers_generated"] == 3
    assert metadata["meta_researcher"]["num_meta_papers_filtered_out"] == 1
    assert metadata["meta_researcher"]["meta_quality_summary"]["passed"] == 2
