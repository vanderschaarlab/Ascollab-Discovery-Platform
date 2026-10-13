import json
from pathlib import Path

from omegaconf import OmegaConf

from env_setup.sandbox_utils import persist_sandbox_images_to_snapshot
from system.snapshot_utils import create_snapshot
from system.trackers import SessionStatsRegistry

PNG_1X1_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jf9sAAAAASUVORK5CYII="
)


def test_persist_sandbox_images_to_snapshot_writes_plot_and_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("PROJECT_DIR", str(tmp_path))
    resp = {"status": "success", "images": [{"type": "png", "data": PNG_1X1_BASE64}]}

    summary = persist_sandbox_images_to_snapshot(
        resp=resp,
        run_name="run_a",
        iteration=3,
        step=5,
        agent_id="agent_1",
        session_id="sid_123",
        job_id="job_456",
        cell_no=7,
        base_snapshots_dir="snapshots",
    )

    assert summary["image_count"] == 1
    assert summary["plot_paths"] == [
        "plots/agent_1_sid-sid_123_job-job_456_cell-7_step-5_img-01.png"
    ]

    iteration_dir = tmp_path / "snapshots" / "run_a" / "iteration_3"
    plot_path = iteration_dir / summary["plot_paths"][0]
    manifest_path = iteration_dir / "plots" / "manifest.jsonl"
    assert plot_path.exists()
    assert manifest_path.exists()

    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    row = rows[0]
    assert row["agent_id"] == "agent_1"
    assert row["iteration"] == 3
    assert row["step"] == 5
    assert row["session_id"] == "sid_123"
    assert row["job_id"] == "job_456"
    assert row["cell_no"] == 7
    assert row["filename"] == plot_path.name
    assert row["path"] == summary["plot_paths"][0]
    assert row["bytes"] > 0
    assert len(row["sha256"]) == 64


def test_persist_sandbox_images_to_snapshot_no_images_creates_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("PROJECT_DIR", str(tmp_path))
    summary = persist_sandbox_images_to_snapshot(
        resp={"status": "success", "images": []},
        run_name="run_b",
        iteration=1,
        step=1,
        agent_id="agent_2",
        session_id=None,
        job_id=None,
        cell_no=None,
        base_snapshots_dir="snapshots",
    )

    assert summary["image_count"] == 0
    assert summary["plot_paths"] == []
    assert not (tmp_path / "snapshots" / "run_b" / "iteration_1").exists()


def test_persist_sandbox_images_to_snapshot_applies_limits_and_warnings(tmp_path, monkeypatch):
    monkeypatch.setenv("PROJECT_DIR", str(tmp_path))
    resp = {
        "status": "success",
        "images": [
            {"type": "png", "data": "not-base64"},
            {"type": "png", "data": PNG_1X1_BASE64},
        ],
    }
    summary = persist_sandbox_images_to_snapshot(
        resp=resp,
        run_name="run_limits",
        iteration=1,
        step=1,
        agent_id="agent_3",
        session_id="sid",
        job_id="job",
        cell_no=1,
        base_snapshots_dir="snapshots",
        max_images_per_execution=1,
    )

    assert summary["image_count"] == 0
    assert summary["skipped_count"] == 2
    assert any("invalid_base64" in warning for warning in summary["warnings"])
    assert any(warning == "max_images_limit:1:dropped_1" for warning in summary["warnings"])


def test_persist_sandbox_images_to_snapshot_skips_failed_execution(tmp_path, monkeypatch):
    monkeypatch.setenv("PROJECT_DIR", str(tmp_path))
    resp = {
        "status": "error",
        "images": [{"type": "png", "data": PNG_1X1_BASE64}],
    }
    summary = persist_sandbox_images_to_snapshot(
        resp=resp,
        run_name="run_error",
        iteration=1,
        step=1,
        agent_id="agent_4",
        session_id="sid",
        job_id="job",
        cell_no=1,
        base_snapshots_dir="snapshots",
    )

    assert summary["image_count"] == 0
    assert summary["plot_paths"] == []
    assert summary["skipped_count"] == 0
    assert "non_success_status:error" in summary["warnings"]
    assert not (tmp_path / "snapshots" / "run_error" / "iteration_1").exists()


def test_persist_sandbox_images_to_snapshot_writes_graph_logic_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("PROJECT_DIR", str(tmp_path))
    resp = {
        "status": "success",
        "stdout": "group_a mean=1.2 group_b mean=0.7 p=0.03",
        "result": "Completed plot generation",
        "images": [{"type": "png", "data": PNG_1X1_BASE64}],
    }
    summary = persist_sandbox_images_to_snapshot(
        resp=resp,
        run_name="run_graph",
        iteration=4,
        step=9,
        agent_id="agent_9",
        session_id="sid_9",
        job_id="job_9",
        cell_no=2,
        base_snapshots_dir="snapshots",
        graph_logic_cfg={"enabled": True, "summary": {"backend": "code_stats_llm"}},
        code=(
            "import matplotlib.pyplot as plt\n"
            "plt.plot([1,2,3],[1.2,1.1,1.3],label='group_a')\n"
            "plt.plot([1,2,3],[0.7,0.8,0.6],label='group_b')\n"
            "plt.xlabel('time')\n"
            "plt.ylabel('score')\n"
            "plt.title('Group Comparison')\n"
        ),
    )

    assert summary["image_count"] == 1
    assert len(summary["plot_filenames"]) == 1
    assert len(summary["plot_handles"]) == 1
    assert len(summary["plot_citations"]) == 1
    assert len(summary["plot_artifacts"]) == 1
    assert summary["plot_summary_backend"] == "code_stats_llm"
    assert isinstance(summary["plot_summary_stats_completeness"], float)

    iteration_dir = tmp_path / "snapshots" / "run_graph" / "iteration_4"
    artifact = summary["plot_artifacts"][0]
    summary_path = iteration_dir / artifact["summary_path"]
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["plot_filename"] == summary["plot_filenames"][0]
    assert payload["generation_backend"] == "code_stats_llm"
    assert "statistics" in payload
    assert isinstance(payload["statistics"].get("series", []), list)
    assert "stats_completeness" in payload["statistics"]
    assert artifact["plot_handle"] == summary["plot_handles"][0]
    assert artifact["plot_citation"] == summary["plot_citations"][0]
    assert artifact["plot_citation"].startswith("[Plot: ")

    summary_manifest = iteration_dir / "plots" / "summary_manifest.jsonl"
    assert summary_manifest.exists()
    rows = [
        json.loads(line)
        for line in summary_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    assert rows[0]["plot_filename"] == summary["plot_filenames"][0]
    assert rows[0]["plot_handle"] == summary["plot_handles"][0]
    assert rows[0]["plot_citation"] == summary["plot_citations"][0]
    assert rows[0]["summary_path"] == artifact["summary_path"]
    assert rows[0]["generation_backend"] == "code_stats_llm"


def test_summary_artifact_contains_descriptive_and_comparative_stats(tmp_path, monkeypatch):
    monkeypatch.setenv("PROJECT_DIR", str(tmp_path))
    resp = {
        "status": "success",
        "stdout": "group_a: 1.1 1.2 1.3 group_b: 0.7 0.8 0.9 p=0.02",
        "images": [{"type": "png", "data": PNG_1X1_BASE64}],
    }
    summary = persist_sandbox_images_to_snapshot(
        resp=resp,
        run_name="run_graph_stats",
        iteration=2,
        step=4,
        agent_id="agent_stats",
        session_id="sid",
        job_id="job",
        cell_no=1,
        base_snapshots_dir="snapshots",
        graph_logic_cfg={"enabled": True, "summary": {"backend": "code_stats_llm"}},
        code=(
            "import matplotlib.pyplot as plt\n"
            "plt.plot([1,2,3],[1.1,1.2,1.3],label='group_a')\n"
            "plt.plot([1,2,3],[0.7,0.8,0.9],label='group_b')\n"
            "plt.title('Comparison')\n"
            "plt.xlabel('x')\n"
            "plt.ylabel('y')\n"
        ),
    )
    iteration_dir = tmp_path / "snapshots" / "run_graph_stats" / "iteration_2"
    artifact = summary["plot_artifacts"][0]
    payload = json.loads((iteration_dir / artifact["summary_path"]).read_text(encoding="utf-8"))
    stats = payload.get("statistics", {})
    assert len(stats.get("series", [])) >= 1
    assert "mean" in stats["series"][0]
    assert len(stats.get("comparisons", [])) >= 1
    assert "delta_mean" in stats["comparisons"][0]


def test_graph_logic_disabled_preserves_legacy_plot_behavior(tmp_path, monkeypatch):
    monkeypatch.setenv("PROJECT_DIR", str(tmp_path))
    resp = {"status": "success", "images": [{"type": "png", "data": PNG_1X1_BASE64}]}
    summary = persist_sandbox_images_to_snapshot(
        resp=resp,
        run_name="run_legacy",
        iteration=1,
        step=3,
        agent_id="agent_legacy",
        session_id="sid",
        job_id="job",
        cell_no=1,
        base_snapshots_dir="snapshots",
        graph_logic_cfg={"enabled": False},
        code="import matplotlib.pyplot as plt\nplt.plot([1,2],[3,4])\n",
    )
    assert summary["image_count"] == 1
    assert summary["plot_paths"]
    assert summary["plot_filenames"] == []
    assert summary["plot_handles"] == []
    assert summary["plot_citations"] == []
    assert summary["plot_summary_paths"] == []
    assert summary["plot_artifacts"] == []
    assert summary["summary_manifest_path"] is None
    iteration_dir = tmp_path / "snapshots" / "run_legacy" / "iteration_1"
    rows = [
        json.loads(line)
        for line in (iteration_dir / "plots" / "manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    assert "plot_filename" not in rows[0]


def test_summary_generation_failure_writes_fallback_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("PROJECT_DIR", str(tmp_path))

    def _boom(**_: object) -> object:
        raise RuntimeError("forced-summary-failure")

    monkeypatch.setattr("env_setup.sandbox_utils.generate_plot_summary_for_image", _boom)
    resp = {"status": "success", "images": [{"type": "png", "data": PNG_1X1_BASE64}]}
    summary = persist_sandbox_images_to_snapshot(
        resp=resp,
        run_name="run_graph_failure",
        iteration=1,
        step=1,
        agent_id="agent_fail",
        session_id="sid",
        job_id="job",
        cell_no=1,
        base_snapshots_dir="snapshots",
        graph_logic_cfg={"enabled": True},
        code="print('hello')",
    )
    assert summary["image_count"] == 1
    assert len(summary["plot_artifacts"]) == 1
    assert any("plot_summary_generation_failed" in w for w in summary["warnings"])
    iteration_dir = tmp_path / "snapshots" / "run_graph_failure" / "iteration_1"
    artifact = summary["plot_artifacts"][0]
    payload = json.loads((iteration_dir / artifact["summary_path"]).read_text(encoding="utf-8"))
    assert payload["generation_backend"] == "heuristic_fallback_only"


def test_graph_logic_default_backend_is_non_vision(tmp_path, monkeypatch):
    monkeypatch.setenv("PROJECT_DIR", str(tmp_path))
    resp = {"status": "success", "images": [{"type": "png", "data": PNG_1X1_BASE64}]}
    summary = persist_sandbox_images_to_snapshot(
        resp=resp,
        run_name="run_graph_default",
        iteration=1,
        step=1,
        agent_id="agent_1",
        session_id="sid",
        job_id="job",
        cell_no=1,
        base_snapshots_dir="snapshots",
        graph_logic_cfg={"enabled": True},
        code="import matplotlib.pyplot as plt\nplt.plot([1,2],[3,4])\n",
    )
    assert summary["plot_summary_backend"] == "code_stats_llm"


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


def test_snapshot_metadata_includes_plots_when_present(tmp_path, monkeypatch):
    monkeypatch.setenv("PROJECT_DIR", str(tmp_path))
    iteration_dir = tmp_path / "snapshots" / "run_c" / "iteration_2"
    plots_dir = iteration_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    (plots_dir / "manifest.jsonl").write_text("", encoding="utf-8")
    (plots_dir / "preexisting.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (plots_dir / "summary_manifest.jsonl").write_text("", encoding="utf-8")
    (plots_dir / "summaries").mkdir(parents=True, exist_ok=True)
    (plots_dir / "summaries" / "preexisting.png.json").write_text("{}", encoding="utf-8")

    out_dir = create_snapshot(
        iteration=2,
        agent_registry=_DummyRegistry(),
        archive=_DummyArchive(),
        private_archive=_DummyPrivateArchive(),
        network_metadata=None,
        session_stats_registry=SessionStatsRegistry(),
        exp_config=OmegaConf.create({"run_name": "run_c", "exp_name": "exp"}),
        run_name="run_c",
        base_snapshots_dir="snapshots",
    )
    assert Path(out_dir) == iteration_dir

    metadata = json.loads((iteration_dir / "snapshot_metadata.json").read_text(encoding="utf-8"))
    assert "plots/" in metadata["files_saved"]
    assert "plots/manifest.jsonl" in metadata["files_saved"]
    assert "plots/summary_manifest.jsonl" in metadata["files_saved"]
    assert "plots/summaries/" in metadata["files_saved"]
