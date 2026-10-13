"""End-to-end: a finished run can be resumed, and the ranker can read its snapshots."""

from __future__ import annotations

import subprocess
import sys

from tests.e2e import harness
from tests.isolation import subprocess_env

CONDITION = "01__ascollab__kirc"


def test_run_resumes_and_ranker_reads_its_snapshots(tmp_path):
    overrides = harness.conditions()[CONDITION]["overrides"]
    first = harness.run_condition(overrides, tmp_path)
    assert first["exit_code"] == 0, first.get("error")
    run_dirs = sorted((tmp_path / "snapshots").glob("*"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    from ranker.snapshot_import import collect_papers, to_ranker_format

    papers = collect_papers([str(run_dir)], iterations="last", cumulative=True)
    assert papers, "ranker found no papers in the snapshot"
    for paper in papers:
        formatted = to_ranker_format(paper)
        assert formatted and all(isinstance(v, str) for v in formatted.values())

    resumed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.e2e.fake_pipeline",
            *harness.sized_overrides(overrides, tmp_path),
            "resume.enabled=true",
            f"resume.snapshot_path={run_dir}",
            "network.iterations=3",
        ],
        cwd=tmp_path,
        env=subprocess_env(PYTHONPATH=str(harness.REPO_ROOT)),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert resumed.returncode == 0, resumed.stderr[-2000:]
    assert (run_dir / "iteration_3").is_dir(), sorted(p.name for p in run_dir.iterdir())
