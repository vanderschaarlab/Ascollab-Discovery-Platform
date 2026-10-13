"""End-to-end: feedback_summary reads a feedback-experiment run written by the pipeline."""

from __future__ import annotations

import json

from scripts.analysis import feedback_summary
from tests.e2e import harness


def test_feedback_summary_reads_a_pipeline_run(tmp_path, monkeypatch):
    overrides = harness.conditions()["04__meta_review_only__kirc"]["overrides"]
    fixture = harness.run_condition(overrides, tmp_path / "run")
    assert fixture["exit_code"] == 0, fixture.get("error")
    (run_dir,) = sorted((tmp_path / "run" / "snapshots").glob("*"))

    out = tmp_path / "summary.json"
    monkeypatch.setattr(
        "sys.argv", ["feedback_summary", str(run_dir), "--output", str(out)], raising=True
    )
    feedback_summary.main()

    (snapshot,) = json.loads(out.read_text(encoding="utf-8"))["snapshots"]
    assert snapshot["iteration"] == harness.ROUNDS
    assert snapshot["metacognition_mode"] == "meta_review_only"
    assert snapshot["agents_with_feedback"] > 0
    assert snapshot["papers_produced"] > 0
    assert snapshot["research"]["sessions"] > 0
