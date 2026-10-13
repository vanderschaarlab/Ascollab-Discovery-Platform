"""End-to-end: a best-effort feedback skip is recorded and replaces the previous feedback."""

from __future__ import annotations

import json
import pickle

from tests.e2e import harness


def test_skipped_feedback_is_recorded_in_snapshot_and_learning_state(tmp_path, monkeypatch):
    monkeypatch.setenv("E2E_SYNTHESIZER_REPLY", "not json")
    # Experiment 04's overrides, which make the feedback best-effort.
    overrides = harness.conditions()["04__meta_review_only__kirc"]["overrides"]
    assert "research.metacognition_feedback.best_effort=true" in overrides
    fixture = harness.run_condition(overrides, tmp_path / "run")
    assert fixture["exit_code"] == 0, fixture.get("error")
    (run_dir,) = sorted((tmp_path / "run" / "snapshots").glob("*"))
    last = run_dir / f"iteration_{harness.ROUNDS}"

    feedback = json.loads((last / "metacognition_feedback.json").read_text(encoding="utf-8"))
    assert feedback["feedback_by_agent"] == {}
    assert feedback["warnings_by_agent"]
    skipped_ids = set(feedback["warnings_by_agent"])

    with (last / "agent_registry_data.pkl").open("rb") as f:
        agents = pickle.load(f)["agents"]
    skipped_agents = [agent for agent in agents if agent.agent_id in skipped_ids]
    assert skipped_agents
    for agent in skipped_agents:
        active = agent.learning_state["metacognition_feedback"]["active"]
        assert "unparseable" in active["skipped"]
        assert "summary" not in active
