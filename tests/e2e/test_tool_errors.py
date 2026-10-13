"""End-to-end: a tool call with a misnamed argument costs the agent a step, not its session."""

from __future__ import annotations

import json

from tests.e2e import harness

CONDITION = "01__ascollab__kirc"


def test_a_misnamed_tool_argument_does_not_end_the_session(tmp_path, monkeypatch):
    # search_internal_archive takes `query_text`; `query` fails the tool's argument schema.
    monkeypatch.setenv(
        "E2E_RESEARCH_FIRST_ACTION",
        json.dumps({"tool": "search_internal_archive", "args": {"query": "kidney cancer"}}),
    )
    fixture = harness.run_condition(harness.conditions()[CONDITION]["overrides"], tmp_path / "run")
    assert fixture["exit_code"] == 0, fixture.get("error")

    expected = json.loads((harness.FIXTURE_DIR / f"{CONDITION}.json").read_text(encoding="utf-8"))
    calls, before = fixture["llm_calls"], expected["llm_calls"]
    # Every session took one extra step and still wrote its paper, so every paper was reviewed.
    sessions = before["persona_expertise"] * harness.ROUNDS
    assert calls["research"] == before["research"] + sessions
    assert calls["reviewer"] == before["reviewer"]
    assert fixture["accepted"] == expected["accepted"]
