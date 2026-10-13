"""End-to-end tests: every experiment condition runs through system.main with scripted fakes.

A failure means a change altered what the pipeline does or writes for that condition. If the
change is intended, run ``E2E_UPDATE=1 pytest tests/e2e`` and commit the fixture diff.
"""

from __future__ import annotations

import os

import pytest

from tests.e2e import harness

CONDITIONS = harness.conditions()
UPDATE = os.environ.get("E2E_UPDATE") == "1"


def test_every_condition_has_an_e2e_fixture_and_none_is_stale():
    fixture_ids = {p.stem for p in harness.FIXTURE_DIR.glob("*.json")}
    if UPDATE:
        harness.FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
        for stale in fixture_ids - set(CONDITIONS):
            (harness.FIXTURE_DIR / f"{stale}.json").unlink()
        return
    assert fixture_ids == set(CONDITIONS)


@pytest.mark.parametrize("fixture_id", sorted(CONDITIONS))
def test_condition_runs_end_to_end(fixture_id, tmp_path):
    fixture = harness.run_condition(CONDITIONS[fixture_id]["overrides"], tmp_path)
    text = harness.dump(fixture)
    path = harness.FIXTURE_DIR / f"{fixture_id}.json"
    if UPDATE:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return
    assert path.exists(), f"missing fixture {path.name}; run E2E_UPDATE=1"
    assert path.read_text(encoding="utf-8") == text, (
        f"{fixture_id} changed; review the diff and regenerate with E2E_UPDATE=1"
    )


def test_fixtures_keep_five_significant_digits_of_long_decimals():
    text = harness.dump(
        {"prompt": "cosine 0.16232170164585114, score 0.45", "x": 0.16232168674468994}
    )

    assert "0.16232," in text and '"x": 0.16232' in text
    assert "0.45" in text and "0.1623217" not in text
