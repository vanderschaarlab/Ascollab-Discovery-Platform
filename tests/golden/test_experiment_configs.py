"""Golden tests: every experiment condition resolves to its recorded config, tools and prompts.

A failure means a change altered what an experiment would do. If the change is intended, run
``GOLDEN_UPDATE=1 pytest tests/golden`` and commit the fixture diff with an explanation.
"""

from __future__ import annotations

import os

import pytest

from tests.golden import harness

CONDITIONS = harness.collect_conditions()
UPDATE = os.environ.get("GOLDEN_UPDATE") == "1"


def test_every_condition_has_a_fixture_and_no_fixture_is_stale():
    fixture_ids = {p.stem for p in harness.FIXTURE_DIR.glob("*.json")}
    if UPDATE:
        harness.FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
        for stale in fixture_ids - set(CONDITIONS):
            (harness.FIXTURE_DIR / f"{stale}.json").unlink()
        return
    assert fixture_ids == set(CONDITIONS)


@pytest.mark.parametrize("fixture_id", sorted(CONDITIONS))
def test_condition_matches_golden_fixture(fixture_id):
    spec = CONDITIONS[fixture_id]
    fixture = harness.build_fixture(spec["config_name"], spec["overrides"])
    harness.assert_no_secrets(fixture)
    text = harness.dump(fixture)
    path = harness.FIXTURE_DIR / f"{fixture_id}.json"
    if UPDATE:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return
    assert path.exists(), f"missing fixture {path.name}; run GOLDEN_UPDATE=1"
    assert path.read_text(encoding="utf-8") == text, (
        f"{fixture_id} changed; review the diff and regenerate with GOLDEN_UPDATE=1"
    )


@pytest.mark.parametrize(
    ("mode", "narrowed"),
    [("default", False), ("experimental", True), ("Experimental", True)],
)
def test_reviewer_prompt_metrics_are_chosen_as_system_main_does(mode, narrowed):
    # system.main narrows the reviewer prompt to the allowlist only in experimental mode, after
    # normalising the mode; otherwise the prompt lists every metric.
    cfg = harness.compose(
        "ascollab",
        [f"review.metrics_pipeline={mode}", "review.metrics_allowlist=[support.lines_of_code]"],
    )
    prompt = harness._reviewer_prompt(cfg)
    assert "metrics.support.lines_of_code" in prompt
    assert ("metrics.soundness.internal_consistency_score" not in prompt) is narrowed


@pytest.mark.parametrize("mode", ["none", "off", "disabled"])
def test_reviewer_prompt_has_no_metrics_guidance_when_metrics_are_off(mode):
    cfg = harness.compose("ascollab", [f"review.metrics_pipeline={mode}"])
    prompt = harness._reviewer_prompt(cfg)
    assert "METRICS" not in prompt and "metric values" not in prompt
    assert "justify the Overall Recommendation." in prompt
