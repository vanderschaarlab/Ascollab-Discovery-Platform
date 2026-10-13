"""Tests run without LangSmith tracing or the experiment scripts' knobs from the shell, and the
repository checks need a git checkout."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from langsmith.utils import get_env_var, tracing_is_enabled

from tests.e2e import harness
from tests.golden import harness as golden_harness
from tests.isolation import (
    is_tracing_variable,
    remove_tracing_variables,
    script_env,
    subprocess_env,
    tracked_files,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACING_ON = {
    "LANGSMITH_TRACING": "true",
    "LANGCHAIN_TRACING_V2": "true",
    "LANGCHAIN_API_KEY": "fake-key",
    "LANGSMITH_ENDPOINT": "http://127.0.0.1:9",
}
_CHILD = "from langsmith.utils import tracing_is_enabled; print(tracing_is_enabled())"


def _set_tracing(monkeypatch) -> None:
    for name, value in TRACING_ON.items():
        monkeypatch.setenv(name, value)


def test_subprocess_env_drops_only_tracing_variables(monkeypatch):
    _set_tracing(monkeypatch)
    monkeypatch.setenv("KEEP_ME", "1")

    env = subprocess_env(EXTRA="2")

    assert not [k for k in env if k.startswith(("LANGCHAIN_", "LANGSMITH_"))]
    assert env["KEEP_ME"] == "1" and env["EXTRA"] == "2" and env["PATH"] == os.environ["PATH"]


def test_a_child_process_started_with_subprocess_env_does_not_trace(monkeypatch):
    _set_tracing(monkeypatch)

    def child(env):
        return subprocess.run(
            [sys.executable, "-c", _CHILD], env=env, capture_output=True, text=True, check=True
        ).stdout.strip()

    assert child(dict(os.environ)) == "True"  # what passing the environment through did
    assert child(subprocess_env()) == "False"


def test_remove_tracing_variables_turns_tracing_off_in_this_process(monkeypatch):
    _set_tracing(monkeypatch)
    get_env_var.cache_clear()
    assert tracing_is_enabled() is True  # and langsmith has now cached that

    remove_tracing_variables(monkeypatch)

    assert not [k for k in os.environ if is_tracing_variable(k)]
    assert tracing_is_enabled() is False


def test_every_test_runs_without_tracing_variables(request):
    assert "_no_langsmith_tracing" in request.fixturenames  # the autouse fixture in conftest.py
    assert not [k for k in os.environ if is_tracing_variable(k)]
    assert tracing_is_enabled() is False


def test_the_e2e_pipeline_subprocess_gets_no_tracing_variables(monkeypatch, tmp_path):
    _set_tracing(monkeypatch)
    seen = {}

    def fake_run(cmd, **kwargs):
        seen.update(kwargs["env"])
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")

    monkeypatch.setattr(harness.subprocess, "run", fake_run)
    monkeypatch.setattr(harness, "_default_k", lambda: 2)
    harness.run_condition([], tmp_path)

    assert seen["E2E_CAPTURE"] == str(tmp_path / "capture.json")
    assert not [k for k in seen if is_tracing_variable(k)]


def test_script_env_passes_only_path_home_and_the_locale(monkeypatch):
    _set_tracing(monkeypatch)
    for knob in ("ITERATIONS", "REVIEWER_COUNTS", "CONFIG_NAME", "BASH_ENV"):
        monkeypatch.setenv(knob, "1")

    env = script_env(DRY_RUN="1")

    assert set(env) <= {"PATH", "HOME", "LANG", "LC_ALL", "DRY_RUN"}
    assert env["PATH"] == os.environ["PATH"] and env["DRY_RUN"] == "1"


def test_collect_conditions_runs_the_scripts_without_exported_knobs(monkeypatch):
    # An exported REVIEWER_COUNTS=1 used to drop the 10__n3/10__n4 conditions, ITERATIONS=3 to
    # shorten every condition.
    monkeypatch.setenv("REVIEWER_COUNTS", "1")
    monkeypatch.setenv("ITERATIONS", "3")
    envs = []

    def fake_run(cmd, **kwargs):
        envs.append(kwargs["env"])
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(golden_harness.subprocess, "run", fake_run)
    assert golden_harness.collect_conditions() == {}

    assert envs and all(e["DRY_RUN"] == "1" for e in envs)
    assert not [e for e in envs if "REVIEWER_COUNTS" in e or "ITERATIONS" in e]


def test_no_test_hands_its_raw_environment_to_a_subprocess():
    raw = ("**os.environ", "env=os.environ", "env=dict(os.environ")
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in sorted((REPO_ROOT / "tests").rglob("*.py"))
        if path != Path(__file__).resolve()
        and any(pattern in path.read_text(encoding="utf-8") for pattern in raw)
    ]
    assert offenders == []


def test_tracked_files_lists_the_files_of_this_checkout():
    files = tracked_files(REPO_ROOT)
    assert "README.md" in files and "tests/isolation.py" in files


def test_tracked_files_skips_outside_a_git_checkout(tmp_path):
    (tmp_path / "README.md").write_text("an exported copy\n", encoding="utf-8")
    with pytest.raises(pytest.skip.Exception, match="not a git checkout"):
        tracked_files(tmp_path)
