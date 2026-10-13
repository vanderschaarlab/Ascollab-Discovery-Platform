"""Keep the tests independent of the developer's environment.

With LangSmith tracing variables set (``LANGSMITH_TRACING`` or ``LANGCHAIN_TRACING_V2``, with an
API key), LangChain's tracer and ``langsmith.traceable`` upload every chat-model call, so a test
run would send the pipeline's prompts, the fakes' replies included, to LangSmith. Tests therefore
run without any ``LANGCHAIN_*`` or ``LANGSMITH_*`` variable: subprocesses get ``subprocess_env()``,
and ``tests/conftest.py`` calls ``remove_tracing_variables`` before every test.

The experiment scripts (``scripts/experiments/*/run.sh``, ``rank.sh``) read knobs such as
``ITERATIONS``, ``NUM_AGENTS``, ``DATASETS``, ``SEEDS``, ``CONFIG_NAME`` or ``REVIEWER_COUNTS`` from
the environment, so a knob exported in the shell would change or drop the conditions the tests
read from their ``DRY_RUN`` output. ``script_env()`` runs them with only ``PATH``, ``HOME`` and the
locale, plus what the test sets.

The repository checks list the tracked files with ``tracked_files()``, which skips the test outside
a git checkout (a ``git archive`` export or a GitHub source zip has no ``.git``).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

TRACING_PREFIXES = ("LANGCHAIN_", "LANGSMITH_")
SCRIPT_ENV_KEEP = ("PATH", "HOME", "LANG", "LC_ALL")


def is_tracing_variable(name: str) -> bool:
    """Return whether an environment variable configures LangChain or LangSmith."""
    return name.upper().startswith(TRACING_PREFIXES)


def subprocess_env(**extra: str) -> dict[str, str]:
    """Return ``os.environ`` without tracing variables, plus ``extra``, for a test subprocess."""
    env = {k: v for k, v in os.environ.items() if not is_tracing_variable(k)}
    env.update(extra)
    return env


def script_env(**extra: str) -> dict[str, str]:
    """Return the environment for an experiment script: ``SCRIPT_ENV_KEEP`` plus ``extra``."""
    env = {k: os.environ[k] for k in SCRIPT_ENV_KEEP if k in os.environ}
    env.update(extra)
    return env


def remove_tracing_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove the tracing variables from this process for the duration of one test."""
    for name in [k for k in os.environ if is_tracing_variable(k)]:
        monkeypatch.delenv(name)
    # langsmith caches its environment lookups, so a value read before the variables were removed
    # (for example at import time) would otherwise keep tracing on.
    from langsmith.utils import get_env_var

    cache_clear = getattr(get_env_var, "cache_clear", None)
    if cache_clear is not None:
        cache_clear()


def tracked_files(repo_root: Path) -> list[str]:
    """Return ``git ls-files`` of the repository; skip the calling test outside a git checkout."""
    import pytest

    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], cwd=repo_root, capture_output=True, text=True
        )
    except FileNotFoundError:
        pytest.skip("git is not installed; this check lists the files tracked in the repository")
    if top.returncode != 0 or Path(top.stdout.strip()).resolve() != repo_root.resolve():
        pytest.skip(
            "not a git checkout (e.g. a git archive or GitHub zip); "
            "this check lists the files tracked in the repository"
        )
    out = subprocess.run(
        ["git", "ls-files"], cwd=repo_root, capture_output=True, text=True, check=True
    ).stdout
    return out.splitlines()
