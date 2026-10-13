"""End-to-end: a replacement agent takes over the backend of the agent it replaces."""

from __future__ import annotations

import re
import subprocess
import sys

from tests.e2e import harness
from tests.isolation import subprocess_env


def test_replacement_agent_keeps_the_replaced_agents_backend(tmp_path):
    overrides = harness.sized_overrides(
        [
            "network.num_agents=3",
            "llm_backends.counts.gpt4o=1",
            "llm_backends.counts.gpt5=1",
            "llm_backends.counts.deepseek=1",
            "llm_backends.counts.llama=0",
            "reputation.enabled=true",
            "reputation_pause.mode=full_pause_permanent",
            "reputation_pause.threshold=10",  # above every score: all agents are candidates
            "reputation_pause.max_paused_agents=1",
            "reputation_pause.replacement.enabled=true",
        ],
        tmp_path,
    )
    proc = subprocess.run(
        [sys.executable, "-m", "tests.e2e.fake_pipeline", *overrides, "network.iterations=1"],
        cwd=tmp_path,
        env=subprocess_env(PYTHONPATH=str(harness.REPO_ROOT)),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    log = (tmp_path / "hydra" / "run.log").read_text(encoding="utf-8")

    assigned = dict(re.findall(r"\[llm-assign\] agent_id=(\S+) backend=(\S+)", log))
    replaced = re.findall(r"\[agent-replace\] .*old_agent_id=(\S+) .*backend=(\S+)", log)
    assert len(replaced) == 1
    old_id, backend = replaced[0]
    assert backend == assigned[old_id]
    # The mix is not re-realised for the replacements, so nothing is "scaled down".
    assert "add up to more than network.num_agents" not in log


def test_replacement_of_an_agent_without_a_backend_uses_the_default_llm():
    from system.main import _replacement_llm_selections

    llm_a, default = object(), object()
    rows = [{"agent_id": "agent_02"}, {"agent_id": "agent_09"}]

    selections = _replacement_llm_selections(rows, {"agent_02": ("gpt5", llm_a)}, default)

    assert selections == [("gpt5", llm_a), ("default", default)]
