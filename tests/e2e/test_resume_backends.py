"""End-to-end: a resumed run keeps every agent on the model it had."""

from __future__ import annotations

import re
import subprocess
import sys
from types import SimpleNamespace

from tests.e2e import harness
from tests.isolation import subprocess_env

OVERRIDES = [
    "exp_name=resume_backends",
    "run_name=resume_backends",
    "network.num_agents=5",
    "llm_backends.counts.gpt4o=2",
    "llm_backends.counts.gpt5=2",
    "llm_backends.counts.deepseek=1",
    "llm_backends.counts.llama=0",
    "research.research_max_iters=12",
    "data.preflight=false",
]


def _backends(tmp_path, name, *extra):
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.e2e.fake_pipeline",
            *OVERRIDES,
            *extra,
            f"hydra.run.dir={tmp_path / name}",
        ],
        cwd=tmp_path,
        env=subprocess_env(PYTHONPATH=str(harness.REPO_ROOT)),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    log = (tmp_path / name / "run.log").read_text(encoding="utf-8")
    return dict(re.findall(r"\[llm-assign\] agent_id=(\S+) backend=(\S+)", log))


def test_resumed_agents_keep_their_backends(tmp_path):
    # The restored registry lists agents in another order than they were created (it changes
    # as papers are accepted); pairing that order with the model plan swapped agents' models.
    first = _backends(tmp_path, "first", "network.iterations=2")
    resumed = _backends(
        tmp_path,
        "resumed",
        "network.iterations=3",
        "resume.enabled=true",
        f"resume.snapshot_path={tmp_path / 'snapshots' / 'resume_backends'}",
    )
    assert len(first) == 5
    assert resumed == first


def test_creation_order_sorts_by_the_number_ending_the_id():
    from system.main import _creation_order

    agents = [SimpleNamespace(agent_id=a) for a in ("agent_10", "x", "agent_02", "agent_1")]
    ordered = [a.agent_id for a in sorted(agents, key=_creation_order)]
    assert ordered == ["agent_1", "agent_02", "agent_10", "x"]
