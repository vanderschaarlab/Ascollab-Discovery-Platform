"""End-to-end: the command rank.sh issues ranks two real (fake-LLM) runs with a scripted judge.

Covers the snapshot lookup override, the ranker config overrides rank.sh passes, and the paper
import from the snapshots the pipeline writes. No real model is called.
"""

from __future__ import annotations

import json
import subprocess
import sys

from tests.e2e import harness
from tests.isolation import script_env, subprocess_env

PAIR = ("indep_homo", "ascollab")


def _rank_command(env_overrides: dict[str, str]) -> list[str]:
    env = script_env(
        **env_overrides,
        DRY_RUN="1",
        DRY_RUN_FORMAT="json",
        DATASETS="kirc",
        RUBRICS="default",
        TOP_N="3",
    )
    out = subprocess.run(
        ["bash", str(harness.REPO_ROOT / "scripts/experiments/01_network_ablations/rank.sh")],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    rows = [json.loads(line) for line in out.splitlines()]
    (row,) = [r for r in rows if r["condition"] == f"{PAIR[0]}_vs_{PAIR[1]}__default"]
    argv = row["argv"]
    return argv[argv.index("ranker.run_sample_and_rank_paired") + 1 :]


def test_rank_sh_command_ranks_two_pipeline_runs(tmp_path):
    conditions = harness.conditions()
    run_dirs = {}
    for condition in PAIR:
        workdir = tmp_path / condition
        fixture = harness.run_condition(conditions[f"01__{condition}__kirc"]["overrides"], workdir)
        assert fixture["exit_code"] == 0, fixture.get("error")
        (run_dirs[condition],) = sorted((workdir / "snapshots").glob("*"))

    overrides = _rank_command(
        {f"SNAPSHOT_DIR_01_KIRC_{c.upper()}_SEED_1": str(d) for c, d in run_dirs.items()}
    )
    assert f'experiments.experiment_a.snapshot_dirs=["{run_dirs[PAIR[0]]}"]' in overrides

    rank_dir = tmp_path / "rank"
    rank_dir.mkdir()
    proc = subprocess.run(
        [sys.executable, "-m", "tests.e2e.fake_ranker", *overrides],
        cwd=rank_dir,
        env=subprocess_env(PYTHONPATH=str(harness.REPO_ROOT)),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]

    out = rank_dir / "outputs/ranker/01/kirc" / f"{PAIR[0]}_vs_{PAIR[1]}" / "default" / "seed1"
    records = json.loads((out / "comparisons.json").read_text(encoding="utf-8"))
    records = records if isinstance(records, list) else records.get("comparisons", [])
    assert records and all(r.get("status") == "completed" for r in records)

    # stat_tests reads the same layout.
    from scripts.analysis import stat_tests

    rows = stat_tests.compute(rank_dir / "outputs/ranker", n_boot=50)
    assert [(r["experiment"], r["dataset"], r["condition_a"]) for r in rows] == [
        ("01", "kirc", PAIR[0])
    ]
    assert rows[0]["n"] == len(records)
