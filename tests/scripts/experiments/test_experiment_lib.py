"""scripts/experiments/lib/common.sh: which snapshot rank.sh ranks and the shared baseline reuses.

Most tests source a copy of the library in a temporary project root that holds fake snapshot
folders. ``emit`` (the only function that executes a command) is replaced by a stub that prints the
command, and ``PYTHON`` is ``false``, so nothing runs; ``DRY_RUN`` is 0 only so that the lookups
read the fake folders. The scripts themselves run with ``DRY_RUN=1``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from tests.isolation import script_env

REPO_ROOT = Path(__file__).resolve().parents[3]
LIB = REPO_ROOT / "scripts" / "experiments" / "lib" / "common.sh"
BASELINE = "ascollab_rb_kirc_baseline_seed_1"
# Replaces emit: print the command instead of running it.
_EMIT_STUB = 'emit() { shift 5; [[ "$1" == "--" ]] && shift; echo "EMIT $*"; }'


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A project root with a copy of lib/common.sh and the kirc bundle it checks for."""
    lib = tmp_path / "scripts" / "experiments" / "lib"
    lib.mkdir(parents=True)
    shutil.copy(LIB, lib / "common.sh")
    (tmp_path / "sys_config" / "dataset").mkdir(parents=True)
    (tmp_path / "sys_config" / "dataset" / "kirc.yaml").write_text("{}\n", encoding="utf-8")
    return tmp_path


def _snapshot(project: Path, name: str, rounds: int, age: int) -> Path:
    """A run folder with rounds 1..rounds finished, last modified ``age`` seconds ago."""
    run = project / "snapshots" / name
    run.mkdir(parents=True)
    for n in range(1, rounds + 1):
        (run / f"iteration_{n}").mkdir()
        (run / f"iteration_{n}" / "snapshot_metadata.json").write_text("{}", encoding="utf-8")
    stamp = time.time() - age
    os.utime(run, (stamp, stamp))
    return run


def _bash(project: Path, script: str, **env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", f"source scripts/experiments/lib/common.sh\n{_EMIT_STUB}\n{script}"],
        cwd=project,
        env={"PATH": os.environ["PATH"], "DRY_RUN": "0", "PYTHON": "false", **env},
        capture_output=True,
        text=True,
        check=False,
    )


def test_lookup_skips_newer_runs_that_did_not_finish(project):
    complete = _snapshot(project, f"{BASELINE}_260101_1200", rounds=20, age=300)
    failed = _snapshot(project, f"{BASELINE}_260102_1200", rounds=0, age=200)  # start-up failure
    partial = _snapshot(project, f"{BASELINE}_260103_1200", rounds=12, age=100)  # stopped early
    smoke = _snapshot(project, f"{BASELINE}_260104_1200", rounds=1, age=0)  # ITERATIONS=1

    proc = _bash(project, "snapshot_dir rb baseline kirc 1")

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == f"snapshots/{complete.name}"
    for skipped in (smoke, partial, failed):
        assert f"skipping snapshots/{skipped.name}: not a complete run" in proc.stderr
    # A smoke run (ITERATIONS=1) counts as complete for a smoke ranking.
    smoke_lookup = _bash(project, "snapshot_dir rb baseline kirc 1", ITERATIONS="1")
    assert smoke_lookup.stdout.strip() == f"snapshots/{smoke.name}"


def test_lookup_fails_when_no_run_finished(project):
    _snapshot(project, f"{BASELINE}_260102_1200", rounds=0, age=0)

    proc = _bash(project, "snapshot_dir rb baseline kirc 1")

    assert proc.returncode != 0
    assert proc.stdout == ""
    assert "ERROR: no complete run for snapshots/ascollab_rb_kirc_baseline_seed_1_*" in proc.stderr


def test_shared_baseline_is_rerun_unless_a_complete_run_exists(project):
    _snapshot(project, f"{BASELINE}_260102_1200", rounds=0, age=100)
    _snapshot(project, f"{BASELINE}_260103_1200", rounds=1, age=0)

    rerun = _bash(project, "run_shared_baseline kirc 1")

    assert rerun.returncode == 0, rerun.stderr
    assert "EMIT false -m system.main" in rerun.stdout
    assert f"exp_name={BASELINE}" in rerun.stdout

    complete = _snapshot(project, f"{BASELINE}_260101_1200", rounds=20, age=300)
    reuse = _bash(project, "run_shared_baseline kirc 1")

    assert reuse.returncode == 0, reuse.stderr
    assert "EMIT" not in reuse.stdout
    assert f"exists (snapshots/{complete.name}); reusing it" in reuse.stderr


def test_override_names_one_run(project):
    # 05 and 10 both have a condition n1, and every seed is a different run.
    run = _snapshot(project, "copied_n1_run", rounds=40, age=0)
    env = {"SNAPSHOT_DIR_05_KIRC_N1_SEED_1": str(run)}

    assert _bash(project, "snapshot_dir 05 n1 kirc 1", **env).stdout.strip() == str(run)
    for other in ("10 n1 kirc 1", "05 n1 kirc 2"):
        proc = _bash(project, f"snapshot_dir {other}", **env)
        assert proc.returncode != 0
        assert "no complete run" in proc.stderr


def test_override_must_be_a_folder_and_warns_when_unfinished(project):
    missing = _bash(
        project, "snapshot_dir 01 ascollab kirc 1", SNAPSHOT_DIR_01_KIRC_ASCOLLAB_SEED_1="nowhere"
    )
    assert missing.returncode != 0
    assert "ERROR: SNAPSHOT_DIR_01_KIRC_ASCOLLAB_SEED_1=nowhere is not a folder" in missing.stderr

    partial = _snapshot(project, "copied_partial_run", rounds=12, age=0)
    proc = _bash(
        project,
        "snapshot_dir 01 ascollab kirc 1",
        SNAPSHOT_DIR_01_KIRC_ASCOLLAB_SEED_1=str(partial),
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == str(partial)
    assert "WARNING" in proc.stderr and "iteration_40/snapshot_metadata.json" in proc.stderr


@pytest.mark.parametrize("rounds", [None, 0, 12])
def test_baseline_override_is_reused_only_when_complete(project, rounds):
    var = "SNAPSHOT_DIR_RB_KIRC_BASELINE_SEED_1"
    path = "nowhere" if rounds is None else str(_snapshot(project, "base", rounds, age=0))

    proc = _bash(project, "run_shared_baseline kirc 1", **{var: path})

    assert proc.returncode != 0
    assert "EMIT" not in proc.stdout
    assert f"ERROR: {var}={path} is not a complete baseline run" in proc.stderr

    complete = _snapshot(project, "complete_base", rounds=20, age=0)
    reuse = _bash(project, "run_shared_baseline kirc 1", **{var: str(complete)})
    assert reuse.returncode == 0, reuse.stderr
    assert "EMIT" not in reuse.stdout
    assert f"{var}={complete}; reusing it" in reuse.stderr


def test_rank_sh_uses_an_override_only_for_its_own_run(tmp_path):
    env = script_env(
        DRY_RUN="1",
        DRY_RUN_FORMAT="json",
        SEEDS="1 2",
        RUBRICS="novelty",
        SNAPSHOT_DIR_05_KIRC_N1_SEED_1=str(tmp_path),
        SNAPSHOT_DIR_10_KIRC_N3_SEED_2=str(tmp_path),
    )
    out = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts/experiments/10_reviewer_count/rank.sh")],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    using = {
        (row["condition"].split("_vs_")[0], row["seed"])
        for row in map(json.loads, out.splitlines())
        if f'experiments.experiment_a.snapshot_dirs=["{tmp_path}"]' in row["argv"]
    }
    assert using == {("n3", "2")}


def _run_script_rounds() -> dict[str, set[str]]:
    """network.iterations of every run each run.sh prints, by experiment ID."""
    env = script_env(DRY_RUN="1", DRY_RUN_FORMAT="json")
    rounds: dict[str, set[str]] = {}
    for script in sorted((REPO_ROOT / "scripts" / "experiments").glob("*/run.sh")):
        out = subprocess.run(
            ["bash", str(script)], env=env, check=True, capture_output=True, text=True
        ).stdout
        for row in map(json.loads, out.splitlines()):
            if row["kind"] == "run":
                (value,) = [a.split("=", 1)[1] for a in row["argv"] if "network.iterations=" in a]
                rounds.setdefault(row["experiment"], set()).add(value)
    return rounds


def test_rounds_for_matches_the_rounds_each_run_script_sets():
    rounds = _run_script_rounds()
    assert set(rounds) == {"01", "02", "04", "05", "06", "07", "08", "09", "10", "rb"}
    script = "; ".join(f"rounds_for {exp}" for exp in sorted(rounds))
    out = subprocess.run(
        ["bash", "-c", f"source {LIB}; {script}"],
        env={"PATH": os.environ["PATH"]},
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    assert dict(zip(sorted(rounds), out, strict=True)) == {
        exp: value for exp, (value,) in rounds.items()
    }


def test_03_rank_sh_does_not_look_up_the_01_snapshot_again():
    # run.sh's cohort manifests record the 01 snapshot; rank.sh must not pass a newer one.
    env = script_env(DRY_RUN="1", DRY_RUN_FORMAT="json")
    out = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts/experiments/03_reviewer_matching/rank.sh")],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    (autoranker,) = [r for r in map(json.loads, out.splitlines()) if r["condition"] == "autoranker"]
    assert "--skip-reviews" in autoranker["argv"]
    assert "--snapshot-dirs" not in autoranker["argv"]


def _dry_run_rows(script: str, **env: str) -> list[dict]:
    run_env = script_env(DRY_RUN="1", DRY_RUN_FORMAT="json", **env)
    out = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts/experiments" / script)],
        env=run_env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [json.loads(line) for line in out.splitlines()]


def test_judge_and_workers_reach_experiment_03_only_when_set():
    (default,) = [r for r in _dry_run_rows("03_reviewer_matching/rank.sh") if r["kind"] == "rank"]
    assert "--judge" not in default["argv"] and "--max-workers" not in default["argv"]

    rows = _dry_run_rows("03_reviewer_matching/rank.sh", JUDGE="gpt4o", MAX_WORKERS="3")
    (given,) = [r for r in rows if r["kind"] == "rank"]
    assert given["argv"][-4:] == ["--judge", "gpt4o", "--max-workers", "3"]


def _rank_config(tmp_path: Path, script: str, **env: str):
    """The ranker overrides of the first command ``script`` prints, and the config they compose.

    The config is composed as the ranker does, with a (template) endpoint file for every backend.
    """
    from hydra import compose, initialize_config_dir

    argv = _dry_run_rows(script, DATASETS="kirc", **env)[0]["argv"]
    overrides = argv[argv.index("ranker.run_sample_and_rank_paired") + 1 :]
    config_dir = tmp_path / "sys_config"
    shutil.copytree(REPO_ROOT / "sys_config", config_dir, ignore=shutil.ignore_patterns("*_api"))
    (config_dir / "llm_api").mkdir()
    template = REPO_ROOT / "sys_config" / "llm_api" / "default_to_complete.yaml"
    for name in ("default", "gpt-4o", "gpt-5", "gpt-5-high", "deepseek", "llama"):
        shutil.copy(template, config_dir / "llm_api" / f"{name}.yaml")
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="sample_and_rank_paired",
            overrides=[o for o in overrides if not o.startswith("hydra.")],
        )
    return overrides, cfg


@pytest.mark.parametrize("judge", [None, "gpt4o", "deepseek", "llama"])
def test_rank_sh_judge_can_be_any_pipeline_backend(tmp_path, judge):
    env = {"RUBRICS": "default"} | ({"JUDGE": judge} if judge else {})
    overrides, cfg = _rank_config(tmp_path, "01_network_ablations/rank.sh", **env)
    assert f"ranker.llm_backend={judge or 'gpt5-high'}" in overrides
    assert "parallelism.max_workers=8" in overrides
    assert cfg.ranker.llm_backend in cfg.endpoints


def test_04_rank_sh_judges_all_pairs_when_fewer_than_n_pairs_exist(tmp_path):
    _, cfg = _rank_config(tmp_path, "04_feedback/rank.sh", RUBRICS="default")
    assert cfg.sampling.mode == "random"
    assert cfg.sampling.n_pairs == 500
    assert cfg.sampling.with_replacement is False
    assert cfg.sampling.cap_n_pairs is True


def test_an_exported_target_does_not_change_any_run():
    # Only 06 sets discovery_dataset.target (per arm); TARGET or V_TARGET in the caller's
    # environment must not reach any run.
    targets = {}
    for script in sorted((REPO_ROOT / "scripts" / "experiments").glob("*/run.sh")):
        rel = script.relative_to(REPO_ROOT / "scripts" / "experiments")
        for row in _dry_run_rows(str(rel), TARGET="kirc_closed_hypotheses", V_TARGET="tcga"):
            if row["kind"] == "run":
                key = (row["experiment"], row["condition"], row["dataset"])
                targets[key] = [a for a in row["argv"] if a.startswith("discovery_dataset.target=")]
    assert targets[("06", "closed", "kirc")] == ["discovery_dataset.target=kirc_closed_hypotheses"]
    assert targets[("06", "open_ended", "kirc")] == ["discovery_dataset.target=tcga"]
    assert len(targets) > 2
    assert all(t == [] for key, t in targets.items() if key[0] != "06")
