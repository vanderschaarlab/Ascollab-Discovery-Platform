"""run_grid draws the AutoRanker step's cohorts from the snapshot the review step recorded."""

import json

import pytest

from scripts.analysis.reviewer_matching import run_grid


def _record(root, dataset, snapshot_dirs):
    """Cohort manifests as the review step's prepare_cohorts writes them."""
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    for name in (f"{dataset}_iter5_cohort0", f"{dataset}_iter5_cohort1"):
        doc = {"dataset": dataset, "snapshot_dirs": snapshot_dirs, "candidate_paper_ids": []}
        (manifests / f"{name}.json").write_text(json.dumps(doc), encoding="utf-8")


@pytest.fixture
def commands(monkeypatch):
    """The commands run_grid runs, recorded instead of run."""
    ran: list[list[str]] = []
    monkeypatch.setattr(run_grid, "_run", ran.append)
    return ran


def _main(monkeypatch, root, *args):
    monkeypatch.setattr(
        "sys.argv",
        ["run_grid", "--datasets", "nhanes", "--root-output-dir", str(root), *args],
    )
    run_grid.main()


def _prepare_cohorts_snapshots(commands):
    (prepare,) = [c for c in commands if "scripts.analysis.reviewer_matching.prepare_cohorts" in c]
    start = prepare.index("--snapshot-dirs") + 1
    return prepare[start : prepare.index("--target-iterations")]


def test_rank_step_reuses_the_recorded_snapshot(tmp_path, monkeypatch, commands):
    _record(tmp_path, "nhanes", ["snapshots/run_a"])

    _main(monkeypatch, tmp_path, "--skip-reviews")

    assert _prepare_cohorts_snapshots(commands) == ["snapshots/run_a"]


def test_the_recorded_snapshot_may_also_be_given(tmp_path, monkeypatch, commands):
    run = tmp_path / "snapshots" / "run_a"
    _record(tmp_path, "nhanes", [str(run)])
    monkeypatch.chdir(tmp_path)

    _main(monkeypatch, tmp_path, "--snapshot-dirs", "nhanes=snapshots/run_a")

    assert _prepare_cohorts_snapshots(commands) == ["snapshots/run_a"]


def test_a_different_snapshot_stops_before_anything_runs(tmp_path, monkeypatch, commands):
    _record(tmp_path, "nhanes", ["snapshots/run_a"])

    with pytest.raises(SystemExit, match=r"drawn from \['snapshots/run_a'\]"):
        _main(monkeypatch, tmp_path, "--snapshot-dirs", "nhanes=snapshots/run_b", "--skip-reviews")
    assert commands == []


def test_without_a_record_the_snapshot_must_be_given(tmp_path, monkeypatch, commands):
    with pytest.raises(SystemExit, match="run the review step first"):
        _main(monkeypatch, tmp_path, "--skip-reviews")

    _main(monkeypatch, tmp_path, "--snapshot-dirs", "nhanes=snapshots/run_a")
    assert _prepare_cohorts_snapshots(commands) == ["snapshots/run_a"]


def test_judge_and_workers_reach_the_autoranker_only_when_given(tmp_path, monkeypatch, commands):
    _record(tmp_path, "nhanes", ["snapshots/run_a"])
    autoranker = "scripts.analysis.reviewer_matching.run_autoranker_for_cohort"

    _main(monkeypatch, tmp_path, "--skip-reviews")
    defaults = [c for c in commands if autoranker in c]
    commands.clear()
    _main(monkeypatch, tmp_path, "--skip-reviews", "--judge", "gpt4o", "--max-workers", "8")
    given = [c for c in commands if autoranker in c]

    assert len(defaults) == len(given) == 2
    assert all("--judge" not in c and "--max-workers" not in c for c in defaults)
    assert all(c[-4:] == ["--judge", "gpt4o", "--max-workers", "8"] for c in given)


def test_analysis_runs_only_with_the_autoranker_step(tmp_path, monkeypatch, commands):
    analysis = "scripts.analysis.reviewer_matching.analyze_results"
    _record(tmp_path, "nhanes", ["snapshots/run_a"])

    _main(monkeypatch, tmp_path, "--skip-autoranker")  # 03's run.sh: reviews only
    assert not any(analysis in c for c in commands)

    _main(monkeypatch, tmp_path, "--skip-reviews")  # 03's rank.sh
    assert sum(analysis in c for c in commands) == 1
