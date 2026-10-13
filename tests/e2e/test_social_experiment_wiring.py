"""End-to-end wiring of experiments 06, 08 and 09: closed task, Markovian memory and reviewer
norms."""

from __future__ import annotations

import json
import subprocess
import sys

from tests.e2e import harness
from tests.isolation import subprocess_env

NORM_A_TEXT = "prioritize papers that are potentially groundbreaking"


def _public_papers_by_round(iteration_dir) -> dict[int, set[str]]:
    """Paper IDs in a round's public archive snapshot, by publication round."""
    doc = json.loads((iteration_dir / "archive_data.json").read_text(encoding="utf-8"))
    by_round: dict[int, set[str]] = {}
    for paper in doc["papers"]:
        by_round.setdefault(int(paper["publication_t"]), set()).add(paper["paper_id"])
    return by_round


def _run_markovian(overrides, workdir, *extra) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.e2e.fake_pipeline",
            *harness.sized_overrides(overrides, workdir),
            *extra,
        ],
        cwd=workdir,
        env=subprocess_env(PYTHONPATH=str(harness.REPO_ROOT)),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]


def test_markovian_memory_prunes_at_each_round_and_after_resume(tmp_path):
    overrides = harness.conditions()["08__markovian_memory__kirc"]["overrides"]
    _run_markovian(overrides, tmp_path, "network.iterations=3")
    (run_dir,) = sorted((tmp_path / "snapshots").glob("*"))
    _run_markovian(
        overrides,
        tmp_path,
        "resume.enabled=true",
        f"resume.snapshot_path={run_dir}",
        "network.iterations=4",
    )

    # Before each round t the public archive keeps only round t-1's papers; the round's snapshot
    # then adds round t's. Rounds 1-3 ran in one process, round 4 after a resume.
    papers = {t: _public_papers_by_round(run_dir / f"iteration_{t}") for t in range(1, 5)}
    for t in range(1, 5):
        assert set(papers[t]) == {t - 1, t} - {0}, (t, papers[t])
        assert papers[t][t], f"round {t} accepted no paper"
        pruning = json.loads(
            (run_dir / f"iteration_{t}" / "social_memory_pruning.json").read_text(encoding="utf-8")
        )
        assert pruning["mode"] == "markovian"
        assert pruning["retained_publication_t"] == t - 1
        # Dropped at round t: round t-2's papers, as the previous snapshot held them (none before
        # round 3; round 4's prune runs after the resume).
        expected = papers[t - 1].get(t - 2, set()) if t > 1 else set()
        assert set(pruning["dropped_paper_ids"]) == expected, t


def _calls(condition: str, workdir, kind: str) -> list[dict]:
    fixture = harness.run_condition(harness.conditions()[condition]["overrides"], workdir)
    assert fixture["exit_code"] == 0, fixture.get("error")
    calls = json.loads((workdir / "capture.json").read_text(encoding="utf-8"))
    return [c for c in calls if c["kind"] == kind]


def test_reviewer_norm_reaches_every_reviewer_prompt(tmp_path):
    with_norm = _calls("09__norm_a_novelty__kirc", tmp_path / "norm", "reviewer")
    assert with_norm and all(NORM_A_TEXT in c["system"] for c in with_norm)

    baseline = _calls("rb__baseline__kirc", tmp_path / "baseline", "reviewer")
    assert baseline and not any(NORM_A_TEXT in c["system"] for c in baseline)


def test_closed_task_goal_opens_every_research_session(tmp_path):
    from env_setup.goals import GOALS

    closed = _calls("06__closed__kirc", tmp_path / "closed", "research")
    assert closed and all(c["human"] == GOALS["kirc_closed_hypotheses"] for c in closed)

    open_ended = _calls("06__open_ended__kirc", tmp_path / "open", "research")
    assert open_ended and all(c["human"] == GOALS["tcga"] for c in open_ended)
