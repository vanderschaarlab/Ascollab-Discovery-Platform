"""stat_tests on synthetic rank.sh outputs (no model calls; numbers here are test fixtures)."""

import json
from pathlib import Path

import pytest

from scripts.analysis import stat_tests


def _record(i: int, a_wins: bool, a_in_slot_a: bool, dataset: str, a: str, b: str) -> dict:
    """One completed comparison; condition a's paper sits in slot A or B."""
    a_label, b_label = f"{dataset}_{a}", f"{dataset}_{b}"
    slot_a, slot_b = (a_label, b_label) if a_in_slot_a else (b_label, a_label)
    slot_a_won = a_wins == a_in_slot_a
    return {
        "status": "completed",
        "score": 1 if slot_a_won else 0,
        "outcome": "A" if slot_a_won else "B",
        "paper_a_id": f"{slot_a}_p{i % 3}",
        "paper_b_id": f"{slot_b}_p{i % 4}",
        "paper_a_experiment": slot_a,
        "paper_b_experiment": slot_b,
    }


def _write(root: Path, dataset: str, a: str, b: str, rubric: str, seed: int, outcomes: list[bool]):
    out = root / "07" / dataset / f"{a}_vs_{b}" / rubric / f"seed{seed}"
    out.mkdir(parents=True)
    records = [_record(i, w, i % 2 == 0, dataset, a, b) for i, w in enumerate(outcomes)]
    records.append({"status": "failed", "paper_a_experiment": "x", "paper_b_experiment": "y"})
    (out / "comparisons.json").write_text(json.dumps(records))


def test_binomial_p_values_match_known_values():
    assert stat_tests.binom_two_sided_p(8, 10) == pytest.approx(0.109375)
    assert stat_tests.binom_two_sided_p(5, 10) == pytest.approx(1.0)
    assert stat_tests.binom_two_sided_p(0, 6) == pytest.approx(2 / 64)
    assert stat_tests.binom_two_sided_p(0, 0) is None


def test_holm_is_monotone_and_ignores_missing():
    assert stat_tests.holm([0.01, None, 0.04, 0.03]) == pytest.approx([0.03, None, 0.06, 0.06])


def test_rank_outputs_become_per_dataset_and_macro_rows(tmp_path):
    root = tmp_path / "outputs" / "ranker"
    _write(root, "kirc", "baseline", "no_reviewers", "default", 1, [True] * 8 + [False] * 2)
    _write(root, "kirc", "baseline", "no_reviewers", "default", 2, [True] * 2 + [False] * 2)
    _write(root, "luad", "baseline", "no_reviewers", "default", 1, [False] * 6)

    rows = stat_tests.compute(root, n_boot=200)
    by_ds = {r["dataset"]: r for r in rows}

    kirc = by_ds["kirc"]
    assert (kirc["experiment"], kirc["condition_a"], kirc["rubric"]) == (
        "07",
        "baseline",
        "default",
    )
    assert (kirc["seeds"], kirc["n"], kirc["wins_a"]) == ("1 2", 14, 10)  # failed records dropped
    assert kirc["win_rate_a"] == pytest.approx(10 / 14)
    assert kirc["binom_p"] == pytest.approx(stat_tests.binom_two_sided_p(10, 14))
    assert kirc["wilson_low"] < kirc["win_rate_a"] < kirc["wilson_high"]
    assert kirc["bootstrap_low"] <= kirc["win_rate_a"] <= kirc["bootstrap_high"]

    assert by_ds["luad"]["win_rate_a"] == 0.0
    assert by_ds["macro"]["win_rate_a"] == pytest.approx((10 / 14 + 0.0) / 2)
    assert by_ds["macro"]["binom_p"] is None

    family = [r["binom_p"] for r in rows]
    assert [r["binom_p_holm"] for r in rows] == pytest.approx(stat_tests.holm(family))


def test_cli_writes_only_to_out(tmp_path):
    root = tmp_path / "ranker"
    _write(root, "kirc", "n1", "n16", "novelty", 1, [True, False, True])
    out = tmp_path / "results"
    before = set(tmp_path.rglob("*"))

    stat_tests.main(["--ranker-dir", str(root), "--out", str(out), "--n-boot", "50"])

    written = set(tmp_path.rglob("*")) - before
    assert written == {out, out / "stat_tests.csv", out / "stat_tests.json"}
    rows = json.loads((out / "stat_tests.json").read_text())
    # "n1" must not be confused with "n16" when resolving which side won.
    assert rows[0]["n"] == 3 and rows[0]["wins_a"] == 2


def test_cli_fails_clearly_without_comparisons(tmp_path):
    with pytest.raises(SystemExit, match="No rank.sh comparisons"):
        stat_tests.main(["--ranker-dir", str(tmp_path), "--out", str(tmp_path / "o")])
