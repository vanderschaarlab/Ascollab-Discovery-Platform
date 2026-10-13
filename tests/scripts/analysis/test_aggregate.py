"""Win-rate aggregation tests against synthetic comparisons.json records."""

from scripts.analysis import aggregate


def _rec(
    status, score, a_exp, b_exp, a_id="pa", b_id="pb", a_snap="run_seed_1", b_snap="run_seed_1"
):
    return {
        "status": status,
        "score": score,
        "paper_a_experiment": a_exp,
        "paper_b_experiment": b_exp,
        "paper_a_id": a_id,
        "paper_b_id": b_id,
        "paper_a_source_snapshot": a_snap,
        "paper_b_source_snapshot": b_snap,
    }


def test_ab_inversion_and_failed_rows_dropped():
    records = [
        # ascollab in slot A wins (score 1)
        _rec("completed", 1, "ascollab", "ablation", a_id="a1"),
        _rec("completed", 1, "ascollab", "ablation", a_id="a2"),
        # ascollab in slot B wins (score 0)
        _rec("completed", 0, "ablation", "ascollab", a_id="x1", b_id="a3"),
        # ascollab in slot A loses (score 0)
        _rec("completed", 0, "ascollab", "ablation", a_id="a4"),
        # failed row is dropped
        _rec("failed", None, "ascollab", "ablation"),
    ]
    pt = aggregate.point_win_rate(records, "ascollab", "ablation")
    assert pt["n"] == 4
    assert pt["wins"] == 3
    assert pt["win_rate"] == 0.75
    # Reverse direction is the complement over completed rows.
    rev = aggregate.point_win_rate(records, "ablation", "ascollab")
    assert rev["wins"] == 1


def test_label_resolution_by_experiment_fields():
    rec = _rec("completed", 1, "norm_a", "norm_b")
    assert aggregate.resolve_condition_a_win(rec, "norm_a", "norm_b") is True
    assert aggregate.resolve_condition_a_win(rec, "norm_b", "norm_a") is False


def test_macro_average_not_equal_to_pooled():
    d_small = [_rec("completed", 1, "a", "b", a_id="s0")]  # rate 1.0, n=1
    d_big = [_rec("completed", 0, "a", "b", a_id=f"b{i}") for i in range(9)]  # rate 0.0, n=9
    macro = aggregate.macro_average(
        {"d_small": d_small, "d_big": d_big}, "a", "b", n_boot=50, seed=1
    )
    # Macro-average of per-dataset rates == mean(1.0, 0.0) == 0.5
    assert abs(macro["win_rate"] - 0.5) < 1e-9
    # Pooled would be 1/10 == 0.1, which macro must NOT equal.
    assert abs(macro["win_rate"] - 0.1) > 0.3
    assert set(macro["per_dataset"]) == {"d_small", "d_big"}


def test_cluster_bootstrap_ci_bounds_and_determinism():
    records = [
        _rec("completed", 1 if i % 2 == 0 else 0, "a", "b", a_id=f"p{i % 4}") for i in range(40)
    ]
    lo1, hi1 = aggregate.cluster_bootstrap_ci(records, "a", "b", n_boot=200, seed=7)
    lo2, hi2 = aggregate.cluster_bootstrap_ci(records, "a", "b", n_boot=200, seed=7)
    assert (lo1, hi1) == (lo2, hi2)  # deterministic under fixed seed
    assert 0.0 <= lo1 <= hi1 <= 1.0
    pt = aggregate.point_win_rate(records, "a", "b")
    assert lo1 <= pt["win_rate"] <= hi1


def test_one_sided_and_empty():
    one_sided = [_rec("completed", 1, "a", "b", a_id=f"p{i}") for i in range(5)]
    assert aggregate.point_win_rate(one_sided, "a", "b")["win_rate"] == 1.0
    empty = aggregate.point_win_rate([], "a", "b")
    assert empty["n"] == 0
    assert empty["win_rate"] is None


def test_per_seed_breakdown():
    records = [
        _rec("completed", 1, "a", "b", a_id="p1", a_snap="run_seed_1"),
        _rec("completed", 0, "a", "b", a_id="p2", a_snap="run_seed_2"),
    ]
    ds = aggregate.aggregate_dataset(records, "a", "b", n_boot=20, seed=1)
    assert set(ds["per_seed"]) == {"1", "2"}
    assert ds["per_seed"]["1"]["win_rate"] == 1.0
    assert ds["per_seed"]["2"]["win_rate"] == 0.0


def test_papers_from_different_seeds_are_separate_clusters():
    # The same paper ID in two seed runs is two papers: one won, the other lost. As one
    # cluster every resample would give exactly 0.5; as two, resamples also give 0 and 1.
    records = [
        _rec("completed", 1, "a", "b", a_id="paper_t01_agent_01", a_snap="run_seed_1"),
        _rec("completed", 0, "a", "b", a_id="paper_t01_agent_01", a_snap="run_seed_2"),
    ]
    lo, hi = aggregate.cluster_bootstrap_ci(records, "a", "b", n_boot=200, seed=3)
    assert (lo, hi) == (0.0, 1.0)
    macro = aggregate.macro_average({"kirc": records}, "a", "b", n_boot=200, seed=3)
    assert (macro["ci_low"], macro["ci_high"]) == (0.0, 1.0)


def test_cluster_key_prefers_the_source_snapshot_dir():
    rec = _rec("completed", 0, "b", "a", b_id="p7", b_snap="seed_1")
    rec["paper_b_source_snapshot_dir"] = "snapshots/run_seed_1_2401"
    assert aggregate._condition_a_paper_id(rec, "a", "b") == "snapshots/run_seed_1_2401::p7"
