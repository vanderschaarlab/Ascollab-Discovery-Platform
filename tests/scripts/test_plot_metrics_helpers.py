from scripts.tools.plot_metrics import _flatten_metrics, _get_paper_status, pareto_front


def test_flatten_metrics_keeps_numeric_leaf_values():
    flat = _flatten_metrics({"support": {"lines_of_code": 3, "label": "x"}})

    assert flat == {"support.lines_of_code": 3.0}


def test_get_paper_status_prefers_private_archive_status():
    status = _get_paper_status(
        {"paper_id": "p1", "iteration": 2},
        {2: {"papers": {"p1": {"status": "accepted"}}}},
    )

    assert status == "accepted"
    assert _get_paper_status({"paper_id": "missing", "iteration": 2}, {}) is None


def test_pareto_front_returns_non_dominated_points_sorted():
    points = [(1, 1), (2, 0), (0, 2), (1, 3), (3, 1)]

    assert pareto_front(points) == [(3, 1), (1, 3)]
