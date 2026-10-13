"""The [signal] diagnostics: each metric once per round, reputation after the round's update."""

import logging
from types import SimpleNamespace

from system.main import _log_signal_diagnostics

METRICS = {
    "paper_1": {"support": {"tool_calls": 1.0}},
    "paper_2": {"support": {"tool_calls": 3.0}},
}
AGENTS = [
    SimpleNamespace(reputation={"reputation_score": 0.2}),
    SimpleNamespace(reputation={"reputation_score": 0.8}),
]


def _signal_lines(caplog):
    return [r.getMessage() for r in caplog.records if "[signal]" in r.getMessage()]


def test_review_call_logs_metrics_only(caplog):
    caplog.set_level(logging.INFO, logger="system.main")
    _log_signal_diagnostics(iteration=1, paper_metrics_by_id=METRICS)
    lines = _signal_lines(caplog)
    assert len(lines) == 1 and "metric=support.tool_calls" in lines[0]


def test_end_of_round_call_logs_reputation_only(caplog):
    caplog.set_level(logging.INFO, logger="system.main")
    _log_signal_diagnostics(
        iteration=1, paper_metrics_by_id=METRICS, agents=AGENTS, include_metrics=False
    )
    lines = _signal_lines(caplog)
    assert len(lines) == 1 and "reputation mean=0.5000" in lines[0]
