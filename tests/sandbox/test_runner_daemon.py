import io
import pickle
import signal
import time

import pytest

import sandbox.runner_daemon as runner


def test_sniff_imports_returns_unique_non_stdlib_top_level_modules():
    imports = runner._sniff_imports(
        """
import os
import pandas as pd
import numpy.linalg
from sklearn.model_selection import train_test_split
from .local import thing
"""
    )

    assert imports == ["pandas", "numpy", "sklearn"]


def test_history_helpers_round_trip_jsonl(tmp_path, monkeypatch):
    history = tmp_path / "history.jsonl"
    monkeypatch.setattr(runner, "HISTORY_FILE", str(history))

    runner._append_history({"cell_no": 1})
    runner._append_history({"cell_no": 2})

    assert runner._read_history_tail(1) == [{"cell_no": 2}]
    assert runner._read_history_all() == [{"cell_no": 1}, {"cell_no": 2}]


def test_reset_namespace_clears_execution_state(monkeypatch):
    runner.EXEC_G["x"] = 1
    runner.CELL_NO = 4

    runner.reset_namespace()

    assert runner.EXEC_G == {"__name__": "__main__"}
    assert runner.EXEC_L is runner.EXEC_G
    assert runner.CELL_NO == 0


def test_format_user_error_includes_exception_and_source_line():
    code = "x = 1\nraise ValueError('bad')"
    try:
        exec(compile(code, "<cell>", "exec"), {})
    except ValueError as exc:
        formatted = runner._format_user_error(code, exc)
    else:
        pytest.fail("expected ValueError")

    assert "ValueError: bad" in formatted
    assert "raise ValueError('bad')" in formatted


def test_execute_success_records_stdout_and_result(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "HISTORY_FILE", str(tmp_path / "history.jsonl"))
    monkeypatch.setattr(runner, "STATE_FILE", str(tmp_path / "state.pkl"))
    runner.reset_namespace()

    result = runner.execute("print('hello')\n1 + 2", timeout_s=2, enable_auto_install=False)

    assert result.status == "success"
    assert result.stdout.strip() == "hello"
    assert result.result == "3"


def test_auto_install_stops_at_its_budget_and_reports_skips(monkeypatch):
    """The install phase runs outside the execution alarm, so it needs a cap."""
    import subprocess

    slept = {"total": 0.0}

    def slow_pip(cmd, **kwargs):
        # Simulate a pip call that burns most of the budget.
        slept["total"] += 0.2
        monkeypatch.setattr(runner.time, "monotonic", lambda: slept["total"])
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 0))

    monkeypatch.setattr(runner.time, "monotonic", lambda: slept["total"])
    monkeypatch.setattr(subprocess, "run", slow_pip)

    logs = runner._auto_install(
        ["definitely_not_a_real_module_a", "definitely_not_a_real_module_b"], budget_s=0.1
    )

    assert [log["module"] for log in logs] == [
        "definitely_not_a_real_module_a",
        "definitely_not_a_real_module_b",
    ]
    # The first package consumed the whole budget; the second is reported as
    # skipped rather than silently dropped.
    assert logs[0].get("skipped") is not True
    assert logs[1]["skipped"] is True
    assert "budget" in logs[1]["pip_output"]


def test_auto_install_caps_each_package_at_the_remaining_budget(monkeypatch):
    import subprocess

    seen_timeouts = []

    def record(cmd, **kwargs):
        seen_timeouts.append(kwargs.get("timeout"))
        raise RuntimeError("pip unavailable")

    monkeypatch.setattr(subprocess, "run", record)

    runner._auto_install(["definitely_not_a_real_module_a"], budget_s=5.0)

    assert seen_timeouts
    assert 0 < seen_timeouts[0] <= 5.0


# --- State budget ----------------------------------------------------
class _Unserializable:
    """Stands in for an oversized array whose serialization must never start."""

    nbytes = 1 << 40

    def __reduce__(self):
        raise AssertionError("oversized value must be rejected before pickling")


def test_int_env_falls_back_on_unset_unparsable_and_non_positive(monkeypatch):
    monkeypatch.delenv("ASC_TEST_ENV", raising=False)
    assert runner._int_env("ASC_TEST_ENV", 7) == 7

    monkeypatch.setenv("ASC_TEST_ENV", "not-a-number")
    assert runner._int_env("ASC_TEST_ENV", 7) == 7

    monkeypatch.setenv("ASC_TEST_ENV", "0")
    assert runner._int_env("ASC_TEST_ENV", 7) == 7

    monkeypatch.setenv("ASC_TEST_ENV", "123")
    assert runner._int_env("ASC_TEST_ENV", 7) == 123


def _fits(value, budget):
    """Whether `value` is written as a state record within `budget`; nothing is left if not."""
    if runner._skip_unpickled(value, budget):
        return False
    buf = io.BytesIO()
    writer = runner._BudgetedWriter(buf, budget)
    written = runner._dump_within_budget(pickle.Pickler(writer, protocol=5), writer, "v", value)
    assert written or buf.getvalue() == b""
    return written


def test_dump_within_budget_accepts_small_and_rejects_large_values():
    assert _fits(b"x" * 16, 1024) is True
    assert _fits(b"x" * 4096, 1024) is False


def test_dump_within_budget_rejects_nested_oversized_value():
    # The real failure mode: the big array is a dict value, so any size probe
    # that only inspects the top-level object would wave this through.
    assert _fits({"X": b"x" * 4096}, 1024) is False


def test_dump_within_budget_rejects_declared_size_without_serializing():
    # _Unserializable raises if pickling is attempted; only the declared
    # nbytes fast path can answer here.
    assert _fits(_Unserializable(), 1024) is False


def test_dump_within_budget_rejects_unpicklable_value():
    assert _fits(lambda: None, 1 << 20) is False


def test_dump_within_budget_counts_array_buffers():
    numpy = pytest.importorskip("numpy")
    array = numpy.zeros(1 << 20, dtype=numpy.float64)  # 8 MiB

    assert _fits(array, 1 << 20) is False
    assert _fits(array, 32 << 20) is True


class _CountsPickling:
    calls = 0

    def __reduce__(self):
        type(self).calls += 1
        return (_CountsPickling, ())


def test_save_state_pickles_each_value_once(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "STATE_FILE", str(tmp_path / "state.pkl"))
    monkeypatch.setattr(runner, "MAX_STATE_BYTES", 1 << 20)
    runner.reset_namespace()
    _CountsPickling.calls = 0
    runner.EXEC_G["obj"] = _CountsPickling()

    assert runner._save_state() == []
    assert _CountsPickling.calls == 1


def _save_and_restore(tmp_path, monkeypatch, **globals_):
    monkeypatch.setattr(runner, "STATE_FILE", str(tmp_path / "state.pkl"))
    monkeypatch.setattr(runner, "MAX_STATE_BYTES", 1 << 20)
    runner.reset_namespace()
    runner.EXEC_G.update(globals_)
    skipped = runner._save_state()
    runner.reset_namespace()
    runner._restore_state()
    return skipped


def test_restore_keeps_objects_shared_between_globals(tmp_path, monkeypatch):
    a = [1, 2]
    # A module and a cell-defined function in between are unpicklable but must not split the
    # globals into separately pickled groups.
    skipped = _save_and_restore(
        tmp_path, monkeypatch, a=a, mod=io, fn=lambda: None, b=a, d={"k": a}
    )

    assert skipped == ["mod", "fn"]
    g = runner.EXEC_G
    assert g["a"] == [1, 2]
    assert g["b"] is g["a"]
    assert g["d"]["k"] is g["a"]


def test_restore_is_correct_after_a_global_that_fails_part_way(tmp_path, monkeypatch):
    # `bad` gets as far as pickling `inner` before the lambda fails, so the pickler's memo
    # names `inner` although its bytes were cut. Reusing that pickler for `good` would write
    # a reference the reader cannot resolve.
    inner = [3]
    after = {"x": "y"}
    skipped = _save_and_restore(
        tmp_path, monkeypatch, bad=[inner, lambda: None], good=inner, after=after, again=after
    )

    assert skipped == ["bad"]
    g = runner.EXEC_G
    assert g["good"] == [3]
    assert g["after"] == {"x": "y"}
    assert g["again"] is g["after"]


def test_restore_state_reads_the_older_single_dict_format(tmp_path, monkeypatch):
    state_file = tmp_path / "state.pkl"
    monkeypatch.setattr(runner, "STATE_FILE", str(state_file))
    state_file.write_bytes(pickle.dumps({"kept": 1, "_ASC_CELL_NO": 4}))
    runner.reset_namespace()

    runner._restore_state()

    assert runner.EXEC_G["kept"] == 1
    assert runner.CELL_NO == 4


class _SlowToPickle:
    """A global whose pickling takes `seconds`, standing in for a large value."""

    def __init__(self, seconds):
        self.seconds = seconds

    def __reduce__(self):
        time.sleep(self.seconds)
        return (_SlowToPickle, (self.seconds,))


def _last_run_end():
    return [r for r in runner._read_history_all() if r.get("event") == "run_end"][-1]


def test_save_that_outlasts_the_cell_timeout_is_complete(tmp_path, monkeypatch):
    # The old daemon saved under the cell's alarm. The alarm fired inside the save, whose
    # `except Exception` swallowed it: the cell still succeeded, but `slow` was silently
    # reported as skipped (or state.pkl was left truncated).
    state_file = tmp_path / "state.pkl"
    monkeypatch.setattr(runner, "HISTORY_FILE", str(tmp_path / "history.jsonl"))
    monkeypatch.setattr(runner, "STATE_FILE", str(state_file))
    monkeypatch.setattr(runner, "MAX_STATE_BYTES", 1 << 20)
    runner.reset_namespace()
    runner.EXEC_G["slow"] = _SlowToPickle(0.5)
    runner._setup_matplotlib()  # import it now: execute's first import would use up the save's time

    result = runner.execute("x = 1", timeout_s=0.2, enable_auto_install=False)

    assert result.status == "success", result.stderr
    assert "state_skipped" not in _last_run_end()
    persisted = runner._read_state_file(str(state_file))
    assert isinstance(persisted["slow"], _SlowToPickle)
    assert persisted["x"] == 1


def test_state_save_stops_at_its_deadline_and_reports_what_it_left_out(tmp_path, monkeypatch):
    # The server stops waiting at timeout + install allowance + 3 s and restarts the container,
    # so a save that ran on past that lost the whole session.
    state_file = tmp_path / "state.pkl"
    monkeypatch.setattr(runner, "HISTORY_FILE", str(tmp_path / "history.jsonl"))
    monkeypatch.setattr(runner, "STATE_FILE", str(state_file))
    monkeypatch.setattr(runner, "MAX_STATE_BYTES", 1 << 20)
    monkeypatch.setattr(runner, "STATE_SAVE_GRACE_S", 1.0)
    runner.reset_namespace()
    runner.EXEC_G["early"] = [1, 2]
    runner.EXEC_G["slow"] = _SlowToPickle(5.0)
    runner.EXEC_G["late"] = "z"
    runner._setup_matplotlib()  # import it now: execute's first import would use up the save's time

    started = time.monotonic()
    result = runner.execute("pass", timeout_s=0.1, enable_auto_install=False)
    elapsed = time.monotonic() - started

    assert result.status == "success", result.stderr
    assert elapsed < 4.0  # stopped at its 1.1 s deadline, not after the 5 s pickle
    assert _last_run_end()["state_skipped"] == ["slow", "late"]
    persisted = runner._read_state_file(str(state_file))
    assert persisted == {"early": [1, 2], "_ASC_CELL_NO": 1}
    # An alarm handled after the save is over does nothing.
    signal.raise_signal(signal.SIGALRM)


def test_save_state_persists_small_globals_and_skips_oversized_ones(tmp_path, monkeypatch):
    state_file = tmp_path / "state.pkl"
    monkeypatch.setattr(runner, "STATE_FILE", str(state_file))
    monkeypatch.setattr(runner, "MAX_STATE_BYTES", 1024)
    runner.reset_namespace()
    runner.EXEC_G["small"] = {"a": 1}
    runner.EXEC_G["huge"] = {"X": b"x" * 8192}
    runner.CELL_NO = 3

    skipped = runner._save_state()

    assert skipped == ["huge"]
    persisted = runner._read_state_file(str(state_file))
    assert persisted["small"] == {"a": 1}
    assert "huge" not in persisted
    assert persisted["_ASC_CELL_NO"] == 3
    # The oversized value must not reach the file either.
    assert state_file.stat().st_size < 1024


def test_save_state_reports_unpicklable_globals_as_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "STATE_FILE", str(tmp_path / "state.pkl"))
    monkeypatch.setattr(runner, "MAX_STATE_BYTES", 1 << 20)
    runner.reset_namespace()
    runner.EXEC_G["fn"] = lambda: None

    assert runner._save_state() == ["fn"]


def test_save_state_round_trips_through_restore_state(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "STATE_FILE", str(tmp_path / "state.pkl"))
    monkeypatch.setattr(runner, "MAX_STATE_BYTES", 1 << 20)
    runner.reset_namespace()
    runner.EXEC_G["kept"] = [1, 2, 3]
    runner.CELL_NO = 5
    runner._save_state()

    runner.reset_namespace()
    runner._restore_state()

    assert runner.EXEC_G["kept"] == [1, 2, 3]
    assert runner.CELL_NO == 5


def test_execute_records_skipped_globals_in_history(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "HISTORY_FILE", str(tmp_path / "history.jsonl"))
    monkeypatch.setattr(runner, "STATE_FILE", str(tmp_path / "state.pkl"))
    monkeypatch.setattr(runner, "MAX_STATE_BYTES", 1024)
    runner.reset_namespace()

    result = runner.execute("big = b'x' * 8192", timeout_s=5, enable_auto_install=False)

    assert result.status == "success"
    run_end = [r for r in runner._read_history_all() if r.get("event") == "run_end"][-1]
    assert run_end["state_skipped"] == ["big"]


def test_execute_omits_state_skipped_when_everything_persists(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "HISTORY_FILE", str(tmp_path / "history.jsonl"))
    monkeypatch.setattr(runner, "STATE_FILE", str(tmp_path / "state.pkl"))
    monkeypatch.setattr(runner, "MAX_STATE_BYTES", 1 << 20)
    runner.reset_namespace()

    runner.execute("small = 1", timeout_s=5, enable_auto_install=False)

    run_end = [r for r in runner._read_history_all() if r.get("event") == "run_end"][-1]
    assert "state_skipped" not in run_end
