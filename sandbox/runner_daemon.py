import ast
import base64
import io
import json
import os
import pickle
import signal
import sys
import time
import traceback
import types
from dataclasses import dataclass
from datetime import datetime
from typing import Any

WORK_DIR = os.environ.get("SANDBOX_RUNNER_WORKDIR", "/work")
INBOX = os.path.join(WORK_DIR, "inbox")
OUTBOX = os.path.join(WORK_DIR, "outbox")
STATE_FILE = os.path.join(WORK_DIR, "state.pkl")
HISTORY_DIR = os.path.join(WORK_DIR, ".history")
HISTORY_FILE = os.path.join(HISTORY_DIR, "history.jsonl")


def _int_env(name: str, default: int) -> int:
    """Read a positive integer environment variable.

    Args:
        name: Environment variable to read.
        default: Value used when unset, unparsable, or not positive.

    Returns:
        int: The configured value, or `default`.
    """
    try:
        value = int(os.environ.get(name, ""))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


# Per-global ceiling on persisted execution state. Round-tripping a value through
# pickle costs several times its own size in transient memory, so a single large
# in-memory dataset would otherwise be re-serialized on every cell.
MAX_STATE_BYTES = _int_env("SANDBOX_MAX_STATE_BYTES", 256 * 1024 * 1024)

for _path in (INBOX, OUTBOX, HISTORY_DIR):
    try:
        os.makedirs(_path, exist_ok=True)
    except Exception:
        pass

EXEC_G = {"__name__": "__main__"}
EXEC_L = EXEC_G
CELL_NO = 0

# The dependency-install phase runs BEFORE the execution alarm is armed, so it
# needs its own budget or a slow install looks like a code timeout to the server.
# The server sends the real value per job via the `install_budget` request key.
DEFAULT_INSTALL_BUDGET_S = 180.0
INSTALL_PER_PACKAGE_TIMEOUT_S = 120.0

# The server waits for a job's result until timeout + install allowance + 3 s after it wrote the
# job, then restarts the container and the session's variables are lost (server_docker.py,
# CodeRunner.run). The state save gets a deadline on the same scale, counted from when the
# daemon starts the job: install allowance + timeout + STATE_SAVE_GRACE_S. So the save ends at
# least 3 - 1.5 = 1.5 s before the server gives up, less the job pick-up delay (one 50 ms poll),
# which leaves time to write the result. The deadline is fixed when the job starts, so a slow
# install phase, cell or figure capture shortens the save instead of pushing it later.
STATE_SAVE_GRACE_S = 1.5


# --- History helpers -------------------------------------------------
def _now_iso() -> str:
    """Return current UTC time in ISO8601 format."""
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _append_history(rec: dict) -> None:
    """Append one JSON record to the per-session history file."""
    try:
        with open(HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _read_history_tail(n: int = 50) -> list[dict]:
    """Read the last n entries from the history file."""
    try:
        with open(HISTORY_FILE, encoding="utf-8") as f:
            lines = f.readlines()
        return [json.loads(x) for x in lines[-n:]]
    except Exception:
        return []


def _read_history_all() -> list[dict]:
    """Read all entries from the history file."""
    try:
        with open(HISTORY_FILE, encoding="utf-8") as f:
            return [json.loads(x) for x in f]
    except Exception:
        return []


def reset_namespace():
    """Reset execution globals and cell counter for a fresh session."""
    global EXEC_G, EXEC_L, CELL_NO
    EXEC_G = {"__name__": "__main__"}
    EXEC_L = EXEC_G
    CELL_NO = 0


def _log(msg: str) -> None:
    print(f"[runner] {msg}", file=sys.stderr, flush=True)


def _format_user_error(code: str, exc: BaseException) -> str:
    """Format user-facing error with source snippet and compact traceback."""
    etype, evalue, tb = sys.exc_info()
    if exc is not None and (etype is None or evalue is None or tb is None):
        etype = type(exc)
        evalue = exc
        tb = exc.__traceback__

    frames = traceback.extract_tb(tb) if tb else []
    user_frames = [f for f in frames if f.filename == "<cell>"]
    last = (user_frames or frames)[-1] if (user_frames or frames) else None

    offending = ""
    if last is not None:
        try:
            src_line = (last.line or "").strip()
            if not src_line and last.lineno and code:
                lines = code.splitlines()
                if 1 <= last.lineno <= len(lines):
                    src_line = lines[last.lineno - 1].strip()
            line_note = f"(line {last.lineno})" if last.lineno else ""
            if src_line:
                offending = f"\n--> {src_line}"
            loc = f" {line_note}" if line_note else ""
        except Exception:
            loc = ""
    else:
        loc = ""

    head = f"{etype.__name__}: {evalue}{loc}"

    tail = ""
    try:
        raw = traceback.format_exception(etype, evalue, tb)
        raw_short = "".join(raw[-8:])
        tail = f"\n\n--- raw traceback (internal) ---\n{raw_short}".rstrip()
    except Exception:
        pass

    return (head + offending + tail).strip()


# state.pkl is a sequence of sections. A section is this header, then pickled (name, value)
# records written by one Pickler: the pickle memo spans the section, so an object that several
# globals refer to is stored once and restored as one object. See `_save_state`.
_STATE_FORMAT = "ascollab-sandbox-state/3"


def _read_state_file(path: str) -> dict[str, Any]:
    """Read persisted globals; a record that fails to load ends the read.

    The records before the failing one are kept; it and every record after it are dropped.
    Each section is read by one Unpickler, as it was written by one Pickler.

    Args:
        path: State file written by `_save_state` (or a single pickled dict, the older format).

    Returns:
        dict[str, Any]: Restored globals (none for an unknown format).
    """
    restored: dict[str, Any] = {}
    with open(path, "rb") as f:
        first = pickle.load(f)
        if isinstance(first, dict):
            return first
        if not (isinstance(first, str) and first == _STATE_FORMAT):
            return restored
        unpickler = pickle.Unpickler(f)
        while True:
            try:
                record = unpickler.load()
                if isinstance(record, str):  # the next section's header
                    unpickler = pickle.Unpickler(f)
                    continue
                name, value = record
            except Exception:  # EOFError at the end of the file, or a corrupt record
                break
            restored[name] = value
    return restored


def _restore_state():
    """Restore persisted globals and cell counter from disk."""
    global CELL_NO
    if os.path.exists(STATE_FILE):
        try:
            EXEC_G.update(_read_state_file(STATE_FILE))
        except Exception:
            pass
    # recover CELL_NO if present in globals
    try:
        CELL_NO = int(EXEC_G.get("_ASC_CELL_NO", 0))
    except Exception:
        CELL_NO = 0


class _BudgetExceeded(Exception):
    """Signals that a value has been shown to exceed the state budget."""


class _BudgetedWriter:
    """Pass-through file wrapper that aborts once one record passes a byte budget.

    Attributes:
        file: Underlying binary file object.
        budget: Maximum number of bytes tolerated before aborting.
        total: Bytes written for the current record (reset by `_dump_within_budget`).
    """

    def __init__(self, f: Any, budget: int) -> None:
        """Initialize the writer.

        Args:
            f: Underlying binary file object.
            budget: Maximum number of bytes tolerated before aborting.
        """
        self.file = f
        self.budget = budget
        self.total = 0

    def write(self, chunk: Any) -> int:
        """Write a pickle chunk, aborting before the budget is exceeded.

        Args:
            chunk: Bytes-like object emitted by the pickler (large array data arrives as a
                `pickle.PickleBuffer`, which has no ``len``).

        Returns:
            int: Number of bytes written.
        """
        self.total += memoryview(chunk).nbytes
        if self.total > self.budget:
            raise _BudgetExceeded
        return self.file.write(chunk)


def _declared_nbytes(value: Any) -> int | None:
    """Read a value's self-reported buffer size, without serializing it.

    Covers the heavyweight array-likes (numpy arrays, pandas objects) whose size
    is known up front, so the common case is rejected before any pickling.

    Args:
        value: Object to size.

    Returns:
        int | None: Reported size in bytes, or None when the object does not
        expose one.
    """
    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, int):
        return nbytes

    memory_usage = getattr(value, "memory_usage", None)
    if callable(memory_usage):
        try:
            usage = memory_usage(deep=True)
            total = getattr(usage, "sum", None)
            return int(total()) if callable(total) else int(usage)
        except Exception:
            return None
    return None


class _SaveDeadline(BaseException):
    """Raised by the save alarm inside a record's pickling when the save runs out of time.

    A BaseException, so a value's own pickling code cannot swallow it with ``except Exception``.
    """


# True only while a record is being pickled under the save alarm. Python may run a signal's
# handler a little after the signal arrives; the handler checks this flag, so an alarm handled
# after the record's try block has ended does nothing.
_SAVE_ALARM_ARMED = False


def _save_alarm_handler(_signum: int, _frame: Any) -> None:
    """SIGALRM handler during the state save: interrupt the record being pickled.

    Raises:
        _SaveDeadline: While a record is being pickled.
    """
    if _SAVE_ALARM_ARMED:
        raise _SaveDeadline


def _skip_unpickled(value: Any, budget: int) -> bool:
    """Whether `value` can be left out of the state without going through the state pickler.

    True for a value whose declared buffer size (numpy, pandas) is over `budget`, and for a
    module, class or function that cannot be pickled. These pickle by name if at all, so trying
    one on its own costs nothing; they are the usual unpicklable globals (modules, and functions
    and classes defined in a cell), and catching them here keeps them from ending a section.

    Args:
        value: Object to persist.
        budget: Maximum serialized size in bytes.

    Returns:
        bool: True when the value is to be left out.
    """
    declared = _declared_nbytes(value)
    if declared is not None and declared > budget:
        return True
    if isinstance(value, (types.ModuleType, type, types.FunctionType)):
        try:
            pickle.dumps(value, protocol=5)
        except Exception:
            return True
    return False


def _dump_within_budget(
    pickler: pickle.Pickler,
    writer: _BudgetedWriter,
    name: str,
    value: Any,
    deadline: float | None = None,
) -> bool:
    """Append one ``(name, value)`` record through `pickler` if it fits the budget and is in time.

    The value is pickled once, straight into the file. Pickling stops as soon as the record
    passes `writer.budget`, and the partial record is truncated away. With a `deadline`, SIGALRM
    interrupts the pickling when it is reached (`_save_state` installs the handler), and the
    partial record is truncated away too. After a failure the pickler's memo may name objects
    whose bytes were cut, so the caller must not use the pickler again.

    Args:
        pickler: Pickler writing to `writer`, shared by the records of one section.
        writer: Budgeted writer around a seekable binary file positioned at the end of the
            previous record.
        name: Global name.
        value: Object to persist.
        deadline: `time.monotonic()` value at which to stop pickling, or None for no limit.

    Returns:
        bool: True when the record was written; False when the value was unpicklable, over
        budget or not pickled by the deadline (the file is then unchanged).
    """
    global _SAVE_ALARM_ARMED
    start = writer.file.tell()
    writer.total = 0
    written = False
    try:
        if deadline is not None:
            _SAVE_ALARM_ARMED = True
            signal.setitimer(signal.ITIMER_REAL, max(deadline - time.monotonic(), 0.001))
        pickler.dump((name, value))
        written = True
    except (Exception, _SaveDeadline):
        pass
    finally:
        if deadline is not None:
            _SAVE_ALARM_ARMED = False
            signal.setitimer(signal.ITIMER_REAL, 0)
    if not written:
        writer.file.seek(start)
        writer.file.truncate()
    return written


def _start_section(f: Any, writer: _BudgetedWriter) -> pickle.Pickler:
    """Write a section header to `f` and return a fresh pickler for the section's records.

    Args:
        f: Binary file being written.
        writer: Budgeted writer around `f`.

    Returns:
        pickle.Pickler: Pickler with an empty memo, writing through `writer`.
    """
    pickle.dump(_STATE_FORMAT, f)
    return pickle.Pickler(writer, protocol=5)


def _save_state(deadline: float | None = None) -> list[str]:
    """Persist picklable globals and the cell counter to disk.

    Each global is pickled once, directly into a temporary file that then replaces the
    state file, so a failed save never leaves a partial state. Globals that are
    unpicklable, or whose serialized form would exceed `MAX_STATE_BYTES`, are left out. So are
    the global being pickled when `deadline` is reached and all globals after it; the state file
    then holds the globals saved before it and the cell counter.

    The globals are pickled by one Pickler, so objects shared between them (``b = a``, or a
    frame held in a dict and in a variable) are stored once and restored shared. A global that
    fails part-way may leave that pickler's memo naming objects whose bytes were cut, so the next
    global starts a new section with a fresh pickler; objects shared across that boundary are
    restored as copies. Values left out by `_skip_unpickled` never touch the pickler.

    Args:
        deadline: `time.monotonic()` value by which the save must end, or None for no limit.
            With a deadline, this installs its own SIGALRM handler (`execute` installs the cell's
            handler again before each cell).

    Returns:
        list[str]: Names of the globals that were not persisted.
    """
    skipped: list[str] = []
    tmp_path = STATE_FILE + ".tmp"
    if deadline is not None:
        signal.signal(signal.SIGALRM, _save_alarm_handler)
    try:
        with open(tmp_path, "wb") as f:
            writer = _BudgetedWriter(f, MAX_STATE_BYTES)
            pickler: pickle.Pickler | None = None
            for k, v in list(EXEC_G.items()):
                if k.startswith("__"):
                    continue
                out_of_time = deadline is not None and time.monotonic() >= deadline
                if out_of_time or _skip_unpickled(v, MAX_STATE_BYTES):
                    skipped.append(k)
                    continue
                if pickler is None:
                    pickler = _start_section(f, writer)
                if not _dump_within_budget(pickler, writer, k, v, deadline):
                    skipped.append(k)
                    pickler = None
            if pickler is None:
                pickler = _start_section(f, writer)
            _dump_within_budget(pickler, writer, "_ASC_CELL_NO", CELL_NO)
        os.replace(tmp_path, STATE_FILE)
    except (Exception, _SaveDeadline):
        # A late alarm outside a record (see `_SAVE_ALARM_ARMED`) must not escape `execute`.
        pass
    finally:
        if deadline is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
    return skipped


@dataclass
class ExecResult:
    """Execution envelope for a single code cell."""

    result: str | None
    stdout: str | None
    stderr: str | None
    images: list[str]
    status: str
    execution_time: float
    pre_install: list[dict[str, Any]] | None


class TimeoutError_(Exception):
    pass


def _timeout_handler(_signum, _frame):
    raise TimeoutError_("Execution timed out")


def _setup_matplotlib():
    """Configure matplotlib to use the non-interactive Agg backend."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.close("all")
    except Exception:
        pass


def _capture_images_b64() -> list[str]:
    """Capture any open matplotlib figures as base64 PNG strings."""
    imgs = []
    try:
        import matplotlib.pyplot as plt

        figs = [plt.figure(i) for i in plt.get_fignums()]
        for fig in figs:
            bio = io.BytesIO()
            fig.savefig(bio, format="png", dpi=100, bbox_inches="tight")
            bio.seek(0)
            imgs.append(base64.b64encode(bio.getvalue()).decode("ascii"))
            plt.close(fig)
    except Exception:
        pass
    return imgs


def _auto_install(
    mods: list[str], *, budget_s: float = DEFAULT_INSTALL_BUDGET_S
) -> list[dict[str, Any]]:
    """Best-effort installation of missing imports, bounded by a total budget.

    Args:
        mods: Module names sniffed from the submitted code.
        budget_s: Total wall-clock cap for the whole phase. Packages not reached
            before it expires are logged as skipped rather than silently dropped,
            so the caller can tell an install problem from a code problem.

    Returns:
        One log dict per module describing whether it was installed or skipped.
    """
    logs: list[dict[str, Any]] = []
    if not mods:
        return logs
    try:
        import importlib
        import subprocess

        install_deadline = time.monotonic() + max(0.0, float(budget_s))

        for m in mods:
            remaining = install_deadline - time.monotonic()
            if remaining <= 0:
                logs.append(
                    {
                        "module": m,
                        "package": m,
                        "installed": False,
                        "skipped": True,
                        "pip_output": f"skipped: auto-install budget of {budget_s:.0f}s exhausted",
                    }
                )
                continue
            ok = False
            pip_out = ""
            try:
                importlib.import_module(m)
                ok = True
            except Exception:
                try:
                    proc = subprocess.run(
                        [sys.executable, "-m", "pip", "install", m],
                        capture_output=True,
                        text=True,
                        timeout=min(INSTALL_PER_PACKAGE_TIMEOUT_S, remaining),
                    )
                    pip_out = (proc.stdout or "") + (proc.stderr or "")
                    importlib.import_module(m)
                    ok = True
                except Exception as e:
                    pip_out = pip_out or str(e)
            logs.append({"module": m, "package": m, "installed": ok, "pip_output": pip_out})
        return logs
    except Exception as e:
        logs.append(
            {
                "module": "unknown",
                "package": "unknown",
                "installed": False,
                "pip_output": str(e),
            }
        )
        return logs


def _sniff_imports(code: str) -> list[str]:
    """Parse code to detect top-level import module names for auto-install."""
    import ast

    skip = {
        "sys",
        "os",
        "json",
        "time",
        "re",
        "math",
        "random",
        "subprocess",
        "pathlib",
        "itertools",
        "functools",
        "collections",
        "typing",
        "dataclasses",
        "statistics",
        "asyncio",
        "threading",
        "multiprocessing",
        "tempfile",
        "shutil",
        "logging",
    }
    try:
        tree = ast.parse(code)
    except Exception:
        return []
    mods: list[str] = []

    class C(ast.NodeVisitor):
        def visit_Import(self, node: ast.Import) -> None:
            for a in node.names:
                r = a.name.split(".")[0]
                if r and r not in mods:
                    mods.append(r)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            if node.level and node.level > 0:
                return
            if node.module:
                r = node.module.split(".")[0]
                if r and r not in mods:
                    mods.append(r)

    C().visit(tree)
    return [m for m in mods if m not in skip]


def execute(
    code: str,
    timeout_s: float,
    enable_auto_install: bool,
    install_budget_s: float = DEFAULT_INSTALL_BUDGET_S,
) -> ExecResult:
    """Execute Python code in a notebook-like REPL cell.

    Args:
        code: Source code to run.
        timeout_s: Wall-clock timeout in seconds, applied to execution only.
        enable_auto_install: Whether to auto-install missing imports.
        install_budget_s: Total wall-clock cap for the dependency-install phase,
            which runs BEFORE the execution alarm is armed. Supplied by the server
            so both sides agree on the job's total budget.

    Returns:
        ExecResult capturing result string, stdout/stderr, images, status, timing, and install logs.
    """
    global CELL_NO
    CELL_NO += 1
    # The state save must end before the server stops waiting for this job; see
    # STATE_SAVE_GRACE_S. The server grants the install allowance only with auto-install.
    install_allowance = install_budget_s if enable_auto_install else 0.0
    save_deadline = (
        time.monotonic() + max(0.0, install_allowance) + max(0.0, timeout_s) + STATE_SAVE_GRACE_S
    )
    _setup_matplotlib()

    pre_install: list[dict[str, Any]] | None = None
    if enable_auto_install:
        pre_install = _auto_install(_sniff_imports(code), budget_s=install_budget_s)

    # record "before" entry
    _append_history(
        {
            "ts": _now_iso(),
            "cell_no": CELL_NO,
            "event": "run_start",
            "code": code,
            "reset": False,
            "auto_install": bool(enable_auto_install),
        }
    )

    old_stdout, old_stderr = sys.stdout, sys.stderr
    stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
    sys.stdout, sys.stderr = stdout_buf, stderr_buf

    start = time.perf_counter()
    last_repr: str | None = None
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, max(0.0, timeout_s))

    try:
        node = ast.parse(code, mode="exec")
        # If the last statement is an expression, eval it to produce a result (REPL-like)
        if node.body and isinstance(node.body[-1], ast.Expr):
            last_expr = ast.Expression(node.body[-1].value)
            if len(node.body) > 1:
                prefix = ast.Module(body=node.body[:-1], type_ignores=[])
                exec(compile(prefix, "<cell>", "exec"), EXEC_G, EXEC_L)
            value = eval(compile(last_expr, "<cell>", "eval"), EXEC_G, EXEC_L)
            try:
                last_repr = repr(value)
            except Exception:
                last_repr = f"<unrepr: {type(value).__name__}>"
        else:
            exec(compile(node, "<cell>", "exec"), EXEC_G, EXEC_L)

        # The cell's code has finished: stop its alarm, so that capturing figures and
        # saving state can never be reported as a timeout of the user's code. The save
        # has its own deadline instead; globals it does not reach are listed as skipped.
        signal.setitimer(signal.ITIMER_REAL, 0)
        images = _capture_images_b64()
        elapsed = time.perf_counter() - start
        skipped_globals = _save_state(deadline=save_deadline)

        _append_history(
            {
                "ts": _now_iso(),
                "cell_no": CELL_NO,
                "event": "run_end",
                "status": "success",
                "stdout_len": len(stdout_buf.getvalue() or ""),
                "stderr_len": len(stderr_buf.getvalue() or ""),
                "result_preview": (last_repr[:200] if last_repr else None),
                **({"state_skipped": skipped_globals} if skipped_globals else {}),
            }
        )

        # attach a small introspection proving stickiness
        EXEC_G_keys = list(EXEC_G.keys())
        introspection = {
            "cell_no": CELL_NO,
            "globals_count": len(EXEC_G_keys),
            "has_pd": "pd" in EXEC_G,
            "has_gene_expression_data": "gene_expression_data" in EXEC_G,
            "history_file": HISTORY_FILE,
        }
        # stash it into pre_install last element as a convenience (or return separately)
        if pre_install is None:
            pre_install = []
        pre_install.append({"introspection": introspection})

        return ExecResult(
            result=last_repr,
            stdout=stdout_buf.getvalue() or None,
            stderr=stderr_buf.getvalue() or None,
            images=images,
            status="success",
            execution_time=elapsed,
            pre_install=pre_install,
        )
    except TimeoutError_:
        images = _capture_images_b64()
        _append_history(
            {
                "ts": _now_iso(),
                "cell_no": CELL_NO,
                "event": "run_end",
                "status": "timeout",
            }
        )
        return ExecResult(
            result=None,
            stdout=stdout_buf.getvalue() or None,
            stderr=f"Execution timed out after {int(timeout_s)}s.",
            images=images,
            status="error",
            execution_time=time.perf_counter() - start,
            pre_install=pre_install,
        )
    except Exception as exc:
        images = _capture_images_b64()
        err = _format_user_error(code, exc)
        # record failure
        _append_history(
            {
                "ts": _now_iso(),
                "cell_no": CELL_NO,
                "event": "run_end",
                "status": "error",
                "error_head": err.split("\n--- raw traceback", 1)[0][:500],
            }
        )
        return ExecResult(
            result=None,
            stdout=stdout_buf.getvalue() or None,
            stderr=err,
            images=images,
            status="error",
            execution_time=time.perf_counter() - start,
            pre_install=pre_install,
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        sys.stdout, sys.stderr = old_stdout, old_stderr


def _write_json(path: str, obj: dict[str, Any]) -> None:
    """Atomically write a JSON document to disk."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def main():
    """Main loop: poll inbox for jobs, execute, and write results to outbox."""
    _restore_state()
    while True:
        try:
            jobs = sorted(fn for fn in os.listdir(INBOX) if fn.endswith(".json"))
            for j in jobs:
                in_path = os.path.join(INBOX, j)
                with open(in_path, encoding="utf-8") as f:
                    req = json.load(f)
                code = req.get("code", "")
                timeout_s = float(req.get("timeout", 120))
                enable_auto_install = bool(req.get("auto_install", True))
                install_budget_s = float(req.get("install_budget", DEFAULT_INSTALL_BUDGET_S))
                shutdown_after = bool(req.get("shutdown_after", False))
                reset = bool(req.get("reset", False))

                _log(f"executing job {j} (shutdown_after={shutdown_after})")

                if reset:
                    reset_namespace()
                    _append_history(
                        {
                            "ts": _now_iso(),
                            "cell_no": CELL_NO,
                            "event": "reset",
                        }
                    )
                result = execute(code, timeout_s, enable_auto_install, install_budget_s)
                # Build an introspection object for the response (defensive).
                # The authoritative run_start/run_end were already written in execute().
                introspection = {
                    "ts": _now_iso(),
                    "cell_no": CELL_NO,
                    "history_file": HISTORY_FILE,
                }
                out_doc = {
                    "result": result.result,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "status": result.status,
                    "execution_time": result.execution_time,
                    "images": [{"type": "png", "data": b64} for b64 in result.images]
                    if result.images
                    else None,
                    "session_metadata": {
                        "created": req.get("created", "unknown"),
                        "lastModified": _now_iso(),
                        "packages": [],
                    },
                    "session_bytes": None,
                    "pre_install": result.pre_install,
                    "job_id": j,
                    "introspection": introspection,
                }

                # Always attach a tail; attach full on error
                out_doc["history_tail"] = _read_history_tail(50)
                if result.status == "error":
                    out_doc["history_full"] = _read_history_all()

                out_path = os.path.join(OUTBOX, j)
                _write_json(out_path, out_doc)
                os.remove(in_path)

                if shutdown_after:
                    _log("shutdown_after=True → exiting container now.")
                    # make sure the json is flushed to disk, then exit
                    sys.stderr.flush()
                    sys.stdout.flush()
                    os._exit(0)
        except Exception:
            traceback.print_exc()
            time.sleep(0.5)
        time.sleep(0.05)


if __name__ == "__main__":
    main()
