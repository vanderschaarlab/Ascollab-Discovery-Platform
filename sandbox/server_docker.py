# sandbox/server_docker.py
import base64
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import docker
from flask import Flask, jsonify, request
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent  # this is sandbox/


ISO = "%Y-%m-%dT%H:%M:%SZ"


def now_iso() -> str:
    return datetime.now(UTC).strftime(ISO)


# ---------------- Config ----------------
class Settings(BaseSettings):
    """Configuration for the sandbox server and container constraints."""

    # storage for per-session /work dirs
    sessions_root: str = Field(default="/tmp/sandbox_sessions", alias="SANDBOX_SESSIONS_ROOT")
    # group sessions under a run id (optional)
    run_id: str | None = Field(
        default_factory=lambda: datetime.now(UTC).strftime("%Y-%m-%d"),
        alias="SANDBOX_RUN_ID",
    )
    # docker image
    image: str = Field(default="agent-sandbox:py311", alias="SANDBOX_IMAGE")
    # API base path
    base_prefix: str = Field(default="/base", alias="SANDBOX_BASE_PREFIX")

    # session limits & timeouts
    max_sessions: int = Field(default=40, alias="SANDBOX_MAX_SESSIONS")
    default_timeout: float = Field(default=120.0, alias="SANDBOX_DEFAULT_TIMEOUT")
    # Extra wall-clock allowance for the daemon's dependency-install phase, which
    # runs before the execution alarm is armed. Only granted when an install can
    # actually happen (see CodeRunner.run).
    install_budget: float = Field(default=180.0, alias="SANDBOX_INSTALL_BUDGET_S")

    # container hardening
    cpus: float = Field(default=2.0, alias="SANDBOX_CPUS")
    memory: str = Field(default="6g", alias="SANDBOX_MEMORY")
    pids_limit: int = Field(default=256, alias="SANDBOX_PIDS_LIMIT")
    read_only_rootfs: bool = Field(default=True, alias="SANDBOX_READ_ONLY_ROOTFS")
    no_new_privileges: bool = Field(default=True, alias="SANDBOX_NO_NEW_PRIVILEGES")
    network_disabled: bool = Field(default=True, alias="SANDBOX_NETWORK_DISABLED")
    user_uid: int = Field(default=1000, alias="SANDBOX_USER_UID")
    user_gid: int = Field(default=999, alias="SANDBOX_USER_GID")

    # per-global ceiling on execution state the runner persists between cells
    max_state_bytes: int = Field(default=256 * 1024 * 1024, alias="SANDBOX_MAX_STATE_BYTES")

    # optional dataset mount (this matches your repo layout idea)
    datasets_dir: str = Field(
        default=str(BASE_DIR / "datasets"),
        alias="SANDBOX_DATASETS_DIR",
    )

    model_config = SettingsConfigDict(
        env_prefix="SANDBOX_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @property
    def effective_root(self) -> str:
        return os.path.join(self.sessions_root, self.run_id) if self.run_id else self.sessions_root


# --------------- Data -------------------
@dataclass
class Session:
    """Record representing a running sandbox session."""

    session_id: str
    created: str
    last_modified: str
    host_workdir: str
    container_id: str
    meta: dict[str, Any] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    busy_since: str | None = None
    busy_started_monotonic: float | None = None
    current_job_id: str | None = None
    current_code_head: str | None = None
    current_timeout: float | None = None
    current_total_budget: float | None = None
    busy_rejects: int = 0
    creating: bool = False
    closing: bool = False
    restarting: bool = False
    closed: bool = False
    slot_released: bool = False

    def metadata(self) -> dict[str, Any]:
        """Return a metadata dict suitable for API responses."""
        meta = {
            "created": self.created,
            "lastModified": self.last_modified,
            "packages": [],
            **self.meta,
        }
        if self.busy_since is not None:
            meta.update(
                {
                    "busy_since": self.busy_since,
                    "busy_seconds": self.busy_seconds(),
                    "current_job_id": self.current_job_id,
                    "current_code_head": self.current_code_head,
                    "current_timeout": self.current_timeout,
                    "current_total_budget": self.current_total_budget,
                    "busy_rejects": self.busy_rejects,
                }
            )
        if self.creating:
            meta["creating"] = True
        if self.closing:
            meta["closing"] = True
        if self.restarting:
            meta["restarting"] = True
        if self.closed:
            meta["closed"] = True
        return meta

    def busy_seconds(self) -> float | None:
        """Return how long the current run has held the session lock."""
        if self.busy_started_monotonic is None:
            return None
        return max(0.0, time.perf_counter() - self.busy_started_monotonic)

    def mark_run_started(
        self,
        *,
        job_id: str,
        code: str,
        timeout: float,
        total_budget: float | None = None,
    ) -> None:
        """Record current run diagnostics while the session lock is held.

        Args:
            job_id: Identifier of the queued job.
            code: Submitted source, used for the diagnostic head line.
            timeout: Execution-phase timeout.
            total_budget: Execution timeout plus any dependency-install allowance.
                reap_stuck derives its threshold from this, so a job legitimately
                inside its install phase is not reaped before its own deadline.
        """
        started = now_iso()
        head = ""
        if isinstance(code, str):
            head = code.strip().splitlines()[0] if code.strip() else ""
        self.busy_since = started
        self.busy_started_monotonic = time.perf_counter()
        self.current_job_id = job_id
        self.current_code_head = head[:160]
        self.current_timeout = float(timeout)
        self.current_total_budget = float(total_budget if total_budget is not None else timeout)
        self.busy_rejects = 0
        self.last_modified = started

    def clear_run_state(self) -> None:
        """Clear current run diagnostics after the lock-protected run ends."""
        self.busy_since = None
        self.busy_started_monotonic = None
        self.current_job_id = None
        self.current_code_head = None
        self.current_timeout = None
        self.current_total_budget = None


# ------------ Docker manager ------------
class ContainerManager:
    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self.dck = docker.from_env(timeout=10)  # seconds per Docker API call

    def create_container(self, workdir_host: str, *, session_id: str = "") -> str:
        """Start a hardened sandbox container bound to the given host workdir.

        Args:
            workdir_host: Host directory bind-mounted at ``/work``.
            session_id: Owning session id, recorded as a Docker label so a
                container orphaned by a hard crash mid-create can be found.

        Returns:
            The id of the running container.

        Raises:
            RuntimeError: If the container is not running within the startup poll.
        """
        # ensure dirs
        Path(workdir_host, "inbox").mkdir(parents=True, exist_ok=True)
        Path(workdir_host, "outbox").mkdir(parents=True, exist_ok=True)

        binds = {
            workdir_host: {"bind": "/work", "mode": "rw"},
        }
        # optional datasets
        if os.path.isdir(self.cfg.datasets_dir):
            binds[self.cfg.datasets_dir] = {"bind": "/datasets", "mode": "ro"}

        sec_opts = ["no-new-privileges"] if self.cfg.no_new_privileges else []
        caps_drop = ["ALL"]

        run_kwargs = {
            "image": self.cfg.image,
            "detach": True,
            "stdin_open": False,
            "tty": False,
            "mem_limit": self.cfg.memory,
            "pids_limit": self.cfg.pids_limit,
            "cpu_quota": int(self.cfg.cpus * 100000),
            "cpu_period": 100000,
            "read_only": self.cfg.read_only_rootfs,
            "security_opt": sec_opts,
            "cap_drop": caps_drop,
            "user": f"{self.cfg.user_uid}:{self.cfg.user_gid}",
            "environment": {"SANDBOX_MAX_STATE_BYTES": str(self.cfg.max_state_bytes)},
            "volumes": binds,
            "tmpfs": {"/tmp": ""},
            "working_dir": "/work",
            "name": f"sandbox_{uuid.uuid4().hex[:8]}",
            # Label so a container orphaned by a hard crash mid-create can be found
            # and removed by hand. Deliberately NOT swept automatically: two sandbox
            # servers can share one Docker daemon, so a label-keyed sweep would
            # delete the other server's containers.
            "labels": {"ascollab.sandbox": "1", "ascollab.session_id": session_id or ""},
        }
        # Only set network_mode when actually disabling networking; do NOT pass None
        if self.cfg.network_disabled:
            run_kwargs["network_mode"] = "none"

        container = self.dck.containers.run(**run_kwargs)

        print(f"[sandbox] started {container.name} ({container.id[:12]})", flush=True)
        startup_deadline = time.monotonic() + 10.0
        while time.monotonic() < startup_deadline:
            container.reload()
            if container.status == "running":
                break
            time.sleep(0.05)
        if container.status != "running":
            try:
                logs = container.logs(tail=200).decode("utf-8", errors="ignore")
            except Exception:
                logs = "<no logs>"
            raise RuntimeError(
                f"Sandbox container exited early (status={container.status}). Logs:\n{logs}"
            )
        return container.id

    def remove_container(self, container_id: str) -> None:
        """Forcefully remove a sandbox container."""
        try:
            c = self.dck.containers.get(container_id)
            c.remove(force=True)
        except Exception:
            pass


# ---------- Session registry ------------
class SessionRegistry:
    """
    This is your original registry, but with:
      - a semaphore to cap concurrent sessions
      - blocking + logging when pool is full
    """

    def __init__(self, cfg: Settings, cm: ContainerManager, logger: logging.Logger):
        self.cfg = cfg
        self.cm = cm
        self.logger = logger
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        Path(self.cfg.effective_root).mkdir(parents=True, exist_ok=True)

        # <= this is the change: cap at max_sessions
        self._sem = threading.Semaphore(cfg.max_sessions)

    def _release_slot_once(self, s: Session) -> None:
        """Release a session capacity slot exactly once."""
        release = False
        with self._lock:
            if not s.slot_released:
                s.slot_released = True
                release = True
        if release:
            self._sem.release()

    def _forget_session(self, s: Session) -> None:
        """Remove a session from the registry if it is still registered."""
        with self._lock:
            current = self._sessions.get(s.session_id)
            if current is s:
                self._sessions.pop(s.session_id, None)

    def _is_registered(self, s: Session) -> bool:
        """Return True while this exact session object is still in the registry."""
        with self._lock:
            return self._sessions.get(s.session_id) is s

    def _clear_queue_files(self, s: Session) -> int:
        """Remove stale inbox/outbox job files after a timeout or restart."""
        removed = 0
        for name in ("inbox", "outbox"):
            queue_dir = Path(s.host_workdir, name)
            if not queue_dir.is_dir():
                continue
            for item in queue_dir.iterdir():
                if not item.is_file():
                    continue
                try:
                    item.unlink()
                    removed += 1
                except Exception:
                    self.logger.warning(
                        "[sandbox] failed to remove stale queue file %s",
                        item,
                        exc_info=True,
                    )
        return removed

    def _create_registered_session(self) -> Session:
        """Register a session, then create its container outside the registry lock.

        The caller must already own a semaphore slot; this method returns that slot
        on every failure path. The registry lock protects only the ``_sessions``
        dict -- holding it across a Docker round-trip serialises every other
        registry request (list, get, shutdown, reap) behind container startup.

        Returns:
            The registered session, with a running container.

        Raises:
            RuntimeError: If the session was shut down while Docker was working.
        """
        # Reserve the session BEFORE the Docker round-trip. Registering first keeps
        # it visible to shutdown_all/reap_stuck for the whole creation window;
        # doing the Docker work first and inserting afterwards would orphan the
        # container if a shutdown raced the create.
        # os.urandom makes the id collision-safe without holding the lock.
        sid = base64.urlsafe_b64encode(os.urandom(12)).decode("ascii").rstrip("=")
        workdir = os.path.join(self.cfg.effective_root, sid)
        s = Session(
            session_id=sid,
            created=now_iso(),
            last_modified=now_iso(),
            host_workdir=workdir,
            container_id="",
            creating=True,
        )
        with self._lock:
            self._sessions[sid] = s

        try:
            s.container_id = self.cm.create_container(workdir, session_id=sid)
            s.creating = False
            s.last_modified = now_iso()

            # A shutdown may have landed while Docker was working. shutdown() would
            # have called remove_container("") -- a no-op -- so the real container
            # is ours to clean up now that we finally know its id.
            if not self._is_registered(s) or s.closing or s.closed:
                self.logger.warning(
                    "[sandbox] session %s was shut down during creation; removing container",
                    sid,
                )
                self.cm.remove_container(s.container_id)
                raise RuntimeError("Session was shut down during creation")
            return s
        except Exception:
            s.creating = False
            self._forget_session(s)
            # Idempotent: a racing shutdown may already have returned the slot.
            self._release_slot_once(s)
            raise

    def create(self) -> Session:
        """Create a session, blocking if capacity is full."""
        # try fast-path first
        if not self._sem.acquire(blocking=False):
            # pool is full — now we log + block
            self.logger.info(
                "[sandbox] max sessions (%d) reached, waiting for a free container...",
                self.cfg.max_sessions,
            )
            # wait in 5s chunks so logs are clear
            got = False
            while not got:
                got = self._sem.acquire(timeout=5.0)
                if not got:
                    self.logger.info(
                        "[sandbox] still waiting for free container... (max=%d, active=%d)",
                        self.cfg.max_sessions,
                        len(self._sessions),
                    )

        # we own a semaphore slot now
        return self._create_registered_session()

    def try_create_nonblocking(self) -> Session | None:
        """Attempt to create a session without blocking when capacity is full."""
        # try to grab a slot; if none, just return None
        if not self._sem.acquire(blocking=False):
            return None

        return self._create_registered_session()

    def list(self) -> list[dict[str, Any]]:
        """List current sessions with metadata."""
        with self._lock:
            res = []
            for s in self._sessions.values():
                status = "busy" if s.lock.locked() else "idle"
                busy_seconds = s.busy_seconds()
                res.append(
                    {
                        "session_id": s.session_id,
                        "created": s.created,
                        "lastModified": s.last_modified,
                        "assigned_cores": None,
                        "mem_mb": self.cfg.memory,
                        "status": status,
                        "busy_since": s.busy_since,
                        "busy_seconds": busy_seconds,
                        "current_job_id": s.current_job_id,
                        "current_timeout": s.current_timeout,
                        "creating": s.creating,
                        "closing": s.closing,
                        "restarting": s.restarting,
                        "metadata": s.metadata(),
                    }
                )
            return res

    def get(self, sid: str) -> Session:
        """Fetch a session by ID or raise KeyError."""
        with self._lock:
            if sid not in self._sessions:
                raise KeyError("Session not found")
            return self._sessions[sid]

    def _restart_locked(self, s: Session) -> None:
        """Restart a session container while the caller already holds s.lock."""
        if s.closed:
            raise RuntimeError("Session is closed")
        s.restarting = True
        old_container_id = s.container_id
        try:
            self._clear_queue_files(s)
            self.cm.remove_container(old_container_id)
            s.container_id = self.cm.create_container(s.host_workdir, session_id=s.session_id)
            s.last_modified = now_iso()
            self.logger.info("[sandbox] session %s restarted", s.session_id)
        except Exception:
            s.closed = True
            self._forget_session(s)
            self._release_slot_once(s)
            self.logger.exception(
                "[sandbox] session %s restart failed; session removed and slot released",
                s.session_id,
            )
            raise
        finally:
            s.restarting = False

    def restart_locked(self, s: Session) -> None:
        """Restart a session whose lock is already held by the caller."""
        self._restart_locked(s)

    def restart(self, sid: str, lock_timeout: float = 10.0) -> dict[str, Any]:
        """Restart the container backing a session with a bounded lock wait."""
        s = self.get(sid)
        acquired = s.lock.acquire(timeout=max(0.0, float(lock_timeout)))
        if not acquired:
            return {
                "status": "error",
                "error": "Session is busy and could not be restarted before timeout",
                "session_id": sid,
                "session_status": "busy",
                "busy_since": s.busy_since,
                "busy_seconds": s.busy_seconds(),
                "current_job_id": s.current_job_id,
            }
        try:
            self._restart_locked(s)
            return {"status": "ok", "session_id": sid, "restarted": True}
        finally:
            s.lock.release()

    def shutdown_locked(self, s: Session) -> dict[str, Any]:
        """Shutdown a session while the caller already holds s.lock."""
        sid = s.session_id
        s.closing = True
        self._forget_session(s)
        try:
            self.cm.remove_container(s.container_id)
            s.closed = True
            s.last_modified = now_iso()
        finally:
            s.closing = False
            self._release_slot_once(s)
            self.logger.info("[sandbox] session %s shut down, slot released", sid)
        return {"status": "ok", "session_id": sid, "forced": False}

    def shutdown(self, sid: str, lock_timeout: float = 10.0) -> dict[str, Any]:
        """Shutdown a session and free its capacity slot with bounded waiting."""
        with self._lock:
            s = self._sessions.pop(sid, None)
        if s is None:
            return {"status": "ok", "session_id": sid, "note": "already shut down"}

        s.closing = True
        acquired = s.lock.acquire(timeout=max(0.0, float(lock_timeout)))
        forced = not acquired
        try:
            self.cm.remove_container(s.container_id)
            s.closed = True
            s.last_modified = now_iso()
        finally:
            s.closing = False
            self._release_slot_once(s)
            if acquired:
                s.lock.release()

        if forced:
            self.logger.warning(
                "[sandbox] session %s force shut down while lock was busy; slot released",
                sid,
            )
        else:
            self.logger.info("[sandbox] session %s shut down, slot released", sid)
        return {"status": "ok", "session_id": sid, "forced": forced}

    def reap_stuck(
        self,
        *,
        max_busy_seconds: float | None = None,
        owner: str | None = None,
    ) -> dict[str, Any]:
        """Force shutdown sessions that have been busy beyond a conservative threshold."""
        with self._lock:
            sessions = list(self._sessions.values())

        reaped: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for s in sessions:
            if owner is not None and s.meta.get("owner") != owner:
                skipped.append({"session_id": s.session_id, "reason": "owner_mismatch"})
                continue
            if not s.lock.locked():
                skipped.append({"session_id": s.session_id, "reason": "idle"})
                continue
            busy_seconds = s.busy_seconds()
            # Derive from the TOTAL budget (execution + install), not the execution
            # timeout alone, or a job still inside its install phase is reapable
            # before its own deadline. Test `is not None` rather than truthiness:
            # `or`-chaining on floats would skip a legitimate 0.0.
            budget = s.current_total_budget
            if budget is None:
                budget = s.current_timeout
            if budget is None:
                budget = self.cfg.default_timeout
            timeout = float(budget or 0.0)
            threshold = (
                float(max_busy_seconds) if max_busy_seconds is not None else (2.0 * timeout + 60.0)
            )
            if busy_seconds is None or busy_seconds < threshold:
                skipped.append(
                    {
                        "session_id": s.session_id,
                        "reason": "below_threshold",
                        "busy_seconds": busy_seconds,
                        "threshold": threshold,
                    }
                )
                continue
            result = self.shutdown(s.session_id, lock_timeout=0.1)
            result.update({"busy_seconds": busy_seconds, "threshold": threshold})
            reaped.append(result)
        return {"status": "ok", "reaped": reaped, "skipped": skipped}


# ------------- Code runner --------------
class CodeRunner:
    def __init__(self, registry: SessionRegistry, cfg: Settings, logger: logging.Logger):
        self.registry = registry
        self.cfg = cfg
        self.logger = logger

    def run(
        self,
        sid: str,
        code: str,
        timeout: float | None,
        auto_install: bool = True,
        shutdown_after: bool = False,
        reset: bool = False,
    ) -> dict[str, Any]:
        """Execute code inside a session, handling busy/timeout/restart logic."""
        s = self.registry.get(sid)
        if not s.lock.acquire(blocking=False):
            # this is the per-session lock: code is running there right now
            try:
                s.busy_rejects += 1
                rejects = s.busy_rejects
            except Exception:
                rejects = 0
            if rejects <= 1 or rejects % 10 == 0:
                self.logger.info(
                    "[sandbox] session %s is busy, rejecting run "
                    "(rejects=%s busy_seconds=%s job=%s)",
                    sid,
                    rejects,
                    s.busy_seconds(),
                    s.current_job_id,
                )
            else:
                self.logger.debug("[sandbox] session %s is busy, rejecting run", sid)
            return self._error(
                "Session is currently busy executing another request.",
                0.0,
                s,
                extra={
                    "busy_retryable": True,
                    "busy_rejects": rejects,
                    "busy_seconds": s.busy_seconds(),
                    "current_job_id": s.current_job_id,
                },
            )
        try:
            started = time.perf_counter()
            effective_timeout = float(timeout or self.cfg.default_timeout)
            # The daemon runs auto-install BEFORE arming its execution alarm, so the
            # server budget must cover both phases or a slow install looks like a code
            # timeout and costs the agent its whole session namespace.
            # Only grant the allowance when an install can actually happen: with
            # network_disabled the phase cannot succeed, and a blanket allowance would
            # also double every session's reap threshold for no benefit.
            can_install = bool(auto_install) and not self.cfg.network_disabled
            install_allowance = float(self.cfg.install_budget) if can_install else 0.0
            total_budget = effective_timeout + install_allowance
            job_id = f"{int(time.time() * 1000)}_{uuid.uuid4().hex}.json"
            s.mark_run_started(
                job_id=job_id,
                code=code,
                timeout=effective_timeout,
                total_budget=total_budget,
            )
            req = {
                "code": code,
                "timeout": effective_timeout,
                "auto_install": bool(auto_install),
                "install_budget": install_allowance,
                "created": s.created,
                "shutdown_after": bool(shutdown_after),
                "reset": bool(reset),
            }
            inbox = os.path.join(s.host_workdir, "inbox", job_id)
            outbox = os.path.join(s.host_workdir, "outbox", job_id)

            # Write atomically: the daemon polls for ``*.json`` and must never read a partial job.
            tmp_inbox = inbox + ".tmp"
            with open(tmp_inbox, "w", encoding="utf-8") as f:
                json.dump(req, f)
            os.replace(tmp_inbox, inbox)

            # The daemon's state save stops 1.5 s before this (runner_daemon.STATE_SAVE_GRACE_S).
            deadline = time.perf_counter() + total_budget + 3.0
            while time.perf_counter() < deadline:
                if s.closed or s.closing:
                    elapsed = time.perf_counter() - started
                    return self._error(
                        "Session shutdown was requested during execution.",
                        elapsed,
                        s,
                        extra={"shutdown_during_execution": True, "job_id": job_id},
                    )
                if os.path.exists(outbox):
                    with open(outbox, encoding="utf-8") as f:
                        doc = json.load(f)
                    os.remove(outbox)
                    s.last_modified = now_iso()
                    # annotate the response so clients/logs can confirm stickiness
                    try:
                        meta = doc.setdefault("session_metadata", {})
                        meta["session_id"] = sid
                        doc.setdefault("session_metadata", {})["session_id"] = sid
                        doc["job_id"] = job_id
                    except Exception:
                        pass
                    # If caller asked for ephemeral run, free the pool slot now.
                    if shutdown_after:
                        try:
                            self.registry.shutdown_locked(s)
                        except Exception:
                            self.logger.exception(
                                "[sandbox] shutdown_after failed for session %s", sid
                            )
                    return doc
                time.sleep(0.05)

            # timed out — restart container to get a fresh interpreter
            if s.closed or s.closing:
                elapsed = time.perf_counter() - started
                return self._error(
                    "Session shutdown was requested during execution.",
                    elapsed,
                    s,
                    extra={"shutdown_during_execution": True, "job_id": job_id},
                )
            self.logger.warning(
                "[sandbox] run in session %s timed out; restarting container while lock is held",
                sid,
            )
            self.registry.restart_locked(s)
            elapsed = time.perf_counter() - started
            overran_install = bool(auto_install) and elapsed > effective_timeout
            return self._error(
                (
                    f"Job exceeded its total budget of {int(total_budget)}s "
                    f"(execution limit {int(effective_timeout)}s"
                    + (
                        f", dependency-install allowance {int(install_allowance)}s"
                        if auto_install
                        else ""
                    )
                    + "). Container restarted; session state was lost."
                    + (
                        " The dependency install phase is the likely cause -- preinstall "
                        "packages in the sandbox image instead of importing them at runtime."
                        if overran_install
                        else ""
                    )
                ),
                elapsed,
                s,
                extra={
                    "timed_out": True,
                    "restarted": True,
                    "job_id": job_id,
                    "likely_install_overrun": overran_install,
                },
            )
        finally:
            # These MUST be independent: a failure to clear run state must never
            # prevent the lock release, or the session is wedged as permanently busy.
            try:
                s.clear_run_state()
            except Exception:
                self.logger.exception(
                    "[sandbox] failed to clear run state for session %s", s.session_id
                )
            try:
                s.lock.release()
            except Exception:
                self.logger.exception(
                    "[sandbox] failed to release lock for session %s", s.session_id
                )

    def _error(
        self,
        msg: str,
        exec_time: float,
        s: Session,
        *,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Standard error envelope for run responses."""
        payload = {
            "result": None,
            "stdout": None,
            "stderr": msg,
            "status": "error",
            "execution_time": exec_time,
            "images": None,
            "session_metadata": s.metadata(),
            "session_bytes": None,
            "pre_install": None,
        }
        if extra:
            payload.update(extra)
        return payload


# -------------- Flask app ---------------
def create_app(args=None) -> Flask:
    """Create and configure the Flask sandbox API server."""
    cfg = Settings()
    print(
        f"[sandbox] sessions_root={cfg.sessions_root} "
        f"datasets_dir={cfg.datasets_dir} "
        f"exists={os.path.isdir(cfg.datasets_dir)}",
        flush=True,
    )
    app = Flask(__name__)

    # logging
    logger = app.logger
    logger.setLevel(logging.INFO)

    cm = ContainerManager(cfg)
    reg = SessionRegistry(cfg, cm, logger)
    runner = CodeRunner(reg, cfg, logger)
    app.config["sandbox_config"] = cfg
    app.config["sandbox_registry"] = reg
    app.config["sandbox_runner"] = runner
    base = cfg.base_prefix.rstrip("/")

    @app.route(f"{base}/sessions", methods=["POST"])
    def create_session_route():
        payload = request.get_json(silent=True) or {}
        meta = {}
        # keep only small, useful tags
        for k in ("owner", "agent_id", "iteration", "session_name"):
            if k in payload:
                meta[k] = payload[k]

        # try to create without blocking
        s = reg.try_create_nonblocking()
        if s is None:
            # pool is full → tell the client explicitly
            return jsonify(
                {
                    "error": "Max sessions reached, please try again later.",
                    "active_sessions": len(reg.list()),
                    "max_sessions": cfg.max_sessions,
                }
            ), 429
        # success — attach metadata so clients can reap safely later
        s.meta = meta
        return jsonify(
            {
                "session_id": s.session_id,
                "created": s.created,
                "assigned_cores": None,
                "mem_mb": cfg.memory,
                "status": "idle",
                "metadata": s.metadata(),
            }
        )

    @app.route(f"{base}/sessions", methods=["GET"])
    def list_sessions_route():
        return jsonify(reg.list())

    @app.route(f"{base}/health", methods=["GET"])
    def health_route():
        sessions = reg.list()
        return jsonify(
            {
                "status": "ok",
                "max_sessions": cfg.max_sessions,
                "active_sessions": len(sessions),
            }
        )

    @app.route(f"{base}/run", methods=["POST"])
    def run_route():
        payload = request.get_json(force=True)
        sid = payload.get("session_id")
        code = payload.get("code")
        timeout = payload.get("timeout")
        auto_install = bool(payload.get("auto_install", True))
        reset = bool(payload.get("reset", False))
        if not sid or code is None:
            return jsonify({"error": "session_id and code are required"}), 400
        try:
            shutdown_after = bool(payload.get("shutdown_after", False))
            resp = runner.run(
                sid,
                code,
                timeout,
                auto_install=auto_install,
                shutdown_after=shutdown_after,
                reset=reset,
            )
            return jsonify(resp)
        except KeyError:
            return jsonify({"error": "Session not found"}), 404
        except Exception as e:
            try:
                s = reg.get(sid)
            except KeyError:
                return jsonify({"error": str(e), "session_id": sid}), 500
            return jsonify(runner._error(str(e), 0.0, s)), 500

    @app.route(f"{base}/restart", methods=["POST"])
    def restart_route():
        payload = request.get_json(force=True)
        sid = payload.get("session_id")
        if not sid:
            return jsonify({"error": "session_id is required"}), 400
        try:
            lock_timeout = float(payload.get("lock_timeout", 10.0))
            result = reg.restart(sid, lock_timeout=lock_timeout)
            status_code = 423 if result.get("status") == "error" else 200
            return jsonify(result), status_code
        except KeyError:
            return jsonify({"error": "Session not found"}), 404
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route(f"{base}/shutdown", methods=["POST"])
    def shutdown_route():
        payload = request.get_json(force=True)
        sid = payload.get("session_id")
        if not sid:
            return jsonify({"error": "session_id is required"}), 400
        try:
            lock_timeout = float(payload.get("lock_timeout", 10.0))
            return jsonify(reg.shutdown(sid, lock_timeout=lock_timeout))
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route(f"{base}/shutdown_all", methods=["POST"])
    def shutdown_all_route():
        items = reg.list()
        errs, done = [], []
        for it in items:
            sid = it.get("session_id") or it.get("id")
            if not sid:
                continue
            try:
                reg.shutdown(sid)
                done.append(sid)
            except Exception as e:
                errs.append({"session_id": sid, "error": str(e)})
        return jsonify({"shut_down": done, "errors": errs})

    @app.route(f"{base}/reap_stuck", methods=["POST"])
    def reap_stuck_route():
        payload = request.get_json(silent=True) or {}
        owner = payload.get("owner")
        if owner is not None:
            owner = str(owner)
        max_busy_seconds = payload.get("max_busy_seconds")
        try:
            max_busy_seconds = float(max_busy_seconds) if max_busy_seconds is not None else None
        except Exception:
            return jsonify({"error": "max_busy_seconds must be numeric"}), 400
        return jsonify(reg.reap_stuck(max_busy_seconds=max_busy_seconds, owner=owner))

    @app.route(f"{base}/stats", methods=["GET"])
    def stats_route():
        sessions = reg.list()
        return jsonify(
            {
                "max_sessions": cfg.max_sessions,
                "active_sessions": len(sessions),
                "sessions": sessions,
                "limits": {"cpus": cfg.cpus, "memory": cfg.memory, "pids": cfg.pids_limit},
            }
        )

    return app


app = create_app()
