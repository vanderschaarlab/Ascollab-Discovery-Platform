import base64
import binascii
import hashlib
import json
import logging
import os
import random
import threading
import time
from pathlib import Path
from typing import Any

import requests
from pydantic import BaseModel, Field

try:
    from langchain_core.tools import StructuredTool
except Exception as exc:
    _STRUCTURED_TOOL_IMPORT_ERROR = exc

    class StructuredTool:  # type: ignore[no-redef]
        """Fallback that keeps client-only sandbox utilities importable."""

        @classmethod
        def from_function(cls, *args, **kwargs):
            raise ImportError(
                "langchain_core is required to build sandbox StructuredTool instances"
            ) from _STRUCTURED_TOOL_IMPORT_ERROR


import atexit
import signal
from datetime import UTC, datetime

from urllib3.util.retry import Retry

from system.plot_summary_artifacts import (
    average as average_values,
)
from system.plot_summary_artifacts import (
    build_fallback_plot_summary,
    build_plot_artifact_record,
    generate_plot_summary_for_image,
    is_graph_logic_enabled,
    normalize_graph_logic_cfg,
    write_plot_summary_artifact,
)
from system.snapshot_paths import resolve_iteration_snapshot_dir

try:
    import fcntl
except Exception:
    fcntl = None

log = logging.getLogger(__name__)

# --- sticky registry for guaranteed cleanup ---
_STICKY_REGISTRY: "set[StickySession]" = set()


def _register_sticky(sticky: "StickySession") -> None:
    """Register a sticky session for process-level cleanup."""
    try:
        _STICKY_REGISTRY.add(sticky)
    except Exception:
        pass


def _unregister_sticky(sticky: "StickySession") -> None:
    """Remove a sticky session from the cleanup registry."""
    try:
        _STICKY_REGISTRY.discard(sticky)
    except Exception:
        pass


def close_all_stickies() -> None:
    """Best-effort close for all stickies on shutdown."""
    for s in list(_STICKY_REGISTRY):
        try:
            s.close()
        except Exception:
            pass


# register process-level cleanup
atexit.register(close_all_stickies)


def _sig_close_all(sig, frame):
    try:
        close_all_stickies()
    finally:
        # default handler behavior
        os._exit(130 if sig == signal.SIGINT else 143)


signal.signal(signal.SIGINT, _sig_close_all)
signal.signal(signal.SIGTERM, _sig_close_all)


def _read_positive_env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
        if value > 0:
            return value
    except Exception:
        pass
    return int(default)


DEFAULT_MAX_PLOTS_PER_EXECUTION = _read_positive_env_int(
    "SANDBOX_MAX_PLOTS_PER_EXECUTION",
    8,
)
DEFAULT_MAX_PLOT_BYTES = _read_positive_env_int(
    "SANDBOX_MAX_PLOT_BYTES",
    8_000_000,
)


# --- special error so callers can queue/retry ----------------
class MaxSessionsError(RuntimeError):
    """Raised when the sandbox reports that the maximum number of sessions has been reached."""


def _log(msg: str) -> None:
    log.debug(f"[sandbox-client] {msg}")


class SandboxClient:
    def __init__(
        self,
        code_timeout_seconds: int = 240,
        base_url: str | None = None,
    ):
        """Initialize the sandbox HTTP client.

        Environment overrides:
          - `SANDBOX_BASE_URL` or `SANDBOX_BASE` (fallback order)
            e.g. http://127.0.0.1:9000 or http://127.0.0.1:9000/base
          - `SANDBOX_AUTH_TOKEN` (optional)

        Args:
            code_timeout_seconds: Default execution timeout per code run.
            base_url: Optional explicit sandbox base URL.
        """
        self.base_url = (
            base_url
            or os.environ.get("SANDBOX_BASE_URL")
            or os.environ.get("SANDBOX_BASE")
            or "http://127.0.0.1:9000"
        ).rstrip("/")

        self.timeout = code_timeout_seconds
        self.auth_token = os.environ.get("SANDBOX_AUTH_TOKEN")

        self.session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=(502, 503, 504),
            allowed_methods=frozenset(["GET", "POST", "DELETE"]),
            raise_on_status=False,
        )
        adapter = requests.adapters.HTTPAdapter(
            max_retries=retry,
            pool_connections=10,
            pool_maxsize=20,
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    # ---------------- helpers ----------------
    def _url(self, path: str) -> str:
        """Join base URL with a relative path."""
        return f"{self.base_url}/{path.lstrip('/')}"

    def _headers(self) -> dict[str, str]:
        """Build JSON headers with optional auth."""
        h = {"Content-Type": "application/json"}
        if self.auth_token:
            h["X-Auth"] = self.auth_token
        return h

    def _run_error_payload(
        self,
        session_id: str,
        stderr: str,
        start_time: float,
        **extra: Any,
    ) -> dict[str, Any]:
        """Build the standard sandbox run error envelope."""
        payload = {
            "result": None,
            "stdout": None,
            "stderr": stderr,
            "status": "error",
            "execution_time": time.time() - start_time,
            "images": None,
            "session_metadata": {
                "created": "unknown",
                "lastModified": "unknown",
                "packages": [],
                "session_id": session_id,
            },
            "session_bytes": None,
            "pre_install": None,
        }
        payload.update(extra)
        return payload

    def _retry_delay(self, base_delay: float, attempt: int, max_delay: float) -> float:
        """Return capped exponential backoff with small jitter."""
        base = max(0.1, float(base_delay))
        cap = max(base, float(max_delay))
        delay = min(cap, base * (2 ** max(0, attempt)))
        return delay * random.uniform(0.8, 1.2)

    # --- selective reaper -------------------
    def _parse_ts_to_epoch(self, ts):
        """Return seconds since epoch for numeric or ISO8601 strings, else None."""
        if isinstance(ts, (int, float)):
            return float(ts)
        if isinstance(ts, str):
            # handle "2025-11-11T15:41:12Z" and "...+00:00"
            try:
                if ts.endswith("Z"):
                    ts = ts[:-1] + "+00:00"
                return datetime.fromisoformat(ts).timestamp()
            except Exception:
                return None
        return None

    def _reap_stale_sessions(
        self,
        *,
        keep_ids: set[str],
        owner: str | None = None,
        max_age_seconds: float = 30.0,
        path: str = "/base/sessions",
        shutdown_path: str = "/base/shutdown",
    ) -> None:
        """
        Best-effort, **safe** reap.

        We ONLY shut down sessions that:
        - are NOT in `keep_ids`
        - AND (if `owner` is given) have metadata.owner == owner
        - AND are older than `max_age_seconds` (using numeric or ISO timestamps)
        """
        try:
            sessions = self.list_sessions(path=path)
        except Exception:
            return

        if not isinstance(sessions, list):
            return

        now = time.time()

        for s in sessions:
            sid = s.get("session_id") or s.get("id")
            if not sid:
                continue
            if sid in keep_ids:
                continue

            meta = s.get("metadata") or s.get("extra_payload") or s.get("meta") or {}
            if owner is not None and meta.get("owner") != owner:
                continue

            last_mod = (
                meta.get("lastModified")
                or meta.get("created")
                or s.get("lastModified")
                or s.get("created")
            )
            t = self._parse_ts_to_epoch(last_mod)
            if isinstance(t, float):
                age = now - t
                if age < max_age_seconds:
                    # too fresh — likely just created
                    continue

            # If we get here: safe to shut down
            try:
                self.shutdown(
                    sid,
                    path=shutdown_path,
                    timeout_seconds=240.0,
                    retries=1,
                    retry_delay=1.0,
                    verify_on_timeout=True,
                )
            except Exception:
                pass

    # ---------------- session lifecycle ----------------
    def create_session(
        self,
        python_packages: list[str] | None = None,
        code_timeout_seconds: int | None = None,
        session_name: str | None = None,
        extra_payload: dict[str, Any] | None = None,
        path: str = "/base/sessions",
        wait_if_full: bool = False,
        wait_timeout: float = 300.0,
        poll_interval: float = 2.0,
    ) -> str:
        """Create a new sandbox session and return its ID.

        Tolerates buggy servers that create a session but respond with HTTP 500/HTML
        by diffing the session list to recover the new ID.

        Args:
            python_packages: Optional list of packages to pre-install.
            code_timeout_seconds: Optional per-session code timeout override.
            session_name: Optional display name for the session.
            extra_payload: Additional JSON payload to include on creation.
            path: Endpoint path for session creation.
            wait_if_full: Whether to wait/retry when capacity is full.
            wait_timeout: Max time to wait when at capacity.
            poll_interval: Delay between capacity retries.

        Returns:
            The created session ID.

        Raises:
            MaxSessionsError: When capacity is full and waiting is disabled or times out.
            RuntimeError: For other creation failures.
        """
        payload: dict[str, Any] = {}
        if python_packages is not None:
            payload["python_packages"] = list(python_packages)
        if code_timeout_seconds is not None:
            payload["code_timeout_seconds"] = int(code_timeout_seconds)
        if session_name:
            payload["session_name"] = str(session_name)
        if extra_payload:
            payload.update(extra_payload)

        start = time.time()
        attempt = 0

        while True:
            attempt += 1

            # we capture the sessions BEFORE the POST
            try:
                before_sessions = self.list_sessions(path=path)
                before_ids = {
                    s.get("session_id") or s.get("id")
                    for s in before_sessions
                    if isinstance(s, dict)
                }
            except Exception:
                before_ids = set()

            try:
                r = self.session.post(
                    self._url(path),
                    headers=self._headers(),
                    json=payload,
                    timeout=30,
                )

                # normal success
                if r.status_code in (200, 201):
                    data = r.json()
                    sid = data.get("session_id") or data.get("id")
                    if not sid:
                        raise RuntimeError(
                            f"Sandbox create_session succeeded but no session_id in response:\n{data}"
                        )
                    return str(sid)

                body_text = r.text or ""

                # capacity case (400/429 + string)
                if r.status_code in (400, 429) and "Max sessions reached" in body_text:
                    if not wait_if_full:
                        raise MaxSessionsError(f"Sandbox is at capacity: {body_text}")
                    elapsed = time.time() - start
                    if elapsed > wait_timeout:
                        raise MaxSessionsError(
                            f"Sandbox still at capacity after {elapsed:.1f}s, giving up. Last response: {body_text}"
                        )
                    _log(
                        f"capacity reached, waiting {poll_interval}s before retry (attempt {attempt})..."
                    )
                    time.sleep(poll_interval)
                    continue

                # --- BUGGY SERVER CASE: 500 but it maybe created the session ---
                if r.status_code == 500:
                    _log("create_session got 500, trying to recover by polling /base/sessions...")
                    recovered_sid = None

                    # try for up to ~8s
                    poll_deadline = time.time() + 8.0
                    while time.time() < poll_deadline and recovered_sid is None:
                        try:
                            after_sessions = self.list_sessions(path=path)
                            after_ids = {
                                s.get("session_id") or s.get("id")
                                for s in after_sessions
                                if isinstance(s, dict)
                            }
                            new_ids = list(after_ids - before_ids)
                            if new_ids:
                                recovered_sid = new_ids[0]
                                _log(f"recovered session id {recovered_sid} after a 500 (delayed)")
                                # IMPORTANT: we no longer do a global reap here.
                                # If you want to clean up, call _reap_stale_sessions() yourself
                                # later with a proper owner + keep_ids.
                                return recovered_sid
                        except Exception as diff_err:
                            _log(f"recovery poll failed once: {diff_err}")
                        # wait a bit and try again
                        time.sleep(0.5)

                    # if we get here, we never saw the new session in /base/sessions
                    # -> find out if we’re actually at capacity and should tell caller to back off
                    try:
                        stats = self.get_server_stats()
                        active = stats.get("active_sessions")
                        max_sess = stats.get("max_sessions")
                        if (
                            isinstance(active, int)
                            and isinstance(max_sess, int)
                            and active >= max_sess
                        ):
                            # tell outer code "this is capacity", not "fatal 500"
                            raise MaxSessionsError(
                                f"Sandbox at capacity (active={active}, max={max_sess}) after 500."
                            )
                    except MaxSessionsError:
                        raise
                    except Exception as st_err:
                        _log(f"could not read server stats after 500: {st_err}")

                    # otherwise: it really is a 500
                    raise RuntimeError(
                        f"Sandbox create_session failed 500 at {path} and recovery/polling failed:\n{body_text}"
                    )

            except requests.ReadTimeout as e:
                raise RuntimeError(f"Sandbox create_session request timed out: {e}") from e
            except requests.RequestException as e:
                raise RuntimeError(f"Sandbox create_session request error: {e!s}") from e

    def list_sessions(self, path: str = "/base/sessions"):
        """List active sandbox sessions."""
        r = self.session.get(self._url(path), headers=self._headers(), timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"Sandbox list_sessions failed {r.status_code}:\n{r.text}")
        return r.json()

    def run(
        self,
        session_id: str,
        code: str,
        auto_install: bool = True,
        path: str = "/base/run",
        shutdown_after: bool = False,  # tell the server to kill the container afterwards
        *,
        reset: bool = False,  # clean interpreter state in the same container
        timeout: int | None = None,
        retries: int = 120,  # tolerate brief per-session busy
        retry_delay: float = 5.0,
        max_retry_delay: float = 30.0,
        busy_retry_timeout_seconds: float | None = None,
        raise_on_capacity: bool = False,  # if True, 429 -> MaxSessionsError
    ):
        """Execute Python code inside a sandbox session.

        Args:
            session_id: Target session identifier.
            code: Python source code to run.
            auto_install: Whether to auto-install missing packages.
            path: Endpoint path for execution.
            shutdown_after: If True, terminate the session after execution.
            reset: Reset interpreter state before execution.
            timeout: Optional timeout override for this run.
            retries: Retries when the session is temporarily busy.
            retry_delay: Backoff between retries.
            max_retry_delay: Maximum delay between busy/transport retries.
            busy_retry_timeout_seconds: Wall-clock budget for busy retries.
            raise_on_capacity: If True, convert capacity errors to MaxSessionsError.

        Returns:
            Execution response dict containing stdout/stderr/result metadata.
        """
        try:
            code_len = len(code) if isinstance(code, str) else 0
            code_head = ""
            if isinstance(code, str):
                code_head = code.strip().splitlines()[0] if code.strip() else ""
            log.debug(
                "[sandbox_run] "
                f"session_id={session_id} code_len={code_len} code_head={code_head[:120]!r}"
            )
        except Exception:
            pass
        effective_timeout = int(timeout if timeout is not None else self.timeout)
        payload = {
            "session_id": session_id,
            "code": code,
            "auto_install": bool(auto_install),
            "timeout": effective_timeout,
            "shutdown_after": bool(shutdown_after),
            "reset": bool(reset),
        }

        start_time = time.time()
        request_timeout = max(effective_timeout + 30, 60)
        try:
            busy_budget = (
                float(busy_retry_timeout_seconds)
                if busy_retry_timeout_seconds is not None
                else float(os.getenv("SANDBOX_BUSY_RETRY_TIMEOUT_SECONDS", "180"))
            )
        except Exception:
            busy_budget = 180.0
        busy_budget = max(0.0, busy_budget)
        busy_attempts = 0
        transport_attempts = 0
        while True:
            try:
                r = self.session.post(
                    self._url(path),
                    headers=self._headers(),
                    json=payload,
                    timeout=request_timeout,
                )
                if r.status_code == 200:
                    data = r.json()
                    # Ensure the response always carries the session id for logging/debug
                    try:
                        meta = (
                            data.setdefault("session_metadata", {})
                            if isinstance(data, dict)
                            else None
                        )
                        if meta is not None and "session_id" not in meta:
                            meta["session_id"] = session_id
                    except Exception:
                        pass

                    is_busy = (
                        isinstance(data, dict)
                        and data.get("status") == "error"
                        and isinstance(data.get("stderr"), str)
                        and "busy" in data["stderr"].lower()
                    )
                    if is_busy:
                        busy_attempts += 1
                        elapsed = time.time() - start_time
                        if busy_attempts <= retries and elapsed < busy_budget:
                            remaining = max(0.0, busy_budget - elapsed)
                            delay = min(
                                remaining,
                                self._retry_delay(retry_delay, busy_attempts - 1, max_retry_delay),
                            )
                            if busy_attempts == 1 or busy_attempts % 5 == 0:
                                _log(
                                    "session busy; retrying "
                                    f"session_id={session_id} attempts={busy_attempts} "
                                    f"elapsed={elapsed:.1f}s sleep={delay:.1f}s"
                                )
                            if delay > 0:
                                time.sleep(delay)
                            continue

                        data["stderr"] = (
                            "Session remained busy after retry budget was exhausted. "
                            f"Original error: {data.get('stderr')}"
                        )
                        data["busy_retry_attempts"] = busy_attempts
                        data["busy_retry_elapsed_seconds"] = elapsed
                        data["busy_retry_timeout_seconds"] = busy_budget
                        return data
                    return data
                # Not found → caller likely used a dead session_id
                if r.status_code == 404:
                    return self._run_error_payload(
                        session_id,
                        "Session not found (404).",
                        start_time,
                    )

                # Capacity
                if r.status_code in (400, 429) and "Max sessions reached" in (r.text or ""):
                    if raise_on_capacity:
                        raise MaxSessionsError(r.text or "Sandbox is at capacity.")
                    return self._run_error_payload(
                        session_id,
                        f"Server at capacity {r.status_code}: {r.text}",
                        start_time,
                    )

                # Generic error envelope
                return self._run_error_payload(
                    session_id,
                    f"Server returned {r.status_code}:\n{r.text}",
                    start_time,
                )
            except Exception as e:
                # Transport-level issue; if we still have retries, back off a bit
                if transport_attempts < retries:
                    delay = self._retry_delay(
                        retry_delay,
                        transport_attempts,
                        max_retry_delay,
                    )
                    time.sleep(delay)
                    transport_attempts += 1
                    continue
                return self._run_error_payload(
                    session_id,
                    f"Transport error: {e!s}",
                    start_time,
                    transport_retry_attempts=transport_attempts,
                )

    def restart(
        self,
        session_id: str,
        path: str = "/base/restart",
        *,
        lock_timeout: float = 10.0,
    ):
        """Restart a sandbox session process."""
        r = self.session.post(
            self._url(path),
            headers=self._headers(),
            json={"session_id": session_id, "lock_timeout": float(lock_timeout)},
            timeout=30,
        )
        if r.status_code != 200:
            raise RuntimeError(f"Sandbox restart failed {r.status_code}:\n{r.text}")
        return r.json()

    def shutdown(
        self,
        session_id: str,
        path: str = "/base/shutdown",
        *,
        timeout_seconds: float = 240.0,
        lock_timeout: float | None = None,
        retries: int = 2,
        retry_delay: float = 1.0,
        verify_on_timeout: bool = True,
        sessions_path: str = "/base/sessions",
    ):
        """Terminate a sandbox session with retries and verification."""
        last_err: Exception | None = None
        for attempt in range(retries + 1):
            try:
                payload: dict[str, Any] = {"session_id": session_id}
                if lock_timeout is not None:
                    payload["lock_timeout"] = float(lock_timeout)
                r = self.session.post(
                    self._url(path),
                    headers=self._headers(),
                    json=payload,
                    timeout=timeout_seconds,
                )
                if r.status_code != 200:
                    raise RuntimeError(f"Sandbox shutdown failed {r.status_code}:\n{r.text}")
                return r.json()
            except Exception as e:
                last_err = e
                if verify_on_timeout:
                    try:
                        sessions = self.list_sessions(path=sessions_path)
                        if isinstance(sessions, list):
                            found = any(
                                (s.get("session_id") or s.get("id")) == session_id for s in sessions
                            )
                            if not found:
                                return {
                                    "status": "ok",
                                    "session_id": session_id,
                                    "note": "already shut down",
                                }
                    except Exception:
                        pass
                if attempt < retries:
                    time.sleep(retry_delay * (attempt + 1))
        if last_err is not None:
            return {"status": "error", "session_id": session_id, "error": str(last_err)}
        return {"status": "error", "session_id": session_id, "error": "unknown error"}

    def shutdown_all_sessions(
        self,
        sessions_path: str = "/base/sessions",
        shutdown_path: str = "/base/shutdown",
    ):
        """Shutdown all sessions visible to the sandbox server."""
        sessions = self.list_sessions(path=sessions_path)
        results = {
            "total_sessions": len(sessions) if isinstance(sessions, list) else 0,
            "shutdown_results": [],
            "errors": [],
        }
        if isinstance(sessions, list):
            for session in sessions:
                sid = session.get("session_id") or session.get("id")
                if not sid:
                    continue
                try:
                    res = self.shutdown(sid, path=shutdown_path)
                    results["shutdown_results"].append(
                        {"session_id": sid, "success": True, "result": res}
                    )
                except Exception as e:
                    results["errors"].append({"session_id": sid, "error": str(e)})
        return results

    def reap_stuck_sessions(
        self,
        *,
        owner: str | None = None,
        max_busy_seconds: float | None = None,
        path: str = "/base/reap_stuck",
    ):
        """Ask the sandbox server to force-reap sessions stuck in busy state."""
        payload: dict[str, Any] = {}
        if owner is not None:
            payload["owner"] = str(owner)
        if max_busy_seconds is not None:
            payload["max_busy_seconds"] = float(max_busy_seconds)
        r = self.session.post(
            self._url(path),
            headers=self._headers(),
            json=payload,
            timeout=30,
        )
        if r.status_code != 200:
            raise RuntimeError(f"Sandbox reap_stuck failed {r.status_code}:\n{r.text}")
        return r.json()

    def get_server_stats(self, path: str = "/base/stats"):
        """Return sandbox server stats if available."""
        try:
            r = self.session.get(self._url(path), headers=self._headers(), timeout=30)
            if r.status_code == 404:
                return {"error": "Stats endpoint not available"}
            if r.status_code != 200:
                raise RuntimeError(f"Sandbox stats failed {r.status_code}:\n{r.text}")
            return r.json()
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                return {"error": "Stats endpoint not available"}
            raise

    def sticky_session(
        self,
        session_id: str | None = None,
        **create_kwargs: Any,
    ) -> "StickySession":
        """Get a per-agent sticky session handle.

        If `session_id` is None, it will be created on first use. `create_kwargs`
        are forwarded to `create_session(...)`.

        Args:
            session_id: Optional existing session to bind.
            **create_kwargs: Parameters used when creating a new session.

        Returns:
            StickySession bound to this client.
        """
        return StickySession(self, session_id=session_id, create_kwargs=create_kwargs)


# --- sticky per-agent session -----------------------
class StickySession:
    """Own one sandbox session_id for an agent and reuse it.

    If the server loses the session (404), optionally recreate once with the
    same creation arguments.
    """

    def __init__(
        self,
        client: "SandboxClient",
        session_id: str | None = None,
        create_kwargs: dict[str, Any] | None = None,
        recreate_on_404: bool = True,
    ):
        """Initialize a StickySession wrapper.

        Args:
            client: SandboxClient used for lifecycle actions.
            session_id: Optional existing session to bind.
            create_kwargs: Parameters for creating a new session if needed.
            recreate_on_404: Whether to recreate once when a session is lost.
        """
        self.client = client
        self.session_id = session_id
        self.create_kwargs = create_kwargs or {}
        self.recreate_on_404 = recreate_on_404
        self._lock = threading.RLock()
        _register_sticky(self)

    def ensure(self) -> str:
        """Ensure a session exists; create on first use."""
        with self._lock:
            if not self.session_id:
                self.session_id = self.client.create_session(**self.create_kwargs)
            return self.session_id

    def run(
        self,
        code: str,
        *,
        timeout: int | None = None,
        auto_install: bool = True,
        reset: bool = False,
        shutdown_after: bool = False,
        retries: int = 120,
        retry_delay: float = 5.0,
    ) -> dict[str, Any]:
        """Execute code within the sticky session, recreating on 404 if allowed."""
        with self._lock:
            sid = self.ensure()
            resp = self.client.run(
                sid,
                code,
                auto_install=auto_install,
                shutdown_after=shutdown_after,
                reset=reset,
                timeout=timeout,
                retries=retries,
                retry_delay=retry_delay,
            )
            # Make sure the response carries the current sid for logging
            try:
                if isinstance(resp, dict):
                    meta = resp.setdefault("session_metadata", {})
                    meta.setdefault("session_id", sid)
            except Exception:
                pass
            # If the session was lost mid-run, optionally recreate once
            if (
                self.recreate_on_404
                and isinstance(resp, dict)
                and resp.get("status") == "error"
                and isinstance(resp.get("stderr"), str)
                and "session not found" in resp["stderr"].lower()
            ):
                self.session_id = self.client.create_session(**self.create_kwargs)
                # retry once with the new session
                sid2 = self.session_id
                resp = self.client.run(
                    sid2,
                    code,
                    auto_install=auto_install,
                    shutdown_after=shutdown_after,
                    reset=reset,
                    timeout=timeout,
                    retries=retries,
                    retry_delay=retry_delay,
                )
                try:
                    if isinstance(resp, dict):
                        meta = resp.setdefault("session_metadata", {})
                        meta.setdefault("session_id", sid2)
                except Exception:
                    pass
            return resp

    def reset(self) -> dict[str, Any]:
        """Clear Python globals in-place, keeping the same container & filesystem."""
        with self._lock:
            sid = self.ensure()
            return self.client.run(sid, "None", reset=True)

    def close(self) -> None:
        """Free the container and release its pool slot."""
        with self._lock:
            if not self.session_id:
                _unregister_sticky(self)
                return
            try:
                self.client.shutdown(self.session_id)
            finally:
                self.session_id = None
                _unregister_sticky(self)


# simple async context to guarantee close
class StickyHandle:
    def __init__(self, sticky: "StickySession"):
        self.sticky = sticky

    async def __aenter__(self):
        return self.sticky.ensure()

    async def __aexit__(self, exc_type, exc, tb):
        try:
            self.sticky.close()
        except Exception:
            pass


# --- helper to build a tool bound to a StickySession ---
def make_code_executor_tool_for_sticky(
    sticky: "StickySession",
    *,
    agent_id: str | None = None,
    retries: int = 120,
    retry_delay: float = 5.0,
) -> StructuredTool:
    """Build an execute_code tool bound to a specific sticky session."""

    class CodeExecInput(BaseModel):
        code: str = Field(
            ...,
            description=(
                "Python code to execute. Keep runs small and incremental; prefer "
                "sampling and checkpoints over full dataset passes."
            ),
        )

    def _exec(code: str) -> dict[str, Any]:
        if agent_id:
            log.debug(
                "[execute_code_tool] "
                f"agent_id={agent_id} session_id={getattr(sticky, 'session_id', None)}"
            )
        # Persist state across calls; never reset/shutdown here.
        return sticky.run(
            code,
            reset=False,
            shutdown_after=False,
            retries=retries,
            retry_delay=retry_delay,
        )

    return StructuredTool.from_function(
        name="execute_code",
        func=_exec,
        description=(
            "Executes Python code in a persistent sandboxed container tied to this agent. "
            "Prefer small, incremental runs with logging/checkpoints; avoid large dense "
            "materializations unless explicitly requested."
        ),
        return_direct=False,
        args_schema=CodeExecInput,
    )


# ---------------------------------------------------------------------
# Tool factories (old + new)
# ---------------------------------------------------------------------
def make_code_executor_tool(
    c: SandboxClient,
    session_id: str,
    *,
    agent_id: str | None = None,
    retries: int = 120,
    retry_delay: float = 5.0,
) -> StructuredTool:
    """Build an execute_code tool bound to a fixed session ID."""

    class CodeExecInput(BaseModel):
        code: str = Field(
            ...,
            description=(
                "Python code to execute. Keep runs small and incremental; prefer "
                "sampling and checkpoints over full dataset passes."
            ),
        )

    def _exec(code: str) -> dict[str, Any]:
        if agent_id:
            log.debug(f"[execute_code_tool] agent_id={agent_id} session_id={session_id}")
        return c.run(session_id, code, retries=retries, retry_delay=retry_delay)

    return StructuredTool.from_function(
        name="execute_code",
        func=_exec,
        description=(
            "Executes Python code in a persistent sandboxed environment. Prefer small, "
            "incremental runs with logging/checkpoints; avoid large dense "
            "materializations unless explicitly requested."
        ),
        return_direct=False,
        args_schema=CodeExecInput,
    )


def _sanitize_token(value: Any, fallback: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return fallback
    cleaned = []
    for ch in raw:
        if ch.isalnum() or ch in ("_", "-", "."):
            cleaned.append(ch)
        else:
            cleaned.append("-")
    out = "".join(cleaned).strip("-._")
    return (out or fallback)[:80]


def _coerce_positive_int(value: Any, default: int) -> int:
    try:
        iv = int(value)
        if iv > 0:
            return iv
    except Exception:
        pass
    return int(default)


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _decode_image_blob(
    data: str,
    *,
    max_image_bytes: int,
) -> tuple[bytes | None, str | None]:
    encoded = str(data or "").strip()
    if not encoded:
        return None, "empty_image_blob"

    if encoded.startswith("data:"):
        try:
            header, encoded = encoded.split(",", 1)
        except ValueError:
            return None, "invalid_data_uri"
        if ";base64" not in header.lower():
            return None, "non_base64_data_uri"
        mime = header.split(";", 1)[0].split(":", 1)[-1].strip().lower()
        if mime and mime not in ("image/png", "image/x-png"):
            return None, f"unsupported_mime:{mime}"

    # Base64 expands by ~4/3. Cap pre-decode size to avoid large allocations.
    max_encoded_len = int(max_image_bytes * 1.5) + 16
    if len(encoded) > max_encoded_len:
        return None, "image_too_large_encoded"

    try:
        payload = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        return None, "invalid_base64"

    if len(payload) > max_image_bytes:
        return None, "image_too_large_decoded"

    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return None, "not_png"

    return payload, None


def _append_jsonl_with_lock(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        if fcntl is not None:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            except Exception:
                pass
        try:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            handle.flush()
        finally:
            if fcntl is not None:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass


def is_sandbox_execution_successful(resp: Any) -> bool:
    """Return True only when the sandbox response explicitly reports success."""
    if not isinstance(resp, dict):
        return False
    return str(resp.get("status", "") or "").strip().lower() == "success"


def persist_sandbox_images_to_snapshot(
    *,
    resp: dict[str, Any],
    run_name: str | None,
    iteration: int | None,
    step: int | None,
    agent_id: str | None,
    session_id: str | None,
    job_id: str | None,
    cell_no: int | None,
    base_snapshots_dir: str = "snapshots",
    max_images_per_execution: int = DEFAULT_MAX_PLOTS_PER_EXECUTION,
    max_image_bytes: int = DEFAULT_MAX_PLOT_BYTES,
    graph_logic_cfg: dict[str, Any] | None = None,
    code: str | None = None,
) -> dict[str, Any]:
    """Persist image blobs from successful sandbox runs into snapshot plots/ files.

    Returns a dict with:
      - plot_paths: relative paths under iteration directory (e.g. plots/foo.png)
      - image_count: number of images persisted
      - skipped_count: number of skipped blobs
      - warnings: machine-friendly warning strings
      - manifest_path: relative manifest path when rows were appended
    """
    out: dict[str, Any] = {
        "plot_paths": [],
        "plot_filenames": [],
        "plot_handles": [],
        "plot_citations": [],
        "plot_summary_paths": [],
        "plot_artifacts": [],
        "image_count": 0,
        "skipped_count": 0,
        "warnings": [],
        "manifest_path": None,
        "summary_manifest_path": None,
        "plot_summary_backend": None,
        "plot_summary_stats_completeness": None,
    }

    if not isinstance(resp, dict):
        out["warnings"].append("response_not_dict")
        return out

    if not is_sandbox_execution_successful(resp):
        status = str(resp.get("status", "") or "").strip().lower() if isinstance(resp, dict) else ""
        out["warnings"].append(f"non_success_status:{status or 'unknown'}")
        return out

    if not run_name:
        out["warnings"].append("missing_run_name")
        return out

    if iteration is None:
        out["warnings"].append("missing_iteration")
        return out

    try:
        iteration_int = int(iteration)
    except Exception:
        out["warnings"].append(f"invalid_iteration:{iteration}")
        return out

    raw_images = resp.get("images") or []
    if not isinstance(raw_images, list) or not raw_images:
        return out

    max_images = _coerce_positive_int(
        max_images_per_execution,
        DEFAULT_MAX_PLOTS_PER_EXECUTION,
    )
    max_bytes = _coerce_positive_int(max_image_bytes, DEFAULT_MAX_PLOT_BYTES)
    graph_cfg = normalize_graph_logic_cfg(graph_logic_cfg)
    graph_enabled = is_graph_logic_enabled(graph_cfg)

    try:
        iteration_dir = resolve_iteration_snapshot_dir(
            run_name=run_name,
            iteration=iteration_int,
            base_snapshots_dir=base_snapshots_dir,
            create=True,
        )
        plots_dir = iteration_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = plots_dir / "manifest.jsonl"

        step_token = _sanitize_token(step, "na")
        agent_token = _sanitize_token(agent_id, "agent")
        sid_token = _sanitize_token(session_id, "sid")
        job_token = _sanitize_token(job_id, "job")
        cell_token = _sanitize_token(cell_no, "na")

        manifest_rows: list[dict[str, Any]] = []
        summary_manifest_rows: list[dict[str, Any]] = []
        persisted_paths: list[str] = []
        persisted_filenames: list[str] = []
        persisted_handles: list[str] = []
        persisted_citations: list[str] = []
        persisted_summary_paths: list[str] = []
        plot_artifacts: list[dict[str, Any]] = []
        summary_backends: list[str] = []
        stats_completeness_vals: list[float | None] = []
        warnings: list[str] = []
        skipped = 0

        for img_idx, img in enumerate(raw_images[:max_images], start=1):
            if not isinstance(img, dict):
                warnings.append(f"image_{img_idx}:not_dict")
                skipped += 1
                continue

            payload, decode_error = _decode_image_blob(
                str(img.get("data", "") or ""),
                max_image_bytes=max_bytes,
            )
            if payload is None:
                warnings.append(f"image_{img_idx}:{decode_error}")
                skipped += 1
                continue

            sha = hashlib.sha256(payload).hexdigest()
            filename = (
                f"{agent_token}_sid-{sid_token}_job-{job_token}_"
                f"cell-{cell_token}_step-{step_token}_img-{img_idx:02d}.png"
            )
            out_path = plots_dir / filename
            if out_path.exists():
                out_path = plots_dir / (f"{out_path.stem}_{sha[:10]}{out_path.suffix}")

            out_path.write_bytes(payload)
            rel_path = out_path.relative_to(iteration_dir).as_posix()
            persisted_paths.append(rel_path)
            plot_filename = None
            if graph_enabled:
                plot_filename = out_path.name
                persisted_filenames.append(plot_filename)

            row: dict[str, Any] = {
                "ts": datetime.now(UTC).isoformat(),
                "run_name": run_name,
                "iteration": iteration_int,
                "step": _int_or_none(step),
                "agent_id": str(agent_id) if agent_id is not None else None,
                "session_id": str(session_id) if session_id is not None else None,
                "job_id": str(job_id) if job_id is not None else None,
                "cell_no": _int_or_none(cell_no),
                "image_index": img_idx,
                "filename": out_path.name,
                "path": rel_path,
                "bytes": len(payload),
                "sha256": sha,
            }
            if plot_filename:
                row["plot_filename"] = plot_filename
            manifest_rows.append(row)

            if graph_enabled and plot_filename:
                summary: dict[str, Any]
                try:
                    candidate = generate_plot_summary_for_image(
                        plot_filename=plot_filename,
                        plot_path=rel_path,
                        image_index=img_idx,
                        code=code,
                        resp=resp,
                        graph_logic_cfg=graph_cfg,
                    )
                    if not isinstance(candidate, dict):
                        raise TypeError("generate_plot_summary_for_image must return a dict")
                    summary = dict(candidate)
                except Exception as e:
                    warnings.append(
                        f"plot_summary_generation_failed:{plot_filename}:{type(e).__name__}:{e}"
                    )
                    summary = build_fallback_plot_summary(
                        plot_filename=plot_filename,
                        plot_path=rel_path,
                        warnings=[f"summary_generation_failed:{type(e).__name__}:{e}"],
                    )
                summary_path = None
                try:
                    summary_path = write_plot_summary_artifact(
                        iteration_dir=iteration_dir,
                        plot_filename=plot_filename,
                        summary=summary,
                    )
                except Exception as e:
                    warnings.append(
                        f"plot_summary_write_failed:{plot_filename}:{type(e).__name__}:{e}"
                    )
                if summary_path:
                    persisted_summary_paths.append(summary_path)
                    artifact = build_plot_artifact_record(
                        plot_filename=plot_filename,
                        plot_path=rel_path,
                        summary_path=summary_path,
                        summary=summary,
                    )
                    plot_artifacts.append(artifact)
                    plot_handle = str(artifact.get("plot_handle", "") or "").strip()
                    plot_citation = str(artifact.get("plot_citation", "") or "").strip()
                    if plot_handle:
                        persisted_handles.append(plot_handle)
                    if plot_citation:
                        persisted_citations.append(plot_citation)
                    summary_backends.append(str(summary.get("generation_backend") or "unknown"))
                    stats_obj = (
                        summary.get("statistics", {})
                        if isinstance(summary.get("statistics", {}), dict)
                        else {}
                    )
                    try:
                        stats_val = float(stats_obj.get("stats_completeness"))
                    except Exception:
                        stats_val = None
                    stats_completeness_vals.append(stats_val)
                    summary_manifest_rows.append(
                        {
                            "ts": datetime.now(UTC).isoformat(),
                            "run_name": run_name,
                            "iteration": iteration_int,
                            "step": _int_or_none(step),
                            "agent_id": str(agent_id) if agent_id is not None else None,
                            "session_id": str(session_id) if session_id is not None else None,
                            "job_id": str(job_id) if job_id is not None else None,
                            "cell_no": _int_or_none(cell_no),
                            "image_index": img_idx,
                            "plot_filename": plot_filename,
                            "plot_handle": str(artifact.get("plot_handle", "") or "").strip(),
                            "plot_citation": str(artifact.get("plot_citation", "") or "").strip(),
                            "plot_path": rel_path,
                            "summary_path": summary_path,
                            "generation_backend": str(
                                summary.get("generation_backend") or "unknown"
                            ),
                            "stats_completeness": stats_obj.get("stats_completeness"),
                        }
                    )

        if len(raw_images) > max_images:
            dropped = len(raw_images) - max_images
            warnings.append(f"max_images_limit:{max_images}:dropped_{dropped}")
            skipped += dropped

        if manifest_rows:
            _append_jsonl_with_lock(manifest_path, manifest_rows)
            out["manifest_path"] = manifest_path.relative_to(iteration_dir).as_posix()
        if summary_manifest_rows:
            summary_manifest_path = plots_dir / "summary_manifest.jsonl"
            _append_jsonl_with_lock(summary_manifest_path, summary_manifest_rows)
            out["summary_manifest_path"] = summary_manifest_path.relative_to(
                iteration_dir
            ).as_posix()

        out["plot_paths"] = persisted_paths
        out["plot_filenames"] = persisted_filenames
        out["plot_handles"] = persisted_handles
        out["plot_citations"] = persisted_citations
        out["plot_summary_paths"] = persisted_summary_paths
        out["plot_artifacts"] = plot_artifacts
        out["image_count"] = len(persisted_paths)
        out["skipped_count"] = skipped
        out["warnings"] = warnings
        if summary_backends:
            uniq = sorted(set(summary_backends))
            out["plot_summary_backend"] = uniq[0] if len(uniq) == 1 else "mixed"
        out["plot_summary_stats_completeness"] = average_values(stats_completeness_vals)
        return out
    except Exception as e:
        out["warnings"].append(f"persist_error:{type(e).__name__}:{e}")
        return out


def extract_sandbox_response(
    resp: dict[str, Any], trim: bool = True, max_chars: int = 10000
) -> tuple[str, list[str], bool]:
    """Convert a sandbox response into LLM-friendly text plus images."""

    def _trim_if_needed(text: str, field_name: str) -> str:
        if not trim or not text or len(text) <= max_chars:
            return text
        return (
            text[:max_chars]
            + f"\n... (trimmed observation due to length, original {field_name} was {len(text)} chars)"
        )

    parts: list[str] = []
    status = resp.get("status", "unknown")
    parts.append(f"Status: {status}")
    was_successful = status == "success"

    stdout = resp.get("stdout")
    if stdout:
        parts.append(f"stdout: {_trim_if_needed(stdout.strip(), 'stdout')}")

    stderr = resp.get("stderr")
    if stderr:
        parts.append(f"stderr: {_trim_if_needed(stderr.strip(), 'stderr')}")

    result = resp.get("result")
    if result:
        parts.append(f"result: {_trim_if_needed(str(result), 'result')}")

    # ---- NEW: expose images to the model + caller ----
    image_blobs: list[str] = []
    raw_images = resp.get("images") or []

    if isinstance(raw_images, list) and raw_images:
        img_summaries = []
        for idx, img in enumerate(raw_images, start=1):
            if not isinstance(img, dict):
                continue
            img_type = img.get("type", "png")
            img_summaries.append({"id": idx, "type": img_type})
            data = img.get("data")
            if isinstance(data, str):
                image_blobs.append(data)

        # Only append IMAGES sentinel if we actually found any
        if img_summaries:
            parts.append("IMAGES = " + json.dumps(img_summaries, ensure_ascii=False))

    # text seen by the LLM; image_blobs usable by the UI layer
    return "\n".join(parts), image_blobs, was_successful


def get_session_id(resp: dict[str, Any]) -> str | None:
    """Extract the session_id from a sandbox response if present."""
    try:
        meta = resp.get("session_metadata") or {}
        sid = meta.get("session_id")
        if sid:
            return str(sid)
    except Exception:
        pass
    return None
