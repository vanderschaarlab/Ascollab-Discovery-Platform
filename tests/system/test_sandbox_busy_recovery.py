import importlib
import json
import sys
import threading
import time
import uuid
from pathlib import Path


class _FakeContainer:
    def __init__(self, name: str):
        self.id = uuid.uuid4().hex
        self.name = name
        self.status = "running"
        self.removed = False

    def reload(self):
        return None

    def logs(self, tail=200):
        return b""

    def remove(self, force=True):
        self.removed = True
        self.status = "removed"


class _FakeContainers:
    def __init__(self):
        self.by_id = {}
        self.created = []

    def run(self, **kwargs):
        c = _FakeContainer(kwargs.get("name", "sandbox_fake"))
        self.by_id[c.id] = c
        self.created.append(c)
        return c

    def get(self, container_id):
        return self.by_id[container_id]


class _FakeDockerClient:
    def __init__(self):
        self.containers = _FakeContainers()


def _load_server_docker(monkeypatch, tmp_path):
    import docker

    fake_client = _FakeDockerClient()
    monkeypatch.setattr(docker, "from_env", lambda timeout=10: fake_client)
    monkeypatch.setenv("SANDBOX_SESSIONS_ROOT", str(tmp_path / "sessions"))
    monkeypatch.setenv("SANDBOX_RUN_ID", "test")
    monkeypatch.setenv("SANDBOX_DATASETS_DIR", str(tmp_path / "datasets"))
    monkeypatch.setenv("SANDBOX_MAX_SESSIONS", "4")
    monkeypatch.setenv("SANDBOX_DEFAULT_TIMEOUT", "1")

    sys.modules.pop("sandbox.server_docker", None)
    module = importlib.import_module("sandbox.server_docker")
    return module, fake_client


def _registry_and_runner(module):
    registry = module.app.config["sandbox_registry"]
    runner = module.app.config["sandbox_runner"]
    return registry, runner


def _write_next_outbox(workdir: str, payload: dict, *, timeout: float = 2.0):
    inbox = Path(workdir) / "inbox"
    outbox = Path(workdir) / "outbox"
    deadline = time.time() + timeout
    while time.time() < deadline:
        jobs = sorted(inbox.glob("*.json"))
        if jobs:
            target = outbox / jobs[0].name
            tmp = outbox / f"{jobs[0].name}.tmp"
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            tmp.replace(target)
            return
        time.sleep(0.01)
    raise AssertionError("runner did not create an inbox job")


def test_docker_timeout_restart_does_not_leave_session_busy(monkeypatch, tmp_path):
    module, fake_client = _load_server_docker(monkeypatch, tmp_path)
    registry, runner = _registry_and_runner(module)
    session = registry.try_create_nonblocking()

    original_perf_counter = module.time.perf_counter
    original_sleep = module.time.sleep
    clock = {"now": 0.0}

    def fast_perf_counter():
        clock["now"] += 2.1
        return clock["now"]

    monkeypatch.setattr(module.time, "perf_counter", fast_perf_counter)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    response = runner.run(session.session_id, "while True: pass", timeout=1)
    monkeypatch.setattr(module.time, "perf_counter", original_perf_counter)
    monkeypatch.setattr(module.time, "sleep", original_sleep)

    assert response["status"] == "error"
    assert response["timed_out"] is True
    assert response["restarted"] is True
    assert not session.lock.locked()
    listed = registry.list()
    assert listed[0]["status"] == "idle"
    assert len(fake_client.containers.created) == 2

    writer = threading.Thread(
        target=_write_next_outbox,
        args=(session.host_workdir, {"status": "ok", "stdout": "ready"}),
    )
    writer.start()
    second = runner.run(session.session_id, "print('ready')", timeout=1)
    writer.join(timeout=2)

    assert second["status"] == "ok"
    assert second["job_id"]
    assert registry.list()[0]["status"] == "idle"


def test_docker_busy_rejection_is_structured(monkeypatch, tmp_path):
    module, _fake_client = _load_server_docker(monkeypatch, tmp_path)
    registry, runner = _registry_and_runner(module)
    session = registry.try_create_nonblocking()

    session.lock.acquire()
    try:
        session.mark_run_started(job_id="job-1", code="print(1)", timeout=10)
        response = runner.run(session.session_id, "print(2)", timeout=1)
    finally:
        session.clear_run_state()
        session.lock.release()

    assert response["status"] == "error"
    assert response["busy_retryable"] is True
    assert response["current_job_id"] == "job-1"
    assert response["busy_rejects"] == 1


def test_docker_shutdown_busy_session_is_bounded_and_forced(monkeypatch, tmp_path):
    module, _fake_client = _load_server_docker(monkeypatch, tmp_path)
    registry, _runner = _registry_and_runner(module)
    session = registry.try_create_nonblocking()

    session.lock.acquire()
    try:
        result = registry.shutdown(session.session_id, lock_timeout=0.01)
    finally:
        session.lock.release()

    assert result["status"] == "ok"
    assert result["forced"] is True
    assert registry.list() == []


def test_docker_reap_stuck_session(monkeypatch, tmp_path):
    module, _fake_client = _load_server_docker(monkeypatch, tmp_path)
    registry, _runner = _registry_and_runner(module)
    session = registry.try_create_nonblocking()
    session.meta["owner"] = "research_t1"

    session.lock.acquire()
    try:
        session.mark_run_started(job_id="stuck-job", code="sleep()", timeout=1)
        session.busy_started_monotonic = time.perf_counter() - 100
        result = registry.reap_stuck(owner="research_t1", max_busy_seconds=1)
    finally:
        session.lock.release()

    assert result["status"] == "ok"
    assert len(result["reaped"]) == 1
    assert result["reaped"][0]["forced"] is True
    assert registry.list() == []


class _BusyResponse:
    status_code = 200
    text = ""

    def json(self):
        return {
            "status": "error",
            "stderr": "Session is currently busy executing another request.",
            "session_metadata": {},
        }


class _OkResponse:
    status_code = 200
    text = ""

    def json(self):
        return {"status": "ok", "stdout": "done", "session_metadata": {}}


def test_client_busy_retry_budget_is_bounded(monkeypatch):
    from env_setup.sandbox_utils import SandboxClient

    client = SandboxClient(base_url="http://sandbox.invalid", code_timeout_seconds=1)
    monkeypatch.setattr(client.session, "post", lambda *args, **kwargs: _BusyResponse())

    response = client.run(
        "sid",
        "print(1)",
        retries=100,
        retry_delay=0.01,
        max_retry_delay=0.01,
        busy_retry_timeout_seconds=0,
    )

    assert response["status"] == "error"
    assert "retry budget was exhausted" in response["stderr"]
    assert response["busy_retry_attempts"] == 1


def test_client_transient_busy_can_recover(monkeypatch):
    from env_setup.sandbox_utils import SandboxClient

    client = SandboxClient(base_url="http://sandbox.invalid", code_timeout_seconds=1)
    responses = [_BusyResponse(), _OkResponse()]
    monkeypatch.setattr(client.session, "post", lambda *args, **kwargs: responses.pop(0))
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    response = client.run(
        "sid",
        "print(1)",
        retries=2,
        retry_delay=0.01,
        max_retry_delay=0.01,
        busy_retry_timeout_seconds=5,
    )

    assert response["status"] == "ok"
    assert response["session_metadata"]["session_id"] == "sid"


def test_client_admin_helpers_send_bounded_control_payloads():
    from env_setup.sandbox_utils import SandboxClient

    class Response:
        status_code = 200
        text = ""

        def json(self):
            return {"status": "ok"}

    client = SandboxClient(base_url="http://sandbox.invalid", code_timeout_seconds=1)
    seen = []

    def post(url, headers=None, json=None, timeout=None):
        seen.append({"url": url, "json": json, "timeout": timeout})
        return Response()

    client.session.post = post

    assert client.restart("sid", lock_timeout=0.25)["status"] == "ok"
    assert seen[-1]["json"] == {"session_id": "sid", "lock_timeout": 0.25}

    assert client.shutdown("sid", lock_timeout=0.5)["status"] == "ok"
    assert seen[-1]["json"] == {"session_id": "sid", "lock_timeout": 0.5}

    assert client.reap_stuck_sessions(owner="research_t1", max_busy_seconds=3)["status"] == "ok"
    assert seen[-1]["json"] == {"owner": "research_t1", "max_busy_seconds": 3.0}


def test_sticky_session_serializes_local_runs():
    from env_setup.sandbox_utils import StickySession

    class FakeClient:
        def __init__(self):
            self.active = 0
            self.max_active = 0
            self.calls = 0
            self.lock = threading.Lock()

        def create_session(self, **kwargs):
            return "sid"

        def run(self, session_id, code, **kwargs):
            with self.lock:
                self.calls += 1
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.02)
            with self.lock:
                self.active -= 1
            return {"status": "ok", "session_metadata": {"session_id": session_id}}

        def shutdown(self, session_id):
            return {"status": "ok"}

    fake = FakeClient()
    sticky = StickySession(fake)
    threads = [threading.Thread(target=sticky.run, args=("print(1)",)) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=2)
    sticky.close()

    assert fake.calls == 2
    assert fake.max_active == 1
