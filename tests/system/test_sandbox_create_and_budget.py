"""Sandbox reliability regressions: install budget, lock leak, container creation.

Reuses the fake-Docker harness shape from ``test_sandbox_busy_recovery`` so the
registry can be exercised without a Docker daemon.
"""

import importlib
import json
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest


class _FakeContainer:
    def __init__(self, name: str, labels: dict | None = None):
        self.id = uuid.uuid4().hex
        self.name = name
        self.labels = labels or {}
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
        self.removed_ids = []

    def run(self, **kwargs):
        c = _FakeContainer(kwargs.get("name", "sandbox_fake"), kwargs.get("labels"))
        self.by_id[c.id] = c
        self.created.append(c)
        return c

    def get(self, container_id):
        return self.by_id[container_id]


class _FakeDockerClient:
    def __init__(self):
        self.containers = _FakeContainers()


def _load_server_docker(monkeypatch, tmp_path, **env):
    import docker

    fake_client = _FakeDockerClient()
    monkeypatch.setattr(docker, "from_env", lambda timeout=10: fake_client)
    monkeypatch.setenv("SANDBOX_SESSIONS_ROOT", str(tmp_path / "sessions"))
    monkeypatch.setenv("SANDBOX_RUN_ID", "test")
    monkeypatch.setenv("SANDBOX_DATASETS_DIR", str(tmp_path / "datasets"))
    monkeypatch.setenv("SANDBOX_MAX_SESSIONS", "2")
    monkeypatch.setenv("SANDBOX_DEFAULT_TIMEOUT", "1")
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    sys.modules.pop("sandbox.server_docker", None)
    module = importlib.import_module("sandbox.server_docker")
    return module, fake_client


def _reply_to_next_job(workdir: str, payload: dict, *, timeout: float = 2.0):
    inbox, outbox = Path(workdir) / "inbox", Path(workdir) / "outbox"
    deadline = time.time() + timeout
    while time.time() < deadline:
        jobs = sorted(inbox.glob("*.json"))
        if jobs:
            request = json.loads(jobs[0].read_text())
            tmp = outbox / f"{jobs[0].name}.tmp"
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            tmp.replace(outbox / jobs[0].name)
            return request
        time.sleep(0.01)
    raise AssertionError("runner did not create an inbox job")


# ---------------------------------------------------------------------------
# Install budget reaches the daemon and widens the server deadline
# ---------------------------------------------------------------------------


def test_install_budget_is_sent_to_the_daemon_when_installs_are_possible(monkeypatch, tmp_path):
    module, _client = _load_server_docker(
        monkeypatch,
        tmp_path,
        SANDBOX_NETWORK_DISABLED="false",
        SANDBOX_INSTALL_BUDGET_S="90",
    )
    registry = module.app.config["sandbox_registry"]
    runner = module.app.config["sandbox_runner"]
    session = registry.try_create_nonblocking()

    captured: dict = {}

    def reply():
        captured.update(
            _reply_to_next_job(session.host_workdir, {"status": "ok", "session_metadata": {}})
        )

    writer = threading.Thread(target=reply)
    writer.start()
    response = runner.run(session.session_id, "import cowsay", timeout=10)
    writer.join(timeout=3)

    assert response["status"] == "ok"
    assert captured["install_budget"] == 90.0
    assert captured["timeout"] == 10.0
    # Execution timeout and install allowance are separate numbers on the wire;
    # the daemon needs both because only the former arms its alarm.
    assert captured["auto_install"] is True


def test_no_install_allowance_when_the_network_is_disabled(monkeypatch, tmp_path):
    module, _client = _load_server_docker(monkeypatch, tmp_path, SANDBOX_INSTALL_BUDGET_S="90")
    registry = module.app.config["sandbox_registry"]
    runner = module.app.config["sandbox_runner"]
    session = registry.try_create_nonblocking()

    captured: dict = {}

    def reply():
        captured.update(
            _reply_to_next_job(session.host_workdir, {"status": "ok", "session_metadata": {}})
        )

    writer = threading.Thread(target=reply)
    writer.start()
    runner.run(session.session_id, "import cowsay", timeout=10)
    writer.join(timeout=3)

    assert captured["install_budget"] == 0.0


def test_mark_run_started_records_the_total_budget(monkeypatch, tmp_path):
    module, _client = _load_server_docker(monkeypatch, tmp_path)
    registry = module.app.config["sandbox_registry"]
    session = registry.try_create_nonblocking()

    session.mark_run_started(job_id="j", code="x = 1", timeout=120.0, total_budget=300.0)

    assert session.current_timeout == 120.0
    assert session.current_total_budget == 300.0
    assert session.metadata()["current_total_budget"] == 300.0


def test_reap_stuck_uses_the_total_budget_not_the_execution_timeout(monkeypatch, tmp_path):
    module, _client = _load_server_docker(monkeypatch, tmp_path)
    registry = module.app.config["sandbox_registry"]
    session = registry.try_create_nonblocking()

    session.lock.acquire()
    try:
        session.mark_run_started(job_id="j", code="x = 1", timeout=120.0, total_budget=300.0)
        # 400s busy: above 2*120+60 = 300 (old threshold) but below 2*300+60 = 660.
        session.busy_started_monotonic = time.perf_counter() - 400
        result = registry.reap_stuck()
    finally:
        session.lock.release()

    assert result["reaped"] == []
    assert result["skipped"][0]["reason"] == "below_threshold"
    assert result["skipped"][0]["threshold"] == pytest.approx(660.0)


# ---------------------------------------------------------------------------
# clear_run_state failure must not skip the lock release
# ---------------------------------------------------------------------------


def test_clear_run_state_failure_still_releases_the_session_lock(monkeypatch, tmp_path):
    module, _client = _load_server_docker(monkeypatch, tmp_path)
    registry = module.app.config["sandbox_registry"]
    runner = module.app.config["sandbox_runner"]
    session = registry.try_create_nonblocking()

    def boom(self):
        raise RuntimeError("state clear exploded")

    monkeypatch.setattr(module.Session, "clear_run_state", boom)

    writer = threading.Thread(
        target=_reply_to_next_job,
        args=(session.host_workdir, {"status": "ok", "session_metadata": {}}),
    )
    writer.start()
    response = runner.run(session.session_id, "print(1)", timeout=2)
    writer.join(timeout=3)

    assert response["status"] == "ok"
    assert session.lock.locked() is False


# ---------------------------------------------------------------------------
# Container creation outside the registry lock, without leaking
# ---------------------------------------------------------------------------


def test_failed_create_removes_the_session_and_returns_the_slot(monkeypatch, tmp_path):
    module, _client = _load_server_docker(monkeypatch, tmp_path)
    registry = module.app.config["sandbox_registry"]

    real_create = registry.cm.create_container
    failing = {"on": True}

    def flaky_create(workdir, session_id=""):
        if failing["on"]:
            raise RuntimeError("docker down")
        return real_create(workdir, session_id=session_id)

    monkeypatch.setattr(registry.cm, "create_container", flaky_create)

    # max_sessions is 2, so exhausting both slots on failures then succeeding
    # twice proves no slot was lost.
    for _ in range(2):
        with pytest.raises(RuntimeError, match="docker down"):
            registry.try_create_nonblocking()
    assert registry.list() == []

    failing["on"] = False
    assert registry.try_create_nonblocking() is not None
    assert registry.try_create_nonblocking() is not None
    assert len(registry.list()) == 2


def test_session_is_visible_while_its_container_is_being_created(monkeypatch, tmp_path):
    module, _client = _load_server_docker(monkeypatch, tmp_path)
    registry = module.app.config["sandbox_registry"]

    release = threading.Event()
    entered = threading.Event()
    real_create = registry.cm.create_container

    def blocking_create(workdir, session_id=""):
        entered.set()
        release.wait(timeout=5)
        return real_create(workdir, session_id=session_id)

    monkeypatch.setattr(registry.cm, "create_container", blocking_create)

    created: list = []
    worker = threading.Thread(target=lambda: created.append(registry.try_create_nonblocking()))
    worker.start()
    assert entered.wait(timeout=5)

    # This is the assertion that catches the leak: the registry answers while
    # Docker is still working, and the pending session is listed as `creating`.
    listed = registry.list()
    assert len(listed) == 1
    assert listed[0]["creating"] is True

    release.set()
    worker.join(timeout=5)
    assert created[0] is not None
    assert registry.list()[0]["creating"] is False


def test_shutdown_during_creation_removes_the_real_container(monkeypatch, tmp_path):
    module, client = _load_server_docker(monkeypatch, tmp_path)
    registry = module.app.config["sandbox_registry"]

    release = threading.Event()
    entered = threading.Event()
    real_create = registry.cm.create_container

    def blocking_create(workdir, session_id=""):
        entered.set()
        release.wait(timeout=5)
        return real_create(workdir, session_id=session_id)

    monkeypatch.setattr(registry.cm, "create_container", blocking_create)

    outcome: list = []

    def create():
        try:
            outcome.append(registry.try_create_nonblocking())
        except Exception as exc:
            outcome.append(exc)

    worker = threading.Thread(target=create)
    worker.start()
    assert entered.wait(timeout=5)

    sid = registry.list()[0]["session_id"]
    registry.shutdown(sid, lock_timeout=0.1)
    release.set()
    worker.join(timeout=5)

    assert isinstance(outcome[0], RuntimeError)
    assert registry.list() == []
    # The container Docker did eventually create was cleaned up, not orphaned.
    assert client.containers.created[-1].removed is True
    # And the slot is free again despite shutdown having released it too.
    assert registry.try_create_nonblocking() is not None


def test_concurrent_creates_are_not_serialised_by_the_registry_lock(monkeypatch, tmp_path):
    module, _client = _load_server_docker(monkeypatch, tmp_path, SANDBOX_MAX_SESSIONS="5")
    registry = module.app.config["sandbox_registry"]

    per_create = 0.15
    real_create = registry.cm.create_container

    def slow_create(workdir, session_id=""):
        time.sleep(per_create)
        return real_create(workdir, session_id=session_id)

    monkeypatch.setattr(registry.cm, "create_container", slow_create)

    started = time.perf_counter()
    threads = [threading.Thread(target=registry.try_create_nonblocking) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    elapsed = time.perf_counter() - started

    assert len(registry.list()) == 5
    # Serialised would be ~5 x per_create; overlapped should be close to one.
    assert elapsed < 3 * per_create


def test_containers_are_labelled_for_manual_orphan_cleanup(monkeypatch, tmp_path):
    module, client = _load_server_docker(monkeypatch, tmp_path)
    registry = module.app.config["sandbox_registry"]

    session = registry.try_create_nonblocking()

    labels = client.containers.created[-1].labels
    assert labels["ascollab.sandbox"] == "1"
    assert labels["ascollab.session_id"] == session.session_id
