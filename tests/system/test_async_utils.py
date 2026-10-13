"""Tests for system.async_utils — safe asyncio runner and arxiv timeout patch."""

import asyncio
import threading
import time

import pytest


class TestSafeAsyncioRun:
    """Verify safe_asyncio_run doesn't hang when orphaned executor threads exist."""

    def test_does_not_hang_on_orphaned_threads(self):
        """Reproduce the exact hang: a sync function in the default executor
        outlives the main coroutine.  stdlib asyncio.run() would hang at
        shutdown_default_executor(wait=True).  safe_asyncio_run must return
        within a few seconds."""
        from system.async_utils import safe_asyncio_run

        blocker_started = threading.Event()
        stop = threading.Event()  # lets us clean up the thread after the test

        def blocking_sync_function():
            """Simulates an orphaned thread doing slow I/O."""
            blocker_started.set()
            stop.wait(60)  # interruptible sleep — cleared when test finishes
            return "done"

        async def main():
            loop = asyncio.get_running_loop()
            # Submit work to the default executor (same path as LangChain tools)
            fut = loop.run_in_executor(None, blocking_sync_function)
            # Wait for the thread to actually start
            while not blocker_started.is_set():
                await asyncio.sleep(0.01)
            # Abandon the future (simulates asyncio.wait_for cancel)
            fut.cancel()
            return 42

        t0 = time.monotonic()
        result = safe_asyncio_run(main())
        elapsed = time.monotonic() - t0
        stop.set()  # release the orphaned thread so it exits promptly

        assert result == 42
        # Must complete quickly — not wait 60s for the sleeping thread
        assert elapsed < 5, (
            f"safe_asyncio_run took {elapsed:.1f}s, likely hung on executor shutdown"
        )

    def test_returns_value_from_normal_coroutine(self):
        """Basic sanity: safe_asyncio_run returns the coroutine's result."""
        from system.async_utils import safe_asyncio_run

        async def simple():
            return "hello"

        assert safe_asyncio_run(simple()) == "hello"

    def test_propagates_exceptions(self):
        """Exceptions from the coroutine must propagate normally."""
        from system.async_utils import safe_asyncio_run

        async def failing():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            safe_asyncio_run(failing())

    def test_multiple_orphaned_threads(self):
        """Multiple orphaned threads should not cause issues."""
        from system.async_utils import safe_asyncio_run

        started = threading.Event()
        stop = threading.Event()
        count = 0

        def slow_sync():
            nonlocal count
            count += 1
            if count >= 3:
                started.set()
            stop.wait(60)

        async def main():
            loop = asyncio.get_running_loop()
            futs = [loop.run_in_executor(None, slow_sync) for _ in range(5)]
            while not started.is_set():
                await asyncio.sleep(0.01)
            for f in futs:
                f.cancel()
            return "ok"

        t0 = time.monotonic()
        result = safe_asyncio_run(main())
        elapsed = time.monotonic() - t0
        stop.set()

        assert result == "ok"
        assert elapsed < 5


class TestArxivTimeoutPatch:
    """Verify that patch_arxiv_client_timeout injects HTTP timeouts."""

    def test_patched_client_has_timeout_adapter(self):
        """After patching, new arxiv.Client sessions must use TimeoutHTTPAdapter."""
        import arxiv

        from system.async_utils import (
            _TimeoutHTTPAdapter,
            patch_arxiv_client_timeout,
        )

        patch_arxiv_client_timeout(timeout=42)

        client = arxiv.Client()
        adapter = client._session.get_adapter("https://export.arxiv.org")
        assert isinstance(adapter, _TimeoutHTTPAdapter)
        assert adapter.timeout == 42

    def test_timeout_adapter_injects_default(self):
        """The adapter must inject the timeout kwarg into send() calls."""
        from unittest.mock import MagicMock, patch

        from system.async_utils import _TimeoutHTTPAdapter

        adapter = _TimeoutHTTPAdapter(timeout=15)
        with patch("requests.adapters.HTTPAdapter.send", return_value=MagicMock()) as mock_send:
            adapter.send(MagicMock(), stream=False)
            _, kwargs = mock_send.call_args
            assert kwargs.get("timeout") == 15

    def test_explicit_timeout_not_overridden(self):
        """If caller already passes timeout, the adapter must not override it."""
        from unittest.mock import MagicMock, patch

        from system.async_utils import _TimeoutHTTPAdapter

        adapter = _TimeoutHTTPAdapter(timeout=15)
        with patch("requests.adapters.HTTPAdapter.send", return_value=MagicMock()) as mock_send:
            adapter.send(MagicMock(), timeout=99)
            _, kwargs = mock_send.call_args
            # setdefault should preserve the caller's explicit value
            assert kwargs.get("timeout") == 99


class TestConfigIntegration:
    """Verify that YAML configs now have the expected timeout fields."""

    @staticmethod
    def _assert_positive_timeout(cfg, *, field: str, label: str) -> None:
        value = cfg.get(field, None)
        if value is None:
            pytest.fail(f"{label} missing '{field}' field")
        try:
            timeout = int(value)
        except (TypeError, ValueError):
            pytest.fail(f"{label} has non-integer '{field}' field")
        if timeout <= 0:
            pytest.fail(f"{label} has non-positive '{field}' field")

    def test_omegaconf_failure_repr_redacts_api_keys(self):
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(
            {
                "api_key": "raw-secret-value",
                "nested": {"access_token": "raw-token-value", "safe": "visible"},
            }
        )

        rendered = repr(cfg)
        assert "raw-secret-value" not in rendered
        assert "raw-token-value" not in rendered
        assert "***REDACTED***" in rendered
        assert "visible" in rendered

    @pytest.mark.parametrize(
        "config_name",
        [
            "default",
            "gpt-5",
            "gpt-5-high",
            "deepseek",
            "llama",
            "gpt-4o",
        ],
    )
    def test_llm_configs_have_timeout(self, config_name):
        from pathlib import Path

        from omegaconf import OmegaConf

        cfg_path = Path("sys_config/llm_api") / f"{config_name}.yaml"
        if not cfg_path.exists():
            pytest.skip(f"{cfg_path} not found")
        cfg = OmegaConf.load(cfg_path)
        self._assert_positive_timeout(cfg, field="timeout", label=f"{config_name}.yaml")

    def test_embedder_config_has_timeout(self):
        from pathlib import Path

        from omegaconf import OmegaConf

        cfg_path = Path("sys_config/embedder_api/default.yaml")
        if not cfg_path.exists():
            pytest.skip("embedder config not found")
        cfg = OmegaConf.load(cfg_path)
        self._assert_positive_timeout(cfg, field="timeout", label="embedder default.yaml")

    def test_base_config_has_arxiv_http_timeout(self):
        from pathlib import Path

        from omegaconf import OmegaConf

        cfg_path = Path("sys_config/ascollab.yaml")
        if not cfg_path.exists():
            pytest.skip("ascollab.yaml not found")
        cfg = OmegaConf.load(cfg_path)
        assert "arxiv_http_timeout_seconds" in cfg.tools
        assert int(cfg.tools.arxiv_http_timeout_seconds) > 0
