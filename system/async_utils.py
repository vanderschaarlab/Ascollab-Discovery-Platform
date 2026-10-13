"""Async utilities for safe event-loop management.

Provides ``safe_asyncio_run`` — a drop-in replacement for ``asyncio.run()``
that won't hang when orphaned thread-pool threads outlive the main coroutine.
Also provides ``patch_arxiv_client_timeout`` to inject HTTP timeouts into the
``arxiv`` library which otherwise blocks indefinitely.
"""

import asyncio
import logging
from typing import TypeVar

import requests.adapters

_T = TypeVar("_T")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# safe_asyncio_run
# ---------------------------------------------------------------------------


def safe_asyncio_run(coro: object) -> _T:
    """Like ``asyncio.run()`` but won't hang on orphaned thread-pool threads.

    When ``asyncio.wait_for()`` cancels a task that had synchronous functions
    running in the default ``ThreadPoolExecutor``, those OS threads cannot be
    interrupted.  The stdlib ``asyncio.run()`` calls
    ``executor.shutdown(wait=True)`` during cleanup, blocking forever.

    This helper replaces that with ``shutdown(wait=False, cancel_futures=True)``
    so the event loop closes promptly.  Orphaned threads continue in the
    background until their I/O naturally completes (bounded by the HTTP
    timeouts we set elsewhere).
    """
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        try:
            # 1. Cancel remaining asyncio tasks (mirrors asyncio._cancel_all_tasks)
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

            # 2. Shutdown async generators
            loop.run_until_complete(loop.shutdown_asyncgens())

            # 3. Shutdown executor WITHOUT waiting for orphaned threads.
            #    wait=False  : don't block on running threads
            #    cancel_futures=True : cancel queued-but-not-started work
            if hasattr(loop, "_default_executor") and loop._default_executor is not None:
                n_threads = len(getattr(loop._default_executor, "_threads", set()))
                if n_threads:
                    log.info(
                        "[safe_asyncio_run] shutting down executor with %d active "
                        "thread(s) (not waiting)",
                        n_threads,
                    )
                loop._default_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        finally:
            asyncio.set_event_loop(None)
            loop.close()


# ---------------------------------------------------------------------------
# arxiv HTTP timeout patch
# ---------------------------------------------------------------------------


class _TimeoutHTTPAdapter(requests.adapters.HTTPAdapter):
    """``requests`` transport adapter that enforces a default timeout."""

    def __init__(self, timeout: float = 30, **kwargs):
        self.timeout = timeout
        super().__init__(**kwargs)

    def send(self, *args, **kwargs):
        kwargs.setdefault("timeout", self.timeout)
        return super().send(*args, **kwargs)


_ARXIV_PATCHED = False


def patch_arxiv_client_timeout(timeout: float = 30) -> None:
    """Monkey-patch ``arxiv.Client`` so every HTTP request has a timeout.

    The ``arxiv`` library creates a bare ``requests.Session()`` and calls
    ``session.get(url)`` with **no** ``timeout`` parameter, which can block
    indefinitely.  This patches ``Client.__init__`` to mount a
    ``_TimeoutHTTPAdapter`` on every new session.

    Safe to call multiple times; only the first call takes effect.
    """
    global _ARXIV_PATCHED
    if _ARXIV_PATCHED:
        return

    try:
        import arxiv
    except ImportError:
        return

    _orig_init = arxiv.Client.__init__

    def _patched_init(self, *a, **kw):
        _orig_init(self, *a, **kw)
        adapter = _TimeoutHTTPAdapter(timeout=timeout)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

    arxiv.Client.__init__ = _patched_init  # type: ignore[assignment]
    _ARXIV_PATCHED = True
    log.info("[async_utils] patched arxiv.Client with HTTP timeout=%ss", timeout)
