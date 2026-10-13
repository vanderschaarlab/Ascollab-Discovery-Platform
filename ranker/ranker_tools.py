"""Tool factories for the ranker agent (arXiv search, sandboxed code execution)."""

import logging
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# arXiv tool
# ---------------------------------------------------------------------------


class _ArxivArgs(BaseModel):
    query: str = Field(..., description="arXiv search query string")


def make_arxiv_tool(cfg: dict[str, Any]) -> StructuredTool:
    """Create an arXiv search tool from config.

    Args:
        cfg: Dict with optional keys ``top_k_results`` and ``load_max_docs``.

    Returns:
        A StructuredTool wrapping arXiv search.
    """
    from langchain_community.tools.arxiv.tool import ArxivQueryRun
    from langchain_community.utilities.arxiv import ArxivAPIWrapper

    wrapper = ArxivAPIWrapper(
        top_k_results=cfg.get("top_k_results", 5),
        load_max_docs=cfg.get("load_max_docs", 5),
    )
    underlying = ArxivQueryRun(api_wrapper=wrapper)
    log.debug("arXiv tool created (top_k=%d)", cfg.get("top_k_results", 5))

    def _run(query: str) -> str:
        return underlying.run({"query": query})

    return StructuredTool.from_function(
        name="arxiv",
        func=_run,
        description="Search arXiv and return summaries of top results.",
        args_schema=_ArxivArgs,
        return_direct=False,
    )


# ---------------------------------------------------------------------------
# Sandbox code-execution tool
# ---------------------------------------------------------------------------


class _SandboxArgs(BaseModel):
    code: str = Field(..., description="Python code to execute in the sandbox.")


class SandboxToolState:
    """Manages the lifecycle of a sandbox session for one ranker invocation.

    A single sticky sandbox session is created lazily on first tool use
    and reused for all subsequent code executions within the same ranker
    invocation. The cleanup method tears down the container.
    """

    def __init__(self, sandbox_client: Any, cfg: dict[str, Any]):
        self._client = sandbox_client
        self._cfg = cfg
        self._sticky = None

    def ensure_session(self):
        """Create the sandbox session lazily on first call."""
        if self._sticky is None:
            session_name = f"ranker_{uuid4().hex[:8]}"
            self._sticky = self._client.sticky_session(
                session_name=session_name,
                extra_payload={"owner": "ranker"},
            )
            self._sticky.ensure()
            log.info("Sandbox session created: %s", session_name)
        return self._sticky

    def run_code(self, code: str) -> str:
        """Execute code and return a formatted text result.

        Args:
            code: Python source code to execute.

        Returns:
            Formatted string with status, stdout, stderr.
        """
        sticky = self.ensure_session()
        timeout = self._cfg.get("code_timeout_seconds", 120)
        resp = sticky.run(code, timeout=timeout, reset=False, shutdown_after=False)

        # Format response as text for the LLM
        parts = []
        status = resp.get("status", "unknown")
        parts.append(f"Status: {status}")

        stdout = resp.get("stdout")
        if stdout:
            parts.append(f"stdout:\n{stdout.strip()}")

        stderr = resp.get("stderr")
        if stderr:
            parts.append(f"stderr:\n{stderr.strip()}")

        result = resp.get("result")
        if result:
            parts.append(f"result: {result}")

        # Note images but don't embed (text-only output)
        images = resp.get("images") or []
        if images:
            parts.append(f"[{len(images)} image(s) produced but not displayed]")

        return "\n".join(parts) if parts else "No output."

    def cleanup(self):
        """Tear down the sandbox session if one was created."""
        if self._sticky is not None:
            try:
                self._sticky.close()
            except Exception:
                pass
            self._sticky = None
            log.debug("Sandbox session cleaned up")


def make_sandbox_tool(
    cfg: dict[str, Any],
    sandbox_client: Any,
) -> tuple[StructuredTool, Callable[[], None]]:
    """Create a sandbox code execution tool.

    Args:
        cfg: Sandbox config dict with ``code_timeout_seconds``, etc.
        sandbox_client: A ``SandboxClient`` instance.

    Returns:
        Tuple of (StructuredTool, cleanup_callable). The cleanup callable
        should be invoked when the ranker session is done to release the
        sandbox container.
    """
    state = SandboxToolState(sandbox_client, cfg)
    log.debug("Sandbox tool created (timeout=%ds)", cfg.get("code_timeout_seconds", 120))

    def _run(code: str) -> str:
        return state.run_code(code)

    tool = StructuredTool.from_function(
        name="sandbox",
        func=_run,
        description=(
            "Execute Python code in a sandboxed container. Use for verifying "
            "claims, running statistical analyses, or testing code from papers. "
            "The environment persists between calls within this session."
        ),
        args_schema=_SandboxArgs,
        return_direct=False,
    )

    return tool, state.cleanup


# ---------------------------------------------------------------------------
# Aggregate builder
# ---------------------------------------------------------------------------


def build_ranker_tools(
    tools_cfg: dict[str, Any],
    sandbox_client: Any = None,
) -> tuple[dict[str, Any], Callable[[], None]]:
    """Build all enabled ranker tools from config.

    Args:
        tools_cfg: Config dict with ``arxiv`` and ``sandbox`` sub-dicts,
            each having an ``enabled`` bool and tool-specific params.
        sandbox_client: Optional SandboxClient, required if sandbox is enabled.

    Returns:
        Tuple of (tools_dict, cleanup_fn).
        tools_dict maps tool name to a dict with keys:
            - "object": StructuredTool instance
            - "tool_signature": human-readable signature string
        cleanup_fn: Callable that cleans up any sandbox sessions.
    """
    tools: dict[str, Any] = {}
    cleanup_fns = []

    arxiv_cfg = tools_cfg.get("arxiv", {})
    if arxiv_cfg.get("enabled", False):
        arxiv_tool = make_arxiv_tool(arxiv_cfg)
        tools["arxiv"] = {
            "object": arxiv_tool,
            "tool_signature": (
                '- arxiv: args = { "query": str } '
                "(Search arXiv and return summaries of top results.)"
            ),
        }

    sandbox_cfg = tools_cfg.get("sandbox", {})
    if sandbox_cfg.get("enabled", False):
        if sandbox_client is None:
            from env_setup.sandbox_utils import SandboxClient

            base_url = sandbox_cfg.get("base_url")
            sandbox_client = SandboxClient(
                code_timeout_seconds=sandbox_cfg.get("code_timeout_seconds", 120),
                base_url=base_url,
            )
        sandbox_tool, sandbox_cleanup = make_sandbox_tool(sandbox_cfg, sandbox_client)
        tools["sandbox"] = {
            "object": sandbox_tool,
            "tool_signature": (
                '- sandbox: args = { "code": str } (Execute Python code in a sandboxed container.)'
            ),
        }
        cleanup_fns.append(sandbox_cleanup)

    def cleanup_all():
        for fn in cleanup_fns:
            try:
                fn()
            except Exception:
                pass

    return tools, cleanup_all
