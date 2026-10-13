from typing import Any

from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from env_setup.langchain_tools import (
    agent_search_tool,
    communicate_tool,
    establish_collaboration_tool,
    paper_search_tool,
)
from env_setup.sandbox_tools import (
    _emit_arxiv_code,
    _emit_search_agent_registry_code,
)
from system.trackers import record_tool_query

SUPPORTED_TOOLS = {
    "arxiv",
    "search_internal_archive",
    "search_agent_registry",
    "establish_collaboration",
    "communicate_with_collaborator",
}


DEFAULTS = {
    "arxiv": {
        "top_k_results": 10,
        "load_max_docs": 10,
        "doc_content_chars_max": None,
    },
    "search_internal_archive": {},
    "search_agent_registry": {},
    "establish_collaboration": {},
    "communicate_with_collaborator": {},
}


def _with_defaults(name: str, cfg: dict | None) -> dict:
    """Return cfg with defaults filled in for our three params, if absent."""
    params = dict(cfg or {})
    for k, v in DEFAULTS[name].items():
        params.setdefault(k, v)  # only set if key missing
    return params


def retrieve_tool_signatures(tools_cfg: dict) -> dict[str, dict[str, str]]:
    """Return code/signature strings for tools for prompt display."""
    tools: dict[str, dict[str, str]] = {}

    for name, cfg in tools_cfg.items():
        if name not in SUPPORTED_TOOLS:
            raise ValueError(f"Unsupported tool: {name}")

        params = _with_defaults(name, cfg)

        if name == "arxiv":
            code, sig = _emit_arxiv_code(params)

        elif name == "search_internal_archive":
            code, sig = _emit_search_agent_registry_code(params)

        else:
            code, sig = "# No code available for this tool.", "No signature available."

        tools[name] = {"code": code, "signature": sig}

    return tools


def _extract_query(payload: Any) -> str:
    """Best-effort extraction of a human-readable query string from tool inputs."""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        for key in ("query", "q", "text", "keywords", "term", "search"):
            val = payload.get(key)
            if isinstance(val, str):
                return val
        # fall back to something informative but bounded
        try:
            import json

            return json.dumps(
                {
                    k: v
                    for k, v in payload.items()
                    if k
                    in (
                        "query",
                        "q",
                        "text",
                        "keywords",
                        "term",
                        "search",
                        "pmcid",
                        "doi",
                        "pmid",
                        "query",
                        "page_size",
                    )
                }
            )[:500]
        except Exception:
            pass
    return repr(payload)[:500]


def make_tools(tools_cfg: dict) -> dict[str, StructuredTool]:
    """Instantiate StructuredTools according to a config dictionary.

    Args:
        tools_cfg: Mapping of tool name to configuration parameters.

    Returns:
        Dictionary of tool name to StructuredTool instances.

    Raises:
        ValueError: If an unsupported tool name is provided.
    """
    tools: dict[str, StructuredTool] = {}

    for name, cfg in tools_cfg.items():
        if name not in SUPPORTED_TOOLS:
            raise ValueError(f"Unsupported tool: {name}")

        params = _with_defaults(name, cfg)

        if name == "arxiv":
            wrapper = ArxivAPIWrapper(**params)
            underlying = ArxivQueryRun(api_wrapper=wrapper)

            def _run_arxiv(query: str, _underlying=underlying) -> str:
                try:
                    record_tool_query("arxiv", query)
                except Exception:
                    pass
                return _underlying.run({"query": query})

            class _ArxivArgs(BaseModel):
                query: str = Field(..., description="arXiv query string")

            tools[name] = StructuredTool.from_function(
                name="arxiv",
                func=_run_arxiv,
                args_schema=_ArxivArgs,
                description="Search arXiv and return a text summary of top results.",
            )
        elif name == "search_internal_archive":
            base = paper_search_tool  # existing StructuredTool

            # Wrap while preserving schema
            def _run_internal_archive(_base=base, **kwargs):
                try:
                    record_tool_query("search_internal_archive", _extract_query(kwargs))
                except Exception:
                    pass
                # delegate with the same input shape
                return _base.invoke(kwargs)

            tools[name] = StructuredTool.from_function(
                name=getattr(base, "name", "search_internal_archive"),
                func=_run_internal_archive,
                args_schema=getattr(base, "args_schema", None),
                description=getattr(base, "description", "Search the internal paper archive."),
            )

        elif name == "search_agent_registry":
            base = agent_search_tool  # existing StructuredTool

            def _run_agent_registry(_base=base, **kwargs):
                try:
                    record_tool_query("search_agent_registry", _extract_query(kwargs))
                except Exception:
                    pass
                return _base.invoke(kwargs)

            tools[name] = StructuredTool.from_function(
                name=getattr(base, "name", "search_agent_registry"),
                func=_run_agent_registry,
                args_schema=getattr(base, "args_schema", None),
                description=getattr(
                    base,
                    "description",
                    "Search the agent registry for candidate collaborators/reviewers.",
                ),
            )

        elif name == "establish_collaboration":  # NEW
            tools[name] = establish_collaboration_tool

        elif name == "communicate_with_collaborator":
            # unchanged (no query to log)
            tools[name] = communicate_tool

        else:
            raise ValueError(f"No implementation for tool: {name}")

    return tools
