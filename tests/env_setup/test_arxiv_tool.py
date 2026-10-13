"""Guards against the arxiv/langchain_community API drift that silently disabled
literature search: arxiv 3.0 removed ``Search.results()``, which
``ArxivAPIWrapper`` still calls, so every agent arxiv call raised AttributeError.
"""

import pytest

pytestmark = pytest.mark.integration


def test_arxiv_search_results_api_is_available():
    """The wrapper calls Search.results(); fail loudly if the pin drifts.

    This assertion needs no network and is the cheap early-warning for an
    incompatible arxiv upgrade.
    """
    import arxiv

    assert hasattr(arxiv.Search, "results"), (
        "arxiv.Search.results() is gone — langchain_community's ArxivAPIWrapper "
        "requires it. Pin arxiv<3 (see sandbox/server_requirements.txt)."
    )


def test_arxiv_query_returns_results():
    """The arxiv tool returns paper text rather than raising."""
    from langchain_community.tools.arxiv.tool import ArxivQueryRun
    from langchain_community.utilities.arxiv import ArxivAPIWrapper

    output = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=1)).run(
        "renal cell carcinoma single-cell"
    )

    assert isinstance(output, str)
    assert "Published:" in output and "Title:" in output


def test_built_arxiv_tool_is_callable():
    """The tool as `make_tools` wires it for agents actually runs end to end."""
    from env_setup.tools import make_tools

    tool = make_tools({"arxiv": {"top_k_results": 1}})["arxiv"]

    output = tool.run({"query": "renal cell carcinoma single-cell"})

    assert isinstance(output, str) and output.strip()
