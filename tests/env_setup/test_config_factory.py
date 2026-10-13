import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

import env_setup.config_factory as config_factory


class _Args(BaseModel):
    query: str = Field(..., description="Search query")
    k: int = Field(3, description="Result count")


def _dummy_tool(query: str, k: int = 3) -> str:
    return f"{query}:{k}"


def test_tool_signature_line_uses_schema_types_and_descriptions():
    tool = StructuredTool.from_function(
        name="dummy",
        func=_dummy_tool,
        args_schema=_Args,
        description="A dummy search tool.\nMore details.",
    )

    line = config_factory.tool_signature_line(tool)

    assert line.startswith("- dummy: args =")
    assert '"query": str' in line
    assert "query = Search query" in line
    assert "A dummy search tool." in line


def test_build_config_materializes_primary_and_collab_tools(monkeypatch):
    tool = StructuredTool.from_function(
        name="arxiv",
        func=_dummy_tool,
        args_schema=_Args,
        description="Search arXiv.",
    )
    monkeypatch.setattr(config_factory, "make_tools", lambda tools_cfg: {"arxiv": tool})
    monkeypatch.setattr(
        config_factory,
        "retrieve_tool_signatures",
        lambda tools_cfg: {"arxiv": {"code": "CODE", "signature": "SIG"}},
    )

    cfg = config_factory.build_config(
        dataset_keys=["tcga_kirc_neurips_hiseqv2"],
        python_package_set_key="tcga",
        tools_cfg={"arxiv": {}},
        goal_key="tcga",
    )

    assert "tcga_kirc_neurips_hiseqv2" in cfg.dataset_description
    assert "arxiv" in cfg.primary_agent_tools
    assert "arxiv" in cfg.collab_agent_tools
    assert cfg.primary_agent_tools["arxiv"].code == "CODE"


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"dataset_keys": ["missing"]}, "Unknown dataset key"),
        ({"python_package_set_key": "missing"}, "Unknown python package key"),
        ({"tools_cfg": {"missing": {}}}, "Unsupported tool"),
        ({"goal_key": "missing"}, "Unknown goal key"),
    ],
)
def test_build_config_rejects_unknown_registry_keys(kwargs, message):
    base = {
        "dataset_keys": ["tcga_kirc_neurips_hiseqv2"],
        "python_package_set_key": "tcga",
        "tools_cfg": {},
        "goal_key": "tcga",
    }
    base.update(kwargs)

    with pytest.raises(ValueError, match=message):
        config_factory.build_config(**base)
