"""The code fallbacks for research.prior_papers match sys_config/ascollab.yaml."""

import ast
import inspect
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from system.research_one_step import research_one_step

REPO_ROOT = Path(__file__).resolve().parents[2]
PRIOR_PAPERS = OmegaConf.load(REPO_ROOT / "sys_config" / "ascollab.yaml").research.prior_papers


def _prior_cfg_get_defaults(path: str) -> dict[str, object]:
    """Defaults of every ``prior_cfg.get(key, default)`` call in a module."""
    defaults: dict[str, object] = {}
    for node in ast.walk(ast.parse((REPO_ROOT / path).read_text(encoding="utf-8"))):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "prior_cfg"
            and len(node.args) == 2
            and all(isinstance(arg, ast.Constant) for arg in node.args)
        ):
            key, default = (arg.value for arg in node.args)
            defaults[str(key)] = default
    return defaults


@pytest.mark.parametrize("path", ["system/main.py", "system/snapshot_utils.py"])
def test_prior_papers_config_fallbacks_match_the_config(path):
    defaults = _prior_cfg_get_defaults(path)
    assert defaults["enabled"] == PRIOR_PAPERS.enabled
    assert defaults["mode"] == PRIOR_PAPERS.mode


def test_research_one_step_prior_papers_defaults_match_the_config():
    params = inspect.signature(research_one_step).parameters
    assert params["prior_papers_enabled"].default == PRIOR_PAPERS.enabled
    assert params["prior_papers_mode"].default == PRIOR_PAPERS.mode
