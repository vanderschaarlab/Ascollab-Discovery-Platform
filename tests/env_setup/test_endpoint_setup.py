"""Endpoint files: only the ones a run needs must exist, and unfilled ones are named clearly."""

import logging
import random
import shutil
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from env_setup.build_llms import (
    EndpointConfigError,
    _mk_embedder,
    _mk_llm,
    build_llm_pool,
    realize_mix,
    require_endpoints,
)

_SYS_CONFIG = Path(__file__).resolve().parents[2] / "sys_config"


def _template(kind: str) -> dict:
    cfg = OmegaConf.load(_SYS_CONFIG / kind / "default_to_complete.yaml")
    return {k: v for k, v in OmegaConf.to_container(cfg, resolve=False).items() if k != "seed"}


def test_missing_required_endpoint_names_the_file():
    with pytest.raises(EndpointConfigError, match=r"sys_config/llm_api/default\.yaml"):
        require_endpoints({"embedder_primary": {}}, ("llm_primary", "embedder_primary"))
    with pytest.raises(EndpointConfigError, match=r"sys_config/embedder_api/default\.yaml"):
        require_endpoints(None, ("embedder_primary",))
    require_endpoints({"llm_primary": {}, "embedder_primary": {}}, ("llm_primary",))


def test_unfilled_template_is_rejected_with_its_file_and_field():
    with pytest.raises(
        EndpointConfigError, match=r"gpt-5\.yaml is not filled in.*azure_deployment"
    ):
        _mk_llm(_template("llm_api"), key="gpt5")
    with pytest.raises(EndpointConfigError, match=r"embedder_api/default\.yaml.*azure_deployment"):
        _mk_embedder(_template("embedder_api"), key="embedder_primary")
    with pytest.raises(EndpointConfigError, match="model is empty"):
        _mk_llm({"provider": "openai", "api_key": "x"}, key="llama")


def test_pool_skips_backends_without_a_file():
    assert build_llm_pool({}, keys=["gpt5"]) == {}
    assert build_llm_pool(None, keys=["gpt5"]) == {}


def _only_templates(directory: str, names: list[str]) -> set[str]:
    """copytree ignore: never copy a filled-in endpoint file (it holds an API key)."""
    if Path(directory) in (_SYS_CONFIG / "llm_api", _SYS_CONFIG / "embedder_api"):
        return {name for name in names if name != "default_to_complete.yaml"}
    return set()


def test_run_config_composes_without_any_endpoint_file(tmp_path):
    # A fresh clone has only the two templates; composing must not fail.
    config_dir = tmp_path / "sys_config"
    shutil.copytree(_SYS_CONFIG, config_dir, ignore=_only_templates)
    for kind in ("llm_api", "embedder_api"):
        assert [f.name for f in (config_dir / kind).iterdir()] == ["default_to_complete.yaml"]
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="ascollab")
        ranker_cfg = compose(config_name="sample_and_rank_paired")
    assert not cfg.get("endpoints")
    assert not ranker_cfg.get("endpoints")
    with pytest.raises(EndpointConfigError):
        require_endpoints(cfg.get("endpoints"), ("llm_primary", "embedder_primary"))


def test_missing_mix_backend_warning_names_its_file(caplog):
    with caplog.at_level(logging.WARNING, logger="env_setup.build_llms"):
        realize_mix({"primary": "p"}, num_agents=2, counts={"gpt4o": 2}, rng=random.Random(0))
    assert "sys_config/llm_api/gpt-4o.yaml" in caplog.text
    assert "docs/models.md" in caplog.text
