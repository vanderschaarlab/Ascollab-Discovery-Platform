"""Any model mix works: only the backends a run uses are built, and missing ones are flagged."""

import logging
import random

import pytest
from omegaconf import OmegaConf

from env_setup import build_llms
from env_setup.build_llms import build_llm_pool, mix_backends, realize_mix

# A filled-in endpoint (never contacted: building a client makes no request) and an unfilled
# template, as copied from sys_config/llm_api/default_to_complete.yaml.
_FILLED = {
    "azure_deployment": "gpt-5",
    "azure_endpoint": "https://example.invalid",
    "api_key": "not-a-key",
    "api_version": "2024-12-01-preview",
    "seed": 1,
}
_TEMPLATE = {"azure_deployment": None, "azure_endpoint": None, "api_key": None, "api_version": None}


def _endpoints(**filled):
    keys = ("llm_primary", "gpt4o", "gpt5", "deepseek", "llama")
    return OmegaConf.create({k: (_FILLED if k in filled else _TEMPLATE) for k in keys})


def test_mix_backends_lists_non_zero_shares_in_order():
    assert mix_backends({"gpt4o": 6, "gpt5": 7, "deepseek": 3, "llama": 0}) == [
        "gpt4o",
        "gpt5",
        "deepseek",
    ]
    assert mix_backends({"gpt5": 0}, {"llama": 0.5, "gpt5": 0.5}) == ["llama", "gpt5"]
    assert mix_backends(None, None) == []


def test_unused_backends_can_stay_unfilled_templates():
    pool = build_llm_pool(_endpoints(llm_primary=True, gpt5=True), keys=["gpt5"])
    assert sorted(pool) == ["gpt5", "primary"]


def test_building_an_unfilled_template_fails():
    # Why only used backends are built: an unfilled endpoint cannot make a client
    # (_mk_llm raises EndpointConfigError, a ValueError naming the file, before any client).
    with pytest.raises(build_llms.EndpointConfigError):
        build_llm_pool(_endpoints(llm_primary=True, gpt5=True), keys=["gpt5", "llama"])


def test_without_keys_every_known_configured_backend_is_built(monkeypatch):
    monkeypatch.setattr(build_llms, "_mk_llm", lambda cfg, key=None: object())
    pool = build_llm_pool(_endpoints())
    assert sorted(pool) == ["deepseek", "gpt4o", "gpt5", "llama", "primary"]


def test_a_new_backend_key_can_be_added(monkeypatch):
    monkeypatch.setattr(build_llms, "_mk_llm", lambda cfg, key=None: object())
    endpoints = OmegaConf.create({"llm_primary": _FILLED, "qwen": _FILLED})
    assert sorted(build_llm_pool(endpoints, keys=["qwen"])) == ["primary", "qwen"]


def test_missing_requested_backend_warns_and_falls_back_to_primary(caplog):
    pool = {"gpt5": "gpt5-client", "primary": "primary-client"}
    with caplog.at_level(logging.WARNING, logger="env_setup.build_llms"):
        plan = realize_mix(pool, num_agents=4, counts={"gpt5": 2, "llama": 2}, rng=random.Random(0))
    assert sorted(k for k, _ in plan) == ["gpt5", "gpt5", "primary", "primary"]
    assert any("['llama']" in r.getMessage() for r in caplog.records)


def test_complete_mix_does_not_warn(caplog):
    pool = {"gpt5": "a", "deepseek": "b", "primary": "c"}
    with caplog.at_level(logging.WARNING, logger="env_setup.build_llms"):
        realize_mix(pool, num_agents=3, counts={"gpt5": 2, "deepseek": 1}, rng=random.Random(0))
    assert not caplog.records


@pytest.mark.parametrize(
    ("n", "expected"),
    # The mixes experiment 05 sets by hand for N >= 2 (mix_for in its run.sh); its N = 1 is the
    # single GPT-5 agent of experiment 01, not a scaled mix.
    [(2, (1, 1, 0)), (3, (1, 1, 1)), (4, (2, 1, 1)), (5, (2, 2, 1)), (10, (4, 3, 3))],
)
def test_counts_above_network_size_scale_down_keeping_proportions(n, expected, caplog):
    pool = {"gpt4o": "a", "gpt5": "b", "deepseek": "c", "primary": "d"}
    counts = {"gpt4o": 6, "gpt5": 5, "deepseek": 5, "llama": 0}
    with caplog.at_level(logging.WARNING, logger="env_setup.build_llms"):
        plan = [k for k, _ in realize_mix(pool, num_agents=n, counts=counts, rng=random.Random(0))]
    assert tuple(plan.count(k) for k in ("gpt4o", "gpt5", "deepseek")) == expected
    assert "scaled to" in caplog.text


def test_primary_alias_can_be_left_out():
    # The ranker builds only its judge: an unfilled default.yaml must not matter.
    endpoints = _endpoints(gpt5=True)  # llm_primary is the unfilled template
    assert sorted(build_llm_pool(endpoints, keys=["gpt5"], primary_alias=False)) == ["gpt5"]
    with pytest.raises(ValueError):
        build_llm_pool(endpoints, keys=["gpt5"])
    primary_only = build_llm_pool(_endpoints(llm_primary=True), keys=["primary"])
    assert sorted(primary_only) == ["primary"]
