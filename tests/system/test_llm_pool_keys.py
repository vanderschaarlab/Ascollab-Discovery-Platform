"""The feedback synthesizer's backend is built, and its endpoint required, only when used."""

import pytest
from omegaconf import OmegaConf

from env_setup.build_llms import EndpointConfigError
from system.main import _llm_pool_keys

_ENDPOINT = {"azure_deployment": "x"}


def _cfg(feedback: dict, endpoints: dict) -> OmegaConf:
    return OmegaConf.create(
        {
            "llm_backends": {"counts": {"gpt4o": 2, "gpt5": 0}},
            "research": {"metacognition_feedback": feedback},
            "endpoints": {"llm_primary": _ENDPOINT, **endpoints},
        }
    )


@pytest.mark.parametrize(
    "feedback",
    [
        {"enabled": False, "mode": "meta_review_only", "llm_backend": "gpt5"},
        {"enabled": True, "mode": "metrics_only", "llm_backend": "gpt5"},
        {"enabled": True, "mode": "none", "llm_backend": "gpt5"},
    ],
    ids=["disabled", "metrics-only", "mode-none"],
)
def test_unused_feedback_backend_is_neither_built_nor_required(feedback):
    assert _llm_pool_keys(_cfg(feedback, {"gpt4o": _ENDPOINT})) == ["gpt4o"]


@pytest.mark.parametrize("mode", ["meta_review_only", "meta_review_with_metrics"])
def test_feedback_backend_is_built_in_llm_modes(mode):
    feedback = {"enabled": True, "mode": mode, "llm_backend": "gpt5"}
    keys = _llm_pool_keys(_cfg(feedback, {"gpt4o": _ENDPOINT, "gpt5": _ENDPOINT}))
    assert keys == ["gpt4o", "gpt5"]


def test_missing_feedback_endpoint_file_stops_the_run_at_start_up():
    feedback = {"enabled": True, "mode": "meta_review_only", "llm_backend": "gpt5"}
    with pytest.raises(EndpointConfigError, match=r"sys_config/llm_api/gpt-5\.yaml"):
        _llm_pool_keys(_cfg(feedback, {"gpt4o": _ENDPOINT}))


def test_primary_feedback_backend_needs_no_extra_file():
    feedback = {"enabled": True, "mode": "meta_review_only", "llm_backend": "primary"}
    assert _llm_pool_keys(_cfg(feedback, {"gpt4o": _ENDPOINT})) == ["gpt4o"]
