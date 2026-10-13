"""Tests for Hydra config loading."""

import pytest


class TestConfig:
    """Test that ranker.yaml loads correctly and has expected structure."""

    def test_hydra_config_loads(self, ranker_cfg):
        assert ranker_cfg.ranker.max_iterations is not None
        assert isinstance(ranker_cfg.ranker.max_iterations, int)
        assert ranker_cfg.ranker.criteria_prompt is not None
        assert len(ranker_cfg.ranker.criteria_prompt) > 0
        assert isinstance(ranker_cfg.ranker.tools.arxiv.enabled, bool)
        assert isinstance(ranker_cfg.ranker.tools.sandbox.enabled, bool)

    @pytest.mark.integration  # needs a filled-in judge endpoint file
    def test_llm_backend_resolution(self, ranker_cfg):
        from env_setup.build_llms import build_llm_pool

        backend_key = ranker_cfg.ranker.llm_backend
        pool = build_llm_pool(
            ranker_cfg.endpoints, keys=[backend_key], primary_alias=backend_key == "primary"
        )
        assert backend_key in pool, (
            f"Configured backend '{backend_key}' not in pool: {list(pool.keys())}"
        )

    def test_retry_config_present(self, ranker_cfg):
        retry = ranker_cfg.ranker.llm_retry
        assert retry.max_retries is not None
        assert retry.base_delay_seconds is not None
        assert retry.max_delay_seconds is not None

    def test_output_parsing_config(self, ranker_cfg):
        parsing = ranker_cfg.ranker.output_parsing
        assert parsing.max_parse_retries is not None
        assert isinstance(parsing.max_parse_retries, int)
