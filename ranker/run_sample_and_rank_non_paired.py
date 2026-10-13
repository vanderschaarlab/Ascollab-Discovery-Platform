"""Hydra CLI entrypoint for the non-paired sample-and-rank batch workflow.

Usage::

    python -m ranker.run_sample_and_rank_non_paired \\
        paper_pool.snapshot_dirs='["snapshots/run_seed_1", "snapshots/run_seed_2"]'
"""

import logging
import sys

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from env_setup.build_llms import ENDPOINT_SETUP_HINT, build_llm_pool, endpoint_file
from system.logging_utils import setup_file_logging_hydra_safe

from .sample_and_rank_non_paired import run_sample_and_rank_non_paired

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path="../sys_config", config_name="sample_and_rank_non_paired"
)
def main(cfg: DictConfig) -> None:
    """Run the non-paired sample-and-rank batch workflow."""

    # --- Setup logging ---
    setup_file_logging_hydra_safe()

    # --- Build LLM ---
    backend_key = cfg.ranker.llm_backend
    pool = build_llm_pool(
        cfg.get("endpoints"), keys=[backend_key], primary_alias=backend_key == "primary"
    )
    if backend_key not in pool:
        print(
            f"Error: no endpoint for ranker backend '{backend_key}': create "
            f"{endpoint_file(backend_key)} ({ENDPOINT_SETUP_HINT}). "
            f"Available: {list(pool.keys())}",
            file=sys.stderr,
        )
        sys.exit(1)
    llm = pool[backend_key]

    # --- Build sandbox client if needed ---
    sandbox_client = None
    tools_cfg = OmegaConf.to_container(cfg.ranker.tools, resolve=True)
    sandbox_cfg = tools_cfg.get("sandbox", {})
    if sandbox_cfg.get("enabled", False):
        from env_setup.sandbox_utils import SandboxClient

        sandbox_client = SandboxClient(
            code_timeout_seconds=sandbox_cfg.get("code_timeout_seconds", 120),
            base_url=sandbox_cfg.get("base_url"),
        )

    # --- Get output dir from Hydra ---
    try:
        output_dir = HydraConfig.get().runtime.output_dir
    except Exception:
        output_dir = None

    # --- Build paper_selection_cfg ---
    paper_selection_cfg = OmegaConf.to_container(cfg.paper_selection, resolve=True)

    # --- Run ---
    result = run_sample_and_rank_non_paired(  # noqa: F841
        paper_pool_cfg=OmegaConf.to_container(cfg.paper_pool, resolve=True),
        sampling_cfg=OmegaConf.to_container(cfg.sampling, resolve=True),
        ranker_cfg=OmegaConf.to_container(cfg.ranker, resolve=True),
        parallelism_cfg=OmegaConf.to_container(cfg.parallelism, resolve=True),
        llm=llm,
        sandbox_client=sandbox_client,
        run_id=cfg.run_name,
        output_dir=output_dir,
        paper_selection_cfg=paper_selection_cfg,
    )


if __name__ == "__main__":
    main()
