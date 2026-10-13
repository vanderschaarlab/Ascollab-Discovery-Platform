"""Standalone Hydra entrypoint for the LLM-judge ranker.

Usage::

    python -m ranker.run_ranker \\
        +paper_a.title="My Paper A" \\
        +paper_a.abstract="..." \\
        +paper_a.manuscript="..." \\
        +paper_b.title="My Paper B" \\
        +paper_b.abstract="..." \\
        +paper_b.manuscript="..." \\
        ranker.llm_backend=gpt5

Or with a JSON input file::

    python -m ranker.run_ranker input_file=path/to/paper_pair.json
"""

import dataclasses
import json
import logging
import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

from env_setup.build_llms import ENDPOINT_SETUP_HINT, build_llm_pool, endpoint_file
from system.logging_utils import setup_file_logging_hydra_safe

from .ranker_core import run_ranker

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../sys_config", config_name="ranker")
def main(cfg: DictConfig) -> None:
    """Run the ranker on a pair of papers provided via CLI or input_file."""

    # --- Setup logging ---
    setup_file_logging_hydra_safe()

    # --- Load papers ---
    if "input_file" in cfg and cfg.input_file is not None:
        with open(cfg.input_file) as f:
            data = json.load(f)
        paper_a = data["paper_a"]
        paper_b = data["paper_b"]
    elif "paper_a" in cfg and "paper_b" in cfg:
        paper_a = OmegaConf.to_container(cfg.paper_a, resolve=True)
        paper_b = OmegaConf.to_container(cfg.paper_b, resolve=True)
    else:
        print(
            "Error: Provide papers via paper_a/paper_b CLI args or input_file=path.json",
            file=sys.stderr,
        )
        sys.exit(1)

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

    # --- Build tools config ---
    ranker_cfg = cfg.ranker
    tools_cfg = (
        OmegaConf.to_container(ranker_cfg.tools, resolve=True) if ranker_cfg.get("tools") else {}
    )
    llm_retry_cfg = (
        OmegaConf.to_container(ranker_cfg.llm_retry, resolve=True)
        if ranker_cfg.get("llm_retry")
        else {}
    )

    # --- Build sandbox client if needed ---
    sandbox_client = None
    sandbox_cfg = tools_cfg.get("sandbox", {})
    if sandbox_cfg.get("enabled", False):
        from env_setup.sandbox_utils import SandboxClient

        sandbox_client = SandboxClient(
            code_timeout_seconds=sandbox_cfg.get("code_timeout_seconds", 120),
            base_url=sandbox_cfg.get("base_url"),
        )

    # --- Log config ---
    tool_names = []
    if tools_cfg.get("arxiv", {}).get("enabled"):
        tool_names.append("arxiv")
    if tools_cfg.get("sandbox", {}).get("enabled"):
        tool_names.append("sandbox")
    mode = ranker_cfg.get("mode", "comparison")
    log.info(
        "Ranker config: mode=%s, backend=%s, max_iterations=%d, tools=%s",
        mode,
        backend_key,
        ranker_cfg.get("max_iterations", 10),
        tool_names or "none",
    )
    log.info("Loaded papers: A=%r, B=%r", paper_a.get("title", "N/A"), paper_b.get("title", "N/A"))

    # --- Run ranker ---
    criteria_prompt = ranker_cfg.get("criteria_prompt", None)
    log.info("Criteria prompt: %s", criteria_prompt or "(default)")
    max_parse_retries = 2
    if ranker_cfg.get("output_parsing"):
        max_parse_retries = ranker_cfg.output_parsing.get("max_parse_retries", 2)

    result = run_ranker(
        paper_a=paper_a,
        paper_b=paper_b,
        llm=llm,
        criteria_prompt=criteria_prompt,
        mode=mode,
        tools_cfg=tools_cfg,
        sandbox_client=sandbox_client,
        max_iterations=ranker_cfg.get("max_iterations", 10),
        llm_retry_cfg=llm_retry_cfg,
        max_parse_retries=max_parse_retries,
    )

    # --- Print results ---
    print("=" * 60)
    print("RANKER RESULT")
    print("=" * 60)
    if result.score is None:
        outcome_label = "No verdict"
    elif mode == "hypothesis_similarity":
        outcome_label = "Same" if result.score == 0 else "Different"
    else:
        outcome_label = "Paper A" if result.score == 1 else "Paper B"
    print(f"Outcome: {outcome_label} (score={result.score})")
    print(f"Turns: {result.turns}")
    print(f"Tokens: input={result.input_tokens}, output={result.output_tokens}")
    if result.tool_calls:
        print(f"Tool calls: {result.tool_calls}")
    print("-" * 60)
    print("Justification:")
    print(result.justification)
    print("=" * 60)

    # Also log the result
    log.info(
        "Result: outcome=%s, turns=%d, tokens=%d/%d (in/out)",
        outcome_label,
        result.turns,
        result.input_tokens,
        result.output_tokens,
    )

    # --- Write trace to Hydra output dir ---
    if result.trace is not None:
        try:
            from hydra.core.hydra_config import HydraConfig

            run_dir = HydraConfig.get().runtime.output_dir
        except Exception:
            run_dir = os.getcwd()

        trace_path = os.path.join(run_dir, "ranker_trace.json")
        with open(trace_path, "w") as f:
            json.dump(dataclasses.asdict(result.trace), f, indent=2)
        log.info("Trace written to %s", trace_path)


if __name__ == "__main__":
    main()
