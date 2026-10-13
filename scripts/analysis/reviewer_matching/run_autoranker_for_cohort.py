"""Run AutoRanker pairwise scoring for one fixed cohort manifest."""

from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from env_setup.build_llms import ENDPOINT_SETUP_HINT, build_llm_pool, endpoint_file
from ranker.sample_and_rank_common import _run_comparisons_parallel, _save_results
from ranker.sample_and_rank_non_paired import (
    _build_summary_non_paired,
    _collect_paper_data_non_paired,
    _generate_pairs_non_paired,
)
from ranker.snapshot_import import (
    build_citation_chains,
    collect_citable_papers,
    collect_dataset_descriptions,
)

from .autoranker_utils import aggregate_win_rate_scores
from .cohort_utils import load_candidate_pool
from .io_utils import read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config-name", default="sample_and_rank_non_paired")
    parser.add_argument("--sampling-mode", choices=["all", "random"], default="all")
    parser.add_argument("--n-pairs", type=int, default=100)
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--judge", help="Judge backend, ranker.llm_backend (default: the config's)")
    parser.add_argument("--max-workers", type=int, help="parallelism.max_workers")
    return parser.parse_args()


def _compose_ranker_config(
    config_name: str, judge: str | None = None, max_workers: int | None = None
) -> DictConfig:
    """Compose the ranker config from ``sys_config``, with the judge and parallelism if given.

    Args:
        config_name: Hydra config name, e.g. ``sample_and_rank_non_paired``.
        judge: Endpoint key of the judge (``ranker.llm_backend``); None keeps the config's.
        max_workers: ``parallelism.max_workers``; None keeps the config's.

    Returns:
        The composed config.
    """
    overrides = []
    if judge:
        overrides.append(f"ranker.llm_backend={judge}")
    if max_workers is not None:
        overrides.append(f"parallelism.max_workers={int(max_workers)}")
    with initialize_config_dir(config_dir=str(Path("sys_config").resolve()), version_base=None):
        return compose(config_name=config_name, overrides=overrides)


def _load_manifest_papers(manifest: dict) -> list[dict]:
    max_iter = max(
        int(p.get("source_iteration", manifest["target_iteration"]))
        for p in manifest["candidate_papers"]
    )
    pool = load_candidate_pool(manifest["snapshot_dirs"], max_iteration=max_iter)
    by_id = {str(p["paper_id"]): p for p in pool}
    missing = [pid for pid in manifest["candidate_paper_ids"] if pid not in by_id]
    if missing:
        raise ValueError(f"Manifest papers missing from snapshot pool: {missing[:5]}")
    return [by_id[pid] for pid in manifest["candidate_paper_ids"]]


def main() -> None:
    args = parse_args()
    manifest = read_json(args.manifest)
    papers = _load_manifest_papers(manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _compose_ranker_config(args.config_name, args.judge, args.max_workers)

    backend_key = cfg.ranker.llm_backend
    pool = build_llm_pool(
        cfg.get("endpoints"), keys=[backend_key], primary_alias=backend_key == "primary"
    )
    if backend_key not in pool:
        raise KeyError(
            f"No endpoint for ranker backend {backend_key!r}: create {endpoint_file(backend_key)} "
            f"({ENDPOINT_SETUP_HINT}). Available: {sorted(pool)}"
        )
    llm = pool[backend_key]

    tools_cfg = OmegaConf.to_container(cfg.ranker.tools, resolve=True)
    sandbox_client = None
    if tools_cfg.get("sandbox", {}).get("enabled", False):
        from env_setup.sandbox_utils import SandboxClient

        sandbox_client = SandboxClient(
            code_timeout_seconds=tools_cfg["sandbox"].get("code_timeout_seconds", 120),
            base_url=tools_cfg["sandbox"].get("base_url"),
        )

    seed = int(args.seed if args.seed is not None else manifest["cohort_seed"])
    sampling_cfg = {
        "mode": args.sampling_mode,
        "n_repeats": args.n_repeats,
        "n_pairs": args.n_pairs,
        "with_replacement": False,
        "allow_self_comparison": False,
        "seed": seed,
    }
    tasks = _generate_pairs_non_paired(papers, sampling_cfg)
    run_id = f"autoranker_{manifest['dataset']}_iter{manifest['target_iteration']}_cohort{manifest['cohort_index']}"
    run_datetime = datetime.now(UTC).isoformat()

    dataset_desc_lookup = collect_dataset_descriptions(papers)
    # Ancestors come from all accepted papers of the cohort's runs up to its iteration.
    cited_papers_lookup = build_citation_chains(
        papers,
        citable=collect_citable_papers(
            manifest["snapshot_dirs"],
            [max(int(p.get("source_iteration", manifest["target_iteration"])) for p in papers)],
        ),
    )
    ranker_cfg = OmegaConf.to_container(cfg.ranker, resolve=True)
    parallelism_cfg = OmegaConf.to_container(cfg.parallelism, resolve=True)

    comparisons, traces = _run_comparisons_parallel(
        tasks,
        llm,
        ranker_cfg,
        sandbox_client,
        manifest["target_iteration"],
        False,
        parallelism_cfg,
        run_id,
        run_datetime,
        dataset_desc_lookup=dataset_desc_lookup,
        cited_papers_lookup=cited_papers_lookup,
    )
    paper_data = _collect_paper_data_non_paired(papers, "cohort")
    summary = _build_summary_non_paired(
        run_id=run_id,
        run_datetime=run_datetime,
        duration_seconds=0.0,
        collection_info={
            "paper_pool": {
                "label": "cohort",
                "snapshot_dirs": manifest["snapshot_dirs"],
                "papers": papers,
            }
        },
        iterations=manifest["target_iteration"],
        cumulative=False,
        sampling_cfg=sampling_cfg,
        papers=papers,
        comparisons=comparisons,
        parallelism_cfg=parallelism_cfg,
        paper_data=paper_data,
        pool_label="cohort",
    )
    _save_results(comparisons, summary, ranker_cfg, {}, traces, str(output_dir), paper_data)
    write_json(output_dir / "autoranker_pairs.json", comparisons)
    scores = aggregate_win_rate_scores(comparisons, paper_ids=manifest["candidate_paper_ids"])
    write_json(output_dir / "autoranker_scores.json", scores)
    write_json(output_dir / "manifest.json", manifest)
    print(f"Wrote AutoRanker outputs to {os.fspath(output_dir)}")


if __name__ == "__main__":
    main()
