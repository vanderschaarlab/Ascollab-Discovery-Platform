"""Replay reviews for a fixed cohort under one reviewer-assignment condition."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from hydra import compose, initialize_config_dir

from env_setup.build_llms import build_llm_and_embedder, build_llm_pool, require_endpoints
from env_setup.config_factory import build_config
from stores import state
from stores.agent_registry import AgentRegistry
from stores.private_paper_archive import PrivatePaperArchive
from system.async_utils import safe_asyncio_run
from system.metareview_one_step import metareview_one_step
from system.review_one_step import review_one_step

from .cohort_utils import load_candidate_pool
from .io_utils import read_json, write_json, write_jsonl
from .replay_utils import (
    build_network_metadata,
    load_agents_from_manifest_sources,
    paper_from_snapshot_dict,
    select_top_k_by_metareview,
    serialize_metareview,
    serialize_review,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--reviewer-assignment-strategy", required=True)
    parser.add_argument("--reviewer-assignment-seed", type=int)
    parser.add_argument("--config-name", default="ascollab")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--accepted-k", nargs="+", type=int, default=[3, 5, 10])
    return parser.parse_args()


def _load_manifest_candidate_dataclasses(manifest: dict) -> list:
    max_iter = max(
        int(p.get("source_iteration", manifest["target_iteration"]))
        for p in manifest["candidate_papers"]
    )
    pool = load_candidate_pool(manifest["snapshot_dirs"], max_iteration=max_iter)
    by_id = {str(p["paper_id"]): p for p in pool}
    missing = [pid for pid in manifest["candidate_paper_ids"] if pid not in by_id]
    if missing:
        raise ValueError(f"Manifest papers missing from snapshot pool: {missing[:5]}")
    return [paper_from_snapshot_dict(by_id[pid]) for pid in manifest["candidate_paper_ids"]]


def configure_logging() -> None:
    """Show the pipeline's INFO log lines on the console, as system.main does.

    This script composes the config with hydra.compose instead of running under Hydra's
    ``@main``, so no handler is configured and Python would show only warnings. The review
    and reviewer-matching code logs its progress (e.g. the ``[reviewer_matching] sourcing``
    line) at INFO.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def main() -> None:
    configure_logging()
    args = parse_args()
    manifest = read_json(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with initialize_config_dir(config_dir=str(Path("sys_config").resolve()), version_base=None):
        cfg = compose(config_name=args.config_name)

    require_endpoints(cfg.get("endpoints"), ("llm_primary", "embedder_primary"))
    llm_default, embedder = build_llm_and_embedder(
        llm_primary=cfg.endpoints.llm_primary,
        embedder_primary=cfg.endpoints.embedder_primary,
    )
    review_backend = str(cfg.review.get("llm_backend", "") or "")
    llm_pool = build_llm_pool(cfg.get("endpoints"), keys=[review_backend] if review_backend else [])

    def llm_for(_agent_id: str):
        return llm_pool.get(cfg.review.get("llm_backend", ""), llm_default)

    candidate_papers = _load_manifest_candidate_dataclasses(manifest)
    agents = load_agents_from_manifest_sources(manifest)
    state.init_state(embedder=embedder)
    state.registry().insert(agents)
    agent_registry = AgentRegistry(embedder=embedder, agents=agents)
    private_archive = PrivatePaperArchive(items=candidate_papers)
    network_metadata = build_network_metadata(agents, candidate_papers)

    tools_cfg = {"search_agent_registry": {}}
    task_config = build_config(
        dataset_keys=list(cfg.discovery_dataset.dataset_keys),
        python_package_set_key=str(cfg.discovery_dataset.target),
        tools_cfg=tools_cfg,
        goal_key=str(cfg.discovery_dataset.target),
    )
    agent_registry_search_tool = task_config.primary_agent_tools["search_agent_registry"].object

    seed = int(
        args.reviewer_assignment_seed
        if args.reviewer_assignment_seed is not None
        else manifest["cohort_seed"]
    )
    review_result = safe_asyncio_run(
        review_one_step(
            iteration=int(manifest["target_iteration"]),
            llm=llm_default,
            llm_for=llm_for,
            candidate_papers=candidate_papers,
            network_metadata=network_metadata,
            private_archive=private_archive,
            agent_registry=agent_registry,
            agent_registry_search_tool=agent_registry_search_tool,
            task_config=task_config,
            num_reviewers_per_paper=int(cfg.review.num_reviewers_per_paper),
            review_max_iterations=int(cfg.review.review_max_iterations),
            include_code_in_reviews=bool(cfg.review.include_code_in_reviews),
            total_session_timeout_seconds=cfg.review.get("total_session_timeout_seconds", None),
            run_name=f"reviewer_matching_{args.condition}",
            allow_self_review=bool(cfg.review.get("allow_self_review", False)),
            require_assertion_justification=bool(
                cfg.review.get("require_assertion_justification", False)
            ),
            reviewer_assignment_strategy=args.reviewer_assignment_strategy,
            reviewer_assignment_seed=seed,
            reviewer_assignment_condition_label=args.condition,
            return_assignment_metadata=True,
        )
    )
    reviews_by_paper, review_logs, assignment_metadata = review_result

    meta_reviews, metareview_logs = safe_asyncio_run(
        metareview_one_step(
            iteration=int(manifest["target_iteration"]),
            llm=llm_default,
            llm_for=llm_for,
            candidate_papers=candidate_papers,
            reviews_by_paper=reviews_by_paper,
            tournament_size=int(cfg.metareview.tournament_size),
            shuffle_before_tournament=bool(cfg.metareview.shuffle_before_tournament),
            total_session_timeout_seconds=cfg.metareview.get("total_session_timeout_seconds", None),
            run_name=f"reviewer_matching_{args.condition}",
        )
    )

    private_archive.add_review_assignment_record(
        f"iteration_{manifest['target_iteration']}_{args.reviewer_assignment_strategy}_{args.condition}",
        assignment_metadata,
    )

    serialized_reviews = {
        pid: [serialize_review(review) for review in reviews]
        for pid, reviews in reviews_by_paper.items()
    }
    serialized_meta = [serialize_metareview(mr) for mr in meta_reviews]
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "review_assignments.json", assignment_metadata)
    write_json(output_dir / "reviews_by_paper.json", serialized_reviews)
    write_json(output_dir / "metareviews.json", serialized_meta)
    for k in sorted(set(args.accepted_k)):
        accepted = select_top_k_by_metareview(meta_reviews, k)
        write_json(output_dir / f"accepted_top{k}.json", {"k": k, "accepted_paper_ids": accepted})

    summary = {
        "dataset": manifest["dataset"],
        "target_iteration": manifest["target_iteration"],
        "cohort_seed": manifest["cohort_seed"],
        "condition": args.condition,
        "reviewer_assignment_strategy": args.reviewer_assignment_strategy,
        "num_candidate_papers": len(candidate_papers),
        "num_reviewed_papers": len(reviews_by_paper),
        "num_reviews_completed": sum(len(v) for v in reviews_by_paper.values()),
        "num_metareviews": len(meta_reviews),
        "accepted": {
            str(k): select_top_k_by_metareview(meta_reviews, k)
            for k in sorted(set(args.accepted_k))
        },
    }
    write_json(output_dir / "condition_summary.json", summary)
    write_jsonl(
        output_dir / "session_stats.jsonl", list(review_logs or []) + list(metareview_logs or [])
    )
    print(f"Wrote review replay outputs to {output_dir}")


if __name__ == "__main__":
    main()
