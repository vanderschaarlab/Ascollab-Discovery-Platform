"""Golden harness: what each experiment condition resolves to, without running it.

For every ``run`` command printed by ``scripts/experiments/*/run.sh`` (``DRY_RUN=1``), this composes
the Hydra config and records a JSON fixture with:

- ``config``: the resolved experiment config (endpoints and run name removed, so no credentials or
  timestamps are ever recorded);
- ``tools``: the research agents' tool names, built the same way ``system.main`` builds them;
- ``prompts``: the research, reviewer and meta-reviewer system prompts rendered by their builders
  for a fixed synthetic agent, with every config-driven flag mapped from the composed config;
- ``reviewer_assignment``: the reviewer assignment for a fixed synthetic set of papers.

A component that cannot be built for a condition is recorded as ``{"error": ...}`` instead of
failing, so the fixtures show exactly which conditions need later work.

Regenerate with ``GOLDEN_UPDATE=1 pytest tests/golden``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tests.isolation import script_env

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
SENSITIVE_KEYS = {"endpoints", "api_key", "azure_endpoint", "azure_deployment", "base_url"}
VOLATILE_KEYS = {"run_name"}


def collect_conditions() -> dict[str, dict[str, Any]]:
    """Return ``{fixture_id: {"config_name": ..., "overrides": [...]}}`` for every run command."""
    env = script_env(DRY_RUN="1", DRY_RUN_FORMAT="json")
    conditions: dict[str, dict[str, Any]] = {}
    for script in sorted((REPO_ROOT / "scripts" / "experiments").glob("*/run.sh")):
        out = subprocess.run(
            ["bash", str(script)], env=env, check=True, capture_output=True, text=True
        ).stdout
        for line in out.splitlines():
            row = json.loads(line)
            if row["kind"] != "run":
                continue
            argv = row["argv"]
            i = argv.index("--config-name")
            fid = f"{row['experiment']}__{row['condition']}__{row['dataset']}"
            conditions.setdefault(fid, {"config_name": argv[i + 1], "overrides": argv[i + 2 :]})
    return conditions


def _strip(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {
            k: _strip(v)
            for k, v in obj.items()
            if k not in SENSITIVE_KEYS and k not in VOLATILE_KEYS
        }
    if isinstance(obj, list):
        return [_strip(x) for x in obj]
    return obj


def _guard(fn):
    try:
        return fn()
    except Exception as exc:  # recorded, not raised: see module docstring
        return {"error": f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"}


def compose(config_name: str, overrides: list[str]):
    from hydra import compose as hydra_compose
    from hydra import initialize_config_dir

    with initialize_config_dir(config_dir=str(REPO_ROOT / "sys_config"), version_base=None):
        return hydra_compose(config_name=config_name, overrides=overrides)


def _resolved_config(cfg) -> dict[str, Any]:
    from omegaconf import OmegaConf

    container = OmegaConf.to_container(cfg, resolve=False)
    stripped = _strip(container)
    return OmegaConf.to_container(OmegaConf.create(stripped), resolve=True)


def _network_flags(cfg) -> tuple[bool, bool]:
    """Mirror of system.main: which tool groups the network setting enables."""
    network = cfg.network
    setting = str(network.get("setting", "network_hetero"))
    default = setting in {"network_homo", "network_hetero"}
    collab = network.get("enable_collaboration_tools")
    memory = network.get("enable_social_memory_tools")
    return (default if collab is None else bool(collab)), (
        default if memory is None else bool(memory)
    )


def _task_config(cfg):
    """Mirror of the tool and task-config construction in system.main."""
    from env_setup.config_factory import build_config
    from system.fixed_workflow import (
        normalize_fixed_workflow_cfg,
        remove_agent_registry_search_tool,
    )

    collab, memory = _network_flags(cfg)
    tools_cfg: dict[str, Any] = {"search_agent_registry": {}}
    if memory:
        tools_cfg["search_internal_archive"] = {}
    if collab:
        tools_cfg["establish_collaboration"] = {}
        tools_cfg["communicate_with_collaborator"] = {}
    arxiv = cfg.tools.get("arxiv", {}) or {}
    if bool(arxiv.get("enabled", True)):
        tools_cfg["arxiv"] = {
            "top_k_results": int(arxiv.get("top_k_results", 3)),
            "load_max_docs": int(arxiv.get("load_max_docs", 5)),
            "doc_content_chars_max": arxiv.get("doc_content_chars_max", None),
        }
    task = build_config(
        dataset_keys=list(cfg.discovery_dataset.dataset_keys),
        python_package_set_key=cfg.discovery_dataset.target,
        tools_cfg=tools_cfg,
        goal_key=cfg.discovery_dataset.target,
    )
    fixed = normalize_fixed_workflow_cfg(cfg.research.get("fixed_workflow", {}) or {})
    remove_agent_registry_search_tool(task.primary_agent_tools, fixed)
    if not memory:
        task.primary_agent_tools.pop("search_agent_registry", None)
        task.primary_agent_tools.pop("search_internal_archive", None)
        task.collab_agent_tools.pop("search_internal_archive", None)
    return task, collab


def _profile():
    from system.schema import AgentNetworkMetadata, AgentProfile

    profile = AgentProfile(
        agent_id="agent_007",
        behavior="Curious and methodical; prefers testable, well-supported claims.",
        expertise="Survival analysis and pathway enrichment in cancer genomics.",
        expertise_topics=["survival analysis", "pathway enrichment"],
        citation_count=3,
        num_accepted_papers=2,
    )
    network = AgentNetworkMetadata(num_agents=16, num_papers=12)
    return profile, network


def _research_prompt(cfg, task) -> str:
    from agents.research.primary_agent_utils import make_primary_agent_system_prompt

    research = cfg.research
    prior = research.get("prior_papers", {}) or {}
    prior_enabled = bool(prior.get("enabled", False))
    prior_mode = str(prior.get("mode", "full"))
    graph = cfg.get("graph_logic", {}) or {}
    profile, network = _profile()
    return make_primary_agent_system_prompt(
        agent_profile=profile,
        agent_network_metadata=network,
        iteration=3,
        tools=task.primary_agent_tools,
        python_packages=task.python_packages,
        dataset_metadata=task.dataset_description,
        prior_papers=[],
        prior_paper_summaries=[],
        prior_papers_enabled=prior_enabled,
        prior_papers_mode=prior_mode,
        self_reflection_prompt_enabled=bool(
            (research.get("self_reflection_prompt", {}) or {}).get("enabled", False)
        ),
        full_citing_enabled=bool(research.get("full_citing", False))
        and prior_enabled
        and prior_mode == "full",
        graph_logic_enabled=bool(graph.get("enabled", False)),
        require_plot_citations_in_research=bool(graph.get("enabled", False))
        and bool(graph.get("require_plot_citations_in_research", False)),
        nature_style_papers_prompt_enabled=bool(
            (research.get("nature_style_papers_prompt", {}) or {}).get("enabled", False)
        ),
    )


def _reviewer_prompt(cfg) -> str:
    from agents.review.rev_agent_utils import make_reviewer_system_prompt
    from system.main import _normalize_metrics_pipeline_mode, _reviewer_metrics_allowlist

    review = cfg.review
    norm = review.get("norm_prompt", {}) or {}
    norm_text = norm.get("text") if bool(norm.get("enabled", False)) else None
    mode, _ = _normalize_metrics_pipeline_mode(review.get("metrics_pipeline", "default"))
    metrics = _reviewer_metrics_allowlist(mode, review.get("metrics_allowlist", None))
    profile, network = _profile()
    return make_reviewer_system_prompt(
        agent_profile=profile,
        agent_network_metadata=network,
        tools=None,
        require_assertion_justification=bool(review.get("require_assertion_justification", False)),
        metrics_allowlist=metrics,
        community_norms=norm_text,
        # As system.main: the guidance is left out only when the metrics pipeline is off.
        include_metrics_guidance=mode != "none",
    )


def _metareviewer_prompt(cfg) -> str:
    from agents.meta_review.meta_rev_agent_utils import make_metareviewer_system_prompt

    reviewerless = bool(
        ((cfg.get("metareview", {}) or {}).get("reviewerless", {}) or {}).get("enabled", False)
    )
    return make_metareviewer_system_prompt(review_free=reviewerless)


class _FakeAgentSearch:
    """Deterministic stand-in for the registry's embedding search: ranks agents by a hash."""

    def __init__(self, ids: list[str]):
        self._ids = ids

    def run(self, payload):
        query = json.dumps(payload, sort_keys=True, default=str)
        return sorted(self._ids, key=lambda a: hashlib.sha256(f"{query}|{a}".encode()).hexdigest())


class _Registry:
    def __init__(self, ids: list[str]):
        from system.schema import AgentProfile

        self._agents = [
            AgentProfile(
                agent_id=a,
                behavior="",
                expertise=f"expertise {a}",
                expertise_topics=[],
                citation_count=0,
                num_accepted_papers=0,
            )
            for a in ids
        ]

    def get_all_agents(self):
        return list(self._agents)


def _reviewer_assignment(cfg) -> dict[str, list[str]]:
    from system.reviewer_matching import assign_papers_to_reviewers
    from system.schema import PaperMetadataPrivate

    review = cfg.review
    n = int(cfg.network.num_agents)
    ids = [f"agent_{i:03d}" for i in range(n)]
    papers = [
        PaperMetadataPrivate(
            paper_id=f"paper_t03_{aid}",
            primary_agent_id=aid,
            collab_agent_ids=[],
            title=f"Synthetic paper by {aid}",
            abstract="Synthetic abstract.",
            manuscript="Synthetic manuscript.",
            citation_count=0,
            publication_t=3,
            cited_paper_ids=[],
        )
        for aid in ids
    ]
    seed = review.get("reviewer_assignment_seed", cfg.get("SEED", 0))
    return assign_papers_to_reviewers(
        strategy=str(review.get("reviewer_assignment_strategy", "expertise_matching")),
        agent_search_tool=_FakeAgentSearch(ids),
        num_reviewers_per_paper=int(review.num_reviewers_per_paper),
        papers=papers,
        agent_registry=_Registry(ids),
        allow_self_review=bool(review.get("allow_self_review", False)),
        seed=int(seed),
        condition_label=review.get("reviewer_assignment_condition_label"),
        iteration=3,
    )


def build_fixture(config_name: str, overrides: list[str]) -> dict[str, Any]:
    cfg = compose(config_name, overrides)
    fixture: dict[str, Any] = {"config_name": config_name, "overrides": overrides}
    fixture["config"] = _guard(lambda: _resolved_config(cfg))
    built = _guard(lambda: _task_config(cfg))
    if isinstance(built, dict):  # error record
        fixture["tools"] = built
        fixture["prompts"] = {"research": built}
    else:
        task, _collab = built
        fixture["tools"] = {
            "primary": sorted(task.primary_agent_tools),
            "collab": sorted(task.collab_agent_tools or {}),
        }
        fixture["prompts"] = {"research": _guard(lambda: _research_prompt(cfg, task))}
    fixture["prompts"]["reviewer"] = _guard(lambda: _reviewer_prompt(cfg))
    fixture["prompts"]["metareviewer"] = _guard(lambda: _metareviewer_prompt(cfg))
    fixture["reviewer_assignment"] = _guard(lambda: _reviewer_assignment(cfg))
    return fixture


def assert_no_secrets(obj: Any, path: str = "") -> None:
    """Fail loudly if a credential-bearing key ever reaches a fixture."""
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            assert k not in SENSITIVE_KEYS, f"sensitive key {k!r} at {path or '/'}"
            assert_no_secrets(v, f"{path}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            assert_no_secrets(v, f"{path}/{i}")


def dump(fixture: dict[str, Any]) -> str:
    return json.dumps(fixture, indent=1, sort_keys=True, default=str) + "\n"
