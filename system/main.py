import json
import logging
import os
import random
import re
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from agents.create_agents import init_agents_profiles
from env_setup.build_llms import (
    build_llm_and_embedder,
    build_llm_pool,
    mix_backends,
    realize_mix,
    require_endpoints,
)
from env_setup.config_factory import build_config
from env_setup.preflight_data import preflight_data
from env_setup.sandbox_utils import SandboxClient
from stores import state
from system.async_utils import patch_arxiv_client_timeout, safe_asyncio_run
from system.bootstrap_iteration import build_bootstrap_selection
from system.fixed_workflow import (
    make_fixed_workflow_context_by_agent,
    make_fixed_workflow_metadata,
    normalize_fixed_workflow_cfg,
    remove_agent_registry_search_tool,
    resolve_fixed_workflow_targets,
    validate_fixed_workflow_compliance,
)
from system.meta_researcher import (
    build_meta_phase_context_by_agent,
    build_meta_source_cards,
    evaluate_meta_research_trigger,
    init_meta_researcher_telemetry,
    normalize_meta_researcher_cfg,
    select_meta_research_agent_ids,
)
from system.metacognition_feedback import (
    build_metacognition_feedback_for_agents,
    make_metacognition_feedback_metadata,
    merge_metacognition_feedback_learning_state,
    normalize_metacognition_feedback_cfg,
    resolve_feedback_source_iteration,
)
from system.metareview_one_step import metareview_one_step
from system.research_one_step import research_one_step, set_session_admission_limit
from system.resume_snapshot import (
    archive_incomplete_tail,
    find_last_complete_iteration,
    load_full_state_from_iteration,
    load_iteration_exp_config,
    resolve_resume_source,
)
from system.review_metrics import build_review_metrics_for_papers
from system.review_one_step import review_one_step
from system.snapshot_paths import resolve_iteration_snapshot_dir, resolve_run_snapshot_dir
from system.snapshot_utils import create_snapshot
from system.trackers import SessionStatsRegistry, session_stats_registry_from_snapshot_payload
from system.update_utils import update_status_and_archives

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("langsmith.client").setLevel(logging.ERROR)
log = logging.getLogger(__name__)  # ✅ define a module logger


def _log_signal_diagnostics(
    *,
    iteration: int,
    paper_metrics_by_id: dict[str, dict[str, Any]] | None,
    agents: list[Any] | None = None,
    include_metrics: bool = True,
) -> None:
    """Log variance diagnostics to detect metric/reputation collapse early.

    Args:
        iteration: Current iteration, shown in each line.
        paper_metrics_by_id: The round's review metrics; nothing is logged without them.
        agents: Agent profiles whose reputation spread is logged.
        include_metrics: Log the per-metric lines (off for the end-of-round call, so each
            metric is logged once per round).
    """
    if not paper_metrics_by_id:
        return

    metric_values: dict[str, list[float]] = {}
    for metrics in paper_metrics_by_id.values():
        for cat, cat_vals in (metrics or {}).items():
            if not isinstance(cat_vals, dict):
                continue
            for key, val in cat_vals.items():
                if isinstance(val, (int, float)):
                    metric_values.setdefault(f"{cat}.{key}", []).append(float(val))

    for metric_key, vals in sorted(metric_values.items()) if include_metrics else ():
        if len(vals) < 2:
            continue
        std = statistics.pstdev(vals)
        if std < 1e-6:
            log.warning(f"[signal][iter={iteration}] collapsed metric {metric_key}: std={std:.6f}")
        else:
            log.info(
                f"[signal][iter={iteration}] metric={metric_key} "
                f"mean={sum(vals) / len(vals):.4f} std={std:.4f} min={min(vals):.4f} max={max(vals):.4f}"
            )

    if agents:
        rep_scores: list[float] = []
        for agent in agents:
            rep = getattr(agent, "reputation", None) or {}
            score = rep.get("reputation_score") if isinstance(rep, dict) else None
            if isinstance(score, (int, float)):
                rep_scores.append(float(score))
        if len(rep_scores) >= 2:
            rep_std = statistics.pstdev(rep_scores)
            log.info(
                f"[signal][iter={iteration}] reputation mean={sum(rep_scores) / len(rep_scores):.4f} "
                f"std={rep_std:.4f} min={min(rep_scores):.4f} max={max(rep_scores):.4f}"
            )
            if rep_std < 0.03:
                log.warning(
                    f"[signal][iter={iteration}] reputation dispersion is very low (std={rep_std:.4f}); "
                    "agent differentiation may be weak."
                )


def maybe_preflight(
    cfg: DictConfig,
    *,
    save_manifest: str | Path | None = None,
    dataset_keys: list[str] | None = None,
) -> dict | None:
    """Run dataset preflight unless disabled; honor strictness settings.

    Args:
        cfg: Hydra config object.
        save_manifest: Optional path to write the manifest JSON.
        dataset_keys: Optional subset of datasets to check.

    Returns:
        Preflight report dict, or None if skipped.
    """
    # allow ENV override too
    env_off = os.getenv("DISABLE_PREFLIGHT", "").lower() in ("1", "true", "yes")
    preflight_enabled = getattr(getattr(cfg, "data", {}), "preflight", True)
    if not preflight_enabled or env_off:
        log.info("[preflight] disabled (cfg/ENV). Skipping dataset probe.")
        return None

    # Defaults/overrides
    dataset_module = getattr(cfg, "datasets_module", "env_setup.datasets")
    base_cfg = getattr(cfg, "base_cfg", "sys_config/ascollab.yaml")
    dataset_root = getattr(cfg, "dataset_root", None)
    sample_n = getattr(cfg, "preflight_sample_n", 5)
    expected_min_sizes = getattr(cfg, "preflight_expect", None)

    # If not explicitly provided, try to use discovery_dataset keys if present
    if dataset_keys is None:
        try:
            dk = list(getattr(cfg.discovery_dataset, "dataset_keys", []))
            dataset_keys = dk if dk else None
        except Exception:
            dataset_keys = None

    report = preflight_data(
        dataset_module=dataset_module,
        base_cfg=base_cfg,
        dataset_root=dataset_root,
        save_manifest=str(save_manifest)
        if save_manifest
        else getattr(cfg, "preflight_manifest", "artifacts/data_manifest.json"),
        sample_n=sample_n,
        expected_min_sizes=expected_min_sizes,
        dataset_keys=dataset_keys,
    )

    if not report.get("ok", False):
        if getattr(getattr(cfg, "data", {}), "preflight_strict", True):
            log.error("[preflight] FAILED and strict mode is on. Aborting.")
            log.error(f"[preflight] Report: {json.dumps(report, indent=2)}")
            raise RuntimeError("Pre-flight failed")
        log.warning("[preflight] FAILED but continuing (strict=false).")
        log.error(f"[preflight] Report: {json.dumps(report, indent=2)}")
    else:
        log.info("[preflight] OK")

    return report


def _creation_order(agent: Any) -> tuple[int, int, str]:
    """Sort key giving agents in creation order: by the number ending their ID (``agent_07``).

    Args:
        agent: Agent profile with an ``agent_id``.

    Returns:
        ``(0, number, id)`` for IDs ending in ``_<number>``, else ``(1, 0, id)`` (after them).
    """
    agent_id = str(getattr(agent, "agent_id", "") or "")
    match = re.search(r"_(\d+)$", agent_id)
    return (0, int(match.group(1)), agent_id) if match else (1, 0, agent_id)


def _assign_llms_to_agents(
    agents: list[Any], llm_selections: list[tuple[str, Any]]
) -> dict[str, tuple[str, Any]]:
    """Build a mapping of agent_id -> (llm_key, llm) from selection plan.

    Args:
        agents: Iterable of agent profiles with `agent_id`/`id`.
        llm_selections: List of (backend_key, llm) at least as long as agents.

    Returns:
        Dict mapping agent_id to a tuple of (llm_key, llm).
    """
    mapping = {}
    for agent, (k, llm) in zip(agents, llm_selections, strict=False):
        agent_id = getattr(agent, "agent_id", None) or agent.id
        mapping[agent_id] = (k, llm)
        # optional: annotate the agent profile so downstream logs can see its backend
        try:
            agent.llm_key = k  # harmless if attrs are dynamic/dataclass
        except Exception:
            pass
    return mapping


def _assigned_backend_key(
    llm_by_agent: Mapping[str, tuple[str, Any]],
    agent_id: str,
) -> str:
    """Return the assigned backend key without using a typed tuple fallback.

    Mypy 2.0.0 can internal-error on `dict.get(key, ("default", llm))` in this
    module. Keeping the fallback branch explicit avoids that inference path.
    """
    entry = llm_by_agent.get(agent_id)
    if entry is None:
        return "default"
    backend_key, _llm = entry
    return str(backend_key or "default")


def _replacement_llm_selections(
    paused_rows: list[Mapping[str, Any]],
    llm_by_agent: Mapping[str, tuple[str, Any]],
    default_llm: Any,
) -> list[tuple[str, Any]]:
    """Backends for replacement agents: each takes over the backend of the agent it replaces.

    This keeps the model mix of the active agents as configured. Re-realising
    ``llm_backends`` for the few replacements would instead scale the whole mix down to them,
    which favours the first-listed backend.

    Args:
        paused_rows: Newly paused agents (each with ``agent_id``), in replacement order.
        llm_by_agent: Mapping of agent_id to (backend_key, llm).
        default_llm: LLM of agents without an assigned backend (``llm_for``'s fallback).

    Returns:
        One (backend_key, llm) per paused agent; ("default", default_llm) when the paused
        agent had no assigned backend.
    """
    selections: list[tuple[str, Any]] = []
    for row in paused_rows:
        entry = llm_by_agent.get(str(row.get("agent_id", "")))
        if entry is None:
            selections.append(("default", default_llm))
        else:
            selections.append(entry)
    return selections


def _extract_llm_runtime_metadata(llm_obj: Any) -> dict[str, Any]:
    deployment = None
    for attr in ("azure_deployment", "deployment_name", "model_name", "model", "deployment"):
        value = getattr(llm_obj, attr, None)
        if value is not None and str(value).strip():
            deployment = str(value).strip()
            break
    reasoning_effort = getattr(llm_obj, "reasoning_effort", None)
    model_kwargs = getattr(llm_obj, "model_kwargs", None)
    if isinstance(model_kwargs, Mapping):
        if reasoning_effort in (None, ""):
            reasoning_effort = model_kwargs.get("reasoning_effort", reasoning_effort)
        if deployment is None:
            model_name = model_kwargs.get("model")
            if model_name is not None and str(model_name).strip():
                deployment = str(model_name).strip()
    return {
        "deployment": (str(deployment).strip() if deployment is not None else None),
        "reasoning_effort": (
            str(reasoning_effort).strip()
            if reasoning_effort is not None and str(reasoning_effort).strip()
            else None
        ),
    }


def _normalize_metrics_pipeline_mode(raw_value: Any) -> tuple[str, bool]:
    """Normalize metrics pipeline mode.

    Returns:
        Tuple of (normalized_mode, fallback_applied).
        normalized_mode is one of: "default", "experimental", "none".
        fallback_applied is True when an unknown value was coerced to "default".
    """
    mode = str(raw_value or "default").strip().lower()
    if mode in {"default", "experimental"}:
        return mode, False
    if mode in {"none", "off", "disabled"}:
        return "none", False
    return "default", True


def _reviewer_metrics_allowlist(metrics_pipeline: str, metrics_allowlist: Any) -> list[str] | None:
    """Return the metrics allowlist the reviewer prompt lists; ``None`` lists every metric.

    Only the experimental pipeline narrows the reviewer prompt to ``review.metrics_allowlist``.

    Args:
        metrics_pipeline: Mode as returned by ``_normalize_metrics_pipeline_mode``.
        metrics_allowlist: ``review.metrics_allowlist`` from the config (any sequence, or None).

    Returns:
        The allowlist as a list in experimental mode, otherwise None.
    """
    if metrics_pipeline != "experimental" or metrics_allowlist is None:
        return None
    return list(metrics_allowlist)


def _resolve_session_create_hard_timeout_seconds(
    research_cfg: Mapping[str, Any] | None,
    *,
    default_seconds: float = 600.0,
) -> float:
    """Resolve session-create timeout with backward-compatible key support."""
    cfg = dict(research_cfg or {}) if isinstance(research_cfg, Mapping) else {}
    raw_timeout = cfg.get(
        "session_create_hard_timeout_seconds",
        cfg.get("session_create_hard_timeout", default_seconds),
    )
    try:
        timeout_seconds = float(raw_timeout)
    except Exception:
        timeout_seconds = float(default_seconds)
    if timeout_seconds <= 0:
        timeout_seconds = float(default_seconds)
    return timeout_seconds


def _resolve_research_admission_limit(
    *,
    max_sessions: int,
    network_num_agents: int,
    max_concurrent_research_agents: Any = None,
) -> tuple[int, int, int | None]:
    """Compute bounded admission size for concurrent research session creation."""
    max_sessions_norm = max(1, int(max_sessions))
    reserve = max(4, max_sessions_norm // 8)
    baseline_capacity = max(1, max_sessions_norm - reserve)
    network_cap = max(1, int(network_num_agents))
    admission = min(baseline_capacity, network_cap)

    override_cap: int | None = None
    if max_concurrent_research_agents is not None:
        try:
            candidate = int(max_concurrent_research_agents)
        except Exception:
            candidate = 0
        if candidate > 0:
            override_cap = candidate
            admission = min(admission, override_cap)

    return admission, reserve, override_cap


def _next_agent_ids(existing_ids: list[str], *, prefix: str, count: int) -> list[str]:
    """Generate unique sequential IDs like '<prefix>_01' beyond current max."""
    if count <= 0:
        return []
    pat = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    max_idx = 0
    width = 2
    for aid in existing_ids:
        m = pat.match(str(aid))
        if not m:
            continue
        digits = m.group(1)
        idx = int(digits)
        max_idx = max(max_idx, idx)
        width = max(width, len(digits))
    start = max_idx + 1
    return [f"{prefix}_{i:0{width}d}" for i in range(start, start + count)]


def _paused_sets(agents, *, iteration: int, pause_cfg: dict) -> tuple[set[str], set[str], set[str]]:
    """Return (pause_full, pause_review, pause_collab) agent_id sets."""
    mode = str(pause_cfg.get("mode", "none") or "none").strip().lower()
    pause_full: set[str] = set()
    pause_review: set[str] = set()
    pause_collab: set[str] = set()
    permanent_modes = {
        "full_pause_permanent",
        "permanent_full_pause",
        "full_pause_permanent_replace",
        "full_pause_permanent_scheduled",
    }

    if mode in ("none", "off", "disabled", "adjust_research_iters"):
        return pause_full, pause_review, pause_collab

    for agent in agents:
        agent_id = getattr(agent, "agent_id", None) or agent.id
        pause_state = getattr(agent, "pause_state", None) or {}
        active = bool(pause_state.get("active", False))
        if not active:
            continue

        state_mode = str(pause_state.get("mode", "none") or "none").strip().lower()
        is_permanent = bool(pause_state.get("permanent", False)) or state_mode in permanent_modes
        until_iter = pause_state.get("until_iteration")
        paused_now = is_permanent or (isinstance(until_iter, int) and iteration <= until_iter)
        if not paused_now:
            continue

        if mode in (
            "full_pause",
            "full_pause_permanent",
            "permanent_full_pause",
            "full_pause_permanent_replace",
            "full_pause_permanent_scheduled",
        ):
            pause_full.add(agent_id)
            pause_review.add(agent_id)
            pause_collab.add(agent_id)
        elif mode == "no_review_or_collab":
            pause_review.add(agent_id)
            pause_collab.add(agent_id)
        elif mode == "no_review":
            pause_review.add(agent_id)

    return pause_full, pause_review, pause_collab


def _research_iter_overrides(
    agents,
    *,
    base_iters: int,
    pause_cfg: dict,
) -> dict[str, int]:
    mode = str(pause_cfg.get("mode", "none") or "none").strip().lower()
    if mode != "adjust_research_iters":
        return {}

    try:
        delta = int(pause_cfg.get("iteration_delta", 0))
    except Exception:
        delta = 0
    if delta == 0:
        return {}

    threshold = float(pause_cfg.get("threshold", 0))
    apply_missing = bool(pause_cfg.get("apply_when_score_missing", False))
    min_iters = int(pause_cfg.get("min_research_iters", 1))

    overrides: dict[str, int] = {}
    for agent in agents:
        agent_id = getattr(agent, "agent_id", None) or agent.id
        rep = getattr(agent, "reputation", None) or {}
        score = rep.get("reputation_score") if isinstance(rep, dict) else None
        if score is None and not apply_missing:
            continue
        below = (score is not None and score < threshold) or (score is None and apply_missing)
        adjusted = base_iters - delta if below else base_iters + delta
        adjusted = max(min_iters, adjusted)
        if adjusted != base_iters:
            overrides[agent_id] = adjusted
    return overrides


def validate_reviewerless_config(exp_cfg: DictConfig, task_overrides: Sequence[str] = ()) -> None:
    """Fail fast when the reviewerless ablation is combined with reviewer-dependent settings.

    The reviewerless ablation (experiment 07) removes reviewers entirely. Any setting that
    consumes reviewer/metareview output would silently make it a multi-factor ablation,
    so we refuse to start. Keys that merely become inert only warn, and only when the
    user set them (``task_overrides`` are the Hydra command-line overrides).
    """
    review_cfg = getattr(exp_cfg, "review", {}) or {}
    metareview_cfg = getattr(exp_cfg, "metareview", {}) or {}
    research_cfg = getattr(exp_cfg, "research", {}) or {}
    reputation_cfg = getattr(exp_cfg, "reputation", {}) or {}
    reputation_pause_cfg = getattr(exp_cfg, "reputation_pause", {}) or {}

    reviewerless = bool((metareview_cfg.get("reviewerless", {}) or {}).get("enabled", False))
    if not reviewerless:
        return

    offenders: list[str] = []
    if bool(reputation_cfg.get("enabled", False)):
        offenders.append("reputation.enabled=true")
    pause_mode = str(reputation_pause_cfg.get("mode", "none") or "none").strip().lower()
    if pause_mode != "none":
        offenders.append(f"reputation_pause.mode={pause_mode} (expected none)")
    if bool((research_cfg.get("metacognition_feedback", {}) or {}).get("enabled", False)):
        offenders.append("research.metacognition_feedback.enabled=true")

    if offenders:
        raise ValueError(
            "metareview.reviewerless.enabled=true is incompatible with settings that "
            "presuppose reviewers (they consume reviewer/metareview output and would make "
            "the reviewerless ablation a multi-factor ablation). Disable: " + "; ".join(offenders)
        )

    # Keys that merely become inert under reviewerless — warn, do not fail.
    inert: list[str] = []
    overridden = {str(ov).lstrip("+~").split("=", 1)[0].strip() for ov in task_overrides}
    if "review.num_reviewers_per_paper" in overridden:
        inert.append("review.num_reviewers_per_paper")
    if bool((review_cfg.get("norm_prompt", {}) or {}).get("enabled", False)):
        inert.append("review.norm_prompt")
    if inert:
        log.warning(
            "[reviewerless] the following keys are inert with no reviewers: %s",
            ", ".join(inert),
        )


_LLM_FEEDBACK_MODES = {"meta_review_only", "meta_review_with_metrics"}


def _llm_pool_keys(exp_cfg: DictConfig) -> list[str]:
    """Backends to build: those with a share in the model mix, plus the feedback backend.

    ``research.metacognition_feedback.llm_backend`` is added only when metacognition
    feedback is enabled in an LLM mode (meta_review_only, meta_review_with_metrics), the
    only case that uses it. Its endpoint file is then required, so a missing file stops the
    run at start-up instead of at the first feedback round.

    Args:
        exp_cfg: The composed run config.

    Returns:
        Endpoint keys for build_llm_pool.

    Raises:
        EndpointConfigError: If the feedback backend is needed and its endpoint file is
            missing.
    """
    llm_backends_cfg = getattr(exp_cfg, "llm_backends", {}) or {}
    keys = mix_backends(
        llm_backends_cfg.get("counts", None), llm_backends_cfg.get("fractions", None)
    )
    feedback_cfg = normalize_metacognition_feedback_cfg(
        exp_cfg.research.get("metacognition_feedback", {}) or {}
    )
    backend = str(feedback_cfg.get("llm_backend") or "").strip()
    if backend and feedback_cfg["mode"] in _LLM_FEEDBACK_MODES and backend != "primary":
        require_endpoints(exp_cfg.get("endpoints"), (backend,))
        if backend not in keys:
            keys.append(backend)
    return keys


@hydra.main(version_base=None, config_path="../sys_config", config_name="ascollab")
def main(exp_cfg: DictConfig):
    """Hydra entrypoint orchestrating research, review, and metareview loops."""
    # Set up logging AFTER Hydra changes CWD, so logs go into the run dir tree
    from system.logging_utils import setup_file_logging_hydra_safe, tee_prints_to_logging

    # Global seeding for reproducibility (best-effort)
    seed = int(exp_cfg.get("SEED", 0) or 0)
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass

    log_path = setup_file_logging_hydra_safe(filename="run.log", level=logging.INFO)
    orig_out, orig_err = tee_prints_to_logging(mirror_stdout=True, mirror_stderr=True)
    log.info(f"[logging] tee active; file -> {log_path}")

    # --- Resolve project root & run dir robustly under Hydra ---
    hydra_cfg = HydraConfig.get()
    project_root = Path(hydra_cfg.runtime.cwd)
    task_overrides = [str(x).strip() for x in (getattr(hydra_cfg.overrides, "task", []) or [])]

    def _has_task_override(path: str) -> bool:
        if not path:
            return False
        prefixes = (f"{path}=", f"+{path}=", f"++{path}=")
        return any(ov.startswith(prefixes) for ov in task_overrides)

    incoming_exp_cfg = exp_cfg
    incoming_resume_cfg = getattr(incoming_exp_cfg, "resume", None) or {}
    resume_enabled = bool(incoming_resume_cfg.get("enabled", False))
    resume_state = None
    resume_input_path: str | None = None
    resume_source_run_dir: Path | None = None
    resume_source_cap_iteration: int | None = None
    resume_archived_dirs: list[Path] = []

    if resume_enabled:
        resume_input_path = str(incoming_resume_cfg.get("snapshot_path", "") or "").strip()
        if not resume_input_path:
            raise ValueError("resume.enabled is true but resume.snapshot_path is missing.")
        incoming_bootstrap_enabled = bool(
            (getattr(incoming_exp_cfg, "bootstrap", None) or {}).get("enabled", False)
        )
        incoming_bootstrap_override = _has_task_override("bootstrap.enabled")
        # In resume mode we load iteration_<last_complete>/exp_config.yaml as baseline.
        # Only treat bootstrap as conflicting when the user explicitly forces it on.
        if incoming_bootstrap_override and incoming_bootstrap_enabled:
            raise ValueError("resume.enabled and bootstrap.enabled cannot both be true.")

        resume_source_run_dir, resolved_cap = resolve_resume_source(
            resume_input_path,
            project_root=project_root,
        )
        resume_source_cap_iteration = int(resolved_cap) if resolved_cap is not None else None
        restored_iteration, restored_iteration_dir = find_last_complete_iteration(
            resume_source_run_dir,
            max_iteration=resume_source_cap_iteration,
        )
        resume_state = load_full_state_from_iteration(restored_iteration_dir)
        restored_cfg = load_iteration_exp_config(restored_iteration_dir)
        saved_bootstrap_enabled = bool(
            (getattr(restored_cfg, "bootstrap", None) or {}).get("enabled", False)
        )
        exp_cfg = OmegaConf.create(OmegaConf.to_container(restored_cfg, resolve=False))

        # Snapshot exp_config is redacted on write; always source runtime endpoint credentials
        # from the active invocation config.
        incoming_endpoints_cfg = getattr(incoming_exp_cfg, "endpoints", None)
        if incoming_endpoints_cfg is not None:
            incoming_endpoints_payload = OmegaConf.to_container(
                incoming_endpoints_cfg, resolve=False
            )
            if isinstance(incoming_endpoints_payload, dict):
                OmegaConf.update(
                    exp_cfg,
                    "endpoints",
                    incoming_endpoints_payload,
                    merge=False,
                    force_add=True,
                )
                log.info(
                    "[resume] using endpoints from active runtime config "
                    "(snapshot exp_config endpoints are redacted)."
                )

        # Keep only continuation-related runtime overrides by default.
        incoming_resume_payload = OmegaConf.to_container(
            OmegaConf.create(dict(incoming_resume_cfg)),
            resolve=True,
        )
        OmegaConf.update(
            exp_cfg,
            "resume",
            dict(incoming_resume_payload) if isinstance(incoming_resume_payload, dict) else {},
            merge=True,
            force_add=True,
        )
        OmegaConf.update(exp_cfg, "resume.enabled", True, merge=False, force_add=True)
        OmegaConf.update(
            exp_cfg, "resume.snapshot_path", resume_input_path, merge=False, force_add=True
        )
        OmegaConf.update(exp_cfg, "bootstrap.enabled", False, merge=False, force_add=True)
        OmegaConf.update(
            exp_cfg, "run_name", str(resume_state.run_name), merge=False, force_add=True
        )
        log.info(
            "[resume] loaded runtime baseline config from %s (saved bootstrap.enabled=%s); "
            "forcing bootstrap.enabled=false in resume mode",
            restored_iteration_dir / "exp_config.yaml",
            saved_bootstrap_enabled,
        )

        if _has_task_override("network.iterations"):
            OmegaConf.update(
                exp_cfg,
                "network.iterations",
                int(incoming_exp_cfg.network.iterations),
                merge=False,
                force_add=True,
            )

        ignored_overrides = []
        for ov in task_overrides:
            normalized = ov.lstrip("+")
            if normalized.startswith("resume.") or normalized.startswith("network.iterations="):
                continue
            ignored_overrides.append(ov)
        if ignored_overrides:
            log.info(
                "[resume] using saved exp_config baseline from %s and ignoring non-resume overrides: %s",
                restored_iteration_dir / "exp_config.yaml",
                ignored_overrides,
            )

        runtime_resume_cfg = getattr(exp_cfg, "resume", None) or {}
        strict_count_match = bool(runtime_resume_cfg.get("strict_agent_count_match", True))
        restored_agent_count = len(resume_state.agents)
        configured_agent_count = int(exp_cfg.network.num_agents)
        if restored_agent_count != configured_agent_count:
            if strict_count_match:
                raise ValueError(
                    f"Resumed snapshot has {restored_agent_count} agents but "
                    f"network.num_agents={configured_agent_count}. "
                    "Set resume.strict_agent_count_match=false to continue with restored agent count."
                )
            log.warning(
                "[resume] overriding network.num_agents from %s to restored agent count %s",
                configured_agent_count,
                restored_agent_count,
            )
            OmegaConf.update(
                exp_cfg,
                "network.num_agents",
                restored_agent_count,
                merge=False,
                force_add=True,
            )

        if bool(runtime_resume_cfg.get("archive_incomplete_tail", True)):
            resume_archived_dirs = archive_incomplete_tail(
                resume_state.run_dir,
                restored_iteration=resume_state.restored_iteration,
            )
            for archived_path in resume_archived_dirs:
                log.info("[resume] archived incomplete iteration dir: %s", archived_path)

    if resume_enabled and resume_state is not None:
        run_name = str(resume_state.run_name)
        run_dir = resume_state.run_dir
        log.info(
            "[resume] source=%s run_dir=%s restored_iteration=%s start_iteration=%s target_iteration=%s",
            resume_input_path,
            run_dir,
            resume_state.restored_iteration,
            int(resume_state.restored_iteration) + 1,
            int(exp_cfg.network.iterations),
        )
        if resume_source_cap_iteration is not None:
            log.info("[resume] iteration cap from source path: <=%s", resume_source_cap_iteration)
    else:
        # Your config already defines run_name; fall back to a timestamp if missing.
        run_name = exp_cfg.get("run_name")
        if not run_name:
            from datetime import datetime

            run_name = f"{exp_cfg.get('exp_name', 'exp')}_{datetime.now().strftime('%y%m%d_%H%M')}"

        run_dir = resolve_run_snapshot_dir(
            run_name=run_name,
            base_snapshots_dir="snapshots",
            project_dir=project_root,
            create=True,
        )

    # --- Preflight data (optional/strictness configurable) ---
    manifest_path = run_dir / "data_manifest.json"
    preflight_report = maybe_preflight(
        exp_cfg,
        save_manifest=manifest_path,
        dataset_keys=list(getattr(exp_cfg.discovery_dataset, "dataset_keys", [])),
    )

    # Persist a compact “data card” alongside the manifest (useful for prompts)
    if preflight_report is not None:
        data_card = {
            "ok": preflight_report.get("ok", False),
            "datasets": {
                k: [
                    {
                        "path": f.get("path"),
                        "exists": f.get("exists"),
                        "n_rows": f.get("schema", {}).get("n_rows"),
                        "n_cols": f.get("schema", {}).get("n_cols"),
                        "columns": (f.get("schema", {}).get("columns") or [])[:10],  # trim
                    }
                    for f in v.get("files", [])
                    if f.get("exists")
                ]
                for k, v in preflight_report.get("datasets", {}).items()
            },
        }
        with open(run_dir / "data_card.json", "w") as fp:
            json.dump(data_card, fp, indent=2)

    # --- Core config / build ---
    network_setting = exp_cfg.network.setting
    assert network_setting in ["network_hetero", "network_homo", "indep_homo"], (
        f"Got unknown network.setting: {network_setting}"
    )
    network_cfg = getattr(exp_cfg, "network", {}) or {}
    if not isinstance(network_cfg, Mapping):
        network_cfg = {}
    default_enable_collaboration_tools = network_setting in {"network_homo", "network_hetero"}
    default_enable_social_memory_tools = network_setting in {"network_homo", "network_hetero"}
    collab_tools_override = network_cfg.get("enable_collaboration_tools")
    social_memory_tools_override = network_cfg.get("enable_social_memory_tools")
    enable_collaboration_tools = (
        default_enable_collaboration_tools
        if collab_tools_override is None
        else bool(collab_tools_override)
    )
    enable_social_memory_tools = (
        default_enable_social_memory_tools
        if social_memory_tools_override is None
        else bool(social_memory_tools_override)
    )
    log.info(
        "[network] setting=%s enable_collaboration_tools=%s enable_social_memory_tools=%s",
        network_setting,
        enable_collaboration_tools,
        enable_social_memory_tools,
    )

    # --- Social-system experiment switches (experiments 07-09). All are off (cumulative) by default. ---
    metareview_cfg_top = getattr(exp_cfg, "metareview", {}) or {}
    reviewerless_enabled = bool(
        (metareview_cfg_top.get("reviewerless", {}) or {}).get("enabled", False)
    )
    validate_reviewerless_config(exp_cfg, task_overrides)

    social_memory_mode = (
        str((network_cfg.get("social_memory", {}) or {}).get("mode", "cumulative") or "cumulative")
        .strip()
        .lower()
    )
    if social_memory_mode not in {"cumulative", "markovian"}:
        log.warning(
            "[social_memory] unknown network.social_memory.mode=%r; falling back to 'cumulative'.",
            social_memory_mode,
        )
        social_memory_mode = "cumulative"

    review_cfg_top = getattr(exp_cfg, "review", {}) or {}
    norm_prompt_cfg = (
        (review_cfg_top.get("norm_prompt", {}) or {}) if isinstance(review_cfg_top, Mapping) else {}
    )
    reviewer_norm_text = None
    if bool(norm_prompt_cfg.get("enabled", False)):
        reviewer_norm_text = norm_prompt_cfg.get("text", None)
        if reviewer_norm_text:
            log.info(
                "[review][norm] injecting reviewer norm label=%s",
                norm_prompt_cfg.get("label"),
            )
        else:
            log.warning(
                "[review][norm] norm_prompt.enabled=true but text is empty; no norm injected."
            )
    log.info(
        "[social_experiments] reviewerless=%s social_memory_mode=%s norm=%s",
        reviewerless_enabled,
        social_memory_mode,
        bool(reviewer_norm_text),
    )

    if exp_cfg.run_name is None:
        raise ValueError("exp_cfg.run_name must be set for tracing purposes.")

    # -- Patch arxiv library to enforce HTTP timeouts (must run before any tool creation) --
    tool_cfg = getattr(exp_cfg, "tools", {}) or {}
    research_cfg_for_tools = getattr(exp_cfg, "research", {}) or {}
    fixed_workflow_cfg_for_tools = normalize_fixed_workflow_cfg(
        research_cfg_for_tools.get("fixed_workflow", {})
        if isinstance(research_cfg_for_tools, Mapping)
        else {}
    )
    arxiv_http_timeout = float(tool_cfg.get("arxiv_http_timeout_seconds", 30))
    patch_arxiv_client_timeout(timeout=arxiv_http_timeout)

    tools_cfg = {
        # Keep registry-search available for reviewer matching logic below.
        "search_agent_registry": {},
    }
    if enable_social_memory_tools:
        tools_cfg["search_internal_archive"] = {}
    if enable_collaboration_tools:
        tools_cfg["establish_collaboration"] = {}
        tools_cfg["communicate_with_collaborator"] = {}
    arxiv_cfg = tool_cfg.get("arxiv", {}) or {}
    if bool(arxiv_cfg.get("enabled", True)):
        tools_cfg["arxiv"] = {
            "top_k_results": int(arxiv_cfg.get("top_k_results", 3)),
            "load_max_docs": int(arxiv_cfg.get("load_max_docs", 5)),
            "doc_content_chars_max": arxiv_cfg.get("doc_content_chars_max", None),
        }

    task_config = build_config(
        dataset_keys=exp_cfg.discovery_dataset.dataset_keys,
        python_package_set_key=exp_cfg.discovery_dataset.target,
        tools_cfg=tools_cfg,
        goal_key=exp_cfg.discovery_dataset.target,
    )
    agent_registry_search_tool = task_config.primary_agent_tools["search_agent_registry"].object
    remove_agent_registry_search_tool(task_config.primary_agent_tools, fixed_workflow_cfg_for_tools)
    if not enable_social_memory_tools:
        task_config.primary_agent_tools.pop("search_agent_registry", None)
        task_config.primary_agent_tools.pop("search_internal_archive", None)
        task_config.collab_agent_tools.pop("search_internal_archive", None)

    # The primary model and the embedder always need an endpoint file; mix backends are checked
    # when the pool is built, the feedback backend in _llm_pool_keys.
    require_endpoints(exp_cfg.get("endpoints"), ("llm_primary", "embedder_primary"))
    # Build the embedder as before; also keep a single default llm for backward compat
    llm_default, embedder = build_llm_and_embedder(
        llm_primary=exp_cfg.endpoints.llm_primary,
        embedder_primary=exp_cfg.endpoints.embedder_primary,
    )

    rng = random.Random(getattr(exp_cfg, "SEED", None) or 42)
    mix_counts = (getattr(exp_cfg, "llm_backends", {}) or {}).get("counts", None)
    mix_fracs = (getattr(exp_cfg, "llm_backends", {}) or {}).get("fractions", None)
    mix_shuffle = (getattr(exp_cfg, "llm_backends", {}) or {}).get("shuffle", True)

    # Build only the backends this run uses, so unused endpoint files can stay templates.
    llm_pool = build_llm_pool(exp_cfg.get("endpoints"), keys=_llm_pool_keys(exp_cfg))

    # Realize the requested mix into a concrete list for num_agents

    llm_selections = realize_mix(
        llm_pool,
        num_agents=exp_cfg.network.num_agents,
        counts=mix_counts,
        fractions=mix_fracs,
        shuffle=mix_shuffle,
        rng=rng,
    )

    # Build/insert agents (resume, bootstrap, or fresh init)
    bootstrap_cfg = getattr(exp_cfg, "bootstrap", None) or {}
    resume_cfg = getattr(exp_cfg, "resume", None) or {}
    bootstrap_enabled = bool(bootstrap_cfg.get("enabled", False))
    resume_enabled = bool(resume_cfg.get("enabled", False))
    if resume_enabled and bootstrap_enabled:
        raise ValueError("resume.enabled and bootstrap.enabled cannot both be true.")

    if bootstrap_enabled:
        warn_noop = bool(bootstrap_cfg.get("warn_if_network_setting_ignored", True))
        fail_noop = bool(bootstrap_cfg.get("fail_if_network_setting_ignored", False))
        msg = (
            f"[bootstrap] bootstrap.enabled=true; network.setting='{network_setting}' does not "
            "reinitialize personas in bootstrap mode."
        )
        if warn_noop:
            log.warning(msg)
        if fail_noop:
            raise ValueError(
                msg
                + " Set bootstrap.fail_if_network_setting_ignored=false to continue intentionally."
            )
    bootstrap_iteration = int(bootstrap_cfg.get("iteration", 1))
    start_iteration = 1
    bootstrap_public_papers = []
    bootstrap_private_papers = []
    bootstrap_reviews_by_paper = {}
    bootstrap_notepad_entries: dict[str, list[dict]] = {}
    resumed_public_papers = []
    resumed_private_papers = []
    resumed_reviews_by_paper: dict[str, list[dict]] = {}
    resumed_review_assignments: dict[str, dict] = {}
    resumed_notepad_entries: dict[str, list[dict]] = {}

    if resume_enabled:
        if resume_state is None:
            raise RuntimeError("resume.enabled=true but no resume state was loaded.")
        start_iteration = int(resume_state.restored_iteration) + 1
        init_agents = list(resume_state.agents or [])
        resumed_public_papers = list(resume_state.public_papers or [])
        resumed_private_papers = list(resume_state.private_papers or [])
        resumed_reviews_by_paper = dict(resume_state.reviews_by_paper or {})
        resumed_review_assignments = dict(getattr(resume_state, "review_assignments", {}) or {})
        resumed_notepad_entries = dict(resume_state.notepad_entries or {})
        log.info(
            "[resume] restored agents=%s public_papers=%s private_papers=%s start_iteration=%s",
            len(init_agents),
            len(resumed_public_papers),
            len(resumed_private_papers),
            start_iteration,
        )
    elif bootstrap_enabled:
        start_iteration = int(bootstrap_cfg.get("start_iteration", bootstrap_iteration + 1))
        pool_path_raw = bootstrap_cfg.get("pool_path")
        if not pool_path_raw:
            raise ValueError("bootstrap.enabled is true but bootstrap.pool_path is missing.")
        try:
            from omegaconf import ListConfig
        except Exception:
            ListConfig = ()
        is_seq = isinstance(pool_path_raw, (list, tuple, ListConfig))
        pool_path_values = pool_path_raw if is_seq else [pool_path_raw]
        pool_paths: list[Path] = []
        for raw in pool_path_values:
            path = Path(str(raw))
            if not path.is_absolute():
                path = (project_root / path).resolve()
            pool_paths.append(path)

        num_agents = int(bootstrap_cfg.get("num_agents", exp_cfg.network.num_agents))
        if num_agents != exp_cfg.network.num_agents:
            raise ValueError("bootstrap.num_agents must match network.num_agents.")

        (
            init_agents,
            bootstrap_public_papers,
            bootstrap_private_papers,
            bootstrap_reviews_by_paper,
            bootstrap_notepad_entries,
        ) = build_bootstrap_selection(
            pool_paths=pool_paths,
            num_agents=num_agents,
            seed=seed,
            max_public_papers_per_agent=bootstrap_cfg.get("max_public_papers_per_agent"),
            max_private_papers_per_agent=bootstrap_cfg.get("max_private_papers_per_agent"),
        )
        log.info(
            f"[bootstrap] selected_agents={len(init_agents)} "
            f"public_papers={len(bootstrap_public_papers)} private_papers={len(bootstrap_private_papers)} "
            f"seed={seed} iteration={bootstrap_iteration}"
        )
    else:
        try:
            init_agents = init_agents_profiles(
                llm=llm_default,  # still pass something; some builders may expect it
                dataset_metadata=task_config.dataset_description,
                n=exp_cfg.network.num_agents,
                network_setting=network_setting,
            )
        except Exception as e:
            log.exception("[main] Failed to initialise agent profiles")
            # Fail fast – don’t try to run the rest of the pipeline
            raise e

    # Apply per-agent metric weights from config (review.metric_weights.{default,by_agent})
    metric_weights_cfg = getattr(exp_cfg.review, "metric_weights", {}) or {}
    default_metric_weights = (
        metric_weights_cfg.get("default", {}) if isinstance(metric_weights_cfg, Mapping) else {}
    )
    metric_weights_by_agent = (
        metric_weights_cfg.get("by_agent", {}) if isinstance(metric_weights_cfg, Mapping) else {}
    )

    def _normalize_weights(raw: Any) -> dict[str, float]:
        weights: dict[str, float] = {}
        if not isinstance(raw, Mapping):
            return weights
        for k, v in raw.items():
            try:
                weights[str(k).lower()] = float(v)
            except Exception:
                continue
        return weights

    default_weights_norm = _normalize_weights(default_metric_weights)
    for agent in init_agents:
        agent_weights = dict(default_weights_norm)
        agent_weights.update(_normalize_weights(metric_weights_by_agent.get(agent.agent_id)))
        agent.metric_weights = agent_weights or None

    # Create a per-agent mapping and a tiny lookup callable (using the freshly created agents).
    # The model plan is per position in creation order (agent_01, agent_02, ...); a resumed
    # registry lists its agents in another order, so restore that order before pairing, or
    # agents would change model on resume.
    assign_order = sorted(init_agents, key=_creation_order) if resume_enabled else init_agents
    llm_by_agent = _assign_llms_to_agents(assign_order, llm_selections)
    default_llm_runtime_metadata = _extract_llm_runtime_metadata(llm_default)

    def llm_for(agent_id: str):
        """Return the agent-specific llm, falling back to the default."""
        entry = llm_by_agent.get(agent_id)
        return entry[1] if entry else llm_default

    def llm_backend_key_for(session_or_agent_id: str) -> str | None:
        sid = str(session_or_agent_id or "").strip()
        if not sid:
            return "default"

        direct = llm_by_agent.get(sid)
        if direct:
            return str(direct[0] or "default")

        if "_t" in sid:
            direct = llm_by_agent.get(sid.split("_t", 1)[0])
            if direct:
                return str(direct[0] or "default")

        for candidate_id, (backend_key, _model) in llm_by_agent.items():
            if sid.startswith(f"{candidate_id}_") or sid.startswith(f"metareview_{candidate_id}_"):
                return str(backend_key or "default")

        return "default"

    # --- Log assignment summary (using the mix plan only) ---
    alloc_summary: dict[str, int] = {}
    for k, _ in llm_selections:
        alloc_summary[k] = alloc_summary.get(k, 0) + 1
    log.info(f"[llm-mix] allocation counts: {alloc_summary}")

    # --- Capacity admission gate + safe cleanup -----------------------------
    # Query server capacity and size the admission gate used inside research_one_step
    _cap_client = SandboxClient()
    _stats = _cap_client.get_server_stats(path="/base/stats") or {}
    _max_sessions = int(_stats.get("max_sessions") or 4)
    research_cfg_runtime = getattr(exp_cfg, "research", {}) or {}
    _admission, _reserve, _admission_override = _resolve_research_admission_limit(
        max_sessions=_max_sessions,
        network_num_agents=int(exp_cfg.network.num_agents),
        max_concurrent_research_agents=research_cfg_runtime.get("max_concurrent_research_agents"),
    )
    set_session_admission_limit(_admission)
    if _admission_override is not None:
        log.info(
            "[capacity] server max_sessions=%s reserve=%s computed_admission=%s "
            "research.max_concurrent_research_agents=%s final_admission=%s",
            _max_sessions,
            _reserve,
            max(1, _max_sessions - _reserve),
            _admission_override,
            _admission,
        )
    else:
        log.info(
            "[capacity] server max_sessions=%s reserve=%s admission gate set to %s",
            _max_sessions,
            _reserve,
            _admission,
        )
    sandbox_requirements_base: dict[str, Any] = {
        "limits": {
            "cpus": _stats.get("cpus"),
            "memory": _stats.get("memory"),
            "pids": _stats.get("pids"),
            "network_disabled": _stats.get("network_disabled"),
        },
        "max_sessions": _max_sessions,
        "admission_limit": _admission,
    }

    # Proactive safe reap for our owner tags (handles leftovers from crashed runs)
    # We’ll reap any idle sessions tagged with this iteration's owner
    # right before we launch that iteration (also done again inside research_one_step on finally).
    def _safe_reap_owner(owner: str, *, max_age_seconds: float = 300.0) -> None:
        try:
            _cap_client._reap_stale_sessions(
                keep_ids=set(),
                owner=owner,
                max_age_seconds=max_age_seconds,
                path="/base/sessions",
                shutdown_path="/base/shutdown",
            )
            s = _cap_client.get_server_stats(path="/base/stats") or {}
            log.info(
                f"[reaper] after reap(owner={owner}): active={s.get('active_sessions')} / max={s.get('max_sessions')}"
            )
        except Exception as _e:
            log.warning(f"[reaper] best-effort reap failed for owner={owner}: {_e}")

    research_cfg = getattr(exp_cfg, "research", {}) or {}
    session_create_hard_timeout_seconds = _resolve_session_create_hard_timeout_seconds(research_cfg)
    notepad_cfg = research_cfg.get("notepad", {}) or {}
    resuming = resume_enabled and resume_state is not None
    archive_init_settings, registry_init_settings = state.retrieval_settings(
        exp_cfg.get("memory"),
        saved_archive=resume_state.archive_settings if resuming else None,
        saved_registry=resume_state.registry_settings if resuming else None,
    )
    log.info(
        "[memory] archive ranking %s; registry ranking %s%s",
        archive_init_settings,
        registry_init_settings,
        " (restored from snapshot)" if resuming else "",
    )
    state_max_k = int(
        registry_init_settings.get("max_k", archive_init_settings.get("max_k", 5)) or 5
    )

    state.init_state(
        embedder=embedder,
        max_k=state_max_k,
        archive_with_citation_boost=bool(archive_init_settings.get("with_citation_boost", False)),
        archive_with_recency_boost=bool(archive_init_settings.get("with_recency_boost", False)),
        archive_w_citation=float(archive_init_settings.get("w_citation", 0.1)),
        archive_w_recency=float(archive_init_settings.get("w_recency", 0.1)),
        registry_with_citation_boost=bool(registry_init_settings.get("with_citation_boost", False)),
        registry_w_sim=float(registry_init_settings.get("w_sim", 0.8)),
        registry_w_cit=float(registry_init_settings.get("w_cit", 0.2)),
        notepad_enabled=bool(notepad_cfg.get("enabled", False)),
        notepad_max_entries_per_agent=int(notepad_cfg.get("max_entries_per_agent", 200) or 200),
        notepad_max_entry_chars=int(notepad_cfg.get("max_entry_chars", 2000) or 2000),
        notepad_base_snapshots_dir="snapshots",
    )
    agent_registry = state.registry()
    archive = state.archive()
    network_metadata = state.network_metadata()
    private_archive = state.private_archive()
    notepad_store = state.notepad_store()

    # Ensure runtime instances reflect restored ranking settings.
    agent_registry.with_citation_boost = bool(
        registry_init_settings.get("with_citation_boost", False)
    )
    agent_registry.w_sim = float(registry_init_settings.get("w_sim", 0.8))
    agent_registry.w_cit = float(registry_init_settings.get("w_cit", 0.2))
    agent_registry.max_k = int(registry_init_settings.get("max_k", 5))
    archive.with_citation_boost = bool(archive_init_settings.get("with_citation_boost", False))
    archive.with_recency_boost = bool(archive_init_settings.get("with_recency_boost", False))
    archive.w_citation = float(archive_init_settings.get("w_citation", 0.1))
    archive.w_recency = float(archive_init_settings.get("w_recency", 0.1))
    archive.max_k = int(archive_init_settings.get("max_k", 5))

    agent_registry.insert(init_agents)
    if resume_enabled and resume_state is not None:
        if resumed_public_papers:
            archive.add(resumed_public_papers)
            # Markovian pruning (experiment 08) of the restored archive happens at the start of the
            # first resumed iteration, exactly as in an uninterrupted run (same pruning
            # record in the snapshot).
        if resumed_private_papers:
            private_archive.add(resumed_private_papers)
        if resumed_reviews_by_paper:
            private_archive.set_reviews(resumed_reviews_by_paper)
        if resumed_review_assignments:
            private_archive.set_review_assignments(resumed_review_assignments)
        if resumed_notepad_entries:
            notepad_store.load_snapshot_data(
                {
                    "entries_by_agent": resumed_notepad_entries,
                }
            )
    elif bootstrap_enabled:
        if bootstrap_public_papers:
            archive.add(bootstrap_public_papers)
        if bootstrap_private_papers:
            private_archive.add(bootstrap_private_papers)
        if bootstrap_reviews_by_paper:
            private_archive.set_reviews(bootstrap_reviews_by_paper)
        if bootstrap_notepad_entries:
            notepad_store.load_snapshot_data(
                {
                    "entries_by_agent": bootstrap_notepad_entries,
                }
            )
    state.update_network_metadata()

    all_agent_profiles = agent_registry.get_all_agents()
    if resume_enabled and resume_state is not None:
        session_stats_registry = session_stats_registry_from_snapshot_payload(
            resume_state.session_stats_payload,
        )
    else:
        session_stats_registry = SessionStatsRegistry()

    # Replace your logging loop with this:
    for agent in all_agent_profiles:
        aid = getattr(agent, "agent_id", None) or agent.id
        backend_key = _assigned_backend_key(llm_by_agent, aid)
        log.info(f"[llm-assign] agent_id={aid} backend={backend_key}")

    if bootstrap_enabled and bool(bootstrap_cfg.get("write_snapshot", True)):
        create_snapshot(
            iteration=bootstrap_iteration,
            agent_registry=agent_registry,
            archive=archive,
            private_archive=private_archive,
            network_metadata=network_metadata,
            session_stats_registry=session_stats_registry,
            exp_config=exp_cfg,
            run_name=exp_cfg.run_name,
        )

    research_cfg = getattr(exp_cfg, "research", {}) or {}
    fixed_workflow_cfg = normalize_fixed_workflow_cfg(research_cfg.get("fixed_workflow", {}) or {})
    prior_cfg = research_cfg.get("prior_papers", {}) or {}
    prior_papers_keep_full = int(prior_cfg.get("keep_full", 1))
    prior_papers_summary_cfg_path = Path(
        prior_cfg.get("summary_cfg_path", "sys_config/llm_api/default.yaml")
    )
    prior_papers_enabled = bool(prior_cfg.get("enabled", False))
    prior_papers_mode = str(prior_cfg.get("mode", "full")).strip().lower()
    self_reflection_prompt_enabled = bool(
        research_cfg.get("self_reflection_prompt", {}).get("enabled", False)
    )
    nature_style_papers_prompt_enabled = bool(
        research_cfg.get("nature_style_papers_prompt", {}).get("enabled", False)
    )
    full_citing_requested = bool(research_cfg.get("full_citing", False))
    full_citing_compatible = prior_papers_enabled and prior_papers_mode == "full"
    full_citing_enabled = bool(full_citing_requested and full_citing_compatible)
    graph_logic_cfg = OmegaConf.to_container(
        getattr(exp_cfg, "graph_logic", {}) or {},
        resolve=True,
    )
    if not isinstance(graph_logic_cfg, dict):
        graph_logic_cfg = {}
    if full_citing_requested and not full_citing_compatible:
        log.warning("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        log.warning(
            "[research.full_citing] requested but incompatible: "
            "requires research.prior_papers.enabled=true and research.prior_papers.mode='full'."
        )
        log.warning(
            f"[research.full_citing] current values: "
            f"prior_papers.enabled={prior_papers_enabled}, prior_papers.mode='{prior_papers_mode}'. "
            "Disabling full_citing for this run."
        )
        log.warning("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    meta_researcher_cfg = normalize_meta_researcher_cfg(
        research_cfg.get("meta_researcher", {}) or {}
    )
    meta_researcher_enabled = bool(meta_researcher_cfg.get("enabled", False))
    if int(meta_researcher_cfg.get("output", {}).get("papers_per_agent", 1) or 1) != 1:
        log.warning(
            "[meta-researcher] output.papers_per_agent != 1 is not yet supported; "
            "using one paper per selected agent."
        )

    target_iteration = int(exp_cfg.network.iterations)
    if start_iteration > target_iteration:
        if resume_enabled and resume_state is not None:
            log.info(
                "[resume] nothing to run: restored_iteration=%s target_iteration=%s",
                resume_state.restored_iteration,
                target_iteration,
            )
        else:
            log.info(
                "[main] nothing to run: start_iteration=%s target_iteration=%s",
                start_iteration,
                target_iteration,
            )
        return

    ##############################
    # Main iterative loop
    ##############################
    for iters in range(start_iteration, target_iteration + 1):
        # proactively reap idle leftovers for this iteration's owner tag
        _safe_reap_owner(owner=f"research_t{iters}", max_age_seconds=300.0)
        paper_metrics_by_id = None
        reviews_by_paper = {}
        meta_reviews = []
        metareview_logs = []
        social_memory_pruning_metadata = None

        # Markovian social memory (experiment 08): prune the PUBLIC archive at the iteration
        # boundary, before the research stage, keeping only papers published at t-1
        # (owner-agnostic). At t=1 there is no t-1, so the public archive is empty.
        # Everything that reads the public archive sees the pruned version: agents'
        # archive search, the reviewer novelty metric, and the public archive_data
        # snapshot. The private archive (the ranker's data source) is not pruned.
        if social_memory_mode == "markovian":
            retained_publication_t = iters - 1
            dropped_ids = archive.prune_older_than(retained_publication_t)
            social_memory_pruning_metadata = {
                "mode": social_memory_mode,
                "iteration": iters,
                "retained_publication_t": retained_publication_t,
                "retained_count": len(getattr(archive, "_entries", {}) or {}),
                "dropped_paper_ids": dropped_ids,
                "dropped_count": len(dropped_ids),
            }
            log.info(
                "[social_memory][markovian] iteration=%s kept publication_t==%s "
                "retained=%s dropped=%s",
                iters,
                retained_publication_t,
                social_memory_pruning_metadata["retained_count"],
                len(dropped_ids),
            )

        # read from config (default True if missing)
        code_collaboration = exp_cfg.research.get("code_collaboration", True)
        pause_cfg = getattr(exp_cfg, "reputation_pause", None) or {}
        pause_full, pause_review, pause_collab = _paused_sets(
            all_agent_profiles,
            iteration=iters,
            pause_cfg=pause_cfg,
        )
        log.info(
            f"[pause-sets] iteration={iters} pause_full={len(pause_full)} "
            f"pause_review={len(pause_review)} pause_collab={len(pause_collab)} "
            f"active_agents={max(0, len(all_agent_profiles) - len(pause_full))}"
        )
        per_agent_research_max_iters = _research_iter_overrides(
            all_agent_profiles,
            base_iters=exp_cfg.research.research_max_iters,
            pause_cfg=pause_cfg,
        )

        metacognition_feedback_by_agent = None
        metacognition_feedback_metadata = None
        fixed_workflow_by_agent = None
        fixed_workflow_metadata = None
        fixed_workflow_assignments = {}
        if bool(fixed_workflow_cfg.get("enabled", False)):
            fixed_workflow_assignments, fixed_workflow_primary_target = (
                resolve_fixed_workflow_targets(
                    all_agent_profiles,
                    iteration=iters,
                    seed=seed,
                    paused_agent_ids=pause_full,
                    cfg=fixed_workflow_cfg,
                )
            )
            fixed_workflow_by_agent = make_fixed_workflow_context_by_agent(
                fixed_workflow_assignments,
                cfg=fixed_workflow_cfg,
                iteration=iters,
            )
            fixed_workflow_metadata = make_fixed_workflow_metadata(
                iteration=iters,
                seed=seed,
                cfg=fixed_workflow_cfg,
                collaborator_map=fixed_workflow_assignments,
                agents=all_agent_profiles,
                primary_target_id=fixed_workflow_primary_target,
            )
            log.info(
                "[fixed-workflow] iteration=%s assignments=%s",
                iters,
                len(fixed_workflow_assignments),
            )
        metacog_raw_cfg = research_cfg.get("metacognition_feedback", {}) or {}
        metacog_cfg = normalize_metacognition_feedback_cfg(metacog_raw_cfg)
        is_first_resumed_iteration = bool(
            resume_state is not None and int(iters) == int(resume_state.restored_iteration) + 1
        )
        metacog_resume_allowed = not is_first_resumed_iteration or bool(
            metacog_cfg.get("apply_on_resume_first_iteration", True)
        )
        if (
            bool(metacog_cfg.get("enabled", False))
            and metacog_cfg.get("mode") != "none"
            and metacog_resume_allowed
        ):
            metacog_active_agents = [
                agent
                for agent in all_agent_profiles
                if str(getattr(agent, "agent_id", None) or getattr(agent, "id", "") or "")
                not in pause_full
            ]
            source_iteration = resolve_feedback_source_iteration(
                iteration=iters,
                restored_iteration=(
                    resume_state.restored_iteration if resume_state is not None else None
                ),
                cfg=metacog_cfg,
            )
            source_snapshot_dir = None
            if (
                resume_state is not None
                and source_iteration is not None
                and int(source_iteration) == int(resume_state.restored_iteration)
            ):
                source_snapshot_dir = resume_state.iteration_dir
            elif source_iteration is not None:
                candidate_source_dir = resolve_iteration_snapshot_dir(
                    run_name=exp_cfg.run_name,
                    iteration=int(source_iteration),
                    project_dir=project_root,
                    create=False,
                )
                if candidate_source_dir.exists():
                    source_snapshot_dir = candidate_source_dir

            metacog_skipped: dict[str, dict[str, Any]] = {}
            if metacog_active_agents:
                feedback_llm = llm_default
                backend_key = str(metacog_cfg.get("llm_backend") or "").strip()
                if backend_key and metacog_cfg.get("mode") in {
                    "meta_review_only",
                    "meta_review_with_metrics",
                }:
                    feedback_llm = llm_pool.get(backend_key)
                    if feedback_llm is None:
                        raise ValueError(
                            f"research.metacognition_feedback.llm_backend={backend_key!r} "
                            f"not found in LLM pool keys={sorted(llm_pool.keys())}"
                        )

                metacognition_feedback_by_agent = build_metacognition_feedback_for_agents(
                    agents=metacog_active_agents,
                    private_archive=private_archive,
                    iteration=iters,
                    restored_iteration=(
                        resume_state.restored_iteration if resume_state is not None else None
                    ),
                    cfg=metacog_cfg,
                    source_snapshot_dir=source_snapshot_dir,
                    llm=feedback_llm,
                    skipped_agents=metacog_skipped,
                )
                for agent in metacog_active_agents:
                    agent_id = getattr(agent, "agent_id", None) or getattr(agent, "id", None)
                    if not agent_id:
                        continue
                    # A skipped agent's skip record replaces the previous round's feedback.
                    payload = (metacognition_feedback_by_agent or {}).get(
                        str(agent_id)
                    ) or metacog_skipped.get(str(agent_id))
                    if not isinstance(payload, Mapping):
                        continue
                    new_learning_state = merge_metacognition_feedback_learning_state(agent, payload)
                    agent.learning_state = new_learning_state
                    try:
                        agent_registry.update(str(agent_id), new_learning_state=new_learning_state)
                    except Exception as e:
                        log.warning(
                            f"[metacognition] failed to persist learning_state for agent_id={agent_id}: {e}"
                        )
            else:
                metacognition_feedback_by_agent = {}
            metacognition_feedback_metadata = make_metacognition_feedback_metadata(
                iteration=iters,
                source_iteration=source_iteration,
                mode=str(metacog_cfg.get("mode", "none")),
                feedback_by_agent=metacognition_feedback_by_agent or {},
                write_artifacts=bool(metacog_cfg.get("write_artifacts", True)),
                skipped_agents=metacog_skipped,
            )
            log.info(
                "[metacognition] iteration=%s mode=%s source_iteration=%s agents_with_feedback=%s",
                iters,
                metacog_cfg.get("mode"),
                source_iteration,
                len(metacognition_feedback_by_agent or {}),
            )

        candidate_papers, research_logs = safe_asyncio_run(
            research_one_step(
                iteration=iters,
                llm=llm_default,
                llm_for=llm_for,
                llm_backend_key_for=llm_backend_key_for,
                default_llm_runtime_metadata=default_llm_runtime_metadata,
                sandbox_requirements=sandbox_requirements_base,
                agents=all_agent_profiles,
                network_metadata=network_metadata,
                private_archive=private_archive,
                task_config=task_config,
                research_max_iters=exp_cfg.research.research_max_iters,
                network_setting=network_setting,
                code_timeout_seconds=exp_cfg.research.code_timeout_seconds,
                total_session_timeout_seconds=exp_cfg.research.total_session_timeout_seconds,
                run_name=exp_cfg.run_name,
                code_collaboration=code_collaboration,
                session_create_hard_timeout_seconds=session_create_hard_timeout_seconds,
                execute_code_retries=exp_cfg.research.get("execute_code_retries", 120),
                execute_code_retry_delay=exp_cfg.research.get(
                    "execute_code_retry_delay_seconds", 5
                ),
                prior_papers_keep_full=prior_papers_keep_full,
                prior_papers_summary_cfg_path=prior_papers_summary_cfg_path,
                prior_papers_enabled=prior_papers_enabled,
                prior_papers_mode=prior_papers_mode,
                self_reflection_prompt_enabled=self_reflection_prompt_enabled,
                nature_style_papers_prompt_enabled=nature_style_papers_prompt_enabled,
                full_citing_enabled=full_citing_enabled,
                graph_logic_cfg=graph_logic_cfg,
                per_agent_research_max_iters=per_agent_research_max_iters,
                paused_agent_ids=pause_full,
                disable_collab_agent_ids=pause_collab,
                metacognition_feedback_by_agent=metacognition_feedback_by_agent,
                fixed_workflow_by_agent=fixed_workflow_by_agent,
            )
        )

        session_stats_registry.add_log(research_logs)
        if bool(fixed_workflow_cfg.get("enabled", False)) and fixed_workflow_metadata is not None:
            compliance_rows = []
            compliant_papers = []
            for paper in candidate_papers or []:
                paper_id = str(getattr(paper, "paper_id", "") or "").strip()
                agent_id = str(getattr(paper, "primary_agent_id", "") or "").strip()
                assigned = str(fixed_workflow_assignments.get(agent_id, "") or "").strip()
                session_log = session_stats_registry.get_research_log_for_agent(iters, agent_id)
                fail_reasons = validate_fixed_workflow_compliance(
                    paper,
                    session_log,
                    assigned,
                    fixed_workflow_cfg,
                )
                passed = not fail_reasons
                compliance_rows.append(
                    {
                        "iteration": int(iters),
                        "paper_id": paper_id,
                        "primary_agent_id": agent_id,
                        "assigned_collaborator_id": assigned,
                        "passed": bool(passed),
                        "fail_reasons": list(fail_reasons),
                    }
                )
                if passed:
                    compliant_papers.append(paper)
                    continue
                paper.status = "fixed_workflow_noncompliant"
                paper.research_validation_status = "fixed_workflow_noncompliant"
                log.warning(
                    "[fixed-workflow][compliance] excluding paper_id=%s agent_id=%s reasons=%s",
                    paper_id,
                    agent_id,
                    fail_reasons,
                )
            fixed_workflow_metadata["compliance"] = compliance_rows
            fixed_workflow_metadata["summary"] = {
                "passed": sum(1 for row in compliance_rows if bool(row.get("passed", False))),
                "failed": sum(1 for row in compliance_rows if not bool(row.get("passed", False))),
                "total": len(compliance_rows),
            }
            candidate_papers = compliant_papers
        meta_researcher_metadata = None
        if meta_researcher_enabled:
            trigger = evaluate_meta_research_trigger(
                iteration=iters,
                cfg=meta_researcher_cfg,
            )
            meta_researcher_metadata = init_meta_researcher_telemetry(
                iteration=iters,
                cfg=meta_researcher_cfg,
                trigger=trigger,
            )
            telemetry_cfg = dict(meta_researcher_cfg.get("telemetry", {}) or {})
            meta_researcher_metadata["write_iteration_file"] = bool(
                telemetry_cfg.get("write_iteration_file", True)
            )
            meta_researcher_metadata["write_meta_quality_report"] = bool(
                telemetry_cfg.get("write_meta_quality_report", True)
            )
            meta_researcher_metadata["write_evidence_map"] = bool(
                telemetry_cfg.get("write_evidence_map", True)
            )
            meta_researcher_metadata["write_meta_claim_checks"] = bool(
                telemetry_cfg.get("write_meta_claim_checks", True)
            )
            meta_researcher_metadata["write_citation_lint"] = bool(
                telemetry_cfg.get("write_citation_lint", True)
            )
            meta_researcher_metadata["write_ranker_pools"] = bool(
                telemetry_cfg.get("write_ranker_pools", True)
            )

            if not bool(trigger.get("triggered", False)):
                meta_researcher_metadata["skipped_reason"] = str(
                    trigger.get("reason", "not_scheduled")
                )
            else:
                source_cards = build_meta_source_cards(
                    public_archive=archive,
                    cfg=meta_researcher_cfg,
                )
                source_ids = [
                    str(card.get("paper_id", "")).strip()
                    for card in source_cards
                    if str(card.get("paper_id", "")).strip()
                ]
                meta_researcher_metadata["selected_source_paper_ids"] = source_ids
                meta_researcher_metadata["num_source_papers"] = len(source_ids)

                min_source_papers = int(
                    meta_researcher_cfg.get("min_public_archive_papers", 0) or 0
                )
                if len(source_ids) < min_source_papers:
                    meta_researcher_metadata["skipped_reason"] = (
                        f"insufficient_public_archive_papers:{len(source_ids)}<{min_source_papers}"
                    )
                else:
                    active_agent_ids = [
                        str(getattr(agent, "agent_id", None) or getattr(agent, "id", "")).strip()
                        for agent in all_agent_profiles
                        if str(getattr(agent, "agent_id", None) or getattr(agent, "id", "")).strip()
                        and str(
                            getattr(agent, "agent_id", None) or getattr(agent, "id", "")
                        ).strip()
                        not in pause_full
                    ]
                    selected_meta_agent_ids = select_meta_research_agent_ids(
                        agents=all_agent_profiles,
                        active_agent_ids=active_agent_ids,
                        cfg=meta_researcher_cfg,
                    )
                    meta_researcher_metadata["selected_agent_ids"] = list(selected_meta_agent_ids)

                    if not selected_meta_agent_ids:
                        meta_researcher_metadata["skipped_reason"] = "no_agents_selected"
                    else:
                        output_cfg = dict(meta_researcher_cfg.get("output", {}) or {})
                        runtime_cfg = dict(meta_researcher_cfg.get("runtime", {}) or {})
                        phase_context_by_agent = build_meta_phase_context_by_agent(
                            agent_ids=selected_meta_agent_ids,
                            source_cards=source_cards,
                            iteration=iters,
                            cfg=meta_researcher_cfg,
                        )
                        selected_meta_set = set(selected_meta_agent_ids)
                        selected_meta_agents = [
                            agent
                            for agent in all_agent_profiles
                            if str(
                                getattr(agent, "agent_id", None) or getattr(agent, "id", "")
                            ).strip()
                            in selected_meta_set
                        ]
                        meta_paper_suffix = (
                            str(output_cfg.get("paper_id_suffix", "meta") or "meta").strip()
                            or "meta"
                        )
                        meta_total_timeout = runtime_cfg.get("total_session_timeout_seconds")
                        if meta_total_timeout is None:
                            meta_total_timeout = exp_cfg.research.total_session_timeout_seconds
                        meta_disable_collab = bool(runtime_cfg.get("disable_collab_tools", False))
                        meta_disable_collab_agent_ids = (
                            set(selected_meta_agent_ids) if meta_disable_collab else set()
                        )
                        meta_disable_collab_agent_ids = meta_disable_collab_agent_ids.union(
                            pause_collab
                        )

                        _safe_reap_owner(
                            owner=f"research_t{iters}_{meta_paper_suffix}",
                            max_age_seconds=300.0,
                        )
                        meta_candidates, meta_research_logs = safe_asyncio_run(
                            research_one_step(
                                iteration=iters,
                                llm=llm_default,
                                llm_for=llm_for,
                                llm_backend_key_for=llm_backend_key_for,
                                default_llm_runtime_metadata=default_llm_runtime_metadata,
                                sandbox_requirements=sandbox_requirements_base,
                                agents=selected_meta_agents,
                                network_metadata=network_metadata,
                                private_archive=private_archive,
                                task_config=task_config,
                                research_max_iters=exp_cfg.research.research_max_iters,
                                network_setting=network_setting,
                                code_timeout_seconds=exp_cfg.research.code_timeout_seconds,
                                total_session_timeout_seconds=meta_total_timeout,
                                run_name=exp_cfg.run_name,
                                code_collaboration=bool(
                                    runtime_cfg.get("code_collaboration", True)
                                ),
                                session_create_hard_timeout_seconds=session_create_hard_timeout_seconds,
                                execute_code_retries=exp_cfg.research.get(
                                    "execute_code_retries", 120
                                ),
                                execute_code_retry_delay=exp_cfg.research.get(
                                    "execute_code_retry_delay_seconds", 5
                                ),
                                prior_papers_keep_full=prior_papers_keep_full,
                                prior_papers_summary_cfg_path=prior_papers_summary_cfg_path,
                                prior_papers_enabled=prior_papers_enabled,
                                prior_papers_mode=prior_papers_mode,
                                self_reflection_prompt_enabled=self_reflection_prompt_enabled,
                                nature_style_papers_prompt_enabled=nature_style_papers_prompt_enabled,
                                full_citing_enabled=full_citing_enabled,
                                graph_logic_cfg=graph_logic_cfg,
                                per_agent_research_max_iters=per_agent_research_max_iters,
                                paused_agent_ids=set(),
                                disable_collab_agent_ids=meta_disable_collab_agent_ids,
                                research_phase="meta_research",
                                paper_id_suffix=meta_paper_suffix,
                                session_id_suffix=meta_paper_suffix,
                                paper_kind="meta_research",
                                phase_context_by_agent=phase_context_by_agent,
                            )
                        )
                        session_stats_registry.add_log(meta_research_logs)
                        telemetry_cfg = dict(meta_researcher_cfg.get("telemetry", {}) or {})
                        write_meta_quality_report = bool(
                            telemetry_cfg.get("write_meta_quality_report", True)
                        )
                        write_evidence_map = bool(telemetry_cfg.get("write_evidence_map", True))
                        write_meta_claim_checks = bool(
                            telemetry_cfg.get("write_meta_claim_checks", True)
                        )
                        write_citation_lint = bool(telemetry_cfg.get("write_citation_lint", True))
                        if write_evidence_map and phase_context_by_agent:
                            first_ctx = next(iter(phase_context_by_agent.values()), {})
                            if isinstance(first_ctx, dict):
                                shared_meta_ctx = first_ctx.get("meta_research_context", {}) or {}
                                if isinstance(shared_meta_ctx, dict):
                                    evidence_graph = shared_meta_ctx.get("evidence_graph")
                                    if isinstance(evidence_graph, dict):
                                        meta_researcher_metadata["evidence_graph"] = dict(
                                            evidence_graph
                                        )
                                        meta_researcher_metadata["evidence_graph_summary"] = dict(
                                            evidence_graph.get("summary", {}) or {}
                                        )

                        promoted_meta_candidates = []
                        meta_quality_reports = []
                        meta_claim_checks_artifacts = []
                        citation_lint_artifacts = []
                        meta_quality_summary = {"passed": 0, "failed": 0, "revised_passed": 0}
                        for paper in meta_candidates or []:
                            validation_status = str(
                                getattr(paper, "research_validation_status", "")
                                or getattr(paper, "status", "")
                            ).strip()
                            gate_status = str(
                                getattr(paper, "meta_quality_gate_status", "") or "not_applicable"
                            ).strip()
                            revision_attempts = int(
                                getattr(paper, "meta_revision_attempts", 0) or 0
                            )
                            fail_reasons = [
                                str(x).strip()
                                for x in (getattr(paper, "meta_quality_fail_reasons", []) or [])
                                if str(x).strip()
                            ]
                            pass_status = validation_status == "under_review" and gate_status in {
                                "passed",
                                "disabled",
                                "not_applicable",
                            }
                            if pass_status:
                                promoted_meta_candidates.append(paper)
                                meta_quality_summary["passed"] += 1
                                if revision_attempts > 0:
                                    meta_quality_summary["revised_passed"] += 1
                            else:
                                meta_quality_summary["failed"] += 1

                            if write_meta_quality_report:
                                meta_quality_reports.append(
                                    {
                                        "paper_id": str(getattr(paper, "paper_id", "") or ""),
                                        "agent_id": str(
                                            getattr(paper, "primary_agent_id", "") or ""
                                        ),
                                        "research_validation_status": validation_status,
                                        "meta_quality_gate_status": gate_status,
                                        "meta_quality_fail_reasons": fail_reasons,
                                        "meta_revision_attempts": revision_attempts,
                                        "meta_supported_claim_count": int(
                                            getattr(paper, "meta_supported_claim_count", 0) or 0
                                        ),
                                        "meta_unsupported_claim_count": int(
                                            getattr(paper, "meta_unsupported_claim_count", 0) or 0
                                        ),
                                        "meta_unresolved_uncertainty_claim_count": int(
                                            getattr(
                                                paper, "meta_unresolved_uncertainty_claim_count", 0
                                            )
                                            or 0
                                        ),
                                    }
                                )
                            if write_meta_claim_checks:
                                meta_claim_checks = getattr(paper, "meta_claim_checks", None)
                                if isinstance(meta_claim_checks, Mapping):
                                    meta_claim_checks_artifacts.append(
                                        {
                                            "paper_id": str(getattr(paper, "paper_id", "") or ""),
                                            "agent_id": str(
                                                getattr(paper, "primary_agent_id", "") or ""
                                            ),
                                            **dict(meta_claim_checks),
                                        }
                                    )
                            if write_citation_lint:
                                citation_lint = getattr(paper, "citation_lint", None)
                                if isinstance(citation_lint, Mapping):
                                    citation_lint_artifacts.append(
                                        {
                                            "paper_id": str(getattr(paper, "paper_id", "") or ""),
                                            "agent_id": str(
                                                getattr(paper, "primary_agent_id", "") or ""
                                            ),
                                            **dict(citation_lint),
                                        }
                                    )

                        if promoted_meta_candidates:
                            candidate_papers = list(candidate_papers or []) + list(
                                promoted_meta_candidates
                            )

                        generated_meta_paper_ids = [
                            str(getattr(paper, "paper_id", "")).strip()
                            for paper in (meta_candidates or [])
                            if str(getattr(paper, "paper_id", "")).strip()
                        ]
                        promoted_meta_paper_ids = [
                            str(getattr(paper, "paper_id", "")).strip()
                            for paper in (promoted_meta_candidates or [])
                            if str(getattr(paper, "paper_id", "")).strip()
                        ]
                        meta_researcher_metadata["generated_meta_paper_ids"] = (
                            generated_meta_paper_ids
                        )
                        meta_researcher_metadata["promoted_meta_paper_ids"] = (
                            promoted_meta_paper_ids
                        )
                        meta_researcher_metadata["num_meta_papers_generated"] = len(
                            generated_meta_paper_ids
                        )
                        meta_researcher_metadata["num_meta_papers"] = len(promoted_meta_paper_ids)
                        meta_researcher_metadata["num_meta_papers_filtered_out"] = max(
                            0, len(generated_meta_paper_ids) - len(promoted_meta_paper_ids)
                        )
                        if write_meta_quality_report:
                            meta_researcher_metadata["meta_quality_reports"] = meta_quality_reports
                        if write_meta_claim_checks:
                            meta_researcher_metadata["meta_claim_checks_artifacts"] = (
                                meta_claim_checks_artifacts
                            )
                        if write_citation_lint:
                            meta_researcher_metadata["citation_lint_artifacts"] = (
                                citation_lint_artifacts
                            )
                        meta_researcher_metadata["meta_quality_summary"] = meta_quality_summary
                        meta_researcher_metadata["agent_failures"] = {
                            str(getattr(paper, "primary_agent_id", "")).strip(): str(
                                getattr(paper, "research_validation_status", "")
                                or getattr(paper, "status", "")
                            )
                            for paper in (meta_candidates or [])
                            if str(getattr(paper, "primary_agent_id", "")).strip()
                            and str(
                                getattr(paper, "research_validation_status", "")
                                or getattr(paper, "status", "")
                            )
                            != "under_review"
                        }

        if candidate_papers:
            metrics_pipeline_raw = exp_cfg.review.get("metrics_pipeline", "default")
            metrics_pipeline, metrics_pipeline_fallback = _normalize_metrics_pipeline_mode(
                metrics_pipeline_raw
            )
            if metrics_pipeline_fallback:
                log.warning(
                    "[review][metrics] unknown metrics_pipeline=%r; falling back to 'default'. "
                    "Valid values: default | experimental | none",
                    metrics_pipeline_raw,
                )
            metrics_allowlist = exp_cfg.review.get("metrics_allowlist", None)
            if metrics_allowlist is not None and not isinstance(metrics_allowlist, list):
                metrics_allowlist = list(metrics_allowlist)

            if metrics_pipeline == "experimental":
                paper_metrics_by_id = build_review_metrics_for_papers(
                    iteration=iters,
                    candidate_papers=candidate_papers,
                    session_stats_registry=session_stats_registry,
                    archive=archive,  # or None if you don’t want semantic distances yet
                    metrics_allowlist=metrics_allowlist,
                )
            elif metrics_pipeline == "none":
                paper_metrics_by_id = None
            else:
                paper_metrics_by_id = build_review_metrics_for_papers(
                    iteration=iters,
                    candidate_papers=candidate_papers,
                    session_stats_registry=session_stats_registry,
                    archive=archive,  # or None if you don’t want semantic distances yet
                )
            if paper_metrics_by_id is None:
                log.info("[review][metrics] disabled (metrics_pipeline=none)")
            else:
                log.debug(f"[review][metrics] {paper_metrics_by_id}")
                # Metrics only: reputation is logged once, after the round's update.
                _log_signal_diagnostics(
                    iteration=iters,
                    paper_metrics_by_id=paper_metrics_by_id,
                )
            for paper in candidate_papers:
                paper_id = str(getattr(paper, "paper_id", "") or "").strip()
                metrics_for_paper = (
                    paper_metrics_by_id.get(paper_id, {})
                    if paper_metrics_by_id is not None and paper_id
                    else {}
                )
                if isinstance(metrics_for_paper, Mapping):
                    paper.review_metrics = dict(metrics_for_paper)
                else:
                    paper.review_metrics = {}
            if reviewerless_enabled:
                # Reviewerless ablation (experiment 07): remove reviewers entirely: no reviewer sessions, no reviewer
                # LLM cost, no review artefacts. The meta-reviewer scores submissions
                # directly. reviews_by_paper stays empty.
                log.info(
                    "[reviewerless] skipping review stage (metareview.reviewerless.enabled=true)"
                )
                reviews_by_paper = {}
            else:
                reviews_by_paper, review_logs = safe_asyncio_run(
                    review_one_step(
                        iteration=iters,
                        llm=llm_default,
                        llm_for=llm_for,
                        llm_backend_key_for=llm_backend_key_for,
                        default_llm_runtime_metadata=default_llm_runtime_metadata,
                        sandbox_requirements=sandbox_requirements_base,
                        candidate_papers=candidate_papers,
                        agent_registry=agent_registry,
                        agent_registry_search_tool=agent_registry_search_tool,
                        private_archive=private_archive,
                        network_metadata=network_metadata,
                        task_config=task_config,
                        num_reviewers_per_paper=exp_cfg.review.num_reviewers_per_paper,
                        review_max_iterations=exp_cfg.review.review_max_iterations,
                        include_code_in_reviews=exp_cfg.review.include_code_in_reviews,
                        total_session_timeout_seconds=exp_cfg.review.total_session_timeout_seconds,
                        run_name=exp_cfg.run_name,
                        paper_metrics_by_id=paper_metrics_by_id,
                        metrics_allowlist=_reviewer_metrics_allowlist(
                            metrics_pipeline, metrics_allowlist
                        ),
                        # Metrics off: no METRICS block, so no guidance to use them.
                        include_metrics_guidance=paper_metrics_by_id is not None,
                        allow_self_review=exp_cfg.review.get("allow_self_review", False),
                        excluded_reviewer_ids=pause_review,
                        require_assertion_justification=exp_cfg.review.get(
                            "require_assertion_justification", False
                        ),
                        graph_logic_cfg=graph_logic_cfg,
                        reviewer_norm_text=reviewer_norm_text,
                        matching_strategy=exp_cfg.review.get("matching_strategy", "registry"),
                        # Deprecated alias. Forwarding `seed` unconditionally here made
                        # review.reviewer_assignment_seed unreachable.
                        matching_seed=exp_cfg.review.get("matching_seed", None),
                        random_matching_seed_offset=exp_cfg.review.get(
                            "random_matching_seed_offset", 0
                        ),
                        reviewer_assignment_strategy=exp_cfg.review.get(
                            "reviewer_assignment_strategy", "expertise_matching"
                        ),
                        reviewer_assignment_seed=exp_cfg.review.get(
                            "reviewer_assignment_seed", exp_cfg.get("SEED", 0)
                        ),
                        reviewer_assignment_condition_label=exp_cfg.review.get(
                            "reviewer_assignment_condition_label", None
                        ),
                    )
                )
                session_stats_registry.add_log(review_logs)

        # Reviewerless mode (experiment 07) has zero reviews by design and still runs meta-review,
        # so the zero-review warning only applies when reviewers are enabled.
        if candidate_papers and not reviewerless_enabled:
            unreviewed = [
                str(getattr(p, "paper_id", "") or "")
                for p in candidate_papers
                if not reviews_by_paper.get(getattr(p, "paper_id", ""))
            ]
            if unreviewed:
                log.warning(
                    "[review] %d/%d candidate papers received ZERO reviews and will skip "
                    "meta-review: %s",
                    len(unreviewed),
                    len(candidate_papers),
                    unreviewed[:10],
                )

        if candidate_papers and (reviews_by_paper or reviewerless_enabled):
            meta_reviews, metareview_logs = safe_asyncio_run(
                metareview_one_step(
                    iteration=iters,
                    llm=llm_default,
                    llm_for=llm_for,
                    llm_backend_key_for=llm_backend_key_for,
                    default_llm_runtime_metadata=default_llm_runtime_metadata,
                    sandbox_requirements=sandbox_requirements_base,
                    candidate_papers=candidate_papers,
                    reviews_by_paper=reviews_by_paper,
                    tournament_size=exp_cfg.metareview.tournament_size,
                    shuffle_before_tournament=exp_cfg.metareview.shuffle_before_tournament,
                    total_session_timeout_seconds=exp_cfg.metareview.get(
                        "total_session_timeout_seconds", None
                    ),
                    run_name=exp_cfg.run_name,
                    require_reviews=not reviewerless_enabled,
                    review_free=reviewerless_enabled,
                )
            )
            session_stats_registry.add_log(metareview_logs)

        if candidate_papers:
            update_summary = update_status_and_archives(
                meta_reviews=meta_reviews,
                candidate_papers=candidate_papers,
                archive=archive,
                private_archive=private_archive,
                agent_registry=agent_registry,
                reputation_cfg=getattr(exp_cfg, "reputation", None),
                pause_cfg=getattr(exp_cfg, "reputation_pause", None),
                iteration=iters,
            )

            # Optional replacement flow for newly paused agents.
            pause_mode = str(pause_cfg.get("mode", "none") or "none").strip().lower()
            replacement_cfg_raw = (
                pause_cfg.get("replacement", {}) if isinstance(pause_cfg, Mapping) else {}
            )
            replacement_cfg = (
                dict(replacement_cfg_raw) if isinstance(replacement_cfg_raw, Mapping) else {}
            )
            replacement_enabled = bool(replacement_cfg.get("enabled", False))
            permanent_modes = {
                "full_pause_permanent",
                "permanent_full_pause",
                "full_pause_permanent_replace",
                "full_pause_permanent_scheduled",
            }
            newly_paused = list(update_summary.get("newly_paused_agents", []) or [])
            if replacement_enabled and pause_mode in permanent_modes and newly_paused:
                replacement_network_setting = str(
                    replacement_cfg.get("network_setting", network_setting)
                )
                replacement_prefix = str(replacement_cfg.get("agent_id_prefix", "agent"))
                replacement_count = len(newly_paused)

                replacement_profiles = init_agents_profiles(
                    llm=llm_default,
                    dataset_metadata=task_config.dataset_description,
                    n=replacement_count,
                    network_setting=replacement_network_setting,
                    agent_id_prefix=replacement_prefix,
                )

                existing_ids = [
                    (getattr(a, "agent_id", None) or getattr(a, "id", ""))
                    for a in agent_registry.get_all_agents()
                ]
                new_ids = _next_agent_ids(
                    existing_ids, prefix=replacement_prefix, count=replacement_count
                )
                for prof, new_id in zip(replacement_profiles, new_ids, strict=False):
                    prof.agent_id = new_id
                    agent_weights = dict(default_weights_norm)
                    agent_weights.update(
                        _normalize_weights(metric_weights_by_agent.get(prof.agent_id))
                    )
                    prof.metric_weights = agent_weights or None

                paused_sorted = sorted(
                    newly_paused,
                    key=lambda row: (
                        float(row.get("reputation_score"))
                        if isinstance(row.get("reputation_score"), (int, float))
                        else float("inf"),
                        str(row.get("agent_id", "")),
                    ),
                )
                # Replacement idx takes over paused_sorted[idx]'s backend (logged below).
                replacement_llm_selections = _replacement_llm_selections(
                    paused_sorted, llm_by_agent, llm_default
                )
                replacement_llm_by_agent = _assign_llms_to_agents(
                    replacement_profiles, replacement_llm_selections
                )
                llm_by_agent.update(replacement_llm_by_agent)

                agent_registry.insert(replacement_profiles)
                state.update_network_metadata()

                for idx, prof in enumerate(replacement_profiles):
                    old = paused_sorted[idx] if idx < len(paused_sorted) else {}
                    old_id = str(old.get("agent_id", "unknown"))
                    old_score = old.get("reputation_score")
                    backend_key, _ = replacement_llm_selections[idx]
                    log.info(
                        f"[agent-replace] iteration={iters} old_agent_id={old_id} "
                        f"old_reputation_score={old_score} new_agent_id={prof.agent_id} "
                        f"backend={backend_key}"
                    )
                log.info(
                    f"[agent-replace-summary] iteration={iters} replacements_requested={replacement_count} "
                    f"replacements_created={len(replacement_profiles)} total_agents={len(agent_registry.get_all_agents())}"
                )

            # Refresh in-memory agent profiles so future iterations see updated stats.
            all_agent_profiles = agent_registry.get_all_agents()
            _log_signal_diagnostics(
                iteration=iters,
                paper_metrics_by_id=paper_metrics_by_id
                if "paper_metrics_by_id" in locals()
                else None,
                agents=all_agent_profiles,
                include_metrics=False,
            )

        # ✅ Save a snapshot of this iteration to disk
        metrics_for_snapshot = paper_metrics_by_id if candidate_papers else None
        create_snapshot(
            iteration=iters,
            agent_registry=agent_registry,
            archive=archive,
            private_archive=private_archive,
            network_metadata=network_metadata,
            session_stats_registry=session_stats_registry,
            exp_config=exp_cfg,
            run_name=exp_cfg.run_name,
            metrics_by_paper=metrics_for_snapshot,
            paper_metadata=candidate_papers if candidate_papers else None,
            meta_researcher_metadata=meta_researcher_metadata,
            metacognition_feedback_metadata=metacognition_feedback_metadata,
            fixed_workflow_metadata=fixed_workflow_metadata,
            social_memory_pruning_metadata=social_memory_pruning_metadata,
        )


if __name__ == "__main__":
    main()
