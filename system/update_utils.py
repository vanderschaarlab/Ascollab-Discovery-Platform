import logging
import math
from collections.abc import Iterable, Mapping
from typing import Any

from stores.agent_registry import AgentRegistry
from stores.paper_archive import PaperArchive
from stores.private_paper_archive import PrivatePaperArchive
from system.schema import PaperMetadataPrivate, PaperMetaReview

log = logging.getLogger(__name__)


def _count_public(arch) -> int:
    """Count public papers in an archive across known shapes; -1 on failure."""
    # Prefer per-agent map if present
    if hasattr(arch, "papers_by_agent") and isinstance(arch.papers_by_agent, dict):
        try:
            return sum(len(v or []) for v in arch.papers_by_agent.values())
        except Exception:
            pass

    # Fallback: flat list
    if hasattr(arch, "papers") and isinstance(arch.papers, list):
        try:
            return len(arch.papers)
        except Exception:
            pass

    # internal entries dict (used by PaperArchive in state.py)
    if hasattr(arch, "_entries") and isinstance(arch._entries, dict):
        try:
            return len([p for p in arch._entries.values() if p is not None])
        except Exception:
            pass

    # Fallback: explicit API
    count_fn = getattr(arch, "count", None)
    if callable(count_fn):
        try:
            return int(count_fn())
        except Exception:
            pass

    return -1


def _as_int_or_none(x, lo=None, hi=None):
    """Safely cast to int with optional bounds; return None if invalid."""
    try:
        v = int(x)
        if lo is not None and v < lo:
            return None
        if hi is not None and v > hi:
            return None
        return v
    except Exception:
        return None


def _as_float_or_none(x, lo=None, hi=None) -> float | None:
    try:
        v = float(x)
        if lo is not None and v < lo:
            return None
        if hi is not None and v > hi:
            return None
        return v
    except Exception:
        return None


def _iter_private_papers(
    private_archive: PrivatePaperArchive,
) -> Iterable[tuple[str, PaperMetadataPrivate]]:
    """Yield (agent_id, paper) pairs from the private archive across known shapes."""
    if hasattr(private_archive, "_entries") and isinstance(private_archive._entries, dict):
        for agent_id, papers in private_archive._entries.items():
            for paper in papers or []:
                yield agent_id, paper
        return

    if hasattr(private_archive, "papers_by_agent") and isinstance(
        private_archive.papers_by_agent, dict
    ):
        for agent_id, papers in private_archive.papers_by_agent.items():
            for paper in papers or []:
                yield agent_id, paper
        return

    if hasattr(private_archive, "papers") and isinstance(private_archive.papers, list):
        for paper in private_archive.papers:
            agent_id = getattr(paper, "primary_agent_id", None)
            if agent_id:
                yield agent_id, paper


def _lookup_score(scores: dict[str, Any], key: str) -> float | None:
    """Case-insensitive lookup for a score key in a score dict."""
    if not isinstance(scores, dict):
        return None
    if key in scores:
        return _as_float_or_none(scores.get(key))
    target = str(key).strip().lower()
    for k, v in scores.items():
        if str(k).strip().lower() == target:
            return _as_float_or_none(v)
    return None


def _compute_agent_reputation(
    private_archive: PrivatePaperArchive,
    *,
    agent_registry: AgentRegistry | None = None,
    score_keys: list[str],
    primary_score_key: str,
    min_score: float = 1.0,
    max_score: float = 5.0,
    include_acceptance_rate: bool = False,
    iteration: int | None = None,
    recency_halflife: float | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute per-agent reputation stats from stored reviews and decisions."""
    per_agent_papers: dict[str, list[PaperMetadataPrivate]] = {}
    for agent_id, paper in _iter_private_papers(private_archive):
        per_agent_papers.setdefault(agent_id, []).append(paper)
    if agent_registry is not None:
        for agent in agent_registry.get_all_agents():
            per_agent_papers.setdefault(agent.agent_id, [])

    reviews_by_paper = getattr(private_archive, "reviews_by_paper", {}) or {}
    reputation_by_agent: dict[str, dict[str, Any]] = {}

    for agent_id, papers in per_agent_papers.items():
        key_weighted_sum: dict[str, float] = dict.fromkeys(score_keys, 0.0)
        key_weight_total: dict[str, float] = dict.fromkeys(score_keys, 0.0)
        weighted_paper_sum = 0.0
        weight_total = 0.0
        papers_with_scores = 0
        accepted = 0
        decided = 0
        weighted_papers = 0
        review_count = 0
        paper_history: list[dict[str, Any]] = []

        for paper in papers:
            paper_id = getattr(paper, "paper_id", None)
            if not paper_id:
                continue

            per_key_values: dict[str, list[float]] = {k: [] for k in score_keys}
            paper_review_count = 0
            for review in reviews_by_paper.get(paper_id, []) or []:
                scores = review.get("scores") if isinstance(review, dict) else None
                if not scores:
                    continue
                saw_any_score = False
                for key in score_keys:
                    score_val = _lookup_score(scores, key)
                    score_val = _as_float_or_none(score_val, lo=min_score, hi=max_score)
                    if score_val is None:
                        continue
                    per_key_values[key].append(score_val)
                    saw_any_score = True
                if saw_any_score:
                    paper_review_count += 1

            paper_key_avgs: dict[str, float] = {}
            for key, values in per_key_values.items():
                if values:
                    paper_key_avgs[key] = sum(values) / len(values)

            if paper_key_avgs:
                papers_with_scores += 1
                paper_avg = sum(paper_key_avgs.values()) / len(paper_key_avgs)
                if iteration is not None and recency_halflife and recency_halflife > 0:
                    age = max(
                        0.0, float(iteration) - float(getattr(paper, "publication_t", iteration))
                    )
                    weight = 0.5 ** (age / float(recency_halflife))
                else:
                    weight = 1.0

                weighted_paper_sum += weight * paper_avg
                weight_total += weight
                weighted_papers += 1
                review_count += paper_review_count
                for key, avg_val in paper_key_avgs.items():
                    key_weighted_sum[key] += weight * avg_val
                    key_weight_total[key] += weight
                paper_history.append(
                    {
                        "publication_t": int(getattr(paper, "publication_t", 0) or 0),
                        "paper_avg": paper_avg,
                        "key_avgs": dict(paper_key_avgs),
                        "review_count": int(paper_review_count),
                    }
                )

            if include_acceptance_rate:
                status = getattr(paper, "status", None)
                if status in ("accepted", "rejected"):
                    decided += 1
                    if status == "accepted":
                        accepted += 1

        score_components = {}
        for key in score_keys:
            denom = key_weight_total[key]
            avg_val = (key_weighted_sum[key] / denom) if denom > 0 else None
            per_paper_vals = [
                h.get("key_avgs", {}).get(key)
                for h in paper_history
                if isinstance(h.get("key_avgs", {}).get(key), (int, float))
            ]
            std_val = None
            if len(per_paper_vals) >= 2:
                mean_val = sum(per_paper_vals) / len(per_paper_vals)
                var = sum((v - mean_val) ** 2 for v in per_paper_vals) / len(per_paper_vals)
                std_val = math.sqrt(max(0.0, var))
            score_components[key] = {
                "avg": avg_val,
                "weight": denom if denom > 0 else 0.0,
                "std": std_val,
            }

        reputation_score = (weighted_paper_sum / weight_total) if weight_total > 0 else None
        acceptance_rate = (accepted / decided) if include_acceptance_rate and decided else None

        # The primary score key ("Overall Recommendation") is the aggregate, not a
        # weighting dimension: `_normalize_metric_key` maps it to "overall", which is
        # absent from the policy-learning target_keys, so ranking it here would waste
        # one of the two weakest slots. Rank the sub-dimensions only.
        sorted_components = sorted(
            (
                (key, comp.get("avg"))
                for key, comp in score_components.items()
                if isinstance(comp.get("avg"), (int, float)) and key != primary_score_key
            ),
            key=lambda kv: float(kv[1]),
        )
        # Guard against overlap when fewer than 4 dimensions have scores; a dimension
        # must never be both the weakest and the strongest.
        n_edge = min(2, len(sorted_components) // 2)
        weakest_dimensions = [k for k, _ in sorted_components[:n_edge]]
        strongest_dimensions = [k for k, _ in sorted_components[len(sorted_components) - n_edge :]][
            ::-1
        ]
        if sorted_components and not weakest_dimensions:
            log.info(
                f"[update] agent '{agent_id}' has only {len(sorted_components)} scored "
                "dimension(s); policy learning has no weakest/strongest split to act on."
            )

        paper_history.sort(key=lambda h: h.get("publication_t", 0))
        trend_window = min(3, len(paper_history))
        overall_trend = None
        score_trends: dict[str, float | None] = dict.fromkeys(score_keys)
        if trend_window >= 1 and len(paper_history) >= 2:
            recent = paper_history[-trend_window:]
            prior = paper_history[:-trend_window]
            if prior:
                recent_avg = sum(float(h["paper_avg"]) for h in recent) / len(recent)
                prior_avg = sum(float(h["paper_avg"]) for h in prior) / len(prior)
                overall_trend = recent_avg - prior_avg
            for key in score_keys:
                recent_key_vals = [
                    float(h["key_avgs"][key])
                    for h in recent
                    if isinstance(h.get("key_avgs", {}).get(key), (int, float))
                ]
                prior_key_vals = [
                    float(h["key_avgs"][key])
                    for h in prior
                    if isinstance(h.get("key_avgs", {}).get(key), (int, float))
                ]
                if recent_key_vals and prior_key_vals:
                    score_trends[key] = (sum(recent_key_vals) / len(recent_key_vals)) - (
                        sum(prior_key_vals) / len(prior_key_vals)
                    )

        suggestions_by_key = {
            "Support": "Increase external evidence breadth and align claims with executed analyses.",
            "Soundness": "Add explicit diagnostic checks and document model assumptions/validity tests.",
            "Significance": "Target higher-impact hypotheses with stronger effect-size evidence.",
            "Originality": "Prioritize lower-overlap hypotheses and introduce method/mechanism novelty.",
            "Overall Recommendation": "Improve the weakest component scores before optimizing overall framing.",
        }
        suggested_actions = [
            suggestions_by_key.get(
                k, f"Improve performance on {k} through targeted iteration planning."
            )
            for k in weakest_dimensions
        ]
        confidence = min(1.0, (review_count / 12.0)) if review_count > 0 else 0.0

        reputation_by_agent[agent_id] = {
            "reputation_score": reputation_score,
            "primary_score_key": primary_score_key,
            "score_keys": list(score_keys),
            "score_components": score_components,
            "papers_total": len(papers),
            "papers_with_scores": papers_with_scores,
            "weighted_papers": weighted_papers,
            "acceptance_rate": acceptance_rate,
            "accepted_papers": accepted,
            "decision_count": decided,
            "n_reviews": review_count,
            "last_updated_iteration": iteration,
            "recency_halflife": recency_halflife,
            "diagnostics": {
                "weakest_dimensions": weakest_dimensions,
                "strongest_dimensions": strongest_dimensions,
                "overall_trend_delta": overall_trend,
                "score_trends": score_trends,
                "confidence": confidence,
                "suggested_actions": suggested_actions,
                "trend_window": trend_window,
            },
        }

    return reputation_by_agent


def _normalize_mode(mode: str | None) -> str:
    if not mode:
        return "none"
    return str(mode).strip().lower()


_PERMANENT_PAUSE_MODES = {
    "full_pause_permanent",
    "permanent_full_pause",
    "full_pause_permanent_replace",
    "full_pause_permanent_scheduled",
}


def _coerce_optional_nonneg_int(value: Any) -> int | None:
    """Parse optional non-negative integer config values.

    Returns None for unset/null-like values.
    """
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in ("", "none", "null"):
        return None
    return _as_int_or_none(value, lo=0)


def _parse_pause_schedule(raw: Any) -> list[dict[str, int]]:
    """Normalize pause schedule config into [{'iteration': int, 'pause_count': int}, ...]."""
    if raw is None or isinstance(raw, (str, bytes)):
        return []
    try:
        items = list(raw)
    except Exception:
        return []
    out: list[dict[str, int]] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        it = _as_int_or_none(item.get("iteration"), lo=1)
        cnt = _as_int_or_none(item.get("pause_count"), lo=1)
        if it is None or cnt is None:
            continue
        out.append({"iteration": it, "pause_count": cnt})
    out.sort(key=lambda d: (d["iteration"], d["pause_count"]))
    return out


def _resolve_score_cfg(reputation_cfg: dict[str, Any], pause_cfg: dict[str, Any]) -> dict[str, Any]:
    """Pick score config for reputation/pause with pause_cfg taking precedence."""

    def _get(cfg: dict[str, Any], key: str, default: Any) -> Any:
        return cfg.get(key, default) if isinstance(cfg, Mapping) else default

    rep_default = _get(reputation_cfg, "score_key", "Overall Recommendation")
    return {
        "score_key": _get(pause_cfg, "score_key", rep_default),
        "additional_score_keys": _get(
            pause_cfg, "additional_score_keys", _get(reputation_cfg, "additional_score_keys", [])
        ),
        "min_score": _get(pause_cfg, "min_score", _get(reputation_cfg, "min_score", 1)),
        "max_score": _get(pause_cfg, "max_score", _get(reputation_cfg, "max_score", 5)),
        "include_acceptance_rate": _get(
            pause_cfg,
            "include_acceptance_rate",
            _get(reputation_cfg, "include_acceptance_rate", False),
        ),
        "recency_halflife": _get(
            pause_cfg,
            "recency_halflife_iterations",
            _get(reputation_cfg, "recency_halflife_iterations", None),
        ),
    }


def _compute_pause_updates(
    *,
    agent_registry: AgentRegistry,
    reputations: dict[str, dict[str, Any]],
    pause_cfg: dict[str, Any],
    iteration: int | None,
) -> dict[str, dict[str, Any]]:
    """Compute per-agent pause state updates based on reputation.

    Temporary pause modes do not extend automatically once set; agents are
    unpaused after duration_iterations even if their reputation remains low.
    Permanent pause modes stay active once triggered.
    """
    mode = _normalize_mode(pause_cfg.get("mode"))
    if mode in ("none", "off", "disabled", "adjust_research_iters"):
        return {}

    threshold = float(pause_cfg.get("threshold", 0))
    permanent_pause = mode in _PERMANENT_PAUSE_MODES
    duration = int(pause_cfg.get("duration_iterations", 0))
    apply_missing = bool(pause_cfg.get("apply_when_score_missing", False))
    if not permanent_pause and duration <= 0:
        return {}

    now = int(iteration or 0)
    updates: dict[str, dict[str, Any]] = {}

    # ---------- permanent pause modes (with optional cap/schedule) ----------
    if permanent_pause:
        max_paused_agents = _coerce_optional_nonneg_int(pause_cfg.get("max_paused_agents"))
        schedule = _parse_pause_schedule(pause_cfg.get("schedule"))
        matched_schedule = [e for e in schedule if int(e["iteration"]) == now]
        schedule_pause_count = sum(int(e["pause_count"]) for e in matched_schedule)

        existing_active: dict[str, bool] = {}
        existing_permanent: dict[str, bool] = {}
        for agent_id in reputations:
            existing_active[agent_id] = False
            existing_permanent[agent_id] = False
            try:
                exists, prof = agent_registry.get_agent(agent_id)
                if exists and prof and getattr(prof, "pause_state", None):
                    ps = prof.pause_state
                    existing_active[agent_id] = bool(ps.get("active", False))
                    state_mode = _normalize_mode(ps.get("mode"))
                    is_perm = (
                        bool(ps.get("permanent", False)) or state_mode in _PERMANENT_PAUSE_MODES
                    )
                    existing_permanent[agent_id] = is_perm
            except Exception:
                continue

        existing_permanent_active = {
            aid
            for aid in reputations
            if existing_active.get(aid, False) and existing_permanent.get(aid, False)
        }

        candidates: list[tuple[str, float, float | None]] = []
        for agent_id, rep in reputations.items():
            score = rep.get("reputation_score") if isinstance(rep, dict) else None
            score = _as_float_or_none(score)

            if agent_id in existing_permanent_active:
                continue

            if mode == "full_pause_permanent_scheduled":
                should_consider = (score is not None) or apply_missing
            else:
                below = (score is not None and score < threshold) or (
                    score is None and apply_missing
                )
                should_consider = below

            if not should_consider:
                continue

            # Missing scores are treated as worst when apply_missing=true.
            sort_score = score if score is not None else float("-inf")
            candidates.append((agent_id, sort_score, score))

        candidates.sort(key=lambda row: (row[1], row[0]))

        if mode == "full_pause_permanent_scheduled":
            requested = schedule_pause_count
        else:
            requested = len(candidates)

        available_slots: int | None = None
        if max_paused_agents is not None:
            available_slots = max(0, max_paused_agents - len(existing_permanent_active))

        requested_candidates = candidates[:requested]
        if available_slots is None:
            selected = requested_candidates
            skipped_due_to_cap: list[tuple[str, float, float | None]] = []
        else:
            selected = requested_candidates[:available_slots]
            skipped_due_to_cap = requested_candidates[available_slots:]

        selected_ids = {aid for aid, _, _ in selected}
        selected_reason = "schedule" if mode == "full_pause_permanent_scheduled" else "threshold"

        def _fmt_pairs(rows: list[tuple[str, float, float | None]]) -> list[str]:
            out = []
            for aid, _sort_val, raw_score in rows:
                out.append(f"{aid}:{'NA' if raw_score is None else f'{raw_score:.4f}'}")
            return out

        log.info(
            f"[pause-policy] mode={mode} iteration={now} threshold={threshold} "
            f"max_paused_agents={max_paused_agents} apply_missing={apply_missing} "
            f"existing_permanent={len(existing_permanent_active)} candidates={len(candidates)} "
            f"requested={requested} schedule_events={matched_schedule if matched_schedule else []}"
        )
        log.info(
            f"[pause-selection] mode={mode} iteration={now} "
            f"selected={_fmt_pairs(selected)} skipped_due_to_cap={_fmt_pairs(skipped_due_to_cap)}"
        )

        for agent_id in reputations:
            if agent_id in existing_permanent_active:
                updates[agent_id] = {
                    "active": True,
                    "mode": mode,
                    "threshold": threshold,
                    "duration_iterations": None,
                    "until_iteration": None,
                    "permanent": True,
                    "last_updated_iteration": now,
                    "action": "continued",
                    "selection_reason": "already_permanent",
                }
                continue

            if agent_id in selected_ids:
                updates[agent_id] = {
                    "active": True,
                    "mode": mode,
                    "threshold": threshold,
                    "duration_iterations": None,
                    "until_iteration": None,
                    "permanent": True,
                    "last_updated_iteration": now,
                    "action": "paused",
                    "selection_reason": selected_reason,
                }
            else:
                updates[agent_id] = {
                    "active": False,
                    "mode": "none",
                    "threshold": threshold,
                    "duration_iterations": None,
                    "until_iteration": None,
                    "permanent": False,
                    "last_updated_iteration": now,
                    "action": "none",
                    "selection_reason": "not_selected",
                }

        return updates

    for agent_id, rep in reputations.items():
        score = rep.get("reputation_score") if isinstance(rep, dict) else None
        below = score is not None and score < threshold
        if score is None and apply_missing:
            below = True

        existing_until = None
        existing_active = False
        try:
            exists, prof = agent_registry.get_agent(agent_id)
            if exists and prof and getattr(prof, "pause_state", None):
                existing_until = prof.pause_state.get("until_iteration")
                existing_active = bool(prof.pause_state.get("active", False))
        except Exception:
            existing_until = None

        if isinstance(existing_until, int) and now <= existing_until:
            updates[agent_id] = {
                "active": True,
                "mode": mode,
                "threshold": threshold,
                "duration_iterations": duration,
                "until_iteration": existing_until,
                "permanent": False,
                "last_updated_iteration": now,
                "action": "continued" if existing_active else "paused",
                "selection_reason": "duration_window",
            }
            continue

        if isinstance(existing_until, int) and now > existing_until and existing_active:
            updates[agent_id] = {
                "active": False,
                "mode": "none",
                "threshold": threshold,
                "duration_iterations": duration,
                "until_iteration": None,
                "permanent": False,
                "last_updated_iteration": now,
                "action": "unpaused",
                "selection_reason": "duration_elapsed",
            }
            continue

        if below:
            new_until = now + duration
            updates[agent_id] = {
                "active": True,
                "mode": mode,
                "threshold": threshold,
                "duration_iterations": duration,
                "until_iteration": new_until,
                "permanent": False,
                "last_updated_iteration": now,
                "action": "paused",
                "selection_reason": "threshold",
            }
        else:
            updates[agent_id] = {
                "active": False,
                "mode": "none",
                "threshold": threshold,
                "duration_iterations": duration,
                "until_iteration": None,
                "permanent": False,
                "last_updated_iteration": now,
                "action": "none",
                "selection_reason": "threshold_not_met",
            }

    return updates


def _normalize_metric_key(name: str) -> str:
    key = str(name or "").strip().lower()
    aliases = {
        "support": "support",
        "soundness": "soundness",
        "significance": "significance",
        "originality": "originality",
        "overall recommendation": "overall",
        "overall": "overall",
    }
    return aliases.get(key, key.replace(" ", "_"))


def _normalize_weights(weights: dict[str, float], default_keys: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in default_keys:
        v = float(weights.get(k, 1.0))
        out[k] = max(0.1, v)
    total = sum(out.values())
    if total <= 0:
        return dict.fromkeys(default_keys, 1.0)
    scale = float(len(default_keys)) / total
    return {k: out[k] * scale for k in default_keys}


def _compute_policy_learning_updates(
    *,
    agent_registry: AgentRegistry,
    reputations: dict[str, dict[str, Any]],
    iteration: int | None,
    policy_cfg: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Generate lightweight per-agent policy updates from reputation diagnostics."""
    cfg = policy_cfg or {}
    if not bool(cfg.get("enabled", True)):
        return {}

    lr = float(cfg.get("learning_rate", 0.15))
    min_delta = float(cfg.get("min_delta_for_update", 0.01))
    target_keys = ["support", "soundness", "significance", "originality"]
    updates: dict[str, dict[str, Any]] = {}

    for agent_id, rep in reputations.items():
        exists, profile = agent_registry.get_agent(agent_id)
        if not exists or profile is None:
            continue

        prev_learning = getattr(profile, "learning_state", None) or {}
        prev_score = _as_float_or_none(prev_learning.get("prev_reputation_score"))
        curr_score = _as_float_or_none(rep.get("reputation_score"))
        delta = None
        if prev_score is not None and curr_score is not None:
            delta = curr_score - prev_score

        current_weights_raw = getattr(profile, "metric_weights", None) or {}
        current_weights = {
            _normalize_metric_key(k): float(v)
            for k, v in current_weights_raw.items()
            if isinstance(v, (int, float))
        }
        weights = _normalize_weights(current_weights, target_keys)

        diag = rep.get("diagnostics", {}) if isinstance(rep, dict) else {}
        weak_dims = [_normalize_metric_key(k) for k in (diag.get("weakest_dimensions") or [])]
        strong_dims = [_normalize_metric_key(k) for k in (diag.get("strongest_dimensions") or [])]
        weak_dims = [k for k in weak_dims if k in target_keys]
        strong_dims = [k for k in strong_dims if k in target_keys]

        should_update = (delta is None) or (abs(delta) >= min_delta) or bool(weak_dims)
        if should_update and weak_dims:
            for k in weak_dims:
                weights[k] += lr
            for k in strong_dims:
                weights[k] -= lr * 0.5
            weights = _normalize_weights(weights, target_keys)

        learning_state = dict(prev_learning)
        learning_state.update(
            {
                "prev_reputation_score": curr_score,
                "delta_reputation_score": delta,
                "last_focus_dimensions": weak_dims,
                "last_updated_iteration": iteration,
                "update_count": int(prev_learning.get("update_count", 0) or 0) + 1,
                "learning_rate": lr,
                "min_delta_for_update": min_delta,
                "applied_update": bool(should_update and weak_dims),
            }
        )

        updates[agent_id] = {
            "metric_weights": weights,
            "learning_state": learning_state,
        }

    return updates


def update_status_and_archives(
    meta_reviews: list[PaperMetaReview],
    candidate_papers: list[PaperMetadataPrivate],
    archive: PaperArchive,
    private_archive: PrivatePaperArchive,
    agent_registry: AgentRegistry,
    *,
    reputation_cfg: dict[str, Any] | None = None,
    pause_cfg: dict[str, Any] | None = None,
    iteration: int | None = None,
) -> dict[str, Any]:
    """Apply meta-review decisions and propagate updates to archives and registry.

    Args:
        meta_reviews: List of meta-review decisions.
        candidate_papers: Papers under consideration this iteration.
        archive: Public archive to add accepted papers and citation updates.
        private_archive: Private archive storing all papers and reviews.
        agent_registry: Registry of agents to update citation/accept counts.

    Returns:
        Dict with update summaries. Key fields:
            - reputations: computed reputation payload by agent_id
            - pause_updates: pause_state updates keyed by agent_id
            - newly_paused_agents: list of {"agent_id", "reputation_score"}
    """
    update_summary: dict[str, Any] = {
        "reputations": {},
        "pause_updates": {},
        "newly_paused_agents": [],
    }
    log.info(
        f"[update] received {len(candidate_papers)} candidates, {len(meta_reviews)} meta-reviews"
    )

    if not meta_reviews or not candidate_papers:
        log.info("[update] nothing to do (no meta-reviews or no candidates)")
        # There is deliberately no score-based fallback decision here: the real
        # decision path sets seven interdependent fields, and a partial fallback
        # would leave papers in a state the archive and citation code do not
        # expect. Make the gap loud instead.
        if candidate_papers and not meta_reviews:
            log.warning(
                f"[update] {len(candidate_papers)} candidate paper(s) have NO "
                "meta-review and will remain undecided this iteration: "
                f"{[getattr(p, 'paper_id', '?') for p in candidate_papers[:10]]}"
            )
        accepted_papers = []
        rejected_papers = []
    else:
        # Index candidates by id
        paper_dict = {p.paper_id: p for p in candidate_papers}

        # Apply meta-review decisions to candidate objects
        for mr in meta_reviews:
            p = paper_dict.get(mr.paper_id)
            if p is None:
                continue
            publication_status = "accepted" if mr.decision == "accept" else "rejected"
            previous_status = str(getattr(p, "status", "") or "").strip()
            if not str(getattr(p, "research_validation_status", "") or "").strip():
                p.research_validation_status = previous_status or "under_review"
            p.publication_status = publication_status
            # Keep historical `status` behavior for compatibility with existing code paths.
            p.status = publication_status
            citation_compliant = getattr(p, "citation_compliant", True)
            if citation_compliant is None:
                citation_compliant = True
            p.citation_compliant = bool(citation_compliant)
            p.is_citable_archive_source = publication_status == "accepted"
            p.citable_with_warnings = bool(p.is_citable_archive_source and not p.citation_compliant)
            p.metareview = mr

        # Collect accepted papers
        accepted_papers = [p for p in candidate_papers if getattr(p, "status", None) == "accepted"]
        rejected_papers = [p for p in candidate_papers if getattr(p, "status", None) == "rejected"]
        accepted_ids = [p.paper_id for p in accepted_papers]
        rejected_ids = [p.paper_id for p in rejected_papers]
        log.info(f"[update] accepted_ids={accepted_ids} (n={len(accepted_ids)})")
        log.info(f"[update] rejected_ids={rejected_ids} (n={len(rejected_ids)})")

        # Build a set of existing public paper_ids to avoid duplicates
        existing_public_ids = set()
        try:
            if hasattr(archive, "papers_by_agent") and isinstance(archive.papers_by_agent, dict):
                for _plist in archive.papers_by_agent.values():
                    for _p in _plist or []:
                        pid = getattr(_p, "paper_id", None)
                        if pid is not None:
                            existing_public_ids.add(pid)
            elif hasattr(archive, "papers") and isinstance(archive.papers, list):
                for _p in archive.papers:
                    pid = getattr(_p, "paper_id", None)
                    if pid is not None:
                        existing_public_ids.add(pid)

            # look inside the internal store used by PaperArchive
            elif hasattr(archive, "_entries") and isinstance(archive._entries, dict):
                for _p in archive._entries.values():
                    pid = getattr(_p, "paper_id", None)
                    if pid is not None:
                        existing_public_ids.add(pid)
        except Exception as e:
            log.warning(f"[update] could not read existing public archive to de-dupe: {e}")

        to_add_public = [
            p for p in accepted_papers if getattr(p, "paper_id", None) not in existing_public_ids
        ]

        # Promote accepted (de-duped) papers to PUBLIC archive
        if to_add_public:
            try:
                archive.add(to_add_public)
            except Exception as e:
                log.warning(f"[update] archive.add failed: {e}")

        # (Optional) Persist accepted to PRIVATE archive too (also de-dupe for safety).
        # `_iter_private_papers` already handles every archive shape, including the
        # `_entries` dict that PrivatePaperArchive actually uses.
        existing_private_ids = set()
        try:
            for _agent_id, _p in _iter_private_papers(private_archive):
                pid = getattr(_p, "paper_id", None)
                if pid is not None:
                    existing_private_ids.add(pid)
        except Exception as e:
            log.warning(f"[update] could not read existing private archive to de-dupe: {e}")

        to_add_private_accept = [
            p for p in accepted_papers if getattr(p, "paper_id", None) not in existing_private_ids
        ]
        if to_add_private_accept:
            try:
                skipped = private_archive.add(to_add_private_accept)
                if skipped:
                    log.warning(
                        f"[update] {len(skipped)} accepted paper(s) already in private "
                        f"archive, skipped: {skipped[:5]}"
                    )
            except Exception as e:
                log.warning(f"[update] private_archive.add failed: {e}")

        to_add_private_reject = [
            p for p in rejected_papers if getattr(p, "paper_id", None) not in existing_private_ids
        ]
        if to_add_private_reject:
            try:
                skipped = private_archive.add(to_add_private_reject)
                if skipped:
                    log.warning(
                        f"[update] {len(skipped)} rejected paper(s) already in private "
                        f"archive, skipped: {skipped[:5]}"
                    )
            except Exception as e:
                log.warning(f"[update] private_archive.add (rejected) failed: {e}")

        # Update citation counts for accepted papers
        # Under Markovian public memory (experiment 08) a cited accepted paper may have been pruned from
        # the public archive. Its private-archive copy still gains the citation. A cited paper
        # that was never accepted (citable when full_citing is off) is not in the public
        # archive either; as in the source, its paper counts are left unchanged. The cited
        # author's registry count gains every citation. Misses are counted for observability.
        public_archive_citation_misses = 0
        pruned_paper_citations = 0
        for p in accepted_papers:
            cited = getattr(p, "cited_paper_ids", None)
            if not cited:
                continue

            for citation in cited:
                try:
                    # Be defensive about schema
                    cited_paper_id = (
                        citation.get("paper_id") if isinstance(citation, dict) else None
                    )
                    cited_agent_id = (
                        citation.get("agent_id") if isinstance(citation, dict) else None
                    )
                    if not cited_paper_id or not cited_agent_id:
                        continue

                    # Update public archive record (if found)
                    try:
                        current_papers = archive.get_most_recent_by_agent(cited_agent_id, k=1000)
                    except Exception:
                        current_papers = []
                    target = next(
                        (
                            cp
                            for cp in current_papers
                            if getattr(cp, "paper_id", None) == cited_paper_id
                        ),
                        None,
                    )
                    new_cit: int | None = None
                    if target is not None:
                        new_cit = int(getattr(target, "citation_count", 0)) + 1
                        try:
                            archive.update_entry(cited_agent_id, cited_paper_id, new_cit)
                        except Exception as e:
                            log.warning(
                                f"[update] archive.update_entry failed for {cited_paper_id}: {e}"
                            )
                    else:
                        public_archive_citation_misses += 1
                        try:
                            private_papers = private_archive.get_by_primary_agent(
                                cited_agent_id, k=None
                            )
                        except Exception as e:
                            log.warning(
                                f"[update] private_archive lookup failed for {cited_paper_id}: {e}"
                            )
                            private_papers = []
                        private_target = next(
                            (
                                pp
                                for pp in private_papers
                                if getattr(pp, "paper_id", None) == cited_paper_id
                                and getattr(pp, "status", None) == "accepted"
                            ),
                            None,
                        )
                        if private_target is not None:
                            pruned_paper_citations += 1
                            new_cit = int(getattr(private_target, "citation_count", 0)) + 1
                    if new_cit is not None:
                        try:
                            private_archive.update_entry(cited_agent_id, cited_paper_id, new_cit)
                        except Exception as e:
                            log.warning(
                                f"[update] private_archive.update_entry failed for {cited_paper_id}: {e}"
                            )

                    # Update agent-level citation counter
                    try:
                        exists, agent_profile = agent_registry.get_agent(cited_agent_id)
                        if exists and agent_profile:
                            new_cit_total = int(getattr(agent_profile, "citation_count", 0)) + 1
                            agent_registry.update(cited_agent_id, new_citations=new_cit_total)
                    except Exception as e:
                        log.warning(
                            f"[update] agent_registry citation update failed for {cited_agent_id}: {e}"
                        )

                except Exception as e:
                    log.warning(f"[update] citation processing error: {e}")

        if public_archive_citation_misses:
            log.info(
                f"[update] {public_archive_citation_misses} citation(s) to papers not in the "
                f"public archive: {pruned_paper_citations} to accepted papers no longer public "
                "(pruned by Markovian social memory), counted in their private-archive copy; "
                f"{public_archive_citation_misses - pruned_paper_citations} to papers never "
                "accepted or not found, paper counts unchanged. Registry counts of the cited "
                "authors are incremented as for any citation."
            )

        # Increment acceptance counts per primary author
        for p in accepted_papers:
            try:
                exists, prof = agent_registry.get_agent(p.primary_agent_id)
                if exists and prof:
                    new_acc = int(getattr(prof, "num_accepted_papers", 0)) + 1
                    agent_registry.update(p.primary_agent_id, new_accepted_papers=new_acc)
            except Exception as e:
                log.warning(
                    f"[update] agent_registry acceptance update failed for {p.primary_agent_id}: {e}"
                )

    pub_count = _count_public(archive)

    if pub_count == -1 and hasattr(archive, "_entries") and isinstance(archive._entries, dict):
        try:
            pub_count = len([p for p in archive._entries.values() if p is not None])
        except Exception:
            pass

    if pub_count == -1:
        try:
            # Prefer an API if your registry exposes IDs
            list_ids = getattr(agent_registry, "list_ids", None)
            agent_ids = list_ids() if callable(list_ids) else []
            if agent_ids:
                total = 0
                for aid in agent_ids:
                    try:
                        items = archive.get_most_recent_by_agent(aid, k=10_000) or []
                        total += len(items)
                    except Exception:
                        continue
                if total > 0:
                    pub_count = total
        except Exception:
            # keep pub_count as-is (0) on any failure
            pass

    log.info(f"[update] public archive now has {pub_count} papers")

    # Update agent reputations + pause states (optional)
    rep_cfg = reputation_cfg or {}
    pause_cfg = pause_cfg or {}
    pause_mode = _normalize_mode(pause_cfg.get("mode"))
    need_reputation = bool(
        rep_cfg.get("enabled", False) or pause_mode not in ("none", "off", "disabled")
    )

    if need_reputation:
        score_cfg = _resolve_score_cfg(rep_cfg, pause_cfg)
        score_key = score_cfg.get("score_key", "Overall Recommendation")
        extra_keys = score_cfg.get("additional_score_keys", []) or []
        score_keys = [score_key] + [k for k in extra_keys if k and k != score_key]
        min_score = float(score_cfg.get("min_score", 1))
        max_score = float(score_cfg.get("max_score", 5))
        include_acceptance_rate = bool(score_cfg.get("include_acceptance_rate", False))
        recency_halflife = score_cfg.get("recency_halflife", None)

        try:
            reputations = _compute_agent_reputation(
                private_archive,
                agent_registry=agent_registry,
                score_keys=score_keys,
                primary_score_key=score_key,
                min_score=min_score,
                max_score=max_score,
                include_acceptance_rate=include_acceptance_rate,
                iteration=iteration,
                recency_halflife=recency_halflife,
            )
        except Exception as e:
            log.warning(f"[update] reputation computation failed: {e}")
            reputations = {}
        update_summary["reputations"] = reputations

        if (rep_cfg.get("enabled", False) or pause_mode == "adjust_research_iters") and reputations:
            for agent_id, rep in reputations.items():
                try:
                    agent_registry.update(agent_id, new_reputation=rep)
                except Exception as e:
                    log.warning(f"[update] reputation update failed for {agent_id}: {e}")

        policy_cfg = rep_cfg.get("policy_learning", {}) if isinstance(rep_cfg, Mapping) else {}
        if rep_cfg.get("enabled", False) and reputations:
            learning_updates = _compute_policy_learning_updates(
                agent_registry=agent_registry,
                reputations=reputations,
                iteration=iteration,
                policy_cfg=policy_cfg,
            )
            for agent_id, upd in learning_updates.items():
                try:
                    agent_registry.update(
                        agent_id,
                        new_metric_weights=upd.get("metric_weights"),
                        new_learning_state=upd.get("learning_state"),
                    )
                except Exception as e:
                    log.warning(f"[update] policy-learning update failed for {agent_id}: {e}")

        if pause_mode not in ("none", "off", "disabled") and reputations:
            pause_updates = _compute_pause_updates(
                agent_registry=agent_registry,
                reputations=reputations,
                pause_cfg=pause_cfg,
                iteration=iteration,
            )
            update_summary["pause_updates"] = pause_updates
            newly_paused_agents: list[dict[str, Any]] = []
            for agent_id, pause_state in pause_updates.items():
                try:
                    agent_registry.update(agent_id, new_pause_state=pause_state)
                    action = pause_state.get("action")
                    rep = reputations.get(agent_id, {})
                    score = rep.get("reputation_score")
                    keys_label = ",".join(rep.get("score_keys") or [])
                    until_iter = pause_state.get("until_iteration")
                    reason = pause_state.get("selection_reason")
                    log.info(
                        f"[pause-action] agent_id={agent_id} action={action} reason={reason} "
                        f"reputation_score={score} score_keys=[{keys_label}] "
                        f"threshold={pause_state.get('threshold')} "
                        f"until_iteration={until_iter} permanent={pause_state.get('permanent')}"
                    )
                    if action == "paused":
                        newly_paused_agents.append(
                            {
                                "agent_id": agent_id,
                                "reputation_score": score,
                                "reason": reason,
                            }
                        )
                except Exception as e:
                    log.warning(f"[update] pause_state update failed for {agent_id}: {e}")
            update_summary["newly_paused_agents"] = newly_paused_agents

    return update_summary
