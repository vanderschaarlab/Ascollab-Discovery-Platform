import hashlib
from collections.abc import Iterable, Mapping
from typing import Any

DEFAULT_FIXED_WORKFLOW_CFG: dict[str, Any] = {
    "enabled": False,
    "assignment_policy": "weakest_agent",
    "tie_break": "seeded_random",
    "self_fallback": "second_weakest",
    "require_establish_before_first_code": True,
    "require_collab_message": True,
    "min_collab_messages": 1,
    "force_tool_target": True,
    "disable_agent_registry_tool": True,
    "write_artifacts": True,
}


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _nonneg_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        return default
    return max(0, parsed)


def normalize_fixed_workflow_cfg(raw: Any) -> dict[str, Any]:
    cfg = dict(DEFAULT_FIXED_WORKFLOW_CFG)
    if isinstance(raw, Mapping):
        cfg.update(dict(raw))

    cfg["enabled"] = _truthy(cfg.get("enabled"))
    cfg["assignment_policy"] = str(cfg.get("assignment_policy") or "weakest_agent").strip().lower()
    if cfg["assignment_policy"] not in {"weakest_agent", "seeded_random"}:
        cfg["assignment_policy"] = "weakest_agent"

    cfg["tie_break"] = str(cfg.get("tie_break") or "seeded_random").strip().lower()
    if cfg["tie_break"] not in {"seeded_random", "agent_id"}:
        cfg["tie_break"] = "seeded_random"

    cfg["self_fallback"] = str(cfg.get("self_fallback") or "second_weakest").strip().lower()
    if cfg["self_fallback"] not in {"second_weakest", "seeded_random"}:
        cfg["self_fallback"] = "second_weakest"

    for key in (
        "require_establish_before_first_code",
        "require_collab_message",
        "force_tool_target",
        "disable_agent_registry_tool",
        "write_artifacts",
    ):
        cfg[key] = _truthy(cfg.get(key))

    cfg["min_collab_messages"] = _nonneg_int(cfg.get("min_collab_messages"), 1)
    return cfg


def _stable_unit_float(*parts: Any) -> float:
    blob = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
    return int(digest, 16) / float(16**16)


def _agent_id(agent: Any) -> str:
    return str(getattr(agent, "agent_id", None) or getattr(agent, "id", "") or "").strip()


def _reputation_score(agent: Any) -> float | None:
    rep = getattr(agent, "reputation", None)
    if not isinstance(rep, Mapping):
        return None
    for key in ("reputation_score", "overall_score", "overall", "score"):
        value = rep.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def agent_weakness_key(
    agent: Any, seed: int, iteration: int, tie_break: str = "seeded_random"
) -> tuple:
    """Build a sort key ordering agents weakest-first.

    Note: an agent with no reputation data sorts as weakest (``-inf``). At
    iteration 1 nobody has reputation, so everyone ties and the tiebreak decides;
    mid-run this means a freshly added replacement agent is picked as weakest.
    That is retained deliberately so the ordering does not change under this fix.

    Args:
        agent: Agent profile to score.
        seed: Run seed, used for the seeded_random tiebreak.
        iteration: Current iteration, used for the seeded_random tiebreak.
        tie_break: "seeded_random" (default) hashes (seed, iteration, agent_id);
            "agent_id" zeroes the hash so the trailing agent_id orders ties
            lexicographically.

    Returns:
        A tuple suitable for use as a sort key; lower sorts as weaker.
    """
    agent_id = _agent_id(agent)
    rep_score = _reputation_score(agent)
    reputation_sort = rep_score if rep_score is not None else float("-inf")
    accepted = int(getattr(agent, "num_accepted_papers", 0) or 0)
    citations = int(getattr(agent, "citation_count", 0) or 0)
    tie = 0.0 if tie_break == "agent_id" else _stable_unit_float(seed, iteration, agent_id)
    return (reputation_sort, accepted, citations, tie, agent_id)


def _seeded_random_order(
    agent_ids: Iterable[str], *, seed: int, iteration: int, label: str
) -> list[str]:
    return sorted(
        [str(aid).strip() for aid in agent_ids if str(aid).strip()],
        key=lambda aid: (_stable_unit_float(seed, iteration, label, aid), aid),
    )


def resolve_fixed_workflow_targets(
    agents: Iterable[Any],
    *,
    iteration: int,
    seed: int,
    paused_agent_ids: Iterable[str] | None = None,
    cfg: Mapping[str, Any] | None = None,
) -> tuple[dict[str, str], str | None]:
    """Resolve the fixed-workflow collaborator assignment for one iteration.

    Args:
        agents: All agent profiles for this iteration.
        iteration: Current iteration index.
        seed: Run seed.
        paused_agent_ids: Agents excluded from being targets or participants.
        cfg: Fixed-workflow config block.

    Returns:
        Tuple of (assignments, primary_target_id). The target is returned explicitly
        so metadata never has to re-derive or infer it -- inference from the map is
        ambiguous for a two-agent run, where every id appears exactly once.
    """
    norm = normalize_fixed_workflow_cfg(cfg)
    if not norm.get("enabled", False):
        return {}, None

    paused = {str(aid).strip() for aid in (paused_agent_ids or []) if str(aid).strip()}
    active_agents = [
        agent for agent in agents if _agent_id(agent) and _agent_id(agent) not in paused
    ]
    if len(active_agents) < 2:
        return {}, None

    policy = str(norm.get("assignment_policy", "weakest_agent"))
    if policy == "seeded_random":
        ordered_ids = _seeded_random_order(
            [_agent_id(agent) for agent in active_agents],
            seed=seed,
            iteration=iteration,
            label="fixed_workflow_seeded_random",
        )
    else:
        ordered_agents = sorted(
            active_agents,
            key=lambda agent: agent_weakness_key(
                agent,
                seed=int(seed),
                iteration=int(iteration),
                tie_break=str(norm.get("tie_break", "seeded_random")),
            ),
        )
        ordered_ids = [_agent_id(agent) for agent in ordered_agents]

    primary_target = ordered_ids[0]
    fallback_target = ordered_ids[1] if len(ordered_ids) > 1 else None
    if norm.get("self_fallback") == "seeded_random":
        random_order = _seeded_random_order(
            ordered_ids,
            seed=seed,
            iteration=iteration,
            label="fixed_workflow_self_fallback",
        )
    else:
        random_order = ordered_ids

    assignments: dict[str, str] = {}
    for agent in active_agents:
        aid = _agent_id(agent)
        if aid != primary_target:
            assignments[aid] = primary_target
            continue
        fallback = fallback_target
        if norm.get("self_fallback") == "seeded_random":
            fallback = next(
                (candidate for candidate in random_order if candidate != aid), fallback_target
            )
        if fallback:
            assignments[aid] = fallback

    return assignments, primary_target


def build_fixed_collaborator_map(
    agents: Iterable[Any],
    *,
    iteration: int,
    seed: int,
    paused_agent_ids: Iterable[str] | None = None,
    cfg: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Backward-compatible wrapper returning only the assignment map.

    Args:
        agents: All agent profiles for this iteration.
        iteration: Current iteration index.
        seed: Run seed.
        paused_agent_ids: Agents excluded from being targets or participants.
        cfg: Fixed-workflow config block.

    Returns:
        Mapping of agent_id -> assigned collaborator id.
    """
    assignments, _primary = resolve_fixed_workflow_targets(
        agents,
        iteration=iteration,
        seed=seed,
        paused_agent_ids=paused_agent_ids,
        cfg=cfg,
    )
    return assignments


def make_fixed_workflow_context_by_agent(
    collaborator_map: Mapping[str, str],
    *,
    cfg: Mapping[str, Any],
    iteration: int,
) -> dict[str, dict[str, Any]]:
    norm = normalize_fixed_workflow_cfg(cfg)
    contexts: dict[str, dict[str, Any]] = {}
    if not norm.get("enabled", False):
        return contexts
    for agent_id, collaborator_id in collaborator_map.items():
        aid = str(agent_id).strip()
        cid = str(collaborator_id).strip()
        if not aid or not cid:
            continue
        contexts[aid] = {
            "enabled": True,
            "iteration": int(iteration),
            "assigned_collaborator_id": cid,
            "assignment_policy": norm.get("assignment_policy"),
            "require_establish_before_first_code": bool(
                norm.get("require_establish_before_first_code")
            ),
            "require_collab_message": bool(norm.get("require_collab_message")),
            "min_collab_messages": int(norm.get("min_collab_messages", 1) or 0),
            "force_tool_target": bool(norm.get("force_tool_target")),
            "disable_agent_registry_tool": bool(norm.get("disable_agent_registry_tool")),
        }
    return contexts


def remove_agent_registry_search_tool(
    primary_agent_tools: Any, cfg: Mapping[str, Any] | None = None
) -> bool:
    norm = normalize_fixed_workflow_cfg(cfg)
    if not (
        norm.get("enabled", False)
        and norm.get("disable_agent_registry_tool", True)
        and isinstance(primary_agent_tools, dict)
    ):
        return False
    return primary_agent_tools.pop("search_agent_registry", None) is not None


def make_fixed_workflow_metadata(
    *,
    iteration: int,
    seed: int = 0,
    cfg: Mapping[str, Any],
    collaborator_map: Mapping[str, str],
    agents: Iterable[Any],
    primary_target_id: str | None = None,
) -> dict[str, Any]:
    """Build the fixed-workflow metadata artifact for one iteration.

    Args:
        iteration: Current iteration index.
        seed: Run seed.
        cfg: Fixed-workflow config block.
        collaborator_map: Assignments produced by resolve_fixed_workflow_targets.
        agents: All agent profiles, including paused ones.
        primary_target_id: The target the assignment function actually chose.
            Passed in rather than recomputed, because recomputation used
            different rules (paused agents included, policy ignored) and
            disagreed with the real assignment.

    Returns:
        The metadata dict written to the fixed-workflow artifact.
    """
    norm = normalize_fixed_workflow_cfg(cfg)
    score_rows: list[dict[str, Any]] = []
    agent_list = [agent for agent in agents if _agent_id(agent)]
    for agent in agent_list:
        aid = _agent_id(agent)
        if not aid:
            continue
        score_rows.append(
            {
                "agent_id": aid,
                "reputation_overall": _reputation_score(agent),
                "num_accepted_papers": int(getattr(agent, "num_accepted_papers", 0) or 0),
                "citation_count": int(getattr(agent, "citation_count", 0) or 0),
            }
        )
    weakest_agent_id = None
    if agent_list:
        weakest_agent_id = _agent_id(
            min(
                agent_list,
                key=lambda agent: agent_weakness_key(
                    agent,
                    seed=int(seed),
                    iteration=int(iteration),
                    tie_break=str(norm.get("tie_break", "seeded_random")),
                ),
            )
        )
    return {
        "enabled": bool(norm.get("enabled", False)),
        "iteration": int(iteration),
        "assignment_policy": str(norm.get("assignment_policy", "weakest_agent")),
        "tie_break": str(norm.get("tie_break", "seeded_random")),
        "self_fallback": str(norm.get("self_fallback", "second_weakest")),
        "write_artifacts": bool(norm.get("write_artifacts", True)),
        # The agent everyone was actually assigned to, as chosen by the assignment
        # function itself.
        "primary_target_id": primary_target_id,
        # Diagnostic only: who the weakness ranking would pick across ALL agents,
        # including paused ones. Never used as the assignment target.
        "weakest_agent_id": weakest_agent_id,
        "assignments": {str(k): str(v) for k, v in collaborator_map.items()},
        "agent_scores": score_rows,
        "compliance": [],
        "summary": {"passed": 0, "failed": 0, "total": 0},
    }


def _timeline(log: Any) -> list[dict[str, Any]]:
    events = getattr(log, "timeline", None)
    return (
        [dict(event) for event in events if isinstance(event, Mapping)]
        if isinstance(events, list)
        else []
    )


def validate_fixed_workflow_compliance(
    paper: Any,
    session_log: Any,
    assigned_collaborator: str | None,
    cfg: Mapping[str, Any] | None = None,
) -> list[str]:
    norm = normalize_fixed_workflow_cfg(cfg)
    if not norm.get("enabled", False):
        return []

    reasons: list[str] = []
    assigned = str(assigned_collaborator or "").strip()
    if not assigned:
        return ["no_assigned_collaborator"]

    collab_ids = {
        str(collab_id).strip()
        for collab_id in (getattr(paper, "collab_agent_ids", []) or [])
        if str(collab_id).strip()
    }
    if assigned not in collab_ids:
        reasons.append("assigned_collaborator_missing_from_paper")

    events = _timeline(session_log)
    established_steps: list[int] = []
    for event in events:
        if event.get("kind") != "tool_result_establish_collaboration":
            continue
        if str(event.get("collab_agent_id", "")).strip() != assigned:
            continue
        try:
            established_steps.append(int(event.get("step", 0) or 0))
        except Exception:
            established_steps.append(0)
    if not established_steps:
        reasons.append("assigned_collaboration_not_established")

    if bool(norm.get("require_collab_message", True)):
        min_messages = int(norm.get("min_collab_messages", 1) or 0)
        # Count COMPLETED interactions only. A single communicate_with_collaborator
        # call emits both a tool_call_* and a tool_result_* event, so counting both
        # kinds double-counts every message. A tool_call with no matching result
        # means the call failed and should not satisfy the requirement.
        message_count = sum(
            1
            for event in events
            if event.get("kind") == "tool_result_communicate_with_collaborator"
        )
        if message_count < min_messages:
            reasons.append("insufficient_collaborator_interactions")

    # Only a violation when the run actually removed the tool; otherwise the agent
    # was legitimately given it (see remove_agent_registry_search_tool, gated on the
    # same flag).
    if bool(norm.get("disable_agent_registry_tool", True)):
        search_events = [
            event
            for event in events
            if (
                event.get("kind") == "tool_result_search_agent_registry"
                or (
                    event.get("kind") == "tool_call"
                    and event.get("tool") == "search_agent_registry"
                )
            )
        ]
        if search_events:
            reasons.append("agent_registry_search_used")

    if bool(norm.get("require_establish_before_first_code", True)) and established_steps:
        first_establish = min(established_steps)
        for event in events:
            if event.get("kind") != "tool_call" or event.get("tool") != "execute_code":
                continue
            try:
                step = int(event.get("step", 0) or 0)
            except Exception:
                step = 0
            if step < first_establish:
                reasons.append("execute_code_before_fixed_collaboration")
                break

    return reasons
