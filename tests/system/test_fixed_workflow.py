from agents.research.primary_agent_utils import make_primary_agent_system_prompt
from agents.research.research_team import (
    _fixed_workflow_block_observation,
    _fixed_workflow_execute_code_block_observation,
    _normalize_fixed_workflow_runtime_context,
)
from system.fixed_workflow import (
    agent_weakness_key,
    build_fixed_collaborator_map,
    make_fixed_workflow_context_by_agent,
    make_fixed_workflow_metadata,
    normalize_fixed_workflow_cfg,
    remove_agent_registry_search_tool,
    resolve_fixed_workflow_targets,
    validate_fixed_workflow_compliance,
)
from system.reviewer_matching import match_papers_to_reviewers_random
from system.schema import AgentNetworkMetadata, AgentProfile, PaperMetadataPrivate
from system.trackers import ResearchStatsLog


def _agent(agent_id, *, reputation=None, accepted=0, citations=0):
    return AgentProfile(
        agent_id=agent_id,
        behavior="careful",
        expertise="biology",
        expertise_topics=["biology"],
        citation_count=citations,
        num_accepted_papers=accepted,
        reputation=({"reputation_score": reputation} if reputation is not None else None),
    )


def _paper(paper_id, author_id, collabs=None):
    return PaperMetadataPrivate(
        paper_id=paper_id,
        primary_agent_id=author_id,
        collab_agent_ids=list(collabs or []),
        title="T",
        abstract="A",
        manuscript="M",
        citation_count=0,
        publication_t=1,
        cited_paper_ids=[],
    )


class _Registry:
    def __init__(self, agents):
        self._agents = list(agents)

    def get_all_agents(self):
        return list(self._agents)


def test_fixed_workflow_assignment_uses_weakest_and_self_fallback():
    agents = [
        _agent("agent_01", reputation=3.0, accepted=2, citations=5),
        _agent("agent_02", reputation=1.0, accepted=0, citations=0),
        _agent("agent_03", reputation=2.0, accepted=1, citations=2),
    ]
    cfg = {"enabled": True, "assignment_policy": "weakest_agent"}

    assignments = build_fixed_collaborator_map(agents, iteration=3, seed=1, cfg=cfg)

    assert assignments["agent_01"] == "agent_02"
    assert assignments["agent_03"] == "agent_02"
    assert assignments["agent_02"] == "agent_03"


def test_fixed_workflow_assignment_is_seed_deterministic_for_ties():
    agents = [_agent("agent_01"), _agent("agent_02"), _agent("agent_03")]
    cfg = {"enabled": True, "assignment_policy": "weakest_agent"}

    a = build_fixed_collaborator_map(agents, iteration=1, seed=42, cfg=cfg)
    b = build_fixed_collaborator_map(agents, iteration=1, seed=42, cfg=cfg)

    assert a == b
    assert all(primary != collaborator for primary, collaborator in a.items())


def test_fixed_workflow_context_and_tool_policy():
    cfg = normalize_fixed_workflow_cfg({"enabled": True})
    contexts = make_fixed_workflow_context_by_agent(
        {"agent_01": "agent_02"},
        cfg=cfg,
        iteration=5,
    )
    tools = {"search_agent_registry": object(), "search_internal_archive": object()}

    removed = remove_agent_registry_search_tool(tools, cfg)

    assert contexts["agent_01"]["assigned_collaborator_id"] == "agent_02"
    assert removed is True
    assert "search_agent_registry" not in tools
    assert "search_internal_archive" in tools


def test_runtime_fixed_workflow_blocks_registry_and_pre_collab_code():
    ctx = _normalize_fixed_workflow_runtime_context(
        {"enabled": True, "assigned_collaborator_id": "agent_02"}
    )

    registry_obs = _fixed_workflow_block_observation("search_agent_registry", ctx)
    code_obs = _fixed_workflow_execute_code_block_observation({"collab_agent_ids": []}, ctx)
    code_after_collab = _fixed_workflow_execute_code_block_observation(
        {"collab_agent_ids": ["agent_02"]},
        ctx,
    )

    assert "disables agent-registry search" in registry_obs
    assert "requires collaboration before code execution" in code_obs
    assert code_after_collab is None


def test_validate_fixed_workflow_compliance_passes_and_fails():
    cfg = {"enabled": True, "require_collab_message": True, "min_collab_messages": 1}
    paper = _paper("p1", "agent_01", collabs=["agent_02"])
    log = ResearchStatsLog()
    log.add_event(
        kind="tool_result_establish_collaboration",
        agent_id="agent_01",
        iteration=1,
        step=1,
        collab_agent_id="agent_02",
    )
    log.add_event(
        kind="tool_call_communicate_with_collaborator",
        agent_id="agent_01",
        iteration=1,
        step=2,
        tool="communicate_with_collaborator",
    )
    log.add_event(
        kind="tool_result_communicate_with_collaborator",
        agent_id="agent_01",
        iteration=1,
        step=2,
        tool="communicate_with_collaborator",
    )
    log.add_event(
        kind="tool_call",
        agent_id="agent_01",
        iteration=1,
        step=3,
        tool="execute_code",
    )

    assert validate_fixed_workflow_compliance(paper, log, "agent_02", cfg) == []

    bad_log = ResearchStatsLog()
    bad_log.add_event(
        kind="tool_call",
        agent_id="agent_01",
        iteration=1,
        step=1,
        tool="search_agent_registry",
    )
    reasons = validate_fixed_workflow_compliance(_paper("p2", "agent_01"), bad_log, "agent_02", cfg)

    assert "assigned_collaborator_missing_from_paper" in reasons
    assert "assigned_collaboration_not_established" in reasons
    assert "agent_registry_search_used" in reasons


def test_one_collaborator_message_counts_once_not_twice():
    """A call + result pair is ONE interaction, so min=2 must fail."""
    paper = _paper("p1", "agent_01", collabs=["agent_02"])
    log = ResearchStatsLog()
    log.add_event(
        kind="tool_result_establish_collaboration",
        agent_id="agent_01",
        iteration=1,
        step=1,
        collab_agent_id="agent_02",
    )
    for kind in (
        "tool_call_communicate_with_collaborator",
        "tool_result_communicate_with_collaborator",
    ):
        log.add_event(kind=kind, agent_id="agent_01", iteration=1, step=2)

    strict = {"enabled": True, "require_collab_message": True, "min_collab_messages": 2}
    lenient = {"enabled": True, "require_collab_message": True, "min_collab_messages": 1}

    assert "insufficient_collaborator_interactions" in validate_fixed_workflow_compliance(
        paper, log, "agent_02", strict
    )
    assert "insufficient_collaborator_interactions" not in validate_fixed_workflow_compliance(
        paper, log, "agent_02", lenient
    )


def test_registry_search_is_only_a_violation_when_the_tool_was_removed():
    """Don't punish an agent for using a tool the run left available."""
    paper = _paper("p1", "agent_01", collabs=["agent_02"])
    log = ResearchStatsLog()
    log.add_event(
        kind="tool_result_establish_collaboration",
        agent_id="agent_01",
        iteration=1,
        step=1,
        collab_agent_id="agent_02",
    )
    log.add_event(
        kind="tool_result_search_agent_registry", agent_id="agent_01", iteration=1, step=2
    )

    base = {"enabled": True, "require_collab_message": False}

    disabled = validate_fixed_workflow_compliance(
        paper, log, "agent_02", {**base, "disable_agent_registry_tool": True}
    )
    allowed = validate_fixed_workflow_compliance(
        paper, log, "agent_02", {**base, "disable_agent_registry_tool": False}
    )

    assert "agent_registry_search_used" in disabled
    assert "agent_registry_search_used" not in allowed


def test_tie_break_agent_id_orders_ties_lexicographically():
    """`tie_break` must actually reach the sort key."""
    agents = [_agent("agent_b"), _agent("agent_a")]

    for seed in range(5):
        ordered = sorted(
            agents,
            key=lambda a: agent_weakness_key(a, seed=seed, iteration=1, tie_break="agent_id"),
        )
        assert [a.agent_id for a in ordered] == ["agent_a", "agent_b"]

    seeded_orders = {
        tuple(
            a.agent_id
            for a in sorted(
                agents,
                key=lambda a: agent_weakness_key(
                    a, seed=seed, iteration=1, tie_break="seeded_random"
                ),
            )
        )
        for seed in range(5)
    }
    assert len(seeded_orders) >= 2


def test_metadata_primary_target_matches_the_real_assignment():
    """Metadata must report the target that was actually assigned."""
    agents = [
        _agent("agent_01", reputation=3.0),
        _agent("agent_02", reputation=1.0),  # weakest overall, but paused
        _agent("agent_03", reputation=2.0),
    ]
    cfg = {"enabled": True, "assignment_policy": "weakest_agent"}

    assignments, primary = resolve_fixed_workflow_targets(
        agents, iteration=1, seed=1, paused_agent_ids=["agent_02"], cfg=cfg
    )
    metadata = make_fixed_workflow_metadata(
        iteration=1,
        seed=1,
        cfg=cfg,
        collaborator_map=assignments,
        agents=agents,
        primary_target_id=primary,
    )

    assert primary == "agent_03"
    assert assignments["agent_01"] == "agent_03"
    assert metadata["primary_target_id"] == "agent_03"
    # The weakness ranking across ALL agents stays available as a diagnostic.
    assert metadata["weakest_agent_id"] == "agent_02"


def test_two_agent_run_reports_a_deterministic_primary_target():
    """The case mode-inference over the map got wrong."""
    agents = [_agent("agent_01", reputation=5.0), _agent("agent_02", reputation=1.0)]
    cfg = {"enabled": True, "assignment_policy": "weakest_agent"}

    assignments, primary = resolve_fixed_workflow_targets(agents, iteration=1, seed=3, cfg=cfg)

    assert primary == "agent_02"
    assert assignments == {"agent_01": "agent_02", "agent_02": "agent_01"}


def test_random_reviewer_matching_is_deterministic_and_excludes_self():
    agents = [_agent("agent_01"), _agent("agent_02"), _agent("agent_03"), _agent("agent_04")]
    papers = [_paper("paper_a", "agent_01"), _paper("paper_b", "agent_02")]
    registry = _Registry(agents)

    first = match_papers_to_reviewers_random(
        num_reviewers_per_paper=2,
        papers=papers,
        agent_registry=registry,
        seed=7,
        iteration=2,
        seed_offset=11,
    )
    second = match_papers_to_reviewers_random(
        num_reviewers_per_paper=2,
        papers=papers,
        agent_registry=registry,
        seed=7,
        iteration=2,
        seed_offset=11,
    )

    assert first == second
    assert "agent_01" not in first["paper_a"]
    assert "agent_02" not in first["paper_b"]
    assert all(len(reviewers) == 2 for reviewers in first.values())


def test_primary_prompt_includes_fixed_workflow_requirement():
    prompt = make_primary_agent_system_prompt(
        agent_profile=_agent("agent_01"),
        agent_network_metadata=AgentNetworkMetadata(num_agents=3),
        iteration=2,
        tools={},
        fixed_workflow_context={
            "enabled": True,
            "assigned_collaborator_id": "agent_02",
        },
    )

    assert "Fixed Workflow Collaboration Requirement" in prompt
    assert "agent_02" in prompt
    assert "Do not call search_agent_registry" in prompt
