from types import SimpleNamespace

from system.meta_researcher import (
    build_meta_phase_context_by_agent,
    build_meta_source_cards,
    evaluate_meta_research_trigger,
    normalize_meta_researcher_cfg,
    select_meta_research_agent_ids,
)


def _paper(
    paper_id: str,
    *,
    publication_t: int,
    citation_count: int,
    publication_status: str = "accepted",
) -> SimpleNamespace:
    return SimpleNamespace(
        paper_id=paper_id,
        title=f"title-{paper_id}",
        abstract=f"abstract-{paper_id}",
        manuscript=f"manuscript-{paper_id}",
        publication_t=publication_t,
        citation_count=citation_count,
        publication_status=publication_status,
    )


def test_trigger_schedule_respects_start_and_every_n():
    cfg = normalize_meta_researcher_cfg(
        {
            "enabled": True,
            "start_iteration": 3,
            "every_n_iterations": 3,
        }
    )
    t2 = evaluate_meta_research_trigger(iteration=2, cfg=cfg)
    t3 = evaluate_meta_research_trigger(iteration=3, cfg=cfg)
    t6 = evaluate_meta_research_trigger(iteration=6, cfg=cfg)
    t7 = evaluate_meta_research_trigger(iteration=7, cfg=cfg)

    assert t2["triggered"] is False
    assert t2["reason"] == "before_start_iteration"
    assert t3["triggered"] is True
    assert t6["triggered"] is True
    assert t7["triggered"] is False
    assert t7["reason"] == "not_scheduled"


def test_select_agents_top_k_reputation():
    agents = [
        SimpleNamespace(agent_id="agent_a", reputation={"reputation_score": 0.20}),
        SimpleNamespace(agent_id="agent_b", reputation={"reputation_score": 0.85}),
        SimpleNamespace(agent_id="agent_c", reputation={"reputation_score": 0.55}),
    ]
    cfg = normalize_meta_researcher_cfg(
        {
            "enabled": True,
            "agent_scope": {"mode": "top_k_reputation", "top_k": 2},
        }
    )
    selected = select_meta_research_agent_ids(
        agents=agents,
        active_agent_ids=["agent_a", "agent_b", "agent_c"],
        cfg=cfg,
    )
    assert selected == ["agent_b", "agent_c"]


def test_source_cards_filter_accepted_sort_and_cap():
    archive = SimpleNamespace(
        _entries={
            0: _paper("p_old", publication_t=2, citation_count=50),
            1: _paper("p_new_low_cit", publication_t=5, citation_count=1),
            2: _paper("p_new_high_cit", publication_t=5, citation_count=10),
            3: _paper(
                "p_rejected", publication_t=100, citation_count=100, publication_status="rejected"
            ),
        }
    )
    cfg = normalize_meta_researcher_cfg(
        {
            "enabled": True,
            "source_corpus": {
                "max_papers": 2,
                "max_chars_per_paper": 50,
                "sort_by": "recency_then_citations",
            },
        }
    )
    cards = build_meta_source_cards(public_archive=archive, cfg=cfg)
    ids = [card["paper_id"] for card in cards]
    assert ids == ["p_new_high_cit", "p_new_low_cit"]
    assert "p_rejected" not in ids


def test_phase_context_contains_validation_and_source_ids():
    cfg = normalize_meta_researcher_cfg(
        {
            "enabled": True,
            "output": {
                "require_contradictions_section": True,
                "require_archive_citations": True,
                "min_archive_citations": 3,
            },
        }
    )
    source_cards = [
        {"paper_id": "p1", "title": "t1"},
        {"paper_id": "p2", "title": "t2"},
    ]
    ctx = build_meta_phase_context_by_agent(
        agent_ids=["agent_01", "agent_02"],
        source_cards=source_cards,
        iteration=4,
        cfg=cfg,
    )
    assert sorted(ctx.keys()) == ["agent_01", "agent_02"]
    assert ctx["agent_01"]["meta_research_mode"] is True
    assert ctx["agent_01"]["meta_research_context"]["source_paper_ids"] == ["p1", "p2"]
    assert ctx["agent_01"]["meta_validation"]["require_contradictions_section"] is True
    assert ctx["agent_01"]["meta_validation"]["min_archive_citations"] == 3
    assert ctx["agent_01"]["meta_validation"]["quality_contract"]["min_supported_major_claims"] == 3
    assert (
        ctx["agent_01"]["meta_validation"]["quality_contract"]["min_code_verified_major_claims"]
        == 1
    )
    assert ctx["agent_01"]["meta_validation"]["validation_gate"]["max_revision_attempts"] == 1
    assert "evidence_graph" in ctx["agent_01"]["meta_research_context"]


def test_normalize_cfg_includes_quality_contract_validation_gate_and_telemetry_defaults():
    cfg = normalize_meta_researcher_cfg({"enabled": True})
    assert cfg["quality_contract"]["require_claim_evidence_table"] is True
    assert cfg["quality_contract"]["require_executable_claim_checks_section"] is True
    assert cfg["quality_contract"]["require_reproducibility_notes_section"] is True
    assert cfg["quality_contract"]["min_code_verified_major_claims"] == 1
    assert cfg["quality_contract"]["max_words"] == 950
    assert cfg["validation_gate"]["enabled"] is True
    assert cfg["validation_gate"]["allow_single_revision_pass"] is True
    assert cfg["validation_gate"]["max_revision_attempts"] == 1
    assert cfg["telemetry"]["write_meta_quality_report"] is True
    assert cfg["telemetry"]["write_evidence_map"] is True
    assert cfg["telemetry"]["write_meta_claim_checks"] is True
    assert cfg["telemetry"]["write_citation_lint"] is True
    assert cfg["telemetry"]["write_ranker_pools"] is True
    assert cfg["runtime"]["code_collaboration"] is True
    assert cfg["runtime"]["disable_collab_tools"] is False
