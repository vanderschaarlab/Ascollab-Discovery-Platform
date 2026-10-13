from system.research_one_step import (
    _build_collab_session_id,
    _build_paper_id,
    _build_research_session_id,
    _coerce_meta_gate_cfg,
    _evaluate_meta_quality_gate,
    _render_targeted_meta_revision,
    _validate_meta_research_report,
)


def test_phase_suffix_makes_unique_session_and_paper_ids():
    primary_session = _build_research_session_id("agent_01", 4, session_id_suffix=None)
    meta_session = _build_research_session_id("agent_01", 4, session_id_suffix="meta")
    primary_paper = _build_paper_id(iteration=4, agent_id="agent_01", paper_id_suffix=None)
    meta_paper = _build_paper_id(iteration=4, agent_id="agent_01", paper_id_suffix="meta")

    assert primary_session == "agent_01_t04"
    assert meta_session == "agent_01_t04_meta"
    assert primary_paper == "paper_t04_agent_01"
    assert meta_paper == "paper_t04_agent_01_meta"
    assert primary_session != meta_session
    assert primary_paper != meta_paper


def test_meta_report_validation_requires_section_and_archive_citations():
    report = "# Title\nX\n# Evidence Map\n..."
    verdict = {"valid_archive_ids": ["p1"]}
    errors = _validate_meta_research_report(
        report=report,
        citation_verdict=verdict,
        require_contradictions_section=True,
        require_archive_citations=True,
        min_archive_citations=2,
    )
    assert len(errors) == 2
    assert "Contradictions in Public Archive" in errors[0]
    assert "at least 2 valid [Archive: ...] citations" in errors[1]


def test_meta_quality_gate_flags_unsupported_bounded_claims():
    report = """
# Title
Meta Study
# Contradictions in Public Archive
- A
# Claim-Evidence Table
| Claim | Source Paper IDs | Evidence Type | Support Strength |
|---|---|---|---|
| C1 | [p1] | literature | moderate |
# Support and Validity Checks
- B
# Contradictions and Resolution
- C
# Evidence Map
- Claim C1 [Archive: p1]
# Bounded Final Claims
- Broad claim without archive citation.
# Executable Claim Checks
- C1 | [Archive: p1] | check_C1 | pass | [Plot: check_C1]
# Reproducibility Notes
- deterministic seeds used
"""
    gate = _evaluate_meta_quality_gate(
        report=report,
        citation_verdict={"valid_archive_ids": ["p1"]},
        validation_cfg={
            "require_contradictions_section": True,
            "require_archive_citations": True,
            "min_archive_citations": 1,
            "quality_contract": {
                "min_supported_major_claims": 1,
                "min_code_verified_major_claims": 0,
                "max_unsupported_major_claims": 0,
                "max_unresolved_uncertainty_claims": 0,
            },
            "validation_gate": {"enabled": True},
        },
    )
    assert gate["status"] == "failed"
    assert gate["unsupported_claim_count"] >= 1
    assert any(
        "unsupported major claims exceed threshold" in reason
        for reason in gate["final_fail_reasons"]
    )


def test_meta_revision_renderer_adds_missing_sections_and_bounded_claims():
    original = "# Title\nX\n# Contradictions in Public Archive\n- Y"
    revised = _render_targeted_meta_revision(
        report=original,
        citation_verdict={"valid_archive_ids": ["p1", "p2"]},
        validation_cfg={
            "min_archive_citations": 2,
            "quality_contract": {"min_supported_major_claims": 2},
        },
        fail_reasons=["missing required section '# Bounded Final Claims'"],
    )
    assert "# Claim-Evidence Table" in revised
    assert "# Support and Validity Checks" in revised
    assert "# Contradictions and Resolution" in revised
    assert "# Bounded Final Claims" in revised
    assert "# Executable Claim Checks" in revised
    assert "# Reproducibility Notes" in revised
    assert "[Archive: p1]" in revised or "[Archive: p2]" in revised


def test_meta_quality_gate_enforces_code_checks_for_bounded_claims():
    report = """
# Title
Meta Study
# Contradictions in Public Archive
- A
# Claim-Evidence Table
| Claim ID | Source Paper IDs | Evidence Type | Support Strength |
|---|---|---|---|
| C1 | [Archive: p1] | literature | moderate |
# Support and Validity Checks
- C1 supported by [Archive: p1]
# Contradictions and Resolution
- C
# Evidence Map
- C1 -> [Archive: p1]
# Bounded Final Claims
- C1: bounded claim [Archive: p1]
# Executable Claim Checks
| claim_id | evidence_refs | code_check_id | verification_status | artifact_refs |
|---|---|---|---|---|
| C1 | [Archive: p1] | check_C1 | not_run | [Plot: check_C1] |
# Reproducibility Notes
- deterministic seed
"""
    gate = _evaluate_meta_quality_gate(
        report=report,
        citation_verdict={"valid_archive_ids": ["p1"]},
        validation_cfg={
            "require_contradictions_section": True,
            "require_archive_citations": True,
            "min_archive_citations": 1,
            "quality_contract": {
                "min_supported_major_claims": 1,
                "min_code_verified_major_claims": 1,
                "max_unsupported_major_claims": 0,
                "max_unresolved_uncertainty_claims": 0,
            },
            "validation_gate": {"enabled": True},
        },
    )
    assert gate["status"] == "failed"
    assert any(
        "not_run code checks found in bounded final claims" in r for r in gate["final_fail_reasons"]
    )
    assert gate["code_verified_major_claim_count"] == 0


def test_meta_quality_gate_fails_when_bounded_claim_check_row_missing():
    report = """
# Title
Meta Study
# Contradictions in Public Archive
- A
# Claim-Evidence Table
| Claim ID | Source Paper IDs | Evidence Type | Support Strength |
|---|---|---|---|
| C1 | [Archive: p1] | literature | moderate |
| C2 | [Archive: p2] | literature | moderate |
# Support and Validity Checks
- C1 and C2 supported
# Contradictions and Resolution
- C
# Evidence Map
- C1 -> [Archive: p1]
- C2 -> [Archive: p2]
# Bounded Final Claims
- C1: bounded claim [Archive: p1]
- C2: bounded claim [Archive: p2]
# Executable Claim Checks
| claim_id | evidence_refs | code_check_id | verification_status | artifact_refs |
|---|---|---|---|---|
| C1 | [Archive: p1] | check_C1 | pass | [Plot: check_C1] |
# Reproducibility Notes
- deterministic seed
"""
    gate = _evaluate_meta_quality_gate(
        report=report,
        citation_verdict={"valid_archive_ids": ["p1", "p2"]},
        validation_cfg={
            "require_contradictions_section": True,
            "require_archive_citations": True,
            "min_archive_citations": 2,
            "quality_contract": {
                "min_supported_major_claims": 1,
                "min_code_verified_major_claims": 1,
                "max_unsupported_major_claims": 0,
                "max_unresolved_uncertainty_claims": 0,
            },
            "validation_gate": {"enabled": True},
        },
    )
    assert gate["status"] == "failed"
    assert any(
        "missing executable claim checks for bounded final claims" in r
        for r in gate["final_fail_reasons"]
    )


def test_meta_revision_renderer_compresses_report_when_over_word_budget():
    long_report = " ".join(f"token{i}" for i in range(1, 120))
    revised = _render_targeted_meta_revision(
        report=long_report,
        citation_verdict={"valid_archive_ids": []},
        validation_cfg={
            "quality_contract": {
                "require_claim_evidence_table": False,
                "require_support_checks_section": False,
                "require_contradiction_resolution_section": False,
                "require_bounded_final_claims_section": False,
                "require_executable_claim_checks_section": False,
                "require_reproducibility_notes_section": False,
                "max_words": 40,
            }
        },
        fail_reasons=["report exceeds max_words: 119 > 40"],
    )
    assert len(revised.split()) <= 41  # allow trailing ellipsis token


def test_coerce_meta_gate_cfg_supports_max_revision_attempts():
    quality_cfg, gate_cfg = _coerce_meta_gate_cfg(
        {
            "quality_contract": {"min_code_verified_major_claims": 2},
            "validation_gate": {
                "enabled": True,
                "allow_single_revision_pass": False,
                "max_revision_attempts": 3,
            },
        }
    )
    assert quality_cfg["min_code_verified_major_claims"] == 2
    assert gate_cfg["max_revision_attempts"] == 3


def test_collaborator_logs_never_reuse_the_collaborators_own_session_id():
    own = _build_research_session_id("agent_06", 2, session_id_suffix=None)
    collab = _build_collab_session_id("agent_06", "agent_07", 2, session_id_suffix=None)
    meta_collab = _build_collab_session_id("agent_06", "agent_07", 2, session_id_suffix="meta")

    assert own == "agent_06_t02"
    assert collab == "agent_06_t02_collab_for_agent_07"
    assert meta_collab == "agent_06_t02_meta_collab_for_agent_07"
