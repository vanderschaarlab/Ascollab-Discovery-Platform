from system.citation_tags import (
    parse_citation_tags,
    rewrite_plot_citation_tags,
    validate_citation_tags,
)


def test_parse_citation_tags_valid_examples():
    text = (
        "Finding A [Archive: paper_t02_agent_03]. "
        "Finding B [Arxiv: BRCA signaling | Smith et al. | 2024]. "
        "Finding C [Code: differential expression test]."
    )
    tags = parse_citation_tags(text)
    assert len(tags) == 3
    assert all(tag["valid"] for tag in tags)
    assert [tag["type"] for tag in tags] == ["archive", "arxiv", "code"]


def test_validate_citation_tags_invalid_arxiv_shape():
    text = "Unsupported tag [Arxiv: title | author]."
    verdict = validate_citation_tags(text)
    assert verdict["is_valid"] is False
    assert any("arxiv tag must be" in err for err in verdict["errors"])


def test_validate_citation_tags_invalid_code_word_count():
    text = "Tag [Code: this description has six words exactly]."
    verdict = validate_citation_tags(text)
    assert verdict["is_valid"] is False
    assert any("<= 5 words" in err for err in verdict["errors"])


def test_validate_citation_tags_require_at_least_one():
    verdict = validate_citation_tags("No tags here.", require_at_least_one=True)
    assert verdict["is_valid"] is False
    assert "at least one citation tag is required" in verdict["errors"]


def test_validate_citation_tags_archive_status_check():
    text = "Reference [Archive: paper_t03_agent_04]."
    verdict = validate_citation_tags(
        text,
        known_archive_status_by_id={"paper_t03_agent_04": "rejected"},
        require_accepted_archive_ids=True,
    )
    assert verdict["is_valid"] is False
    assert any("non-accepted" in err for err in verdict["errors"])


def test_unknown_archive_id_is_recorded_but_not_enforced_in_phase_1(caplog):
    """Fabricated ids are counted via a log line, not an error."""
    text = "Real [Archive: p1]. Fake [Archive: p_ghost]."

    with caplog.at_level("WARNING", logger="system.citation_tags"):
        verdict = validate_citation_tags(
            text,
            known_archive_status_by_id={"p1": "accepted"},
            require_accepted_archive_ids=True,
        )

    assert verdict["is_valid"] is True
    records = [r for r in caplog.records if "[citation][unknown_archive_ids]" in r.getMessage()]
    assert len(records) == 1
    assert "p_ghost" in records[0].getMessage()


def test_unknown_archive_id_is_an_error_in_phase_2():
    text = "Real [Archive: p1]. Fake [Archive: p_ghost]."

    verdict = validate_citation_tags(
        text,
        known_archive_status_by_id={"p1": "accepted"},
        require_accepted_archive_ids=True,
        unknown_archive_ids_are_errors=True,
    )

    assert verdict["is_valid"] is False
    assert len([err for err in verdict["errors"] if "p_ghost" in err]) == 1
    assert not any("p1]" in err for err in verdict["errors"])


def test_empty_archive_still_runs_the_unknown_id_check(caplog):
    """The check used to be skipped entirely on an empty archive."""
    with caplog.at_level("WARNING", logger="system.citation_tags"):
        lenient = validate_citation_tags(
            "Fabricated [Archive: p_ghost].",
            known_archive_status_by_id={},
            require_accepted_archive_ids=True,
        )
    strict = validate_citation_tags(
        "Fabricated [Archive: p_ghost].",
        known_archive_status_by_id={},
        require_accepted_archive_ids=True,
        unknown_archive_ids_are_errors=True,
    )

    assert lenient["is_valid"] is True
    assert any("[citation][unknown_archive_ids]" in r.getMessage() for r in caplog.records)
    assert strict["is_valid"] is False


def test_validate_citation_tags_code_evidence_requires_successful_exec_when_enabled():
    text = "Empirical support [Code: pilot regression]."
    verdict = validate_citation_tags(
        text,
        require_backed_code_tags=True,
        successful_code_executions_current_iteration=0,
        code_script="print('no run')",
    )
    assert verdict["is_valid"] is False
    assert any("successful execute_code run" in err for err in verdict["errors"])


def test_validate_citation_tags_code_evidence_requires_script_when_enabled():
    text = "Empirical support [Code: pilot regression]."
    verdict = validate_citation_tags(
        text,
        require_backed_code_tags=True,
        successful_code_executions_current_iteration=1,
        code_script="",
    )
    assert verdict["is_valid"] is False
    assert any("non-empty code_script" in err for err in verdict["errors"])


def test_validate_citation_tags_code_evidence_backed_when_enabled():
    text = "Empirical support [Code: pilot regression]."
    verdict = validate_citation_tags(
        text,
        require_backed_code_tags=True,
        successful_code_executions_current_iteration=2,
        code_script="print('ran')",
    )
    assert verdict["is_valid"] is True
    code_evidence = verdict.get("code_evidence", {})
    assert code_evidence.get("code_tag_count") == 1
    assert code_evidence.get("is_backed") is True


def test_parse_citation_tags_valid_plot_tag():
    text = "Figure evidence [Plot: agent_01_sid-sid_job-job_cell-2_step-5_img-01.png]."
    tags = parse_citation_tags(text)
    assert len(tags) == 1
    assert tags[0]["type"] == "plot"
    assert tags[0]["valid"] is True
    assert tags[0]["plot_filename"] == "agent_01_sid-sid_job-job_cell-2_step-5_img-01.png"


def test_validate_citation_tags_rejects_unknown_plot_filename_when_required():
    text = "Figure evidence [Plot: unknown_plot.png]."
    verdict = validate_citation_tags(
        text,
        known_plot_filenames=["agent_01_sid-sid_job-job_cell-2_step-5_img-01.png"],
        require_known_plot_filenames=True,
    )
    assert verdict["is_valid"] is False
    assert any("unknown plot citation token" in err for err in verdict["errors"])


def test_validate_citation_tags_rejects_legacy_canonical_plot_token_after_cutover():
    text = "Figure evidence [Plot: plot_t03_agent_01_s02_i01]."
    verdict = validate_citation_tags(
        text,
        known_plot_filenames=["agent_01_sid-sid_job-job_cell-2_step-5_img-01.png"],
        require_known_plot_filenames=True,
    )
    assert verdict["is_valid"] is False
    assert any("unknown plot citation token" in err for err in verdict["errors"])


def test_parse_citation_tags_normalizes_plot_filename_from_path_and_quotes():
    text = "Figure evidence [Plot: `plots\\\\agent_01_sid-sid_job-job_cell-2_step-5_img-01.png`]."
    tags = parse_citation_tags(text)
    assert len(tags) == 1
    assert tags[0]["valid"] is True
    assert tags[0]["plot_filename"] == "agent_01_sid-sid_job-job_cell-2_step-5_img-01.png"


def test_validate_citation_tags_accepts_normalized_plot_filename_variants():
    text = "Figure evidence [Plot: plots/agent_01_sid-sid_job-job_cell-2_step-5_img-01.png.]."
    verdict = validate_citation_tags(
        text,
        known_plot_filenames=["agent_01_sid-sid_job-job_cell-2_step-5_img-01.png"],
        require_known_plot_filenames=True,
    )
    assert verdict["is_valid"] is True
    assert verdict["valid_plot_filenames"] == ["agent_01_sid-sid_job-job_cell-2_step-5_img-01.png"]


def test_validate_citation_tags_requires_at_least_one_plot_tag_when_enabled():
    verdict = validate_citation_tags(
        "No plot citation here.",
        require_at_least_one_plot_tag=True,
    )
    assert verdict["is_valid"] is False
    assert any(
        "at least one [Plot: ...] citation tag is required" in err for err in verdict["errors"]
    )


def test_validate_citation_tags_accepts_plot_handles_with_mapping():
    text = "Figure evidence [Plot: plot_ab12cd34]."
    verdict = validate_citation_tags(
        text,
        known_plot_filenames=["agent_01_sid-sid_job-job_cell-2_step-5_img-01.png"],
        known_plot_handles=["plot_ab12cd34"],
        plot_handle_to_filename={
            "plot_ab12cd34": "agent_01_sid-sid_job-job_cell-2_step-5_img-01.png"
        },
        require_known_plot_filenames=True,
    )
    assert verdict["is_valid"] is True
    assert verdict["valid_plot_handles"] == ["plot_ab12cd34"]
    assert verdict["valid_plot_filenames"] == ["agent_01_sid-sid_job-job_cell-2_step-5_img-01.png"]


def test_rewrite_plot_citation_tags_prefers_handle_tokens():
    original = "Figure evidence [Plot: plots/agent_01_sid-sid_job-job_cell-2_step-5_img-01.png]."
    rewritten = rewrite_plot_citation_tags(
        original,
        known_plot_filenames=["agent_01_sid-sid_job-job_cell-2_step-5_img-01.png"],
        known_plot_handles=["plot_ab12cd34"],
        plot_handle_to_filename={
            "plot_ab12cd34": "agent_01_sid-sid_job-job_cell-2_step-5_img-01.png"
        },
        prefer_plot_handles=True,
    )
    assert rewritten["changed"] is True
    assert "[Plot: plot_ab12cd34]" in rewritten["text"]
    assert rewritten["replacements"]
