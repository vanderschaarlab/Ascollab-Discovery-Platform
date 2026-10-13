from system.validation import validate_paper_payload


def _valid_payload():
    return {
        "title": "Validated RNA survival biomarkers in KIRC cohorts",
        "abstract": "A" * 140,
        "manuscript": (
            "Methods\n"
            + "We analyzed the allowed dataset with cross-validation and adjusted models. "
            + "Results show calibrated survival associations with confidence intervals. "
            + "Findings are replicated across folds. "
            + "M" * 760
        ),
        "datasets_used": "tcga_kirc_neurips_hiseqv2",
        "citations": [{"title": "Known evidence", "id": "paper_01"}],
    }


def test_validate_paper_payload_normalizes_valid_payload():
    payload, errors = validate_paper_payload(
        payload_or_text=_valid_payload(),
        allowed_dataset_keys=["tcga_kirc_neurips_hiseqv2"],
        allowed_citation_ids=["paper_01"],
    )

    assert errors == []
    assert payload["datasets_used"] == ["tcga_kirc_neurips_hiseqv2"]
    assert payload["citations"] == ["Known evidence paper_01"]


def test_validate_paper_payload_extracts_json_from_fenced_text():
    payload = _valid_payload()
    text = "prefix\n```json\n" + __import__("json").dumps(payload) + "\n```\nsuffix"

    parsed, errors = validate_paper_payload(
        payload_or_text=text,
        allowed_dataset_keys=["tcga_kirc_neurips_hiseqv2"],
    )

    assert errors == []
    assert parsed["title"] == payload["title"]


def test_validate_paper_payload_rejects_unknown_dataset_and_access_excuse():
    payload = _valid_payload()
    payload["datasets_used"] = ["unknown"]
    payload["manuscript"] = payload["manuscript"] + " We had no direct access to the data."

    _parsed, errors = validate_paper_payload(
        payload_or_text=payload,
        allowed_dataset_keys=["tcga_kirc_neurips_hiseqv2"],
    )

    assert any("unknown keys" in err for err in errors)
    assert any("no/indirect dataset access" in err for err in errors)


def test_validate_paper_payload_reports_parse_failure():
    payload, errors = validate_paper_payload(
        payload_or_text="not json",
        allowed_dataset_keys=["tcga_kirc_neurips_hiseqv2"],
    )

    assert payload == {}
    assert errors == ["Could not parse JSON object from model output"]
