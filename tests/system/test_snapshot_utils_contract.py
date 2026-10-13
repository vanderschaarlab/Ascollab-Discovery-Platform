from pathlib import Path

from system.snapshot_utils import (
    _build_ranker_pool_records,
    _flatten_private_archive_papers,
    _redact_api_keys,
    _to_plain,
    _write_jsonl,
)


def test_redact_api_keys_recurses_through_mappings_and_lists():
    redacted = _redact_api_keys(
        {"api_key": "secret", "nested": [{"api_key": "secret2", "keep": "value"}]}
    )

    assert redacted == {
        "api_key": "REDACTED",
        "nested": [{"api_key": "REDACTED", "keep": "value"}],
    }


def test_flatten_private_archive_papers_supports_legacy_nested_shape():
    flat = _flatten_private_archive_papers(
        {"papers": [[{"paper_id": "p1"}], {"paper_id": "p2"}, "ignored"]}
    )

    assert flat == [{"paper_id": "p1"}, {"paper_id": "p2"}]


def test_build_ranker_pool_records_filters_rejected_timeout_and_meta():
    records = _build_ranker_pool_records(
        private_archive_data={
            "papers": [
                {"paper_id": "accepted", "status": "accepted", "publication_t": 2},
                {"paper_id": "rejected", "status": "rejected", "publication_t": 1},
                {"paper_id": "timeout_timeout", "status": "accepted", "publication_t": 3},
                {
                    "paper_id": "meta",
                    "status": "accepted",
                    "publication_t": 4,
                    "paper_kind": "meta_research",
                },
            ]
        },
        include_rejected=False,
        exclude_timeout=True,
    )

    assert [paper["paper_id"] for paper in records["all"]] == ["accepted", "meta"]
    assert [paper["paper_id"] for paper in records["standard"]] == ["accepted"]
    assert [paper["paper_id"] for paper in records["meta"]] == ["meta"]


def test_write_jsonl_serializes_dataclasses(tmp_path, private_paper_factory):
    path = tmp_path / "rows.jsonl"

    _write_jsonl(path, [{"paper": private_paper_factory(paper_id="p1")}])

    text = path.read_text()
    assert '"paper_id": "p1"' in text
    assert "Path" in _to_plain(Path("x"))
