from system.review_one_step import _build_assignment_metadata
from system.schema import PaperMetadataPrivate


def _paper(pid, author):
    return PaperMetadataPrivate(
        paper_id=pid,
        primary_agent_id=author,
        collab_agent_ids=[],
        title=pid,
        abstract="a",
        manuscript="m",
        citation_count=0,
        publication_t=1,
        cited_paper_ids=[],
    )


def test_assignment_metadata_counts_load_and_self_review_violations():
    papers = {"p1": _paper("p1", "a1"), "p2": _paper("p2", "a2")}
    metadata = _build_assignment_metadata(
        strategy="random",
        seed=5,
        condition_label="condition_b",
        num_reviewers_per_paper=2,
        allow_self_review=False,
        excluded_reviewer_ids={"a4"},
        assignments={"p1": ["a1", "a3"], "p2": ["a3"]},
        papers_by_id=papers,
    )

    assert metadata["strategy"] == "random"
    assert metadata["diagnostics"]["reviewer_load"] == {"a1": 1, "a3": 2}
    assert metadata["diagnostics"]["num_self_review_violations"] == 1
    assert metadata["diagnostics"]["num_underfilled_papers"] == 1
    assert metadata["excluded_reviewer_ids"] == ["a4"]
