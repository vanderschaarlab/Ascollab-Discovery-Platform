import pytest
from langchain_core.embeddings import Embeddings

from stores.paper_archive import PaperArchive
from stores.private_paper_archive import PrivatePaperArchive


class _FixedEmbeddings(Embeddings):
    """Embedder returning hand-picked unit vectors, including a negative one."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.vectors[text] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.vectors[text]


def test_paper_archive_returns_most_recent_for_agent(fake_embeddings, paper_factory):
    archive = PaperArchive(fake_embeddings)
    archive.add(
        [
            paper_factory(paper_id="old", publication_t=1),
            paper_factory(paper_id="new", publication_t=3),
            paper_factory(paper_id="mid", publication_t=2),
        ]
    )

    papers = archive.get_most_recent_by_agent("agent_01", k=2)

    assert [paper.paper_id for paper in papers] == ["new", "mid"]


def test_paper_archive_rejects_dimension_mismatch(fake_embeddings, paper_factory):
    archive = PaperArchive(fake_embeddings)
    archive.add(paper_factory(paper_id="p1"))
    fake_embeddings.dim = 4

    with pytest.raises(AssertionError, match="Embedding dimension mismatch"):
        archive.add(paper_factory(paper_id="p2"))


def test_paper_archive_updates_citations_monotonically(fake_embeddings, paper_factory):
    archive = PaperArchive(fake_embeddings)
    archive.add(paper_factory(citation_count=2))

    archive.update_entry("agent_01", "paper_01", 5)

    assert archive.get_most_recent_by_agent("agent_01")[0].citation_count == 5
    with pytest.raises(ValueError, match="cannot be less"):
        archive.update_entry("agent_01", "paper_01", 4)
    with pytest.raises(ValueError, match="Paper not found"):
        archive.update_entry("agent_02", "missing", 1)


def test_paper_archive_boosted_query_prefers_higher_combined_score(fake_embeddings, paper_factory):
    archive = PaperArchive(
        fake_embeddings,
        with_citation_boost=True,
        w_citation=10.0,
        max_k=5,
    )
    archive.add(
        [
            paper_factory(paper_id="low_cite", abstract="same topic", citation_count=0),
            paper_factory(paper_id="high_cite", abstract="same topic", citation_count=100),
        ]
    )

    results = archive.query_semantic("same topic", k=1)

    assert results[0][0].paper_id == "high_cite"


def test_boosted_query_keeps_scores_aligned_with_papers(paper_factory):
    """A negative inner product is a valid neighbour, not a filter signal."""
    embeddings = _FixedEmbeddings(
        {
            "query": [1.0, 0.0],
            "near": [1.0, 0.0],
            "mid": [0.5, 0.8660254],
            "opposed": [-0.6, 0.8],
        }
    )
    archive = PaperArchive(
        embeddings, with_citation_boost=True, w_citation=1.0, w_recency=0.0, max_k=5
    )
    archive.add(
        [
            paper_factory(paper_id="near", abstract="near", citation_count=0),
            paper_factory(paper_id="mid", abstract="mid", citation_count=100),
            paper_factory(paper_id="opposed", abstract="opposed", citation_count=50),
        ]
    )

    results = archive.query_semantic("query", k=3)

    # sims_norm = [1.0, 0.6875, 0.0]; cit_norm = [0.0, 1.0, ~0.852]
    assert [paper.paper_id for paper, _ in results] == ["mid", "near", "opposed"]
    by_id = {paper.paper_id: score for paper, score in results}
    assert by_id["mid"] == pytest.approx(1.6875, abs=1e-3)
    assert by_id["near"] == pytest.approx(1.0, abs=1e-3)
    assert by_id["opposed"] == pytest.approx(0.852, abs=1e-2)


def test_query_semantic_with_similarity_reports_the_raw_inner_product(paper_factory):
    """Metric code needs the similarity, not the boosted ranking score."""
    embeddings = _FixedEmbeddings({"query": [1.0, 0.0], "near": [1.0, 0.0], "opposed": [-0.6, 0.8]})
    archive = PaperArchive(embeddings, with_citation_boost=True, w_citation=1.0, max_k=5)
    archive.add(
        [
            paper_factory(paper_id="near", abstract="near", citation_count=0),
            paper_factory(paper_id="opposed", abstract="opposed", citation_count=9),
        ]
    )

    results = archive.query_semantic_with_similarity("query", k=2)

    sims = {paper.paper_id: raw for paper, _score, raw in results}
    assert sims["near"] == pytest.approx(1.0, abs=1e-5)
    assert sims["opposed"] == pytest.approx(-0.6, abs=1e-5)


def test_query_semantic_handles_k_larger_than_archive_and_none(fake_embeddings, paper_factory):
    """k=None must not raise, and k above ntotal must not pad with sentinels."""
    archive = PaperArchive(fake_embeddings, with_citation_boost=True, max_k=10)
    archive.add([paper_factory(paper_id="only", abstract="only paper")])

    assert len(archive.query_semantic("only paper", k=8)) == 1
    assert len(archive.query_semantic("only paper", k=None)) == 1
    assert archive.query_semantic("only paper", k=0) == []


def test_private_archive_add_skips_duplicates_without_dropping_the_batch(private_paper_factory):
    """One duplicate must not swallow every later paper in the batch."""
    archive = PrivatePaperArchive([private_paper_factory(paper_id="dup")])

    skipped = archive.add(
        [
            private_paper_factory(paper_id="first"),
            private_paper_factory(paper_id="dup", primary_agent_id="agent_02"),
            private_paper_factory(paper_id="third"),
        ]
    )

    assert skipped == ["dup"]
    stored = {paper.paper_id for paper in archive.get_by_primary_agent("agent_01", k=None)}
    assert stored == {"dup", "first", "third"}
    # Skip means skip: the stored copy keeps its original owner.
    assert archive.get_by_primary_agent("agent_02", k=None) == []


def test_private_archive_add_can_still_raise_on_duplicates(private_paper_factory):
    archive = PrivatePaperArchive([private_paper_factory(paper_id="dup")])

    with pytest.raises(ValueError, match="already exists"):
        archive.add([private_paper_factory(paper_id="dup")], on_duplicate="raise")


def test_private_archive_enforces_identity_and_review_records(private_paper_factory):
    archive = PrivatePaperArchive()
    paper = private_paper_factory(paper_id="paper_a", citation_count=1)
    archive.add(paper)

    with pytest.raises(ValueError, match="already exists"):
        archive.add(
            private_paper_factory(paper_id="paper_a", primary_agent_id="agent_02"),
            on_duplicate="raise",
        )
    with pytest.raises(ValueError, match="negative"):
        archive.add(private_paper_factory(paper_id="paper_b", citation_count=-1))

    archive.add_reviews_for_iteration(
        3,
        {
            "paper_a": [
                {
                    "reviewer_id": "reviewer_01",
                    "review_text": "Good.",
                    "scores": {"overall": 0.8},
                }
            ]
        },
    )

    stored = archive.reviews_by_paper["paper_a"][0]
    assert stored == {
        "iteration": 3,
        "reviewer_id": "reviewer_01",
        "review_text": "Good.",
        "scores": {"overall": 0.8},
    }


def test_private_archive_assignments_and_ordering(private_paper_factory):
    archive = PrivatePaperArchive(
        [
            private_paper_factory(paper_id="p2", publication_t=2),
            private_paper_factory(paper_id="p1", publication_t=1),
        ]
    )
    archive.add_review_assignment_record("assign_1", {"paper_id": "p1"})
    archive.set_review_assignments({"assign_2": {"paper_id": "p2"}})

    assert list(archive.review_assignments) == ["assign_2"]
    assert [paper.paper_id for paper in archive.get_by_primary_agent("agent_01", k=None)] == [
        "p1",
        "p2",
    ]
