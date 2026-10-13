"""PaperArchive.prune_older_than semantics (Markovian memory, experiment 08)."""

from stores.paper_archive import PaperArchive
from system.schema import PaperMetadata


def _paper(pid, agent, t, cit=0):
    return PaperMetadata(
        paper_id=pid,
        primary_agent_id=agent,
        collab_agent_ids=[],
        title=f"T{pid}",
        abstract=f"abstract about topic {pid}",
        manuscript="m",
        citation_count=cit,
        publication_t=t,
        cited_paper_ids=[],
    )


def _archive(fake_embeddings):
    arc = PaperArchive(fake_embeddings)
    arc.add(
        [
            _paper("p1", "a1", 1, cit=5),
            _paper("p2", "a1", 1, cit=2),
            _paper("p3", "a2", 2, cit=1),
            _paper("p4", "a2", 3, cit=0),
        ]
    )
    return arc


def test_prune_removes_older_and_returns_ids(fake_embeddings):
    arc = _archive(fake_embeddings)
    dropped = arc.prune_older_than(2)  # keep publication_t >= 2
    assert sorted(dropped) == ["p1", "p2"]
    remaining = {p.paper_id for p in arc._entries.values()}
    assert remaining == {"p3", "p4"}
    assert arc._index.ntotal == 2


def test_prune_is_owner_agnostic(fake_embeddings):
    arc = _archive(fake_embeddings)
    # a1 authored both old papers; both are dropped even though they are the owner's.
    arc.prune_older_than(2)
    assert arc.get_most_recent_by_agent("a1") == []
    assert [p.paper_id for p in arc.get_most_recent_by_agent("a2")] == ["p4", "p3"]


def test_prune_recomputes_max_citations_and_keeps_ids_monotonic(fake_embeddings):
    arc = _archive(fake_embeddings)
    arc.prune_older_than(2)
    # survivors p3(cit=1), p4(cit=0)
    assert arc._max_citations_seen == 1
    next_id_before = arc._next_faiss_id
    arc.add([_paper("p5", "a3", 4, cit=9)])
    assert arc._next_faiss_id == next_id_before + 1
    ids = {p.paper_id for p, _ in arc.query_semantic("topic", k=5)}
    assert "p5" in ids


def test_query_and_recent_return_only_survivors(fake_embeddings):
    arc = _archive(fake_embeddings)
    arc.prune_older_than(2)
    found = {p.paper_id for p, _ in arc.query_semantic("topic", k=5)}
    assert found <= {"p3", "p4"}


def test_prune_to_empty_is_well_defined(fake_embeddings):
    arc = _archive(fake_embeddings)
    arc.prune_older_than(999)  # drop everything
    assert arc._entries == {}
    assert arc._index is None
    assert arc.query_semantic("anything", k=3) == []
    assert arc.get_most_recent_by_agent("a1") == []
    # Re-adding after emptying still works.
    arc.add([_paper("p9", "a9", 5, cit=0)])
    assert [p.paper_id for p, _ in arc.query_semantic("topic", k=3)] == ["p9"]


def test_prune_empty_archive_no_crash(fake_embeddings):
    assert PaperArchive(fake_embeddings).prune_older_than(1) == []


def test_prune_noop_when_nothing_older(fake_embeddings):
    arc = _archive(fake_embeddings)
    assert arc.prune_older_than(1) == []  # all publication_t >= 1
    assert arc._index.ntotal == 4
