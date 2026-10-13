"""Regression: replaying an iteration must not drop papers from the archive."""

import logging

from stores.agent_registry import AgentRegistry
from stores.paper_archive import PaperArchive
from stores.private_paper_archive import PrivatePaperArchive
from system.update_utils import update_status_and_archives


def _run(meta_reviews, papers, archive, private_archive, registry):
    return update_status_and_archives(
        meta_reviews=meta_reviews,
        candidate_papers=papers,
        archive=archive,
        private_archive=private_archive,
        agent_registry=registry,
        iteration=1,
    )


def test_replayed_iteration_keeps_one_copy_and_loses_nothing(
    fake_embeddings, private_paper_factory, agent_profile_factory, metareview_factory
):
    archive = PaperArchive(fake_embeddings)
    private_archive = PrivatePaperArchive()
    registry = AgentRegistry(fake_embeddings, [agent_profile_factory(agent_id="agent_01")])

    papers = [
        private_paper_factory(paper_id="p1", abstract="first paper"),
        private_paper_factory(paper_id="p2", abstract="second paper"),
        private_paper_factory(paper_id="p3", abstract="third paper"),
    ]
    meta_reviews = [metareview_factory(paper_id=pid) for pid in ("p1", "p2", "p3")]

    _run(meta_reviews, papers, archive, private_archive, registry)
    # Second, identical pass: every paper is now a duplicate.
    _run(meta_reviews, papers, archive, private_archive, registry)

    stored = [p.paper_id for p in private_archive.get_by_primary_agent("agent_01", k=None)]
    assert sorted(stored) == ["p1", "p2", "p3"]


def test_missing_meta_reviews_leave_a_loud_warning(
    fake_embeddings, private_paper_factory, agent_profile_factory, caplog
):
    """The dead score-based fallback is gone; the gap must be visible."""
    caplog.set_level(logging.DEBUG)
    archive = PaperArchive(fake_embeddings)
    private_archive = PrivatePaperArchive()
    registry = AgentRegistry(fake_embeddings, [agent_profile_factory(agent_id="agent_01")])

    _run([], [private_paper_factory(paper_id="p1")], archive, private_archive, registry)

    out = caplog.text
    assert "1 candidate paper(s) have NO meta-review" in out
    assert "'p1'" in out


def test_a_single_duplicate_does_not_drop_later_papers(
    fake_embeddings, private_paper_factory, agent_profile_factory, metareview_factory
):
    archive = PaperArchive(fake_embeddings)
    # Pre-seed the private archive with the middle paper only, so the dedupe
    # pre-filter has to be the thing that spots it.
    private_archive = PrivatePaperArchive([private_paper_factory(paper_id="p2")])
    registry = AgentRegistry(fake_embeddings, [agent_profile_factory(agent_id="agent_01")])

    papers = [
        private_paper_factory(paper_id="p1", abstract="first paper"),
        private_paper_factory(paper_id="p2", abstract="second paper"),
        private_paper_factory(paper_id="p3", abstract="third paper"),
    ]
    meta_reviews = [metareview_factory(paper_id=pid) for pid in ("p1", "p2", "p3")]

    _run(meta_reviews, papers, archive, private_archive, registry)

    stored = [p.paper_id for p in private_archive.get_by_primary_agent("agent_01", k=None)]
    assert sorted(stored) == ["p1", "p2", "p3"]
