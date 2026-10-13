"""Experiment 08: a citation to a paper pruned from the public archive still counts in the private archive."""

import logging

from stores.agent_registry import AgentRegistry
from stores.paper_archive import PaperArchive
from stores.private_paper_archive import PrivatePaperArchive
from system.update_utils import update_status_and_archives


def _cite(private_paper_factory, cited_id, cited_agent):
    return private_paper_factory(
        paper_id="citing",
        primary_agent_id="agent_02",
        abstract="builds on earlier work",
        cited_paper_ids=[{"paper_id": cited_id, "agent_id": cited_agent}],
    )


def _setup(fake_embeddings, private_paper_factory, paper_factory, agent_profile_factory):
    archive = PaperArchive(fake_embeddings)
    archive.add(paper_factory(paper_id="old", primary_agent_id="agent_01", citation_count=1))
    private_archive = PrivatePaperArchive(
        [
            private_paper_factory(
                paper_id="old", primary_agent_id="agent_01", citation_count=1, status="accepted"
            ),
            private_paper_factory(
                paper_id="rejected",
                primary_agent_id="agent_01",
                citation_count=0,
                status="rejected",
            ),
        ]
    )
    registry = AgentRegistry(
        fake_embeddings,
        [
            agent_profile_factory(agent_id="agent_01", citation_count=1),
            agent_profile_factory(agent_id="agent_02", citation_count=0),
        ],
    )
    return archive, private_archive, registry


def _accept(paper, archive, private_archive, registry, metareview_factory):
    update_status_and_archives(
        meta_reviews=[metareview_factory(paper_id=paper.paper_id)],
        candidate_papers=[paper],
        archive=archive,
        private_archive=private_archive,
        agent_registry=registry,
        iteration=3,
    )


def _private_count(private_archive, agent_id, paper_id):
    papers = private_archive.get_by_primary_agent(agent_id, k=None)
    return next(p.citation_count for p in papers if p.paper_id == paper_id)


def test_citation_to_a_public_paper_updates_both_archives(
    fake_embeddings, private_paper_factory, paper_factory, agent_profile_factory, metareview_factory
):
    archive, private_archive, registry = _setup(
        fake_embeddings, private_paper_factory, paper_factory, agent_profile_factory
    )
    _accept(
        _cite(private_paper_factory, "old", "agent_01"),
        archive,
        private_archive,
        registry,
        metareview_factory,
    )
    assert archive.get_most_recent_by_agent("agent_01", k=10)[0].citation_count == 2
    assert _private_count(private_archive, "agent_01", "old") == 2
    assert registry.get_agent("agent_01")[1].citation_count == 2


def test_citation_to_a_pruned_paper_still_counts_privately(
    fake_embeddings,
    private_paper_factory,
    paper_factory,
    agent_profile_factory,
    metareview_factory,
    caplog,
):
    caplog.set_level(logging.DEBUG)
    archive, private_archive, registry = _setup(
        fake_embeddings, private_paper_factory, paper_factory, agent_profile_factory
    )
    archive.prune_older_than(2)  # Markovian memory drops "old" (publication_t=1)
    assert archive.get_most_recent_by_agent("agent_01", k=10) == []

    _accept(
        _cite(private_paper_factory, "old", "agent_01"),
        archive,
        private_archive,
        registry,
        metareview_factory,
    )
    assert _private_count(private_archive, "agent_01", "old") == 2
    assert registry.get_agent("agent_01")[1].citation_count == 2
    assert "1 to accepted papers no longer public" in caplog.text


def test_citation_to_a_rejected_paper_changes_only_the_registry(
    fake_embeddings,
    private_paper_factory,
    paper_factory,
    agent_profile_factory,
    metareview_factory,
    caplog,
):
    """With full_citing off a rejected paper can be cited; as in the source, its count stays."""
    caplog.set_level(logging.INFO)
    archive, private_archive, registry = _setup(
        fake_embeddings, private_paper_factory, paper_factory, agent_profile_factory
    )
    _accept(
        _cite(private_paper_factory, "rejected", "agent_01"),
        archive,
        private_archive,
        registry,
        metareview_factory,
    )
    assert _private_count(private_archive, "agent_01", "rejected") == 0
    assert registry.get_agent("agent_01")[1].citation_count == 2
    assert "0 to accepted papers no longer public" in caplog.text
    assert "1 to papers never accepted or not found" in caplog.text


def test_failed_private_lookup_still_counts_the_citation_in_the_registry(
    fake_embeddings,
    private_paper_factory,
    paper_factory,
    agent_profile_factory,
    metareview_factory,
    monkeypatch,
):
    archive, private_archive, registry = _setup(
        fake_embeddings, private_paper_factory, paper_factory, agent_profile_factory
    )
    archive.prune_older_than(2)
    real_lookup = private_archive.get_by_primary_agent

    def failing_lookup(agent_id, k=None):
        if agent_id == "agent_01":
            raise RuntimeError("lookup failed")
        return real_lookup(agent_id, k=k)

    monkeypatch.setattr(private_archive, "get_by_primary_agent", failing_lookup)
    _accept(
        _cite(private_paper_factory, "old", "agent_01"),
        archive,
        private_archive,
        registry,
        metareview_factory,
    )
    assert registry.get_agent("agent_01")[1].citation_count == 2
