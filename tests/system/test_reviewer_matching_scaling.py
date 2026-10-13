"""Reviewer-count scaling (experiment 10): expertise assignment distinctness + salting."""

from system.reviewer_matching import (
    assign_papers_to_reviewers,
    match_papers_to_reviewers,
)
from system.schema import AgentProfile, PaperMetadataPrivate


class DummyRegistry:
    def __init__(self, ids):
        self._agents = [
            AgentProfile(
                agent_id=aid,
                behavior="",
                expertise=f"expertise {aid}",
                expertise_topics=[],
                citation_count=0,
                num_accepted_papers=0,
            )
            for aid in ids
        ]

    def get_all_agents(self):
        return list(self._agents)


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


def _registry(n=16):
    return DummyRegistry([f"a{i}" for i in range(1, n + 1)])


def _papers():
    return [_paper("p1", "a1"), _paper("p2", "a2"), _paper("p3", "a3")]


def test_expertise_yields_exactly_n_distinct_for_each_count():
    registry = _registry(16)
    papers = _papers()
    for n in (1, 2, 3, 4):
        assignments = assign_papers_to_reviewers(
            strategy="expertise_matching",
            agent_search_tool=None,  # forces the registry fallback pool
            num_reviewers_per_paper=n,
            papers=papers,
            agent_registry=registry,
            allow_self_review=False,
        )
        for paper in papers:
            rids = assignments[paper.paper_id]
            assert len(rids) == n
            assert len(set(rids)) == n
            assert paper.primary_agent_id not in rids  # author excluded


def test_expertise_excludes_reviewer_ids():
    registry = _registry(8)
    papers = _papers()
    excluded = ["a5", "a6"]
    assignments = assign_papers_to_reviewers(
        strategy="expertise_matching",
        agent_search_tool=None,
        num_reviewers_per_paper=3,
        papers=papers,
        agent_registry=registry,
        exclude_reviewer_ids=excluded,
    )
    for rids in assignments.values():
        assert not (set(rids) & set(excluded))


def test_condition_label_salts_reproducibly():
    registry = _registry(16)
    papers = _papers()

    def run(label):
        return match_papers_to_reviewers(
            agent_search_tool=None,
            num_reviewers_per_paper=3,
            papers=papers,
            agent_registry=registry,
            seed=1,
            condition_label=label,
        )

    assert run("n_reviewers=1") == run("n_reviewers=1")  # reproducible
    assert run("n_reviewers=1") != run("n_reviewers=2")  # independently salted


def test_default_condition_label_matches_dispatcher_default():
    """condition_label=None keys exactly like the dispatcher's default label.

    An unset label is deterministic and identical to the dispatcher default; setting a
    label (as experiment 10's ``n_reviewers=k`` arms do) only adds a salt.
    """
    registry = _registry(16)
    papers = _papers()
    kwargs = {
        "agent_search_tool": None,
        "num_reviewers_per_paper": 3,
        "papers": papers,
        "agent_registry": registry,
        "seed": 1,
    }
    default = match_papers_to_reviewers(**kwargs)
    explicit_none = match_papers_to_reviewers(**kwargs, condition_label=None)
    dispatched = assign_papers_to_reviewers(strategy="expertise_matching", **kwargs)
    assert default == explicit_none == dispatched
    assert default == match_papers_to_reviewers(**kwargs)  # reproducible


def test_reviewer_count_labels_salt_through_the_dispatcher_with_an_iteration():
    """review_one_step passes seed + iteration + label; experiment 10's arms must stay salted."""
    registry = _registry(16)
    papers = _papers()

    def run(n, label, seed=1, iteration=3):
        return assign_papers_to_reviewers(
            strategy="expertise_matching",
            agent_search_tool=None,
            num_reviewers_per_paper=n,
            papers=papers,
            agent_registry=registry,
            seed=seed,
            iteration=iteration,
            condition_label=label,
        )

    for n in (1, 2, 3, 4):
        label = f"n_reviewers={n}"
        first = run(n, label)
        assert first == run(n, label)  # reproducible
        for paper in papers:
            rids = first[paper.paper_id]
            assert len(rids) == len(set(rids)) == n
            assert paper.primary_agent_id not in rids
    # Same n, different label => independently salted.
    assert run(3, "n_reviewers=3") != run(3, "n_reviewers=4")
    # The seed still moves the assignment for a fixed label.
    assert run(3, "n_reviewers=3", seed=1) != run(3, "n_reviewers=3", seed=2)


def test_small_pool_returns_available_without_crash():
    # 2 eligible reviewers (minus the author) but 3 requested.
    registry = _registry(3)
    papers = [_paper("p1", "a1")]
    assignments = assign_papers_to_reviewers(
        strategy="expertise_matching",
        agent_search_tool=None,
        num_reviewers_per_paper=3,
        papers=papers,
        agent_registry=registry,
        allow_self_review=False,
    )
    rids = assignments["p1"]
    assert "a1" not in rids
    assert len(set(rids)) <= 2  # cannot exceed the eligible pool
