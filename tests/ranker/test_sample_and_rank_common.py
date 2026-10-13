"""Tests for ranker.sample_and_rank_common — shared helpers for ranking pipelines."""

from ranker.sample_and_rank_common import (
    _metareview_sort_key,
    _top_n,
    _top_n_per_iteration,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metareview(paper_id, overall_score, rank="1/4", decision="accept"):
    """Return a metareview dict matching the PaperMetaReview schema."""
    return {
        "paper_id": paper_id,
        "meta_review_text": f"Review of {paper_id}",
        "overall_score": overall_score,
        "rank": rank,
        "justification": None,
        "decision": decision,
    }


def _make_collected_paper(paper_id, agent_id, pub_t, overall_score=None):
    """Return a paper dict as returned by collect_papers() (post-conversion)."""
    mr = _make_metareview(paper_id, overall_score) if overall_score is not None else None
    return {
        "title": f"Title of {paper_id}",
        "abstract": f"Abstract of {paper_id}",
        "manuscript": f"Manuscript of {paper_id}",
        "code": "",
        "paper_id": paper_id,
        "source_snapshot": "test_run",
        "source_snapshot_dir": "/tmp/test",
        "source_iteration": pub_t,
        "primary_agent_id": agent_id,
        "metareview": mr,
        "status": "accepted",
        "reviews": [],
        "collab_agent_ids": [],
        "citation_count": 0,
        "cited_paper_ids": [],
        "code_script": None,
        "publication_t": pub_t,
    }


# ---------------------------------------------------------------------------
# Tests: _metareview_sort_key
# ---------------------------------------------------------------------------


class TestMetareviewSortKey:
    """Verify sort key helper handles nulls correctly."""

    def test_valid_score(self):
        p = {"metareview": {"overall_score": 0.75}}
        assert _metareview_sort_key(p) == 0.75

    def test_none_metareview(self):
        p = {"metareview": None}
        assert _metareview_sort_key(p) == float("-inf")

    def test_missing_metareview(self):
        p = {}
        assert _metareview_sort_key(p) == float("-inf")

    def test_none_overall_score(self):
        p = {"metareview": {"overall_score": None}}
        assert _metareview_sort_key(p) == float("-inf")


# ---------------------------------------------------------------------------
# Tests: _top_n (direct calls)
# ---------------------------------------------------------------------------


class TestTopNBasic:
    """Top-N with enough papers: returns exactly N papers sorted by score."""

    def test_basic(self):
        papers = [_make_collected_paper(f"p{i}", f"ag{i}", 1, 0.1 * (i + 1)) for i in range(5)]
        result = _top_n(papers, 3, "test")
        assert len(result) == 3
        # Should be highest scoring: p4(0.5), p3(0.4), p2(0.3)
        ids = [p["paper_id"] for p in result]
        assert ids == ["p4", "p3", "p2"]


class TestTopNFewerThanN:
    """Top-N with fewer papers than N: all papers kept, no error."""

    def test_fewer_than_n(self):
        papers = [_make_collected_paper("p0", "ag0", 1, 0.5)]
        result = _top_n(papers, 10, "test")
        assert len(result) == 1


class TestTopNNullMetareviews:
    """Top-N with null metareviews: papers without scores sort to the bottom."""

    def test_null_metareviews(self):
        papers = [
            _make_collected_paper("p0", "ag0", 1, 0.9),
            _make_collected_paper("p1", "ag1", 1, None),
            _make_collected_paper("p2", "ag2", 1, 0.5),
        ]
        result = _top_n(papers, 2, "test")
        assert len(result) == 2
        assert [p["paper_id"] for p in result] == ["p0", "p2"]


class TestTopNDisabled:
    """When n >= len(papers), all papers are returned (equivalent to disabled)."""

    def test_all_returned_when_n_exceeds_count(self):
        papers = [_make_collected_paper(f"p{i}", f"ag{i}", 1, 0.5) for i in range(5)]
        result = _top_n(papers, 100, "test")
        assert len(result) == 5


# ---------------------------------------------------------------------------
# Tests: _top_n_per_iteration (direct calls)
# ---------------------------------------------------------------------------


class TestPerIterationBasic:
    """Per-iteration top-N: each iteration contributes exactly N papers."""

    def test_per_iteration_basic(self):
        papers = [
            # Iteration 1: 4 papers
            _make_collected_paper("p1", "ag1", 1, 0.9),
            _make_collected_paper("p2", "ag2", 1, 0.7),
            _make_collected_paper("p3", "ag3", 1, 0.3),
            _make_collected_paper("p4", "ag4", 1, 0.1),
            # Iteration 2: 4 papers
            _make_collected_paper("p5", "ag5", 2, 0.8),
            _make_collected_paper("p6", "ag6", 2, 0.6),
            _make_collected_paper("p7", "ag7", 2, 0.4),
            _make_collected_paper("p8", "ag8", 2, 0.2),
            # Iteration 3: 4 papers
            _make_collected_paper("p9", "ag9", 3, 1.0),
            _make_collected_paper("p10", "ag10", 3, 0.5),
            _make_collected_paper("p11", "ag11", 3, 0.3),
            _make_collected_paper("p12", "ag12", 3, 0.1),
        ]
        result = _top_n_per_iteration(papers, 2, "test")
        assert len(result) == 6
        ids = {p["paper_id"] for p in result}
        assert ids == {"p1", "p2", "p5", "p6", "p9", "p10"}


class TestPerIterationUnevenGroups:
    """Per-iteration top-N with uneven groups: smaller groups contribute all papers."""

    def test_per_iteration_uneven_groups(self):
        papers = [
            # Iteration 1: 5 papers
            _make_collected_paper("p1", "ag1", 1, 0.9),
            _make_collected_paper("p2", "ag2", 1, 0.8),
            _make_collected_paper("p3", "ag3", 1, 0.7),
            _make_collected_paper("p4", "ag4", 1, 0.6),
            _make_collected_paper("p5", "ag5", 1, 0.5),
            # Iteration 2: 1 paper
            _make_collected_paper("p6", "ag6", 2, 0.4),
        ]
        result = _top_n_per_iteration(papers, 3, "test")
        assert len(result) == 4  # 3 from iter 1, 1 from iter 2
        ids = {p["paper_id"] for p in result}
        assert ids == {"p1", "p2", "p3", "p6"}


class TestPerIterationNullScores:
    """Per-iteration top-N with null scores: null-scored papers sort to bottom."""

    def test_per_iteration_null_scores(self):
        papers = [
            # Iteration 1: p1(0.9), p2(None), p3(0.5)
            _make_collected_paper("p1", "ag1", 1, 0.9),
            _make_collected_paper("p2", "ag2", 1, None),
            _make_collected_paper("p3", "ag3", 1, 0.5),
            # Iteration 2: p4(None), p5(0.6), p6(None)
            _make_collected_paper("p4", "ag4", 2, None),
            _make_collected_paper("p5", "ag5", 2, 0.6),
            _make_collected_paper("p6", "ag6", 2, None),
        ]
        result = _top_n_per_iteration(papers, 2, "test")
        assert len(result) == 4
        # Iteration 1: p1(0.9), p3(0.5)
        # Iteration 2: p5(0.6), then one null-scored paper (p4 or p6)
        ids = {p["paper_id"] for p in result}
        assert "p1" in ids
        assert "p3" in ids
        assert "p5" in ids
        # One of p4/p6 should fill the second slot in iter 2
        assert len(ids & {"p4", "p6"}) == 1


class TestPerIterationSingleIteration:
    """Per-iteration top-N with single iteration: equivalent to global top-N."""

    def test_per_iteration_single_iteration(self):
        papers = [_make_collected_paper(f"p{i}", f"ag{i}", 1, 0.1 * (i + 1)) for i in range(5)]
        result = _top_n_per_iteration(papers, 3, "test")
        assert len(result) == 3
        ids = [p["paper_id"] for p in result]
        assert ids == ["p4", "p3", "p2"]


class TestPerIterationEmptyInput:
    """Per-iteration top-N with empty input: returns empty list."""

    def test_per_iteration_empty_input(self):
        result = _top_n_per_iteration([], 5, "test")
        assert result == []
