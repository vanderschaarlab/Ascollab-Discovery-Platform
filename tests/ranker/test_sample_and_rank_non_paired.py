"""Tests for ranker.sample_and_rank_non_paired -- batch non-paired pairwise comparison workflow."""

import json
import os
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from ranker.sample_and_rank_common import _run_single_comparison
from ranker.sample_and_rank_non_paired import (
    _apply_paper_selection_non_paired,
    _gather_papers_non_paired,
    _generate_pairs_non_paired,
    run_sample_and_rank_non_paired,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ai_msg(content: str) -> AIMessage:
    """Create an AIMessage with usage metadata for testing."""
    msg = AIMessage(content=content)
    msg.usage_metadata = {
        "input_tokens": 10,
        "output_tokens": 20,
        "total_tokens": 30,
    }
    return msg


_VERDICT_A = "Verdict:\n- Winner: A\n- Justification: Paper A is clearly better"
_VERDICT_B = "Verdict:\n- Winner: B\n- Justification: Paper B is superior"


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


def _make_paper(paper_id, agent_id, pub_t, code_script=None, metareview=None, status="accepted"):
    """Return a minimal paper dict matching the snapshot schema."""
    return {
        "paper_id": paper_id,
        "primary_agent_id": agent_id,
        "collab_agent_ids": [],
        "title": f"Title of {paper_id}",
        "abstract": f"Abstract of {paper_id}",
        "manuscript": f"Manuscript of {paper_id}",
        "citation_count": 0,
        "publication_t": pub_t,
        "cited_paper_ids": [],
        "code_script": code_script,
        "metareview": metareview,
        "status": status,
    }


def _make_timeout_paper(paper_id, agent_id, pub_t):
    """Return a timed-out paper matching the snapshot schema."""
    return {
        "paper_id": f"{paper_id}_timeout",
        "primary_agent_id": agent_id,
        "collab_agent_ids": [],
        "title": f"Incomplete Research (Timeout) - Agent {agent_id}",
        "abstract": "Research was terminated due to timeout before completion.",
        "manuscript": "TIMEOUT",
        "citation_count": 0,
        "publication_t": pub_t,
        "cited_paper_ids": [],
        "code_script": "",
        "metareview": None,
        "status": "rejected",
    }


def _make_review(iteration, reviewer_id):
    """Return a minimal review record."""
    return {
        "iteration": iteration,
        "reviewer_id": reviewer_id,
        "review_text": f"Review by {reviewer_id} for iteration {iteration}",
        "scores": {
            "Support": 2,
            "Soundness": 2,
            "Significance": 2,
            "Originality": 2,
            "Overall Recommendation": 2,
        },
        "debated": False,
        "debate_rounds": 0,
    }


def _write_iteration(
    run_dir,
    iteration,
    papers,
    run_name,
    write_exp_config=True,
    reviews_by_paper=None,
    write_public_archive=True,
):
    """Write snapshot files for one iteration."""
    iter_dir = run_dir / f"iteration_{iteration}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "iteration": iteration,
        "timestamp": "20260213_120000",
        "run_name": run_name,
        "snapshot_created_at": "2026-02-13T12:00:00.000000",
        "num_agents": 2,
        "num_papers_public": len([p for p in papers if p.get("status") == "accepted"]),
        "files_saved": ["private_archive_data.json", "snapshot_metadata.json", "exp_config.yaml"],
    }
    (iter_dir / "snapshot_metadata.json").write_text(json.dumps(meta, indent=2))

    if reviews_by_paper is None:
        reviews_by_paper = {}

    private_archive = {
        "papers": [papers],
        "reviews_by_paper": reviews_by_paper,
        "with_citation_boost": False,
        "with_recency_boost": False,
        "w_citation": 0.1,
        "w_recency": 0.1,
        "max_k": 5,
    }
    (iter_dir / "private_archive_data.json").write_text(json.dumps(private_archive, indent=2))

    if write_public_archive:
        accepted_papers = [p for p in papers if p.get("status") == "accepted"]
        public_archive = {
            "papers": accepted_papers,
            "with_citation_boost": False,
            "with_recency_boost": False,
            "w_citation": 0.1,
            "w_recency": 0.1,
            "max_k": 5,
        }
        (iter_dir / "archive_data.json").write_text(json.dumps(public_archive, indent=2))

    if write_exp_config:
        exp_config = f"exp_name: {run_name}\nrun_name: {run_name}\n"
        (iter_dir / "exp_config.yaml").write_text(exp_config)


def _make_metrics_record(paper_id, agent_id, paper_title, iteration):
    """Return a metrics record matching the metrics.jsonl schema."""
    return {
        "iteration": iteration,
        "paper_id": paper_id,
        "primary_agent_id": agent_id,
        "paper_title": paper_title,
        "metrics": {
            "support": {"external_papers_read": 3, "evidence_breadth_index": 5.0},
            "soundness": {"internal_consistency_score": 0.8},
            "significance": {"clinical_biological_relevance": 0.6},
            "originality": {"hypothesis_overlap_jaccard": 0.2},
        },
    }


def _write_metrics_jsonl(run_dir, iteration, records):
    """Write a metrics.jsonl file for the given iteration."""
    iter_dir = run_dir / f"iteration_{iteration}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r) for r in records]
    (iter_dir / "metrics.jsonl").write_text("\n".join(lines) + "\n")


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
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def single_pool_snapshots(tmp_path):
    """Build 2 snapshot directories treated as a single pool.

    Snapshot 1: 2 iterations, ~2 papers per iteration (4 papers cumulative).
    Snapshot 2: 2 iterations, ~2 papers per iteration (3 papers cumulative).
    Metrics written for iteration 2 papers.
    Returns dict with paths, run names, a paper_pool_cfg dict, and expected paper counts.
    """
    # ---- Snapshot 1 ----
    snap1 = tmp_path / "snap_1"
    snap1_name = "snap_1_260213_1200"

    s1_t1 = [
        _make_paper("paper_t01_agent_01", "agent_01", 1, code_script="print('s1')"),
        _make_paper("paper_t01_agent_02", "agent_02", 1),
    ]
    s1_t2 = s1_t1 + [
        _make_paper("paper_t02_agent_01", "agent_01", 2, code_script="x = 1"),
        _make_paper("paper_t02_agent_02", "agent_02", 2),
    ]
    _write_iteration(snap1, 1, s1_t1, snap1_name)
    _write_iteration(snap1, 2, s1_t2, snap1_name)

    # ---- Snapshot 2 ----
    snap2 = tmp_path / "snap_2"
    snap2_name = "snap_2_260213_1200"

    s2_t1 = [
        _make_paper("paper_t01_agent_03", "agent_03", 1),
        _make_paper("paper_t01_agent_04", "agent_04", 1, code_script="y = 2"),
    ]
    s2_t2 = s2_t1 + [
        _make_paper("paper_t02_agent_03", "agent_03", 2),
    ]
    _write_iteration(snap2, 1, s2_t1, snap2_name)
    _write_iteration(snap2, 2, s2_t2, snap2_name)

    # Write metrics.jsonl for iteration 2 papers
    _write_metrics_jsonl(
        snap1,
        2,
        [
            _make_metrics_record(
                "paper_t02_agent_01", "agent_01", "Title of paper_t02_agent_01", 2
            ),
            _make_metrics_record(
                "paper_t02_agent_02", "agent_02", "Title of paper_t02_agent_02", 2
            ),
        ],
    )
    _write_metrics_jsonl(
        snap2,
        2,
        [
            _make_metrics_record(
                "paper_t02_agent_03", "agent_03", "Title of paper_t02_agent_03", 2
            ),
        ],
    )

    return {
        "snap1": snap1,
        "snap1_name": snap1_name,
        "snap2": snap2,
        "snap2_name": snap2_name,
        "paper_pool_cfg": {
            "label": "my_pool",
            "snapshot_dirs": [str(snap1), str(snap2)],
        },
        # Expected counts for iterations="last", cumulative=True:
        # snap1: 4 papers (t1: 2, t2: 2, cumulative up to t=2)
        # snap2: 3 papers (t1: 2, t2: 1, cumulative up to t=2)
        # Total: 7
        "expected_cumulative_count": 7,
        # For cumulative=False, iterations="last":
        # snap1: 2 papers (t=2 only)
        # snap2: 1 paper (t=2 only)
        # Total: 3
        "expected_noncumulative_count": 3,
    }


@pytest.fixture
def mock_llm():
    """MagicMock that returns an AIMessage with a valid verdict on every call."""
    mock = MagicMock()
    mock.invoke.return_value = _make_ai_msg(_VERDICT_A)
    return mock


# ---------------------------------------------------------------------------
# Tests: Paper Gathering (Single Pool)
# ---------------------------------------------------------------------------


class TestGatherSingleSnapshot:
    """1. test_gather_single_snapshot"""

    def test_gather_single_snapshot(self, single_pool_snapshots):
        snap1 = single_pool_snapshots["snap1"]
        snap1_name = single_pool_snapshots["snap1_name"]

        papers, collection_info = _gather_papers_non_paired(
            [str(snap1)],
            "test_pool",
            {"iterations": "last", "cumulative": True},
        )

        # snap1: cumulative up to t=2 -> 2 at t=1 + 2 at t=2 = 4
        assert len(papers) == 4

        # All papers should have source_snapshot set correctly
        for p in papers:
            assert p["source_snapshot"] == snap1_name

        # collection_info should have paper_pool key
        assert "paper_pool" in collection_info
        assert collection_info["paper_pool"]["label"] == "test_pool"


class TestGatherMultipleSnapshots:
    """2. test_gather_multiple_snapshots"""

    def test_gather_multiple_snapshots(self, single_pool_snapshots):
        papers, collection_info = _gather_papers_non_paired(
            [str(single_pool_snapshots["snap1"]), str(single_pool_snapshots["snap2"])],
            "my_pool",
            {"iterations": "last", "cumulative": True},
        )

        # snap1: 4 cumulative, snap2: 3 cumulative -> total 7
        assert len(papers) == single_pool_snapshots["expected_cumulative_count"]

        # source_snapshot_dir values should cover both input dirs
        dirs = {p["source_snapshot_dir"] for p in papers}
        assert len(dirs) == 2
        assert str(single_pool_snapshots["snap1"]) in dirs
        assert str(single_pool_snapshots["snap2"]) in dirs


class TestGatherCumulativeFalse:
    """3. test_gather_cumulative_false"""

    def test_gather_cumulative_false(self, single_pool_snapshots):
        papers, _ = _gather_papers_non_paired(
            [str(single_pool_snapshots["snap1"]), str(single_pool_snapshots["snap2"])],
            "my_pool",
            {"iterations": "last", "cumulative": False},
        )

        # Only last-iteration papers: snap1: 2 (t=2), snap2: 1 (t=2)
        assert len(papers) == single_pool_snapshots["expected_noncumulative_count"]
        for p in papers:
            assert p["source_iteration"] == 2


class TestGatherIterationExplicit:
    """4. test_gather_iteration_explicit"""

    def test_gather_iteration_explicit(self, single_pool_snapshots):
        papers, _ = _gather_papers_non_paired(
            [str(single_pool_snapshots["snap1"]), str(single_pool_snapshots["snap2"])],
            "my_pool",
            {"iterations": [1], "cumulative": True},
        )

        # Only iteration 1 papers: snap1: 2 (t=1), snap2: 2 (t=1)
        for p in papers:
            assert p["source_iteration"] == 1
        assert len(papers) == 4


class TestGatherWithIncludeRejected:
    """5. test_gather_with_include_rejected"""

    def test_gather_with_include_rejected(self, tmp_path):
        snap = tmp_path / "snap_rejected"
        snap_name = "snap_rejected_run"

        papers = [
            _make_paper("accepted_01", "agent_01", 1, status="accepted"),
            _make_paper("rejected_01", "agent_02", 1, status="rejected"),
        ]
        _write_iteration(snap, 1, papers, snap_name)

        # include_rejected=True: both papers included
        result_incl, _ = _gather_papers_non_paired(
            [str(snap)],
            "pool",
            {"iterations": "last", "cumulative": True, "include_rejected": True},
        )
        assert len(result_incl) == 2
        statuses = {p["status"] for p in result_incl}
        assert "rejected" in statuses
        assert "accepted" in statuses

        # include_rejected=False (default): only accepted
        result_excl, _ = _gather_papers_non_paired(
            [str(snap)],
            "pool",
            {"iterations": "last", "cumulative": True, "include_rejected": False},
        )
        assert len(result_excl) == 1
        assert result_excl[0]["status"] == "accepted"


class TestGatherWithExcludeTimeout:
    """6. test_gather_with_exclude_timeout"""

    def test_gather_with_exclude_timeout(self, tmp_path):
        snap = tmp_path / "snap_timeout"
        snap_name = "snap_timeout_run"

        papers = [
            _make_paper("normal_01", "agent_01", 1, status="accepted"),
            _make_timeout_paper("timed_out_01", "agent_02", 1),
        ]
        _write_iteration(snap, 1, papers, snap_name)

        # exclude_timeout=True (default): timeout papers excluded
        result_excl, _ = _gather_papers_non_paired(
            [str(snap)],
            "pool",
            {
                "iterations": "last",
                "cumulative": True,
                "exclude_timeout": True,
                "include_rejected": True,
            },
        )
        assert len(result_excl) == 1
        assert result_excl[0]["paper_id"] == "normal_01"

        # exclude_timeout=False: timeout papers included
        result_incl, _ = _gather_papers_non_paired(
            [str(snap)],
            "pool",
            {
                "iterations": "last",
                "cumulative": True,
                "exclude_timeout": False,
                "include_rejected": True,
            },
        )
        assert len(result_incl) == 2
        pids = {p["paper_id"] for p in result_incl}
        assert "timed_out_01_timeout" in pids


# ---------------------------------------------------------------------------
# Tests: Paper Selection (Non-Paired)
# ---------------------------------------------------------------------------


class TestTopNAppliedToPool:
    """7. test_top_n_applied_to_pool"""

    def test_top_n_applied_to_pool(self):
        papers = [
            _make_collected_paper(f"p{i}", f"ag{i}", 1, overall_score=0.1 * (i + 1))
            for i in range(10)
        ]

        cfg = {"top_n": {"enabled": True, "n": 3}}
        result, info = _apply_paper_selection_non_paired(papers, cfg)

        assert len(result) == 3
        # Top 3 by score: p9(1.0), p8(0.9), p7(0.8)
        ids = [p["paper_id"] for p in result]
        assert ids == ["p9", "p8", "p7"]
        assert info["top_n"]["enabled"] is True
        assert info["top_n"]["n"] == 3
        assert info["top_n"]["count_after"] == 3


class TestTopNDisabled:
    """8. test_top_n_disabled"""

    def test_top_n_disabled(self):
        papers = [_make_collected_paper(f"p{i}", f"ag{i}", 1, overall_score=0.5) for i in range(5)]

        cfg = {"top_n": {"enabled": False}}
        result, info = _apply_paper_selection_non_paired(papers, cfg)

        assert len(result) == 5
        assert info["top_n"]["enabled"] is False
        assert info["final_count"] == 5


class TestBalanceNotInConfig:
    """9. test_balance_not_in_config"""

    def test_balance_not_in_config(self):
        papers = [_make_collected_paper(f"p{i}", f"ag{i}", 1, overall_score=0.5) for i in range(5)]

        # Even if balance settings are passed, they should have no effect
        # because _apply_paper_selection_non_paired does not process balance.
        cfg = {
            "top_n": {"enabled": False},
            "balance": {"enabled": True, "method": "random", "seed": 42},
        }
        result, info = _apply_paper_selection_non_paired(papers, cfg)

        # All papers pass through -- no balance logic
        assert len(result) == 5
        assert "balance" not in info


# ---------------------------------------------------------------------------
# Tests: Paper selection -- Per-iteration Top-N (non-paired)
# ---------------------------------------------------------------------------


class TestPerIterationTopNNonPaired:
    """Per-iteration top-N works on the single pool."""

    def test_per_iteration_top_n_non_paired(self):
        papers = [
            # Iteration 1: 4 papers
            _make_collected_paper("p1", "ag1", 1, 0.9),
            _make_collected_paper("p2", "ag2", 1, 0.7),
            _make_collected_paper("p3", "ag3", 1, 0.5),
            _make_collected_paper("p4", "ag4", 1, 0.3),
            # Iteration 2: 3 papers
            _make_collected_paper("p5", "ag5", 2, 0.8),
            _make_collected_paper("p6", "ag6", 2, 0.6),
            _make_collected_paper("p7", "ag7", 2, 0.4),
            # Iteration 3: 5 papers
            _make_collected_paper("p8", "ag8", 3, 1.0),
            _make_collected_paper("p9", "ag9", 3, 0.9),
            _make_collected_paper("p10", "ag10", 3, 0.7),
            _make_collected_paper("p11", "ag11", 3, 0.5),
            _make_collected_paper("p12", "ag12", 3, 0.3),
        ]

        cfg = {"top_n": {"enabled": True, "n": 2, "per_iteration": True}}
        result, info = _apply_paper_selection_non_paired(papers, cfg)

        assert len(result) == 6  # 2 from each of 3 iterations
        assert info["top_n"]["per_iteration"] is True
        assert info["top_n"]["count_after"] == 6


class TestPerIterationFalseIsGlobalTopNNonPaired:
    """per_iteration=False preserves existing global top-N behaviour."""

    def test_per_iteration_false_is_global_top_n_non_paired(self):
        papers = [
            _make_collected_paper("p1", "ag1", 1, 0.9),
            _make_collected_paper("p2", "ag2", 1, 0.7),
            _make_collected_paper("p3", "ag3", 1, 0.5),
            _make_collected_paper("p4", "ag4", 1, 0.3),
            _make_collected_paper("p5", "ag5", 2, 0.8),
            _make_collected_paper("p6", "ag6", 2, 0.6),
            _make_collected_paper("p7", "ag7", 2, 0.4),
            _make_collected_paper("p8", "ag8", 3, 1.0),
            _make_collected_paper("p9", "ag9", 3, 0.9),
            _make_collected_paper("p10", "ag10", 3, 0.7),
            _make_collected_paper("p11", "ag11", 3, 0.5),
            _make_collected_paper("p12", "ag12", 3, 0.3),
        ]

        cfg = {"top_n": {"enabled": True, "n": 2, "per_iteration": False}}
        result, info = _apply_paper_selection_non_paired(papers, cfg)

        # Global top 2: p8(1.0), p1(0.9) or p9(0.9) — top 2 by score overall
        assert len(result) == 2
        assert info["top_n"]["per_iteration"] is False


class TestPerIterationDisabledIgnoresFlag:
    """per_iteration is ignored when top-N is disabled."""

    def test_per_iteration_disabled_top_n_ignores_flag_non_paired(self):
        papers = [
            _make_collected_paper("p1", "ag1", 1, 0.9),
            _make_collected_paper("p2", "ag2", 1, 0.7),
            _make_collected_paper("p3", "ag3", 2, 0.8),
            _make_collected_paper("p4", "ag4", 2, 0.6),
        ]

        cfg = {"top_n": {"enabled": False, "n": 1, "per_iteration": True}}
        result, info = _apply_paper_selection_non_paired(papers, cfg)

        assert len(result) == 4  # All papers pass through
        assert info["top_n"]["enabled"] is False


class TestPerIterationDefaultsFalseNonPaired:
    """Missing per_iteration key defaults to global top-N (backward compat)."""

    def test_per_iteration_defaults_to_false(self):
        papers = [
            _make_collected_paper("p1", "ag1", 1, 0.9),
            _make_collected_paper("p2", "ag2", 1, 0.7),
            _make_collected_paper("p3", "ag3", 2, 0.8),
        ]

        # No per_iteration key
        cfg = {"top_n": {"enabled": True, "n": 5}}
        result, info = _apply_paper_selection_non_paired(papers, cfg)

        assert len(result) == 3  # All pass through (n=5 > 3)
        assert info["top_n"]["per_iteration"] is False


# ---------------------------------------------------------------------------
# Tests: Pair Generation (Non-Paired)
# ---------------------------------------------------------------------------


class TestPairsAllModeUnordered:
    """10. test_pairs_all_mode_unordered"""

    def test_pairs_all_mode_unordered(self):
        n = 5
        papers = [_make_collected_paper(f"p{i}", f"ag{i}", 1) for i in range(n)]

        sampling_cfg = {"mode": "all", "n_repeats": 1, "seed": 42, "allow_self_comparison": False}
        tasks = _generate_pairs_non_paired(papers, sampling_cfg)

        expected = n * (n - 1) // 2  # C(5, 2) = 10
        assert len(tasks) == expected

        # No self-comparison
        for t in tasks:
            assert t["paper_a"]["paper_id"] != t["paper_b"]["paper_id"]

        # Each unordered pair appears exactly once
        logical_pairs = []
        for t in tasks:
            pair = frozenset([t["paper_a"]["paper_id"], t["paper_b"]["paper_id"]])
            logical_pairs.append(pair)
        assert len(logical_pairs) == len(set(logical_pairs))


class TestPairsAllModeWithRepeats:
    """11. test_pairs_all_mode_with_repeats"""

    def test_pairs_all_mode_with_repeats(self):
        n = 4
        papers = [_make_collected_paper(f"p{i}", f"ag{i}", 1) for i in range(n)]

        sampling_cfg = {"mode": "all", "n_repeats": 3, "seed": 42, "allow_self_comparison": False}
        tasks = _generate_pairs_non_paired(papers, sampling_cfg)

        expected = n * (n - 1) // 2 * 3  # C(4, 2) * 3 = 18
        assert len(tasks) == expected

        repeat_indices = {t["repeat_index"] for t in tasks}
        assert repeat_indices == {0, 1, 2}


class TestPairsAllModeWithSelfComparison:
    """12. test_pairs_all_mode_with_self_comparison"""

    def test_pairs_all_mode_with_self_comparison(self):
        n = 4
        papers = [_make_collected_paper(f"p{i}", f"ag{i}", 1) for i in range(n)]

        sampling_cfg = {"mode": "all", "n_repeats": 1, "seed": 42, "allow_self_comparison": True}
        tasks = _generate_pairs_non_paired(papers, sampling_cfg)

        # C(n,2) + n = n*(n+1)/2
        expected = n * (n + 1) // 2
        assert len(tasks) == expected

        # Exactly n pairs have the same paper in both positions
        self_pairs = [t for t in tasks if t["paper_a"]["paper_id"] == t["paper_b"]["paper_id"]]
        assert len(self_pairs) == n


class TestPairsRandomWithoutReplacement:
    """13. test_pairs_random_without_replacement"""

    def test_pairs_random_without_replacement(self):
        n = 6
        papers = [_make_collected_paper(f"p{i}", f"ag{i}", 1) for i in range(n)]
        pool_size = n * (n - 1) // 2  # C(6,2) = 15  # noqa: F841
        n_pairs = 5  # Less than pool_size

        sampling_cfg = {"mode": "random", "n_pairs": n_pairs, "with_replacement": False, "seed": 42}
        tasks = _generate_pairs_non_paired(papers, sampling_cfg)

        assert len(tasks) == n_pairs

        # No duplicate unordered pairs
        logical_pairs = []
        for t in tasks:
            pair = frozenset([t["paper_a"]["paper_id"], t["paper_b"]["paper_id"]])
            logical_pairs.append(pair)
        assert len(logical_pairs) == len(set(logical_pairs))


class TestPairsRandomWithReplacement:
    """14. test_pairs_random_with_replacement"""

    def test_pairs_random_with_replacement(self):
        n = 4
        papers = [_make_collected_paper(f"p{i}", f"ag{i}", 1) for i in range(n)]
        pool_size = n * (n - 1) // 2  # C(4,2) = 6
        n_pairs = pool_size + 5  # More than pool size

        sampling_cfg = {"mode": "random", "n_pairs": n_pairs, "with_replacement": True, "seed": 42}
        tasks = _generate_pairs_non_paired(papers, sampling_cfg)

        # No error, correct count
        assert len(tasks) == n_pairs


class TestPairsRandomOverflowError:
    """15. test_pairs_random_overflow_error"""

    def test_pairs_random_overflow_error(self):
        n = 4
        papers = [_make_collected_paper(f"p{i}", f"ag{i}", 1) for i in range(n)]
        pool_size = n * (n - 1) // 2  # C(4,2) = 6
        n_pairs = pool_size + 1

        sampling_cfg = {"mode": "random", "n_pairs": n_pairs, "with_replacement": False, "seed": 42}
        with pytest.raises(ValueError, match="Cannot sample"):
            _generate_pairs_non_paired(papers, sampling_cfg)


class TestPairsReproducible:
    """16. test_pairs_reproducible"""

    def test_pairs_reproducible(self):
        n = 6
        papers = [_make_collected_paper(f"p{i}", f"ag{i}", 1) for i in range(n)]

        sampling_cfg_42 = {"mode": "random", "n_pairs": 5, "with_replacement": False, "seed": 42}
        tasks_42a = _generate_pairs_non_paired(papers, sampling_cfg_42)
        tasks_42b = _generate_pairs_non_paired(papers, sampling_cfg_42)

        # Same seed -> same pairs
        ids_a = [(t["paper_a"]["paper_id"], t["paper_b"]["paper_id"]) for t in tasks_42a]
        ids_b = [(t["paper_a"]["paper_id"], t["paper_b"]["paper_id"]) for t in tasks_42b]
        assert ids_a == ids_b

        # Different seed -> different pairs (with high probability)
        sampling_cfg_99 = {"mode": "random", "n_pairs": 5, "with_replacement": False, "seed": 99}
        tasks_99 = _generate_pairs_non_paired(papers, sampling_cfg_99)
        ids_99 = [(t["paper_a"]["paper_id"], t["paper_b"]["paper_id"]) for t in tasks_99]
        assert ids_a != ids_99


class TestPairsABRandomization:
    """17. test_pairs_ab_randomization"""

    def test_pairs_ab_randomization(self):
        # Use enough papers so that randomization produces both orderings
        n = 8
        papers = [_make_collected_paper(f"p{i}", f"ag{i}", 1) for i in range(n)]

        sampling_cfg = {"mode": "all", "n_repeats": 1, "seed": 42}
        tasks = _generate_pairs_non_paired(papers, sampling_cfg)

        # For non-paired, experiment fields are None, so we check that
        # for a given logical pair, paper X appears as both A and B across
        # different comparisons.
        # Collect which paper_ids appear in A vs B position
        a_ids = {t["paper_a"]["paper_id"] for t in tasks}
        b_ids = {t["paper_b"]["paper_id"] for t in tasks}

        # With enough pairs and randomization, a paper should appear in both positions
        both = a_ids & b_ids
        assert len(both) > 0, "Expected at least one paper to appear in both A and B positions"


class TestNoSelfComparisonDefault:
    """18. test_no_self_comparison_default"""

    def test_no_self_comparison_default(self):
        n = 5
        papers = [_make_collected_paper(f"p{i}", f"ag{i}", 1) for i in range(n)]

        # Default config: allow_self_comparison absent
        sampling_cfg = {"mode": "all", "n_repeats": 1, "seed": 42}
        tasks = _generate_pairs_non_paired(papers, sampling_cfg)

        for t in tasks:
            assert t["paper_a"]["paper_id"] != t["paper_b"]["paper_id"], (
                f"Self-comparison found: {t['paper_a']['paper_id']}"
            )


# ---------------------------------------------------------------------------
# Tests: Comparison Records (Non-Paired)
# ---------------------------------------------------------------------------


class TestSingleComparisonSuccess:
    """19. test_single_comparison_success"""

    def test_single_comparison_success(self, single_pool_snapshots, mock_llm):
        papers, _ = _gather_papers_non_paired(
            [str(single_pool_snapshots["snap1"])],
            "pool",
            {"iterations": "last", "cumulative": False},
        )

        sampling_cfg = {"mode": "all", "n_repeats": 1, "seed": 42}
        tasks = _generate_pairs_non_paired(papers, sampling_cfg)

        ranker_cfg = {
            "max_iterations": 5,
            "tools": {},
            "llm_retry": {},
            "output_parsing": {"max_parse_retries": 2},
        }

        record = _run_single_comparison(
            task=tasks[0],
            llm=mock_llm,
            ranker_cfg=ranker_cfg,
            sandbox_client=None,
            iteration_selection="last",
        )

        assert record["status"] == "completed"
        assert record["score"] in (0, 1)
        assert record["paper_a_id"] == tasks[0]["paper_a"]["paper_id"]
        assert record["paper_b_id"] == tasks[0]["paper_b"]["paper_id"]
        assert record["comparison_id"] == tasks[0]["comparison_id"]


class TestSingleComparisonFailure:
    """20. test_single_comparison_failure"""

    def test_single_comparison_failure(self, single_pool_snapshots):
        papers, _ = _gather_papers_non_paired(
            [str(single_pool_snapshots["snap1"])],
            "pool",
            {"iterations": "last", "cumulative": False},
        )

        sampling_cfg = {"mode": "all", "n_repeats": 1, "seed": 42}
        tasks = _generate_pairs_non_paired(papers, sampling_cfg)

        failing_llm = MagicMock()
        failing_llm.invoke.side_effect = RuntimeError("LLM exploded")

        ranker_cfg = {
            "max_iterations": 5,
            "tools": {},
            "llm_retry": {},
            "output_parsing": {"max_parse_retries": 2},
        }

        record = _run_single_comparison(
            task=tasks[0],
            llm=failing_llm,
            ranker_cfg=ranker_cfg,
            sandbox_client=None,
            iteration_selection="last",
        )

        assert record["status"] == "failed"
        assert record["error"] is not None
        assert "LLM exploded" in record["error"]


class TestExperimentFieldsAreNull:
    """21. test_experiment_fields_are_null"""

    def test_experiment_fields_are_null(self, single_pool_snapshots, mock_llm):
        papers, _ = _gather_papers_non_paired(
            [str(single_pool_snapshots["snap1"])],
            "pool",
            {"iterations": "last", "cumulative": False},
        )

        sampling_cfg = {"mode": "all", "n_repeats": 1, "seed": 42}
        tasks = _generate_pairs_non_paired(papers, sampling_cfg)

        ranker_cfg = {
            "max_iterations": 5,
            "tools": {},
            "llm_retry": {},
            "output_parsing": {"max_parse_retries": 2},
        }

        record = _run_single_comparison(
            task=tasks[0],
            llm=mock_llm,
            ranker_cfg=ranker_cfg,
            sandbox_client=None,
            iteration_selection="last",
        )

        assert record["paper_a_experiment"] is None
        assert record["paper_b_experiment"] is None


# ---------------------------------------------------------------------------
# Tests: End-to-End (Mocked LLM)
# ---------------------------------------------------------------------------


class TestE2EOutputFilesExist:
    """22. test_e2e_output_files_exist"""

    def test_e2e_output_files_exist(self, single_pool_snapshots, mock_llm, tmp_path):
        output_dir = str(tmp_path / "output_e2e")

        ranker_cfg = {
            "max_iterations": 5,
            "tools": {},
            "llm_retry": {},
            "output_parsing": {"max_parse_retries": 2},
        }

        result = run_sample_and_rank_non_paired(
            paper_pool_cfg=single_pool_snapshots["paper_pool_cfg"],
            sampling_cfg={"mode": "all", "n_repeats": 1, "seed": 42},
            ranker_cfg=ranker_cfg,
            parallelism_cfg={"max_workers": 2},
            llm=mock_llm,
            run_id="test_e2e_run",
            output_dir=output_dir,
            paper_selection_cfg={"iterations": "last", "cumulative": False},
        )

        assert "comparisons" in result
        assert "summary" in result
        assert "output_dir" in result

        # comparisons.json exists and is valid JSON list
        comparisons_path = os.path.join(output_dir, "comparisons.json")
        assert os.path.exists(comparisons_path)
        with open(comparisons_path) as f:
            comparisons_data = json.load(f)
        assert isinstance(comparisons_data, list)
        assert len(comparisons_data) > 0

        # run_summary.json exists
        summary_path = os.path.join(output_dir, "run_summary.json")
        assert os.path.exists(summary_path)

        # ranker_config.json exists
        config_path = os.path.join(output_dir, "ranker_config.json")
        assert os.path.exists(config_path)

        # experiment_configs/ has YAML files
        exp_configs_dir = os.path.join(output_dir, "experiment_configs")
        assert os.path.isdir(exp_configs_dir)
        yaml_files = [f for f in os.listdir(exp_configs_dir) if f.endswith(".yaml")]
        assert len(yaml_files) >= 1

        # ranker_traces/ has one JSON per completed comparison
        summary_data = result["summary"]
        traces_dir = os.path.join(output_dir, "ranker_traces")
        if summary_data["results"]["completed"] > 0:
            assert os.path.isdir(traces_dir)
            trace_files = [f for f in os.listdir(traces_dir) if f.endswith(".json")]
            assert len(trace_files) == summary_data["results"]["completed"]

        # papers/paper_data.json exists
        paper_data_path = os.path.join(output_dir, "papers", "paper_data.json")
        assert os.path.exists(paper_data_path)


class TestE2EComparisonsPandasLoadable:
    """23. test_e2e_comparisons_pandas_loadable"""

    def test_e2e_comparisons_pandas_loadable(self, single_pool_snapshots, mock_llm, tmp_path):
        pd = pytest.importorskip("pandas")

        output_dir = str(tmp_path / "output_pandas")

        ranker_cfg = {
            "max_iterations": 5,
            "tools": {},
            "llm_retry": {},
            "output_parsing": {"max_parse_retries": 2},
        }

        run_sample_and_rank_non_paired(
            paper_pool_cfg=single_pool_snapshots["paper_pool_cfg"],
            sampling_cfg={"mode": "all", "n_repeats": 1, "seed": 42},
            ranker_cfg=ranker_cfg,
            parallelism_cfg={"max_workers": 2},
            llm=mock_llm,
            run_id="test_pandas_run",
            output_dir=output_dir,
            paper_selection_cfg={"iterations": "last", "cumulative": False},
        )

        comparisons_path = os.path.join(output_dir, "comparisons.json")
        df = pd.read_json(comparisons_path)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

        expected_cols = [
            "comparison_id",
            "run_id",
            "status",
            "score",
            "outcome",
            "paper_a_id",
            "paper_b_id",
            "paper_a_experiment",
            "paper_b_experiment",
            "turns",
            "input_tokens",
            "output_tokens",
            "duration_seconds",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"


class TestE2EPaperDataStructure:
    """24. test_e2e_paper_data_structure"""

    def test_e2e_paper_data_structure(self, single_pool_snapshots, mock_llm, tmp_path):
        output_dir = str(tmp_path / "output_paper_data")

        ranker_cfg = {
            "max_iterations": 5,
            "tools": {},
            "llm_retry": {},
            "output_parsing": {"max_parse_retries": 2},
        }

        run_sample_and_rank_non_paired(
            paper_pool_cfg=single_pool_snapshots["paper_pool_cfg"],
            sampling_cfg={"mode": "all", "n_repeats": 1, "seed": 42},
            ranker_cfg=ranker_cfg,
            parallelism_cfg={"max_workers": 2},
            llm=mock_llm,
            run_id="test_paper_data_run",
            output_dir=output_dir,
            paper_selection_cfg={"iterations": "last", "cumulative": False},
        )

        paper_data_path = os.path.join(output_dir, "papers", "paper_data.json")
        assert os.path.exists(paper_data_path)

        with open(paper_data_path) as f:
            paper_data = json.load(f)

        assert isinstance(paper_data, list)
        assert len(paper_data) > 0

        for rec in paper_data:
            assert "paper_id" in rec
            assert "experiment" in rec
            assert "experiment_condition" in rec
            assert "source_snapshot" in rec
            assert "metrics" in rec
            assert "metareview" in rec
            assert "status" in rec

            # Non-paired: experiment == pool_label, experiment_condition is null
            assert rec["experiment"] == "my_pool"
            assert rec["experiment_condition"] is None


class TestE2ESummaryStructure:
    """25. test_e2e_summary_structure"""

    def test_e2e_summary_structure(self, single_pool_snapshots, mock_llm, tmp_path):
        output_dir = str(tmp_path / "output_summary")

        ranker_cfg = {
            "max_iterations": 5,
            "tools": {},
            "llm_retry": {},
            "output_parsing": {"max_parse_retries": 2},
        }

        result = run_sample_and_rank_non_paired(  # noqa: F841
            paper_pool_cfg=single_pool_snapshots["paper_pool_cfg"],
            sampling_cfg={"mode": "all", "n_repeats": 1, "seed": 42},
            ranker_cfg=ranker_cfg,
            parallelism_cfg={"max_workers": 2},
            llm=mock_llm,
            run_id="test_summary_run",
            output_dir=output_dir,
            paper_selection_cfg={"iterations": "last", "cumulative": False},
        )

        summary_path = os.path.join(output_dir, "run_summary.json")
        with open(summary_path) as f:
            summary = json.load(f)

        # paper_pool key in experiments (not experiment_a/experiment_b)
        assert "experiments" in summary
        assert "paper_pool" in summary["experiments"]
        assert "experiment_a" not in summary["experiments"]
        assert "experiment_b" not in summary["experiments"]

        # Pool label matches
        assert summary["experiments"]["paper_pool"]["label"] == "my_pool"

        # sampling.total_unique_pairs matches C(n, 2)
        n_papers = summary["experiments"]["paper_pool"]["num_papers"]
        expected_pairs = n_papers * (n_papers - 1) // 2
        assert summary["sampling"]["total_unique_pairs"] == expected_pairs

        # paper_data section
        assert "paper_data" in summary
        assert "total" in summary["paper_data"]
        assert "with_system_metrics" in summary["paper_data"]
        assert "without_system_metrics" in summary["paper_data"]


class TestE2EMixedSuccessFailure:
    """26. test_e2e_mixed_success_failure"""

    def test_e2e_mixed_success_failure(self, single_pool_snapshots, tmp_path):
        output_dir = str(tmp_path / "output_mixed")

        call_count = 0

        def mock_invoke(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Simulated LLM failure")
            return _make_ai_msg(_VERDICT_A)

        mixed_llm = MagicMock()
        mixed_llm.invoke.side_effect = mock_invoke

        ranker_cfg = {
            "max_iterations": 5,
            "tools": {},
            "llm_retry": {},
            "output_parsing": {"max_parse_retries": 2},
        }

        result = run_sample_and_rank_non_paired(
            paper_pool_cfg=single_pool_snapshots["paper_pool_cfg"],
            sampling_cfg={"mode": "all", "n_repeats": 1, "seed": 42},
            ranker_cfg=ranker_cfg,
            parallelism_cfg={"max_workers": 1},  # Sequential to control failure order
            llm=mixed_llm,
            run_id="test_mixed_run",
            output_dir=output_dir,
            paper_selection_cfg={"iterations": "last", "cumulative": False},
        )

        comparisons = result["comparisons"]
        statuses = {c["status"] for c in comparisons}
        assert "failed" in statuses
        assert "completed" in statuses

        # Failed records have error field
        failed = [c for c in comparisons if c["status"] == "failed"]
        assert len(failed) >= 1
        for f in failed:
            assert f["error"] is not None

        # File also contains both statuses
        comparisons_path = os.path.join(output_dir, "comparisons.json")
        with open(comparisons_path) as fh:
            data = json.load(fh)
        statuses_file = {c["status"] for c in data}
        assert "failed" in statuses_file
        assert "completed" in statuses_file


class TestE2ECumulativeInRecords:
    """27. test_e2e_cumulative_in_records"""

    def test_e2e_cumulative_in_records(self, single_pool_snapshots, mock_llm, tmp_path):
        output_dir = str(tmp_path / "output_cumulative")

        ranker_cfg = {
            "max_iterations": 5,
            "tools": {},
            "llm_retry": {},
            "output_parsing": {"max_parse_retries": 2},
        }

        result = run_sample_and_rank_non_paired(
            paper_pool_cfg=single_pool_snapshots["paper_pool_cfg"],
            sampling_cfg={"mode": "all", "n_repeats": 1, "seed": 42},
            ranker_cfg=ranker_cfg,
            parallelism_cfg={"max_workers": 2},
            llm=mock_llm,
            run_id="test_cumulative_run",
            output_dir=output_dir,
            paper_selection_cfg={"iterations": "last", "cumulative": True},
        )

        # Every comparison record has cumulative: true
        for c in result["comparisons"]:
            assert "cumulative" in c
            assert c["cumulative"] is True

        # Summary has cumulative: true
        assert result["summary"]["cumulative"] is True


# ---------------------------------------------------------------------------
# Tests: Integration (Real LLM)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestIntegrationRealLlm:
    """28. test_integration_real_llm"""

    def test_integration_real_llm(self, llm, tmp_path):
        # Build minimal synthetic snapshot: 3 papers -> C(3,2) = 3 pairs
        snap = tmp_path / "int_snap"
        snap_name = "int_snap_run"

        papers = [
            _make_paper("paper_t01_agent_01", "agent_01", 1, code_script="print('hello')"),
            _make_paper("paper_t01_agent_02", "agent_02", 1),
            _make_paper("paper_t01_agent_03", "agent_03", 1),
        ]
        _write_iteration(snap, 1, papers, snap_name)

        output_dir = str(tmp_path / "int_output")

        paper_pool_cfg = {
            "label": "integration_pool",
            "snapshot_dirs": [str(snap)],
        }

        ranker_cfg = {
            "max_iterations": 5,
            "tools": {},
            "llm_retry": {"max_retries": 3, "base_delay_seconds": 2.0, "max_delay_seconds": 30.0},
            "output_parsing": {"max_parse_retries": 2},
        }

        result = run_sample_and_rank_non_paired(
            paper_pool_cfg=paper_pool_cfg,
            sampling_cfg={"mode": "all", "n_repeats": 1, "seed": 42},
            ranker_cfg=ranker_cfg,
            parallelism_cfg={"max_workers": 2},
            llm=llm,
            run_id="integration_test_np",
            output_dir=output_dir,
            paper_selection_cfg={"iterations": "last", "cumulative": False},
        )

        comparisons = result["comparisons"]
        assert len(comparisons) == 3  # C(3,2) = 3 pairs

        # All completed
        for c in comparisons:
            assert c["status"] == "completed"
            assert c["score"] in (0, 1)

        # Output files exist
        assert os.path.exists(os.path.join(output_dir, "comparisons.json"))
        assert os.path.exists(os.path.join(output_dir, "run_summary.json"))
        assert os.path.exists(os.path.join(output_dir, "ranker_config.json"))

        # Experiment fields are None
        for c in comparisons:
            assert c["paper_a_experiment"] is None
            assert c["paper_b_experiment"] is None
