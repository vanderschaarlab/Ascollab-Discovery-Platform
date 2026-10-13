"""Tests for ranker.sample_and_rank_paired — batch paired pairwise comparison workflow."""

import json
import logging
import os
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from ranker.sample_and_rank_common import (
    _metareview_sort_key,
    _run_single_comparison,
)
from ranker.sample_and_rank_paired import (
    _apply_paper_selection,
    _collect_paper_data,
    _gather_papers,
    _generate_pairs,
    _print_selection_summary,
    run_sample_and_rank_paired,
)
from ranker.snapshot_import import collect_metrics, collect_papers

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


def _make_paper(
    paper_id,
    agent_id,
    pub_t,
    code_script=None,
    metareview=None,
    status="accepted",
    paper_kind="research",
):
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
        "paper_kind": paper_kind,
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
    """Write snapshot files for one iteration.

    Args:
        papers: Flat list of paper dicts (will be wrapped in a nested list
                for private_archive_data.json).
        reviews_by_paper: Optional dict of paper_id -> review list.
        write_public_archive: If True, also write archive_data.json with
            only accepted papers (for backward compat testing).
    """
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

    # Private archive: nested list (one group) + reviews
    if reviews_by_paper is None:
        reviews_by_paper = {}

    private_archive = {
        "papers": [papers],  # Wrap in a single agent group
        "reviews_by_paper": reviews_by_paper,
        "with_citation_boost": False,
        "with_recency_boost": False,
        "w_citation": 0.1,
        "w_recency": 0.1,
        "max_k": 5,
    }
    (iter_dir / "private_archive_data.json").write_text(json.dumps(private_archive, indent=2))

    # Optionally write public archive
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def two_experiment_snapshots(tmp_path):
    """Build two fake experiments on disk.

    Experiment A: 2 snapshot folders (seeds), 2 iterations each, ~2 papers per iteration.
    Experiment B: 1 snapshot folder, 2 iterations, ~2 papers per iteration.
    """
    # ---- Experiment A, seed 1 ----
    exp_a_seed1 = tmp_path / "exp_a_seed_1"
    exp_a_seed1_name = "exp_a_seed_1_260213_1200"

    a1_t1 = [
        _make_paper("paper_t01_agent_01", "agent_01", 1, code_script="print('a1')"),
        _make_paper("paper_t01_agent_02", "agent_02", 1),
    ]
    a1_t2 = a1_t1 + [
        _make_paper("paper_t02_agent_01", "agent_01", 2, code_script="x = 1"),
        _make_paper("paper_t02_agent_02", "agent_02", 2),
    ]
    _write_iteration(exp_a_seed1, 1, a1_t1, exp_a_seed1_name)
    _write_iteration(exp_a_seed1, 2, a1_t2, exp_a_seed1_name)

    # ---- Experiment A, seed 2 ----
    exp_a_seed2 = tmp_path / "exp_a_seed_2"
    exp_a_seed2_name = "exp_a_seed_2_260213_1200"

    a2_t1 = [
        _make_paper("paper_t01_agent_03", "agent_03", 1),
        _make_paper("paper_t01_agent_04", "agent_04", 1, code_script="y = 2"),
    ]
    a2_t2 = a2_t1 + [
        _make_paper("paper_t02_agent_03", "agent_03", 2),
    ]
    _write_iteration(exp_a_seed2, 1, a2_t1, exp_a_seed2_name)
    _write_iteration(exp_a_seed2, 2, a2_t2, exp_a_seed2_name)

    # ---- Experiment B, seed 1 ----
    exp_b_seed1 = tmp_path / "exp_b_seed_1"
    exp_b_seed1_name = "exp_b_seed_1_260213_1300"

    b1_t1 = [
        _make_paper("paper_t01_agent_05", "agent_05", 1, code_script="import os"),
        _make_paper("paper_t01_agent_06", "agent_06", 1),
    ]
    b1_t2 = b1_t1 + [
        _make_paper("paper_t02_agent_05", "agent_05", 2),
        _make_paper("paper_t02_agent_06", "agent_06", 2, code_script="z = 3"),
    ]
    _write_iteration(exp_b_seed1, 1, b1_t1, exp_b_seed1_name)
    _write_iteration(exp_b_seed1, 2, b1_t2, exp_b_seed1_name)

    # Write metrics.jsonl for iteration 2 papers in each snapshot
    # exp_a_seed1: metrics for t=2 papers
    _write_metrics_jsonl(
        exp_a_seed1,
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
    # exp_a_seed1: metrics for t=1 papers
    _write_metrics_jsonl(
        exp_a_seed1,
        1,
        [
            _make_metrics_record(
                "paper_t01_agent_01", "agent_01", "Title of paper_t01_agent_01", 1
            ),
            _make_metrics_record(
                "paper_t01_agent_02", "agent_02", "Title of paper_t01_agent_02", 1
            ),
        ],
    )
    # exp_a_seed2: metrics for t=2 papers
    _write_metrics_jsonl(
        exp_a_seed2,
        2,
        [
            _make_metrics_record(
                "paper_t02_agent_03", "agent_03", "Title of paper_t02_agent_03", 2
            ),
        ],
    )
    # exp_b_seed1: metrics for t=2 papers
    _write_metrics_jsonl(
        exp_b_seed1,
        2,
        [
            _make_metrics_record(
                "paper_t02_agent_05", "agent_05", "Title of paper_t02_agent_05", 2
            ),
            _make_metrics_record(
                "paper_t02_agent_06", "agent_06", "Title of paper_t02_agent_06", 2
            ),
        ],
    )

    return {
        "exp_a_seed1": exp_a_seed1,
        "exp_a_seed1_name": exp_a_seed1_name,
        "exp_a_seed2": exp_a_seed2,
        "exp_a_seed2_name": exp_a_seed2_name,
        "exp_b_seed1": exp_b_seed1,
        "exp_b_seed1_name": exp_b_seed1_name,
        # Convenience: experiment configs as dicts
        "experiments_cfg_single": {
            "experiment_a": {
                "label": "baseline",
                "snapshot_dirs": [str(exp_a_seed1)],
            },
            "experiment_b": {
                "label": "treatment",
                "snapshot_dirs": [str(exp_b_seed1)],
            },
        },
        "experiments_cfg_multi": {
            "experiment_a": {
                "label": "baseline",
                "snapshot_dirs": [str(exp_a_seed1), str(exp_a_seed2)],
            },
            "experiment_b": {
                "label": "treatment",
                "snapshot_dirs": [str(exp_b_seed1)],
            },
        },
        # Paper counts for iterations="last" (pub_t=2):
        # exp_a_seed1: 2 papers at t=2
        # exp_a_seed2: 1 paper at t=2
        # exp_b_seed1: 2 papers at t=2
    }


@pytest.fixture
def mock_llm():
    """MagicMock that returns an AIMessage with a valid verdict on every call."""
    mock = MagicMock()
    mock.invoke.return_value = _make_ai_msg(_VERDICT_A)
    return mock


# ---------------------------------------------------------------------------
# Tests: Paper gathering
# ---------------------------------------------------------------------------


def _metrics_by_pid(papers):
    """collect_metrics keyed by bare paper_id (the ``<snapshot dir>::`` prefix dropped).

    Papers from different snapshots with the same paper_id would collide; the fixtures' IDs differ
    across snapshots, also where one side reads two (TestCollectMetricsMultipleSnapshots).
    """
    return {uid.split("::", 1)[1]: m for uid, m in collect_metrics(papers).items()}


class TestGatherSingleFolderPerExperiment:
    """1. test_gather_single_folder_per_experiment"""

    def test_gather_single_folder_per_experiment(self, two_experiment_snapshots):
        cfg = two_experiment_snapshots["experiments_cfg_single"]
        papers_a, papers_b, info = _gather_papers(
            cfg, paper_selection_cfg={"iterations": "last", "cumulative": True}
        )

        # exp_a_seed1: cumulative up to t=2 -> 2 at t=1 + 2 at t=2 = 4
        assert len(papers_a) == 4
        # exp_b_seed1: cumulative up to t=2 -> 2 at t=1 + 2 at t=2 = 4
        assert len(papers_b) == 4

        # Check provenance
        for p in papers_a:
            assert p["source_snapshot"] == two_experiment_snapshots["exp_a_seed1_name"]
        for p in papers_b:
            assert p["source_snapshot"] == two_experiment_snapshots["exp_b_seed1_name"]


class TestGatherMultipleFolders:
    """2. test_gather_multiple_folders"""

    def test_gather_multiple_folders(self, two_experiment_snapshots):
        cfg = two_experiment_snapshots["experiments_cfg_multi"]
        papers_a, papers_b, info = _gather_papers(
            cfg, paper_selection_cfg={"iterations": "last", "cumulative": True}
        )

        # exp_a_seed1: cumulative t<=2 -> 4, exp_a_seed2: cumulative t<=2 -> 3 -> total 7
        assert len(papers_a) == 7
        # exp_b_seed1: cumulative t<=2 -> 4
        assert len(papers_b) == 4

        # source_snapshot_dir should differ between seeds
        dirs_a = {p["source_snapshot_dir"] for p in papers_a}
        assert len(dirs_a) == 2


class TestGatherIterationLast:
    """3. test_gather_iteration_last"""

    def test_gather_iteration_last(self, two_experiment_snapshots):
        cfg = two_experiment_snapshots["experiments_cfg_single"]
        papers_a, papers_b, _ = _gather_papers(
            cfg, paper_selection_cfg={"iterations": "last", "cumulative": True}
        )

        # Cumulative: all papers up to and including the highest iteration (t<=2)
        for p in papers_a:
            assert p["source_iteration"] <= 2
        for p in papers_b:
            assert p["source_iteration"] <= 2


class TestGatherIterationExplicit:
    """4. test_gather_iteration_explicit"""

    def test_gather_iteration_explicit(self, two_experiment_snapshots):
        cfg = two_experiment_snapshots["experiments_cfg_single"]
        papers_a, papers_b, _ = _gather_papers(
            cfg, paper_selection_cfg={"iterations": [1], "cumulative": True}
        )

        # Only papers from iteration 1
        for p in papers_a:
            assert p["source_iteration"] == 1
        for p in papers_b:
            assert p["source_iteration"] == 1


class TestSourceSnapshotDirPresent:
    """5. test_source_snapshot_dir_present"""

    def test_source_snapshot_dir_present(self, two_experiment_snapshots):
        cfg = two_experiment_snapshots["experiments_cfg_multi"]
        papers_a, papers_b, _ = _gather_papers(
            cfg, paper_selection_cfg={"iterations": "last", "cumulative": False}
        )

        for p in papers_a + papers_b:
            assert "source_snapshot_dir" in p
            assert isinstance(p["source_snapshot_dir"], str)
            assert len(p["source_snapshot_dir"]) > 0

        # Verify each paper's source_snapshot_dir matches one of the input dirs
        all_input_dirs = set()
        for exp_key in ("experiment_a", "experiment_b"):
            for sd in cfg[exp_key]["snapshot_dirs"]:
                all_input_dirs.add(str(sd))

        for p in papers_a + papers_b:
            assert p["source_snapshot_dir"] in all_input_dirs


# ---------------------------------------------------------------------------
# Tests: Pair sampling
# ---------------------------------------------------------------------------


class TestPairsAllMode:
    """6. test_pairs_all_mode"""

    def test_pairs_all_mode(self, two_experiment_snapshots):
        cfg = two_experiment_snapshots["experiments_cfg_single"]
        papers_a, papers_b, _ = _gather_papers(
            cfg, paper_selection_cfg={"iterations": "last", "cumulative": False}
        )

        sampling_cfg = {"mode": "all", "n_repeats": 1, "seed": 42}
        tasks = _generate_pairs(papers_a, papers_b, sampling_cfg, "baseline", "treatment")

        expected = len(papers_a) * len(papers_b)
        assert len(tasks) == expected


class TestPairsAllModeNRepeats:
    """7. test_pairs_all_mode_n_repeats"""

    def test_pairs_all_mode_n_repeats(self, two_experiment_snapshots):
        cfg = two_experiment_snapshots["experiments_cfg_single"]
        papers_a, papers_b, _ = _gather_papers(
            cfg, paper_selection_cfg={"iterations": "last", "cumulative": False}
        )

        sampling_cfg = {"mode": "all", "n_repeats": 3, "seed": 42}
        tasks = _generate_pairs(papers_a, papers_b, sampling_cfg, "baseline", "treatment")

        expected = len(papers_a) * len(papers_b) * 3
        assert len(tasks) == expected

        # Each unique pair (by paper IDs) should appear exactly 3 times
        from collections import Counter

        pair_counter = Counter()
        for t in tasks:
            pair = (t["paper_a"]["paper_id"], t["paper_b"]["paper_id"])
            pair_counter[pair] += 1

        # Due to A/B randomization, the same logical pair may appear with swapped IDs.
        # Instead, check total tasks = unique_pairs * n_repeats and repeat_index range.
        repeat_indices = [t["repeat_index"] for t in tasks]
        assert set(repeat_indices) == {0, 1, 2}


class TestPairsRandomWithoutReplacement:
    """8. test_pairs_random_without_replacement"""

    def test_pairs_random_without_replacement(self, two_experiment_snapshots):
        cfg = two_experiment_snapshots["experiments_cfg_single"]
        papers_a, papers_b, _ = _gather_papers(
            cfg, paper_selection_cfg={"iterations": "last", "cumulative": False}
        )

        product_size = len(papers_a) * len(papers_b)
        n_pairs = min(product_size, 3)

        sampling_cfg = {"mode": "random", "n_pairs": n_pairs, "with_replacement": False, "seed": 42}
        tasks = _generate_pairs(papers_a, papers_b, sampling_cfg, "baseline", "treatment")

        assert len(tasks) == n_pairs

        # No duplicate comparison IDs
        ids = [t["comparison_id"] for t in tasks]
        assert len(ids) == len(set(ids))

        # No duplicate logical paper pairs (regardless of A/B position)
        logical_pairs = set()
        for t in tasks:
            pair = frozenset([t["paper_a"]["paper_id"], t["paper_b"]["paper_id"]])
            assert pair not in logical_pairs, f"Duplicate logical pair: {pair}"
            logical_pairs.add(pair)


class TestPairsRandomWithReplacement:
    """9. test_pairs_random_with_replacement"""

    def test_pairs_random_with_replacement(self, two_experiment_snapshots):
        cfg = two_experiment_snapshots["experiments_cfg_single"]
        papers_a, papers_b, _ = _gather_papers(
            cfg, paper_selection_cfg={"iterations": "last", "cumulative": False}
        )

        product_size = len(papers_a) * len(papers_b)
        n_pairs = product_size + 5  # More than product size

        sampling_cfg = {"mode": "random", "n_pairs": n_pairs, "with_replacement": True, "seed": 42}
        tasks = _generate_pairs(papers_a, papers_b, sampling_cfg, "baseline", "treatment")

        # No error raised, correct count
        assert len(tasks) == n_pairs


class TestPairsRandomWithoutReplacementOverflow:
    """10. test_pairs_random_without_replacement_overflow"""

    def test_pairs_random_without_replacement_overflow(self, two_experiment_snapshots):
        cfg = two_experiment_snapshots["experiments_cfg_single"]
        papers_a, papers_b, _ = _gather_papers(
            cfg, paper_selection_cfg={"iterations": "last", "cumulative": False}
        )

        product_size = len(papers_a) * len(papers_b)
        n_pairs = product_size + 1

        sampling_cfg = {"mode": "random", "n_pairs": n_pairs, "with_replacement": False, "seed": 42}
        with pytest.raises(ValueError, match="Cannot sample"):
            _generate_pairs(papers_a, papers_b, sampling_cfg, "baseline", "treatment")

    def test_cap_n_pairs_judges_every_possible_pair(self, two_experiment_snapshots, caplog):
        # What 04's rank.sh asks for: 500 pairs, or all of them when fewer exist.
        cfg = two_experiment_snapshots["experiments_cfg_single"]
        papers_a, papers_b, _ = _gather_papers(
            cfg, paper_selection_cfg={"iterations": "last", "cumulative": False}
        )
        product_size = len(papers_a) * len(papers_b)
        sampling_cfg = {"mode": "random", "n_pairs": 500, "cap_n_pairs": True, "seed": 42}

        with caplog.at_level(logging.WARNING, logger="ranker.sample_and_rank_paired"):
            tasks = _generate_pairs(papers_a, papers_b, sampling_cfg, "baseline", "treatment")

        pairs = {frozenset((t["paper_a"]["paper_id"], t["paper_b"]["paper_id"])) for t in tasks}
        assert len(tasks) == len(pairs) == product_size
        assert f"Only {product_size} pairs are possible" in caplog.text
        # Enough pairs: the cap changes nothing.
        enough = {"mode": "random", "n_pairs": product_size, "seed": 42}
        assert _generate_pairs(papers_a, papers_b, {**enough, "cap_n_pairs": True}, "a", "b") == (
            _generate_pairs(papers_a, papers_b, enough, "a", "b")
        )


class TestPairsRandomReproducible:
    """11. test_pairs_random_reproducible"""

    def test_pairs_random_reproducible(self, two_experiment_snapshots):
        cfg = two_experiment_snapshots["experiments_cfg_single"]
        papers_a, papers_b, _ = _gather_papers(
            cfg, paper_selection_cfg={"iterations": "last", "cumulative": False}
        )

        sampling_cfg_42 = {"mode": "random", "n_pairs": 3, "with_replacement": False, "seed": 42}
        tasks_42a = _generate_pairs(papers_a, papers_b, sampling_cfg_42, "baseline", "treatment")
        tasks_42b = _generate_pairs(papers_a, papers_b, sampling_cfg_42, "baseline", "treatment")

        # Same seed -> same pairs (same comparison_ids and paper assignments)
        ids_a = [(t["paper_a"]["paper_id"], t["paper_b"]["paper_id"]) for t in tasks_42a]
        ids_b = [(t["paper_a"]["paper_id"], t["paper_b"]["paper_id"]) for t in tasks_42b]
        assert ids_a == ids_b

        # Different seed -> different pairs (with high probability)
        sampling_cfg_99 = {"mode": "random", "n_pairs": 3, "with_replacement": False, "seed": 99}
        tasks_99 = _generate_pairs(papers_a, papers_b, sampling_cfg_99, "baseline", "treatment")
        ids_99 = [(t["paper_a"]["paper_id"], t["paper_b"]["paper_id"]) for t in tasks_99]
        # Very unlikely to match (4 choose 3 = 4 options, different seed should differ)
        assert ids_a != ids_99


class TestPairsABRandomization:
    """12. test_pairs_ab_randomization"""

    def test_pairs_ab_randomization(self, two_experiment_snapshots):
        cfg = two_experiment_snapshots["experiments_cfg_multi"]
        papers_a, papers_b, _ = _gather_papers(
            cfg, paper_selection_cfg={"iterations": "last", "cumulative": False}
        )

        # Use all mode with enough pairs for randomization to show both orderings
        sampling_cfg = {"mode": "all", "n_repeats": 1, "seed": 42}
        tasks = _generate_pairs(papers_a, papers_b, sampling_cfg, "baseline", "treatment")

        # Check that both orderings appear (baseline as A and treatment as A)
        a_experiments = {t["paper_a_experiment"] for t in tasks}
        # With enough pairs, both "baseline" and "treatment" should appear as paper_a
        assert "baseline" in a_experiments or "treatment" in a_experiments
        # With randomization, we expect both
        assert len(a_experiments) == 2, (
            f"Expected both experiments in A position, got: {a_experiments}"
        )


# ---------------------------------------------------------------------------
# Tests: Ranking (mocked LLM)
# ---------------------------------------------------------------------------


class TestSingleComparisonSuccess:
    """13. test_single_comparison_success"""

    def test_single_comparison_success(self, two_experiment_snapshots, mock_llm):
        cfg = two_experiment_snapshots["experiments_cfg_single"]
        papers_a, papers_b, _ = _gather_papers(
            cfg, paper_selection_cfg={"iterations": "last", "cumulative": False}
        )

        sampling_cfg = {"mode": "all", "n_repeats": 1, "seed": 42}
        tasks = _generate_pairs(papers_a, papers_b, sampling_cfg, "baseline", "treatment")

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
    """14. test_single_comparison_failure"""

    def test_single_comparison_failure(self, two_experiment_snapshots):
        cfg = two_experiment_snapshots["experiments_cfg_single"]
        papers_a, papers_b, _ = _gather_papers(
            cfg, paper_selection_cfg={"iterations": "last", "cumulative": False}
        )

        sampling_cfg = {"mode": "all", "n_repeats": 1, "seed": 42}
        tasks = _generate_pairs(papers_a, papers_b, sampling_cfg, "baseline", "treatment")

        # LLM that always raises
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


class TestSingleComparisonUnparsedVerdict:
    """An unparseable verdict must not be recorded as a Paper B win."""

    def test_unparsed_verdict_records_no_decision(self, two_experiment_snapshots, monkeypatch):
        from ranker import sample_and_rank_common
        from ranker.ranker_schemas import RankerResult

        cfg = two_experiment_snapshots["experiments_cfg_single"]
        papers_a, papers_b, _ = _gather_papers(
            cfg, paper_selection_cfg={"iterations": "last", "cumulative": False}
        )
        tasks = _generate_pairs(
            papers_a, papers_b, {"mode": "all", "n_repeats": 1, "seed": 42}, "baseline", "treatment"
        )

        monkeypatch.setattr(
            sample_and_rank_common,
            "run_ranker",
            lambda **kwargs: RankerResult(
                score=None, valid_result=False, justification="", raw_response=""
            ),
        )

        record = _run_single_comparison(
            task=tasks[0],
            llm=MagicMock(),
            ranker_cfg={"max_iterations": 5, "tools": {}, "llm_retry": {}},
            sandbox_client=None,
            iteration_selection="last",
        )

        assert record["status"] == "unparsed_verdict"
        assert record["score"] is None
        assert record["outcome"] is None
        assert record["error"] is None


# ---------------------------------------------------------------------------
# Tests: End-to-end (mocked LLM)
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """15. test_end_to_end"""

    def test_end_to_end(self, two_experiment_snapshots, mock_llm, tmp_path):
        cfg = two_experiment_snapshots["experiments_cfg_single"]
        output_dir = str(tmp_path / "output")

        ranker_cfg = {
            "max_iterations": 5,
            "tools": {},
            "llm_retry": {},
            "output_parsing": {"max_parse_retries": 2},
        }

        result = run_sample_and_rank_paired(
            experiments_cfg=cfg,
            sampling_cfg={"mode": "all", "n_repeats": 1, "seed": 42},
            ranker_cfg=ranker_cfg,
            parallelism_cfg={"max_workers": 2},
            llm=mock_llm,
            run_id="test_run_001",
            output_dir=output_dir,
            paper_selection_cfg={"iterations": "last", "cumulative": False},
        )

        # Return dict has required keys
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

        # run_summary.json exists with correct counts
        summary_path = os.path.join(output_dir, "run_summary.json")
        assert os.path.exists(summary_path)
        with open(summary_path) as f:
            summary_data = json.load(f)
        assert summary_data["results"]["completed"] + summary_data["results"]["failed"] == len(
            comparisons_data
        )

        # ranker_config.json exists
        config_path = os.path.join(output_dir, "ranker_config.json")
        assert os.path.exists(config_path)

        # experiment_configs/ has YAML files
        exp_configs_dir = os.path.join(output_dir, "experiment_configs")
        assert os.path.isdir(exp_configs_dir)
        yaml_files = [f for f in os.listdir(exp_configs_dir) if f.endswith(".yaml")]
        # Should have one per snapshot folder (2 dirs in single mode: 1 for A, 1 for B)
        assert len(yaml_files) >= 1

        # ranker_traces/ has one JSON per completed comparison
        traces_dir = os.path.join(output_dir, "ranker_traces")
        if summary_data["results"]["completed"] > 0:
            assert os.path.isdir(traces_dir)
            trace_files = [f for f in os.listdir(traces_dir) if f.endswith(".json")]
            assert len(trace_files) == summary_data["results"]["completed"]


class TestComparisonsPandasLoadable:
    """16. test_comparisons_pandas_loadable"""

    def test_comparisons_pandas_loadable(self, two_experiment_snapshots, mock_llm, tmp_path):
        pd = pytest.importorskip("pandas")

        cfg = two_experiment_snapshots["experiments_cfg_single"]
        output_dir = str(tmp_path / "output_pandas")

        ranker_cfg = {
            "max_iterations": 5,
            "tools": {},
            "llm_retry": {},
            "output_parsing": {"max_parse_retries": 2},
        }

        run_sample_and_rank_paired(
            experiments_cfg=cfg,
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

        # Check expected columns are present
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


class TestFailedComparisonInOutput:
    """17. test_failed_comparison_in_output"""

    def test_failed_comparison_in_output(self, two_experiment_snapshots, tmp_path):
        cfg = two_experiment_snapshots["experiments_cfg_single"]
        output_dir = str(tmp_path / "output_failures")

        call_count = 0

        def mock_invoke(messages):
            nonlocal call_count
            call_count += 1
            # Fail on the first comparison's first call; succeed on all others
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

        result = run_sample_and_rank_paired(
            experiments_cfg=cfg,
            sampling_cfg={"mode": "all", "n_repeats": 1, "seed": 42},
            ranker_cfg=ranker_cfg,
            parallelism_cfg={"max_workers": 1},  # Sequential to control failure order
            llm=mixed_llm,
            run_id="test_fail_run",
            output_dir=output_dir,
            paper_selection_cfg={"iterations": "last", "cumulative": False},
        )

        comparisons = result["comparisons"]

        # At least one failed and others completed
        statuses = {c["status"] for c in comparisons}
        assert "failed" in statuses
        assert "completed" in statuses

        # Failed one has error message
        failed = [c for c in comparisons if c["status"] == "failed"]
        assert len(failed) >= 1
        for f in failed:
            assert f["error"] is not None

        # comparisons.json contains both
        comparisons_path = os.path.join(output_dir, "comparisons.json")
        with open(comparisons_path) as fh:
            data = json.load(fh)
        statuses_file = {c["status"] for c in data}
        assert "failed" in statuses_file
        assert "completed" in statuses_file


# ---------------------------------------------------------------------------
# Tests: Paper data collection (renamed from paper metrics)
# ---------------------------------------------------------------------------


class TestCollectMetricsReadsCorrectIteration:
    """19. test_collect_metrics_reads_correct_iteration

    Synthetic snapshot with multiple iteration_N/metrics.jsonl files;
    verify the correct file is read per paper based on source_iteration.
    """

    def test_collect_metrics_reads_correct_iteration(self, tmp_path):
        snap = tmp_path / "snap_multi_iter"

        # Iteration 1: paper at t=1
        papers_t1 = [_make_paper("paper_t01_agent_01", "agent_01", 1)]
        _write_iteration(snap, 1, papers_t1, "snap_run")
        _write_metrics_jsonl(
            snap,
            1,
            [
                _make_metrics_record("paper_t01_agent_01", "agent_01", "Title t01", 1),
            ],
        )

        # Iteration 2: cumulative archive has t1+t2 papers
        papers_t2 = papers_t1 + [_make_paper("paper_t02_agent_02", "agent_02", 2)]
        _write_iteration(snap, 2, papers_t2, "snap_run")
        _write_metrics_jsonl(
            snap,
            2,
            [
                _make_metrics_record("paper_t02_agent_02", "agent_02", "Title t02", 2),
            ],
        )

        # Collect all papers (both iterations)
        papers = collect_papers(str(snap), iterations="all")
        assert len(papers) == 2

        metrics = _metrics_by_pid(papers)

        # Both papers should have metrics
        assert "paper_t01_agent_01" in metrics
        assert "paper_t02_agent_02" in metrics

        # Verify correct iteration was read for each
        assert metrics["paper_t01_agent_01"]["iteration"] == 1
        assert metrics["paper_t02_agent_02"]["iteration"] == 2


class TestCollectMetricsMissingFile:
    """20. test_collect_metrics_missing_file

    metrics.jsonl absent for an iteration; verify no error, paper omitted from result.
    """

    def test_collect_metrics_missing_file(self, tmp_path):
        snap = tmp_path / "snap_no_metrics"

        # Iteration 1: no metrics.jsonl
        papers_t1 = [_make_paper("paper_t01_agent_01", "agent_01", 1)]
        _write_iteration(snap, 1, papers_t1, "snap_run")
        # No _write_metrics_jsonl call

        # Iteration 2: has metrics
        papers_t2 = papers_t1 + [_make_paper("paper_t02_agent_02", "agent_02", 2)]
        _write_iteration(snap, 2, papers_t2, "snap_run")
        _write_metrics_jsonl(
            snap,
            2,
            [
                _make_metrics_record("paper_t02_agent_02", "agent_02", "Title t02", 2),
            ],
        )

        papers = collect_papers(str(snap), iterations="all")
        metrics = _metrics_by_pid(papers)

        # t=1 paper has no metrics (file missing), t=2 paper does
        assert "paper_t01_agent_01" not in metrics
        assert "paper_t02_agent_02" in metrics


class TestCollectMetricsMultipleSnapshots:
    """21. test_collect_metrics_multiple_snapshots

    Papers from two snapshot dirs; verify metrics are gathered from both.
    """

    def test_collect_metrics_multiple_snapshots(self, two_experiment_snapshots):
        cfg = two_experiment_snapshots["experiments_cfg_multi"]
        papers_a, papers_b, _ = _gather_papers(
            cfg, paper_selection_cfg={"iterations": "last", "cumulative": False}
        )

        # papers_a: 2 from seed1 (t=2) + 1 from seed2 (t=2) = 3
        # papers_b: 2 from seed1 (t=2)
        metrics_a = _metrics_by_pid(papers_a)
        metrics_b = _metrics_by_pid(papers_b)

        # All 3 experiment A papers should have metrics
        assert len(metrics_a) == 3
        assert "paper_t02_agent_01" in metrics_a
        assert "paper_t02_agent_02" in metrics_a
        assert "paper_t02_agent_03" in metrics_a

        # All 2 experiment B papers should have metrics
        assert len(metrics_b) == 2
        assert "paper_t02_agent_05" in metrics_b
        assert "paper_t02_agent_06" in metrics_b

        # Verify the metrics came from the right snapshots
        dirs_a = {metrics_a[pid]["paper_id"]: pid for pid in metrics_a}
        assert len(dirs_a) == 3


class TestPaperDataSavedInOutput:
    """22. test_paper_data_saved_in_output

    End-to-end run with mocked ranker; verify papers/paper_data.json exists
    and is valid JSON with the expected records.
    """

    def test_paper_data_saved_in_output(self, two_experiment_snapshots, mock_llm, tmp_path):
        cfg = two_experiment_snapshots["experiments_cfg_single"]
        output_dir = str(tmp_path / "output_paper_data")

        ranker_cfg = {
            "max_iterations": 5,
            "tools": {},
            "llm_retry": {},
            "output_parsing": {"max_parse_retries": 2},
        }

        result = run_sample_and_rank_paired(  # noqa: F841
            experiments_cfg=cfg,
            sampling_cfg={"mode": "all", "n_repeats": 1, "seed": 42},
            ranker_cfg=ranker_cfg,
            parallelism_cfg={"max_workers": 2},
            llm=mock_llm,
            run_id="test_paper_data_run",
            output_dir=output_dir,
            paper_selection_cfg={"iterations": "last", "cumulative": False},
        )

        # papers/paper_data.json should exist
        paper_data_path = os.path.join(output_dir, "papers", "paper_data.json")
        assert os.path.exists(paper_data_path)

        # Old path should NOT exist
        old_metrics_path = os.path.join(output_dir, "paper_metrics", "metrics.json")
        assert not os.path.exists(old_metrics_path)

        with open(paper_data_path) as f:
            paper_data = json.load(f)

        assert isinstance(paper_data, list)
        # With iterations="last", cumulative=False (t=2): 2 papers from A + 2 from B = 4
        assert len(paper_data) == 4

        # Each record has the expected fields
        for rec in paper_data:
            assert "paper_id" in rec
            assert "experiment" in rec
            assert "experiment_condition" in rec
            assert "source_snapshot" in rec
            assert "source_snapshot_dir" in rec
            assert "source_iteration" in rec
            assert "primary_agent_id" in rec
            assert "title" in rec
            assert "metrics" in rec
            assert "metareview" in rec
            assert "reviews" in rec
            assert "status" in rec

        # All papers in this test have system metrics (metrics.jsonl exists)
        for rec in paper_data:
            assert isinstance(rec["metrics"], dict)

        # Verify experiment labels are correct
        experiments = {rec["experiment"] for rec in paper_data}
        assert experiments == {"baseline", "treatment"}

        # Verify experiment_condition is A or B
        conditions = {rec["experiment_condition"] for rec in paper_data}
        assert conditions == {"A", "B"}

        # Verify run_summary.json has paper_data section (NOT paper_metrics)
        summary_path = os.path.join(output_dir, "run_summary.json")
        with open(summary_path) as f:
            summary = json.load(f)
        assert "paper_data" in summary
        assert "paper_metrics" not in summary
        assert summary["paper_data"]["total"] == 4
        assert summary["paper_data"]["without_system_metrics"] == 0


class TestPaperDataWithoutSystemMetrics:
    """23. Papers without system metrics still appear in paper_data.json.

    No metrics.jsonl in any snapshot; verify papers/paper_data.json
    is created with all papers, each having metrics=null.
    """

    def test_papers_without_system_metrics(self, mock_llm, tmp_path):
        # Build snapshots without any metrics.jsonl
        exp_a = tmp_path / "no_metrics_a"
        exp_b = tmp_path / "no_metrics_b"

        papers_a = [_make_paper("paper_t01_agent_01", "agent_01", 1)]
        papers_b = [_make_paper("paper_t01_agent_02", "agent_02", 1)]

        _write_iteration(exp_a, 1, papers_a, "no_metrics_a_run")
        _write_iteration(exp_b, 1, papers_b, "no_metrics_b_run")

        output_dir = str(tmp_path / "output_no_metrics")

        experiments_cfg = {
            "experiment_a": {
                "label": "baseline",
                "snapshot_dirs": [str(exp_a)],
            },
            "experiment_b": {
                "label": "treatment",
                "snapshot_dirs": [str(exp_b)],
            },
        }

        ranker_cfg = {
            "max_iterations": 5,
            "tools": {},
            "llm_retry": {},
            "output_parsing": {"max_parse_retries": 2},
        }

        result = run_sample_and_rank_paired(  # noqa: F841
            experiments_cfg=experiments_cfg,
            sampling_cfg={"mode": "all", "n_repeats": 1, "seed": 42},
            ranker_cfg=ranker_cfg,
            parallelism_cfg={"max_workers": 1},
            llm=mock_llm,
            run_id="test_no_metrics_run",
            output_dir=output_dir,
            paper_selection_cfg={"iterations": "last", "cumulative": False},
        )

        # papers/paper_data.json IS created (all papers included)
        paper_data_path = os.path.join(output_dir, "papers", "paper_data.json")
        assert os.path.exists(paper_data_path)

        with open(paper_data_path) as f:
            paper_data = json.load(f)
        assert len(paper_data) == 2
        for rec in paper_data:
            assert rec["metrics"] is None
            assert rec["metareview"] is None

        # Summary should reflect no system metrics
        summary_path = os.path.join(output_dir, "run_summary.json")
        with open(summary_path) as f:
            summary = json.load(f)
        assert summary["paper_data"]["total"] == 2
        assert summary["paper_data"]["without_system_metrics"] == 2


# ---------------------------------------------------------------------------
# Tests: Integration (marked, requires real LLM)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestIntegrationRealLlm:
    """18. test_integration_real_llm"""

    def test_integration_real_llm(self, llm, tmp_path):
        # Build minimal synthetic snapshots: 2 papers in A, 1 paper in B -> 2 comparisons
        exp_a = tmp_path / "int_exp_a"
        exp_b = tmp_path / "int_exp_b"

        papers_a = [
            _make_paper("paper_t01_agent_01", "agent_01", 1, code_script="print('hello')"),
            _make_paper("paper_t01_agent_02", "agent_02", 1),
        ]
        papers_b = [
            _make_paper("paper_t01_agent_03", "agent_03", 1),
        ]

        _write_iteration(exp_a, 1, papers_a, "int_exp_a_name")
        _write_iteration(exp_b, 1, papers_b, "int_exp_b_name")

        output_dir = str(tmp_path / "int_output")

        experiments_cfg = {
            "experiment_a": {
                "label": "baseline",
                "snapshot_dirs": [str(exp_a)],
            },
            "experiment_b": {
                "label": "treatment",
                "snapshot_dirs": [str(exp_b)],
            },
        }

        ranker_cfg = {
            "max_iterations": 5,
            "tools": {},
            "llm_retry": {"max_retries": 3, "base_delay_seconds": 2.0, "max_delay_seconds": 30.0},
            "output_parsing": {"max_parse_retries": 2},
        }

        result = run_sample_and_rank_paired(
            experiments_cfg=experiments_cfg,
            sampling_cfg={"mode": "all", "n_repeats": 1, "seed": 42},
            ranker_cfg=ranker_cfg,
            parallelism_cfg={"max_workers": 2},
            llm=llm,
            run_id="integration_test",
            output_dir=output_dir,
            paper_selection_cfg={"iterations": "last", "cumulative": False},
        )

        comparisons = result["comparisons"]
        assert len(comparisons) == 2  # 2 x 1 = 2 pairs

        # All completed
        for c in comparisons:
            assert c["status"] == "completed"
            assert c["score"] in (0, 1)

        # Output files exist
        assert os.path.exists(os.path.join(output_dir, "comparisons.json"))
        assert os.path.exists(os.path.join(output_dir, "run_summary.json"))
        assert os.path.exists(os.path.join(output_dir, "ranker_config.json"))


# ---------------------------------------------------------------------------
# Tests: Cumulative iteration filtering
# ---------------------------------------------------------------------------


class TestGatherCumulativeFalse:
    """Verify old (per-iteration) behaviour is accessible via cumulative=False."""

    def test_gather_cumulative_false(self, two_experiment_snapshots):
        cfg = two_experiment_snapshots["experiments_cfg_single"]
        papers_a, papers_b, _ = _gather_papers(
            cfg, paper_selection_cfg={"iterations": "last", "cumulative": False}
        )
        # Old behaviour: only t=2
        assert len(papers_a) == 2
        assert all(p["source_iteration"] == 2 for p in papers_a)


class TestCumulativeInComparisonRecord:
    """Verify the cumulative flag is persisted in records and summary."""

    def test_cumulative_in_record(self, two_experiment_snapshots, mock_llm, tmp_path):
        cfg = two_experiment_snapshots["experiments_cfg_single"]
        output_dir = str(tmp_path / "output_cumulative")

        ranker_cfg = {
            "max_iterations": 5,
            "tools": {},
            "llm_retry": {},
            "output_parsing": {"max_parse_retries": 2},
        }

        result = run_sample_and_rank_paired(
            experiments_cfg=cfg,
            sampling_cfg={"mode": "all", "n_repeats": 1, "seed": 42},
            ranker_cfg=ranker_cfg,
            parallelism_cfg={"max_workers": 2},
            llm=mock_llm,
            run_id="test_cumulative_run",
            output_dir=output_dir,
            paper_selection_cfg={"iterations": "last", "cumulative": True},
        )

        for c in result["comparisons"]:
            assert "cumulative" in c
            assert c["cumulative"] is True
        assert result["summary"]["cumulative"] is True


# ---------------------------------------------------------------------------
# Tests: Metareview passthrough in collect_papers
# ---------------------------------------------------------------------------


class TestMetareviewPassthrough:
    """Verify collect_papers() carries metareview data through."""

    def test_metareview_present(self, tmp_path):
        snap = tmp_path / "snap_mr"
        mr = _make_metareview("paper_t01_agent_01", 0.75)
        papers = [_make_paper("paper_t01_agent_01", "agent_01", 1, metareview=mr)]
        _write_iteration(snap, 1, papers, "snap_mr_run")

        collected = collect_papers(str(snap), iterations="last")
        assert len(collected) == 1
        assert collected[0]["metareview"] is not None
        assert collected[0]["metareview"]["overall_score"] == 0.75
        assert collected[0]["metareview"]["paper_id"] == "paper_t01_agent_01"

    def test_metareview_null(self, tmp_path):
        snap = tmp_path / "snap_mr_null"
        papers = [_make_paper("paper_t01_agent_01", "agent_01", 1)]
        _write_iteration(snap, 1, papers, "snap_mr_null_run")

        collected = collect_papers(str(snap), iterations="last")
        assert len(collected) == 1
        assert collected[0]["metareview"] is None

    def test_to_ranker_format_excludes_metareview(self, tmp_path):
        from ranker.snapshot_import import to_ranker_format

        snap = tmp_path / "snap_mr_strip"
        mr = _make_metareview("paper_t01_agent_01", 0.5)
        papers = [_make_paper("paper_t01_agent_01", "agent_01", 1, metareview=mr)]
        _write_iteration(snap, 1, papers, "snap_mr_strip_run")

        collected = collect_papers(str(snap), iterations="last")
        ranker_dict = to_ranker_format(collected[0])
        assert "metareview" not in ranker_dict
        assert set(ranker_dict.keys()) == {"title", "abstract", "manuscript", "code"}


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
# Tests: Paper selection -- Top-N
# ---------------------------------------------------------------------------


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


class TestTopNDisabled:
    """Top-N disabled: no filtering occurs."""

    def test_top_n_disabled(self):
        papers_a = [_make_collected_paper(f"a{i}", f"ag{i}", 1, 0.5) for i in range(5)]
        papers_b = [_make_collected_paper(f"b{i}", f"ag{i}", 1, 0.5) for i in range(3)]

        cfg = {"top_n": {"enabled": False}, "balance": {"enabled": False}}
        result_a, result_b, info = _apply_paper_selection(papers_a, papers_b, cfg)

        assert len(result_a) == 5
        assert len(result_b) == 3
        assert info["top_n"]["enabled"] is False


class TestTopNBasic:
    """Top-N with enough papers: each set gets exactly N papers, sorted by score."""

    def test_top_n_basic(self):
        papers_a = [
            _make_collected_paper("a0", "ag0", 1, 0.3),
            _make_collected_paper("a1", "ag1", 1, 0.9),
            _make_collected_paper("a2", "ag2", 1, 0.7),
            _make_collected_paper("a3", "ag3", 1, 0.1),
            _make_collected_paper("a4", "ag4", 1, 0.5),
        ]
        papers_b = [
            _make_collected_paper("b0", "ag0", 1, 0.2),
            _make_collected_paper("b1", "ag1", 1, 0.8),
            _make_collected_paper("b2", "ag2", 1, 0.6),
        ]

        cfg = {"top_n": {"enabled": True, "n": 3}, "balance": {"enabled": False}}
        result_a, result_b, info = _apply_paper_selection(papers_a, papers_b, cfg)

        assert len(result_a) == 3
        assert len(result_b) == 3

        # Top 3 from A by score: a1(0.9), a2(0.7), a4(0.5)
        ids_a = [p["paper_id"] for p in result_a]
        assert ids_a == ["a1", "a2", "a4"]

        # Top 3 from B (all 3): b1(0.8), b2(0.6), b0(0.2)
        ids_b = [p["paper_id"] for p in result_b]
        assert ids_b == ["b1", "b2", "b0"]


class TestTopNFewerThanN:
    """Top-N with fewer papers than N: all papers kept, no error."""

    def test_top_n_fewer_than_n(self):
        papers_a = [_make_collected_paper("a0", "ag0", 1, 0.5)]
        papers_b = [_make_collected_paper("b0", "ag0", 1, 0.3)]

        cfg = {"top_n": {"enabled": True, "n": 10}, "balance": {"enabled": False}}
        result_a, result_b, info = _apply_paper_selection(papers_a, papers_b, cfg)

        assert len(result_a) == 1
        assert len(result_b) == 1


class TestTopNNullMetareviews:
    """Top-N with null metareviews: papers without scores sort to the bottom."""

    def test_top_n_null_metareviews(self):
        papers_a = [
            _make_collected_paper("a0", "ag0", 1, 0.9),
            _make_collected_paper("a1", "ag1", 1, None),  # no metareview
            _make_collected_paper("a2", "ag2", 1, 0.5),
            _make_collected_paper("a3", "ag3", 1, None),  # no metareview
        ]
        papers_b = [_make_collected_paper("b0", "ag0", 1, 0.5)]

        cfg = {"top_n": {"enabled": True, "n": 2}, "balance": {"enabled": False}}
        result_a, result_b, info = _apply_paper_selection(papers_a, papers_b, cfg)

        assert len(result_a) == 2
        # a0(0.9) and a2(0.5) should be selected (null scores excluded)
        ids = [p["paper_id"] for p in result_a]
        assert ids == ["a0", "a2"]


# ---------------------------------------------------------------------------
# Tests: Paper selection -- Per-iteration Top-N (paired)
# ---------------------------------------------------------------------------


class TestPerIterationTopNPaired:
    """Per-iteration top-N applied independently to each experiment."""

    def test_per_iteration_top_n_paired(self):
        # Experiment A: 2 iterations, 4 papers each
        papers_a = [
            _make_collected_paper("a1", "ag1", 1, 0.9),
            _make_collected_paper("a2", "ag2", 1, 0.7),
            _make_collected_paper("a3", "ag3", 1, 0.3),
            _make_collected_paper("a4", "ag4", 1, 0.1),
            _make_collected_paper("a5", "ag5", 2, 0.8),
            _make_collected_paper("a6", "ag6", 2, 0.6),
            _make_collected_paper("a7", "ag7", 2, 0.4),
            _make_collected_paper("a8", "ag8", 2, 0.2),
        ]
        # Experiment B: 3 iterations, 3 papers each
        papers_b = [
            _make_collected_paper("b1", "ag1", 1, 0.9),
            _make_collected_paper("b2", "ag2", 1, 0.5),
            _make_collected_paper("b3", "ag3", 1, 0.3),
            _make_collected_paper("b4", "ag4", 2, 0.8),
            _make_collected_paper("b5", "ag5", 2, 0.6),
            _make_collected_paper("b6", "ag6", 2, 0.4),
            _make_collected_paper("b7", "ag7", 3, 1.0),
            _make_collected_paper("b8", "ag8", 3, 0.7),
            _make_collected_paper("b9", "ag9", 3, 0.2),
        ]

        cfg = {
            "top_n": {"enabled": True, "n": 2, "per_iteration": True},
            "balance": {"enabled": False},
        }
        result_a, result_b, info = _apply_paper_selection(papers_a, papers_b, cfg)

        # A: 2 iterations × 2 = 4 papers
        assert len(result_a) == 4
        assert info["top_n"]["count_a_after"] == 4
        # B: 3 iterations × 2 = 6 papers
        assert len(result_b) == 6
        assert info["top_n"]["count_b_after"] == 6
        assert info["top_n"]["per_iteration"] is True


class TestPerIterationFalseIsGlobalTopNPaired:
    """per_iteration=False preserves existing global top-N behaviour."""

    def test_per_iteration_false_is_global_top_n_paired(self):
        # Same data as above
        papers_a = [
            _make_collected_paper("a1", "ag1", 1, 0.9),
            _make_collected_paper("a2", "ag2", 1, 0.7),
            _make_collected_paper("a3", "ag3", 1, 0.3),
            _make_collected_paper("a4", "ag4", 1, 0.1),
            _make_collected_paper("a5", "ag5", 2, 0.8),
            _make_collected_paper("a6", "ag6", 2, 0.6),
            _make_collected_paper("a7", "ag7", 2, 0.4),
            _make_collected_paper("a8", "ag8", 2, 0.2),
        ]
        papers_b = [
            _make_collected_paper("b1", "ag1", 1, 0.9),
            _make_collected_paper("b2", "ag2", 1, 0.5),
            _make_collected_paper("b3", "ag3", 1, 0.3),
            _make_collected_paper("b4", "ag4", 2, 0.8),
            _make_collected_paper("b5", "ag5", 2, 0.6),
            _make_collected_paper("b6", "ag6", 2, 0.4),
            _make_collected_paper("b7", "ag7", 3, 1.0),
            _make_collected_paper("b8", "ag8", 3, 0.7),
            _make_collected_paper("b9", "ag9", 3, 0.2),
        ]

        cfg = {
            "top_n": {"enabled": True, "n": 2, "per_iteration": False},
            "balance": {"enabled": False},
        }
        result_a, result_b, info = _apply_paper_selection(papers_a, papers_b, cfg)

        # Global top-2 from A: a1(0.9), a5(0.8)
        assert len(result_a) == 2
        ids_a = {p["paper_id"] for p in result_a}
        assert ids_a == {"a1", "a5"}
        # Global top-2 from B: b7(1.0), b1(0.9)
        assert len(result_b) == 2
        ids_b = {p["paper_id"] for p in result_b}
        assert ids_b == {"b7", "b1"}
        assert info["top_n"]["per_iteration"] is False


class TestPerIterationTopNThenBalancePaired:
    """Per-iteration top-N followed by balance equalization."""

    def test_per_iteration_top_n_then_balance_paired(self):
        # Experiment A: 2 iterations × 3 papers each
        papers_a = [
            _make_collected_paper("a1", "ag1", 1, 0.9),
            _make_collected_paper("a2", "ag2", 1, 0.7),
            _make_collected_paper("a3", "ag3", 1, 0.5),
            _make_collected_paper("a4", "ag4", 2, 0.8),
            _make_collected_paper("a5", "ag5", 2, 0.6),
            _make_collected_paper("a6", "ag6", 2, 0.4),
        ]
        # Experiment B: 1 iteration × 4 papers
        papers_b = [
            _make_collected_paper("b1", "ag1", 1, 0.9),
            _make_collected_paper("b2", "ag2", 1, 0.7),
            _make_collected_paper("b3", "ag3", 1, 0.5),
            _make_collected_paper("b4", "ag4", 1, 0.3),
        ]

        cfg = {
            "top_n": {"enabled": True, "n": 2, "per_iteration": True},
            "balance": {"enabled": True, "method": "random", "seed": 42},
        }
        result_a, result_b, info = _apply_paper_selection(papers_a, papers_b, cfg)

        # After top-N: A has 4 (2 per iter × 2 iters), B has 2 (2 from 1 iter)
        assert info["top_n"]["count_a_after"] == 4
        assert info["top_n"]["count_b_after"] == 2
        # After balance: A cut to 2 to match B
        assert len(result_a) == 2
        assert len(result_b) == 2
        assert info["balance"]["enabled"] is True


class TestPerIterationDefaultsFalsePaired:
    """Missing per_iteration key defaults to global top-N (backward compat)."""

    def test_per_iteration_defaults_to_false(self):
        papers_a = [
            _make_collected_paper("a1", "ag1", 1, 0.9),
            _make_collected_paper("a2", "ag2", 1, 0.7),
            _make_collected_paper("a3", "ag3", 2, 0.8),
        ]
        papers_b = [
            _make_collected_paper("b1", "ag1", 1, 0.5),
        ]

        # No per_iteration key at all
        cfg = {
            "top_n": {"enabled": True, "n": 5},
            "balance": {"enabled": False},
        }
        result_a, result_b, info = _apply_paper_selection(papers_a, papers_b, cfg)

        # All 3 pass through (n=5 > 3)
        assert len(result_a) == 3
        assert info["top_n"]["per_iteration"] is False


# ---------------------------------------------------------------------------
# Tests: Paper selection -- Balance
# ---------------------------------------------------------------------------


class TestBalanceDisabled:
    """Balance disabled: unequal sets unchanged."""

    def test_balance_disabled(self):
        papers_a = [_make_collected_paper(f"a{i}", f"ag{i}", 1, 0.5) for i in range(5)]
        papers_b = [_make_collected_paper(f"b{i}", f"ag{i}", 1, 0.5) for i in range(3)]

        cfg = {"top_n": {"enabled": False}, "balance": {"enabled": False}}
        result_a, result_b, info = _apply_paper_selection(papers_a, papers_b, cfg)

        assert len(result_a) == 5
        assert len(result_b) == 3


class TestBalanceEqualSets:
    """Balance with equal sets: no-op."""

    def test_balance_equal_sets(self):
        papers_a = [_make_collected_paper(f"a{i}", f"ag{i}", 1, 0.5) for i in range(3)]
        papers_b = [_make_collected_paper(f"b{i}", f"ag{i}", 1, 0.5) for i in range(3)]

        cfg = {
            "top_n": {"enabled": False},
            "balance": {"enabled": True, "method": "random", "seed": 42},
        }
        result_a, result_b, info = _apply_paper_selection(papers_a, papers_b, cfg)

        assert len(result_a) == 3
        assert len(result_b) == 3


class TestBalanceRandomCutsLargerSet:
    """Balance with method=random: larger set sampled down."""

    def test_balance_random(self):
        papers_a = [_make_collected_paper(f"a{i}", f"ag{i}", 1, 0.5) for i in range(6)]
        papers_b = [_make_collected_paper(f"b{i}", f"ag{i}", 1, 0.5) for i in range(3)]

        cfg = {
            "top_n": {"enabled": False},
            "balance": {"enabled": True, "method": "random", "seed": 42},
        }
        result_a, result_b, info = _apply_paper_selection(papers_a, papers_b, cfg)

        assert len(result_a) == 3
        assert len(result_b) == 3
        # All remaining A papers should be from the original set
        ids_a = {p["paper_id"] for p in result_a}
        assert ids_a.issubset({f"a{i}" for i in range(6)})

    def test_balance_random_reproducible(self):
        """Same seed gives same result."""
        papers_a = [_make_collected_paper(f"a{i}", f"ag{i}", 1, 0.5) for i in range(6)]
        papers_b = [_make_collected_paper(f"b{i}", f"ag{i}", 1, 0.5) for i in range(3)]

        cfg = {
            "top_n": {"enabled": False},
            "balance": {"enabled": True, "method": "random", "seed": 42},
        }
        r1a, _, _ = _apply_paper_selection(papers_a, papers_b, cfg)
        r2a, _, _ = _apply_paper_selection(papers_a, papers_b, cfg)

        ids1 = [p["paper_id"] for p in r1a]
        ids2 = [p["paper_id"] for p in r2a]
        assert ids1 == ids2

    def test_balance_random_cuts_b_when_larger(self):
        """When B is larger, B gets cut."""
        papers_a = [_make_collected_paper(f"a{i}", f"ag{i}", 1, 0.5) for i in range(2)]
        papers_b = [_make_collected_paper(f"b{i}", f"ag{i}", 1, 0.5) for i in range(5)]

        cfg = {
            "top_n": {"enabled": False},
            "balance": {"enabled": True, "method": "random", "seed": 42},
        }
        result_a, result_b, info = _apply_paper_selection(papers_a, papers_b, cfg)

        assert len(result_a) == 2
        assert len(result_b) == 2


class TestBalanceTopMetareview:
    """Balance with method=top_metareview: larger set cut to best by score."""

    def test_balance_top_metareview(self):
        papers_a = [
            _make_collected_paper("a0", "ag0", 1, 0.1),
            _make_collected_paper("a1", "ag1", 1, 0.9),
            _make_collected_paper("a2", "ag2", 1, 0.5),
            _make_collected_paper("a3", "ag3", 1, 0.7),
            _make_collected_paper("a4", "ag4", 1, 0.3),
        ]
        papers_b = [
            _make_collected_paper("b0", "ag0", 1, 0.8),
            _make_collected_paper("b1", "ag1", 1, 0.4),
        ]

        cfg = {
            "top_n": {"enabled": False},
            "balance": {"enabled": True, "method": "top_metareview", "seed": 42},
        }
        result_a, result_b, info = _apply_paper_selection(papers_a, papers_b, cfg)

        assert len(result_a) == 2
        assert len(result_b) == 2
        # Best 2 from A: a1(0.9) and a3(0.7)
        ids_a = [p["paper_id"] for p in result_a]
        assert ids_a == ["a1", "a3"]


class TestBalanceTopMetareviewWithNulls:
    """Balance top_metareview with null scores in larger set."""

    def test_balance_top_metareview_nulls(self):
        papers_a = [
            _make_collected_paper("a0", "ag0", 1, 0.8),
            _make_collected_paper("a1", "ag1", 1, None),
            _make_collected_paper("a2", "ag2", 1, 0.3),
            _make_collected_paper("a3", "ag3", 1, None),
        ]
        papers_b = [_make_collected_paper("b0", "ag0", 1, 0.5)]

        cfg = {
            "top_n": {"enabled": False},
            "balance": {"enabled": True, "method": "top_metareview", "seed": 42},
        }
        result_a, result_b, info = _apply_paper_selection(papers_a, papers_b, cfg)

        assert len(result_a) == 1
        assert result_a[0]["paper_id"] == "a0"


# ---------------------------------------------------------------------------
# Tests: Paper selection -- Combined top-N + balance
# ---------------------------------------------------------------------------


class TestTopNThenBalance:
    """Top-N applied first, then balance."""

    def test_top_n_then_balance(self):
        # A: 10 papers, B: 3 papers
        papers_a = [_make_collected_paper(f"a{i}", f"ag{i}", 1, 0.1 * (i + 1)) for i in range(10)]
        papers_b = [_make_collected_paper(f"b{i}", f"ag{i}", 1, 0.1 * (i + 1)) for i in range(3)]

        cfg = {
            "top_n": {"enabled": True, "n": 5},
            "balance": {"enabled": True, "method": "top_metareview", "seed": 42},
        }
        result_a, result_b, info = _apply_paper_selection(papers_a, papers_b, cfg)

        # After top-5: A->5 (a9,a8,a7,a6,a5), B->3 (b2,b1,b0)
        # After balance: A->3 (top 3: a9,a8,a7)
        assert len(result_a) == 3
        assert len(result_b) == 3

        ids_a = [p["paper_id"] for p in result_a]
        assert ids_a == ["a9", "a8", "a7"]

    def test_top_n_already_equalizes(self):
        """When top-N makes sets equal, balance is a no-op."""
        papers_a = [_make_collected_paper(f"a{i}", f"ag{i}", 1, 0.5) for i in range(5)]
        papers_b = [_make_collected_paper(f"b{i}", f"ag{i}", 1, 0.5) for i in range(5)]

        cfg = {
            "top_n": {"enabled": True, "n": 3},
            "balance": {"enabled": True, "method": "random", "seed": 42},
        }
        result_a, result_b, info = _apply_paper_selection(papers_a, papers_b, cfg)

        assert len(result_a) == 3
        assert len(result_b) == 3

    def test_both_disabled_preserves_original(self):
        """Both disabled: original behavior preserved."""
        papers_a = [_make_collected_paper(f"a{i}", f"ag{i}", 1, 0.5) for i in range(5)]
        papers_b = [_make_collected_paper(f"b{i}", f"ag{i}", 1, 0.5) for i in range(3)]

        cfg = {"top_n": {"enabled": False}, "balance": {"enabled": False}}
        result_a, result_b, info = _apply_paper_selection(papers_a, papers_b, cfg)

        assert len(result_a) == 5
        assert len(result_b) == 3


# ---------------------------------------------------------------------------
# Tests: Paper selection -- Selection info
# ---------------------------------------------------------------------------


class TestSelectionInfo:
    """Verify _apply_paper_selection returns correct selection_info."""

    def test_info_top_n_only(self):
        papers_a = [_make_collected_paper(f"a{i}", f"ag{i}", 1, 0.5) for i in range(5)]
        papers_b = [_make_collected_paper(f"b{i}", f"ag{i}", 1, 0.5) for i in range(3)]

        cfg = {"top_n": {"enabled": True, "n": 2}, "balance": {"enabled": False}}
        _, _, info = _apply_paper_selection(papers_a, papers_b, cfg)

        assert info["original_count_a"] == 5
        assert info["original_count_b"] == 3
        assert info["top_n"]["enabled"] is True
        assert info["top_n"]["n"] == 2
        assert info["top_n"]["count_a_after"] == 2
        assert info["top_n"]["count_b_after"] == 2
        assert info["balance"]["enabled"] is False
        assert info["final_count_a"] == 2
        assert info["final_count_b"] == 2

    def test_info_balance_only(self):
        papers_a = [_make_collected_paper(f"a{i}", f"ag{i}", 1, 0.5) for i in range(5)]
        papers_b = [_make_collected_paper(f"b{i}", f"ag{i}", 1, 0.5) for i in range(2)]

        cfg = {
            "top_n": {"enabled": False},
            "balance": {"enabled": True, "method": "random", "seed": 42},
        }
        _, _, info = _apply_paper_selection(papers_a, papers_b, cfg)

        assert info["top_n"]["enabled"] is False
        assert info["balance"]["enabled"] is True
        assert info["balance"]["method"] == "random"
        assert info["balance"]["count_a_before"] == 5
        assert info["balance"]["count_a_after"] == 2
        assert info["final_count_a"] == 2
        assert info["final_count_b"] == 2


# ---------------------------------------------------------------------------
# Tests: _print_selection_summary -- filtering info in PAPER SELECTION report
# ---------------------------------------------------------------------------


class TestSelectionSummaryFiltering:
    """Verify _print_selection_summary includes filtering info."""

    def _build_selection_info(
        self, include_rejected, exclude_timeout, counts_a=None, counts_b=None
    ):
        """Build a minimal selection_info dict with filtering data."""
        if counts_a is None:
            counts_a = {"accepted": 5, "rejected": 0, "timeout": 0}
        if counts_b is None:
            counts_b = {"accepted": 3, "rejected": 0, "timeout": 0}
        return {
            "original_count_a": counts_a["accepted"] + counts_a["rejected"],
            "original_count_b": counts_b["accepted"] + counts_b["rejected"],
            "top_n": {"enabled": False},
            "balance": {"enabled": False},
            "final_count_a": counts_a["accepted"] + counts_a["rejected"],
            "final_count_b": counts_b["accepted"] + counts_b["rejected"],
            "filtering": {
                "include_rejected": include_rejected,
                "exclude_timeout": exclude_timeout,
                "counts_a": counts_a,
                "counts_b": counts_b,
            },
        }

    def test_defaults_accepted_only(self, capsys):
        info = self._build_selection_info(False, True)
        _print_selection_summary(info)
        out = capsys.readouterr().out
        assert "Status filter: accepted only" in out
        assert "Timeout filter: excluded" in out

    def test_include_rejected_shows_counts(self, capsys):
        info = self._build_selection_info(
            True,
            True,
            counts_a={"accepted": 5, "rejected": 3, "timeout": 0},
            counts_b={"accepted": 4, "rejected": 2, "timeout": 0},
        )
        _print_selection_summary(info)
        out = capsys.readouterr().out
        assert "Status filter: accepted + rejected" in out
        assert "Experiment A: 5 accepted, 3 rejected" in out
        assert "Experiment B: 4 accepted, 2 rejected" in out

    def test_include_timeout_shows_counts(self, capsys):
        info = self._build_selection_info(
            True,
            False,
            counts_a={"accepted": 5, "rejected": 3, "timeout": 1},
            counts_b={"accepted": 4, "rejected": 2, "timeout": 2},
        )
        _print_selection_summary(info)
        out = capsys.readouterr().out
        assert "Timeout filter: included (1 in A, 2 in B)" in out

    def test_include_timeout_zero_found(self, capsys):
        info = self._build_selection_info(
            True,
            False,
            counts_a={"accepted": 5, "rejected": 1, "timeout": 0},
            counts_b={"accepted": 3, "rejected": 1, "timeout": 0},
        )
        _print_selection_summary(info)
        out = capsys.readouterr().out
        assert "Timeout filter: included (0 found)" in out

    def test_no_filtering_key_uses_defaults(self, capsys):
        """If filtering info is absent, prints default labels."""
        info = {
            "original_count_a": 5,
            "original_count_b": 3,
            "top_n": {"enabled": False},
            "balance": {"enabled": False},
            "final_count_a": 5,
            "final_count_b": 3,
        }
        _print_selection_summary(info)
        out = capsys.readouterr().out
        assert "Status filter: accepted only" in out
        assert "Timeout filter: excluded" in out


# ---------------------------------------------------------------------------
# Tests: _gather_papers records filtering info in collection_info
# ---------------------------------------------------------------------------


class TestGatherPapersFilteringInfo:
    """Verify _gather_papers populates collection_info['filtering']."""

    def test_filtering_info_defaults(self, two_experiment_snapshots):
        cfg = two_experiment_snapshots["experiments_cfg_single"]
        _, _, info = _gather_papers(
            cfg, paper_selection_cfg={"iterations": "last", "cumulative": False}
        )

        filt = info["filtering"]
        assert filt["include_rejected"] is False
        assert filt["exclude_timeout"] is True
        assert filt["counts_a"]["accepted"] > 0
        assert filt["counts_b"]["accepted"] > 0

    def test_filtering_info_with_include_rejected(self, tmp_path):
        snap_a = tmp_path / "filt_snap_a"
        snap_b = tmp_path / "filt_snap_b"

        papers_a = [
            _make_paper("pa1", "ag1", 1, status="accepted"),
            _make_paper("pa2", "ag2", 1, status="rejected"),
        ]
        papers_b = [
            _make_paper("pb1", "ag3", 1, status="accepted"),
        ]

        _write_iteration(snap_a, 1, papers_a, "filt_a_run")
        _write_iteration(snap_b, 1, papers_b, "filt_b_run")

        cfg = {
            "experiment_a": {"label": "a", "snapshot_dirs": [str(snap_a)]},
            "experiment_b": {"label": "b", "snapshot_dirs": [str(snap_b)]},
        }
        psel = {"include_rejected": True, "exclude_timeout": True}
        _, _, info = _gather_papers(
            cfg, paper_selection_cfg={**psel, "iterations": "last", "cumulative": False}
        )

        filt = info["filtering"]
        assert filt["include_rejected"] is True
        assert filt["counts_a"]["accepted"] == 1
        assert filt["counts_a"]["rejected"] == 1
        assert filt["counts_b"]["accepted"] == 1
        assert filt["counts_b"]["rejected"] == 0

    def test_filtering_info_with_per_side_paper_kind_filters(self, tmp_path):
        snap_a = tmp_path / "kind_snap_a"
        snap_b = tmp_path / "kind_snap_b"

        papers_a = [
            _make_paper("pa_std", "ag1", 1, status="accepted", paper_kind="research"),
            _make_paper("pa_meta", "ag2", 1, status="accepted", paper_kind="meta_research"),
        ]
        papers_b = [
            _make_paper("pb_std", "ag3", 1, status="accepted", paper_kind="research"),
            _make_paper("pb_meta", "ag4", 1, status="accepted", paper_kind="meta_research"),
        ]

        _write_iteration(snap_a, 1, papers_a, "kind_a_run")
        _write_iteration(snap_b, 1, papers_b, "kind_b_run")

        cfg = {
            "experiment_a": {"label": "a", "snapshot_dirs": [str(snap_a)]},
            "experiment_b": {"label": "b", "snapshot_dirs": [str(snap_b)]},
        }
        psel = {
            "iterations": "last",
            "cumulative": False,
            "paper_kind_filter_a": "meta_only",
            "paper_kind_filter_b": "standard_only",
        }
        papers_a_filtered, papers_b_filtered, info = _gather_papers(cfg, paper_selection_cfg=psel)

        assert {p["paper_id"] for p in papers_a_filtered} == {"pa_meta"}
        assert {p["paper_id"] for p in papers_b_filtered} == {"pb_std"}
        assert info["filtering"]["paper_kind_filter_a"] == "meta_only"
        assert info["filtering"]["paper_kind_filter_b"] == "standard_only"


# ---------------------------------------------------------------------------
# Tests: _collect_paper_data -- experiment_condition and metareview
# ---------------------------------------------------------------------------


class TestCollectPaperDataExperimentCondition:
    """Verify experiment_condition field in _collect_paper_data output."""

    def test_experiment_condition(self, two_experiment_snapshots):
        cfg = two_experiment_snapshots["experiments_cfg_single"]
        papers_a, papers_b, _ = _gather_papers(
            cfg, paper_selection_cfg={"iterations": "last", "cumulative": False}
        )

        records = _collect_paper_data(papers_a, papers_b, "baseline", "treatment")

        conditions = {r["experiment_condition"] for r in records}
        assert conditions == {"A", "B"}

        for r in records:
            if r["experiment"] == "baseline":
                assert r["experiment_condition"] == "A"
            else:
                assert r["experiment_condition"] == "B"


class TestCollectPaperDataMetareview:
    """Verify metareview field in _collect_paper_data output."""

    def test_metareview_present(self, tmp_path):
        snap_a = tmp_path / "snap_mr_a"
        snap_b = tmp_path / "snap_mr_b"

        mr_a = _make_metareview("paper_t01_agent_01", 0.75)
        papers_a_raw = [_make_paper("paper_t01_agent_01", "agent_01", 1, metareview=mr_a)]
        papers_b_raw = [_make_paper("paper_t01_agent_02", "agent_02", 1)]

        _write_iteration(snap_a, 1, papers_a_raw, "snap_mr_a_run")
        _write_iteration(snap_b, 1, papers_b_raw, "snap_mr_b_run")

        papers_a = collect_papers(str(snap_a), iterations="last")
        papers_b = collect_papers(str(snap_b), iterations="last")

        records = _collect_paper_data(papers_a, papers_b, "exp_a", "exp_b")

        a_rec = [r for r in records if r["paper_id"] == "paper_t01_agent_01"][0]
        b_rec = [r for r in records if r["paper_id"] == "paper_t01_agent_02"][0]

        assert a_rec["metareview"] is not None
        assert a_rec["metareview"]["overall_score"] == 0.75
        assert b_rec["metareview"] is None


class TestCollectPaperDataAllPapersIncluded:
    """All papers are included even without system metrics."""

    def test_all_papers_included(self, tmp_path):
        snap = tmp_path / "snap_no_sys_metrics"
        papers_raw = [
            _make_paper("paper_t01_agent_01", "agent_01", 1),
            _make_paper("paper_t01_agent_02", "agent_02", 1),
        ]
        _write_iteration(snap, 1, papers_raw, "snap_run")
        # No metrics.jsonl written

        papers = collect_papers(str(snap), iterations="last")
        records = _collect_paper_data(papers, [], "exp_a", "exp_b")

        assert len(records) == 2
        for r in records:
            assert r["metrics"] is None
            assert r["experiment_condition"] == "A"


# ---------------------------------------------------------------------------
# Tests: End-to-end with paper_selection
# ---------------------------------------------------------------------------


class TestEndToEndWithPaperSelection:
    """Full run_sample_and_rank_paired with paper_selection enabled."""

    def test_e2e_top_n(self, mock_llm, tmp_path):
        """Top-N filtering reduces paper count and is reflected in summary."""
        exp_a = tmp_path / "e2e_top_n_a"
        exp_b = tmp_path / "e2e_top_n_b"

        # A: 4 papers with different metareview scores
        papers_a = [
            _make_paper("p_a1", "ag1", 1, metareview=_make_metareview("p_a1", 0.9)),
            _make_paper("p_a2", "ag2", 1, metareview=_make_metareview("p_a2", 0.7)),
            _make_paper("p_a3", "ag3", 1, metareview=_make_metareview("p_a3", 0.3)),
            _make_paper("p_a4", "ag4", 1, metareview=_make_metareview("p_a4", 0.1)),
        ]
        # B: 2 papers
        papers_b = [
            _make_paper("p_b1", "ag5", 1, metareview=_make_metareview("p_b1", 0.8)),
            _make_paper("p_b2", "ag6", 1, metareview=_make_metareview("p_b2", 0.4)),
        ]

        _write_iteration(exp_a, 1, papers_a, "e2e_top_n_a_run")
        _write_iteration(exp_b, 1, papers_b, "e2e_top_n_b_run")

        output_dir = str(tmp_path / "output_e2e_top_n")
        experiments_cfg = {
            "experiment_a": {"label": "baseline", "snapshot_dirs": [str(exp_a)]},
            "experiment_b": {"label": "treatment", "snapshot_dirs": [str(exp_b)]},
        }
        ranker_cfg = {
            "max_iterations": 5,
            "tools": {},
            "llm_retry": {},
            "output_parsing": {"max_parse_retries": 2},
        }

        result = run_sample_and_rank_paired(
            experiments_cfg=experiments_cfg,
            sampling_cfg={"mode": "all", "n_repeats": 1, "seed": 42},
            ranker_cfg=ranker_cfg,
            parallelism_cfg={"max_workers": 1},
            llm=mock_llm,
            run_id="test_e2e_top_n",
            output_dir=output_dir,
            paper_selection_cfg={
                "iterations": "last",
                "cumulative": True,
                "top_n": {"enabled": True, "n": 2},
                "balance": {"enabled": False},
            },
        )

        # After top-2: A->2, B->2 -> 4 comparisons
        assert len(result["comparisons"]) == 4  # 2 x 2

        # Summary should have paper_selection info
        summary = result["summary"]
        assert "paper_selection" in summary
        assert summary["paper_selection"]["top_n"]["enabled"] is True
        assert summary["paper_selection"]["final_count_a"] == 2
        assert summary["paper_selection"]["final_count_b"] == 2

        # papers/paper_data.json should have the filtered papers
        paper_data_path = os.path.join(output_dir, "papers", "paper_data.json")
        assert os.path.exists(paper_data_path)
        with open(paper_data_path) as f:
            paper_data = json.load(f)
        assert len(paper_data) == 4  # 2 from A + 2 from B

        # All should have metareview data
        for rec in paper_data:
            assert rec["metareview"] is not None

    def test_e2e_balance(self, mock_llm, tmp_path):
        """Balance equalizes sets and summary reflects it."""
        exp_a = tmp_path / "e2e_bal_a"
        exp_b = tmp_path / "e2e_bal_b"

        papers_a = [
            _make_paper(
                f"p_a{i}", f"ag{i}", 1, metareview=_make_metareview(f"p_a{i}", 0.1 * (i + 1))
            )
            for i in range(5)
        ]
        papers_b = [
            _make_paper("p_b1", "ag10", 1, metareview=_make_metareview("p_b1", 0.5)),
        ]

        _write_iteration(exp_a, 1, papers_a, "e2e_bal_a_run")
        _write_iteration(exp_b, 1, papers_b, "e2e_bal_b_run")

        output_dir = str(tmp_path / "output_e2e_bal")
        experiments_cfg = {
            "experiment_a": {"label": "baseline", "snapshot_dirs": [str(exp_a)]},
            "experiment_b": {"label": "treatment", "snapshot_dirs": [str(exp_b)]},
        }
        ranker_cfg = {
            "max_iterations": 5,
            "tools": {},
            "llm_retry": {},
            "output_parsing": {"max_parse_retries": 2},
        }

        result = run_sample_and_rank_paired(
            experiments_cfg=experiments_cfg,
            sampling_cfg={"mode": "all", "n_repeats": 1, "seed": 42},
            ranker_cfg=ranker_cfg,
            parallelism_cfg={"max_workers": 1},
            llm=mock_llm,
            run_id="test_e2e_bal",
            output_dir=output_dir,
            paper_selection_cfg={
                "iterations": "last",
                "cumulative": True,
                "top_n": {"enabled": False},
                "balance": {"enabled": True, "method": "top_metareview", "seed": 42},
            },
        )

        # After balance: A->1, B->1 -> 1 comparison
        assert len(result["comparisons"]) == 1

        summary = result["summary"]
        assert "paper_selection" in summary
        assert summary["paper_selection"]["balance"]["enabled"] is True
        assert summary["paper_selection"]["final_count_a"] == 1
        assert summary["paper_selection"]["final_count_b"] == 1


# ---------------------------------------------------------------------------
# Tests: T13-T21 -- New paper data features
# ---------------------------------------------------------------------------


class TestPaperDataOutputPath:
    """T13: Output path is papers/paper_data.json, old path does NOT exist."""

    def test_paper_data_output_path(self, two_experiment_snapshots, mock_llm, tmp_path):
        cfg = two_experiment_snapshots["experiments_cfg_single"]
        output_dir = str(tmp_path / "output_path_test")

        ranker_cfg = {
            "max_iterations": 5,
            "tools": {},
            "llm_retry": {},
            "output_parsing": {"max_parse_retries": 2},
        }

        run_sample_and_rank_paired(
            experiments_cfg=cfg,
            sampling_cfg={"mode": "all", "n_repeats": 1, "seed": 42},
            ranker_cfg=ranker_cfg,
            parallelism_cfg={"max_workers": 1},
            llm=mock_llm,
            run_id="test_path",
            output_dir=output_dir,
            paper_selection_cfg={"iterations": "last", "cumulative": False},
        )

        # New path exists
        assert os.path.exists(os.path.join(output_dir, "papers", "paper_data.json"))
        # Old path does NOT exist
        assert not os.path.exists(os.path.join(output_dir, "paper_metrics", "metrics.json"))


class TestPaperDataHasAllFields:
    """T14: paper_data.json records have all required fields."""

    def test_paper_data_has_all_fields(self, tmp_path, mock_llm):
        exp_a = tmp_path / "fields_a"
        exp_b = tmp_path / "fields_b"

        reviews = {
            "paper_t01_agent_01": [_make_review(1, "agent_10")],
        }

        papers_a = [_make_paper("paper_t01_agent_01", "agent_01", 1, code_script="x=1")]
        papers_a[0]["collab_agent_ids"] = ["agent_05"]
        papers_a[0]["citation_count"] = 2
        papers_a[0]["cited_paper_ids"] = ["ref_1"]

        papers_b = [_make_paper("paper_t01_agent_02", "agent_02", 1)]

        _write_iteration(exp_a, 1, papers_a, "fields_a_run", reviews_by_paper=reviews)
        _write_iteration(exp_b, 1, papers_b, "fields_b_run")

        output_dir = str(tmp_path / "output_fields")
        experiments_cfg = {
            "experiment_a": {"label": "baseline", "snapshot_dirs": [str(exp_a)]},
            "experiment_b": {"label": "treatment", "snapshot_dirs": [str(exp_b)]},
        }
        ranker_cfg = {
            "max_iterations": 5,
            "tools": {},
            "llm_retry": {},
            "output_parsing": {"max_parse_retries": 2},
        }

        run_sample_and_rank_paired(
            experiments_cfg=experiments_cfg,
            sampling_cfg={"mode": "all", "n_repeats": 1, "seed": 42},
            ranker_cfg=ranker_cfg,
            parallelism_cfg={"max_workers": 1},
            llm=mock_llm,
            run_id="test_fields",
            output_dir=output_dir,
            paper_selection_cfg={"iterations": "last", "cumulative": False},
        )

        paper_data_path = os.path.join(output_dir, "papers", "paper_data.json")
        with open(paper_data_path) as f:
            paper_data = json.load(f)

        for rec in paper_data:
            # Original paper fields
            assert "collab_agent_ids" in rec
            assert "citation_count" in rec
            assert "cited_paper_ids" in rec
            assert "status" in rec
            assert "code_script" in rec
            assert "publication_t" in rec
            # Provenance fields
            assert "source_snapshot" in rec
            assert "source_snapshot_dir" in rec
            assert "source_iteration" in rec
            # Reviews
            assert "reviews" in rec
            # Experiment context
            assert "experiment" in rec
            assert "experiment_condition" in rec
            # Metrics
            assert "metrics" in rec
            # Metareview
            assert "metareview" in rec


class TestPaperDataIncludesReviews:
    """T15: Reviews are included in paper_data.json."""

    def test_paper_data_includes_reviews(self, tmp_path, mock_llm):
        exp_a = tmp_path / "reviews_a"
        exp_b = tmp_path / "reviews_b"

        reviews = {
            "paper_t01_agent_01": [
                _make_review(1, "agent_10"),
                _make_review(1, "agent_11"),
            ],
        }

        papers_a = [_make_paper("paper_t01_agent_01", "agent_01", 1)]
        papers_b = [_make_paper("paper_t01_agent_02", "agent_02", 1)]

        _write_iteration(exp_a, 1, papers_a, "reviews_a_run", reviews_by_paper=reviews)
        _write_iteration(exp_b, 1, papers_b, "reviews_b_run")

        output_dir = str(tmp_path / "output_reviews")
        experiments_cfg = {
            "experiment_a": {"label": "baseline", "snapshot_dirs": [str(exp_a)]},
            "experiment_b": {"label": "treatment", "snapshot_dirs": [str(exp_b)]},
        }
        ranker_cfg = {
            "max_iterations": 5,
            "tools": {},
            "llm_retry": {},
            "output_parsing": {"max_parse_retries": 2},
        }

        run_sample_and_rank_paired(
            experiments_cfg=experiments_cfg,
            sampling_cfg={"mode": "all", "n_repeats": 1, "seed": 42},
            ranker_cfg=ranker_cfg,
            parallelism_cfg={"max_workers": 1},
            llm=mock_llm,
            run_id="test_reviews",
            output_dir=output_dir,
            paper_selection_cfg={"iterations": "last", "cumulative": False},
        )

        paper_data_path = os.path.join(output_dir, "papers", "paper_data.json")
        with open(paper_data_path) as f:
            paper_data = json.load(f)

        a_rec = [r for r in paper_data if r["paper_id"] == "paper_t01_agent_01"][0]
        b_rec = [r for r in paper_data if r["paper_id"] == "paper_t01_agent_02"][0]

        assert len(a_rec["reviews"]) == 2
        assert a_rec["reviews"][0]["reviewer_id"] == "agent_10"
        assert b_rec["reviews"] == []


class TestSummaryHasPaperDataKey:
    """T16: run_summary.json has 'paper_data' key, NOT 'paper_metrics'."""

    def test_summary_has_paper_data_key(self, two_experiment_snapshots, mock_llm, tmp_path):
        cfg = two_experiment_snapshots["experiments_cfg_single"]
        output_dir = str(tmp_path / "output_summary_key")

        ranker_cfg = {
            "max_iterations": 5,
            "tools": {},
            "llm_retry": {},
            "output_parsing": {"max_parse_retries": 2},
        }

        result = run_sample_and_rank_paired(
            experiments_cfg=cfg,
            sampling_cfg={"mode": "all", "n_repeats": 1, "seed": 42},
            ranker_cfg=ranker_cfg,
            parallelism_cfg={"max_workers": 1},
            llm=mock_llm,
            run_id="test_summary_key",
            output_dir=output_dir,
            paper_selection_cfg={"iterations": "last", "cumulative": False},
        )

        summary = result["summary"]
        assert "paper_data" in summary
        assert "paper_metrics" not in summary

        # Verify it has the expected sub-fields
        pd = summary["paper_data"]
        assert "total" in pd
        assert "experiment_a" in pd
        assert "experiment_b" in pd
        assert "with_system_metrics" in pd
        assert "without_system_metrics" in pd


class TestE2EWithIncludeRejected:
    """T17: End-to-end with include_rejected=True."""

    def test_e2e_with_include_rejected(self, mock_llm, tmp_path):
        exp_a = tmp_path / "rejected_a"
        exp_b = tmp_path / "rejected_b"

        papers_a = [
            _make_paper("p_a1", "ag1", 1, status="accepted"),
            _make_paper("p_a2", "ag2", 1, status="rejected"),
        ]
        papers_b = [
            _make_paper("p_b1", "ag3", 1, status="accepted"),
            _make_paper("p_b2", "ag4", 1, status="rejected"),
        ]
        timeout_paper = _make_timeout_paper("p_a3", "ag5", 1)
        papers_a.append(timeout_paper)

        _write_iteration(exp_a, 1, papers_a, "rejected_a_run")
        _write_iteration(exp_b, 1, papers_b, "rejected_b_run")

        output_dir = str(tmp_path / "output_rejected")
        experiments_cfg = {
            "experiment_a": {"label": "baseline", "snapshot_dirs": [str(exp_a)]},
            "experiment_b": {"label": "treatment", "snapshot_dirs": [str(exp_b)]},
        }
        ranker_cfg = {
            "max_iterations": 5,
            "tools": {},
            "llm_retry": {},
            "output_parsing": {"max_parse_retries": 2},
        }

        result = run_sample_and_rank_paired(
            experiments_cfg=experiments_cfg,
            sampling_cfg={"mode": "all", "n_repeats": 1, "seed": 42},
            ranker_cfg=ranker_cfg,
            parallelism_cfg={"max_workers": 1},
            llm=mock_llm,
            run_id="test_rejected",
            output_dir=output_dir,
            paper_selection_cfg={
                "iterations": "last",
                "cumulative": False,
                "include_rejected": True,
                "exclude_timeout": True,
                "top_n": {"enabled": False},
                "balance": {"enabled": False},
            },
        )

        # papers/paper_data.json should include accepted + rejected (no timeout)
        paper_data_path = os.path.join(output_dir, "papers", "paper_data.json")
        with open(paper_data_path) as f:
            paper_data = json.load(f)

        statuses = {r["status"] for r in paper_data}
        assert "accepted" in statuses
        assert "rejected" in statuses

        # No timeout papers
        for r in paper_data:
            assert not r["paper_id"].endswith("_timeout")
            assert r["manuscript"] != "TIMEOUT"

        # A: 2 (accepted + rejected, no timeout), B: 2 -> total 4
        assert len(paper_data) == 4

        # Comparisons should exist: 2x2 = 4
        assert len(result["comparisons"]) == 4


class TestE2EWithTimeoutPapers:
    """T18: End-to-end with timeout papers explicitly included."""

    def test_e2e_with_timeout_papers(self, mock_llm, tmp_path):
        exp_a = tmp_path / "timeout_a"
        exp_b = tmp_path / "timeout_b"

        papers_a = [
            _make_paper("p_a1", "ag1", 1, status="accepted"),
            _make_timeout_paper("p_a2", "ag2", 1),
        ]
        papers_b = [
            _make_paper("p_b1", "ag3", 1, status="accepted"),
        ]

        _write_iteration(exp_a, 1, papers_a, "timeout_a_run")
        _write_iteration(exp_b, 1, papers_b, "timeout_b_run")

        output_dir = str(tmp_path / "output_timeout")
        experiments_cfg = {
            "experiment_a": {"label": "baseline", "snapshot_dirs": [str(exp_a)]},
            "experiment_b": {"label": "treatment", "snapshot_dirs": [str(exp_b)]},
        }
        ranker_cfg = {
            "max_iterations": 5,
            "tools": {},
            "llm_retry": {},
            "output_parsing": {"max_parse_retries": 2},
        }

        result = run_sample_and_rank_paired(  # noqa: F841
            experiments_cfg=experiments_cfg,
            sampling_cfg={"mode": "all", "n_repeats": 1, "seed": 42},
            ranker_cfg=ranker_cfg,
            parallelism_cfg={"max_workers": 1},
            llm=mock_llm,
            run_id="test_timeout",
            output_dir=output_dir,
            paper_selection_cfg={
                "iterations": "last",
                "cumulative": False,
                "include_rejected": True,
                "exclude_timeout": False,
                "top_n": {"enabled": False},
                "balance": {"enabled": False},
            },
        )

        paper_data_path = os.path.join(output_dir, "papers", "paper_data.json")
        with open(paper_data_path) as f:
            paper_data = json.load(f)

        # Timeout paper should be present
        timeout_papers = [r for r in paper_data if r["manuscript"] == "TIMEOUT"]
        assert len(timeout_papers) == 1

        # A: 2 (accepted + timeout), B: 1 -> total 3
        assert len(paper_data) == 3


class TestE2EDefaultBackwardCompatible:
    """T19: Default behavior is backward compatible (except path rename)."""

    def test_e2e_default_backward_compatible(self, two_experiment_snapshots, mock_llm, tmp_path):
        cfg = two_experiment_snapshots["experiments_cfg_single"]
        output_dir = str(tmp_path / "output_default")

        ranker_cfg = {
            "max_iterations": 5,
            "tools": {},
            "llm_retry": {},
            "output_parsing": {"max_parse_retries": 2},
        }

        result = run_sample_and_rank_paired(
            experiments_cfg=cfg,
            sampling_cfg={"mode": "all", "n_repeats": 1, "seed": 42},
            ranker_cfg=ranker_cfg,
            parallelism_cfg={"max_workers": 1},
            llm=mock_llm,
            run_id="test_default",
            output_dir=output_dir,
            paper_selection_cfg={"iterations": "last", "cumulative": False},
        )

        # Only accepted papers (same as before)
        paper_data_path = os.path.join(output_dir, "papers", "paper_data.json")
        assert os.path.exists(paper_data_path)

        with open(paper_data_path) as f:
            paper_data = json.load(f)

        for rec in paper_data:
            assert rec["status"] == "accepted"

        # New path used
        assert os.path.exists(paper_data_path)
        # Old path does NOT exist
        assert not os.path.exists(os.path.join(output_dir, "paper_metrics", "metrics.json"))

        # Summary uses new key
        assert "paper_data" in result["summary"]
        assert "paper_metrics" not in result["summary"]


class TestPaperSelectionWorksWithRejectedPapers:
    """T20: Top-N selection works with rejected papers."""

    def test_paper_selection_works_with_rejected_papers(self, mock_llm, tmp_path):
        exp_a = tmp_path / "sel_rej_a"
        exp_b = tmp_path / "sel_rej_b"

        papers_a = [
            _make_paper(
                "p_a1", "ag1", 1, status="accepted", metareview=_make_metareview("p_a1", 0.9)
            ),
            _make_paper(
                "p_a2",
                "ag2",
                1,
                status="rejected",
                metareview=_make_metareview("p_a2", 0.7, decision="reject"),
            ),
            _make_paper(
                "p_a3", "ag3", 1, status="accepted", metareview=_make_metareview("p_a3", 0.3)
            ),
        ]
        papers_b = [
            _make_paper(
                "p_b1", "ag4", 1, status="accepted", metareview=_make_metareview("p_b1", 0.8)
            ),
        ]

        _write_iteration(exp_a, 1, papers_a, "sel_rej_a_run")
        _write_iteration(exp_b, 1, papers_b, "sel_rej_b_run")

        output_dir = str(tmp_path / "output_sel_rej")
        experiments_cfg = {
            "experiment_a": {"label": "baseline", "snapshot_dirs": [str(exp_a)]},
            "experiment_b": {"label": "treatment", "snapshot_dirs": [str(exp_b)]},
        }
        ranker_cfg = {
            "max_iterations": 5,
            "tools": {},
            "llm_retry": {},
            "output_parsing": {"max_parse_retries": 2},
        }

        result = run_sample_and_rank_paired(  # noqa: F841
            experiments_cfg=experiments_cfg,
            sampling_cfg={"mode": "all", "n_repeats": 1, "seed": 42},
            ranker_cfg=ranker_cfg,
            parallelism_cfg={"max_workers": 1},
            llm=mock_llm,
            run_id="test_sel_rej",
            output_dir=output_dir,
            paper_selection_cfg={
                "iterations": "last",
                "cumulative": False,
                "include_rejected": True,
                "exclude_timeout": True,
                "top_n": {"enabled": True, "n": 2},
                "balance": {"enabled": False},
            },
        )

        paper_data_path = os.path.join(output_dir, "papers", "paper_data.json")
        with open(paper_data_path) as f:
            paper_data = json.load(f)

        # Top-2 from A includes rejected paper (p_a1=0.9, p_a2=0.7)
        a_papers = [r for r in paper_data if r["experiment"] == "baseline"]
        a_ids = {r["paper_id"] for r in a_papers}
        assert "p_a1" in a_ids
        assert "p_a2" in a_ids
        assert len(a_papers) == 2

        # Rejected paper is included
        a_statuses = {r["status"] for r in a_papers}
        assert "rejected" in a_statuses


@pytest.mark.integration
class TestIntegrationWithRejectedPapers:
    """T21: Integration test with rejected papers (requires real LLM)."""

    def test_integration_with_rejected_papers(self, llm, tmp_path):
        exp_a = tmp_path / "int_rej_a"
        exp_b = tmp_path / "int_rej_b"

        papers_a = [
            _make_paper("p_a1", "ag1", 1, status="accepted", code_script="print('a')"),
            _make_paper("p_a2", "ag2", 1, status="rejected"),
        ]
        papers_b = [
            _make_paper("p_b1", "ag3", 1, status="accepted"),
            _make_paper("p_b2", "ag4", 1, status="rejected"),
        ]

        reviews = {
            "p_a1": [_make_review(1, "agent_10")],
            "p_b1": [_make_review(1, "agent_11")],
        }

        _write_iteration(exp_a, 1, papers_a, "int_rej_a_run", reviews_by_paper=reviews)
        _write_iteration(exp_b, 1, papers_b, "int_rej_b_run", reviews_by_paper=reviews)

        output_dir = str(tmp_path / "int_rej_output")
        experiments_cfg = {
            "experiment_a": {"label": "baseline", "snapshot_dirs": [str(exp_a)]},
            "experiment_b": {"label": "treatment", "snapshot_dirs": [str(exp_b)]},
        }
        ranker_cfg = {
            "max_iterations": 5,
            "tools": {},
            "llm_retry": {"max_retries": 3, "base_delay_seconds": 2.0, "max_delay_seconds": 30.0},
            "output_parsing": {"max_parse_retries": 2},
        }

        result = run_sample_and_rank_paired(
            experiments_cfg=experiments_cfg,
            sampling_cfg={"mode": "all", "n_repeats": 1, "seed": 42},
            ranker_cfg=ranker_cfg,
            parallelism_cfg={"max_workers": 2},
            llm=llm,
            run_id="int_rej_test",
            output_dir=output_dir,
            paper_selection_cfg={
                "iterations": "last",
                "cumulative": False,
                "include_rejected": True,
                "exclude_timeout": True,
                "top_n": {"enabled": False},
                "balance": {"enabled": False},
            },
        )

        # Should have completed comparisons
        comparisons = result["comparisons"]
        assert len(comparisons) > 0

        # papers/paper_data.json should have reviews
        paper_data_path = os.path.join(output_dir, "papers", "paper_data.json")
        assert os.path.exists(paper_data_path)
        with open(paper_data_path) as f:
            paper_data = json.load(f)

        # All papers collected with reviews
        for rec in paper_data:
            assert "reviews" in rec
