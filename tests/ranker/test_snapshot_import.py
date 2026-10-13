"""Tests for ranker.snapshot_import -- snapshot paper collector."""

import json

import pytest

from ranker.snapshot_import import collect_papers, to_ranker_format

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
        "files_saved": ["private_archive_data.json", "snapshot_metadata.json"],
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

    # Optionally write public archive (for backward compat testing)
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


# ---------------------------------------------------------------------------
# Fixture: build two fake snapshot runs on disk
# ---------------------------------------------------------------------------


@pytest.fixture
def snapshots(tmp_path):
    """Build two fake snapshot runs.

    run_A: 3 iterations
        iter 1: 2 papers (one with code, one without)
        iter 2: +1 paper  (cumulative: 3)
        iter 3: +1 paper  (cumulative: 4)

    run_B: 2 iterations
        iter 1: 2 papers
        iter 2: +1 paper  (cumulative: 3)
    """
    # ---- run_A ----
    run_a = tmp_path / "run_A"
    run_a_name = "run_A_260213_1200"

    papers_t1 = [
        _make_paper("paper_t01_agent_01", "agent_01", 1, code_script="print('hello')"),
        _make_paper("paper_t01_agent_02", "agent_02", 1, code_script=None),
    ]
    papers_t2 = papers_t1 + [
        _make_paper("paper_t02_agent_01", "agent_01", 2, code_script="x = 1"),
    ]
    papers_t3 = papers_t2 + [
        _make_paper("paper_t03_agent_02", "agent_02", 3, code_script=""),
    ]

    _write_iteration(run_a, 1, papers_t1, run_a_name)
    _write_iteration(run_a, 2, papers_t2, run_a_name)
    _write_iteration(run_a, 3, papers_t3, run_a_name)

    # ---- run_B ----
    run_b = tmp_path / "run_B"
    run_b_name = "run_B_260213_1300"

    papers_b_t1 = [
        _make_paper("paper_t01_agent_03", "agent_03", 1, code_script="import os"),
        _make_paper("paper_t01_agent_04", "agent_04", 1),
    ]
    papers_b_t2 = papers_b_t1 + [
        _make_paper("paper_t02_agent_03", "agent_03", 2, code_script="y = 2"),
    ]
    _write_iteration(run_b, 1, papers_b_t1, run_b_name)
    _write_iteration(run_b, 2, papers_b_t2, run_b_name)

    return {
        "run_A": run_a,
        "run_A_name": run_a_name,
        "run_B": run_b,
        "run_B_name": run_b_name,
    }


# ---------------------------------------------------------------------------
# Fixture: snapshot with mixed paper types (accepted, rejected, timeout)
# ---------------------------------------------------------------------------


@pytest.fixture
def mixed_snapshot(tmp_path):
    """Snapshot with 2 accepted + 2 rejected + 1 timeout paper."""
    run = tmp_path / "mixed_run"
    run_name = "mixed_run_260213"

    accepted_1 = _make_paper(
        "paper_t01_agent_01", "agent_01", 1, code_script="print('a')", status="accepted"
    )
    accepted_1["citation_count"] = 3
    accepted_1["collab_agent_ids"] = ["agent_05"]
    accepted_1["cited_paper_ids"] = ["old_paper_1"]

    accepted_2 = _make_paper(
        "paper_t01_agent_02", "agent_02", 1, code_script="print('b')", status="accepted"
    )

    rejected_1 = _make_paper("paper_t01_agent_03", "agent_03", 1, status="rejected")
    rejected_2 = _make_paper("paper_t01_agent_04", "agent_04", 1, status="rejected")

    timeout_1 = _make_timeout_paper("paper_t01_agent_05", "agent_05", 1)

    all_papers = [accepted_1, accepted_2, rejected_1, rejected_2, timeout_1]

    reviews = {
        "paper_t01_agent_01": [
            _make_review(1, "agent_10"),
            _make_review(1, "agent_11"),
        ],
        "paper_t01_agent_02": [],
        "paper_t01_agent_03": [_make_review(1, "agent_12")],
    }

    _write_iteration(run, 1, all_papers, run_name, reviews_by_paper=reviews)

    return {
        "run_dir": run,
        "run_name": run_name,
        "all_papers": all_papers,
        "reviews": reviews,
    }


# ---------------------------------------------------------------------------
# Tests: Basic functionality (updated to use private_archive_data.json)
# ---------------------------------------------------------------------------


def test_collect_last_iteration(snapshots):
    """Default iterations='last' with cumulative=True returns all papers up to max iteration."""
    papers = collect_papers(snapshots["run_A"])
    # run_A: cumulative up to t=3 → all 4 papers
    assert len(papers) == 4
    assert {p["source_iteration"] for p in papers} == {1, 2, 3}


def test_collect_all_iterations(snapshots):
    """iterations='all' returns papers from every iteration."""
    papers = collect_papers(snapshots["run_A"], iterations="all")
    # 2 from t1 + 1 from t2 + 1 from t3 = 4 total
    assert len(papers) == 4
    iters = {p["source_iteration"] for p in papers}
    assert iters == {1, 2, 3}


def test_collect_specific_iterations(snapshots):
    """Explicit iteration list filters correctly (cumulative=False)."""
    papers = collect_papers(snapshots["run_A"], iterations=[1, 3], cumulative=False)
    iters = {p["source_iteration"] for p in papers}
    assert iters == {1, 3}
    # 2 papers from t1 + 1 paper from t3 = 3
    assert len(papers) == 3


def test_output_has_ranker_fields(snapshots):
    """Every returned paper has the four ranker-compatible string fields."""
    papers = collect_papers(snapshots["run_A"], iterations="all")
    for p in papers:
        for key in ("title", "abstract", "manuscript", "code"):
            assert key in p, f"Missing key: {key}"
            assert isinstance(p[key], str), f"{key} should be str, got {type(p[key])}"


def test_provenance_fields(snapshots):
    """Every returned paper has correct provenance metadata."""
    papers = collect_papers(snapshots["run_A"], iterations="all")
    for p in papers:
        assert "paper_id" in p
        assert "source_snapshot" in p
        assert "source_iteration" in p
        assert "primary_agent_id" in p
        assert p["source_snapshot"] == snapshots["run_A_name"]


def test_code_mapping(snapshots):
    """code_script is mapped to 'code'; None/empty becomes empty string."""
    papers = collect_papers(snapshots["run_A"], iterations=[1])
    by_id = {p["paper_id"]: p for p in papers}

    # agent_01 paper has code_script="print('hello')"
    assert by_id["paper_t01_agent_01"]["code"] == "print('hello')"
    # agent_02 paper has code_script=None
    assert by_id["paper_t01_agent_02"]["code"] == ""


def test_to_ranker_format(snapshots):
    """to_ranker_format strips provenance, keeps only ranker keys."""
    papers = collect_papers(snapshots["run_A"])
    ranker_dict = to_ranker_format(papers[0])
    assert set(ranker_dict.keys()) == {"title", "abstract", "manuscript", "code"}
    # No provenance keys
    for key in ("paper_id", "source_snapshot", "source_iteration", "primary_agent_id"):
        assert key not in ranker_dict


def test_multiple_snapshots_last(snapshots):
    """'last' with cumulative=True returns all papers per snapshot."""
    papers = collect_papers(
        [snapshots["run_A"], snapshots["run_B"]],
        iterations="last",
    )
    sources = {p["source_snapshot"] for p in papers}
    assert sources == {snapshots["run_A_name"], snapshots["run_B_name"]}
    # run_A: 4 cumulative papers, run_B: 3 cumulative papers → 7 total
    assert len(papers) == 7


def test_single_path_as_string(snapshots):
    """A single string path (not a list) works as convenience input."""
    papers = collect_papers(str(snapshots["run_A"]))
    assert len(papers) > 0


def test_missing_snapshot_dir():
    """Non-existent snapshot directory raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        collect_papers("/nonexistent/path/to/snapshot")


def test_bad_iteration_numbers(snapshots):
    """Requesting an iteration that doesn't exist raises ValueError."""
    with pytest.raises(ValueError, match="99"):
        collect_papers(snapshots["run_A"], iterations=[99])


def test_multiple_snapshots_specific_iterations(snapshots):
    """Explicit iteration list applied uniformly across multiple snapshots (cumulative=False)."""
    # Both run_A and run_B have iterations 1 and 2
    papers = collect_papers(
        [snapshots["run_A"], snapshots["run_B"]],
        iterations=[1, 2],
        cumulative=False,
    )
    # run_A: 2 from iter 1 + 1 from iter 2 = 3;  run_B: 2 from iter 1 + 1 from iter 2 = 3 → 6 total
    assert len(papers) == 6
    assert all(p["source_iteration"] in {1, 2} for p in papers)
    sources = {p["source_snapshot"] for p in papers}
    assert sources == {snapshots["run_A_name"], snapshots["run_B_name"]}


def test_multiple_snapshots_specific_iteration_missing_in_one(snapshots):
    """Iteration 3 exists in run_A but not run_B -- should raise ValueError."""
    with pytest.raises(ValueError, match="3"):
        collect_papers(
            [snapshots["run_A"], snapshots["run_B"]],
            iterations=[3],
            cumulative=False,
        )


def test_source_snapshot_dir_present(snapshots):
    """Every paper has source_snapshot_dir matching the input folder path."""
    papers = collect_papers(snapshots["run_A"], iterations="all")
    for p in papers:
        assert "source_snapshot_dir" in p
        assert p["source_snapshot_dir"] == str(snapshots["run_A"])

    # Multi-snapshot: each paper's dir matches the corresponding input
    papers_multi = collect_papers(
        [snapshots["run_A"], snapshots["run_B"]],
        iterations="last",
    )
    valid_dirs = {str(snapshots["run_A"]), str(snapshots["run_B"])}
    for p in papers_multi:
        assert p["source_snapshot_dir"] in valid_dirs


# ---------------------------------------------------------------------------
# Tests: cumulative iteration filtering
# ---------------------------------------------------------------------------


def test_collect_last_iteration_non_cumulative(snapshots):
    """iterations='last' with cumulative=False returns only the final iteration's papers."""
    papers = collect_papers(snapshots["run_A"], cumulative=False)
    assert len(papers) == 1
    assert all(p["source_iteration"] == 3 for p in papers)


def test_cumulative_single_iteration_cutoff(snapshots):
    """iterations=[2] with cumulative=True returns papers from t<=2."""
    papers = collect_papers(snapshots["run_A"], iterations=[2], cumulative=True)
    assert len(papers) == 3  # 2 from t=1, 1 from t=2
    assert all(p["source_iteration"] <= 2 for p in papers)


def test_cumulative_single_iteration_cutoff_non_cumulative(snapshots):
    """iterations=[2] with cumulative=False returns papers from t==2 only."""
    papers = collect_papers(snapshots["run_A"], iterations=[2], cumulative=False)
    assert len(papers) == 1
    assert all(p["source_iteration"] == 2 for p in papers)


def test_cumulative_multi_element_list_raises(snapshots):
    """Multi-element list with cumulative=True raises ValueError."""
    with pytest.raises(ValueError, match="cumulative.*not supported"):
        collect_papers(snapshots["run_A"], iterations=[1, 3], cumulative=True)


def test_cumulative_all_iterations_unchanged(snapshots):
    """iterations='all' returns all papers regardless of cumulative flag."""
    papers_true = collect_papers(snapshots["run_A"], iterations="all", cumulative=True)
    papers_false = collect_papers(snapshots["run_A"], iterations="all", cumulative=False)
    assert len(papers_true) == len(papers_false) == 4


def test_cumulative_last_equals_all(snapshots):
    """iterations='last' with cumulative=True returns the same papers as iterations='all'."""
    papers_last = collect_papers(snapshots["run_A"], iterations="last", cumulative=True)
    papers_all = collect_papers(snapshots["run_A"], iterations="all")
    ids_last = {p["paper_id"] for p in papers_last}
    ids_all = {p["paper_id"] for p in papers_all}
    assert ids_last == ids_all


def test_cumulative_multiple_snapshots_cutoff(snapshots):
    """iterations=[1] cumulative=True on multiple snapshots returns t<=1 from each."""
    papers = collect_papers(
        [snapshots["run_A"], snapshots["run_B"]],
        iterations=[1],
        cumulative=True,
    )
    # run_A: 2 papers at t=1, run_B: 2 papers at t=1 → 4 total
    assert len(papers) == 4
    assert all(p["source_iteration"] <= 1 for p in papers)


def test_collect_last_ignores_incomplete_later_iteration(snapshots):
    """A partial highest-numbered iteration directory should not break 'last'."""
    partial_iter = snapshots["run_A"] / "iteration_4"
    (partial_iter / "plots").mkdir(parents=True)

    papers = collect_papers(snapshots["run_A"], iterations="last", cumulative=True)

    assert len(papers) == 4
    assert {p["source_iteration"] for p in papers} == {1, 2, 3}


def test_cutoff_reads_requested_iteration_when_later_iteration_is_incomplete(snapshots):
    """Explicit cumulative cutoffs should not read from a later partial iteration."""
    partial_iter = snapshots["run_A"] / "iteration_4"
    (partial_iter / "plots").mkdir(parents=True)

    papers = collect_papers(snapshots["run_A"], iterations=[2], cumulative=True)

    assert len(papers) == 3
    assert all(p["source_iteration"] <= 2 for p in papers)


# ---------------------------------------------------------------------------
# Tests: metareview passthrough
# ---------------------------------------------------------------------------


def test_metareview_present_in_output(tmp_path):
    """collect_papers() includes the metareview block from the archive."""
    run = tmp_path / "run_mr"
    metareview = {
        "paper_id": "paper_t01_agent_01",
        "meta_review_text": "Good paper",
        "overall_score": 0.85,
        "rank": "1/2",
        "justification": None,
        "decision": "accept",
    }
    paper = _make_paper("paper_t01_agent_01", "agent_01", 1)
    paper["metareview"] = metareview
    _write_iteration(run, 1, [paper], "run_mr_name")

    papers = collect_papers(str(run), iterations="last")
    assert len(papers) == 1
    assert papers[0]["metareview"] is not None
    assert papers[0]["metareview"]["overall_score"] == 0.85
    assert papers[0]["metareview"]["decision"] == "accept"


def test_metareview_null_when_absent(snapshots):
    """Papers without metareview have metareview=None."""
    papers = collect_papers(snapshots["run_A"], iterations="last")
    for p in papers:
        assert "metareview" in p
        assert p["metareview"] is None


def test_to_ranker_format_excludes_metareview(tmp_path):
    """to_ranker_format strips metareview along with other provenance fields."""
    run = tmp_path / "run_mr_strip"
    paper = _make_paper("paper_t01_agent_01", "agent_01", 1)
    paper["metareview"] = {
        "paper_id": "paper_t01_agent_01",
        "meta_review_text": "text",
        "overall_score": 0.5,
        "rank": "1/1",
        "justification": None,
        "decision": "accept",
    }
    _write_iteration(run, 1, [paper], "run_mr_strip_name")

    papers = collect_papers(str(run), iterations="last")
    ranker_dict = to_ranker_format(papers[0])
    assert "metareview" not in ranker_dict
    assert set(ranker_dict.keys()) == {"title", "abstract", "manuscript", "code"}


# ---------------------------------------------------------------------------
# Tests: T1-T12 -- New paper gathering features
# ---------------------------------------------------------------------------


class TestDefaultCollectsAcceptedOnly:
    """T1: Default settings collect only accepted papers."""

    def test_default_collects_accepted_only(self, mixed_snapshot):
        papers = collect_papers(mixed_snapshot["run_dir"])
        assert len(papers) == 2
        assert all(p["status"] == "accepted" for p in papers)


class TestIncludeRejectedTrue:
    """T2: include_rejected=True includes rejected papers (but not timeouts)."""

    def test_include_rejected_true(self, mixed_snapshot):
        papers = collect_papers(mixed_snapshot["run_dir"], include_rejected=True)
        # 2 accepted + 2 rejected = 4 (timeout excluded by default)
        assert len(papers) == 4
        statuses = {p["status"] for p in papers}
        assert statuses == {"accepted", "rejected"}


class TestExcludeTimeoutFalse:
    """T3: exclude_timeout=False includes all papers including timeouts."""

    def test_exclude_timeout_false(self, mixed_snapshot):
        papers = collect_papers(
            mixed_snapshot["run_dir"],
            include_rejected=True,
            exclude_timeout=False,
        )
        # 2 accepted + 2 rejected + 1 timeout = 5
        assert len(papers) == 5


class TestExcludeTimeoutDefaultRemovesTimeouts:
    """T4: Default exclude_timeout=True removes timeout papers."""

    def test_exclude_timeout_default_removes_timeouts(self, tmp_path):
        run = tmp_path / "timeout_test"
        accepted = _make_paper("paper_t01_agent_01", "agent_01", 1, status="accepted")
        timeout = _make_timeout_paper("paper_t01_agent_02", "agent_02", 1)

        _write_iteration(run, 1, [accepted, timeout], "timeout_run")

        papers = collect_papers(run, include_rejected=True)
        assert len(papers) == 1
        assert papers[0]["paper_id"] == "paper_t01_agent_01"


class TestTimeoutDetectionByIdSuffix:
    """T5: Timeout detection works via paper_id suffix alone."""

    def test_timeout_detection_by_id_suffix(self, tmp_path):
        run = tmp_path / "id_suffix_test"
        # Paper with _timeout suffix but NON-TIMEOUT manuscript
        timeout_by_id = _make_paper(
            "paper_t01_agent_01_timeout",
            "agent_01",
            1,
            status="rejected",
        )
        timeout_by_id["manuscript"] = "Some normal manuscript"  # NOT "TIMEOUT"

        accepted = _make_paper("paper_t01_agent_02", "agent_02", 1, status="accepted")

        _write_iteration(run, 1, [accepted, timeout_by_id], "id_suffix_run")

        papers = collect_papers(run, include_rejected=True, exclude_timeout=True)
        assert len(papers) == 1
        assert papers[0]["paper_id"] == "paper_t01_agent_02"


class TestTimeoutDetectionByManuscript:
    """T6: Timeout detection works via manuscript field alone."""

    def test_timeout_detection_by_manuscript(self, tmp_path):
        run = tmp_path / "manuscript_test"
        # Paper with TIMEOUT manuscript but no _timeout suffix in ID
        timeout_by_manuscript = _make_paper(
            "paper_t01_agent_01",
            "agent_01",
            1,
            status="rejected",
        )
        timeout_by_manuscript["manuscript"] = "TIMEOUT"

        accepted = _make_paper("paper_t01_agent_02", "agent_02", 1, status="accepted")

        _write_iteration(run, 1, [accepted, timeout_by_manuscript], "manuscript_run")

        papers = collect_papers(run, include_rejected=True, exclude_timeout=True)
        assert len(papers) == 1
        assert papers[0]["paper_id"] == "paper_t01_agent_02"


class TestAllPaperFieldsPresent:
    """T7: All original paper fields are passed through."""

    def test_all_paper_fields_present(self, mixed_snapshot):
        papers = collect_papers(mixed_snapshot["run_dir"])
        assert len(papers) >= 1
        p = papers[0]  # First accepted paper (has extra fields populated)

        # Original snapshot fields
        assert "collab_agent_ids" in p
        assert "citation_count" in p
        assert "cited_paper_ids" in p
        assert "code_script" in p
        assert "status" in p
        assert "publication_t" in p
        assert "paper_id" in p
        assert "primary_agent_id" in p
        assert "title" in p
        assert "abstract" in p
        assert "manuscript" in p
        assert "metareview" in p

        # Provenance fields
        assert "source_snapshot" in p
        assert "source_snapshot_dir" in p
        assert "source_iteration" in p
        assert "code" in p

        # Reviews
        assert "reviews" in p

        # Verify specific values from the first accepted paper
        assert p["citation_count"] == 3
        assert p["collab_agent_ids"] == ["agent_05"]
        assert p["cited_paper_ids"] == ["old_paper_1"]


class TestReviewsAttached:
    """T8: Reviews are correctly attached to papers."""

    def test_reviews_attached(self, mixed_snapshot):
        papers = collect_papers(mixed_snapshot["run_dir"])
        by_id = {p["paper_id"]: p for p in papers}

        # Paper 1 has 2 reviews
        p1 = by_id["paper_t01_agent_01"]
        assert "reviews" in p1
        assert len(p1["reviews"]) == 2
        assert p1["reviews"][0]["reviewer_id"] == "agent_10"
        assert p1["reviews"][1]["reviewer_id"] == "agent_11"
        assert "scores" in p1["reviews"][0]

        # Paper 2 has 0 reviews
        p2 = by_id["paper_t01_agent_02"]
        assert "reviews" in p2
        assert p2["reviews"] == []


class TestIterationFilteringWithRejected:
    """T9: Iteration filtering works correctly with rejected papers."""

    def test_iteration_filtering_with_rejected(self, tmp_path):
        run = tmp_path / "iter_rejected_test"

        # Build 3 iterations with mixed statuses
        papers_t1 = [
            _make_paper("paper_t01_agent_01", "agent_01", 1, status="accepted"),
            _make_paper("paper_t01_agent_02", "agent_02", 1, status="rejected"),
            _make_timeout_paper("paper_t01_agent_03", "agent_03", 1),
        ]
        papers_t2 = papers_t1 + [
            _make_paper("paper_t02_agent_01", "agent_01", 2, status="accepted"),
            _make_paper("paper_t02_agent_02", "agent_02", 2, status="rejected"),
        ]
        papers_t3 = papers_t2 + [
            _make_paper("paper_t03_agent_01", "agent_01", 3, status="accepted"),
        ]

        _write_iteration(run, 1, papers_t1, "iter_run")
        _write_iteration(run, 2, papers_t2, "iter_run")
        _write_iteration(run, 3, papers_t3, "iter_run")

        # iterations="last" with include_rejected=True, cumulative
        papers = collect_papers(run, iterations="last", include_rejected=True)
        # All papers up to max_t=3, excluding timeout: 2+2+1=5
        assert len(papers) == 5
        statuses = {p["status"] for p in papers}
        assert statuses == {"accepted", "rejected"}
        # No timeout papers
        assert all(not p["paper_id"].endswith("_timeout") for p in papers)


class TestStatusFieldInOutput:
    """T10: Status field is passed through in output."""

    def test_status_field_in_output(self, tmp_path):
        run = tmp_path / "status_test"
        papers_raw = [
            _make_paper("paper_t01_agent_01", "agent_01", 1, status="accepted"),
            _make_paper("paper_t01_agent_02", "agent_02", 1, status="rejected"),
        ]
        _write_iteration(run, 1, papers_raw, "status_run")

        papers = collect_papers(run, include_rejected=True)
        assert len(papers) == 2

        statuses = {p["paper_id"]: p["status"] for p in papers}
        assert statuses["paper_t01_agent_01"] == "accepted"
        assert statuses["paper_t01_agent_02"] == "rejected"


class TestToRankerFormatUnchanged:
    """T11: to_ranker_format still returns only ranker fields."""

    def test_to_ranker_format_unchanged(self, mixed_snapshot):
        papers = collect_papers(mixed_snapshot["run_dir"])
        assert len(papers) >= 1
        ranker_dict = to_ranker_format(papers[0])
        assert set(ranker_dict.keys()) == {"title", "abstract", "manuscript", "code"}
        # No reviews, no provenance, no extra fields
        assert "reviews" not in ranker_dict
        assert "paper_id" not in ranker_dict
        assert "status" not in ranker_dict
        assert "citation_count" not in ranker_dict


class TestMultipleSnapshotsWithIncludeRejected:
    """T12: Multi-snapshot collection works with include_rejected flag."""

    def test_multiple_snapshots_with_include_rejected(self, tmp_path):
        snap_a = tmp_path / "snap_a"
        snap_b = tmp_path / "snap_b"

        papers_a = [
            _make_paper("paper_t01_agent_01", "agent_01", 1, status="accepted"),
            _make_paper("paper_t01_agent_02", "agent_02", 1, status="rejected"),
        ]
        papers_b = [
            _make_paper("paper_t01_agent_03", "agent_03", 1, status="accepted"),
            _make_paper("paper_t01_agent_04", "agent_04", 1, status="rejected"),
        ]

        _write_iteration(snap_a, 1, papers_a, "snap_a_run")
        _write_iteration(snap_b, 1, papers_b, "snap_b_run")

        papers = collect_papers([snap_a, snap_b], include_rejected=True)
        assert len(papers) == 4

        # Check provenance: each paper's source_snapshot_dir is correct
        dirs = {p["source_snapshot_dir"] for p in papers}
        assert dirs == {str(snap_a), str(snap_b)}

        # Both statuses present
        statuses = {p["status"] for p in papers}
        assert statuses == {"accepted", "rejected"}


def test_paper_kind_filter_standard_and_meta(tmp_path):
    run = tmp_path / "kind_filter_run"
    papers = [
        _make_paper("paper_t01_std", "agent_01", 1, paper_kind="research"),
        _make_paper("paper_t01_meta", "agent_02", 1, paper_kind="meta_research"),
    ]
    _write_iteration(run, 1, papers, "kind_filter_run_name")

    standard_only = collect_papers(run, paper_kind_filter="standard_only")
    meta_only = collect_papers(run, paper_kind_filter="meta_only")
    all_papers = collect_papers(run, paper_kind_filter="all")

    assert {p["paper_id"] for p in standard_only} == {"paper_t01_std"}
    assert {p["paper_id"] for p in meta_only} == {"paper_t01_meta"}
    assert {p["paper_id"] for p in all_papers} == {"paper_t01_std", "paper_t01_meta"}


def test_collect_papers_prefers_ranker_pools_when_available(tmp_path):
    run = tmp_path / "ranker_pool_run"
    paper_private = _make_paper("paper_private_archive", "agent_01", 1, paper_kind="research")
    _write_iteration(run, 1, [paper_private], "ranker_pool_run_name")

    iter_dir = run / "iteration_1"
    ranker_pool_dir = iter_dir / "ranker_pools"
    ranker_pool_dir.mkdir(parents=True, exist_ok=True)
    (ranker_pool_dir / "papers_all.jsonl").write_text(
        json.dumps(_make_paper("paper_from_pool", "agent_99", 1, paper_kind="research")) + "\n",
        encoding="utf-8",
    )
    (ranker_pool_dir / "papers_standard.jsonl").write_text(
        json.dumps(_make_paper("paper_from_pool", "agent_99", 1, paper_kind="research")) + "\n",
        encoding="utf-8",
    )
    (ranker_pool_dir / "papers_meta.jsonl").write_text("", encoding="utf-8")
    (ranker_pool_dir / "manifest.json").write_text(
        json.dumps(
            {
                "filters": {"include_rejected": False, "exclude_timeout": True},
                "counts": {"all": 1, "standard": 1, "meta": 0},
                "schema_version": 1,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    papers = collect_papers(
        run, iterations="last", cumulative=True, include_rejected=False, exclude_timeout=True
    )
    assert len(papers) == 1
    assert papers[0]["paper_id"] == "paper_from_pool"


def test_invalid_paper_kind_filter_raises(snapshots):
    with pytest.raises(ValueError, match="paper_kind_filter"):
        collect_papers(snapshots["run_A"], paper_kind_filter="invalid_filter")
