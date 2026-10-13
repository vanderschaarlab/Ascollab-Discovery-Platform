"""Tests for the cited papers feature.

Tests cover:
- Citation chain resolution (build_citation_chains)
- Prompt formatting (cited papers sections in ranker user message)
- Configuration (include_cited_papers in default config)
- Snapshot integration (citation chains from on-disk snapshots)
"""

import json

import pytest

from ranker import sample_and_rank_common
from ranker.ranker_utils import make_ranker_user_message
from ranker.snapshot_import import (
    build_citation_chains,
    collect_citable_papers,
    collect_metrics,
    collect_papers,
    paper_uid,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_paper_with_citations(paper_id, agent_id, pub_t, cited_refs=None, code=""):
    """Return a paper dict in the format returned by collect_papers()."""
    return {
        "paper_id": paper_id,
        "primary_agent_id": agent_id,
        "publication_t": pub_t,
        "cited_paper_ids": cited_refs or [],
        "title": f"Title of {paper_id}",
        "abstract": f"Abstract of {paper_id}",
        "manuscript": f"Manuscript of {paper_id}",
        "code": code,
        "code_script": code or None,
        "source_snapshot": "test_run",
        "source_snapshot_dir": "/tmp/test",
        "source_iteration": pub_t,
        "status": "accepted",
        "citation_count": 0,
        "collab_agent_ids": [],
        "metareview": None,
        "reviews": [],
    }


def _by_pid(chains):
    """Chains keyed by paper_id, for tests over a single run."""
    return {uid.split("::", 1)[1]: chain for uid, chain in chains.items()}


def _write_iteration(run_dir, iteration, papers, run_name, reviews_by_paper=None):
    """Write snapshot files for one iteration (minimal version for citation tests)."""
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

    if reviews_by_paper is None:
        reviews_by_paper = {}

    private_archive = {
        "papers": [papers],
        "reviews_by_paper": reviews_by_paper,
    }
    (iter_dir / "private_archive_data.json").write_text(json.dumps(private_archive, indent=2))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def paper_chain():
    """A set of papers forming a citation chain for testing.

    paper_t01_agent_01 (t=1, cites nothing)
    paper_t01_agent_02 (t=1, cites nothing, has code)
    paper_t02_agent_01 (t=2, cites [paper_t01_agent_01, paper_t01_agent_02])
    paper_t03_agent_01 (t=3, cites [paper_t02_agent_01])
    paper_t04_agent_01 (t=4, cites [paper_t03_agent_01, paper_t01_agent_02])
    """
    p1 = _make_paper_with_citations("paper_t01_agent_01", "agent_01", 1)
    p2 = _make_paper_with_citations("paper_t01_agent_02", "agent_02", 1, code="print('hello')")
    p3 = _make_paper_with_citations(
        "paper_t02_agent_01",
        "agent_01",
        2,
        cited_refs=[
            {"paper_id": "paper_t01_agent_01", "agent_id": "agent_01"},
            {"paper_id": "paper_t01_agent_02", "agent_id": "agent_02"},
        ],
    )
    p4 = _make_paper_with_citations(
        "paper_t03_agent_01",
        "agent_01",
        3,
        cited_refs=[
            {"paper_id": "paper_t02_agent_01", "agent_id": "agent_01"},
        ],
    )
    p5 = _make_paper_with_citations(
        "paper_t04_agent_01",
        "agent_01",
        4,
        cited_refs=[
            {"paper_id": "paper_t03_agent_01", "agent_id": "agent_01"},
            {"paper_id": "paper_t01_agent_02", "agent_id": "agent_02"},
        ],
    )
    return [p1, p2, p3, p4, p5]


@pytest.fixture
def paper_no_citations():
    """A single paper with no citations."""
    return _make_paper_with_citations("paper_t05_agent_01", "agent_01", 5)


# Fixtures for prompt formatting tests


@pytest.fixture
def paper_a():
    """A basic paper for message formatting tests."""
    return {
        "title": "Analysis of Gene Expression",
        "abstract": "We analyzed gene expression data.",
        "manuscript": "Introduction: Gene expression analysis...",
        "code": "import pandas as pd\ndf = pd.read_csv('data.csv')",
    }


@pytest.fixture
def paper_b():
    """A second basic paper for message formatting tests."""
    return {
        "title": "Theoretical Framework",
        "abstract": "A theoretical analysis.",
        "manuscript": "Introduction: We present a theoretical framework...",
        "code": "",
    }


@pytest.fixture
def cp1():
    """A cited paper with code."""
    return {
        "title": "Foundational Study",
        "abstract": "We established the foundation.",
        "manuscript": "This is the foundational manuscript.",
        "code": "print('foundation')",
        "paper_id": "paper_t01_agent_02",
    }


@pytest.fixture
def cp2():
    """A cited paper without code."""
    return {
        "title": "Early Work",
        "abstract": "Early exploration of the field.",
        "manuscript": "This is the early work manuscript.",
        "code": "",
        "paper_id": "paper_t02_agent_05",
    }


@pytest.fixture
def cp3():
    """A third cited paper with code."""
    return {
        "title": "Intermediate Results",
        "abstract": "Building on prior work.",
        "manuscript": "This is the intermediate manuscript.",
        "code": "x = compute()",
        "paper_id": "paper_t03_agent_01",
    }


# ---------------------------------------------------------------------------
# Tests: build_citation_chains
# ---------------------------------------------------------------------------


class TestBuildCitationChains:
    """Tests for build_citation_chains() in ranker/snapshot_import.py."""

    def test_no_citations(self, paper_no_citations):
        """Paper with empty cited_paper_ids gets an empty chain."""
        chains = _by_pid(build_citation_chains([paper_no_citations]))
        assert chains[paper_no_citations["paper_id"]] == []

    def test_direct_citations_only(self, paper_chain):
        """Paper citing two papers (which cite nothing) gets exactly those two, sorted."""
        chains = _by_pid(build_citation_chains(paper_chain))
        chain = chains["paper_t02_agent_01"]
        assert len(chain) == 2
        assert chain[0]["paper_id"] == "paper_t01_agent_01"
        assert chain[1]["paper_id"] == "paper_t01_agent_02"

    def test_transitive_chain(self, paper_chain):
        """Paper citing a paper which cites others gets the full transitive chain."""
        chains = _by_pid(build_citation_chains(paper_chain))
        chain = chains["paper_t03_agent_01"]
        assert len(chain) == 3
        ids = [p["paper_id"] for p in chain]
        assert ids == [
            "paper_t01_agent_01",
            "paper_t01_agent_02",
            "paper_t02_agent_01",
        ]

    def test_deep_chain_with_diamond(self, paper_chain):
        """Paper at end of chain with diamond pattern gets all ancestors, no duplicates."""
        chains = _by_pid(build_citation_chains(paper_chain))
        chain = chains["paper_t04_agent_01"]
        assert len(chain) == 4  # All four prior papers, no duplicates
        ids = [p["paper_id"] for p in chain]
        # Sorted by (publication_t, paper_id):
        assert ids == [
            "paper_t01_agent_01",  # t=1
            "paper_t01_agent_02",  # t=1
            "paper_t02_agent_01",  # t=2
            "paper_t03_agent_01",  # t=3
        ]

    def test_reverse_ancestral_order(self, paper_chain):
        """Chain is sorted by publication_t ascending, paper_id as tiebreaker."""
        chains = _by_pid(build_citation_chains(paper_chain))
        chain = chains["paper_t04_agent_01"]
        pub_times = [p["publication_t"] for p in chain]
        assert pub_times == sorted(pub_times)
        # Within same publication_t, paper_id is sorted
        t1_papers = [p for p in chain if p["publication_t"] == 1]
        assert [p["paper_id"] for p in t1_papers] == sorted(p["paper_id"] for p in t1_papers)

    def test_missing_cited_paper_skipped(self, paper_chain):
        """If a cited_paper_ids entry references a missing paper, it is skipped."""
        paper = _make_paper_with_citations(
            "paper_t05",
            "agent_01",
            5,
            cited_refs=[
                {"paper_id": "nonexistent_paper", "agent_id": "agent_99"},
                {"paper_id": "paper_t01_agent_01", "agent_id": "agent_01"},
            ],
        )
        all_papers = [paper_chain[0], paper]  # paper_t01_agent_01 + paper_t05
        chains = _by_pid(build_citation_chains(all_papers))
        assert len(chains["paper_t05"]) == 1
        assert chains["paper_t05"][0]["paper_id"] == "paper_t01_agent_01"

    def test_empty_input(self):
        """Empty papers list returns empty dict."""
        assert build_citation_chains([]) == {}

    def test_all_papers_get_entry(self, paper_chain):
        """Every paper in the input gets an entry in the output dict."""
        chains = _by_pid(build_citation_chains(paper_chain))
        for p in paper_chain:
            assert p["paper_id"] in chains


# ---------------------------------------------------------------------------
# Tests: Prompt formatting
# ---------------------------------------------------------------------------


class TestCitedPapersPromptSection:
    """Tests for the prompt formatting in ranker/ranker_utils.py."""

    def test_cited_papers_section_present(self, paper_a, paper_b, cp1, cp2):
        """When enabled and cited papers provided, ## Cited Papers heading appears."""
        msg = make_ranker_user_message(
            paper_a,
            paper_b,
            include_cited_papers=True,
            cited_papers_a=[cp1, cp2],
        )
        paper_a_section = msg[: msg.index("---")]
        assert "## Cited Papers" in paper_a_section

    def test_cited_papers_section_absent_when_disabled(self, paper_a, paper_b, cp1):
        """When disabled, no ## Cited Papers heading appears even with data."""
        msg = make_ranker_user_message(
            paper_a,
            paper_b,
            include_cited_papers=False,
            cited_papers_a=[cp1],
        )
        assert "## Cited Papers" not in msg

    def test_cited_papers_section_present_when_no_citations(self, paper_a, paper_b):
        """When enabled but no cited papers, section is emitted with explicit message."""
        msg = make_ranker_user_message(
            paper_a,
            paper_b,
            include_cited_papers=True,
            cited_papers_a=[],
            cited_papers_b=None,
        )
        assert "## Cited Papers" in msg
        assert "This paper does not cite any other papers." in msg

    def test_cited_papers_count_in_intro(self, paper_a, paper_b, cp1, cp2, cp3):
        """Introductory line correctly states the number of cited papers."""
        msg = make_ranker_user_message(
            paper_a,
            paper_b,
            include_cited_papers=True,
            cited_papers_a=[cp1, cp2, cp3],
        )
        paper_a_section = msg[: msg.index("---")]
        assert "3 previously published papers" in paper_a_section

    def test_cited_paper_subheadings(self, paper_a, paper_b, cp1, cp2):
        """Each cited paper gets a ### Cited Paper X of N (paper_id) heading."""
        msg = make_ranker_user_message(
            paper_a,
            paper_b,
            include_cited_papers=True,
            cited_papers_a=[cp1, cp2],
        )
        paper_a_section = msg[: msg.index("---")]
        assert "### Cited Paper 1 of 2" in paper_a_section
        assert "### Cited Paper 2 of 2" in paper_a_section
        assert cp1["paper_id"] in paper_a_section
        assert cp2["paper_id"] in paper_a_section

    def test_cited_paper_content(self, paper_a, paper_b, cp1):
        """Each cited paper shows title, abstract, and manuscript."""
        msg = make_ranker_user_message(
            paper_a,
            paper_b,
            include_cited_papers=True,
            cited_papers_a=[cp1],
        )
        paper_a_section = msg[: msg.index("---")]
        assert f"Title: {cp1['title']}" in paper_a_section
        assert f"Abstract: {cp1['abstract']}" in paper_a_section
        assert cp1["manuscript"] in paper_a_section

    def test_cited_paper_code_shown_when_enabled(self, paper_a, paper_b):
        """When include_code=True and cited paper has code, #### Code appears."""
        cp_with_code = {
            "title": "T",
            "abstract": "A",
            "manuscript": "M",
            "code": "print('hello')",
            "paper_id": "p1",
        }
        msg = make_ranker_user_message(
            paper_a,
            paper_b,
            include_code=True,
            include_cited_papers=True,
            cited_papers_a=[cp_with_code],
        )
        paper_a_section = msg[: msg.index("---")]
        assert "#### Code" in paper_a_section
        assert "print('hello')" in paper_a_section

    def test_cited_paper_code_hidden_when_disabled(self, paper_a, paper_b):
        """When include_code=False, no code section for cited papers."""
        cp_with_code = {
            "title": "T",
            "abstract": "A",
            "manuscript": "M",
            "code": "print('hello')",
            "paper_id": "p1",
        }
        msg = make_ranker_user_message(
            paper_a,
            paper_b,
            include_code=False,
            include_cited_papers=True,
            cited_papers_a=[cp_with_code],
        )
        assert "#### Code" not in msg

    def test_cited_paper_no_code_no_placeholder(self, paper_a, paper_b):
        """When include_code=True but cited paper has no code, no #### Code appears."""
        cp_no_code = {
            "title": "T",
            "abstract": "A",
            "manuscript": "M",
            "code": "",
            "paper_id": "p2",
        }
        msg = make_ranker_user_message(
            paper_a,
            paper_b,
            include_code=True,
            include_cited_papers=True,
            cited_papers_a=[cp_no_code],
        )
        paper_a_section = msg[: msg.index("---")]
        # The main paper's Code section exists
        assert "## Code" in paper_a_section
        # No #### Code sub-heading for the cited paper
        assert "#### Code" not in paper_a_section

    def test_no_dataset_description_in_cited_papers(self, paper_a, paper_b, cp1):
        """Dataset descriptions are never included in cited papers."""
        msg = make_ranker_user_message(
            paper_a,
            paper_b,
            include_dataset_description=True,
            include_cited_papers=True,
            cited_papers_a=[cp1],
            dataset_descriptions_a={"tcga_luad_hiseqv2": "description"},
        )
        paper_a_section = msg[: msg.index("---")]
        # The main paper's dataset description appears once
        ds_count = paper_a_section.count("## Dataset Description")
        assert ds_count == 1

    def test_section_order(self, paper_a, paper_b, cp1):
        """Cited Papers section comes after Code, which comes after Manuscript."""
        msg = make_ranker_user_message(
            paper_a,
            paper_b,
            include_code=True,
            include_dataset_description=True,
            include_cited_papers=True,
            cited_papers_a=[cp1],
            dataset_descriptions_a={"k": "v"},
        )
        paper_a_section = msg[: msg.index("---")]
        ds_pos = paper_a_section.index("## Dataset Description")
        ms_pos = paper_a_section.index("## Manuscript")
        code_pos = paper_a_section.index("## Code")
        cited_pos = paper_a_section.index("## Cited Papers")
        assert ds_pos < ms_pos < code_pos < cited_pos

    def test_independent_cited_papers_per_paper(self, paper_a, paper_b, cp1):
        """Paper A and Paper B can have different cited papers."""
        msg = make_ranker_user_message(
            paper_a,
            paper_b,
            include_cited_papers=True,
            cited_papers_a=[cp1],
            cited_papers_b=None,
        )
        paper_a_section = msg[: msg.index("# Paper B")]
        paper_b_section = msg[msg.index("# Paper B") :]
        assert "## Cited Papers" in paper_a_section
        assert cp1["paper_id"] in paper_a_section
        # Paper B has no citations — section present but with explicit message
        assert "## Cited Papers" in paper_b_section
        assert "This paper does not cite any other papers." in paper_b_section
        assert cp1["paper_id"] not in paper_b_section

    def test_backward_compat_no_cited_papers_args(self, paper_a, paper_b):
        """Calling without cited papers args shows the 'does not cite' message."""
        msg = make_ranker_user_message(paper_a, paper_b)
        assert "## Cited Papers" in msg
        assert "This paper does not cite any other papers." in msg


# ---------------------------------------------------------------------------
# Tests: Configuration
# ---------------------------------------------------------------------------


class TestCitedPapersConfig:
    """Tests for configuration in sys_config/ranker_cfg/default.yaml."""

    def test_default_config_has_include_cited_papers(self, ranker_cfg):
        """The default ranker config contains include_cited_papers set to true."""
        assert ranker_cfg.ranker.include_cited_papers is True


# ---------------------------------------------------------------------------
# Tests: Snapshot integration
# ---------------------------------------------------------------------------


class TestBuildCitationChainsWithSnapshot:
    """Tests using on-disk fake snapshots."""

    def test_citation_chain_from_snapshot(self, tmp_path):
        """Collect papers from snapshot, then build citation chains."""
        run = tmp_path / "citation_run"

        p1 = {
            "paper_id": "paper_t01_agent_01",
            "primary_agent_id": "agent_01",
            "collab_agent_ids": [],
            "title": "Title of paper_t01_agent_01",
            "abstract": "Abstract of paper_t01_agent_01",
            "manuscript": "Manuscript of paper_t01_agent_01",
            "citation_count": 0,
            "publication_t": 1,
            "cited_paper_ids": [],
            "code_script": None,
            "metareview": None,
            "status": "accepted",
        }
        p2 = {
            "paper_id": "paper_t02_agent_01",
            "primary_agent_id": "agent_01",
            "collab_agent_ids": [],
            "title": "Title of paper_t02_agent_01",
            "abstract": "Abstract of paper_t02_agent_01",
            "manuscript": "Manuscript of paper_t02_agent_01",
            "citation_count": 0,
            "publication_t": 2,
            "cited_paper_ids": [
                {"paper_id": "paper_t01_agent_01", "agent_id": "agent_01"},
            ],
            "code_script": "x = 1",
            "metareview": None,
            "status": "accepted",
        }

        # Write both papers in the latest (only) iteration
        all_papers = [p1, p2]
        _write_iteration(run, 2, all_papers, "citation_run_name")

        papers = collect_papers(run)
        chains = _by_pid(build_citation_chains(papers))
        chain = chains["paper_t02_agent_01"]
        assert len(chain) == 1
        assert chain[0]["paper_id"] == "paper_t01_agent_01"

    def test_citation_chain_preserves_paper_fields(self, tmp_path):
        """Each paper in the citation chain retains all its original fields."""
        run = tmp_path / "fields_run"

        p1 = {
            "paper_id": "paper_t01_agent_01",
            "primary_agent_id": "agent_01",
            "collab_agent_ids": [],
            "title": "Title of paper_t01_agent_01",
            "abstract": "Abstract of paper_t01_agent_01",
            "manuscript": "Manuscript of paper_t01_agent_01",
            "citation_count": 0,
            "publication_t": 1,
            "cited_paper_ids": [],
            "code_script": "code_here",
            "metareview": None,
            "status": "accepted",
        }
        p2 = {
            "paper_id": "paper_t02_agent_01",
            "primary_agent_id": "agent_01",
            "collab_agent_ids": [],
            "title": "Title of paper_t02_agent_01",
            "abstract": "Abstract of paper_t02_agent_01",
            "manuscript": "Manuscript of paper_t02_agent_01",
            "citation_count": 0,
            "publication_t": 2,
            "cited_paper_ids": [
                {"paper_id": "paper_t01_agent_01", "agent_id": "agent_01"},
            ],
            "code_script": None,
            "metareview": None,
            "status": "accepted",
        }

        _write_iteration(run, 2, [p1, p2], "fields_run_name")

        papers = collect_papers(run)
        chains = _by_pid(build_citation_chains(papers))
        chain = chains["paper_t02_agent_01"]
        cited = chain[0]
        assert "title" in cited
        assert "manuscript" in cited
        assert "code" in cited
        assert "paper_id" in cited
        assert "publication_t" in cited


# ---------------------------------------------------------------------------
# Tests: several runs pooled, and papers outside the selection
# ---------------------------------------------------------------------------


def _run_papers(run_dir: str, tag: str):
    """Two papers of one run: t02 cites t01. IDs repeat across runs; titles carry the run."""
    t1 = _make_paper_with_citations("paper_t01_agent_01", "agent_01", 1)
    t2 = _make_paper_with_citations(
        "paper_t02_agent_01",
        "agent_01",
        2,
        cited_refs=[{"paper_id": "paper_t01_agent_01", "agent_id": "agent_01"}],
    )
    for p in (t1, t2):
        p["source_snapshot_dir"] = run_dir
        p["title"] = f"{p['title']} ({tag})"
    return t1, t2


class TestCitationChainsAcrossRuns:
    """Paper IDs repeat across runs; selections leave ancestors out."""

    def test_chains_are_keyed_by_run_and_resolved_within_each_run(self):
        a1, a2 = _run_papers("snapshots/run_a", "A")
        b1, b2 = _run_papers("snapshots/run_b", "B")

        chains = build_citation_chains([a1, a2, b1, b2])

        assert chains[paper_uid(a2)] == [a1]
        assert chains[paper_uid(b2)] == [b1]
        assert len(chains) == 4

    def test_ancestors_outside_the_selection_come_from_citable(self):
        a1, a2 = _run_papers("snapshots/run_a", "A")
        b1, b2 = _run_papers("snapshots/run_b", "B")

        # Only the newest paper of each run was selected for ranking.
        assert build_citation_chains([a2, b2])[paper_uid(a2)] == []
        chains = build_citation_chains([a2, b2], citable=[a1, a2, b1, b2])

        assert chains[paper_uid(a2)] == [a1]
        assert chains[paper_uid(b2)] == [b1]

    def test_each_comparison_gets_its_own_runs_chain(self, monkeypatch):
        a1, a2 = _run_papers("snapshots/run_a", "A")
        b1, b2 = _run_papers("snapshots/run_b", "B")
        chains = build_citation_chains([a2, b2], citable=[a1, b1])
        seen = {}

        def fake_run_ranker(**kwargs):
            seen["a"] = kwargs["cited_papers_a"]
            seen["b"] = kwargs["cited_papers_b"]
            raise RuntimeError("stop after capturing the inputs")

        monkeypatch.setattr(sample_and_rank_common, "run_ranker", fake_run_ranker)
        task = {
            "comparison_id": "c0",
            "paper_a": a2,
            "paper_b": b2,
            "paper_a_experiment": "run_a",
            "paper_b_experiment": "run_b",
        }
        record = sample_and_rank_common._run_single_comparison(
            task, None, {"mode": "comparison"}, None, "last", cited_papers_lookup=chains
        )

        assert record["status"] == "failed"  # the fake judge stops after capturing

        assert [p["title"] for p in seen["a"]] == [a1["title"]]
        assert [p["title"] for p in seen["b"]] == [b1["title"]]

    def test_citable_papers_and_metrics_from_two_runs_on_disk(self, tmp_path):
        runs = {}
        for tag in ("a", "b"):
            run = tmp_path / f"run_{tag}"
            t1, t2 = _run_papers(str(run), tag.upper())
            for p in (t1, t2):
                p["source_snapshot_dir"] = ""  # written as stored by the pipeline
            _write_iteration(run, 2, [t1, t2], f"run_{tag}")
            (run / "iteration_2" / "metrics.jsonl").write_text(
                "".join(
                    json.dumps({"paper_id": p["paper_id"], "metrics": {"run": tag}}) + "\n"
                    for p in (t1, t2)
                )
            )
            runs[tag] = str(run)

        selected = collect_papers(list(runs.values()))
        selected = [p for p in selected if p["paper_id"] == "paper_t02_agent_01"]
        citable = collect_citable_papers(list(runs.values()), "last")
        chains = build_citation_chains(selected, citable=citable)
        metrics = collect_metrics(selected)

        assert len(selected) == 2 and len(citable) == 4
        for p in selected:
            run_tag = "A" if p["source_snapshot_dir"] == runs["a"] else "B"
            assert [c["title"] for c in chains[paper_uid(p)]] == [
                f"Title of paper_t01_agent_01 ({run_tag})"
            ]
            assert metrics[paper_uid(p)]["metrics"] == {"run": run_tag.lower()}
