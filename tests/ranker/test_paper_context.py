"""Tests for additional paper context features (code section + dataset descriptions).

Unit tests for the prompt formatting changes and the dataset description
collection utility. No LLM calls.
"""

import json

import pytest
import yaml

from ranker.ranker_utils import make_ranker_user_message
from ranker.snapshot_import import collect_dataset_descriptions

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def paper_with_code():
    """Paper dict with a non-empty code field."""
    return {
        "title": "Analysis of Gene Expression",
        "abstract": "We analyzed gene expression data.",
        "manuscript": "Introduction: Gene expression analysis...",
        "code": "import pandas as pd\ndf = pd.read_csv('data.csv')\nprint(df.head())",
    }


@pytest.fixture
def paper_without_code():
    """Paper dict with an empty code field."""
    return {
        "title": "Theoretical Framework",
        "abstract": "A theoretical analysis.",
        "manuscript": "Introduction: We present a theoretical framework...",
        "code": "",
    }


@pytest.fixture
def sample_dataset_descriptions():
    """Simple dataset descriptions for testing."""
    return {
        "tcga_luad_hiseqv2": "TCGA LUAD RNA-seq expression matrix.",
        "tcga_luad_rppa": "TCGA LUAD RPPA protein expression matrix.",
    }


def _write_exp_config(path, dataset_keys):
    """Helper: write a minimal exp_config.yaml with the given dataset keys."""
    config = {
        "exp_name": "test_experiment",
        "discovery_dataset": {
            "dataset_keys": dataset_keys,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(config, f)


def _write_snapshot_metadata(iter_dir, run_name="test_run"):
    """Helper: write a minimal snapshot_metadata.json."""
    meta = {
        "iteration": 1,
        "timestamp": "20260213_120000",
        "run_name": run_name,
        "snapshot_created_at": "2026-02-13T12:00:00.000000",
        "num_agents": 2,
        "num_papers_public": 1,
        "files_saved": ["private_archive_data.json", "snapshot_metadata.json"],
    }
    iter_dir.mkdir(parents=True, exist_ok=True)
    (iter_dir / "snapshot_metadata.json").write_text(json.dumps(meta, indent=2))


# ---------------------------------------------------------------------------
# TestUserMessageCodeSection
# ---------------------------------------------------------------------------


class TestUserMessageCodeSection:
    """Tests for the code section behavior in make_ranker_user_message()."""

    def test_code_section_with_code(self, paper_with_code, paper_without_code):
        msg = make_ranker_user_message(
            paper_with_code,
            paper_without_code,
            include_code=True,
        )
        # Paper A section should have ## Code and the code text
        paper_a_section = msg[: msg.index("---")]
        assert "## Code" in paper_a_section
        assert "import pandas as pd" in paper_a_section

    def test_code_section_without_code(self, paper_without_code):
        msg = make_ranker_user_message(
            paper_without_code,
            paper_without_code,
            include_code=True,
        )
        assert "## Code" in msg
        assert "No code was provided for this paper." in msg

    def test_code_section_disabled(self, paper_with_code, paper_without_code):
        msg = make_ranker_user_message(
            paper_with_code,
            paper_without_code,
            include_code=False,
        )
        assert "## Code" not in msg

    def test_code_section_disabled_no_code(self, paper_without_code):
        msg = make_ranker_user_message(
            paper_without_code,
            paper_without_code,
            include_code=False,
        )
        assert "## Code" not in msg
        assert "No code was provided" not in msg


# ---------------------------------------------------------------------------
# TestUserMessageDatasetSection
# ---------------------------------------------------------------------------


class TestUserMessageDatasetSection:
    """Tests for the dataset description section in make_ranker_user_message()."""

    def test_dataset_description_present(self, paper_without_code):
        descriptions_a = {"tcga_luad_hiseqv2": "LUAD expression description text here."}
        msg = make_ranker_user_message(
            paper_without_code,
            paper_without_code,
            include_dataset_description=True,
            dataset_descriptions_a=descriptions_a,
        )
        paper_a_section = msg[: msg.index("---")]
        assert "## Dataset Description" in paper_a_section
        assert "Dataset 1: tcga_luad_hiseqv2" in paper_a_section
        assert "LUAD expression description text here." in paper_a_section

    def test_dataset_description_multiple_datasets(self, paper_without_code):
        descriptions_a = {
            "tcga_luad_hiseqv2": "LUAD expression description.",
            "tcga_luad_rppa": "LUAD RPPA description.",
        }
        msg = make_ranker_user_message(
            paper_without_code,
            paper_without_code,
            include_dataset_description=True,
            dataset_descriptions_a=descriptions_a,
        )
        paper_a_section = msg[: msg.index("---")]
        assert "Dataset 1:" in paper_a_section
        assert "Dataset 2:" in paper_a_section

    def test_dataset_description_none(self, paper_without_code):
        msg = make_ranker_user_message(
            paper_without_code,
            paper_without_code,
            include_dataset_description=True,
            dataset_descriptions_a=None,
        )
        paper_a_section = msg[: msg.index("---")]
        assert "## Dataset Description" in paper_a_section
        assert "No dataset description available" in paper_a_section

    def test_dataset_description_disabled(self, paper_without_code, sample_dataset_descriptions):
        msg = make_ranker_user_message(
            paper_without_code,
            paper_without_code,
            include_dataset_description=False,
            dataset_descriptions_a=sample_dataset_descriptions,
        )
        assert "## Dataset Description" not in msg

    def test_dataset_description_different_per_paper(self, paper_without_code):
        descriptions_a = {"tcga_luad_hiseqv2": "LUAD expression data."}
        descriptions_b = {"tcga_kirc_neurips_hiseqv2": "TCGA KIRC RNA-seq data."}
        msg = make_ranker_user_message(
            paper_without_code,
            paper_without_code,
            include_dataset_description=True,
            dataset_descriptions_a=descriptions_a,
            dataset_descriptions_b=descriptions_b,
        )
        paper_a_section = msg[: msg.index("---")]
        paper_b_section = msg[msg.index("---") :]

        assert "tcga_luad_hiseqv2" in paper_a_section
        assert "tcga_kirc_neurips_hiseqv2" not in paper_a_section

        assert "tcga_kirc_neurips_hiseqv2" in paper_b_section
        assert "tcga_luad_hiseqv2" not in paper_b_section


# ---------------------------------------------------------------------------
# TestUserMessageStructure
# ---------------------------------------------------------------------------


class TestUserMessageStructure:
    """Tests for overall section ordering and formatting."""

    def test_section_order(self, paper_with_code, sample_dataset_descriptions):
        msg = make_ranker_user_message(
            paper_with_code,
            paper_with_code,
            include_code=True,
            include_dataset_description=True,
            dataset_descriptions_a=sample_dataset_descriptions,
            dataset_descriptions_b=sample_dataset_descriptions,
        )
        # Check ordering within Paper A's section
        paper_a_section = msg[: msg.index("---")]
        ds_pos = paper_a_section.index("## Dataset Description")
        ms_pos = paper_a_section.index("## Manuscript")
        code_pos = paper_a_section.index("## Code")
        assert ds_pos < ms_pos < code_pos

    def test_separator_between_papers(self, paper_without_code):
        msg = make_ranker_user_message(paper_without_code, paper_without_code)
        assert "\n---\n" in msg

    def test_manuscript_section_heading(self, paper_without_code):
        msg = make_ranker_user_message(paper_without_code, paper_without_code)
        assert "## Manuscript" in msg
        assert "Title:" in msg
        assert "Abstract:" in msg

    def test_backward_compat_defaults(self, paper_without_code):
        """Call with no keyword args to verify backward-compatible defaults."""
        msg = make_ranker_user_message(paper_without_code, paper_without_code)
        assert "## Manuscript" in msg
        assert "## Code" in msg
        assert "No code was provided for this paper." in msg
        assert "## Dataset Description" in msg
        assert "No dataset description available" in msg


# ---------------------------------------------------------------------------
# TestCollectDatasetDescriptions
# ---------------------------------------------------------------------------


class TestCollectDatasetDescriptions:
    """Tests for collect_dataset_descriptions() in ranker/snapshot_import.py."""

    def test_basic_lookup(self, tmp_path):
        snap_dir = tmp_path / "snap_1"
        iter_dir = snap_dir / "iteration_1"
        _write_snapshot_metadata(iter_dir)
        _write_exp_config(
            iter_dir / "exp_config.yaml",
            dataset_keys=["tcga_kirc_neurips_hiseqv2"],
        )
        papers = [{"source_snapshot_dir": str(snap_dir)}]
        result = collect_dataset_descriptions(papers)

        assert str(snap_dir) in result
        assert "tcga_kirc_neurips_hiseqv2" in result[str(snap_dir)]
        # Value should be a non-empty string from the DATASETS registry
        assert len(result[str(snap_dir)]["tcga_kirc_neurips_hiseqv2"]) > 0

    def test_multiple_dataset_keys(self, tmp_path):
        snap_dir = tmp_path / "snap_multi"
        iter_dir = snap_dir / "iteration_1"
        _write_snapshot_metadata(iter_dir)
        _write_exp_config(
            iter_dir / "exp_config.yaml",
            dataset_keys=["tcga_kirc_neurips_hiseqv2", "tcga_kirc_neurips_survival"],
        )
        papers = [{"source_snapshot_dir": str(snap_dir)}]
        result = collect_dataset_descriptions(papers)

        desc = result[str(snap_dir)]
        assert "tcga_kirc_neurips_hiseqv2" in desc
        assert "tcga_kirc_neurips_survival" in desc

    def test_unknown_dataset_key_skipped(self, tmp_path):
        snap_dir = tmp_path / "snap_unknown"
        iter_dir = snap_dir / "iteration_1"
        _write_snapshot_metadata(iter_dir)
        _write_exp_config(
            iter_dir / "exp_config.yaml",
            dataset_keys=["tcga_kirc_neurips_hiseqv2", "nonexistent_key"],
        )
        papers = [{"source_snapshot_dir": str(snap_dir)}]
        result = collect_dataset_descriptions(papers)

        desc = result[str(snap_dir)]
        assert "tcga_kirc_neurips_hiseqv2" in desc
        assert "nonexistent_key" not in desc

    def test_missing_exp_config(self, tmp_path):
        snap_dir = tmp_path / "snap_no_config"
        iter_dir = snap_dir / "iteration_1"
        _write_snapshot_metadata(iter_dir)
        # No exp_config.yaml written
        papers = [{"source_snapshot_dir": str(snap_dir)}]
        result = collect_dataset_descriptions(papers)

        assert str(snap_dir) in result
        assert result[str(snap_dir)] == {}

    def test_papers_from_different_snapshots(self, tmp_path):
        # Snapshot 1
        snap_1 = tmp_path / "snap_1"
        iter_dir_1 = snap_1 / "iteration_1"
        _write_snapshot_metadata(iter_dir_1)
        _write_exp_config(
            iter_dir_1 / "exp_config.yaml",
            dataset_keys=["tcga_kirc_neurips_hiseqv2"],
        )

        # Snapshot 2
        snap_2 = tmp_path / "snap_2"
        iter_dir_2 = snap_2 / "iteration_1"
        _write_snapshot_metadata(iter_dir_2)
        _write_exp_config(
            iter_dir_2 / "exp_config.yaml",
            dataset_keys=["tcga_luad_hiseqv2"],
        )

        papers = [
            {"source_snapshot_dir": str(snap_1)},
            {"source_snapshot_dir": str(snap_2)},
        ]
        result = collect_dataset_descriptions(papers)

        assert len(result) == 2
        assert "tcga_kirc_neurips_hiseqv2" in result[str(snap_1)]
        assert "tcga_luad_hiseqv2" in result[str(snap_2)]

    def test_empty_papers_list(self):
        result = collect_dataset_descriptions([])
        assert result == {}
