"""Integration tests for additional paper context features.

These tests verify that the new code and dataset description features
work end-to-end through run_ranker() with a real LLM.
Marked with @pytest.mark.integration.
"""

import pytest

from ranker.ranker_core import run_ranker


@pytest.fixture
def sample_dataset_descriptions():
    """Short dataset descriptions for integration testing."""
    return {
        "tcga_kirc_neurips_hiseqv2": (
            "Gene expression RNA-seq data for TCGA Kidney Renal Clear Cell "
            "Carcinoma (KIRC). 20530 genes x 606 samples."
        ),
    }


@pytest.fixture
def paper_a_with_code():
    return {
        "title": "Statistical analysis of gene expression in TCGA-KIRC",
        "abstract": (
            "We performed differential expression analysis on 530 TCGA-KIRC "
            "samples, identifying 47 significantly dysregulated genes."
        ),
        "manuscript": (
            "Introduction: Kidney renal clear cell carcinoma is the most "
            "common subtype. Methods: We used DESeq2. Results: 47 genes "
            "were significant. Discussion: Results are consistent."
        ),
        "code": "import pandas as pd\ndf = pd.read_csv('data.csv')\nprint(df.describe())",
    }


@pytest.fixture
def paper_b_basic():
    return {
        "title": "Thoughts on cancer",
        "abstract": "Cancer is a very important disease. We think about it.",
        "manuscript": ("Cancer is bad. More research is needed. We believe our approach is novel."),
    }


@pytest.mark.integration
class TestCodeSectionIntegration:
    def test_run_ranker_with_code_enabled(self, llm, paper_a_with_code, paper_b_basic):
        result = run_ranker(
            paper_a=paper_a_with_code,
            paper_b=paper_b_basic,
            llm=llm,
            include_code=True,
        )
        assert result.score in (0, 1)
        assert result.justification

    def test_run_ranker_with_code_disabled(self, llm, paper_a_with_code, paper_b_basic):
        result = run_ranker(
            paper_a=paper_a_with_code,
            paper_b=paper_b_basic,
            llm=llm,
            include_code=False,
        )
        assert result.score in (0, 1)
        assert result.justification


@pytest.mark.integration
class TestDatasetDescriptionIntegration:
    def test_run_ranker_with_dataset_descriptions(
        self,
        llm,
        paper_a_with_code,
        paper_b_basic,
        sample_dataset_descriptions,
    ):
        result = run_ranker(
            paper_a=paper_a_with_code,
            paper_b=paper_b_basic,
            llm=llm,
            include_dataset_description=True,
            dataset_descriptions_a=sample_dataset_descriptions,
        )
        assert result.score in (0, 1)
        assert result.justification

    def test_run_ranker_all_features_enabled(
        self,
        llm,
        paper_a_with_code,
        paper_b_basic,
        sample_dataset_descriptions,
    ):
        result = run_ranker(
            paper_a=paper_a_with_code,
            paper_b=paper_b_basic,
            llm=llm,
            include_code=True,
            include_dataset_description=True,
            dataset_descriptions_a=sample_dataset_descriptions,
            dataset_descriptions_b=sample_dataset_descriptions,
        )
        assert result.score in (0, 1)
        assert result.justification
