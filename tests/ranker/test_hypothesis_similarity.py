"""Tests for hypothesis similarity ranker mode.

Covers verdict parsing, system prompt construction, dispatch helpers,
integration tests (real LLM), pipeline integration tests (mock LLM),
and configuration loading.
"""

import os
from unittest.mock import MagicMock

import pytest
from hydra import compose, initialize_config_dir
from langchain_core.messages import AIMessage

from ranker.ranker_core import run_ranker
from ranker.ranker_schemas import RankerResult
from ranker.ranker_utils import (
    HYPOTHESIS_SIMILARITY_NUDGE_MESSAGE,
    NUDGE_MESSAGE,
    get_system_prompt_builder,
    get_verdict_parser,
    make_hypothesis_similarity_system_prompt,
    make_ranker_system_prompt,
    parse_hypothesis_similarity_verdict,
    parse_verdict,
)
from ranker.sample_and_rank_common import _run_single_comparison

# Absolute path to the sys_config directory
_SYS_CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "sys_config"))


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


def _make_paper(paper_id, agent_id, pub_t):
    """Return a minimal paper dict matching the snapshot schema.

    Includes the ``code`` key that ``to_ranker_format()`` expects.
    """
    return {
        "paper_id": paper_id,
        "primary_agent_id": agent_id,
        "collab_agent_ids": [],
        "title": f"Title of {paper_id}",
        "abstract": f"Abstract of {paper_id}",
        "manuscript": f"Manuscript of {paper_id}",
        "code": "",
        "citation_count": 0,
        "publication_t": pub_t,
        "cited_paper_ids": [],
        "code_script": None,
        "metareview": None,
        "status": "accepted",
        "source_snapshot": "snap_1",
        "source_snapshot_dir": "/tmp/snap_1",
        "source_iteration": pub_t,
    }


def _mock_llm_hypothesis_similarity():
    """Return a mock LLM that always returns a 'Same' verdict."""
    mock = MagicMock()
    response = AIMessage(
        content="Verdict:\n- Decision: Same\n- Justification: Both papers test the same hypothesis."
    )
    response.usage_metadata = {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}
    mock.invoke.return_value = response
    return mock


def _mock_llm_different():
    """Return a mock LLM that always returns a 'Different' verdict."""
    mock = MagicMock()
    response = AIMessage(
        content="Verdict:\n- Decision: Different\n- Justification: The papers test different hypotheses."
    )
    response.usage_metadata = {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}
    mock.invoke.return_value = response
    return mock


def _mock_llm_comparison():
    """Return a mock LLM that returns a comparison-mode verdict."""
    mock = MagicMock()
    response = AIMessage(content="Verdict:\n- Winner: A\n- Justification: Paper A is better.")
    response.usage_metadata = {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}
    mock.invoke.return_value = response
    return mock


# ---------------------------------------------------------------------------
# 1. Verdict Parsing Tests
# ---------------------------------------------------------------------------


class TestParseHypothesisSimilarityVerdict:
    """Tests for parse_hypothesis_similarity_verdict()."""

    def test_structured_same(self):
        """1.1. Structured format — same."""
        text = "Verdict:\n- Decision: Same\n- Justification: Both papers test gene X upregulation."
        result, valid = parse_hypothesis_similarity_verdict(text)
        assert valid is True
        assert result["score"] == 0
        assert "gene X upregulation" in result["justification"]

    def test_structured_different(self):
        """1.2. Structured format — different."""
        text = "Verdict:\n- Decision: Different\n- Justification: Paper A studies cancer, Paper B studies diabetes."
        result, valid = parse_hypothesis_similarity_verdict(text)
        assert valid is True
        assert result["score"] == 1
        assert "cancer" in result["justification"]

    def test_messy_whitespace(self):
        """1.3. Messy whitespace."""
        text = "  verdict :\n  -  decision:  Same \n- justification: reasons here"
        result, valid = parse_hypothesis_similarity_verdict(text)
        assert valid is True
        assert result["score"] == 0

    def test_markdown_headers(self):
        """1.4. Markdown headers."""
        text = "## Verdict\n- Decision: Different\n- Justification: Different focus areas"
        result, valid = parse_hypothesis_similarity_verdict(text)
        assert valid is True
        assert result["score"] == 1

    def test_standalone_decision_line(self):
        """1.5. Standalone Decision line (fallback)."""
        text = "After analysis, Decision: Same. Because both papers address the same question."
        result, valid = parse_hypothesis_similarity_verdict(text)
        assert valid is True
        assert result["score"] == 0

    def test_nl_hypotheses_are_the_same(self):
        """1.6. Natural language — 'hypotheses are the same'."""
        text = "The hypotheses are the same, as both papers investigate gene expression in kidney cancer."
        result, valid = parse_hypothesis_similarity_verdict(text)
        assert valid is True
        assert result["score"] == 0

    def test_nl_hypotheses_are_different(self):
        """1.7. Natural language — 'hypotheses are different'."""
        text = (
            "The hypotheses are different because the papers address unrelated research questions."
        )
        result, valid = parse_hypothesis_similarity_verdict(text)
        assert valid is True
        assert result["score"] == 1

    def test_nl_papers_test_same_hypothesis(self):
        """1.8. Natural language — 'papers test the same hypothesis'."""
        text = "These papers test the same hypothesis regarding drug resistance mechanisms."
        result, valid = parse_hypothesis_similarity_verdict(text)
        assert valid is True
        assert result["score"] == 0

    def test_no_verdict(self):
        """1.9. No verdict at all."""
        text = "I need more information to make a decision."
        _, valid = parse_hypothesis_similarity_verdict(text)
        assert valid is False

    def test_empty_string(self):
        """1.10a. Empty string."""
        _, valid = parse_hypothesis_similarity_verdict("")
        assert valid is False

    def test_none_input(self):
        """1.10b. None input."""
        _, valid = parse_hypothesis_similarity_verdict(None)
        assert valid is False

    def test_comparison_format_not_parsed(self):
        """1.11. Old comparison format is NOT parsed."""
        text = "Verdict:\n- Winner: A\n- Justification: Paper A is better"
        _, valid = parse_hypothesis_similarity_verdict(text)
        assert valid is False


# ---------------------------------------------------------------------------
# 2. System Prompt Construction Tests
# ---------------------------------------------------------------------------


class TestHypothesisSimilaritySystemPrompt:
    """Tests for make_hypothesis_similarity_system_prompt()."""

    def test_contains_hypothesis_task(self):
        """2.1. Contains hypothesis comparison task."""
        prompt = make_hypothesis_similarity_system_prompt("Some criteria.")
        assert "same hypothesis" in prompt.lower() or "different hypothes" in prompt.lower()

    def test_contains_criteria_prompt(self):
        """2.2. Contains criteria prompt."""
        criteria = "Custom criterion about what counts as same."
        prompt = make_hypothesis_similarity_system_prompt(criteria)
        assert "Custom criterion about what counts as same." in prompt

    def test_correct_verdict_format(self):
        """2.3. Contains correct verdict format."""
        prompt = make_hypothesis_similarity_system_prompt("criteria")
        assert "Decision:" in prompt
        assert "Same or Different" in prompt
        assert "Winner:" not in prompt

    def test_no_tools_no_action(self):
        """2.4. No tools — no tool instructions."""
        prompt = make_hypothesis_similarity_system_prompt("criteria", tool_signatures=None)
        assert "Action:" not in prompt

    def test_with_tools(self):
        """2.5. With tools — includes tool instructions."""
        prompt = make_hypothesis_similarity_system_prompt(
            "criteria",
            tool_signatures={"arxiv": "arxiv(query: str): Search arXiv."},
        )
        assert "arxiv" in prompt
        assert "Action:" in prompt

    def test_section_notes_included(self):
        """2.6. Section notes included."""
        prompt = make_hypothesis_similarity_system_prompt(
            "criteria",
            section_notes_block="Each paper has a Manuscript section.",
        )
        assert "Manuscript section" in prompt


# ---------------------------------------------------------------------------
# 3. Dispatch Helper Tests
# ---------------------------------------------------------------------------


class TestModeDispatch:
    """Tests for get_system_prompt_builder() and get_verdict_parser()."""

    def test_comparison_builder(self):
        """3.1a. Comparison mode returns correct builder."""
        builder = get_system_prompt_builder("comparison")
        assert builder is make_ranker_system_prompt

    def test_comparison_parser(self):
        """3.1b. Comparison mode returns correct parser."""
        parse_fn, nudge = get_verdict_parser("comparison")
        assert parse_fn is parse_verdict
        assert nudge == NUDGE_MESSAGE

    def test_hypothesis_similarity_builder(self):
        """3.2a. Hypothesis similarity mode returns correct builder."""
        builder = get_system_prompt_builder("hypothesis_similarity")
        assert builder is make_hypothesis_similarity_system_prompt

    def test_hypothesis_similarity_parser(self):
        """3.2b. Hypothesis similarity mode returns correct parser."""
        parse_fn, nudge = get_verdict_parser("hypothesis_similarity")
        assert parse_fn is parse_hypothesis_similarity_verdict
        assert nudge == HYPOTHESIS_SIMILARITY_NUDGE_MESSAGE

    def test_unknown_mode_builder(self):
        """3.3a. Unknown mode raises ValueError for builder."""
        with pytest.raises(ValueError, match="Unknown ranker mode"):
            get_system_prompt_builder("invalid_mode")

    def test_unknown_mode_parser(self):
        """3.3b. Unknown mode raises ValueError for parser."""
        with pytest.raises(ValueError, match="Unknown ranker mode"):
            get_verdict_parser("invalid_mode")


# ---------------------------------------------------------------------------
# 4. Integration Tests (Real LLM)
# ---------------------------------------------------------------------------


@pytest.fixture
def same_hypothesis_papers():
    """Two papers testing the same hypothesis with different methods."""
    paper_a = {
        "title": "Differential gene expression analysis of VHL in TCGA-KIRC",
        "abstract": (
            "We hypothesize that VHL loss-of-function mutations drive upregulation "
            "of HIF-target genes in kidney renal clear cell carcinoma. We test this "
            "using RNA-seq differential expression analysis on 530 TCGA-KIRC samples."
        ),
        "manuscript": (
            "Introduction: VHL is the most commonly mutated gene in KIRC...\n\n"
            "Hypothesis: VHL loss-of-function mutations lead to upregulation of "
            "HIF-target genes in KIRC, driving tumor progression.\n\n"
            "Methods: RNA-seq analysis of 530 TCGA-KIRC samples using DESeq2...\n\n"
            "Results: 47 HIF-target genes significantly upregulated in VHL-mutant samples..."
        ),
    }
    paper_b = {
        "title": "Proteomic validation of VHL-HIF axis in clear cell renal carcinoma",
        "abstract": (
            "We hypothesize that VHL loss-of-function mutations cause upregulation "
            "of HIF-target genes in clear cell renal carcinoma. We validate this "
            "using mass spectrometry-based proteomics on 120 clinical specimens."
        ),
        "manuscript": (
            "Introduction: The VHL-HIF axis is central to KIRC pathogenesis...\n\n"
            "Hypothesis: VHL loss-of-function leads to increased HIF-target "
            "protein levels in KIRC tumors.\n\n"
            "Methods: TMT-labeled quantitative proteomics on 120 KIRC specimens...\n\n"
            "Results: 31 HIF-target proteins significantly elevated in VHL-mutant tumors..."
        ),
    }
    return paper_a, paper_b


@pytest.fixture
def different_hypothesis_papers():
    """Two papers testing clearly different hypotheses."""
    paper_a = {
        "title": "Differential gene expression analysis of VHL in TCGA-KIRC",
        "abstract": (
            "We hypothesize that VHL loss-of-function mutations drive upregulation "
            "of HIF-target genes in kidney renal clear cell carcinoma."
        ),
        "manuscript": (
            "Introduction: VHL is the most commonly mutated gene in KIRC...\n\n"
            "Hypothesis: VHL mutations lead to HIF-target gene upregulation in KIRC.\n\n"
            "Methods: RNA-seq analysis of 530 TCGA-KIRC samples...\n\n"
            "Results: 47 HIF-target genes significantly upregulated..."
        ),
    }
    paper_b = {
        "title": "Machine learning prediction of antibiotic resistance in E. coli",
        "abstract": (
            "We hypothesize that genomic features can predict antibiotic resistance "
            "phenotypes in Escherichia coli using gradient-boosted classifiers."
        ),
        "manuscript": (
            "Introduction: Antibiotic resistance is a global health crisis...\n\n"
            "Hypothesis: Genomic sequence features (SNPs, gene presence/absence) "
            "can accurately predict antibiotic resistance in E. coli.\n\n"
            "Methods: Random forest and XGBoost classifiers on 10,000 E. coli genomes...\n\n"
            "Results: AUC of 0.94 for resistance prediction..."
        ),
    }
    return paper_a, paper_b


@pytest.mark.integration
class TestHypothesisSimilarityIntegration:
    """Integration tests that call a real LLM."""

    def test_same_hypothesis_returns_same(self, llm, same_hypothesis_papers):
        """4.3. Same hypothesis → score 0."""
        paper_a, paper_b = same_hypothesis_papers
        result = run_ranker(
            paper_a=paper_a,
            paper_b=paper_b,
            llm=llm,
            mode="hypothesis_similarity",
            max_iterations=5,
        )
        assert isinstance(result, RankerResult)
        assert result.score == 0  # Same hypothesis
        assert len(result.justification) > 0

    def test_different_hypotheses_returns_different(self, llm, different_hypothesis_papers):
        """4.4. Different hypotheses → score 1."""
        paper_a, paper_b = different_hypothesis_papers
        result = run_ranker(
            paper_a=paper_a,
            paper_b=paper_b,
            llm=llm,
            mode="hypothesis_similarity",
            max_iterations=5,
        )
        assert isinstance(result, RankerResult)
        assert result.score == 1  # Different hypotheses
        assert len(result.justification) > 0

    def test_returns_valid_result_structure(self, llm, same_hypothesis_papers):
        """4.5. Returns valid RankerResult structure."""
        paper_a, paper_b = same_hypothesis_papers
        result = run_ranker(
            paper_a=paper_a,
            paper_b=paper_b,
            llm=llm,
            mode="hypothesis_similarity",
            max_iterations=5,
        )
        assert result.score in (0, 1)
        assert isinstance(result.justification, str)
        assert result.turns >= 1
        assert result.input_tokens > 0
        assert result.output_tokens > 0

    def test_custom_criteria(self, llm, same_hypothesis_papers):
        """4.6. Custom criteria prompt works."""
        paper_a, paper_b = same_hypothesis_papers
        result = run_ranker(
            paper_a=paper_a,
            paper_b=paper_b,
            llm=llm,
            mode="hypothesis_similarity",
            criteria_prompt=(
                "Two hypotheses are the same ONLY if they test the exact same "
                "gene in the exact same organism using the exact same data type."
            ),
            max_iterations=5,
        )
        assert isinstance(result, RankerResult)
        assert result.score in (0, 1)


# ---------------------------------------------------------------------------
# 5. Pipeline Integration Tests (Mocked LLM)
# ---------------------------------------------------------------------------


class TestHypothesisSimilarityPipeline:
    """Pipeline integration tests with mocked LLM."""

    def test_single_comparison_hypothesis_similarity(self):
        """5.2. _run_single_comparison with hypothesis_similarity mode produces outcome='same'."""
        mock_llm = _mock_llm_hypothesis_similarity()
        ranker_cfg = {"mode": "hypothesis_similarity", "max_iterations": 2}
        task = {
            "comparison_id": "cmp_0001",
            "paper_a": _make_paper("p1", "agent1", 1),
            "paper_b": _make_paper("p2", "agent2", 1),
            "paper_a_experiment": "exp_a",
            "paper_b_experiment": "exp_b",
        }
        record = _run_single_comparison(task, mock_llm, ranker_cfg, None, "last")
        assert record["status"] == "completed"
        assert record["score"] == 0
        assert record["outcome"] == "same"

    def test_single_comparison_different(self):
        """5.3. _run_single_comparison with 'Different' verdict."""
        mock_llm = _mock_llm_different()
        ranker_cfg = {"mode": "hypothesis_similarity", "max_iterations": 2}
        task = {
            "comparison_id": "cmp_0002",
            "paper_a": _make_paper("p1", "agent1", 1),
            "paper_b": _make_paper("p2", "agent2", 1),
            "paper_a_experiment": "exp_a",
            "paper_b_experiment": "exp_b",
        }
        record = _run_single_comparison(task, mock_llm, ranker_cfg, None, "last")
        assert record["status"] == "completed"
        assert record["score"] == 1
        assert record["outcome"] == "different"

    def test_single_comparison_comparison_mode(self):
        """5.4. Comparison mode still works (backward compatibility)."""
        mock_llm = _mock_llm_comparison()
        ranker_cfg = {"mode": "comparison", "max_iterations": 2}
        task = {
            "comparison_id": "cmp_0003",
            "paper_a": _make_paper("p1", "agent1", 1),
            "paper_b": _make_paper("p2", "agent2", 1),
            "paper_a_experiment": "exp_a",
            "paper_b_experiment": "exp_b",
        }
        record = _run_single_comparison(task, mock_llm, ranker_cfg, None, "last")
        assert record["outcome"] in ("A", "B")

    def test_default_mode_is_comparison(self):
        """5.5. Default mode is 'comparison' (backward compatibility)."""
        mock_llm = _mock_llm_comparison()
        ranker_cfg = {"max_iterations": 2}  # no mode specified
        task = {
            "comparison_id": "cmp_0004",
            "paper_a": _make_paper("p1", "agent1", 1),
            "paper_b": _make_paper("p2", "agent2", 1),
            "paper_a_experiment": "exp_a",
            "paper_b_experiment": "exp_b",
        }
        record = _run_single_comparison(task, mock_llm, ranker_cfg, None, "last")
        assert record["outcome"] in ("A", "B")


# ---------------------------------------------------------------------------
# 6. Configuration Test
# ---------------------------------------------------------------------------


class TestHypothesisSimilarityConfig:
    """Tests for config loading."""

    def test_load_hypothesis_similarity_config(self):
        """6.1. Load hypothesis_similarity.yaml."""
        with initialize_config_dir(config_dir=_SYS_CONFIG_DIR, version_base=None):
            cfg = compose(
                config_name="ranker",
                overrides=["ranker_cfg@ranker=hypothesis_similarity"],
            )
        assert cfg.ranker.mode == "hypothesis_similarity"
        cp = cfg.ranker.criteria_prompt.lower()
        assert "same" in cp or "hypothesis" in cp

    def test_default_config_has_comparison_mode(self):
        """6.2. Default config has mode=comparison."""
        with initialize_config_dir(config_dir=_SYS_CONFIG_DIR, version_base=None):
            cfg = compose(config_name="ranker")
        assert cfg.ranker.mode == "comparison"
