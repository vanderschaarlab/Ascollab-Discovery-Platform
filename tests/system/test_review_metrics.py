"""Basic tests for system/review_metrics.py.

Goal: verify that each metric helper produces sensible output for
representative inputs, not exhaustive coverage. Tests are grouped by
metric category: helpers, support, soundness, significance, originality,
and integration (build_review_metrics_for_papers).
"""

import math
from collections import Counter
from types import SimpleNamespace

import pytest

from system.review_metrics import (
    REVIEW_METRIC_CATALOG,
    _approx_claim_evidence_alignment,
    _clinical_biological_relevance,
    _compute_originality_metrics,
    _compute_significance_metrics,
    _compute_soundness_metrics,
    _compute_support_metrics,
    _concept_entropy,
    _concept_pair_rarity,
    _cosine_similarity,
    _detect_evidence_types,
    _detect_statistical_validity_checks,
    _effect_size_strength,
    _extract_concept_tokens,
    _extract_hypothesis_tokens_from_text,
    _heuristic_internal_consistency_score,
    _hypothesis_overlap_jaccard,
    _mechanistic_novelty_flag,
    _methodological_novelty_flag,
    _percentile_rank,
    _split_sentences,
    _tokenize,
    build_review_metrics_for_papers,
    normalize_metric_allowlist,
    review_metric_catalog,
)
from system.schema import PaperMetadataPrivate
from system.trackers import ResearchStatsLog, SessionStats, SessionStatsRegistry

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_paper(
    paper_id="test_paper",
    primary_agent_id="agent_01",
    title="Test Paper",
    abstract="We hypothesize a novel biomarker for survival.",
    manuscript="",
    cited_paper_ids=None,
    code_script="",
) -> PaperMetadataPrivate:
    return PaperMetadataPrivate(
        paper_id=paper_id,
        primary_agent_id=primary_agent_id,
        collab_agent_ids=[],
        title=title,
        abstract=abstract,
        manuscript=manuscript or abstract,
        citation_count=0,
        publication_t=1,
        cited_paper_ids=cited_paper_ids or [],
        code_script=code_script,
    )


RICH_MANUSCRIPT = (
    "We propose a novel method to analyze gene expression. "
    "In this study, we performed differential expression analysis on 530 TCGA samples. "
    "Results show p < 0.001 and 95% CI: 0.74-0.82. "
    "Cox regression hazard ratio: 1.8 with overall survival as endpoint. "
    "The Kaplan-Meier survival plot confirmed significance. "
    "Pearson correlation r=0.65 was observed. "
    "AUC=0.83 for the classifier. "
    "Cohen's d=0.6 indicates a medium effect size. "
    "Sensitivity analysis and robustness check confirmed findings. "
    "Cross-validation with train-test split was used. "
    "We applied Bonferroni correction for multiple testing. "
    "Variance inflation factor and residuals were examined. "
    "The proportional hazards assumption was verified. "
    "Normality was confirmed via Shapiro-Wilk test. "
    "Patient cohort included 200 individuals from a clinical trial. "
    "This biomarker shows therapeutic relevance for drug response."
)

WEAK_MANUSCRIPT = (
    "We looked at some data. The results are interesting. "
    "Post-hoc analysis only was performed. "
    "The model was unadjusted and underpowered. "
    "Small sample size limits our conclusions."
)


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_split_sentences_basic(self):
        text = "First sentence. Second sentence! Third?"
        sents = _split_sentences(text)
        assert len(sents) == 3
        assert sents[0] == "First sentence."
        assert sents[1] == "Second sentence!"
        assert sents[2] == "Third?"

    def test_split_sentences_empty(self):
        assert _split_sentences("") == []
        assert _split_sentences(None) == []

    def test_tokenize_basic(self):
        tokens = _tokenize("Hello World 123 p-value foo_bar")
        assert "hello" in tokens
        assert "world" in tokens
        assert "p-value" in tokens
        assert "foo_bar" in tokens

    def test_tokenize_empty(self):
        assert _tokenize("") == []
        assert _tokenize(None) == []

    def test_extract_concept_tokens_filters_stopwords(self):
        tokens = _extract_concept_tokens("the patient has cancer and mutation")
        assert "the" not in tokens
        assert "and" not in tokens
        assert "has" not in tokens
        assert "patient" in tokens
        assert "cancer" in tokens
        assert "mutation" in tokens

    def test_extract_concept_tokens_filters_short(self):
        tokens = _extract_concept_tokens("go to we an")
        # all are <= 2 chars or stopwords
        assert len(tokens) == 0

    def test_extract_concept_tokens_filters_digits(self):
        tokens = _extract_concept_tokens("123 456 gene")
        assert "123" not in tokens
        assert "gene" in tokens

    def test_cosine_similarity_identical(self):
        a = Counter({"gene": 3, "expression": 2})
        sim = _cosine_similarity(a, a)
        assert abs(sim - 1.0) < 1e-6

    def test_cosine_similarity_disjoint(self):
        a = Counter({"gene": 1})
        b = Counter({"drug": 1})
        assert _cosine_similarity(a, b) == 0.0

    def test_cosine_similarity_empty(self):
        assert _cosine_similarity(Counter(), Counter({"a": 1})) == 0.0

    def test_percentile_rank(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _percentile_rank(vals, 3.0) == 60.0  # 3 out of 5 <= 3.0
        assert _percentile_rank(vals, 1.0) == 20.0
        assert _percentile_rank(vals, 5.0) == 100.0

    def test_percentile_rank_empty(self):
        assert _percentile_rank([], 1.0) == 50.0

    def test_extract_hypothesis_tokens_with_cue(self):
        text = "We hypothesize that gene X drives tumor growth. This is background."
        tokens = _extract_hypothesis_tokens_from_text(text)
        assert "gene" in tokens
        assert "tumor" in tokens
        assert "growth" in tokens

    def test_extract_hypothesis_tokens_fallback(self):
        # No hypothesis cue -> falls back to first 2 sentences
        text = "Gene X is important. It regulates apoptosis."
        tokens = _extract_hypothesis_tokens_from_text(text)
        assert len(tokens) > 0
        assert "gene" in tokens

    def test_extract_hypothesis_tokens_empty(self):
        assert _extract_hypothesis_tokens_from_text("") == set()
        assert _extract_hypothesis_tokens_from_text(None) == set()


# ---------------------------------------------------------------------------
# Support metrics
# ---------------------------------------------------------------------------


class TestSupportMetrics:
    def test_detect_evidence_types_numerical(self):
        types = _detect_evidence_types("The p < 0.05 indicates significance with 95% CI.")
        assert "numerical_data" in types

    def test_detect_evidence_types_statistical_test(self):
        types = _detect_evidence_types("We used a t-test and ANOVA to compare groups.")
        assert "statistical_test" in types

    def test_detect_evidence_types_dag(self):
        types = _detect_evidence_types(
            "We constructed a directed acyclic graph for causal inference."
        )
        assert "causal_graph_dag" in types

    def test_detect_evidence_types_survival(self):
        types = _detect_evidence_types("Kaplan-Meier curves showed hazard ratio differences.")
        assert "survival_plot_hazard_ratio" in types

    def test_detect_evidence_types_ignores_incidental_hr_and_os(self):
        """Bare \\bhr\\b and \\bos\\b flagged '3 hr incubation' and stray 'OS'."""
        assert "survival_plot_hazard_ratio" not in _detect_evidence_types(
            "Cells were fixed after a 3 hr incubation on the OS."
        )
        assert "survival_plot_hazard_ratio" in _detect_evidence_types(
            "The adjusted HR = 1.8 was significant."
        )
        assert "survival_plot_hazard_ratio" in _detect_evidence_types(
            "We modelled OS.time against treatment arm."
        )

    def test_detect_evidence_types_regression(self):
        types = _detect_evidence_types("Pearson correlation and regression analysis were applied.")
        assert "correlation_regression_clustering" in types

    def test_detect_evidence_types_empty(self):
        types = _detect_evidence_types("This is a generic sentence with no evidence language.")
        assert types == []

    def test_detect_evidence_types_rich(self):
        types = _detect_evidence_types(RICH_MANUSCRIPT)
        # Should detect multiple types
        assert len(types) >= 3

    def test_claim_evidence_alignment_bow_returns_bounded(self):
        score = _approx_claim_evidence_alignment(RICH_MANUSCRIPT, embedder=None)
        assert 0.0 <= score <= 1.0

    def test_claim_evidence_alignment_short_text(self):
        score = _approx_claim_evidence_alignment("Short.", embedder=None)
        assert score == 0.5  # neutral for < 2 sentences

    def test_compute_support_no_stats(self):
        paper = _make_paper(manuscript=RICH_MANUSCRIPT)
        result = _compute_support_metrics(paper, RICH_MANUSCRIPT, stats=None, embedder=None)
        assert "external_papers_read" in result
        assert result["external_papers_read"] == 0
        assert result["successful_code_executions"] == 0
        assert result["lines_of_code"] == 0
        assert isinstance(result["evidence_types_present"], list)
        assert len(result["evidence_types_present"]) > 0
        assert isinstance(result["claim_evidence_alignment_max_cosine"], float)

    def test_compute_support_with_stats(self):
        paper = _make_paper(manuscript=RICH_MANUSCRIPT)
        stats = SimpleNamespace(
            tool_calls={"arxiv": 3, "search_internal_archive": 2},
            code_executions_successful=5,
            code_executions_failed=1,
            code_executions=6,
            lines_of_code=120,
            timeline=[],
        )
        result = _compute_support_metrics(paper, RICH_MANUSCRIPT, stats=stats, embedder=None)
        assert result["external_papers_read"] == 5  # 3 + 2
        assert result["successful_code_executions"] == 5
        assert result["lines_of_code"] == 120

    def test_compute_support_with_collab_logs(self):
        """Collab-agent tool calls should be counted in external_papers_read."""
        paper = _make_paper(manuscript=RICH_MANUSCRIPT)
        collab_log_a = SimpleNamespace(tool_calls={"arxiv": 4})
        collab_log_b = SimpleNamespace(tool_calls={"arxiv": 1, "search_internal_archive": 2})
        stats = SimpleNamespace(
            tool_calls={"arxiv": 1},
            code_executions_successful=0,
            code_executions_failed=0,
            code_executions=0,
            lines_of_code=0,
            timeline=[],
            collab_logs={"agent_a": collab_log_a, "agent_b": collab_log_b},
        )
        result = _compute_support_metrics(paper, RICH_MANUSCRIPT, stats=stats, embedder=None)
        # primary(1) + collab_a(4) + collab_b(1+2) = 8
        assert result["external_papers_read"] == 8

    def test_compute_support_collab_logs_empty(self):
        """Empty collab_logs should not affect the count."""
        paper = _make_paper(manuscript=RICH_MANUSCRIPT)
        stats = SimpleNamespace(
            tool_calls={"arxiv": 3},
            code_executions_successful=0,
            code_executions_failed=0,
            code_executions=0,
            lines_of_code=0,
            timeline=[],
            collab_logs={},
        )
        result = _compute_support_metrics(paper, RICH_MANUSCRIPT, stats=stats, embedder=None)
        assert result["external_papers_read"] == 3

    def test_evidence_breadth_index(self):
        paper = _make_paper(manuscript=RICH_MANUSCRIPT)
        stats = SimpleNamespace(
            tool_calls={"arxiv": 2},
            code_executions_successful=1,
            code_executions_failed=0,
            code_executions=1,
            lines_of_code=10,
            timeline=[],
        )
        result = _compute_support_metrics(paper, RICH_MANUSCRIPT, stats=stats, embedder=None)
        evidence_types = result["evidence_types_present"]
        expected_breadth = 2 + len(set(evidence_types))
        assert result["evidence_breadth_index"] == expected_breadth

    def test_statistical_code_cells_from_timeline(self):
        paper = _make_paper(manuscript=RICH_MANUSCRIPT)
        timeline = [
            {
                "kind": "tool_call",
                "tool": "execute_code",
                "args": {"code": "import scipy\nfrom scipy.stats import ttest_ind"},
            },
            {"kind": "tool_call", "tool": "execute_code", "args": {"code": "print('hello')"}},
            {
                "kind": "tool_call",
                "tool": "execute_code",
                "args": {"code": "import statsmodels\nresult = ols(...)"},
            },
        ]
        stats = SimpleNamespace(
            tool_calls={"arxiv": 0},
            code_executions_successful=3,
            code_executions_failed=0,
            code_executions=3,
            lines_of_code=30,
            timeline=timeline,
        )
        result = _compute_support_metrics(paper, RICH_MANUSCRIPT, stats=stats, embedder=None)
        # 2 of 3 cells have statistical patterns (scipy, statsmodels)
        assert result["num_statistical_code_cells"] == 2

    def test_analytical_depth_equals_stat_cells(self):
        paper = _make_paper(manuscript=RICH_MANUSCRIPT)
        stats = SimpleNamespace(
            tool_calls={},
            code_executions_successful=2,
            code_executions_failed=0,
            code_executions=2,
            lines_of_code=20,
            timeline=[
                {
                    "kind": "tool_call",
                    "tool": "execute_code",
                    "args": {"code": "from lifelines import CoxPHFitter"},
                },
                {"kind": "tool_call", "tool": "execute_code", "args": {"code": "import pingouin"}},
            ],
        )
        result = _compute_support_metrics(paper, RICH_MANUSCRIPT, stats=stats, embedder=None)
        assert result["analytical_depth"] == result["num_statistical_code_cells"]


# ---------------------------------------------------------------------------
# Soundness metrics
# ---------------------------------------------------------------------------


class TestSoundnessMetrics:
    def test_internal_consistency_rich(self):
        score = _heuristic_internal_consistency_score(RICH_MANUSCRIPT)
        assert 0.0 <= score <= 1.0
        # Rich manuscript has many positive cues -> should be above neutral (0.45)
        assert score > 0.5

    def test_internal_consistency_weak(self):
        score = _heuristic_internal_consistency_score(WEAK_MANUSCRIPT)
        assert 0.0 <= score <= 1.0
        # Weak manuscript has negative cues -> should be below neutral
        assert score < 0.45

    def test_internal_consistency_empty(self):
        score = _heuristic_internal_consistency_score("")
        assert score == 0.45  # neutral starting point, no patterns match

    def test_validity_checks_all_pass(self):
        text = (
            "Variance inflation factor was acceptable. "
            "Normality was confirmed via Shapiro-Wilk. "
            "Residuals showed no pattern."
        )
        passed, total = _detect_statistical_validity_checks(text)
        assert total >= 2
        assert passed == total

    def test_validity_checks_some_fail(self):
        text = (
            "Variance inflation factor was acceptable. "
            "Heteroscedasticity was detected, violating assumptions. "
            "Normality was confirmed via Shapiro-Wilk."
        )
        passed, total = _detect_statistical_validity_checks(text)
        assert total >= 2
        assert passed < total

    def test_validity_checks_empty(self):
        passed, total = _detect_statistical_validity_checks("No statistical content here.")
        assert passed == 0
        assert total == 0

    def test_compute_soundness(self):
        result = _compute_soundness_metrics(RICH_MANUSCRIPT, stats=None)
        assert "internal_consistency_score" in result
        assert "num_statistical_validity_checks_passed" in result
        assert "num_statistical_validity_checks_total" in result
        assert "diagnostic_validity_pass_rate" in result
        assert 0.0 <= result["diagnostic_validity_pass_rate"] <= 1.0

    def test_diagnostic_validity_pass_rate_zero_total(self):
        result = _compute_soundness_metrics("Nothing here.", stats=None)
        assert result["diagnostic_validity_pass_rate"] == 0.0
        assert result["num_statistical_validity_checks_total"] == 0


# ---------------------------------------------------------------------------
# Significance metrics
# ---------------------------------------------------------------------------


class TestSignificanceMetrics:
    def test_clinical_relevance_high(self):
        text = "Patient cohort from clinical trial showed drug response and toxicity endpoints with biomarker."
        score = _clinical_biological_relevance(text)
        assert score > 0.5

    def test_clinical_relevance_zero(self):
        score = _clinical_biological_relevance("Abstract algebra and topology of manifolds.")
        assert score == 0.0

    def test_clinical_relevance_saturation(self):
        # Score saturates at 1.0 (hits / 8.0)
        text = " ".join(
            [
                "patient",
                "patients",
                "cohort",
                "trial",
                "clinical",
                "therapy",
                "treatment",
                "drug",
                "response",
                "toxicity",
            ]
        )
        score = _clinical_biological_relevance(text)
        assert score == 1.0

    def test_clinical_relevance_bounded(self):
        score = _clinical_biological_relevance(RICH_MANUSCRIPT)
        assert 0.0 <= score <= 1.0

    def test_effect_size_hazard_ratio(self):
        score = _effect_size_strength("hazard ratio: 1.8")
        assert score > 0.0
        assert score <= 1.0

    def test_effect_size_auc(self):
        score = _effect_size_strength("AUC=0.83")
        assert score > 0.0

    def test_effect_size_cohens_d(self):
        score = _effect_size_strength("Cohen's d=0.6")
        assert score > 0.0

    def test_effect_size_empty(self):
        score = _effect_size_strength("No effect sizes reported here.")
        assert score == 0.0

    def test_effect_size_multiple(self):
        score = _effect_size_strength("hazard ratio: 1.8; AUC=0.83; Cohen's d=0.6")
        assert score > 0.0  # average of multiple extractions

    def test_effect_size_ratio_at_one(self):
        # HR of 1.0 -> log(1.0) = 0 -> score = 0
        score = _effect_size_strength("hazard ratio: 1.0")
        assert score == 0.0

    def test_effect_size_negative_ratio_ignored(self):
        score = _effect_size_strength("hazard ratio: -0.5")
        assert score == 0.0  # negative ratios are skipped

    def test_compute_significance_no_archive(self):
        paper = _make_paper(manuscript=RICH_MANUSCRIPT)
        result, _ = _compute_significance_metrics(paper, RICH_MANUSCRIPT, archive=None)
        assert "clinical_biological_relevance" in result
        assert "clinical_relevance_score" in result
        assert "effect_size_strength" in result
        assert result["distance_to_prior_mean"] is None
        assert result["clinical_biological_relevance"] == result["clinical_relevance_score"]

    def test_distance_to_prior_uses_the_raw_similarity_not_the_boosted_score(self):
        """With boosting on, the ranking score is not a similarity."""
        from system.review_metrics import _semantic_distances_to_prior

        paper = _make_paper(manuscript=RICH_MANUSCRIPT)

        class _Archive:
            def query_semantic(self, query_text, k=3):
                # Boosted ranking scores: the top hit is always normalised to 1.0.
                return [(SimpleNamespace(paper_id="x"), 1.0), (SimpleNamespace(paper_id="y"), 0.0)]

            def query_semantic_with_similarity(self, query_text, k=3):
                return [
                    (SimpleNamespace(paper_id="x"), 1.0, 0.30),
                    (SimpleNamespace(paper_id="y"), 0.0, 0.10),
                ]

        distances = _semantic_distances_to_prior(paper, RICH_MANUSCRIPT, _Archive(), k=5)

        assert distances == [pytest.approx(0.70), pytest.approx(0.90)]

    def test_single_neighbour_reports_no_novelty_signal(self):
        """One neighbour used to yield distance=1.0, i.e. 'maximally novel'."""
        from system.review_metrics import _semantic_distances_to_prior

        paper = _make_paper(manuscript=RICH_MANUSCRIPT)

        class _Archive:
            def query_semantic_with_similarity(self, query_text, k=3):
                return [(SimpleNamespace(paper_id="x"), 1.0, 0.9)]

        assert _semantic_distances_to_prior(paper, RICH_MANUSCRIPT, _Archive(), k=5) == []


# ---------------------------------------------------------------------------
# Originality metrics
# ---------------------------------------------------------------------------


class TestOriginalityMetrics:
    def test_methodological_novelty_true(self):
        text = "We introduce a novel pipeline for gene expression analysis."
        assert _methodological_novelty_flag(text) is True

    def test_methodological_novelty_false(self):
        text = "Standard methods were applied to the dataset."
        assert _methodological_novelty_flag(text) is False

    def test_methodological_novelty_first_application(self):
        text = "This is the first application of deep learning to this workflow."
        assert _methodological_novelty_flag(text) is True

    def test_methodological_novelty_empty(self):
        assert _methodological_novelty_flag("") is False

    def test_mechanistic_novelty_true(self):
        assert _mechanistic_novelty_flag("We discovered a novel mechanism for apoptosis.") is True
        assert _mechanistic_novelty_flag("A previously unreported interaction was found.") is True

    def test_mechanistic_novelty_false(self):
        assert _mechanistic_novelty_flag("Standard pathway analysis was performed.") is False

    def test_hypothesis_overlap_no_archive(self):
        paper = _make_paper(manuscript="We hypothesize gene X drives cancer.")
        result = _hypothesis_overlap_jaccard(paper, archive=None)
        assert result is None  # no prior papers -> None

    def test_concept_entropy_basic(self):
        tokens = ["gene", "cancer", "mutation", "gene", "expression"]
        entropy = _concept_entropy(tokens)
        assert entropy is not None
        assert entropy > 0.0

    def test_concept_entropy_uniform(self):
        # All unique tokens -> max entropy for that count
        tokens = ["gene", "cancer", "mutation", "expression"]
        entropy = _concept_entropy(tokens)
        expected = -4 * (0.25 * math.log(0.25))
        assert abs(entropy - expected) < 1e-6

    def test_concept_entropy_single_token(self):
        tokens = ["gene", "gene", "gene"]
        entropy = _concept_entropy(tokens)
        assert entropy == 0.0  # only one unique token -> zero entropy

    def test_concept_entropy_empty(self):
        assert _concept_entropy([]) is None

    def test_concept_pair_rarity_basic(self):
        concepts = ["gene", "mutation", "cancer", "survival"]
        archive_texts = [
            "gene mutation in cancer patients",
            "survival analysis of gene expression",
            "novel drug targets in cancer",
        ]
        rarity = _concept_pair_rarity(concepts, archive_texts)
        assert rarity is not None
        assert rarity > 0.0

    def test_concept_pair_rarity_no_archive(self):
        assert _concept_pair_rarity(["gene", "cancer"], []) is None

    def test_concept_pair_rarity_short_tokens_filtered(self):
        # tokens <= 3 chars are filtered out by _concept_pair_rarity
        assert _concept_pair_rarity(["abc", "def"], ["abc def text"]) is None

    def test_compute_originality_no_archive(self):
        paper = _make_paper(manuscript=RICH_MANUSCRIPT)
        result = _compute_originality_metrics(paper, RICH_MANUSCRIPT, archive=None, embedder=None)
        assert "hypothesis_overlap_jaccard" in result
        assert "semantic_novelty_knn_distance" in result
        assert "concept_entropy" in result
        assert "methodological_novelty_flag" in result
        assert "mechanistic_novelty_flag" in result
        assert result["semantic_novelty_knn_distance"] is None
        assert result["hypothesis_overlap_jaccard"] is None

    def test_compute_originality_concept_entropy_nonzero(self):
        paper = _make_paper(manuscript=RICH_MANUSCRIPT)
        result = _compute_originality_metrics(paper, RICH_MANUSCRIPT, archive=None, embedder=None)
        # Rich manuscript should produce some concept tokens -> nonzero entropy
        assert result["concept_entropy"] is not None
        assert result["concept_entropy"] > 0.0


# ---------------------------------------------------------------------------
# Integration: build_review_metrics_for_papers
# ---------------------------------------------------------------------------


class TestBuildReviewMetrics:
    def test_basic_output_structure(self):
        paper = _make_paper(manuscript=RICH_MANUSCRIPT)
        metrics = build_review_metrics_for_papers(
            iteration=1,
            candidate_papers=[paper],
            session_stats_registry=SessionStatsRegistry(),
            archive=None,
            embedder=None,
        )
        assert paper.paper_id in metrics
        block = metrics[paper.paper_id]
        assert "support" in block
        assert "soundness" in block
        assert "significance" in block
        assert "originality" in block

    def test_all_expected_keys_present(self):
        paper = _make_paper(manuscript=RICH_MANUSCRIPT)
        metrics = build_review_metrics_for_papers(
            iteration=1,
            candidate_papers=[paper],
            session_stats_registry=SessionStatsRegistry(),
            archive=None,
            embedder=None,
        )
        block = metrics[paper.paper_id]
        for category, keys in REVIEW_METRIC_CATALOG.items():
            assert category in block, f"Missing category: {category}"
            for key in keys:
                assert key in block[category], f"Missing key {category}.{key}"

    def test_multiple_papers(self):
        papers = [
            _make_paper(paper_id="p1", manuscript=RICH_MANUSCRIPT),
            _make_paper(paper_id="p2", manuscript=WEAK_MANUSCRIPT),
        ]
        metrics = build_review_metrics_for_papers(
            iteration=1,
            candidate_papers=papers,
            session_stats_registry=SessionStatsRegistry(),
            archive=None,
            embedder=None,
        )
        assert "p1" in metrics
        assert "p2" in metrics
        # Rich paper should generally score higher on soundness
        rich_soundness = metrics["p1"]["soundness"]["internal_consistency_score"]
        weak_soundness = metrics["p2"]["soundness"]["internal_consistency_score"]
        assert rich_soundness > weak_soundness

    def test_empty_candidates(self):
        metrics = build_review_metrics_for_papers(
            iteration=1,
            candidate_papers=[],
            session_stats_registry=SessionStatsRegistry(),
            archive=None,
            embedder=None,
        )
        assert metrics == {}

    def test_collaborator_work_comes_from_the_authors_session(self):
        """Support counts add each collaborator's log inside the author's session.

        The collaborator's own research session that round (its own paper) does not count.
        """
        paper = PaperMetadataPrivate(
            paper_id="p_collab",
            primary_agent_id="agent_01",
            collab_agent_ids=["agent_02"],
            title="Collab Paper",
            abstract="We hypothesize a novel biomarker.",
            manuscript=RICH_MANUSCRIPT,
            citation_count=0,
            publication_t=1,
            cited_paper_ids=[],
            code_script="import pandas as pd\ndf = pd.read_csv('data.csv')\n",
            research_session_id="agent_01_t01",
        )
        iteration = 1

        collab_in_session = ResearchStatsLog()
        collab_in_session.lines_of_code = 70
        collab_in_session.code_executions = 3
        collab_in_session.code_executions_successful = 3

        primary_stats = ResearchStatsLog()
        primary_stats.lines_of_code = 50
        primary_stats.code_executions = 2
        primary_stats.code_executions_successful = 2
        primary_stats.collab_logs = {"agent_02": collab_in_session}

        collaborators_own_session = ResearchStatsLog()
        collaborators_own_session.lines_of_code = 999
        collaborators_own_session.code_executions = 9
        collaborators_own_session.code_executions_successful = 9

        registry = SessionStatsRegistry()
        registry.add_log(
            [
                SessionStats(
                    iteration=iteration,
                    session_id="agent_01_t01",
                    session_agent_id="agent_01",
                    type="research",
                    stats_logs=primary_stats,
                ),
                SessionStats(
                    iteration=iteration,
                    session_id="agent_02_t01_collab_for_agent_01",
                    session_agent_id="agent_02",
                    type="research",
                    stats_logs=collab_in_session,
                ),
                SessionStats(
                    iteration=iteration,
                    session_id="agent_02_t01",
                    session_agent_id="agent_02",
                    type="research",
                    stats_logs=collaborators_own_session,
                ),
            ]
        )

        metrics = build_review_metrics_for_papers(
            iteration=iteration,
            candidate_papers=[paper],
            session_stats_registry=registry,
            archive=None,
            embedder=None,
        )
        support = metrics["p_collab"]["support"]
        assert support["lines_of_code"] == 120  # 50 + 70
        assert support["successful_code_executions"] == 5  # 2 + 3


# ---------------------------------------------------------------------------
# Allowlist / catalog
# ---------------------------------------------------------------------------


class TestAllowlistAndCatalog:
    def test_review_metric_catalog_structure(self):
        catalog = review_metric_catalog()
        assert set(catalog.keys()) == {"support", "soundness", "significance", "originality"}
        for cat, keys in catalog.items():
            assert isinstance(keys, list)
            assert len(keys) > 0

    def test_normalize_allowlist_none(self):
        allow, unknown = normalize_metric_allowlist(None)
        catalog = review_metric_catalog()
        for cat in catalog:
            assert allow[cat] == set(catalog[cat])
        assert unknown == []

    def test_normalize_allowlist_category_wildcard(self):
        allow, unknown = normalize_metric_allowlist(["support"])
        catalog = review_metric_catalog()
        assert allow["support"] == set(catalog["support"])
        assert allow["soundness"] == set()  # not requested

    def test_normalize_allowlist_star(self):
        allow, _ = normalize_metric_allowlist(["support.*"])
        catalog = review_metric_catalog()
        assert allow["support"] == set(catalog["support"])

    def test_normalize_allowlist_specific_metric(self):
        allow, unknown = normalize_metric_allowlist(["support.external_papers_read"])
        assert "external_papers_read" in allow["support"]
        assert len(allow["support"]) == 1
        assert unknown == []

    def test_normalize_allowlist_unknown(self):
        allow, unknown = normalize_metric_allowlist(["support.nonexistent_metric"])
        assert "nonexistent_metric" not in allow["support"]
        assert "support.nonexistent_metric" in unknown

    def test_normalize_allowlist_unknown_category(self):
        _, unknown = normalize_metric_allowlist(["fake_category"])
        assert "fake_category" in unknown

    def test_allowlist_filters_output(self):
        paper = _make_paper(manuscript=RICH_MANUSCRIPT)
        metrics = build_review_metrics_for_papers(
            iteration=1,
            candidate_papers=[paper],
            session_stats_registry=SessionStatsRegistry(),
            archive=None,
            embedder=None,
            metrics_allowlist=["support.external_papers_read", "soundness"],
        )
        block = metrics[paper.paper_id]
        assert "support" in block
        assert "external_papers_read" in block["support"]
        # Only the one requested support metric should be present
        assert len(block["support"]) == 1
        # All soundness metrics should be present
        assert "soundness" in block
        assert len(block["soundness"]) == len(REVIEW_METRIC_CATALOG["soundness"])


# ---------------------------------------------------------------------------
# Bug regression tests (document known issues)
# ---------------------------------------------------------------------------


class TestBugRegressions:
    def test_or_no_false_positive(self):
        """Regression: 'or' in prose must not be parsed as odds ratio."""
        text = "Group A or 5 patients received treatment."
        assert _effect_size_strength(text) == 0.0

    def test_or_with_separator_still_works(self):
        """OR with = or : separator should still be captured."""
        assert _effect_size_strength("or=1.5") > 0.0
        assert _effect_size_strength("or: 2.1") > 0.0

    def test_r_no_false_positive(self):
        """Regression: standalone 'r' + number must not match without separator."""
        text = "See section r 3 for details."
        assert _effect_size_strength(text) == 0.0

    def test_r_with_separator_still_works(self):
        """r with = or : separator should still be captured."""
        assert _effect_size_strength("r=0.65") > 0.0
        assert _effect_size_strength("r: 0.72") > 0.0
