from system.review_metrics import _effect_size_strength, build_review_metrics_for_papers
from system.schema import PaperMetadataPrivate
from system.trackers import SessionStatsRegistry


def _make_paper() -> PaperMetadataPrivate:
    return PaperMetadataPrivate(
        paper_id="paper_t01_agent_01",
        primary_agent_id="agent_01",
        collab_agent_ids=[],
        title="Test paper",
        abstract="We test a new hypothesis in a clinical cohort.",
        manuscript=(
            "We propose a novel method. "
            "In a patient cohort, hazard ratio: 1.8 and AUC=0.83 indicate clinical relevance. "
            "We report confidence interval 95% CI and model diagnostics."
        ),
        citation_count=0,
        publication_t=1,
        cited_paper_ids=[],
        code_script="",
        metareview=None,
        status="under_review",
    )


def test_effect_size_strength_detects_common_patterns() -> None:
    text = "hazard ratio: 1.8; AUC=0.83; Cohen's d = 0.6"
    score = _effect_size_strength(text)
    assert score > 0.0


def test_metrics_contract_keys_present() -> None:
    paper = _make_paper()
    metrics = build_review_metrics_for_papers(
        iteration=1,
        candidate_papers=[paper],
        session_stats_registry=SessionStatsRegistry(),
        archive=None,
        embedder=None,
    )
    m = metrics[paper.paper_id]

    assert set(m.keys()) == {"support", "soundness", "significance", "originality"}

    assert {
        "external_papers_read",
        "successful_code_executions",
        "lines_of_code",
        "evidence_types_present",
        "num_statistical_code_cells",
        "evidence_breadth_index",
        "analytical_depth",
        "claim_evidence_alignment_max_cosine",
    }.issubset(set(m["support"].keys()))

    assert {
        "internal_consistency_score",
        "num_statistical_validity_checks_passed",
        "num_statistical_validity_checks_total",
        "diagnostic_validity_pass_rate",
    }.issubset(set(m["soundness"].keys()))

    assert {
        "distance_to_prior_mean",
        "distance_to_prior_min",
        "distance_to_prior_max",
        "clinical_biological_relevance",
        "clinical_relevance_score",
        "effect_size_strength",
    }.issubset(set(m["significance"].keys()))

    assert {
        "hypothesis_overlap_jaccard",
        "semantic_novelty_knn_distance",
        "embedding_novelty_percentile",
        "concept_pair_rarity",
        "concept_entropy",
        "methodological_novelty_flag",
        "mechanistic_novelty_flag",
    }.issubset(set(m["originality"].keys()))
