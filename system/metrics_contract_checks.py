"""Runtime checks for review metric and parser contracts.

This module provides lightweight assertions that can be invoked in CI or locally.
"""

from agents.review.rev_agent_utils import parse_review_text
from system.review_metrics import _effect_size_strength, build_review_metrics_for_papers
from system.schema import PaperMetadataPrivate
from system.trackers import SessionStatsRegistry

REQUIRED_METRIC_KEYS: dict[str, set[str]] = {
    "support": {
        "external_papers_read",
        "successful_code_executions",
        "lines_of_code",
        "evidence_types_present",
        "num_statistical_code_cells",
        "evidence_breadth_index",
        "analytical_depth",
        "claim_evidence_alignment_max_cosine",
    },
    "soundness": {
        "internal_consistency_score",
        "num_statistical_validity_checks_passed",
        "num_statistical_validity_checks_total",
        "diagnostic_validity_pass_rate",
    },
    "significance": {
        "distance_to_prior_mean",
        "distance_to_prior_min",
        "distance_to_prior_max",
        "clinical_biological_relevance",
        "clinical_relevance_score",
        "effect_size_strength",
    },
    "originality": {
        "hypothesis_overlap_jaccard",
        "semantic_novelty_knn_distance",
        "embedding_novelty_percentile",
        "concept_pair_rarity",
        "concept_entropy",
        "methodological_novelty_flag",
        "mechanistic_novelty_flag",
    },
}


def assert_effect_size_parsing() -> None:
    score = _effect_size_strength("hazard ratio: 1.8; AUC=0.83; Cohen's d=0.6")
    if score <= 0.0:
        raise AssertionError("effect_size_strength failed to parse common patterns")


def assert_metrics_contract() -> None:
    paper = PaperMetadataPrivate(
        paper_id="paper_contract_check",
        primary_agent_id="agent_01",
        collab_agent_ids=[],
        title="Contract Check",
        abstract="We test a clinical hypothesis.",
        manuscript="We propose a novel method in a patient cohort with hazard ratio: 1.8 and AUC=0.83.",
        citation_count=0,
        publication_t=1,
        cited_paper_ids=[],
        code_script="",
        metareview=None,
        status="under_review",
    )
    metrics = build_review_metrics_for_papers(
        iteration=1,
        candidate_papers=[paper],
        session_stats_registry=SessionStatsRegistry(),
        archive=None,
        embedder=None,
    )
    metric_block = metrics[paper.paper_id]

    for category, required in REQUIRED_METRIC_KEYS.items():
        present = set((metric_block.get(category) or {}).keys())
        missing = required - present
        if missing:
            raise AssertionError(f"metrics contract missing keys in {category}: {sorted(missing)}")


def assert_decimal_review_scores() -> None:
    review = """
# Review Text
1. Summary
Strong work.
2. Motivation
Clear.
3. Claims and Evidence
Reasonable.
4. Soundness of Methodology
Mostly sound.
5. Relation to Prior Knowledge
Adequate.
6. Novelty and Significance
Moderate novelty.
7. Other Comments
None.

# Quantitative Ratings
- Support: 3.5
- Soundness: 3
- Significance: 2.5
- Originality: 3.5
- Overall Recommendation: 3.5
""".strip()
    parsed, ok = parse_review_text(review)
    if not ok:
        raise AssertionError("review parser did not accept decimal scores")
    if parsed.get("Support") != 3.5 or parsed.get("Overall Recommendation") != 3.5:
        raise AssertionError("review parser parsed unexpected values for decimal scores")


def run_all_contract_checks() -> None:
    assert_effect_size_parsing()
    assert_metrics_contract()
    assert_decimal_review_scores()


if __name__ == "__main__":
    run_all_contract_checks()
    print("OK: metrics/review contract checks passed")
