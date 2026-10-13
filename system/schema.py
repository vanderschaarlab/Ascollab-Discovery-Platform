from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentProfile:
    """Profile for an agent/author/reviewer in the network.

    Attributes:
        agent_id: Unique identifier for the agent.
        behavior: Behavioral description or persona of the agent.
        expertise: Primary expertise summary.
        expertise_topics: List of specific expertise topics.
        citation_count: Total citations accrued by the agent.
        num_accepted_papers: Count of accepted papers authored by the agent.
    """

    agent_id: str
    behavior: str
    expertise: str
    expertise_topics: list[str]
    citation_count: int
    num_accepted_papers: int
    metric_weights: dict[str, float] | None = None
    reputation: dict[str, Any] | None = None
    pause_state: dict[str, Any] | None = None
    learning_state: dict[str, Any] | None = None


@dataclass
class PaperMetaReview:
    """Meta-review summarizing and ranking a paper.

    Attributes:
        paper_id: Identifier of the paper being meta-reviewed.
        meta_review_text: Text of the meta-review.
        overall_score: Numeric overall score assigned.
        rank: Rank string (e.g., '1/4') within a tournament.
        justification: Reasoning for the decision.
        decision: Final decision status (default 'under_review').
    """

    paper_id: str
    meta_review_text: str
    overall_score: float
    rank: str
    justification: str
    decision: str = "under_review"


@dataclass
class PaperMetadata:
    """Public paper metadata for review and archival.

    Attributes:
        paper_id: Unique paper identifier.
        primary_agent_id: Agent ID of the primary author.
        collab_agent_ids: List of collaborator agent IDs.
        title: Paper title.
        abstract: Paper abstract.
        manuscript: Full manuscript text.
        citation_count: Citation count for the paper.
        publication_t: Publication time/index.
        cited_paper_ids: List of cited paper references as dicts.
    """

    paper_id: str
    primary_agent_id: str
    collab_agent_ids: list[str]
    title: str
    abstract: str
    manuscript: str
    citation_count: int
    publication_t: int
    cited_paper_ids: list[dict[str, str]]


@dataclass
class PaperMetadataPrivate(PaperMetadata):
    """Private paper metadata with optional code and metareview."""

    code_script: str | None = None
    metareview: PaperMetaReview | None = None
    # Optional per-paper review metrics captured at review time.
    review_metrics: dict[str, Any] = field(default_factory=dict)
    paper_kind: str = "research"
    research_phase: str = "primary"
    research_session_id: str | None = None
    meta_research_source_paper_ids: list[str] = field(default_factory=list)
    meta_research_trigger_iteration: int | None = None
    meta_quality_gate_status: str = "not_applicable"
    meta_quality_fail_reasons: list[str] = field(default_factory=list)
    meta_revision_attempts: int = 0
    meta_supported_claim_count: int = 0
    meta_unsupported_claim_count: int = 0
    meta_unresolved_uncertainty_claim_count: int = 0
    meta_code_verified_major_claim_count: int = 0
    meta_claim_checks: dict[str, Any] = field(default_factory=dict)
    citation_lint: dict[str, Any] = field(default_factory=dict)
    # Backward-compatible runtime status (historically overloaded).
    status: str = "under_review"
    # Publication decision from metareview/update stage.
    publication_status: str = "under_review"
    # Research-stage validation status before publication decision is applied.
    research_validation_status: str = "under_review"
    # Citation integrity diagnostics from research-stage validation.
    citation_compliant: bool = True
    citation_errors: list[str] = field(default_factory=list)
    citation_error_count: int = 0
    invalid_archive_refs: list[str] = field(default_factory=list)
    # Trust policy visibility flags.
    is_citable_archive_source: bool = False
    citable_with_warnings: bool = False
    # Graph-logic plot artifacts emitted from execute_code runs in this iteration.
    plot_artifacts: list[dict[str, Any]] = field(default_factory=list)
    plot_filenames: list[str] = field(default_factory=list)
    plot_handles: list[str] = field(default_factory=list)
    plot_handle_to_filename: dict[str, str] = field(default_factory=dict)


@dataclass
class PaperReview:
    """Individual reviewer evaluation of a paper."""

    paper_id: str
    reviewer_agent_id: str
    review_text: str
    score: dict[str, float]


@dataclass
class SectionCritiqueIssue:
    """Structured issue extracted during a section-focused critique pass."""

    issue_id: str
    section: str
    severity: str
    category: str
    claim: str
    evidence: list[str]
    confidence: float
    actionable_fix: str


@dataclass
class SectionCritique:
    """Structured critique for a single manuscript section."""

    section: str
    summary: str
    major_issues: list[SectionCritiqueIssue]
    minor_issues: list[SectionCritiqueIssue]
    section_scores: dict[str, float]
    raw_text: str = ""


@dataclass
class ReviewSynthesisBundle:
    """Intermediate bundle used to condense section critiques into a final review."""

    section_critiques: list[SectionCritique]
    canonical_issues: list[dict[str, Any]]
    conflicts: list[dict[str, Any]]
    final_prioritized_issues: list[dict[str, Any]]


@dataclass
class AgentNetworkMetadata:
    """Aggregate statistics describing the agent/paper network."""

    num_agents: int = 0
    min_agent_citation_count: int = 0
    max_agent_citation_count: int = 0
    avg_agent_citation_count: int = 0
    min_agent_accepted_papers: int = 0
    avg_agent_accepted_papers: int = 0
    max_agent_accepted_papers: int = 0
    num_papers: int = 0
    min_paper_citation_count: int = 0
    max_paper_citation_count: int = 0
    avg_paper_citation_count: int = 0
