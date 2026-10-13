"""Offline review replay helpers for fixed reviewer-matching cohorts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from system.bootstrap_iteration import _paper_from_dict, load_pool
from system.schema import AgentNetworkMetadata, PaperMetadataPrivate, PaperMetaReview, PaperReview


def paper_from_snapshot_dict(paper: dict[str, Any]) -> PaperMetadataPrivate:
    """Convert a snapshot paper dict into the runtime dataclass."""
    return _paper_from_dict(paper)


def select_top_k_by_metareview(meta_reviews: list[PaperMetaReview], k: int) -> list[str]:
    """Return top-k paper ids by metareview score with deterministic tie-breaks."""
    ranked = sorted(
        meta_reviews,
        key=lambda mr: (
            -float(getattr(mr, "overall_score", 0.0) or 0.0),
            str(getattr(mr, "paper_id", "")),
        ),
    )
    return [mr.paper_id for mr in ranked[: max(0, int(k))]]


def serialize_review(review: PaperReview) -> dict[str, Any]:
    return {
        "paper_id": review.paper_id,
        "reviewer_id": review.reviewer_agent_id,
        "review_text": review.review_text,
        "scores": dict(review.score or {}),
    }


def serialize_metareview(meta_review: PaperMetaReview) -> dict[str, Any]:
    return {
        "paper_id": meta_review.paper_id,
        "meta_review_text": meta_review.meta_review_text,
        "overall_score": meta_review.overall_score,
        "rank": meta_review.rank,
        "justification": meta_review.justification,
        "decision": meta_review.decision,
    }


def load_agents_from_manifest_sources(manifest: dict[str, Any]) -> list[Any]:
    """Load agent profiles from the latest listed source iteration dirs."""
    agents_by_id: dict[str, Any] = {}
    for snap_dir in manifest.get("snapshot_dirs", []):
        iteration = max(
            int(p.get("source_iteration", manifest["target_iteration"]))
            for p in manifest["candidate_papers"]
        )
        iter_dir = Path(snap_dir) / f"iteration_{iteration}"
        agents, *_ = load_pool(iter_dir)
        for agent in agents:
            if getattr(agent, "agent_id", None):
                agents_by_id.setdefault(agent.agent_id, agent)
    return list(agents_by_id.values())


def build_network_metadata(
    agents: list[Any], papers: list[PaperMetadataPrivate]
) -> AgentNetworkMetadata:
    citations = [int(getattr(a, "citation_count", 0) or 0) for a in agents]
    accepted = [int(getattr(a, "num_accepted_papers", 0) or 0) for a in agents]
    paper_citations = [int(getattr(p, "citation_count", 0) or 0) for p in papers]
    return AgentNetworkMetadata(
        num_agents=len(agents),
        min_agent_citation_count=min(citations) if citations else 0,
        max_agent_citation_count=max(citations) if citations else 0,
        avg_agent_citation_count=int(sum(citations) / len(citations)) if citations else 0,
        min_agent_accepted_papers=min(accepted) if accepted else 0,
        avg_agent_accepted_papers=int(sum(accepted) / len(accepted)) if accepted else 0,
        max_agent_accepted_papers=max(accepted) if accepted else 0,
        num_papers=len(papers),
        min_paper_citation_count=min(paper_citations) if paper_citations else 0,
        max_paper_citation_count=max(paper_citations) if paper_citations else 0,
        avg_paper_citation_count=int(sum(paper_citations) / len(paper_citations))
        if paper_citations
        else 0,
    )
