from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

DEFAULT_META_RESEARCHER_CFG: dict[str, Any] = {
    "enabled": False,
    "every_n_iterations": 3,
    "start_iteration": 3,
    "min_public_archive_papers": 4,
    "agent_scope": {
        "mode": "all_active",
        "top_k": None,
        "agent_ids": [],
    },
    "source_corpus": {
        "max_papers": 24,
        "max_chars_per_paper": 1800,
        "sort_by": "recency_then_citations",
    },
    "output": {
        "papers_per_agent": 1,
        "paper_id_suffix": "meta",
        "require_contradictions_section": True,
        "require_archive_citations": True,
        "min_archive_citations": 3,
    },
    "runtime": {
        "code_collaboration": True,
        "disable_collab_tools": False,
        "total_session_timeout_seconds": None,
    },
    "quality_contract": {
        "require_claim_evidence_table": True,
        "require_support_checks_section": True,
        "require_contradiction_resolution_section": True,
        "require_bounded_final_claims_section": True,
        "require_executable_claim_checks_section": True,
        "require_reproducibility_notes_section": True,
        "min_supported_major_claims": 3,
        "min_code_verified_major_claims": 1,
        "max_unsupported_major_claims": 0,
        "max_unresolved_uncertainty_claims": 0,
        "max_words": 950,
    },
    "validation_gate": {
        "enabled": True,
        "max_revision_attempts": 1,
        "allow_single_revision_pass": True,
        "fail_on_unresolved_nan_claims": True,
        "fail_on_missing_evidence_mapping": True,
    },
    "telemetry": {
        "write_iteration_file": True,
        "write_meta_quality_report": True,
        "write_evidence_map": True,
        "write_meta_claim_checks": True,
        "write_citation_lint": True,
        "write_ranker_pools": True,
    },
}


def _as_int(value: Any, default: int, *, min_value: int | None = None) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = int(default)
    if min_value is not None:
        parsed = max(int(min_value), parsed)
    return parsed


def _as_optional_int(
    value: Any, default: int | None = None, *, min_value: int | None = None
) -> int | None:
    if value is None:
        return default
    try:
        parsed = int(value)
    except Exception:
        return default
    if min_value is not None:
        parsed = max(int(min_value), parsed)
    return parsed


def _safe_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def normalize_meta_researcher_cfg(raw_cfg: Any) -> dict[str, Any]:
    cfg = _safe_mapping(raw_cfg)
    agent_scope_raw = _safe_mapping(cfg.get("agent_scope"))
    source_corpus_raw = _safe_mapping(cfg.get("source_corpus"))
    output_raw = _safe_mapping(cfg.get("output"))
    runtime_raw = _safe_mapping(cfg.get("runtime"))
    quality_contract_raw = _safe_mapping(cfg.get("quality_contract"))
    validation_gate_raw = _safe_mapping(cfg.get("validation_gate"))
    telemetry_raw = _safe_mapping(cfg.get("telemetry"))

    scope_mode = (
        str(agent_scope_raw.get("mode", DEFAULT_META_RESEARCHER_CFG["agent_scope"]["mode"]))
        .strip()
        .lower()
    )
    if scope_mode not in {"all_active", "top_k_reputation", "explicit_ids"}:
        scope_mode = "all_active"

    agent_ids: list[str] = []
    for raw in list(agent_scope_raw.get("agent_ids", []) or []):
        aid = str(raw).strip()
        if aid:
            agent_ids.append(aid)
    deduped_agent_ids: list[str] = []
    seen_ids: set[str] = set()
    for aid in agent_ids:
        if aid in seen_ids:
            continue
        seen_ids.add(aid)
        deduped_agent_ids.append(aid)

    normalized: dict[str, Any] = {
        "enabled": bool(cfg.get("enabled", DEFAULT_META_RESEARCHER_CFG["enabled"])),
        "every_n_iterations": _as_int(
            cfg.get("every_n_iterations", DEFAULT_META_RESEARCHER_CFG["every_n_iterations"]),
            DEFAULT_META_RESEARCHER_CFG["every_n_iterations"],
            min_value=1,
        ),
        "start_iteration": _as_int(
            cfg.get("start_iteration", DEFAULT_META_RESEARCHER_CFG["start_iteration"]),
            DEFAULT_META_RESEARCHER_CFG["start_iteration"],
            min_value=1,
        ),
        "min_public_archive_papers": _as_int(
            cfg.get(
                "min_public_archive_papers",
                DEFAULT_META_RESEARCHER_CFG["min_public_archive_papers"],
            ),
            DEFAULT_META_RESEARCHER_CFG["min_public_archive_papers"],
            min_value=0,
        ),
        "agent_scope": {
            "mode": scope_mode,
            "top_k": _as_optional_int(agent_scope_raw.get("top_k"), None, min_value=1),
            "agent_ids": deduped_agent_ids,
        },
        "source_corpus": {
            "max_papers": _as_int(
                source_corpus_raw.get(
                    "max_papers",
                    DEFAULT_META_RESEARCHER_CFG["source_corpus"]["max_papers"],
                ),
                DEFAULT_META_RESEARCHER_CFG["source_corpus"]["max_papers"],
                min_value=1,
            ),
            "max_chars_per_paper": _as_int(
                source_corpus_raw.get(
                    "max_chars_per_paper",
                    DEFAULT_META_RESEARCHER_CFG["source_corpus"]["max_chars_per_paper"],
                ),
                DEFAULT_META_RESEARCHER_CFG["source_corpus"]["max_chars_per_paper"],
                min_value=200,
            ),
            "sort_by": str(
                source_corpus_raw.get(
                    "sort_by",
                    DEFAULT_META_RESEARCHER_CFG["source_corpus"]["sort_by"],
                )
            )
            .strip()
            .lower()
            or DEFAULT_META_RESEARCHER_CFG["source_corpus"]["sort_by"],
        },
        "output": {
            "papers_per_agent": _as_int(
                output_raw.get(
                    "papers_per_agent", DEFAULT_META_RESEARCHER_CFG["output"]["papers_per_agent"]
                ),
                DEFAULT_META_RESEARCHER_CFG["output"]["papers_per_agent"],
                min_value=1,
            ),
            "paper_id_suffix": str(
                output_raw.get(
                    "paper_id_suffix", DEFAULT_META_RESEARCHER_CFG["output"]["paper_id_suffix"]
                )
            ).strip()
            or DEFAULT_META_RESEARCHER_CFG["output"]["paper_id_suffix"],
            "require_contradictions_section": bool(
                output_raw.get(
                    "require_contradictions_section",
                    DEFAULT_META_RESEARCHER_CFG["output"]["require_contradictions_section"],
                )
            ),
            "require_archive_citations": bool(
                output_raw.get(
                    "require_archive_citations",
                    DEFAULT_META_RESEARCHER_CFG["output"]["require_archive_citations"],
                )
            ),
            "min_archive_citations": _as_int(
                output_raw.get(
                    "min_archive_citations",
                    DEFAULT_META_RESEARCHER_CFG["output"]["min_archive_citations"],
                ),
                DEFAULT_META_RESEARCHER_CFG["output"]["min_archive_citations"],
                min_value=0,
            ),
        },
        "runtime": {
            "code_collaboration": bool(
                runtime_raw.get(
                    "code_collaboration",
                    DEFAULT_META_RESEARCHER_CFG["runtime"]["code_collaboration"],
                )
            ),
            "disable_collab_tools": bool(
                runtime_raw.get(
                    "disable_collab_tools",
                    DEFAULT_META_RESEARCHER_CFG["runtime"]["disable_collab_tools"],
                )
            ),
            "total_session_timeout_seconds": _as_optional_int(
                runtime_raw.get("total_session_timeout_seconds"),
                DEFAULT_META_RESEARCHER_CFG["runtime"]["total_session_timeout_seconds"],
                min_value=1,
            ),
        },
        "quality_contract": {
            "require_claim_evidence_table": bool(
                quality_contract_raw.get(
                    "require_claim_evidence_table",
                    DEFAULT_META_RESEARCHER_CFG["quality_contract"]["require_claim_evidence_table"],
                )
            ),
            "require_support_checks_section": bool(
                quality_contract_raw.get(
                    "require_support_checks_section",
                    DEFAULT_META_RESEARCHER_CFG["quality_contract"][
                        "require_support_checks_section"
                    ],
                )
            ),
            "require_contradiction_resolution_section": bool(
                quality_contract_raw.get(
                    "require_contradiction_resolution_section",
                    DEFAULT_META_RESEARCHER_CFG["quality_contract"][
                        "require_contradiction_resolution_section"
                    ],
                )
            ),
            "require_bounded_final_claims_section": bool(
                quality_contract_raw.get(
                    "require_bounded_final_claims_section",
                    DEFAULT_META_RESEARCHER_CFG["quality_contract"][
                        "require_bounded_final_claims_section"
                    ],
                )
            ),
            "require_executable_claim_checks_section": bool(
                quality_contract_raw.get(
                    "require_executable_claim_checks_section",
                    DEFAULT_META_RESEARCHER_CFG["quality_contract"][
                        "require_executable_claim_checks_section"
                    ],
                )
            ),
            "require_reproducibility_notes_section": bool(
                quality_contract_raw.get(
                    "require_reproducibility_notes_section",
                    DEFAULT_META_RESEARCHER_CFG["quality_contract"][
                        "require_reproducibility_notes_section"
                    ],
                )
            ),
            "min_supported_major_claims": _as_int(
                quality_contract_raw.get(
                    "min_supported_major_claims",
                    DEFAULT_META_RESEARCHER_CFG["quality_contract"]["min_supported_major_claims"],
                ),
                DEFAULT_META_RESEARCHER_CFG["quality_contract"]["min_supported_major_claims"],
                min_value=0,
            ),
            "min_code_verified_major_claims": _as_int(
                quality_contract_raw.get(
                    "min_code_verified_major_claims",
                    DEFAULT_META_RESEARCHER_CFG["quality_contract"][
                        "min_code_verified_major_claims"
                    ],
                ),
                DEFAULT_META_RESEARCHER_CFG["quality_contract"]["min_code_verified_major_claims"],
                min_value=0,
            ),
            "max_unsupported_major_claims": _as_int(
                quality_contract_raw.get(
                    "max_unsupported_major_claims",
                    DEFAULT_META_RESEARCHER_CFG["quality_contract"]["max_unsupported_major_claims"],
                ),
                DEFAULT_META_RESEARCHER_CFG["quality_contract"]["max_unsupported_major_claims"],
                min_value=0,
            ),
            "max_unresolved_uncertainty_claims": _as_int(
                quality_contract_raw.get(
                    "max_unresolved_uncertainty_claims",
                    DEFAULT_META_RESEARCHER_CFG["quality_contract"][
                        "max_unresolved_uncertainty_claims"
                    ],
                ),
                DEFAULT_META_RESEARCHER_CFG["quality_contract"][
                    "max_unresolved_uncertainty_claims"
                ],
                min_value=0,
            ),
            "max_words": _as_int(
                quality_contract_raw.get(
                    "max_words",
                    DEFAULT_META_RESEARCHER_CFG["quality_contract"]["max_words"],
                ),
                DEFAULT_META_RESEARCHER_CFG["quality_contract"]["max_words"],
                min_value=1,
            ),
        },
        "validation_gate": {
            "enabled": bool(
                validation_gate_raw.get(
                    "enabled",
                    DEFAULT_META_RESEARCHER_CFG["validation_gate"]["enabled"],
                )
            ),
            "max_revision_attempts": _as_int(
                validation_gate_raw.get(
                    "max_revision_attempts",
                    1
                    if bool(
                        validation_gate_raw.get(
                            "allow_single_revision_pass",
                            DEFAULT_META_RESEARCHER_CFG["validation_gate"][
                                "allow_single_revision_pass"
                            ],
                        )
                    )
                    else 0,
                ),
                1
                if bool(
                    validation_gate_raw.get(
                        "allow_single_revision_pass",
                        DEFAULT_META_RESEARCHER_CFG["validation_gate"][
                            "allow_single_revision_pass"
                        ],
                    )
                )
                else 0,
                min_value=0,
            ),
            "allow_single_revision_pass": bool(
                validation_gate_raw.get(
                    "allow_single_revision_pass",
                    DEFAULT_META_RESEARCHER_CFG["validation_gate"]["allow_single_revision_pass"],
                )
            ),
            "fail_on_unresolved_nan_claims": bool(
                validation_gate_raw.get(
                    "fail_on_unresolved_nan_claims",
                    DEFAULT_META_RESEARCHER_CFG["validation_gate"]["fail_on_unresolved_nan_claims"],
                )
            ),
            "fail_on_missing_evidence_mapping": bool(
                validation_gate_raw.get(
                    "fail_on_missing_evidence_mapping",
                    DEFAULT_META_RESEARCHER_CFG["validation_gate"][
                        "fail_on_missing_evidence_mapping"
                    ],
                )
            ),
        },
        "telemetry": {
            "write_iteration_file": bool(
                telemetry_raw.get(
                    "write_iteration_file",
                    DEFAULT_META_RESEARCHER_CFG["telemetry"]["write_iteration_file"],
                )
            ),
            "write_meta_quality_report": bool(
                telemetry_raw.get(
                    "write_meta_quality_report",
                    DEFAULT_META_RESEARCHER_CFG["telemetry"]["write_meta_quality_report"],
                )
            ),
            "write_evidence_map": bool(
                telemetry_raw.get(
                    "write_evidence_map",
                    DEFAULT_META_RESEARCHER_CFG["telemetry"]["write_evidence_map"],
                )
            ),
            "write_meta_claim_checks": bool(
                telemetry_raw.get(
                    "write_meta_claim_checks",
                    DEFAULT_META_RESEARCHER_CFG["telemetry"]["write_meta_claim_checks"],
                )
            ),
            "write_citation_lint": bool(
                telemetry_raw.get(
                    "write_citation_lint",
                    DEFAULT_META_RESEARCHER_CFG["telemetry"]["write_citation_lint"],
                )
            ),
            "write_ranker_pools": bool(
                telemetry_raw.get(
                    "write_ranker_pools",
                    DEFAULT_META_RESEARCHER_CFG["telemetry"]["write_ranker_pools"],
                )
            ),
        },
    }
    return normalized


def evaluate_meta_research_trigger(*, iteration: int, cfg: Mapping[str, Any]) -> dict[str, Any]:
    every_n = _as_int(cfg.get("every_n_iterations", 1), 1, min_value=1)
    start_iteration = _as_int(cfg.get("start_iteration", 1), 1, min_value=1)
    enabled = bool(cfg.get("enabled", False))

    triggered = False
    reason = "disabled"
    if enabled:
        if iteration < start_iteration:
            reason = "before_start_iteration"
        else:
            offset = iteration - start_iteration
            triggered = (offset % every_n) == 0
            reason = "scheduled" if triggered else "not_scheduled"

    return {
        "iteration": int(iteration),
        "every_n_iterations": int(every_n),
        "start_iteration": int(start_iteration),
        "enabled": bool(enabled),
        "triggered": bool(triggered),
        "reason": reason,
    }


def _agent_id(agent: Any) -> str:
    return str(getattr(agent, "agent_id", None) or getattr(agent, "id", "")).strip()


def select_meta_research_agent_ids(
    *,
    agents: list[Any],
    active_agent_ids: Iterable[str],
    cfg: Mapping[str, Any],
) -> list[str]:
    active_set = {str(x).strip() for x in active_agent_ids if str(x).strip()}
    if not active_set:
        return []

    scope = _safe_mapping(cfg.get("agent_scope"))
    mode = str(scope.get("mode", "all_active") or "all_active").strip().lower()

    agents_by_id: dict[str, Any] = {}
    active_ordered: list[str] = []
    for agent in agents:
        aid = _agent_id(agent)
        if not aid or aid not in active_set:
            continue
        agents_by_id[aid] = agent
        active_ordered.append(aid)

    if mode == "explicit_ids":
        explicit = [
            str(x).strip() for x in list(scope.get("agent_ids", []) or []) if str(x).strip()
        ]
        seen: set[str] = set()
        out: list[str] = []
        for aid in explicit:
            if aid in seen or aid not in agents_by_id:
                continue
            seen.add(aid)
            out.append(aid)
        return out

    if mode == "top_k_reputation":
        top_k = _as_optional_int(scope.get("top_k"), None, min_value=1)
        rows: list[tuple[float, str]] = []
        for aid in active_ordered:
            agent = agents_by_id.get(aid)
            rep = getattr(agent, "reputation", None) or {}
            score = rep.get("reputation_score") if isinstance(rep, Mapping) else None
            score_val = float(score) if isinstance(score, (int, float)) else float("-inf")
            rows.append((score_val, aid))
        rows.sort(key=lambda row: (row[0], row[1]), reverse=True)
        selected = [aid for _score, aid in rows]
        if top_k is not None:
            selected = selected[:top_k]
        return selected

    # default: all active agents
    return active_ordered


def _iter_archive_papers(public_archive: Any) -> list[Any]:
    entries = getattr(public_archive, "_entries", {}) or {}
    if isinstance(entries, Mapping):
        return [paper for paper in entries.values() if paper is not None]
    if isinstance(entries, list):
        return [paper for paper in entries if paper is not None]
    return []


def _is_accepted_source(paper: Any) -> bool:
    publication_status = getattr(paper, "publication_status", None)
    if isinstance(publication_status, str) and publication_status.strip():
        return publication_status.strip().lower() == "accepted"

    status = getattr(paper, "status", None)
    if isinstance(status, str) and status.strip():
        lowered = status.strip().lower()
        # Public archive normally contains accepted papers only. If status exists,
        # keep strict accepted filtering for trust-policy consistency.
        return lowered == "accepted"

    # Public PaperMetadata entries often do not have status fields.
    return True


def _paper_sort_key(paper: Any, *, sort_by: str) -> tuple[Any, ...]:
    try:
        publication_t = int(getattr(paper, "publication_t", 0) or 0)
    except Exception:
        publication_t = 0
    try:
        citation_count = int(getattr(paper, "citation_count", 0) or 0)
    except Exception:
        citation_count = 0
    paper_id = str(getattr(paper, "paper_id", "") or "")
    if sort_by == "citations_then_recency":
        return (-citation_count, -publication_t, paper_id)
    # default
    return (-publication_t, -citation_count, paper_id)


def build_meta_source_cards(
    *,
    public_archive: Any,
    cfg: Mapping[str, Any],
) -> list[dict[str, Any]]:
    source_cfg = _safe_mapping(cfg.get("source_corpus"))
    max_papers = _as_int(
        source_cfg.get("max_papers", DEFAULT_META_RESEARCHER_CFG["source_corpus"]["max_papers"]),
        DEFAULT_META_RESEARCHER_CFG["source_corpus"]["max_papers"],
        min_value=1,
    )
    max_chars = _as_int(
        source_cfg.get(
            "max_chars_per_paper",
            DEFAULT_META_RESEARCHER_CFG["source_corpus"]["max_chars_per_paper"],
        ),
        DEFAULT_META_RESEARCHER_CFG["source_corpus"]["max_chars_per_paper"],
        min_value=200,
    )
    sort_by = (
        str(source_cfg.get("sort_by", DEFAULT_META_RESEARCHER_CFG["source_corpus"]["sort_by"]))
        .strip()
        .lower()
        or DEFAULT_META_RESEARCHER_CFG["source_corpus"]["sort_by"]
    )

    accepted = [p for p in _iter_archive_papers(public_archive) if _is_accepted_source(p)]
    accepted.sort(key=lambda p: _paper_sort_key(p, sort_by=sort_by))
    selected = accepted[:max_papers]

    cards: list[dict[str, Any]] = []
    for paper in selected:
        paper_id = str(getattr(paper, "paper_id", "") or "").strip()
        if not paper_id:
            continue
        title = str(getattr(paper, "title", "") or "Untitled").strip()
        abstract = str(getattr(paper, "abstract", "") or "").strip()
        manuscript = str(getattr(paper, "manuscript", "") or "").strip()
        excerpt_source = manuscript or abstract
        excerpt = excerpt_source[:max_chars].strip()
        cards.append(
            {
                "paper_id": paper_id,
                "title": title,
                "abstract": abstract,
                "manuscript_excerpt": excerpt,
                "publication_iteration": int(getattr(paper, "publication_t", 0) or 0),
                "citation_count": int(getattr(paper, "citation_count", 0) or 0),
            }
        )
    return cards


def _split_candidate_claims(text: str, *, max_claims: int = 4) -> list[str]:
    cleaned = " ".join(str(text or "").replace("\n", " ").split())
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    claims: list[str] = []
    seen: set[str] = set()
    for part in parts:
        claim = part.strip(" -;:")
        if len(claim) < 40:
            continue
        key = claim.lower()
        if key in seen:
            continue
        seen.add(key)
        claims.append(claim)
        if len(claims) >= max(1, int(max_claims)):
            break
    if claims:
        return claims
    return [cleaned[:240].strip()]


def _infer_evidence_type(claim_text: str) -> str:
    lower = claim_text.lower()
    if any(
        token in lower
        for token in ("p=", "confidence interval", "odds ratio", "hazard ratio", "regression")
    ):
        return "statistical"
    if any(
        token in lower
        for token in ("experiment", "benchmark", "simulation", "model", "pipeline run", "code")
    ):
        return "code_result"
    if any(token in lower for token in ("review", "literature", "prior work", "study", "trial")):
        return "literature"
    return "derived"


def _infer_support_strength(claim_text: str) -> str:
    lower = claim_text.lower()
    if any(
        token in lower
        for token in ("uncertain", "may", "might", "possibly", "suggests", "inconclusive")
    ):
        return "weak"
    if any(char.isdigit() for char in claim_text) and any(
        token in lower
        for token in ("increase", "decrease", "effect", "improve", "reduction", "association")
    ):
        return "strong"
    return "moderate"


def _claim_polarity(claim_text: str) -> int:
    lower = claim_text.lower()
    positive_tokens = ("increase", "improve", "higher", "positive", "benefit", "upregulat")
    negative_tokens = ("decrease", "worse", "lower", "negative", "harm", "downregulat", "no effect")
    pos = any(token in lower for token in positive_tokens)
    neg = any(token in lower for token in negative_tokens)
    if pos and not neg:
        return 1
    if neg and not pos:
        return -1
    return 0


def _claim_keywords(claim_text: str) -> set[str]:
    lower = claim_text.lower()
    tokens = re.findall(r"[a-z]{4,}", lower)
    stopwords = {
        "that",
        "this",
        "with",
        "from",
        "into",
        "were",
        "their",
        "these",
        "those",
        "across",
        "paper",
        "study",
        "result",
        "results",
        "findings",
        "evidence",
    }
    return {tok for tok in tokens if tok not in stopwords}


def build_meta_evidence_graph(
    *,
    source_cards: list[dict[str, Any]],
    max_claims_per_source: int = 4,
    max_contradiction_pairs: int = 24,
) -> dict[str, Any]:
    claim_nodes: list[dict[str, Any]] = []
    for card in source_cards:
        if not isinstance(card, Mapping):
            continue
        paper_id = str(card.get("paper_id", "")).strip()
        if not paper_id:
            continue
        source_text = str(card.get("abstract", "") or "").strip()
        if not source_text:
            source_text = str(card.get("manuscript_excerpt", "") or "").strip()
        claims = _split_candidate_claims(source_text, max_claims=max_claims_per_source)
        for idx, claim in enumerate(claims, start=1):
            claim_nodes.append(
                {
                    "claim_id": f"{paper_id}_c{idx}",
                    "paper_id": paper_id,
                    "claim_text": claim,
                    "evidence_type": _infer_evidence_type(claim),
                    "support_strength": _infer_support_strength(claim),
                    "polarity": _claim_polarity(claim),
                    "keywords": sorted(_claim_keywords(claim)),
                }
            )

    contradiction_candidates: list[dict[str, Any]] = []
    for i, left in enumerate(claim_nodes):
        left_keywords = set(left.get("keywords", []) or [])
        if not left_keywords:
            continue
        for right in claim_nodes[i + 1 :]:
            if left.get("paper_id") == right.get("paper_id"):
                continue
            right_keywords = set(right.get("keywords", []) or [])
            if not right_keywords:
                continue
            overlap = sorted(left_keywords.intersection(right_keywords))
            if len(overlap) < 2:
                continue
            left_polarity = int(left.get("polarity", 0) or 0)
            right_polarity = int(right.get("polarity", 0) or 0)
            contradictory = (left_polarity * right_polarity) == -1
            contradiction_candidates.append(
                {
                    "claim_id_a": str(left.get("claim_id", "")),
                    "claim_id_b": str(right.get("claim_id", "")),
                    "paper_id_a": str(left.get("paper_id", "")),
                    "paper_id_b": str(right.get("paper_id", "")),
                    "shared_keywords": overlap[:8],
                    "is_potential_contradiction": bool(contradictory),
                }
            )
            if len(contradiction_candidates) >= max(1, int(max_contradiction_pairs)):
                break
        if len(contradiction_candidates) >= max(1, int(max_contradiction_pairs)):
            break

    return {
        "source_paper_ids": [
            str(card.get("paper_id", "")).strip()
            for card in source_cards
            if str(card.get("paper_id", "")).strip()
        ],
        "claim_nodes": claim_nodes,
        "contradiction_candidates": contradiction_candidates,
        "summary": {
            "num_source_cards": len(source_cards),
            "num_claim_nodes": len(claim_nodes),
            "num_contradiction_candidates": len(contradiction_candidates),
        },
    }


def build_meta_phase_context_by_agent(
    *,
    agent_ids: list[str],
    source_cards: list[dict[str, Any]],
    iteration: int,
    cfg: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    output_cfg = _safe_mapping(cfg.get("output"))
    quality_cfg = _safe_mapping(cfg.get("quality_contract"))
    validation_gate_cfg = _safe_mapping(cfg.get("validation_gate"))
    source_ids = [
        str(card.get("paper_id", "")).strip()
        for card in source_cards
        if str(card.get("paper_id", "")).strip()
    ]
    evidence_graph = build_meta_evidence_graph(source_cards=source_cards)
    shared_context = {
        "trigger_iteration": int(iteration),
        "source_cards": [dict(card) for card in source_cards],
        "source_paper_ids": source_ids,
        "evidence_graph": evidence_graph,
        "requirements": {
            "require_contradictions_section": bool(
                output_cfg.get("require_contradictions_section", True)
            ),
            "require_archive_citations": bool(output_cfg.get("require_archive_citations", True)),
            "min_archive_citations": _as_int(
                output_cfg.get("min_archive_citations", 3), 3, min_value=0
            ),
        },
        "quality_contract": {
            "require_claim_evidence_table": bool(
                quality_cfg.get("require_claim_evidence_table", True)
            ),
            "require_support_checks_section": bool(
                quality_cfg.get("require_support_checks_section", True)
            ),
            "require_contradiction_resolution_section": bool(
                quality_cfg.get("require_contradiction_resolution_section", True)
            ),
            "require_bounded_final_claims_section": bool(
                quality_cfg.get("require_bounded_final_claims_section", True)
            ),
            "require_executable_claim_checks_section": bool(
                quality_cfg.get("require_executable_claim_checks_section", True)
            ),
            "require_reproducibility_notes_section": bool(
                quality_cfg.get("require_reproducibility_notes_section", True)
            ),
            "min_supported_major_claims": _as_int(
                quality_cfg.get("min_supported_major_claims", 3), 3, min_value=0
            ),
            "min_code_verified_major_claims": _as_int(
                quality_cfg.get("min_code_verified_major_claims", 1), 1, min_value=0
            ),
            "max_unsupported_major_claims": _as_int(
                quality_cfg.get("max_unsupported_major_claims", 0), 0, min_value=0
            ),
            "max_unresolved_uncertainty_claims": _as_int(
                quality_cfg.get("max_unresolved_uncertainty_claims", 0), 0, min_value=0
            ),
            "max_words": _as_int(quality_cfg.get("max_words", 950), 950, min_value=100),
        },
        "validation_gate": {
            "enabled": bool(validation_gate_cfg.get("enabled", True)),
            "max_revision_attempts": _as_int(
                validation_gate_cfg.get(
                    "max_revision_attempts",
                    1 if bool(validation_gate_cfg.get("allow_single_revision_pass", True)) else 0,
                ),
                1 if bool(validation_gate_cfg.get("allow_single_revision_pass", True)) else 0,
                min_value=0,
            ),
            "allow_single_revision_pass": bool(
                validation_gate_cfg.get("allow_single_revision_pass", True)
            ),
            "fail_on_unresolved_nan_claims": bool(
                validation_gate_cfg.get("fail_on_unresolved_nan_claims", True)
            ),
            "fail_on_missing_evidence_mapping": bool(
                validation_gate_cfg.get("fail_on_missing_evidence_mapping", True)
            ),
        },
    }
    by_agent: dict[str, dict[str, Any]] = {}
    for aid in agent_ids:
        aid_clean = str(aid).strip()
        if not aid_clean:
            continue
        validation_payload = dict(shared_context["requirements"])
        validation_payload["quality_contract"] = dict(shared_context["quality_contract"])
        validation_payload["validation_gate"] = dict(shared_context["validation_gate"])
        by_agent[aid_clean] = {
            "meta_research_mode": True,
            "meta_research_context": dict(shared_context),
            "meta_validation": validation_payload,
        }
    return by_agent


def init_meta_researcher_telemetry(
    *,
    iteration: int,
    cfg: Mapping[str, Any],
    trigger: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "enabled": bool(cfg.get("enabled", False)),
        "iteration": int(iteration),
        "trigger": {
            "iteration": int(trigger.get("iteration", iteration) or iteration),
            "every_n_iterations": int(
                trigger.get("every_n_iterations", cfg.get("every_n_iterations", 1)) or 1
            ),
            "start_iteration": int(
                trigger.get("start_iteration", cfg.get("start_iteration", 1)) or 1
            ),
            "triggered": bool(trigger.get("triggered", False)),
            "reason": str(trigger.get("reason", "")),
        },
        "selected_agent_ids": [],
        "selected_source_paper_ids": [],
        "generated_meta_paper_ids": [],
        "agent_failures": {},
        "meta_quality_reports": [],
        "meta_quality_summary": {
            "passed": 0,
            "failed": 0,
            "revised_passed": 0,
        },
        "num_source_papers": 0,
        "num_meta_papers": 0,
        "skipped_reason": None,
        "meta_claim_checks_artifacts": [],
        "citation_lint_artifacts": [],
    }
