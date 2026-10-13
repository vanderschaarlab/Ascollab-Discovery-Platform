import dataclasses
import json
import logging
import pickle
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from omegaconf import DictConfig, OmegaConf

from stores import state as _global_state
from stores.agent_registry import AgentRegistry
from stores.paper_archive import PaperArchive
from stores.private_paper_archive import PrivatePaperArchive
from system.citation_tags import validate_citation_tags
from system.schema import AgentNetworkMetadata
from system.snapshot_paths import (
    resolve_iteration_snapshot_dir,
    resolve_snapshot_run_name,
)
from system.trackers import SessionStatsRegistry

log = logging.getLogger(__name__)


def _redact_api_keys(obj):
    """Recursively replace any `api_key` entries with the sentinel value."""
    if isinstance(obj, Mapping):
        return {
            k: "REDACTED" if str(k) == "api_key" else _redact_api_keys(v) for k, v in obj.items()
        }
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        return [_redact_api_keys(v) for v in obj]
    return obj


def _to_plain(obj):
    """Convert nested objects (dataclasses/pydantic/etc.) to JSON-serializable primitives."""
    # Primitives
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # Mappings
    if isinstance(obj, Mapping):
        return {str(k): _to_plain(v) for k, v in obj.items()}

    # Sequences (but not str/bytes handled above)
    if isinstance(obj, Sequence) and not isinstance(obj, (bytes, bytearray)):
        return [_to_plain(v) for v in obj]

    # Dataclasses
    if dataclasses.is_dataclass(obj):
        return _to_plain(dataclasses.asdict(obj))

    # Pydantic v2
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        try:
            return _to_plain(obj.model_dump())
        except Exception:
            pass

    # Pydantic v1
    if hasattr(obj, "dict") and callable(obj.dict):
        try:
            return _to_plain(obj.dict())
        except Exception:
            pass

    # Generic objects: fall back to __dict__ (exclude privates/callables)
    if hasattr(obj, "__dict__"):
        d = {k: v for k, v in vars(obj).items() if not k.startswith("_") and not callable(v)}
        return _to_plain(d)

    # Last resort: string repr
    return repr(obj)


def _empty_citation_counts() -> dict[str, int]:
    return {
        "total": 0,
        "valid": 0,
        "invalid": 0,
        "archive": 0,
        "arxiv": 0,
        "code": 0,
        "plot": 0,
    }


def _accumulate_citation_counts(dst: dict[str, int], src: dict[str, Any]) -> None:
    for key in ("total", "valid", "invalid", "archive", "arxiv", "code", "plot"):
        dst[key] = int(dst.get(key, 0)) + int((src or {}).get(key, 0))


def _paper_status_lookup(private_archive: PrivatePaperArchive) -> dict[str, str]:
    lookup: dict[str, str] = {}
    entries = getattr(private_archive, "_entries", {}) or {}
    if isinstance(entries, dict):
        for _agent_id, papers in entries.items():
            for paper in papers or []:
                pid = str(getattr(paper, "paper_id", "") or "").strip()
                if not pid:
                    continue
                lookup[pid] = str(getattr(paper, "status", "") or "").strip().lower()
    return lookup


def _paper_citation_compliance_lookup(private_archive: PrivatePaperArchive) -> dict[str, bool]:
    lookup: dict[str, bool] = {}
    entries = getattr(private_archive, "_entries", {}) or {}
    if isinstance(entries, dict):
        for _agent_id, papers in entries.items():
            for paper in papers or []:
                pid = str(getattr(paper, "paper_id", "") or "").strip()
                if not pid:
                    continue
                compliant = getattr(paper, "citation_compliant", True)
                lookup[pid] = bool(True if compliant is None else compliant)
    return lookup


def _flatten_private_archive_papers(
    private_archive_data: Mapping[str, Any],
) -> list[dict[str, Any]]:
    papers_raw = list((private_archive_data or {}).get("papers", []) or [])
    flat: list[dict[str, Any]] = []
    for item in papers_raw:
        if isinstance(item, list):
            for paper in item:
                if isinstance(paper, Mapping):
                    flat.append(dict(paper))
        elif isinstance(item, Mapping):
            flat.append(dict(item))
    return flat


def _is_timeout_paper_record(paper: Mapping[str, Any]) -> bool:
    paper_id = str((paper or {}).get("paper_id", "") or "").strip()
    manuscript = str((paper or {}).get("manuscript", "") or "").strip()
    return paper_id.endswith("_timeout") or manuscript == "TIMEOUT"


def _normalize_paper_kind(paper: Mapping[str, Any]) -> str:
    raw = str((paper or {}).get("paper_kind", "research") or "research").strip().lower()
    return "meta_research" if raw == "meta_research" else "research"


def _ranker_pool_sort_key(paper: Mapping[str, Any]) -> tuple[int, str]:
    try:
        publication_t = int((paper or {}).get("publication_t", 0) or 0)
    except Exception:
        publication_t = 0
    paper_id = str((paper or {}).get("paper_id", "") or "").strip()
    return (publication_t, paper_id)


def _build_ranker_pool_records(
    *,
    private_archive_data: Mapping[str, Any],
    include_rejected: bool,
    exclude_timeout: bool,
) -> dict[str, list[dict[str, Any]]]:
    flat_papers = _flatten_private_archive_papers(private_archive_data)
    filtered: list[dict[str, Any]] = []
    for paper in flat_papers:
        status = str((paper or {}).get("status", "") or "").strip().lower()
        if not include_rejected and status != "accepted":
            continue
        if exclude_timeout and _is_timeout_paper_record(paper):
            continue
        filtered.append(dict(paper))

    filtered.sort(key=_ranker_pool_sort_key)
    all_records = filtered
    standard_records = [p for p in all_records if _normalize_paper_kind(p) == "research"]
    meta_records = [p for p in all_records if _normalize_paper_kind(p) == "meta_research"]
    return {
        "all": all_records,
        "standard": standard_records,
        "meta": meta_records,
    }


def _write_jsonl(path, rows: list[Mapping[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_to_plain(dict(row)), ensure_ascii=False, default=str) + "\n")


def _lookup_successful_code_executions(
    *,
    iteration: int,
    primary_agent_id: str,
    session_stats_registry: SessionStatsRegistry,
) -> int:
    stats = None
    if hasattr(session_stats_registry, "get_research_log_for_agent"):
        try:
            stats = session_stats_registry.get_research_log_for_agent(iteration, primary_agent_id)
        except Exception:
            stats = None
    if stats is None:
        reg = getattr(session_stats_registry, "registry", {}) or {}
        iter_data = reg.get(iteration) or reg.get(str(iteration)) or {}
        research_logs = iter_data.get("research", {}) if isinstance(iter_data, dict) else {}
        if isinstance(research_logs, dict):
            stats = (
                research_logs.get(f"{primary_agent_id}_t{int(iteration):02d}")
                or research_logs.get(f"{primary_agent_id}_t{iteration}")
                or research_logs.get(primary_agent_id)
            )
    if stats is None:
        return 0
    stats_obj = getattr(stats, "stats_logs", stats)
    try:
        return int(getattr(stats_obj, "code_executions_successful", 0) or 0)
    except Exception:
        return 0


def _build_citation_snapshot_metadata(
    *,
    iteration: int,
    paper_metadata: list[Any] | None,
    private_archive: PrivatePaperArchive,
    session_stats_registry: SessionStatsRegistry,
    require_research_tags: bool,
    require_review_tags: bool,
) -> dict[str, Any]:
    status_lookup = _paper_status_lookup(private_archive)
    compliance_lookup = _paper_citation_compliance_lookup(private_archive)
    research_entries: list[dict[str, Any]] = []
    review_entries: list[dict[str, Any]] = []
    research_totals = _empty_citation_counts()
    review_totals = _empty_citation_counts()
    research_code_evidence_totals: dict[str, int] = {
        "papers_with_code_tags": 0,
        "papers_with_unbacked_code_tags": 0,
        "total_code_tags": 0,
    }
    citation_integrity_totals: dict[str, Any] = {
        "accepted_noncompliant_count": 0,
        "accepted_noncompliant_paper_ids": [],
        "archive_edges_total": 0,
        "archive_edges_to_noncompliant_count": 0,
        "papers_citing_noncompliant_count": 0,
        "papers_citing_noncompliant_ids": [],
        "noncompliant_cited_paper_ids": [],
        "alerts": [],
    }

    local_publication_status: dict[str, str] = {}
    local_citation_compliant: dict[str, bool] = {}
    for paper in paper_metadata or []:
        paper_id = str(getattr(paper, "paper_id", "") or "").strip()
        if not paper_id:
            continue
        publication_status = (
            str(getattr(paper, "publication_status", "") or getattr(paper, "status", ""))
            .strip()
            .lower()
        )
        if publication_status:
            local_publication_status[paper_id] = publication_status
        compliant = getattr(paper, "citation_compliant", True)
        local_citation_compliant[paper_id] = bool(True if compliant is None else compliant)

    for paper in paper_metadata or []:
        paper_id = str(getattr(paper, "paper_id", "") or "").strip()
        primary_agent_id = str(getattr(paper, "primary_agent_id", "") or "").strip()
        manuscript = str(getattr(paper, "manuscript", "") or "")
        code_script = str(getattr(paper, "code_script", "") or "")
        successful_code_execs = _lookup_successful_code_executions(
            iteration=iteration,
            primary_agent_id=primary_agent_id,
            session_stats_registry=session_stats_registry,
        )
        verdict = validate_citation_tags(
            manuscript,
            require_at_least_one=require_research_tags,
            known_archive_status_by_id=status_lookup,
            require_accepted_archive_ids=True,
            require_backed_code_tags=require_research_tags,
            successful_code_executions_current_iteration=successful_code_execs,
            code_script=code_script,
        )
        counts = dict(verdict.get("counts", {}))
        _accumulate_citation_counts(research_totals, counts)
        code_evidence = dict(verdict.get("code_evidence", {}))
        code_tag_count = int(code_evidence.get("code_tag_count", 0) or 0)
        research_code_evidence_totals["total_code_tags"] += code_tag_count
        if code_tag_count > 0:
            research_code_evidence_totals["papers_with_code_tags"] += 1
            if not bool(code_evidence.get("is_backed", False)):
                research_code_evidence_totals["papers_with_unbacked_code_tags"] += 1
        publication_status = local_publication_status.get(paper_id) or status_lookup.get(
            paper_id, ""
        )
        publication_status = str(publication_status or "").strip().lower()
        research_validation_status = str(
            getattr(paper, "research_validation_status", "") or ""
        ).strip()
        paper_errors = [str(e) for e in getattr(paper, "citation_errors", []) or []]
        if not paper_errors:
            paper_errors = [str(e) for e in verdict.get("errors", [])]
        paper_compliant = local_citation_compliant.get(
            paper_id, bool(verdict.get("is_valid", False))
        )
        is_citable_archive_source = bool(publication_status == "accepted")
        citable_with_warnings = bool(is_citable_archive_source and not paper_compliant)
        if is_citable_archive_source and not paper_compliant:
            citation_integrity_totals["accepted_noncompliant_count"] += 1
            if paper_id not in citation_integrity_totals["accepted_noncompliant_paper_ids"]:
                citation_integrity_totals["accepted_noncompliant_paper_ids"].append(paper_id)

        invalid_archive_refs: list[str] = [
            str(pid).strip()
            for pid in (getattr(paper, "invalid_archive_refs", []) or [])
            if str(pid).strip()
        ]
        if not invalid_archive_refs:
            for cited_pid in verdict.get("valid_archive_ids", []):
                pid = str(cited_pid).strip()
                if not pid:
                    continue
                cited_status = str(status_lookup.get(pid, "") or "").strip().lower()
                if cited_status and cited_status != "accepted" and pid not in invalid_archive_refs:
                    invalid_archive_refs.append(pid)

        cited_noncompliant_archive_ids: list[str] = []
        for cited_pid in verdict.get("valid_archive_ids", []):
            pid = str(cited_pid).strip()
            if not pid:
                continue
            citation_integrity_totals["archive_edges_total"] += 1
            cited_status = (
                str(local_publication_status.get(pid) or status_lookup.get(pid, "")).strip().lower()
            )
            cited_compliant = bool(
                local_citation_compliant.get(pid, compliance_lookup.get(pid, True))
            )
            if cited_status == "accepted" and not cited_compliant:
                citation_integrity_totals["archive_edges_to_noncompliant_count"] += 1
                if pid not in cited_noncompliant_archive_ids:
                    cited_noncompliant_archive_ids.append(pid)
                if pid not in citation_integrity_totals["noncompliant_cited_paper_ids"]:
                    citation_integrity_totals["noncompliant_cited_paper_ids"].append(pid)
        if cited_noncompliant_archive_ids:
            citation_integrity_totals["papers_citing_noncompliant_count"] += 1
            if paper_id not in citation_integrity_totals["papers_citing_noncompliant_ids"]:
                citation_integrity_totals["papers_citing_noncompliant_ids"].append(paper_id)

        research_entries.append(
            {
                "paper_id": paper_id,
                "primary_agent_id": primary_agent_id,
                "status": str(getattr(paper, "status", "") or ""),
                "publication_status": publication_status,
                "research_validation_status": research_validation_status,
                "citation_compliant": bool(paper_compliant),
                "is_citable_archive_source": is_citable_archive_source,
                "citable_with_warnings": citable_with_warnings,
                "is_valid": bool(verdict.get("is_valid", False)),
                "counts": counts,
                "errors": [str(e) for e in verdict.get("errors", [])],
                "citation_errors": paper_errors,
                "citation_error_count": len(paper_errors),
                "invalid_archive_refs": invalid_archive_refs,
                "cited_noncompliant_archive_ids": cited_noncompliant_archive_ids,
                "code_evidence": {
                    "code_tag_count": code_tag_count,
                    "successful_code_executions_current_iteration": int(successful_code_execs),
                    "code_script_chars": len(code_script.strip()),
                    "is_backed": bool(code_evidence.get("is_backed", False)),
                    "errors": [str(e) for e in code_evidence.get("errors", [])],
                },
                "tags": [
                    {
                        "type": str(tag.get("type", "")),
                        "raw": str(tag.get("raw", "")),
                        "value": str(tag.get("value", "")),
                        "valid": bool(tag.get("valid", False)),
                        "error": str(tag.get("error", "")),
                    }
                    for tag in verdict.get("tags", [])
                ],
            }
        )

    if citation_integrity_totals["accepted_noncompliant_count"] > 0:
        citation_integrity_totals["alerts"].append(
            {
                "severity": "HIGH",
                "message": (
                    f"{citation_integrity_totals['accepted_noncompliant_count']} accepted paper(s) "
                    "are citation-noncompliant but remain citable under trust policy"
                ),
                "paper_ids": citation_integrity_totals["accepted_noncompliant_paper_ids"][:50],
            }
        )
    if citation_integrity_totals["archive_edges_to_noncompliant_count"] > 0:
        citation_integrity_totals["alerts"].append(
            {
                "severity": "HIGH",
                "message": (
                    f"{citation_integrity_totals['archive_edges_to_noncompliant_count']} archive citation edge(s) "
                    "target accepted-but-noncompliant papers"
                ),
                "paper_ids": citation_integrity_totals["papers_citing_noncompliant_ids"][:50],
                "cited_paper_ids": citation_integrity_totals["noncompliant_cited_paper_ids"][:50],
            }
        )

    reviews_by_paper = getattr(private_archive, "reviews_by_paper", {}) or {}
    if isinstance(reviews_by_paper, dict):
        for paper_id, entries in reviews_by_paper.items():
            for entry in entries or []:
                if int(entry.get("iteration", -1)) != int(iteration):
                    continue
                review_text = str(entry.get("review_text", "") or "")
                verdict = validate_citation_tags(
                    review_text,
                    require_at_least_one=require_review_tags,
                )
                counts = dict(verdict.get("counts", {}))
                _accumulate_citation_counts(review_totals, counts)
                review_entries.append(
                    {
                        "paper_id": str(paper_id),
                        "reviewer_id": str(entry.get("reviewer_id", "") or ""),
                        "is_valid": bool(verdict.get("is_valid", False)),
                        "counts": counts,
                        "errors": [str(e) for e in verdict.get("errors", [])],
                        "tags": [
                            {
                                "type": str(tag.get("type", "")),
                                "raw": str(tag.get("raw", "")),
                                "value": str(tag.get("value", "")),
                                "valid": bool(tag.get("valid", False)),
                                "error": str(tag.get("error", "")),
                            }
                            for tag in verdict.get("tags", [])
                        ],
                    }
                )

    return {
        "schema": {
            "archive": "[Archive: paper_id]",
            "arxiv": "[Arxiv: paper title | author | year]",
            "code": "[Code: brief description <=5 words]",
        },
        "trust_policy": {
            "archive_citable_if": "publication_status == accepted",
            "accepted_noncompliant_are_citable": True,
            "note": (
                "Accepted papers remain citable even when citation_compliant=false. "
                "Integrity alerts and edge-level diagnostics expose associated risk."
            ),
        },
        "research": {
            "strict_required": bool(require_research_tags),
            "totals": research_totals,
            "code_evidence_totals": research_code_evidence_totals,
            "citation_integrity": citation_integrity_totals,
            "entries": research_entries,
        },
        "review": {
            "strict_required": bool(require_review_tags),
            "totals": review_totals,
            "entries": review_entries,
        },
    }


def create_snapshot(
    iteration: int,
    agent_registry: AgentRegistry,
    archive: PaperArchive,
    private_archive: PrivatePaperArchive,
    network_metadata: AgentNetworkMetadata,
    session_stats_registry: SessionStatsRegistry,
    exp_config: DictConfig,
    *,
    run_name: str | None = None,
    base_snapshots_dir: str = "snapshots",
    metrics_by_paper: dict[str, dict[str, Any]] | None = None,
    paper_metadata: list[Any] | None = None,
    meta_researcher_metadata: dict[str, Any] | None = None,
    metacognition_feedback_metadata: dict[str, Any] | None = None,
    fixed_workflow_metadata: dict[str, Any] | None = None,
    social_memory_pruning_metadata: dict[str, Any] | None = None,
) -> str:
    """Persist a full snapshot of the current run state for auditing.

    Args:
        iteration: Current iteration index.
        agent_registry: Registry containing all agent profiles.
        archive: Public paper archive.
        private_archive: Private archive including reviews.
        network_metadata: Aggregate agent/paper network metadata.
        session_stats_registry: Registry of per-session logs.
        exp_config: Experiment configuration.
        run_name: Optional run name (defaults to config name).
        base_snapshots_dir: Base directory for storing snapshots.

    Returns:
        Path to the snapshot directory as a string.
    """
    exp_dir = resolve_snapshot_run_name(run_name=run_name, exp_config=exp_config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_dir = resolve_iteration_snapshot_dir(
        run_name=exp_dir,
        iteration=iteration,
        base_snapshots_dir=base_snapshots_dir,
        create=True,
    )

    if metrics_by_paper:
        metrics_path = snapshot_dir / "metrics.jsonl"
        paper_lookup = {
            getattr(p, "paper_id", None): p
            for p in (paper_metadata or [])
            if getattr(p, "paper_id", None)
        }
        with metrics_path.open("a", encoding="utf-8") as mf:
            for pid, metrics in metrics_by_paper.items():
                paper = paper_lookup.get(pid)
                mf.write(
                    json.dumps(
                        {
                            "iteration": iteration,
                            "paper_id": pid,
                            "primary_agent_id": getattr(paper, "primary_agent_id", None)
                            if paper
                            else None,
                            "paper_title": getattr(paper, "title", None) if paper else None,
                            "metrics": metrics,
                        },
                        ensure_ascii=False,
                        default=str,
                    )
                    + "\n"
                )

    meta_researcher_file_written = False
    if isinstance(meta_researcher_metadata, Mapping):
        write_meta_file = bool(meta_researcher_metadata.get("write_iteration_file", True))
        if write_meta_file:
            meta_researcher_path = snapshot_dir / "meta_researcher.json"
            with meta_researcher_path.open("w", encoding="utf-8") as mf:
                json.dump(
                    _to_plain(dict(meta_researcher_metadata)),
                    mf,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
            meta_researcher_file_written = True

    metacognition_feedback_file_written = False
    if isinstance(metacognition_feedback_metadata, Mapping) and bool(
        metacognition_feedback_metadata.get("write_artifacts", True)
    ):
        feedback_path = snapshot_dir / "metacognition_feedback.json"
        with feedback_path.open("w", encoding="utf-8") as ff:
            json.dump(
                _to_plain(dict(metacognition_feedback_metadata)),
                ff,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        metacognition_feedback_file_written = True

    social_memory_pruning_written = False
    if isinstance(social_memory_pruning_metadata, Mapping):
        social_memory_pruning_path = snapshot_dir / "social_memory_pruning.json"
        with social_memory_pruning_path.open("w", encoding="utf-8") as sf:
            json.dump(
                _to_plain(dict(social_memory_pruning_metadata)),
                sf,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        social_memory_pruning_written = True

    fixed_workflow_assignments_written = False
    fixed_workflow_compliance_written = False
    if isinstance(fixed_workflow_metadata, Mapping) and bool(
        fixed_workflow_metadata.get("write_artifacts", True)
    ):
        assignments_payload = {
            key: value
            for key, value in dict(fixed_workflow_metadata).items()
            if key != "compliance"
        }
        fixed_assignments_path = snapshot_dir / "fixed_workflow_assignments.json"
        with fixed_assignments_path.open("w", encoding="utf-8") as ff:
            json.dump(
                _to_plain(assignments_payload),
                ff,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        fixed_workflow_assignments_written = True

        compliance_rows = fixed_workflow_metadata.get("compliance", []) or []
        if isinstance(compliance_rows, list):
            fixed_compliance_path = snapshot_dir / "fixed_workflow_compliance.jsonl"
            with fixed_compliance_path.open("w", encoding="utf-8") as cf:
                for row in compliance_rows:
                    if isinstance(row, Mapping):
                        cf.write(
                            json.dumps(_to_plain(dict(row)), ensure_ascii=False, default=str) + "\n"
                        )
            fixed_workflow_compliance_written = True

    # --- dump per-session timelines for full auditability ---
    try:
        timeline_dir = snapshot_dir / "timelines"
        timeline_dir.mkdir(parents=True, exist_ok=True)

        # Expect SessionStatsRegistry from system.trackers
        iter_logs = session_stats_registry.registry.get(iteration, {})

        for phase, sessions in iter_logs.items():  # e.g. "research", "review"
            for session_id, session_log in sessions.items():
                timeline = getattr(session_log, "timeline", None)
                if not timeline:
                    continue

                if phase == "review":
                    out_path = timeline_dir / f"{session_id}_timeline.jsonl"
                else:
                    out_path = timeline_dir / f"{phase}_{session_id}_timeline.jsonl"
                with out_path.open("w", encoding="utf-8") as f:
                    for ev in timeline:
                        # default=str just in case, but we only put primitives in payloads above
                        f.write(json.dumps(ev, ensure_ascii=False, default=str) + "\n")
    except Exception as e:
        # best-effort: snapshot should not fail just because of logging
        log.warning(f"[snapshot] failed to write timelines for iteration {iteration}: {e}")

    extra_snapshot_files: list[str] = []

    try:
        agent_registry_data = {
            "agents": agent_registry.get_all_agents(),
            "max_citations_seen": getattr(agent_registry, "_max_citations_seen", 0),
            "with_citation_boost": getattr(agent_registry, "with_citation_boost", False),
            "w_sim": getattr(agent_registry, "w_sim", 0.8),
            "w_cit": getattr(agent_registry, "w_cit", 0.2),
            "max_k": getattr(agent_registry, "max_k", 5),
        }
        agent_registry_path = snapshot_dir / "agent_registry_data.pkl"
        with open(agent_registry_path, "wb") as f:
            pickle.dump(agent_registry_data, f)

        archive_data = {
            "papers": [],
            "with_citation_boost": getattr(archive, "with_citation_boost", False),
            "with_recency_boost": getattr(archive, "with_recency_boost", False),
            "w_citation": getattr(archive, "w_citation", 0.1),
            "w_recency": getattr(archive, "w_recency", 0.1),
            "max_k": getattr(archive, "max_k", 5),
        }
        if hasattr(archive, "_entries") and archive._entries:
            archive_data["papers"] = [_to_plain(v) for v in archive._entries.values()]
        archive_path = snapshot_dir / "archive_data.pkl"
        with open(archive_path, "wb") as f:
            pickle.dump(archive_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(snapshot_dir / "archive_data.json", "w", encoding="utf-8") as jf:
            json.dump(archive_data, jf, ensure_ascii=False, indent=2)

        private_archive_data = {
            "papers": [],
            "reviews_by_paper": {},
            "review_assignments": {},
            "with_citation_boost": getattr(private_archive, "with_citation_boost", False),
            "with_recency_boost": getattr(private_archive, "with_recency_boost", False),
            "w_citation": getattr(private_archive, "w_citation", 0.1),
            "w_recency": getattr(private_archive, "w_recency", 0.1),
            "max_k": getattr(private_archive, "max_k", 5),
        }

        # Existing paper serialisation
        if hasattr(private_archive, "_entries") and private_archive._entries:
            private_archive_data["papers"] = [
                _to_plain(v) for v in private_archive._entries.values()
            ]

        # serialise stored reviews (if present)
        if hasattr(private_archive, "reviews_by_paper"):
            try:
                private_archive_data["reviews_by_paper"] = {
                    paper_id: [dict(entry) for entry in entries]
                    for paper_id, entries in private_archive.reviews_by_paper.items()
                }
            except Exception as e:
                log.warning(f"[snapshot] failed to serialise reviews_by_paper: {e}")

        if hasattr(private_archive, "review_assignments"):
            try:
                private_archive_data["review_assignments"] = {
                    str(record_id): dict(record or {})
                    for record_id, record in private_archive.review_assignments.items()
                }
            except Exception as e:
                log.warning(f"[snapshot] failed to serialise review_assignments: {e}")

        private_archive_path = snapshot_dir / "private_archive_data.pkl"
        with open(private_archive_path, "wb") as f:
            pickle.dump(private_archive_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(snapshot_dir / "private_archive_data.json", "w", encoding="utf-8") as jf:
            json.dump(private_archive_data, jf, ensure_ascii=False, indent=2)

        if isinstance(meta_researcher_metadata, Mapping):
            write_meta_quality_report = bool(
                meta_researcher_metadata.get("write_meta_quality_report", True)
            )
            write_evidence_map = bool(meta_researcher_metadata.get("write_evidence_map", True))
            write_meta_claim_checks = bool(
                meta_researcher_metadata.get("write_meta_claim_checks", True)
            )
            write_citation_lint = bool(meta_researcher_metadata.get("write_citation_lint", True))

            if write_meta_quality_report:
                meta_quality_reports = (
                    meta_researcher_metadata.get("meta_quality_reports", []) or []
                )
                if isinstance(meta_quality_reports, list) and meta_quality_reports:
                    with open(
                        snapshot_dir / "meta_quality_report.json", "w", encoding="utf-8"
                    ) as f:
                        json.dump(
                            _to_plain(meta_quality_reports),
                            f,
                            ensure_ascii=False,
                            indent=2,
                            default=str,
                        )
                    extra_snapshot_files.append("meta_quality_report.json")

            if write_evidence_map:
                evidence_graph = meta_researcher_metadata.get("evidence_graph", None)
                if isinstance(evidence_graph, Mapping):
                    with open(snapshot_dir / "evidence_map.json", "w", encoding="utf-8") as f:
                        json.dump(
                            _to_plain(dict(evidence_graph)),
                            f,
                            ensure_ascii=False,
                            indent=2,
                            default=str,
                        )
                    extra_snapshot_files.append("evidence_map.json")

            if write_meta_claim_checks:
                meta_claim_checks = (
                    meta_researcher_metadata.get("meta_claim_checks_artifacts", []) or []
                )
                if isinstance(meta_claim_checks, list) and meta_claim_checks:
                    with open(snapshot_dir / "meta_claim_checks.json", "w", encoding="utf-8") as f:
                        json.dump(
                            _to_plain(meta_claim_checks),
                            f,
                            ensure_ascii=False,
                            indent=2,
                            default=str,
                        )
                    extra_snapshot_files.append("meta_claim_checks.json")

            if write_citation_lint:
                citation_lint = meta_researcher_metadata.get("citation_lint_artifacts", []) or []
                if isinstance(citation_lint, list) and citation_lint:
                    with open(snapshot_dir / "citation_lint.json", "w", encoding="utf-8") as f:
                        json.dump(
                            _to_plain(citation_lint), f, ensure_ascii=False, indent=2, default=str
                        )
                    extra_snapshot_files.append("citation_lint.json")

        write_ranker_pools = True
        if isinstance(meta_researcher_metadata, Mapping):
            write_ranker_pools = bool(meta_researcher_metadata.get("write_ranker_pools", True))
        if write_ranker_pools:
            include_rejected = False
            exclude_timeout = True
            pools = _build_ranker_pool_records(
                private_archive_data=private_archive_data,
                include_rejected=include_rejected,
                exclude_timeout=exclude_timeout,
            )
            ranker_pool_dir = snapshot_dir / "ranker_pools"
            ranker_pool_dir.mkdir(parents=True, exist_ok=True)

            papers_all_path = ranker_pool_dir / "papers_all.jsonl"
            papers_standard_path = ranker_pool_dir / "papers_standard.jsonl"
            papers_meta_path = ranker_pool_dir / "papers_meta.jsonl"
            manifest_path = ranker_pool_dir / "manifest.json"

            _write_jsonl(papers_all_path, pools["all"])
            _write_jsonl(papers_standard_path, pools["standard"])
            _write_jsonl(papers_meta_path, pools["meta"])
            with open(manifest_path, "w", encoding="utf-8") as mf:
                json.dump(
                    {
                        "run_name": exp_dir,
                        "iteration": int(iteration),
                        "generated_at_utc": datetime.utcnow().replace(microsecond=0).isoformat()
                        + "Z",
                        "counts": {
                            "all": len(pools["all"]),
                            "standard": len(pools["standard"]),
                            "meta": len(pools["meta"]),
                        },
                        "filters": {
                            "include_rejected": bool(include_rejected),
                            "exclude_timeout": bool(exclude_timeout),
                        },
                        "source_files": ["private_archive_data.json"],
                        "schema_version": 1,
                    },
                    mf,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )

            extra_snapshot_files.extend(
                [
                    "ranker_pools/",
                    "ranker_pools/papers_all.jsonl",
                    "ranker_pools/papers_standard.jsonl",
                    "ranker_pools/papers_meta.jsonl",
                    "ranker_pools/manifest.json",
                ]
            )

        notepad_data: dict[str, Any] = {
            "entries_by_agent": {},
            "notepad_file_paths": {},
            "config": {},
        }
        try:
            store = _global_state.notepad_store()
            if store is not None:
                candidate = store.snapshot_data()
                if isinstance(candidate, Mapping):
                    notepad_data = dict(candidate)
        except Exception:
            pass
        notepad_path_pkl = snapshot_dir / "notepad_data.pkl"
        with open(notepad_path_pkl, "wb") as f:
            pickle.dump(notepad_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(snapshot_dir / "notepad_data.json", "w", encoding="utf-8") as jf:
            json.dump(notepad_data, jf, ensure_ascii=False, indent=2, default=str)

        session_stats_data = {"registry": {}}
        if hasattr(session_stats_registry, "registry"):
            # Not `iteration`: the snapshot's own iteration is used after this loop.
            for reg_iteration, iteration_data in session_stats_registry.registry.items():
                session_stats_data["registry"][reg_iteration] = {}
                for log_type, agents_data in iteration_data.items():
                    session_stats_data["registry"][reg_iteration][log_type] = {}
                    for agent_id, agent_log in agents_data.items():
                        log_dict = {}
                        session_agent_id = None
                        if hasattr(agent_log, "session_agent_id") and hasattr(
                            agent_log, "stats_logs"
                        ):
                            session_agent_id = getattr(agent_log, "session_agent_id", None)
                            stats_log = getattr(agent_log, "stats_logs", agent_log)
                        else:
                            stats_log = agent_log
                        for field_name, field_value in stats_log.__dict__.items():
                            # convert defaultdicts to plain dicts for pickle/json friendliness
                            if field_name in ("tool_calls", "tool_inputs") and isinstance(
                                field_value, Mapping
                            ):
                                log_dict[field_name] = dict(
                                    field_value
                                )  # normalize defaultdict -> dict
                            else:
                                log_dict[field_name] = field_value
                        log_dict["log_type"] = type(stats_log).__name__
                        if session_agent_id is not None:
                            log_dict["session_agent_id"] = session_agent_id
                        session_stats_data["registry"][reg_iteration][log_type][agent_id] = log_dict

        session_stats_path = snapshot_dir / "session_stats_data.pkl"
        with open(session_stats_path, "wb") as f:
            pickle.dump(session_stats_data, f)
            total_logs = 0
            research_logs = 0
            review_logs = 0
            for iteration_data in session_stats_data["registry"].values():
                for log_type, agents_data in iteration_data.items():
                    count = len(agents_data)
                    total_logs += count
                    if log_type == "research":
                        research_logs += count
                    elif log_type == "review":
                        review_logs += count
        config_path = snapshot_dir / "exp_config.yaml"
        exp_config_cleaned = OmegaConf.create(
            _redact_api_keys(OmegaConf.to_container(exp_config, resolve=False))
        )
        with open(config_path, "w") as f:
            OmegaConf.save(exp_config_cleaned, f)

        research_cfg = getattr(exp_config, "research", None) or {}
        review_cfg = getattr(exp_config, "review", None) or {}
        try:
            prior_cfg = research_cfg.get("prior_papers", {}) or {}
        except Exception:
            prior_cfg = {}
        try:
            prior_enabled = bool(prior_cfg.get("enabled", False))
        except Exception:
            prior_enabled = False
        try:
            prior_mode = str(prior_cfg.get("mode", "full")).strip().lower()
        except Exception:
            prior_mode = "summary"
        try:
            full_citing_requested = bool(research_cfg.get("full_citing", False))
        except Exception:
            full_citing_requested = False
        full_citing_effective = bool(
            full_citing_requested and prior_enabled and prior_mode == "full"
        )
        try:
            review_require_citations = bool(
                review_cfg.get("require_assertion_justification", False)
            )
        except Exception:
            review_require_citations = False
        citation_tags_metadata = _build_citation_snapshot_metadata(
            iteration=iteration,
            paper_metadata=paper_metadata,
            private_archive=private_archive,
            session_stats_registry=session_stats_registry,
            require_research_tags=full_citing_effective,
            require_review_tags=review_require_citations,
        )

        files_saved = [
            "agent_registry_data.pkl",
            "archive_data.pkl",
            "archive_data.json",
            "private_archive_data.pkl",
            "private_archive_data.json",
            "notepad_data.pkl",
            "notepad_data.json",
            "session_stats_data.pkl",
            "exp_config.yaml",
        ] + (["metrics.jsonl"] if metrics_by_paper else [])
        if meta_researcher_file_written:
            files_saved.append("meta_researcher.json")
        if metacognition_feedback_file_written:
            files_saved.append("metacognition_feedback.json")
        if fixed_workflow_assignments_written:
            files_saved.append("fixed_workflow_assignments.json")
        if fixed_workflow_compliance_written:
            files_saved.append("fixed_workflow_compliance.jsonl")
        if social_memory_pruning_written:
            files_saved.append("social_memory_pruning.json")
        if extra_snapshot_files:
            files_saved.extend([path for path in extra_snapshot_files if path not in files_saved])
        plots_dir = snapshot_dir / "plots"
        if plots_dir.exists() and plots_dir.is_dir():
            files_saved.append("plots/")
            plots_manifest = plots_dir / "manifest.jsonl"
            if plots_manifest.exists():
                files_saved.append("plots/manifest.jsonl")
            summary_manifest = plots_dir / "summary_manifest.jsonl"
            if summary_manifest.exists():
                files_saved.append("plots/summary_manifest.jsonl")
            summaries_dir = plots_dir / "summaries"
            if summaries_dir.exists() and summaries_dir.is_dir():
                files_saved.append("plots/summaries/")
        run_prefix = f"{base_snapshots_dir}/{exp_dir}/"
        for _agent_id, raw_path in (notepad_data.get("notepad_file_paths", {}) or {}).items():
            path_text = str(raw_path or "").strip()
            if not path_text:
                continue
            normalized = path_text.replace("\\", "/")
            if normalized.startswith(run_prefix):
                normalized = normalized[len(run_prefix) :]
            files_saved.append(normalized)

        meta_researcher_summary = None
        if isinstance(meta_researcher_metadata, Mapping):
            trigger_payload = meta_researcher_metadata.get("trigger", {}) or {}
            if not isinstance(trigger_payload, Mapping):
                trigger_payload = {}
            meta_researcher_summary = {
                "triggered": bool(trigger_payload.get("triggered", False)),
                "num_source_papers": int(meta_researcher_metadata.get("num_source_papers", 0) or 0),
                "num_meta_papers": int(meta_researcher_metadata.get("num_meta_papers", 0) or 0),
                "num_meta_papers_generated": int(
                    meta_researcher_metadata.get(
                        "num_meta_papers_generated",
                        meta_researcher_metadata.get("num_meta_papers", 0),
                    )
                    or 0
                ),
                "num_meta_papers_filtered_out": int(
                    meta_researcher_metadata.get("num_meta_papers_filtered_out", 0) or 0
                ),
                "skipped_reason": meta_researcher_metadata.get("skipped_reason"),
            }
            quality_summary = meta_researcher_metadata.get("meta_quality_summary", {}) or {}
            if isinstance(quality_summary, Mapping):
                meta_researcher_summary["meta_quality_summary"] = {
                    "passed": int(quality_summary.get("passed", 0) or 0),
                    "failed": int(quality_summary.get("failed", 0) or 0),
                    "revised_passed": int(quality_summary.get("revised_passed", 0) or 0),
                }

        metadata = {
            "iteration": iteration,
            "timestamp": timestamp,
            "run_name": exp_dir,
            "snapshot_created_at": datetime.now().isoformat(),
            "num_agents": len(agent_registry_data["agents"]),
            "num_papers_public": len(archive_data["papers"]),
            "citation_tags": citation_tags_metadata,
            "files_saved": files_saved,
        }
        if meta_researcher_summary is not None:
            metadata["meta_researcher"] = meta_researcher_summary
        if metacognition_feedback_file_written and isinstance(
            metacognition_feedback_metadata, Mapping
        ):
            metadata["metacognition_feedback"] = {
                "mode": str(metacognition_feedback_metadata.get("mode", "none")),
                "source_iteration": metacognition_feedback_metadata.get("source_iteration"),
                "agent_count": len(
                    metacognition_feedback_metadata.get("feedback_by_agent", {}) or {}
                ),
            }

        metadata_path = snapshot_dir / "snapshot_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return str(snapshot_dir)

    except Exception:
        raise
