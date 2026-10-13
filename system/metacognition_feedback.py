from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

VALID_MODES = {"none", "meta_review_only", "meta_review_with_metrics", "metrics_only"}
DEFAULT_AXES = ("originality", "significance", "soundness", "support")

METRIC_DEFINITIONS: dict[str, dict[str, str]] = {
    "support": {
        "external_papers_read": "Count of literature/archive retrieval calls.",
        "successful_code_executions": "Successful analysis code executions.",
        "lines_of_code": "Total executed code lines.",
        "evidence_types_present": "Evidence modalities detected in the manuscript.",
        "num_statistical_code_cells": "Statistical analysis code cells detected.",
        "evidence_breadth_index": "Literature breadth plus evidence modality breadth.",
        "analytical_depth": "Approximate depth of statistical analysis.",
        "claim_evidence_alignment_max_cosine": "Similarity between claim-like and evidence-like sentences.",
    },
    "soundness": {
        "internal_consistency_score": "Heuristic consistency and rigor score in [0, 1].",
        "num_statistical_validity_checks_passed": "Distinct validity checks mentioned without failure cues.",
        "num_statistical_validity_checks_total": "Distinct statistical validity checks detected.",
        "diagnostic_validity_pass_rate": "Passed validity checks divided by detected checks.",
    },
    "significance": {
        "clinical_biological_relevance": "Clinical/biological keyword relevance score in [0, 1].",
        "clinical_relevance_score": "Alias of clinical_biological_relevance.",
        "effect_size_strength": "Extracted effect-size strength score in [0, 1].",
        "distance_to_prior_mean": "Mean semantic distance to archive neighbors.",
        "distance_to_prior_min": "Minimum semantic distance to archive neighbors.",
        "distance_to_prior_max": "Maximum semantic distance to archive neighbors.",
    },
    "originality": {
        "hypothesis_overlap_jaccard": "Average overlap with prior hypothesis tokens; lower is more novel.",
        "concept_pair_rarity": "Rarity of concept pairs in the archive; higher is rarer.",
        "concept_entropy": "Entropy over concepts; higher means broader concept mix.",
        "semantic_novelty_knn_distance": "Mean semantic distance to archive neighbors.",
        "embedding_novelty_percentile": "Batch percentile rank of semantic novelty.",
        "methodological_novelty_flag": "Whether a novel method/approach is claimed.",
        "mechanistic_novelty_flag": "Whether a novel mechanism/pathway is claimed.",
    },
}

DEFAULT_METRIC_CATALOG: dict[str, list[str]] = {
    category: list(definitions.keys()) for category, definitions in METRIC_DEFINITIONS.items()
}


def normalize_metacognition_feedback_cfg(raw: Any) -> dict[str, Any]:
    """Normalize optional metacognition config.

    Missing config is intentionally equivalent to disabled mode so old configs
    and old snapshots remain compatible.
    """
    data = _to_mapping_dict(raw)
    mode = str(data.get("mode", "none") or "none").strip().lower()
    enabled = _as_bool(data.get("enabled", False), False)
    if not enabled:
        mode = "none"
    if mode not in VALID_MODES:
        raise ValueError(
            f"Unknown metacognition_feedback.mode={mode!r}; expected one of {sorted(VALID_MODES)}"
        )

    axes = data.get("axes", DEFAULT_AXES)
    if isinstance(axes, str):
        axes = [x.strip() for x in axes.split(",")]
    axes_norm = [str(x).strip().lower() for x in (axes or DEFAULT_AXES) if str(x).strip()]
    axes_norm = [x for x in axes_norm if x in DEFAULT_AXES] or list(DEFAULT_AXES)

    metrics_allowlist = data.get("metrics_allowlist", None)
    if isinstance(metrics_allowlist, str):
        metrics_allowlist = [x.strip() for x in metrics_allowlist.split(",") if x.strip()]
    elif metrics_allowlist is not None and not isinstance(metrics_allowlist, list):
        try:
            metrics_allowlist = list(metrics_allowlist)
        except Exception:
            metrics_allowlist = None

    return {
        "enabled": enabled,
        "mode": mode,
        "apply_on_resume_first_iteration": _as_bool(
            data.get("apply_on_resume_first_iteration", True), True
        ),
        "source_iteration": data.get("source_iteration", "previous"),
        "include_rejected": _as_bool(data.get("include_rejected", True), True),
        "include_timeout": _as_bool(data.get("include_timeout", False), False),
        "include_collaborator_papers": _as_bool(
            data.get("include_collaborator_papers", True), True
        ),
        "max_papers_per_agent": _as_int(data.get("max_papers_per_agent", 3), 3),
        "max_reviews_per_paper": _as_int(data.get("max_reviews_per_paper", 4), 4),
        "max_chars_per_review": _as_int(data.get("max_chars_per_review", 1200), 1200),
        "max_chars_per_paper": _as_int(data.get("max_chars_per_paper", 1800), 1800),
        "axes": axes_norm,
        "metrics_allowlist": metrics_allowlist,
        "llm_backend": data.get("llm_backend", None),
        "strict_source_required": _as_bool(data.get("strict_source_required", True), True),
        "best_effort": _as_bool(data.get("best_effort", False), False),
        "write_artifacts": _as_bool(data.get("write_artifacts", True), True),
    }


def resolve_feedback_source_iteration(
    iteration: int,
    restored_iteration: int | None,
    cfg: Mapping[str, Any],
) -> int | None:
    """Resolve which completed iteration should provide feedback."""
    if not _as_bool((cfg or {}).get("enabled", False), False):
        return None
    if str((cfg or {}).get("mode", "none") or "none").strip().lower() == "none":
        return None

    raw = (cfg or {}).get("source_iteration", "previous")
    if isinstance(raw, int):
        return int(raw)

    text = str(raw or "previous").strip().lower()
    if text == "previous":
        previous = int(iteration) - 1
        return previous if previous >= 1 else None
    if text == "restored":
        return int(restored_iteration) if restored_iteration is not None else None
    try:
        return int(text)
    except Exception as exc:
        raise ValueError(f"Invalid metacognition source_iteration={raw!r}") from exc


def collect_feedback_source_records(
    private_archive: Any,
    iteration: int,
    include_rejected: bool,
    include_timeout: bool,
    *,
    max_chars_per_paper: int = 1800,
    max_reviews_per_paper: int = 4,
    max_chars_per_review: int = 1200,
) -> list[dict[str, Any]]:
    """Collect previous-iteration paper/review/metareview records from a private archive."""
    papers = _flatten_archive_papers(private_archive)
    reviews_by_paper = _reviews_by_paper(private_archive)
    out: list[dict[str, Any]] = []

    for paper in papers:
        publication_t = _as_int(_get(paper, "publication_t", -1), -1)
        if publication_t != int(iteration):
            continue

        status = _status_for_paper(paper)
        if _is_timeout_or_incomplete(paper, status) and not include_timeout:
            continue
        if status == "rejected" and not include_rejected:
            continue
        if status not in {"accepted", "rejected", "under_review"} and not include_timeout:
            continue

        paper_id = str(_get(paper, "paper_id", "") or "").strip()
        if not paper_id:
            continue

        metareview_obj = _get(paper, "metareview", None)
        metareview = _metareview_to_dict(metareview_obj)
        raw_reviews = reviews_by_paper.get(paper_id, []) or []
        reviews: list[dict[str, Any]] = []
        for row in raw_reviews[: max(0, int(max_reviews_per_paper))]:
            if not isinstance(row, Mapping):
                continue
            reviews.append(
                {
                    "reviewer_id": row.get("reviewer_id"),
                    "scores": dict(row.get("scores") or {}),
                    "review_text": _truncate(
                        str(row.get("review_text", "") or ""), max_chars_per_review
                    ),
                }
            )

        metrics = _get(paper, "review_metrics", {})
        out.append(
            {
                "paper_id": paper_id,
                "primary_agent_id": str(_get(paper, "primary_agent_id", "") or "").strip(),
                "collab_agent_ids": [
                    str(x).strip()
                    for x in (_get(paper, "collab_agent_ids", []) or [])
                    if str(x).strip()
                ],
                "title": str(_get(paper, "title", "") or "Untitled").strip(),
                "abstract": _truncate(str(_get(paper, "abstract", "") or ""), max_chars_per_paper),
                "manuscript_excerpt": _truncate(
                    str(_get(paper, "manuscript", "") or ""), max_chars_per_paper
                ),
                "status": status,
                "publication_t": publication_t,
                "metareview": metareview,
                "reviews": reviews,
                "metrics": dict(metrics) if isinstance(metrics, Mapping) else {},
            }
        )

    out.sort(key=lambda r: (str(r.get("primary_agent_id", "")), str(r.get("paper_id", ""))))
    return out


def load_snapshot_metrics(snapshot_iteration_dir: str | Path | None) -> dict[str, dict[str, Any]]:
    """Load metrics.jsonl from a snapshot iteration directory if available."""
    if snapshot_iteration_dir is None:
        return {}
    path = Path(snapshot_iteration_dir) / "metrics.jsonl"
    if not path.exists():
        return {}

    metrics: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, Mapping):
                continue
            paper_id = str(row.get("paper_id", "") or "").strip()
            payload = row.get("metrics", {})
            if paper_id and isinstance(payload, Mapping):
                metrics[paper_id] = dict(payload)
    return metrics


def group_records_by_agent(
    records: Iterable[Mapping[str, Any]],
    include_collaborator_papers: bool,
) -> dict[str, list[dict[str, Any]]]:
    """Group source records by primary author and optionally collaborators."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, str]] = set()
    for record in records or []:
        paper_id = str(record.get("paper_id", "") or "").strip()
        agent_ids = [str(record.get("primary_agent_id", "") or "").strip()]
        if include_collaborator_papers:
            agent_ids.extend(str(x).strip() for x in (record.get("collab_agent_ids", []) or []))
        for agent_id in agent_ids:
            if not agent_id or not paper_id:
                continue
            key = (agent_id, paper_id)
            if key in seen:
                continue
            seen.add(key)
            grouped.setdefault(agent_id, []).append(dict(record))
    for rows in grouped.values():
        rows.sort(key=lambda r: str(r.get("paper_id", "")))
    return grouped


def format_metrics_only_feedback(
    agent_id: str,
    records: list[Mapping[str, Any]],
    metric_catalog: Mapping[str, list[str]],
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """Build deterministic feedback payload from metric definitions and scores."""
    axes = list(cfg.get("axes", DEFAULT_AXES) or DEFAULT_AXES)
    records = _limit_records(records, cfg)
    metric_catalog = _apply_metric_allowlist(metric_catalog or {}, cfg.get("metrics_allowlist"))
    metric_catalog = _filter_metric_catalog_by_axes(metric_catalog, axes)
    raw_metrics = {
        str(r.get("paper_id", "")): _filter_metrics_by_catalog(r.get("metrics"), metric_catalog)
        for r in records
        if str(r.get("paper_id", "")).strip()
    }

    axes_payload: dict[str, dict[str, list[str]]] = {}
    for axis in axes:
        metric_names = list((metric_catalog or {}).get(axis, []) or [])
        values = _axis_numeric_values(raw_metrics, axis)
        strengths: list[str] = []
        weaknesses: list[str] = []
        if values:
            best_key, best_val = max(values, key=lambda item: item[1])
            low_values = [(k, v) for k, v in values if v < 0.35]
            strengths.append(f"Highest observed {axis} signal: {best_key}={best_val:.3g}.")
            for key, value in low_values[:3]:
                weaknesses.append(f"Low observed {axis} signal: {key}={value:.3g}.")
        else:
            weaknesses.append(
                f"No numeric {axis} metric scores were available in the source records."
            )

        metric_list = ", ".join(metric_names[:8]) if metric_names else "no configured metrics"
        axes_payload[axis] = {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "actions": [
                f"Use the {axis} scores and definitions to choose one concrete improvement target.",
                f"Relevant {axis} metrics: {metric_list}.",
            ],
        }

    return {
        "source_iteration": _source_iteration_from_records(records),
        "mode": "metrics_only",
        "summary": (
            f"Direct metric feedback for {agent_id}: review the prior round's metric definitions "
            "and scores, then target the weakest available dimension in the next paper."
        ),
        "axes": axes_payload,
        "source_paper_ids": [
            str(r.get("paper_id", "")) for r in records if str(r.get("paper_id", "")).strip()
        ],
        "raw_metrics": raw_metrics,
        "metric_definitions": _metric_definitions_for_catalog(metric_catalog),
    }


def make_feedback_synthesizer_user_message(
    agent_id: str,
    records: list[Mapping[str, Any]],
    include_metrics: bool,
    cfg: Mapping[str, Any],
) -> str:
    """Build the LLM user message for agent-targeted feedback synthesis."""
    axes = list(cfg.get("axes", DEFAULT_AXES) or DEFAULT_AXES)
    records = _limit_records(records, cfg)
    metric_catalog = _apply_metric_allowlist(_review_metric_catalog(), cfg.get("metrics_allowlist"))
    metric_catalog = _filter_metric_catalog_by_axes(metric_catalog, axes)
    rows: list[str] = []
    for idx, record in enumerate(records, start=1):
        parts = [
            f"Paper {idx}",
            f"paper_id: {record.get('paper_id')}",
            f"status: {record.get('status')}",
            f"title: {record.get('title')}",
            f"abstract: {record.get('abstract')}",
            f"manuscript_excerpt: {record.get('manuscript_excerpt')}",
        ]
        metareview = record.get("metareview") or {}
        if isinstance(metareview, Mapping) and metareview:
            parts.append(
                "metareview: "
                + json.dumps(
                    {
                        "overall_score": metareview.get("overall_score"),
                        "decision": metareview.get("decision"),
                        "rank": metareview.get("rank"),
                        "justification": metareview.get("justification"),
                        "meta_review_text": metareview.get("meta_review_text"),
                    },
                    ensure_ascii=False,
                    default=str,
                )
            )
        reviews = record.get("reviews") or []
        if reviews:
            parts.append("reviews: " + json.dumps(reviews, ensure_ascii=False, default=str))
        if include_metrics:
            filtered_metrics = _filter_metrics_by_catalog(record.get("metrics"), metric_catalog)
            parts.append(
                "metrics: " + json.dumps(filtered_metrics, ensure_ascii=False, default=str)
            )
        rows.append("\n".join(parts))

    metric_context = ""
    if include_metrics:
        metric_definitions = _metric_definitions_for_catalog(metric_catalog)
        compact_defs: list[str] = []
        for category in axes:
            defs = metric_definitions.get(category, {})
            if not isinstance(defs, Mapping):
                continue
            entries = "; ".join(f"{name}: {desc}" for name, desc in list(defs.items())[:8])
            if entries:
                compact_defs.append(f"- {category}: {entries}")
        if compact_defs:
            metric_context = (
                "Short internal metric definitions:\n" + "\n".join(compact_defs) + "\n\n"
            )

    return (
        f"Write concise metacognitive feedback for research agent {agent_id}.\n"
        "Use only the previous-round evidence below. This is planning guidance, not manuscript evidence.\n"
        f"Axes to cover: {', '.join(axes)}.\n"
        + metric_context
        + "Return strict JSON only with this shape:\n"
        "{\n"
        '  "summary": "one short paragraph",\n'
        '  "axes": {\n'
        '    "originality": {"strengths": [], "weaknesses": [], "actions": []},\n'
        '    "significance": {"strengths": [], "weaknesses": [], "actions": []},\n'
        '    "soundness": {"strengths": [], "weaknesses": [], "actions": []},\n'
        '    "support": {"strengths": [], "weaknesses": [], "actions": []}\n'
        "  }\n"
        "}\n\n"
        "Previous-round records:\n" + ("\n\n---\n\n".join(rows) if rows else "No source records.")
    )


def parse_feedback_synthesizer_output(text: str) -> dict[str, Any]:
    """Parse strict JSON feedback from the synthesizer."""
    payload = _extract_json_object(text)
    if not isinstance(payload, Mapping):
        raise ValueError("Feedback synthesizer output must be a JSON object.")
    summary = str(payload.get("summary", "") or "").strip()
    axes = payload.get("axes")
    if not summary or not isinstance(axes, Mapping):
        raise ValueError("Feedback synthesizer output requires summary and axes.")

    normalized_axes: dict[str, dict[str, list[str]]] = {}
    for axis in DEFAULT_AXES:
        raw_axis = axes.get(axis)
        if not isinstance(raw_axis, Mapping):
            raise ValueError(f"Feedback synthesizer output missing axes.{axis}.")
        normalized_axes[axis] = {
            "strengths": _list_of_strings(raw_axis.get("strengths")),
            "weaknesses": _list_of_strings(raw_axis.get("weaknesses")),
            "actions": _list_of_strings(raw_axis.get("actions")),
        }
    return {"summary": summary, "axes": normalized_axes}


def build_metacognition_feedback_for_agents(
    *,
    agents: Iterable[Any] | None = None,
    private_archive: Any,
    iteration: int,
    restored_iteration: int | None,
    cfg: Mapping[str, Any],
    source_snapshot_dir: str | Path | None = None,
    llm: Any = None,
    skipped_agents: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build active metacognition payloads keyed by agent id.

    Disabled config returns immediately and has no side effects.

    Args:
        agents: Active agent profiles; when given, only their records are used.
        private_archive: Private archive holding the source round's papers.
        iteration: Current iteration.
        restored_iteration: Iteration restored from a snapshot, when resuming.
        cfg: The ``research.metacognition_feedback`` config.
        source_snapshot_dir: Snapshot directory of the source round, for its metrics.
        llm: Chat model for the LLM feedback modes.
        skipped_agents: If given, filled with a skip record (``source_iteration``,
            ``mode``, ``skipped`` reason, ``_warnings`` and ``_llm_usage``) for each agent
            whose feedback a best-effort run skipped this round. Skipped agents are not
            in the returned dict.

    Returns:
        Feedback payloads keyed by agent id.

    Raises:
        ValueError: If an LLM mode has no ``llm``, or, in a strict run, if source data is
            missing or a feedback-synthesizer reply cannot be parsed after two attempts.
    """
    norm = normalize_metacognition_feedback_cfg(cfg)
    if not norm["enabled"] or norm["mode"] == "none":
        return {}

    source_iteration = resolve_feedback_source_iteration(iteration, restored_iteration, norm)
    if source_iteration is None:
        msg = "Could not resolve metacognition feedback source iteration."
        if _allow_best_effort(norm):
            log.warning("[metacognition] %s", msg)
            return {}
        raise ValueError(msg)

    records = collect_feedback_source_records(
        private_archive,
        source_iteration,
        include_rejected=norm["include_rejected"],
        include_timeout=norm["include_timeout"],
        max_chars_per_paper=norm["max_chars_per_paper"],
        max_reviews_per_paper=norm["max_reviews_per_paper"],
        max_chars_per_review=norm["max_chars_per_review"],
    )
    if source_snapshot_dir is not None:
        snapshot_metrics = load_snapshot_metrics(source_snapshot_dir)
        _merge_snapshot_metrics(records, snapshot_metrics)

    if not records:
        msg = f"No metacognition source records found for iteration {source_iteration}."
        if _allow_best_effort(norm):
            log.warning("[metacognition] %s", msg)
            return {}
        raise ValueError(msg)

    metric_catalog = _review_metric_catalog()
    metric_catalog = _apply_metric_allowlist(metric_catalog, norm["metrics_allowlist"])
    metric_catalog = _filter_metric_catalog_by_axes(metric_catalog, norm["axes"])
    records = _copy_records_with_filtered_metrics(records, metric_catalog)
    grouped = group_records_by_agent(records, norm["include_collaborator_papers"])
    allowed_agent_ids = _agent_id_set(agents)
    if allowed_agent_ids:
        grouped = {aid: rows for aid, rows in grouped.items() if aid in allowed_agent_ids}
        missing_agent_ids = sorted(allowed_agent_ids.difference(grouped.keys()))
        if missing_agent_ids and not _allow_best_effort(norm):
            preview = ", ".join(missing_agent_ids[:10])
            if len(missing_agent_ids) > 10:
                preview += f", ... (+{len(missing_agent_ids) - 10} more)"
            raise ValueError(
                "Missing metacognition source records for active agents "
                f"from iteration {source_iteration}: {preview}"
            )
        if missing_agent_ids:
            log.warning(
                "[metacognition] missing source records for %d active agents from iteration %s",
                len(missing_agent_ids),
                source_iteration,
            )
    if not grouped:
        msg = f"No metacognition source records matched active agents for iteration {source_iteration}."
        if _allow_best_effort(norm):
            log.warning("[metacognition] %s", msg)
            return {}
        raise ValueError(msg)

    feedback_by_agent: dict[str, dict[str, Any]] = {}
    mode = norm["mode"]
    include_metrics = mode == "meta_review_with_metrics"
    if mode in {"meta_review_only", "meta_review_with_metrics"} and llm is None:
        raise ValueError(f"metacognition_feedback.mode={mode!r} requires an LLM.")

    for agent_id, agent_records in grouped.items():
        agent_records = _limit_records(agent_records, norm)
        source_warnings = _source_record_warnings(agent_id, agent_records, mode=mode)
        blocking_warnings = _blocking_source_warnings(source_warnings, mode=mode)
        if blocking_warnings and not _allow_best_effort(norm):
            raise ValueError(
                "Missing required metacognition source data for "
                f"agent_id={agent_id}: " + "; ".join(blocking_warnings)
            )
        if mode == "metrics_only":
            payload = format_metrics_only_feedback(agent_id, agent_records, metric_catalog, norm)
        else:
            user_message = make_feedback_synthesizer_user_message(
                agent_id,
                agent_records,
                include_metrics=include_metrics,
                cfg=norm,
            )
            # An unparseable reply is retried once. If it still fails, best-effort runs skip
            # this agent's feedback for the round; strict runs stop, as before.
            parsed = None
            parse_error: Exception | None = None
            llm_usage: dict[str, Any] | None = None  # summed over both attempts
            for attempt in (1, 2):
                response = _invoke_feedback_llm(
                    llm,
                    _feedback_synthesizer_messages(user_message),
                    stage_label=f"metacognition_feedback.{mode}.{agent_id}",
                )
                usage = getattr(response, "usage_metadata", None)
                if isinstance(usage, Mapping):
                    llm_usage = _add_usage(llm_usage or {}, usage)
                try:
                    parsed = parse_feedback_synthesizer_output(_message_content(response))
                    break
                except (ValueError, TypeError) as exc:
                    parse_error = exc
                    log.warning(
                        "[metacognition] unparseable feedback-synthesizer reply for "
                        "agent_id=%s (attempt %d/2): %s",
                        agent_id,
                        attempt,
                        exc,
                    )
            if parsed is None:
                if _allow_best_effort(norm):
                    log.warning(
                        "[metacognition] no feedback for agent_id=%s this round "
                        "(synthesizer reply unparseable twice).",
                        agent_id,
                    )
                    reason = "feedback-synthesizer reply unparseable twice; no feedback this round"
                    if skipped_agents is not None:
                        skipped_agents[agent_id] = {
                            "source_iteration": int(source_iteration),
                            "mode": mode,
                            "skipped": reason,
                            "_warnings": [reason],
                            **({"_llm_usage": llm_usage} if llm_usage is not None else {}),
                        }
                    continue
                raise ValueError(
                    f"Feedback synthesizer reply for agent_id={agent_id} could not be parsed "
                    f"after 2 attempts: {parse_error}"
                ) from parse_error
            payload = {
                "source_iteration": int(source_iteration),
                "mode": mode,
                "summary": parsed["summary"],
                "axes": parsed["axes"],
                "source_paper_ids": [
                    str(r.get("paper_id", ""))
                    for r in agent_records
                    if str(r.get("paper_id", "")).strip()
                ],
                "raw_metrics": (
                    {
                        str(r.get("paper_id", "")): dict(r.get("metrics") or {})
                        for r in agent_records
                        if str(r.get("paper_id", "")).strip()
                    }
                    if include_metrics
                    else {}
                ),
            }
            if llm_usage is not None:
                payload["_llm_usage"] = llm_usage
        payload["source_iteration"] = int(source_iteration)
        payload["mode"] = mode
        if source_warnings:
            payload["_warnings"] = source_warnings
        feedback_by_agent[agent_id] = payload

    return feedback_by_agent


def make_metacognition_feedback_metadata(
    *,
    iteration: int,
    source_iteration: int | None,
    mode: str,
    feedback_by_agent: Mapping[str, Any],
    write_artifacts: bool = True,
    skipped_agents: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build snapshot metadata for already-constructed feedback payloads.

    Args:
        iteration: Current iteration.
        source_iteration: The round the feedback was built from.
        mode: Feedback mode.
        feedback_by_agent: Payloads from build_metacognition_feedback_for_agents.
        write_artifacts: Whether the feedback artefact is written.
        skipped_agents: Skip records from build_metacognition_feedback_for_agents; their
            warnings and token usage are recorded, but they are not in ``feedback_by_agent``.

    Returns:
        The ``metacognition_feedback`` snapshot metadata.
    """
    sanitized_feedback: dict[str, Any] = {}
    llm_usage_by_agent: dict[str, Any] = {}
    for agent_id, payload in (feedback_by_agent or {}).items():
        if not isinstance(payload, Mapping):
            continue
        sanitized_payload = dict(payload)
        usage = sanitized_payload.pop("_llm_usage", None)
        if isinstance(usage, Mapping):
            llm_usage_by_agent[str(agent_id)] = dict(usage)
        sanitized_feedback[str(agent_id)] = sanitized_payload
    counts = {
        str(agent_id): len((payload or {}).get("source_paper_ids", []) or [])
        for agent_id, payload in sanitized_feedback.items()
        if isinstance(payload, Mapping)
    }
    warnings = {
        str(agent_id): list(payload.get("_warnings", []) or [])
        for agent_id, payload in sanitized_feedback.items()
        if isinstance(payload, Mapping) and payload.get("_warnings")
    }
    for agent_id, record in (skipped_agents or {}).items():
        if not isinstance(record, Mapping):
            continue
        if record.get("_warnings"):
            warnings[str(agent_id)] = list(record["_warnings"])
        if isinstance(record.get("_llm_usage"), Mapping):
            llm_usage_by_agent[str(agent_id)] = dict(record["_llm_usage"])
    metadata = {
        "iteration": int(iteration),
        "source_iteration": source_iteration,
        "mode": str(mode or "none"),
        "feedback_by_agent": sanitized_feedback,
        "source_paper_counts_by_agent": counts,
        "warnings_by_agent": warnings,
        "write_artifacts": _as_bool(write_artifacts, True),
    }
    if llm_usage_by_agent:
        metadata["llm_usage_by_agent"] = llm_usage_by_agent
    return metadata


def merge_metacognition_feedback_learning_state(
    agent: Any,
    feedback_payload: Mapping[str, Any],
    *,
    history_limit: int = 10,
) -> dict[str, Any]:
    """Return merged learning_state with namespaced metacognition feedback."""
    learning_state = dict(getattr(agent, "learning_state", None) or {})
    previous = learning_state.get("metacognition_feedback", {})
    previous = dict(previous) if isinstance(previous, Mapping) else {}
    history = [dict(x) for x in (previous.get("history", []) or []) if isinstance(x, Mapping)]
    payload = {
        key: value for key, value in dict(feedback_payload or {}).items() if key != "_llm_usage"
    }
    learning_state["metacognition_feedback"] = {
        "active": payload,
        "history": [*history, payload][-max(1, int(history_limit)) :],
    }
    return learning_state


def _feedback_synthesizer_system_prompt() -> str:
    return (
        "You synthesize previous-round scientific review evidence into concise, agent-specific "
        "planning feedback. Do not make accept/reject decisions. Do not introduce new scientific "
        "claims. Return strict JSON only."
    )


def _feedback_synthesizer_messages(user_message: str) -> list[Any]:
    """Build chat messages without making pure-helper imports depend on LangChain."""
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        return [
            SystemMessage(content=_feedback_synthesizer_system_prompt()),
            HumanMessage(content=user_message),
        ]
    except ModuleNotFoundError:
        return [
            {"role": "system", "content": _feedback_synthesizer_system_prompt()},
            {"role": "user", "content": user_message},
        ]


def _invoke_feedback_llm(llm: Any, messages: list[Any], *, stage_label: str) -> Any:
    """Invoke the sidecar feedback LLM, using retry support when available."""
    try:
        from system.llm_retry import invoke_with_content_filter_retry

        return invoke_with_content_filter_retry(
            llm,
            messages,
            stage_label=stage_label,
            logger=log,
        )
    except ModuleNotFoundError as exc:
        invoke = getattr(llm, "invoke", None)
        if callable(invoke):
            return invoke(messages)
        raise RuntimeError(
            "Metacognition feedback LLM mode requires an LLM with invoke(...) "
            "or the project's LangChain/OpenAI dependencies."
        ) from exc


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", "none", "null", ""}:
        return False
    return bool(default)


def _to_mapping_dict(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return dict(raw)
    items = getattr(raw, "items", None)
    if callable(items):
        try:
            return dict(items())
        except Exception:
            return {}
    return {}


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _truncate(text: str, max_chars: int) -> str:
    try:
        limit = max(0, int(max_chars))
    except Exception:
        limit = 0
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _flatten_archive_papers(private_archive: Any) -> list[Any]:
    if private_archive is None:
        return []
    if isinstance(private_archive, Mapping):
        raw = private_archive.get("papers", [])
        return _flatten_paper_rows(raw)
    entries = getattr(private_archive, "_entries", {}) or {}
    if isinstance(entries, Mapping):
        papers: list[Any] = []
        for rows in entries.values():
            papers.extend(list(rows or []))
        return papers
    return []


def _flatten_paper_rows(raw: Any) -> list[Any]:
    flat: list[Any] = []
    for item in raw or []:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    return flat


def _reviews_by_paper(private_archive: Any) -> dict[str, list[dict[str, Any]]]:
    if private_archive is None:
        return {}
    if isinstance(private_archive, Mapping):
        raw = private_archive.get("reviews_by_paper", {}) or {}
    else:
        raw = getattr(private_archive, "reviews_by_paper", {}) or {}
    if not isinstance(raw, Mapping):
        return {}
    return {str(k): [dict(x) for x in (v or []) if isinstance(x, Mapping)] for k, v in raw.items()}


def _status_for_paper(paper: Any) -> str:
    return (
        str(
            _get(
                paper,
                "publication_status",
                _get(paper, "status", _get(paper, "research_validation_status", "under_review")),
            )
            or "under_review"
        )
        .strip()
        .lower()
    )


def _is_timeout_or_incomplete(paper: Any, status: str) -> bool:
    paper_id = str(_get(paper, "paper_id", "") or "")
    manuscript = str(_get(paper, "manuscript", "") or "")
    return (
        status in {"timeout", "incomplete"}
        or paper_id.endswith("_timeout")
        or paper_id.endswith("_incomplete")
        or manuscript.strip() == "TIMEOUT"
    )


def _metareview_to_dict(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return dict(raw)
    return {
        "paper_id": getattr(raw, "paper_id", None),
        "meta_review_text": getattr(raw, "meta_review_text", None),
        "overall_score": getattr(raw, "overall_score", None),
        "rank": getattr(raw, "rank", None),
        "justification": getattr(raw, "justification", None),
        "decision": getattr(raw, "decision", None),
    }


def _limit_records(
    records: list[Mapping[str, Any]], cfg: Mapping[str, Any]
) -> list[dict[str, Any]]:
    limit = max(1, _as_int((cfg or {}).get("max_papers_per_agent", 3), 3))
    return [dict(r) for r in list(records or [])[:limit]]


def _source_iteration_from_records(records: list[Mapping[str, Any]]) -> int | None:
    for record in records or []:
        value = record.get("publication_t")
        try:
            return int(value)
        except Exception:
            continue
    return None


def _review_metric_catalog() -> dict[str, list[str]]:
    """Return the canonical review metric catalog, with a lightweight fallback."""
    try:
        from system.review_metrics import review_metric_catalog

        return review_metric_catalog()
    except ModuleNotFoundError:
        return {category: list(names) for category, names in DEFAULT_METRIC_CATALOG.items()}


def _normalize_metric_allowlist(
    metrics_allowlist: list[str] | None,
) -> tuple[dict[str, set[str]], list[str]]:
    """Normalize a metric allowlist without requiring the full metrics stack."""
    try:
        from system.review_metrics import normalize_metric_allowlist

        return normalize_metric_allowlist(metrics_allowlist)
    except ModuleNotFoundError:
        catalog = _review_metric_catalog()
        unknown: list[str] = []
        if not metrics_allowlist:
            return {cat: set(keys) for cat, keys in catalog.items()}, unknown

        allow: dict[str, set[str]] = {cat: set() for cat in catalog}
        for raw in metrics_allowlist:
            entry = str(raw).strip()
            if not entry:
                continue
            if "." not in entry:
                if entry in catalog:
                    allow[entry].update(catalog[entry])
                else:
                    unknown.append(entry)
                continue
            category, metric = entry.split(".", 1)
            if category not in catalog:
                unknown.append(entry)
                continue
            if metric in {"*", "all"}:
                allow[category].update(catalog[category])
            elif metric in catalog[category]:
                allow[category].add(metric)
            else:
                unknown.append(entry)
        return allow, unknown


def _apply_metric_allowlist(
    metric_catalog: Mapping[str, list[str]],
    metrics_allowlist: list[str] | None,
) -> dict[str, list[str]]:
    allow, _unknown = _normalize_metric_allowlist(metrics_allowlist)
    out: dict[str, list[str]] = {}
    for category, names in metric_catalog.items():
        allowed = allow.get(category, set(names))
        out[category] = [name for name in names if name in allowed]
    return out


def _filter_metric_catalog_by_axes(
    metric_catalog: Mapping[str, list[str]],
    axes: Iterable[Any],
) -> dict[str, list[str]]:
    axis_set = {str(axis).strip().lower() for axis in (axes or []) if str(axis).strip()}
    if not axis_set:
        axis_set = set(DEFAULT_AXES)
    return {
        str(category): list(names or [])
        for category, names in (metric_catalog or {}).items()
        if str(category).strip().lower() in axis_set
    }


def _filter_metric_definitions(
    metric_catalog: Mapping[str, list[str]],
    metrics_allowlist: list[str] | None,
) -> dict[str, dict[str, str]]:
    filtered_catalog = _apply_metric_allowlist(metric_catalog, metrics_allowlist)
    return _metric_definitions_for_catalog(filtered_catalog)


def _metric_definitions_for_catalog(
    metric_catalog: Mapping[str, list[str]],
) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for category, names in (metric_catalog or {}).items():
        defs = METRIC_DEFINITIONS.get(category, {})
        out[category] = {name: defs.get(name, "Review metric score.") for name in names}
    return out


def _merge_snapshot_metrics(
    records: list[dict[str, Any]], snapshot_metrics: Mapping[str, Any]
) -> None:
    if not snapshot_metrics:
        return
    for record in records:
        paper_id = str(record.get("paper_id", "") or "")
        if not paper_id:
            continue
        if isinstance(record.get("metrics"), Mapping) and record.get("metrics"):
            continue
        metrics = snapshot_metrics.get(paper_id)
        if isinstance(metrics, Mapping):
            record["metrics"] = dict(metrics)


def _copy_records_with_filtered_metrics(
    records: list[dict[str, Any]],
    metric_catalog: Mapping[str, list[str]],
) -> list[dict[str, Any]]:
    """Copy records and retain only metacognition-allowed metric scores."""
    out: list[dict[str, Any]] = []
    for record in records or []:
        row = dict(record)
        row["metrics"] = _filter_metrics_by_catalog(row.get("metrics"), metric_catalog)
        out.append(row)
    return out


def _filter_metrics_by_catalog(
    metrics: Any,
    metric_catalog: Mapping[str, list[str]],
) -> dict[str, dict[str, Any]]:
    if not isinstance(metrics, Mapping):
        return {}
    filtered: dict[str, dict[str, Any]] = {}
    for category, names in (metric_catalog or {}).items():
        raw_values = metrics.get(category, {})
        if not isinstance(raw_values, Mapping):
            continue
        allowed = {str(name) for name in (names or [])}
        kept = {str(key): value for key, value in raw_values.items() if str(key) in allowed}
        if kept:
            filtered[str(category)] = kept
    return filtered


def _axis_numeric_values(
    raw_metrics: Mapping[str, Mapping[str, Any]], axis: str
) -> list[tuple[str, float]]:
    values: list[tuple[str, float]] = []
    for paper_id, metrics in (raw_metrics or {}).items():
        axis_metrics = metrics.get(axis, {}) if isinstance(metrics, Mapping) else {}
        if not isinstance(axis_metrics, Mapping):
            continue
        for key, value in axis_metrics.items():
            if isinstance(value, bool):
                values.append((f"{paper_id}.{key}", 1.0 if value else 0.0))
            elif isinstance(value, (int, float)):
                values.append((f"{paper_id}.{key}", float(value)))
    return values


def _record_has_review_or_metareview(record: Mapping[str, Any]) -> bool:
    reviews = record.get("reviews", [])
    if isinstance(reviews, list) and any(isinstance(r, Mapping) and r for r in reviews):
        return True
    metareview = record.get("metareview", {})
    if isinstance(metareview, Mapping):
        return any(
            key != "paper_id" and value not in (None, "", [], {})
            for key, value in metareview.items()
        )
    return False


def _record_has_metrics(record: Mapping[str, Any]) -> bool:
    metrics = record.get("metrics", {})
    if not isinstance(metrics, Mapping) or not metrics:
        return False
    for values in metrics.values():
        if isinstance(values, Mapping) and values:
            return True
    return False


def _source_record_warnings(
    agent_id: str,
    records: list[Mapping[str, Any]],
    *,
    mode: str,
) -> list[str]:
    warnings: list[str] = []
    if mode in {"meta_review_only", "meta_review_with_metrics"} and not any(
        _record_has_review_or_metareview(record) for record in records
    ):
        warnings.append(
            f"agent_id={agent_id} has no reviewer or metareview text in source records."
        )
    if mode in {"meta_review_with_metrics", "metrics_only"} and not any(
        _record_has_metrics(record) for record in records
    ):
        warnings.append(f"agent_id={agent_id} has no internal review metrics in source records.")
    return warnings


def _blocking_source_warnings(warnings: list[str], *, mode: str) -> list[str]:
    if mode in {"meta_review_only", "meta_review_with_metrics"}:
        return [
            warning
            for warning in warnings
            if "no reviewer or metareview text" in warning
            or (mode == "meta_review_with_metrics" and "no internal review metrics" in warning)
        ]
    if mode == "metrics_only":
        return [warning for warning in warnings if "no internal review metrics" in warning]
    return []


def _add_usage(total: Mapping[str, Any], usage: Mapping[str, Any]) -> dict[str, Any]:
    """Add one reply's token usage to a running total (nested counts too)."""
    merged = dict(total)
    for key, value in usage.items():
        if isinstance(value, Mapping):
            previous = merged.get(key)
            merged[key] = _add_usage(previous if isinstance(previous, Mapping) else {}, value)
        elif isinstance(value, int | float) and not isinstance(value, bool):
            previous = merged.get(key, 0)
            merged[key] = (previous if isinstance(previous, int | float) else 0) + value
        else:
            merged.setdefault(key, value)
    return merged


def _message_content(response: Any) -> str:
    if isinstance(response, str):
        return response
    content = getattr(response, "content", None)
    if content is None:
        return str(response or "")
    return content if isinstance(content, str) else str(content)


def _allow_best_effort(cfg: Mapping[str, Any]) -> bool:
    return _as_bool(cfg.get("best_effort", False), False) and not _as_bool(
        cfg.get("strict_source_required", True),
        True,
    )


def _extract_json_object(text: str) -> Any:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start : end + 1])
        raise


def _list_of_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("Expected a list of strings.")
    out = [str(x).strip() for x in value if str(x).strip()]
    return out[:8]


def _agent_id_set(agents: Iterable[Any] | None) -> set[str]:
    out: set[str] = set()
    for agent in agents or []:
        agent_id = str(getattr(agent, "agent_id", None) or getattr(agent, "id", "") or "").strip()
        if agent_id:
            out.add(agent_id)
    return out
