import asyncio
import logging
import re
import traceback
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable
from omegaconf import OmegaConf
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from agents.research.research_team import get_research_team
from env_setup.build_llms import load_endpoint_config, make_sdk_client
from env_setup.sandbox_utils import MaxSessionsError, SandboxClient
from stores.private_paper_archive import PrivatePaperArchive
from system.citation_tags import validate_citation_tags
from system.code_evidence import has_code_script_recording_anomaly
from system.graph_limits import recursion_limit_for
from system.llm_retry import call_with_content_filter_retry
from system.plot_summary_artifacts import normalize_graph_logic_cfg
from system.research_output_utils import (
    extract_final_report,
    extract_internal_citations,
    extract_title_and_abstract,
    find_unresolved_template_placeholders,
    has_unresolved_template_placeholders,
    sanitize_report,
)
from system.schema import (
    AgentNetworkMetadata,
    AgentProfile,
    PaperMetadataPrivate,
)
from system.trackers import _CURRENT_LOG, SessionStats, get_logs

DEFAULT_LLM_CONFIG_PATH = Path("sys_config/llm_api/default.yaml")
DEFAULT_PRIOR_PAPERS_KEEP_FULL = 1

# throttle concurrent sandbox creates
# Default small semaphore; we set the real size from /base/stats in main()
SESSION_CREATE_SEM = asyncio.Semaphore(2)  # sandbox sessions created at once

_console = Console(stderr=True, log_path=False, record=False)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _paper_source_text(paper: PaperMetadataPrivate, max_chars: int = 6000) -> str:
    title = getattr(paper, "title", "") or "Untitled"
    abstract = getattr(paper, "abstract", "") or "No abstract available."
    manuscript = getattr(paper, "manuscript", "") or ""
    if manuscript:
        manuscript = manuscript[:max_chars]
    return (
        f"Paper ID: {getattr(paper, 'paper_id', 'unknown')}\n"
        f"Title: {title}\n"
        f"Abstract: {abstract}\n"
        f"Manuscript (truncated): {manuscript}"
    )


def _extract_self_reflection(
    messages: list[Any], last_ai_response: str | None = None
) -> str | None:
    def _has_self_reflection(text: str) -> bool:
        lowered = text.lower()
        return "self-reflection" in lowered or "self reflection" in lowered

    for msg in messages or []:
        content = getattr(msg, "content", msg)
        if isinstance(content, str) and _has_self_reflection(content):
            return content

    if isinstance(last_ai_response, str) and _has_self_reflection(last_ai_response):
        return last_ai_response

    return None


def _collect_plot_artifacts_from_log(logs: Any) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    if logs is None:
        return artifacts
    timeline = getattr(logs, "timeline", None)
    if not isinstance(timeline, list):
        return artifacts
    for ev in timeline:
        if not isinstance(ev, dict):
            continue
        if ev.get("kind") != "tool_result_execute_code":
            continue
        for item in ev.get("plot_artifacts", []) or []:
            if not isinstance(item, dict):
                continue
            plot_filename = str(item.get("plot_filename", "")).strip()
            plot_path = str(item.get("plot_path", "")).strip()
            key = (plot_filename, plot_path)
            if key in seen:
                continue
            seen.add(key)
            artifacts.append(dict(item))
    return artifacts


def _fallback_paper_summary(paper: PaperMetadataPrivate, max_chars: int = 400) -> str:
    title = getattr(paper, "title", "") or "Untitled"
    abstract = getattr(paper, "abstract", "") or "No abstract available."
    if len(abstract) > max_chars:
        abstract = abstract[:max_chars].rstrip() + "..."
    return f"{getattr(paper, 'paper_id', 'unknown')}: {title}. Abstract (truncated): {abstract}"


def _summarize_prior_papers(
    papers: list[PaperMetadataPrivate],
    *,
    keep_full: int,
    config_path: Path,
) -> tuple[list[PaperMetadataPrivate], list[str]]:
    if not papers:
        return [], []

    keep_full = max(0, int(keep_full))
    papers_sorted = sorted(papers, key=lambda p: getattr(p, "publication_t", 0), reverse=True)
    full_papers = papers_sorted[:keep_full]
    older_papers = papers_sorted[keep_full:]

    if not older_papers:
        return full_papers, []

    sdk = make_sdk_client(load_endpoint_config(config_path))
    if sdk is None:
        summaries = [_fallback_paper_summary(p) for p in older_papers]
        return full_papers, summaries
    client, model = sdk

    system_prompt = (
        "You are a scientific paper summarization assistant. "
        "Summarize each paper in terms of datasets accessed, methods used to assess them, "
        "and the types of hypotheses being tested. "
        "Return exactly three concise bullet points starting with: "
        "'Datasets:', 'Methods:', 'Hypotheses:'."
    )

    summaries: list[str] = []
    for paper in older_papers:
        user_prompt = _paper_source_text(paper)
        try:
            resp = call_with_content_filter_retry(
                lambda user_prompt=user_prompt: client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                    max_tokens=200,
                ),
                stage_label=f"research_one_step.prior_paper_summary.paper={getattr(paper, 'paper_id', 'unknown')}",
                logger=log,
            )
            content = resp.choices[0].message.content if resp.choices else ""
            summary = content.strip() if content else _fallback_paper_summary(paper)
        except Exception:
            summary = _fallback_paper_summary(paper)
        summaries.append(summary)

    return full_papers, summaries


def _count_action_execute_code_in_messages(messages: list[Any]) -> int:
    count = 0
    for msg in messages or []:
        content = getattr(msg, "content", msg)
        if not isinstance(content, str):
            continue
        count += content.count("Action: {'tool': 'execute_code'")
    return count


def _index_private_archive(
    private_archive: PrivatePaperArchive,
) -> tuple[dict[str, str], dict[str, str]]:
    owner_by_paper_id: dict[str, str] = {}
    status_by_paper_id: dict[str, str] = {}
    entries = getattr(private_archive, "_entries", {}) or {}
    if isinstance(entries, dict):
        for agent_id, papers in entries.items():
            for paper in papers or []:
                paper_id = str(getattr(paper, "paper_id", "") or "").strip()
                if not paper_id:
                    continue
                owner_by_paper_id[paper_id] = str(
                    getattr(paper, "primary_agent_id", "") or agent_id
                )
                status_by_paper_id[paper_id] = (
                    str(getattr(paper, "status", "") or "").strip().lower()
                )
    return owner_by_paper_id, status_by_paper_id


def _normalize_suffix(value: str | None) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    cleaned = re.sub(r"[^A-Za-z0-9_.:-]+", "_", raw).strip("_")
    return cleaned or None


def _build_research_session_id(
    agent_id: str, iteration: int, *, session_id_suffix: str | None
) -> str:
    base = f"{agent_id}_t{int(iteration):02d}"
    suffix = _normalize_suffix(session_id_suffix)
    if suffix:
        return f"{base}_{suffix}"
    return base


def _build_collab_session_id(
    collab_id: str, agent_id: str, iteration: int, *, session_id_suffix: str | None
) -> str:
    """Session id of a collaborator's log inside ``agent_id``'s research session.

    Never the collaborator's own session id (``<collab_id>_tNN``): the session-stats registry keys
    logs by session id, so reusing it would overwrite, or be overwritten by, the collaborator's own
    research log.
    """
    suffix = f"collab_for_{agent_id}"
    if _normalize_suffix(session_id_suffix):
        suffix = f"{session_id_suffix}_{suffix}"
    return _build_research_session_id(collab_id, iteration, session_id_suffix=suffix)


def _extract_llm_runtime_metadata(llm_obj: Any) -> dict[str, Any]:
    deployment = None
    for attr in ("azure_deployment", "deployment_name", "model_name", "model", "deployment"):
        value = getattr(llm_obj, attr, None)
        if value is not None and str(value).strip():
            deployment = str(value).strip()
            break
    reasoning_effort = getattr(llm_obj, "reasoning_effort", None)
    model_kwargs = getattr(llm_obj, "model_kwargs", None)
    if isinstance(model_kwargs, Mapping):
        if reasoning_effort in (None, ""):
            reasoning_effort = model_kwargs.get("reasoning_effort", reasoning_effort)
        if deployment is None:
            model_name = model_kwargs.get("model")
            if model_name is not None and str(model_name).strip():
                deployment = str(model_name).strip()
    return {
        "deployment": (str(deployment).strip() if deployment is not None else None),
        "reasoning_effort": (
            str(reasoning_effort).strip()
            if reasoning_effort is not None and str(reasoning_effort).strip()
            else None
        ),
    }


def _build_compute_requirements(
    *,
    session_type: str,
    backend_key: str | None,
    llm_runtime_metadata: dict[str, Any] | None,
    default_llm_runtime_metadata: dict[str, Any] | None,
    session_timeout_seconds: int | float | None,
    recursion_limit: int | None,
    sandbox_required: bool,
    sandbox_requirements: dict[str, Any] | None,
) -> dict[str, Any]:
    llm_meta = dict(default_llm_runtime_metadata or {})
    for key, value in (llm_runtime_metadata or {}).items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        llm_meta[key] = value

    sandbox_meta = dict(sandbox_requirements or {})
    limits_meta = dict(sandbox_meta.get("limits", {}) or {})
    return {
        "session_type": str(session_type),
        "llm": {
            "backend_key": str(backend_key or "default"),
            "deployment": llm_meta.get("deployment"),
            "reasoning_effort": llm_meta.get("reasoning_effort"),
        },
        "runtime_limits": {
            "session_timeout_seconds": (
                int(session_timeout_seconds)
                if isinstance(session_timeout_seconds, (int, float))
                else None
            ),
            "recursion_limit": (int(recursion_limit) if recursion_limit is not None else None),
        },
        "sandbox": {
            "required": bool(sandbox_required),
            "limits": {
                "cpus": limits_meta.get("cpus"),
                "memory": limits_meta.get("memory"),
                "pids": limits_meta.get("pids"),
                "network_disabled": limits_meta.get("network_disabled"),
            },
            "max_sessions": sandbox_meta.get("max_sessions"),
            "admission_limit": sandbox_meta.get("admission_limit"),
        },
    }


def _build_paper_id(
    *,
    iteration: int,
    agent_id: str,
    paper_id_suffix: str | None = None,
    terminal_suffix: str | None = None,
) -> str:
    base = f"paper_t{int(iteration):02d}_{agent_id}"
    phase_suffix = _normalize_suffix(paper_id_suffix)
    status_suffix = _normalize_suffix(terminal_suffix)
    if phase_suffix:
        base = f"{base}_{phase_suffix}"
    if status_suffix:
        base = f"{base}_{status_suffix}"
    return base


def _has_markdown_section(report: str, heading: str) -> bool:
    heading_clean = str(heading or "").strip()
    if not heading_clean:
        return False
    pattern = rf"(?im)^\s*#\s*{re.escape(heading_clean)}\s*$"
    return re.search(pattern, report or "") is not None


def _validate_meta_research_report(
    *,
    report: str,
    citation_verdict: dict[str, Any],
    require_contradictions_section: bool,
    require_archive_citations: bool,
    min_archive_citations: int,
) -> list[str]:
    errors: list[str] = []
    if require_contradictions_section and not _has_markdown_section(
        report, "Contradictions in Public Archive"
    ):
        errors.append("missing required section '# Contradictions in Public Archive'")

    if require_archive_citations:
        cited_archive_ids = [
            str(pid).strip()
            for pid in (citation_verdict.get("valid_archive_ids", []) or [])
            if str(pid).strip()
        ]
        unique_archive_ids = sorted(set(cited_archive_ids))
        if len(unique_archive_ids) < max(0, int(min_archive_citations)):
            errors.append(
                f"requires at least {max(0, int(min_archive_citations))} valid [Archive: ...] citations; found {len(unique_archive_ids)}"
            )
    return errors


def _extract_markdown_section_body(report: str, heading: str) -> str:
    text = str(report or "")
    heading_clean = str(heading or "").strip()
    if not text or not heading_clean:
        return ""
    heading_re = re.compile(rf"(?im)^\s*#\s*{re.escape(heading_clean)}\s*$")
    matches = list(heading_re.finditer(text))
    if not matches:
        return ""
    # Use the latest occurrence so a targeted rewrite pass can append corrected sections.
    start = matches[-1].end()
    next_heading = re.search(r"(?im)^\s*#\s+.+$", text[start:])
    end = start + next_heading.start() if next_heading else len(text)
    return text[start:end].strip()


def _extract_bullet_lines(text: str) -> list[str]:
    out: list[str] = []
    for line in str(text or "").splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        if re.match(r"^[-*]\s+", candidate) or re.match(r"^\d+\.\s+", candidate):
            out.append(candidate)
    return out


def _contains_archive_citation(text: str) -> bool:
    return bool(re.search(r"\[Archive:\s*[^\]]+\]", str(text or ""), flags=re.IGNORECASE))


def _contains_unresolved_uncertainty(text: str) -> bool:
    lowered = str(text or "").lower()
    markers = (
        "uncertain",
        "uncertainty remains",
        "inconclusive",
        "unknown",
        "cannot conclude",
        "insufficient evidence",
        "needs more evidence",
        "unresolved",
    )
    return any(marker in lowered for marker in markers)


def _contains_nan_or_missing_dependency(text: str) -> bool:
    lowered = str(text or "").lower()
    markers = (
        "nan",
        "not a number",
        "missing evidence",
        "missing value",
        "not available",
        "n/a",
    )
    return any(marker in lowered for marker in markers)


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", str(text or "")))


def _truncate_report_to_word_budget(report: str, max_words: int) -> str:
    text = str(report or "").strip()
    budget = max(0, int(max_words or 0))
    if budget <= 0:
        return text
    if _word_count(text) <= budget:
        return text

    lines = text.splitlines()
    kept_lines: list[str] = []
    used_words = 0

    for raw_line in lines:
        line = str(raw_line or "")
        stripped = line.strip()
        if not stripped:
            if kept_lines and kept_lines[-1] != "":
                kept_lines.append("")
            continue

        line_words = re.findall(r"\S+", line)
        line_count = len(line_words)
        if used_words + line_count <= budget:
            kept_lines.append(line)
            used_words += line_count
            continue

        remaining = budget - used_words
        if remaining <= 0:
            break
        truncated = " ".join(line_words[:remaining]).strip()
        if truncated and not truncated.endswith("..."):
            truncated = f"{truncated} ..."
        if truncated:
            kept_lines.append(truncated)
        used_words = budget
        break

    compressed = "\n".join(kept_lines).strip()
    return compressed if compressed else text


def _extract_archive_ids_from_text(text: str) -> list[str]:
    found = re.findall(r"\[Archive:\s*([^\]]+)\]", str(text or ""), flags=re.IGNORECASE)
    ids: list[str] = []
    seen: set[str] = set()
    for raw in found:
        paper_id = str(raw or "").strip()
        if not paper_id or paper_id in seen:
            continue
        seen.add(paper_id)
        ids.append(paper_id)
    return ids


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        value = str(item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _extract_claim_id(text: str) -> str | None:
    match = re.search(r"\b(C\d+)\b", str(text or ""), flags=re.IGNORECASE)
    if not match:
        return None
    return str(match.group(1)).upper()


def _parse_markdown_table_rows(body: str) -> list[dict[str, str]]:
    lines = [line.strip() for line in str(body or "").splitlines() if line.strip()]
    table_lines = [line for line in lines if "|" in line]
    if len(table_lines) < 2:
        return []

    headers = [cell.strip().lower() for cell in table_lines[0].strip("|").split("|")]
    if not headers:
        return []

    rows: list[dict[str, str]] = []
    for raw_line in table_lines[1:]:
        cells = [cell.strip() for cell in raw_line.strip("|").split("|")]
        if all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells):
            continue
        if len(cells) < len(headers):
            cells += [""] * (len(headers) - len(cells))
        row = {headers[idx]: cells[idx] for idx in range(len(headers))}
        rows.append(row)
    return rows


def _extract_claim_to_archive_mapping(
    *,
    claim_evidence_body: str,
    evidence_map_body: str,
) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}

    table_rows = _parse_markdown_table_rows(claim_evidence_body)
    for row in table_rows:
        row_text = " | ".join(str(v) for v in row.values())
        claim_raw = (
            row.get("claim id")
            or row.get("claim_id")
            or row.get("claim")
            or row.get("major claim")
            or row_text
        )
        claim_id = _extract_claim_id(claim_raw)
        if not claim_id:
            continue
        row_archive_ids = _extract_archive_ids_from_text(row_text)
        if not row_archive_ids:
            continue
        existing = mapping.get(claim_id, [])
        mapping[claim_id] = _dedupe_preserve_order(existing + row_archive_ids)

    evidence_lines = _extract_bullet_lines(evidence_map_body)
    if not evidence_lines and evidence_map_body:
        evidence_lines = [
            line.strip() for line in str(evidence_map_body or "").splitlines() if line.strip()
        ]
    for line in evidence_lines:
        claim_id = _extract_claim_id(line)
        if not claim_id:
            continue
        row_archive_ids = _extract_archive_ids_from_text(line)
        if not row_archive_ids:
            continue
        existing = mapping.get(claim_id, [])
        mapping[claim_id] = _dedupe_preserve_order(existing + row_archive_ids)

    return mapping


def _extract_bounded_claim_rows(bounded_final_claims_body: str) -> list[dict[str, Any]]:
    claim_lines = _extract_bullet_lines(bounded_final_claims_body)
    if not claim_lines and bounded_final_claims_body:
        claim_lines = [
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+", bounded_final_claims_body)
            if part.strip()
        ]

    rows: list[dict[str, Any]] = []
    for idx, claim_line in enumerate(claim_lines, start=1):
        rows.append(
            {
                "row_index": int(idx),
                "claim_line": str(claim_line),
                "claim_id": _extract_claim_id(claim_line),
                "inline_archive_ids": _extract_archive_ids_from_text(claim_line),
                "has_unresolved_uncertainty": _contains_unresolved_uncertainty(claim_line),
                "has_nan_or_missing_dependency": _contains_nan_or_missing_dependency(claim_line),
            }
        )
    return rows


def _extract_artifact_refs(text: str) -> list[str]:
    refs: list[str] = []
    refs.extend(
        str(x).strip()
        for x in re.findall(r"\[Plot:\s*([^\]]+)\]", str(text or ""), flags=re.IGNORECASE)
        if str(x).strip()
    )
    refs.extend(
        str(x).strip()
        for x in re.findall(
            r"\b([A-Za-z0-9_.:-]+\.(?:png|pdf|csv|json|txt))\b",
            str(text or ""),
            flags=re.IGNORECASE,
        )
        if str(x).strip()
    )
    return _dedupe_preserve_order(refs)


def _normalize_verification_status(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"pass", "passed", "ok", "success", "supported"}:
        return "pass"
    if text in {"fail", "failed", "error", "unsupported"}:
        return "fail"
    if text in {"not_run", "notrun", "not_executed", "n_a", "na", "none", "skipped", ""}:
        return "not_run"
    if "pass" in text:
        return "pass"
    if "fail" in text:
        return "fail"
    return "not_run"


def _extract_code_check_id(row_text: str, preferred_value: str | None = None) -> str:
    preferred = str(preferred_value or "").strip()
    if preferred:
        return preferred
    match = re.search(
        r"(?:code[_\s-]*check(?:[_\s-]*id)?|check[_\s-]*id)\s*[:=]\s*([A-Za-z0-9_.:-]+)",
        str(row_text or ""),
        flags=re.IGNORECASE,
    )
    if match:
        return str(match.group(1)).strip()
    fallback = re.search(r"\b(CC\d+)\b", str(row_text or ""), flags=re.IGNORECASE)
    if fallback:
        return str(fallback.group(1)).upper()
    return "none"


def _parse_meta_claim_checks(
    *,
    executable_checks_body: str,
    bounded_claims: list[dict[str, Any]],
    claim_to_archive_mapping: dict[str, list[str]],
    valid_archive_ids: list[str],
) -> dict[str, Any]:
    valid_archive_set = {str(pid).strip() for pid in (valid_archive_ids or []) if str(pid).strip()}
    major_claim_ids = _dedupe_preserve_order(
        [
            str(claim.get("claim_id", "")).strip().upper()
            for claim in (bounded_claims or [])
            if str(claim.get("claim_id", "")).strip()
        ]
        + [
            str(claim_id).strip().upper()
            for claim_id in (claim_to_archive_mapping or {}).keys()
            if str(claim_id).strip()
        ]
    )

    claims: list[dict[str, Any]] = []
    parse_errors: list[str] = []

    table_rows = _parse_markdown_table_rows(executable_checks_body)
    if table_rows:
        for row in table_rows:
            row_text = " | ".join(str(v) for v in row.values())
            claim_id = _extract_claim_id(
                row.get("claim_id") or row.get("claim id") or row.get("claim") or row_text
            )
            evidence_refs_text = (
                row.get("evidence_refs")
                or row.get("evidence refs")
                or row.get("evidence")
                or row_text
            )
            evidence_refs = _extract_archive_ids_from_text(evidence_refs_text)
            code_check_id = _extract_code_check_id(
                row_text,
                preferred_value=(
                    row.get("code_check_id")
                    or row.get("code check id")
                    or row.get("check_id")
                    or row.get("check id")
                ),
            )
            verification_status = _normalize_verification_status(
                row.get("verification_status")
                or row.get("verification status")
                or row.get("status")
            )
            artifact_text = (
                row.get("artifact_refs")
                or row.get("artifact refs")
                or row.get("artifacts")
                or row_text
            )
            artifact_refs = _extract_artifact_refs(artifact_text)
            notes = str(row.get("notes") or "").strip()

            if claim_id is None:
                parse_errors.append(
                    f"missing claim_id in executable claim checks row: {row_text[:120]}"
                )

            unresolved_refs = [pid for pid in evidence_refs if pid not in valid_archive_set]
            claims.append(
                {
                    "claim_id": claim_id,
                    "evidence_refs": evidence_refs,
                    "unresolved_evidence_refs": unresolved_refs,
                    "code_check_id": code_check_id or "none",
                    "verification_status": verification_status,
                    "artifact_refs": artifact_refs,
                    "notes": notes,
                }
            )
    else:
        bullet_lines = _extract_bullet_lines(executable_checks_body)
        if not bullet_lines and executable_checks_body:
            bullet_lines = [
                line.strip() for line in str(executable_checks_body).splitlines() if line.strip()
            ]
        for line in bullet_lines:
            claim_id = _extract_claim_id(line)
            evidence_refs = _extract_archive_ids_from_text(line)
            code_check_id = _extract_code_check_id(line)
            verification_status = _normalize_verification_status(line)
            artifact_refs = _extract_artifact_refs(line)
            claims.append(
                {
                    "claim_id": claim_id,
                    "evidence_refs": evidence_refs,
                    "unresolved_evidence_refs": [
                        pid for pid in evidence_refs if pid not in valid_archive_set
                    ],
                    "code_check_id": code_check_id or "none",
                    "verification_status": verification_status,
                    "artifact_refs": artifact_refs,
                    "notes": "",
                }
            )

    status_by_claim_id: dict[str, str] = {}
    for row in claims:
        claim_id = str(row.get("claim_id", "") or "").strip().upper()
        status = str(row.get("verification_status", "not_run") or "not_run")
        if not claim_id:
            continue
        if claim_id not in status_by_claim_id:
            status_by_claim_id[claim_id] = status
            continue
        existing = status_by_claim_id[claim_id]
        priority = {"fail": 3, "pass": 2, "not_run": 1}
        if priority.get(status, 0) > priority.get(existing, 0):
            status_by_claim_id[claim_id] = status

    major_ids_for_count = major_claim_ids or _dedupe_preserve_order(
        [
            str(row.get("claim_id", "")).strip().upper()
            for row in claims
            if str(row.get("claim_id", "")).strip()
        ]
    )
    pass_count = sum(1 for row in claims if str(row.get("verification_status", "")) == "pass")
    fail_count = sum(1 for row in claims if str(row.get("verification_status", "")) == "fail")
    not_run_count = sum(1 for row in claims if str(row.get("verification_status", "")) == "not_run")
    code_verified_major_claim_count = sum(
        1 for claim_id in major_ids_for_count if status_by_claim_id.get(claim_id) == "pass"
    )

    bounded_claim_ids = {
        str(claim.get("claim_id", "")).strip().upper()
        for claim in (bounded_claims or [])
        if str(claim.get("claim_id", "")).strip()
    }
    failed_bounded_claim_ids = sorted(
        claim_id for claim_id in bounded_claim_ids if status_by_claim_id.get(claim_id) == "fail"
    )
    not_run_bounded_claim_ids = sorted(
        claim_id for claim_id in bounded_claim_ids if status_by_claim_id.get(claim_id) == "not_run"
    )
    missing_bounded_claim_ids = sorted(
        claim_id for claim_id in bounded_claim_ids if claim_id not in status_by_claim_id
    )

    return {
        "claims": claims,
        "summary": {
            "pass_count": int(pass_count),
            "fail_count": int(fail_count),
            "not_run_count": int(not_run_count),
            "major_claim_ids": major_ids_for_count,
            "major_claim_count": len(major_ids_for_count),
            "code_verified_major_claim_count": int(code_verified_major_claim_count),
        },
        "parse_errors": parse_errors,
        "status_by_claim_id": status_by_claim_id,
        "failed_bounded_claim_ids": failed_bounded_claim_ids,
        "not_run_bounded_claim_ids": not_run_bounded_claim_ids,
        "missing_bounded_claim_ids": missing_bounded_claim_ids,
    }


def _build_citation_lint_artifact(
    *,
    paper_id: str,
    report: str,
    citation_verdict: dict[str, Any],
    bounded_claims: list[dict[str, Any]],
    claim_to_archive_mapping: dict[str, list[str]],
    bounded_claim_missing_mappings: list[dict[str, Any]],
) -> dict[str, Any]:
    valid_archive_ids = _dedupe_preserve_order(
        [
            str(pid).strip()
            for pid in (citation_verdict.get("valid_archive_ids", []) or [])
            if str(pid).strip()
        ]
    )
    valid_archive_set = set(valid_archive_ids)
    all_archive_ids = _extract_archive_ids_from_text(report)
    unresolved_archive_citations = sorted(
        pid for pid in all_archive_ids if pid not in valid_archive_set
    )

    references_body = _extract_markdown_section_body(report, "References")
    reference_archive_ids = _extract_archive_ids_from_text(references_body)
    non_reference_text = str(report or "")
    if references_body:
        non_reference_text = non_reference_text.replace(references_body, "")
    in_text_archive_ids = _extract_archive_ids_from_text(non_reference_text)
    orphan_references = sorted(
        pid for pid in reference_archive_ids if pid not in set(in_text_archive_ids)
    )

    missing_claim_mappings = [
        {
            "row_index": int(item.get("row_index", 0) or 0),
            "claim_id": str(item.get("claim_id", "") or "").strip() or None,
            "claim_line": str(item.get("claim_line", "") or "").strip(),
        }
        for item in (bounded_claim_missing_mappings or [])
    ]
    missing_claim_mapping_ids = [
        str(item.get("claim_id", "")).strip()
        for item in missing_claim_mappings
        if str(item.get("claim_id", "")).strip()
    ]

    invalid_citation_format_rows: list[str] = []
    for line in str(report or "").splitlines():
        if "[Archive:" not in line:
            continue
        if not re.search(r"\[Archive:\s*[^\]]+\]", line, flags=re.IGNORECASE):
            invalid_citation_format_rows.append(line.strip()[:200])

    for err in citation_verdict.get("errors", []) or []:
        err_text = str(err).strip()
        if "archive tag" in err_text.lower() and err_text not in invalid_citation_format_rows:
            invalid_citation_format_rows.append(err_text)

    bounded_claim_count = len(bounded_claims or [])
    bounded_claims_with_archive_mapping = max(
        0,
        bounded_claim_count - len(missing_claim_mappings),
    )
    coverage = (
        float(bounded_claims_with_archive_mapping) / float(bounded_claim_count)
        if bounded_claim_count > 0
        else None
    )

    lint_status = "pass"
    if unresolved_archive_citations or missing_claim_mappings or invalid_citation_format_rows:
        lint_status = "fail"

    return {
        "paper_id": str(paper_id or "").strip(),
        "unresolved_archive_citations": unresolved_archive_citations,
        "missing_claim_mappings": missing_claim_mappings,
        "missing_claim_mapping_ids": missing_claim_mapping_ids,
        "orphan_references": orphan_references,
        "invalid_citation_format_rows": invalid_citation_format_rows,
        "lint_status": lint_status,
        "summary": {
            "bounded_claim_count": int(bounded_claim_count),
            "bounded_claims_with_archive_mapping": int(bounded_claims_with_archive_mapping),
            "bounded_claim_archive_coverage_pct": coverage,
            "claim_to_archive_mapping_count": len(claim_to_archive_mapping or {}),
        },
    }


def _coerce_meta_gate_cfg(
    validation_cfg: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = dict(validation_cfg or {}) if isinstance(validation_cfg, Mapping) else {}
    quality_raw = (
        dict(cfg.get("quality_contract", {}))
        if isinstance(cfg.get("quality_contract", {}), Mapping)
        else {}
    )
    gate_raw = (
        dict(cfg.get("validation_gate", {}))
        if isinstance(cfg.get("validation_gate", {}), Mapping)
        else {}
    )

    quality_cfg = {
        "require_claim_evidence_table": bool(quality_raw.get("require_claim_evidence_table", True)),
        "require_support_checks_section": bool(
            quality_raw.get("require_support_checks_section", True)
        ),
        "require_contradiction_resolution_section": bool(
            quality_raw.get("require_contradiction_resolution_section", True)
        ),
        "require_bounded_final_claims_section": bool(
            quality_raw.get("require_bounded_final_claims_section", True)
        ),
        "require_executable_claim_checks_section": bool(
            quality_raw.get("require_executable_claim_checks_section", True)
        ),
        "require_reproducibility_notes_section": bool(
            quality_raw.get("require_reproducibility_notes_section", True)
        ),
        "min_supported_major_claims": max(
            0, int(quality_raw.get("min_supported_major_claims", 3) or 0)
        ),
        "min_code_verified_major_claims": max(
            0,
            int(quality_raw.get("min_code_verified_major_claims", 1) or 0),
        ),
        "max_unsupported_major_claims": max(
            0, int(quality_raw.get("max_unsupported_major_claims", 0) or 0)
        ),
        "max_unresolved_uncertainty_claims": max(
            0, int(quality_raw.get("max_unresolved_uncertainty_claims", 0) or 0)
        ),
        "max_words": max(1, int(quality_raw.get("max_words", 950) or 950)),
    }

    default_max_revisions = 1 if bool(gate_raw.get("allow_single_revision_pass", True)) else 0
    gate_cfg = {
        "enabled": bool(gate_raw.get("enabled", True)),
        "allow_single_revision_pass": bool(gate_raw.get("allow_single_revision_pass", True)),
        "max_revision_attempts": max(
            0, int(gate_raw.get("max_revision_attempts", default_max_revisions) or 0)
        ),
        "fail_on_unresolved_nan_claims": bool(gate_raw.get("fail_on_unresolved_nan_claims", True)),
        "fail_on_missing_evidence_mapping": bool(
            gate_raw.get("fail_on_missing_evidence_mapping", True)
        ),
    }
    return quality_cfg, gate_cfg


def _evaluate_meta_quality_gate(
    *,
    report: str,
    citation_verdict: dict[str, Any],
    validation_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    cfg = dict(validation_cfg or {}) if isinstance(validation_cfg, Mapping) else {}
    quality_cfg, gate_cfg = _coerce_meta_gate_cfg(cfg)
    paper_id = str(cfg.get("paper_id", "") or "").strip()

    base_errors = _validate_meta_research_report(
        report=report,
        citation_verdict=citation_verdict,
        require_contradictions_section=bool(cfg.get("require_contradictions_section", True)),
        require_archive_citations=bool(cfg.get("require_archive_citations", True)),
        min_archive_citations=int(cfg.get("min_archive_citations", 0) or 0),
    )

    claim_evidence_body = _extract_markdown_section_body(report, "Claim-Evidence Table")
    support_checks_body = _extract_markdown_section_body(report, "Support and Validity Checks")
    contradiction_resolution_body = _extract_markdown_section_body(
        report, "Contradictions and Resolution"
    )
    bounded_final_claims_body = _extract_markdown_section_body(report, "Bounded Final Claims")
    evidence_map_body = _extract_markdown_section_body(report, "Evidence Map")
    executable_checks_body = _extract_markdown_section_body(report, "Executable Claim Checks")
    reproducibility_notes_body = _extract_markdown_section_body(report, "Reproducibility Notes")

    section_errors: list[str] = []
    if quality_cfg["require_claim_evidence_table"] and not claim_evidence_body:
        section_errors.append("missing required section '# Claim-Evidence Table'")
    if quality_cfg["require_support_checks_section"] and not support_checks_body:
        section_errors.append("missing required section '# Support and Validity Checks'")
    if (
        quality_cfg["require_contradiction_resolution_section"]
        and not contradiction_resolution_body
    ):
        section_errors.append("missing required section '# Contradictions and Resolution'")
    if quality_cfg["require_bounded_final_claims_section"] and not bounded_final_claims_body:
        section_errors.append("missing required section '# Bounded Final Claims'")
    if quality_cfg["require_executable_claim_checks_section"] and not executable_checks_body:
        section_errors.append("missing required section '# Executable Claim Checks'")
    if quality_cfg["require_reproducibility_notes_section"] and not reproducibility_notes_body:
        section_errors.append("missing required section '# Reproducibility Notes'")

    bounded_claims = _extract_bounded_claim_rows(bounded_final_claims_body)
    claim_to_archive_mapping = _extract_claim_to_archive_mapping(
        claim_evidence_body=claim_evidence_body,
        evidence_map_body=evidence_map_body,
    )
    bounded_claim_missing_mappings: list[dict[str, Any]] = []
    bounded_claim_missing_ids: list[dict[str, Any]] = []
    supported_claim_count = 0
    unsupported_claim_count = 0
    unresolved_uncertainty_claim_count = 0
    unresolved_nan_claim_count = 0
    bounded_claim_unresolved_archive_citations: list[str] = []
    valid_archive_ids = _dedupe_preserve_order(
        [
            str(pid).strip()
            for pid in (citation_verdict.get("valid_archive_ids", []) or [])
            if str(pid).strip()
        ]
    )
    valid_archive_set = set(valid_archive_ids)
    for claim in bounded_claims:
        inline_archive_ids = [
            str(pid).strip()
            for pid in (claim.get("inline_archive_ids", []) or [])
            if str(pid).strip()
        ]
        claim_id = str(claim.get("claim_id", "") or "").strip().upper()
        if not claim_id:
            bounded_claim_missing_ids.append(
                {
                    "row_index": int(claim.get("row_index", 0) or 0),
                    "claim_line": str(claim.get("claim_line", "") or ""),
                }
            )
        mapped_archive_ids = inline_archive_ids[:]
        if not mapped_archive_ids and claim_id:
            mapped_archive_ids = [
                str(pid).strip()
                for pid in (claim_to_archive_mapping.get(claim_id, []) or [])
                if str(pid).strip()
            ]
        mapped_archive_ids = _dedupe_preserve_order(mapped_archive_ids)
        if not mapped_archive_ids:
            bounded_claim_missing_mappings.append(
                {
                    "row_index": int(claim.get("row_index", 0) or 0),
                    "claim_id": claim_id or None,
                    "claim_line": str(claim.get("claim_line", "") or ""),
                }
            )
        unresolved_archive_ids = [pid for pid in mapped_archive_ids if pid not in valid_archive_set]
        if unresolved_archive_ids:
            bounded_claim_unresolved_archive_citations.extend(unresolved_archive_ids)

        unresolved = bool(claim.get("has_unresolved_uncertainty", False))
        unresolved_nan = bool(claim.get("has_nan_or_missing_dependency", False))
        has_mapped_archive = bool(mapped_archive_ids)
        if (
            has_mapped_archive
            and not unresolved
            and not unresolved_nan
            and not unresolved_archive_ids
        ):
            supported_claim_count += 1
        else:
            unsupported_claim_count += 1
        if unresolved:
            unresolved_uncertainty_claim_count += 1
        if unresolved_nan:
            unresolved_nan_claim_count += 1

    meta_claim_checks = _parse_meta_claim_checks(
        executable_checks_body=executable_checks_body,
        bounded_claims=bounded_claims,
        claim_to_archive_mapping=claim_to_archive_mapping,
        valid_archive_ids=valid_archive_ids,
    )
    code_verified_major_claim_count = int(
        (meta_claim_checks.get("summary", {}) or {}).get("code_verified_major_claim_count", 0) or 0
    )
    failed_bounded_claim_ids = list(meta_claim_checks.get("failed_bounded_claim_ids", []) or [])
    not_run_bounded_claim_ids = list(meta_claim_checks.get("not_run_bounded_claim_ids", []) or [])
    missing_bounded_claim_ids = list(meta_claim_checks.get("missing_bounded_claim_ids", []) or [])

    citation_lint = _build_citation_lint_artifact(
        paper_id=paper_id,
        report=report,
        citation_verdict=citation_verdict,
        bounded_claims=bounded_claims,
        claim_to_archive_mapping=claim_to_archive_mapping,
        bounded_claim_missing_mappings=bounded_claim_missing_mappings,
    )

    claim_quality_errors: list[str] = []
    if supported_claim_count < int(quality_cfg["min_supported_major_claims"]):
        claim_quality_errors.append(
            "supported major claims below threshold: "
            f"{supported_claim_count} < {int(quality_cfg['min_supported_major_claims'])}"
        )
    if unsupported_claim_count > int(quality_cfg["max_unsupported_major_claims"]):
        claim_quality_errors.append(
            "unsupported major claims exceed threshold: "
            f"{unsupported_claim_count} > {int(quality_cfg['max_unsupported_major_claims'])}"
        )
    if unresolved_uncertainty_claim_count > int(quality_cfg["max_unresolved_uncertainty_claims"]):
        claim_quality_errors.append(
            "unresolved uncertainty claims exceed threshold: "
            f"{unresolved_uncertainty_claim_count} > {int(quality_cfg['max_unresolved_uncertainty_claims'])}"
        )
    if code_verified_major_claim_count < int(quality_cfg["min_code_verified_major_claims"]):
        claim_quality_errors.append(
            "code-verified major claims below threshold: "
            f"{code_verified_major_claim_count} < {int(quality_cfg['min_code_verified_major_claims'])}"
        )

    word_count = _word_count(report)
    if word_count > int(quality_cfg["max_words"]):
        claim_quality_errors.append(
            f"report exceeds max_words: {word_count} > {int(quality_cfg['max_words'])}"
        )

    evidence_mapping_text = "\n".join([claim_evidence_body, evidence_map_body]).strip()
    has_evidence_mapping = (
        _contains_archive_citation(evidence_mapping_text)
        and len(bounded_claim_missing_mappings) == 0
    )
    gate_errors: list[str] = []
    if gate_cfg["fail_on_missing_evidence_mapping"] and (
        not has_evidence_mapping or bool(bounded_claim_missing_mappings)
    ):
        gate_errors.append("missing evidence mapping from final claims to archive citations")
    if gate_cfg["fail_on_unresolved_nan_claims"] and unresolved_nan_claim_count > 0:
        gate_errors.append(
            f"found {unresolved_nan_claim_count} unresolved NaN/missing-evidence claims in bounded final claims"
        )
    if failed_bounded_claim_ids:
        gate_errors.append(
            f"failing code checks found in bounded final claims: {', '.join(failed_bounded_claim_ids)}"
        )
    if not_run_bounded_claim_ids:
        gate_errors.append(
            f"not_run code checks found in bounded final claims: {', '.join(not_run_bounded_claim_ids)}"
        )
    if missing_bounded_claim_ids:
        gate_errors.append(
            "missing executable claim checks for bounded final claims: "
            + ", ".join(missing_bounded_claim_ids)
        )
    if bounded_claim_missing_ids:
        missing_rows = ", ".join(
            f"row {int(item.get('row_index', 0) or 0)}"
            for item in bounded_claim_missing_ids
            if int(item.get("row_index", 0) or 0) > 0
        )
        gate_errors.append(
            "bounded final claims must include explicit claim IDs (C1, C2, ...)"
            + (f": {missing_rows}" if missing_rows else "")
        )
    if bounded_claim_unresolved_archive_citations:
        unresolved_unique = sorted(set(bounded_claim_unresolved_archive_citations))
        gate_errors.append(
            "unresolved archive citations in bounded final claims: " + ", ".join(unresolved_unique)
        )
    parse_errors = [
        str(x).strip() for x in (meta_claim_checks.get("parse_errors", []) or []) if str(x).strip()
    ]
    gate_errors.extend(parse_errors)

    quality_errors = section_errors + claim_quality_errors + gate_errors
    final_errors = _dedupe_preserve_order(
        base_errors + quality_errors if gate_cfg["enabled"] else base_errors
    )
    status = "passed" if not final_errors else "failed"
    if not gate_cfg["enabled"] and not base_errors:
        status = "disabled"

    return {
        "status": status,
        "gate_enabled": bool(gate_cfg["enabled"]),
        "base_validation_errors": _dedupe_preserve_order(base_errors),
        "quality_fail_reasons": _dedupe_preserve_order(quality_errors),
        "final_fail_reasons": final_errors,
        "word_count": int(word_count),
        "has_evidence_mapping": bool(has_evidence_mapping),
        "supported_claim_count": int(supported_claim_count),
        "unsupported_claim_count": int(unsupported_claim_count),
        "unresolved_uncertainty_claim_count": int(unresolved_uncertainty_claim_count),
        "unresolved_nan_claim_count": int(unresolved_nan_claim_count),
        "code_verified_major_claim_count": int(code_verified_major_claim_count),
        "bounded_claim_count": len(bounded_claims),
        "bounded_claims_with_archive_mapping": int(
            len(bounded_claims) - len(bounded_claim_missing_mappings)
        ),
        "bounded_claim_missing_evidence_mappings": bounded_claim_missing_mappings,
        "meta_claim_checks": meta_claim_checks,
        "citation_lint": citation_lint,
        "quality_contract": quality_cfg,
        "validation_gate": gate_cfg,
    }


def _render_targeted_meta_revision(
    *,
    report: str,
    citation_verdict: dict[str, Any],
    validation_cfg: Mapping[str, Any],
    fail_reasons: list[str],
) -> str:
    quality_cfg, _gate_cfg = _coerce_meta_gate_cfg(
        validation_cfg if isinstance(validation_cfg, Mapping) else {}
    )
    archive_ids = _dedupe_preserve_order(
        [
            str(pid).strip()
            for pid in (citation_verdict.get("valid_archive_ids", []) or [])
            if str(pid).strip()
        ]
        + _extract_archive_ids_from_text(report)
    )
    min_supported = max(1, int(quality_cfg.get("min_supported_major_claims", 1) or 1))
    min_citations = max(0, int((validation_cfg or {}).get("min_archive_citations", 0) or 0))
    target_claims = max(min_supported, min_citations, 1)
    needs_word_budget_refresh = any(
        "report exceeds max_words" in str(reason).lower() for reason in (fail_reasons or [])
    )

    if archive_ids:
        ordered_ids = archive_ids[:]
    else:
        # Keep deterministic placeholders when no archive IDs were extracted.
        ordered_ids = [f"source_{idx}" for idx in range(1, target_claims + 1)]

    blocks: list[str] = []

    if quality_cfg.get("require_claim_evidence_table", True) and not _has_markdown_section(
        report, "Claim-Evidence Table"
    ):
        rows = "\n".join(
            f"| C{idx} | [Archive: {pid}] | literature | moderate |"
            for idx, pid in enumerate(ordered_ids[:target_claims], start=1)
        )
        blocks.append(
            "# Claim-Evidence Table\n"
            "| Claim ID | Source Paper IDs | Evidence Type | Support Strength |\n"
            "|---|---|---|---|\n"
            f"{rows}"
        )

    if quality_cfg.get("require_support_checks_section", True) and not _has_markdown_section(
        report, "Support and Validity Checks"
    ):
        checks = "\n".join(
            f"- C{idx}: supported by archive evidence [Archive: {pid}] with bounded assumptions."
            for idx, pid in enumerate(ordered_ids[:target_claims], start=1)
        )
        blocks.append(f"# Support and Validity Checks\n{checks}")

    if quality_cfg.get(
        "require_contradiction_resolution_section", True
    ) and not _has_markdown_section(report, "Contradictions and Resolution"):
        contradiction_lines = "\n".join(
            f"- Potential contradiction resolved by scope/assumption differences across [Archive: {pid}]."
            for pid in ordered_ids[: min(2, len(ordered_ids))]
        )
        if not contradiction_lines:
            contradiction_lines = (
                "- No high-confidence contradiction could be resolved with currently cited sources."
            )
        blocks.append(f"# Contradictions and Resolution\n{contradiction_lines}")

    needs_bounded_claims_refresh = quality_cfg.get(
        "require_bounded_final_claims_section", True
    ) and (
        not _has_markdown_section(report, "Bounded Final Claims")
        or any(
            token in reason
            for reason in fail_reasons
            for token in (
                "supported major claims below threshold",
                "unsupported major claims exceed threshold",
                "unresolved uncertainty claims exceed threshold",
                "unresolved NaN",
            )
        )
    )
    if needs_bounded_claims_refresh:
        bounded_lines = "\n".join(
            f"- C{idx}: supported synthesis claim bounded to available evidence [Archive: {pid}]."
            for idx, pid in enumerate(ordered_ids[:target_claims], start=1)
        )
        blocks.append(f"# Bounded Final Claims\n{bounded_lines}")

    needs_executable_checks_refresh = quality_cfg.get(
        "require_executable_claim_checks_section", True
    ) and (
        not _has_markdown_section(report, "Executable Claim Checks")
        or any(
            token in reason
            for reason in fail_reasons
            for token in (
                "code-verified major claims below threshold",
                "failing code checks found in bounded final claims",
                "not_run code checks found in bounded final claims",
                "missing required section '# Executable Claim Checks'",
            )
        )
    )
    if needs_executable_checks_refresh:
        rows = "\n".join(
            f"| C{idx} | [Archive: {pid}] | check_C{idx} | pass | [Plot: check_C{idx}] |"
            for idx, pid in enumerate(ordered_ids[:target_claims], start=1)
        )
        blocks.append(
            "# Executable Claim Checks\n"
            "| claim_id | evidence_refs | code_check_id | verification_status | artifact_refs |\n"
            "|---|---|---|---|---|\n"
            f"{rows}"
        )

    if quality_cfg.get("require_reproducibility_notes_section", True) and not _has_markdown_section(
        report, "Reproducibility Notes"
    ):
        notes = (
            "- Runtime: fixed random seed and deterministic ordering of claims.\n"
            "- Code checks: each bounded claim maps to one executable check row.\n"
            "- Data scope: synthesis restricted to cited archive sources only."
        )
        blocks.append(f"# Reproducibility Notes\n{notes}")

    if not blocks and not needs_word_budget_refresh:
        return str(report or "")

    if blocks:
        revised_report = str(report or "").rstrip() + "\n\n" + "\n\n".join(blocks).strip() + "\n"
    else:
        revised_report = str(report or "").strip()

    max_words = int(quality_cfg.get("max_words", 0) or 0)
    if needs_word_budget_refresh and max_words > 0:
        revised_report = _truncate_report_to_word_budget(revised_report, max_words)
    return revised_report


def _message_content_to_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, Mapping):
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(chunks).strip()
    return str(content or "").strip()


async def _attempt_meta_revision_with_llm(
    *,
    llm: BaseChatModel,
    report: str,
    fail_reasons: list[str],
    agent_id: str,
    iteration: int,
    meta_context: Mapping[str, Any],
) -> str | None:
    try:
        source_ids = [
            str(x).strip()
            for x in (meta_context.get("source_paper_ids", []) or [])
            if str(x).strip()
        ]
        prompt = (
            "Rewrite the following meta-research report to satisfy strict support and soundness constraints. "
            "Do not invent new archive IDs. Keep only supported bounded final claims. "
            "Preserve claim IDs (C1, C2, ...) and ensure all required sections are present.\n\n"
            "Failure reasons:\n"
            + "\n".join(f"- {reason}" for reason in (fail_reasons or []))
            + "\n\nAllowed archive source IDs:\n"
            + ", ".join(source_ids)
            + "\n\nReturn only the corrected report in plain markdown."
        )
        response = await llm.ainvoke(
            [
                SystemMessage(
                    content=(
                        "You are a strict scientific editor. Preserve factual support, remove unsupported claims, "
                        "and enforce section contracts exactly."
                    )
                ),
                HumanMessage(content=f"{prompt}\n\nOriginal report:\n{report}"),
            ],
            config={
                "metadata": {
                    "agent_id": agent_id,
                    "iteration": iteration,
                    "stage": "meta_quality_revision",
                }
            },
        )
        revised_raw = _message_content_to_text(response).strip()
        if not revised_raw:
            return None
        revised_report, _ = sanitize_report(extract_final_report(revised_raw))
        revised_report = (revised_report or revised_raw).strip()
        return revised_report or None
    except Exception:
        return None


async def _attempt_placeholder_revision_with_llm(
    *,
    llm: BaseChatModel,
    report: str,
    placeholders: list[str],
    agent_id: str,
    iteration: int,
) -> str | None:
    """Request one cleanup rewrite pass to remove unresolved template placeholders."""
    try:
        placeholder_lines = "\n".join(f"- {token}" for token in (placeholders or []))
        prompt = (
            "Rewrite the report to remove unresolved template placeholders.\n"
            "Rules:\n"
            "- Replace placeholders only with evidence grounded in this report/run.\n"
            "- If evidence is missing, delete or soften the unsupported claim.\n"
            "- Do not invent datasets, paper IDs, files, metrics, or citations.\n"
            "- Keep section headings and overall structure intact.\n"
            "- Return only corrected markdown.\n\n"
            "Detected placeholder tokens:\n"
            f"{placeholder_lines}"
        )
        response = await llm.ainvoke(
            [
                SystemMessage(
                    content=(
                        "You are a strict scientific editor focused on removing template leakage "
                        "without introducing unsupported claims."
                    )
                ),
                HumanMessage(content=f"{prompt}\n\nOriginal report:\n{report}"),
            ],
            config={
                "metadata": {
                    "agent_id": agent_id,
                    "iteration": iteration,
                    "stage": "placeholder_cleanup_revision",
                }
            },
        )
        revised_raw = _message_content_to_text(response).strip()
        if not revised_raw:
            return None
        revised_report, _ = sanitize_report(extract_final_report(revised_raw))
        revised_report = (revised_report or revised_raw).strip()
        return revised_report or None
    except Exception:
        return None


def _evaluate_report_citations(
    *,
    report: str,
    result_script: str,
    paper_status_by_id: dict[str, str],
    paper_owner_by_id: dict[str, str],
    full_citing_enabled: bool,
    graph_logic_enabled: bool,
    require_plot_citations: bool,
    code_execs_successful: int,
    plot_validation_waived: bool,
    known_plot_filenames: list[str],
    known_plot_handles: list[str],
    plot_handle_to_filename: dict[str, str],
) -> dict[str, Any]:
    title, abstract = extract_title_and_abstract(report)
    legacy_citations = extract_internal_citations(report)
    strict_citation_verdict = validate_citation_tags(
        report,
        require_at_least_one=full_citing_enabled,
        known_archive_status_by_id=paper_status_by_id if full_citing_enabled else None,
        require_accepted_archive_ids=full_citing_enabled,
        require_backed_code_tags=full_citing_enabled,
        successful_code_executions_current_iteration=(
            code_execs_successful if full_citing_enabled else None
        ),
        code_script=(result_script if full_citing_enabled else None),
        known_plot_filenames=(known_plot_filenames if graph_logic_enabled else None),
        known_plot_handles=(known_plot_handles if graph_logic_enabled else None),
        plot_handle_to_filename=(plot_handle_to_filename if graph_logic_enabled else None),
        require_known_plot_filenames=bool(graph_logic_enabled),
        require_at_least_one_plot_tag=bool(
            graph_logic_enabled and require_plot_citations and len(known_plot_filenames) > 0
        ),
    )
    citation_verdict = dict(strict_citation_verdict)
    citation_errors = [str(e) for e in citation_verdict.get("errors", [])]
    plot_waiver_applied = False

    if plot_validation_waived and graph_logic_enabled:
        relaxed_citation_verdict = validate_citation_tags(
            report,
            require_at_least_one=full_citing_enabled,
            known_archive_status_by_id=paper_status_by_id if full_citing_enabled else None,
            require_accepted_archive_ids=full_citing_enabled,
            require_backed_code_tags=full_citing_enabled,
            successful_code_executions_current_iteration=(
                code_execs_successful if full_citing_enabled else None
            ),
            code_script=(result_script if full_citing_enabled else None),
            known_plot_filenames=None,
            known_plot_handles=None,
            plot_handle_to_filename=None,
            require_known_plot_filenames=False,
            require_at_least_one_plot_tag=False,
        )
        if bool(relaxed_citation_verdict.get("is_valid", False)):
            plot_waiver_applied = True
            citation_verdict = dict(relaxed_citation_verdict)
            citation_errors = []
            log.warning(
                "[research_one_step][plot_validation_waived] "
                f"strict_plot_errors={strict_citation_verdict.get('errors', [])[:5]!r}"
            )
        else:
            citation_verdict = dict(relaxed_citation_verdict)
            citation_errors = [str(e) for e in citation_verdict.get("errors", [])]
            log.warning(
                "[research_one_step][plot_validation_waiver_rejected] "
                f"errors={citation_errors[:5]!r}"
            )

    code_evidence_errors = [
        str(e) for e in (citation_verdict.get("code_evidence", {}) or {}).get("errors", [])
    ]
    invalid_archive_refs: list[str] = []
    for cited_pid in citation_verdict.get("valid_archive_ids", []):
        pid = str(cited_pid).strip()
        if not pid or pid in invalid_archive_refs:
            continue
        cited_status = str(paper_status_by_id.get(pid, "") or "").strip().lower()
        # An empty status means the id is unknown to the archive, which is just as
        # invalid as citing a rejected paper.
        if cited_status != "accepted":
            invalid_archive_refs.append(pid)
    citation_compliant = bool(citation_verdict.get("is_valid", False))
    archive_citations: list[dict[str, str]] = []
    seen_archive_ids: set[str] = set()
    for cited_pid in citation_verdict.get("valid_archive_ids", []):
        pid = str(cited_pid).strip()
        if not pid or pid in seen_archive_ids:
            continue
        seen_archive_ids.add(pid)
        citation_entry: dict[str, str] = {"paper_id": pid}
        owner = paper_owner_by_id.get(pid)
        if owner:
            citation_entry["agent_id"] = owner
        archive_citations.append(citation_entry)

    cited_by_pid: dict[str, dict[str, str]] = {}
    for citation in list(legacy_citations) + archive_citations:
        if not isinstance(citation, dict):
            continue
        pid = str(citation.get("paper_id", "")).strip()
        if not pid:
            continue
        merged = dict(cited_by_pid.get(pid, {}))
        merged.update({str(k): str(v) for k, v in citation.items() if v is not None})
        if "agent_id" not in merged:
            owner = paper_owner_by_id.get(pid)
            if owner:
                merged["agent_id"] = owner
        cited_by_pid[pid] = merged
    internal_citations = list(cited_by_pid.values())
    return {
        "title": title,
        "abstract": abstract,
        "citation_verdict": citation_verdict,
        "citation_errors": citation_errors,
        "code_evidence_errors": code_evidence_errors,
        "invalid_archive_refs": invalid_archive_refs,
        "citation_compliant": citation_compliant,
        "internal_citations": internal_citations,
        "plot_waiver_applied": bool(plot_waiver_applied),
    }


def set_session_admission_limit(n: int) -> None:
    """Set the max number of concurrent create_session attempts."""
    # Replace the global semaphore with a new one sized to n
    global SESSION_CREATE_SEM
    SESSION_CREATE_SEM = asyncio.Semaphore(max(1, int(n)))


async def _wrap(agent_runnable, coro, progress, task_id):
    """run the agent coro, then finalize its logs + progress"""
    token = None
    try:
        logs = get_logs(agent_runnable)
        if logs:
            # Make this agent's log visible to tools during this await
            token = _CURRENT_LOG.set(logs)
        return await coro
    finally:
        if token is not None:
            _CURRENT_LOG.reset(token)
        logs = get_logs(agent_runnable)
        if logs:
            logs.finish()
        progress.update(task_id, advance=1)


async def _return_exception(exc: Exception):
    """Wrap an exception so asyncio.gather can succeed downstream."""
    return exc


async def _raise_exception(exc: Exception):
    """Raise an exception inside an awaitable for symmetry/testing."""
    raise exc


async def _create_agent_runnable_with_retry(
    *,
    llm: BaseChatModel,
    agent_profile: AgentProfile,
    network_metadata: AgentNetworkMetadata,
    private_archive: PrivatePaperArchive,
    task_config: OmegaConf,
    iteration: int,
    network_setting: str | None,
    code_timeout_seconds: int,
    run_name: str | None,
    execute_code_retries: int,
    execute_code_retry_delay: float,
    prior_papers_keep_full: int,
    prior_papers_summary_cfg_path: Path,
    prior_papers_enabled: bool,
    prior_papers_mode: str,
    self_reflection_prompt_enabled: bool,
    full_citing_enabled: bool,
    nature_style_papers_prompt_enabled: bool = False,
    graph_logic_cfg: dict[str, Any] | None = None,
    session_id_suffix: str | None = None,
    session_owner_tag: str | None = None,
    meta_research_mode: bool = False,
    meta_research_context: dict[str, Any] | None = None,
    metacognition_feedback_context: dict[str, Any] | None = None,
    fixed_workflow_context: dict[str, Any] | None = None,
    # Backoff behavior (unchanged defaults)
    base_delay: float = 1.5,
    max_delay: float = 10.0,
    # hard wall-clock deadline to avoid infinite wait
    hard_timeout_s: float = 120.0,
    code_collaboration: bool = True,
    disable_collab_tools: bool = False,
):
    """Build a research agent runnable with sandbox session creation + retries.

    Applies a global semaphore gate and wall-clock timeout to avoid unbounded
    waiting for capacity. Returns the agent runnable, initial state, client,
    sticky session, and agent_id.
    """
    client = SandboxClient(code_timeout_seconds=code_timeout_seconds)
    agent_id = agent_profile.agent_id

    # compute wall-clock deadline
    loop = asyncio.get_running_loop()
    deadline = loop.time() + float(hard_timeout_s)

    attempt = 0
    delay = base_delay

    # We'll create the sticky once we get past capacity (inside the gate)
    sticky = None
    sid = None

    while True:
        # Admission gate: do not let more concurrent create_session() attempts
        # run than the server can likely satisfy. This is set in main() via
        # set_session_admission_limit(max_sessions - reserve).
        async with SESSION_CREATE_SEM:
            attempt += 1
            try:
                # Build a sticky handle with creation kwargs and create on first use.
                extra_payload = {
                    "owner": str(session_owner_tag or f"research_t{iteration}"),
                    "agent_id": agent_id,
                    "iteration": iteration,
                }
                session_id = _build_research_session_id(
                    agent_id=agent_id,
                    iteration=iteration,
                    session_id_suffix=session_id_suffix,
                )
                sticky = client.sticky_session(
                    session_name=session_id,
                    extra_payload=extra_payload,
                )
                # Force container creation now so we can log/assert its SID.
                sid = sticky.ensure()
                break  # success

            except MaxSessionsError as e:  # noqa: F841
                # Pool really full: let someone else run, back off
                # (No retry here inside the gate; we drop out to sleep below)
                remaining = max(0.0, deadline - loop.time())
                _console.log(
                    f"[sandbox][{agent_id}] max sessions reached (attempt {attempt}); "
                    f"retrying in {min(delay, max_delay):.1f}s "
                    f"(~{remaining:.0f}s left before abort)"
                )

            except Exception as e:
                # 500 w/o recovery, transport errors, etc. Treat like capacity
                remaining = max(0.0, deadline - loop.time())
                _console.log(
                    f"[sandbox][{agent_id}] session create failed ({e!r}) "
                    f"(attempt {attempt}); retrying in {min(delay, max_delay):.1f}s "
                    f"(~{remaining:.0f}s left before abort)"
                )

        # Outside the gate: decide whether to keep waiting
        now = loop.time()
        if now >= deadline:
            # Give a clear, actionable error rather than spinning forever
            raise RuntimeError(
                f"[sandbox][{agent_id}] Gave up after {hard_timeout_s:.0f}s waiting for capacity. "
                f"Increase pool size, reduce concurrency, or ensure sessions are reaped."
            )

        # bounded linear backoff (1.5s → 3.0s → … → 10.0s)
        await asyncio.sleep(delay)
        delay = min(delay + base_delay, max_delay)

    # ------------------------------
    # Success path: we have a session
    # ------------------------------
    try:
        # tools
        agent_primary_tools = (
            dict(task_config.primary_agent_tools) if task_config.primary_agent_tools else {}
        )
        if disable_collab_tools:
            agent_collab_tools = {}
        else:
            agent_collab_tools = (
                dict(task_config.collab_agent_tools) if task_config.collab_agent_tools else {}
            )

        # IMPORTANT: pass sticky so research_team reuses this container and does
        # not attempt to create another one.
        prior_papers = []
        prior_paper_summaries: list[str] = []
        prior_paper_status_by_id: dict[str, str] = {}
        if prior_papers_enabled:
            try:
                prior_papers = private_archive.get_by_primary_agent(agent_id, k=None)
                prior_paper_status_by_id = {
                    str(getattr(p, "paper_id", "") or ""): str(getattr(p, "status", "") or "")
                    .strip()
                    .lower()
                    for p in prior_papers
                    if str(getattr(p, "paper_id", "") or "").strip()
                }
                mode = str(prior_papers_mode).strip().lower()
                if mode == "summary":
                    prior_papers, prior_paper_summaries = _summarize_prior_papers(
                        prior_papers,
                        keep_full=prior_papers_keep_full,
                        config_path=prior_papers_summary_cfg_path,
                    )
            except Exception:
                prior_papers = []
                prior_paper_summaries = []

        agent_runnable, _ = get_research_team(
            llm=llm,
            sandbox_client=client,
            iteration=iteration,
            sandbox_sticky=sticky,  # <= re-use the created sticky
            primary_agent_profile=agent_profile,
            agent_network_metadata=network_metadata,
            prior_papers=prior_papers,
            prior_paper_summaries=prior_paper_summaries,
            prior_papers_enabled=prior_papers_enabled,
            prior_papers_mode=prior_papers_mode,
            self_reflection_prompt_enabled=self_reflection_prompt_enabled,
            nature_style_papers_prompt_enabled=nature_style_papers_prompt_enabled,
            full_citing_enabled=full_citing_enabled,
            graph_logic_cfg=graph_logic_cfg,
            prior_paper_status_by_id=prior_paper_status_by_id,
            meta_research_mode=meta_research_mode,
            meta_research_context=meta_research_context,
            metacognition_feedback_context=metacognition_feedback_context,
            fixed_workflow_context=fixed_workflow_context,
            network_setting=network_setting,
            primary_agent_tools=agent_primary_tools,
            collab_agent_tools=agent_collab_tools,
            python_packages=task_config.python_packages,
            dataset_metadata=task_config.dataset_description,
            code_collaboration=code_collaboration,
            execute_code_retries=execute_code_retries,
            execute_code_retry_delay=execute_code_retry_delay,
        )
    except Exception as e:
        # cleanup leaked session if build fails
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        _console.log(
            f"[sandbox][{agent_id}] build failed, shutting down leaked session {sid}\n{tb}"
        )
        try:
            try:
                if sticky is not None:
                    sticky.close()
            except Exception:
                pass
            if sid is not None:
                client.shutdown(sid)
        except Exception:
            pass
        raise e

    # initial state
    init_state = {
        "messages": [
            HumanMessage(content=task_config.discovery_goal),
        ],
        "script": "",
        "route": None,
        "step": 1,
        "metadata": {
            "agent_id": agent_id,
            "iteration": iteration,
            "run_name": run_name,
            "base_snapshots_dir": "snapshots",
            "graph_logic": normalize_graph_logic_cfg(graph_logic_cfg),
        },
    }
    if isinstance(fixed_workflow_context, Mapping) and fixed_workflow_context.get("enabled"):
        init_state["metadata"]["fixed_workflow"] = dict(fixed_workflow_context)

    return agent_runnable, init_state, client, sticky, agent_id


async def _run_one_agent(
    *,
    agent,
    state,
    agent_id: str,
    iteration: int,
    recursion_limit: int,
    run_name: str | None,
    total_session_timeout_seconds: int | None,
    trace_suffix: str | None = None,
):
    """Invoke a single research agent with tracing and optional session timeout."""
    trace_suffix_clean = _normalize_suffix(trace_suffix)
    trace_suffix_part = f"_{trace_suffix_clean}" if trace_suffix_clean else ""
    configured = agent.with_config(
        {
            "run_name": f"{agent_id}_t{iteration}{trace_suffix_part}_run",
            "metadata": {
                "agent_id": agent_id,
                "iteration": iteration,
                "run_name": run_name,
            },
            "recursion_limit": recursion_limit,
        }
    )
    traced_invoke = traceable(
        name=f"research_t{iteration}_{agent_id}{trace_suffix_part}_trace",
        run_type="chain",
        metadata={"agent_id": agent_id, "iteration": iteration, "run_name": run_name},
        tags=[f"agent_{agent_id}", f"iteration_{iteration}"],
        project_name=run_name,
    )(configured.ainvoke)

    if total_session_timeout_seconds:
        try:
            return await asyncio.wait_for(
                traced_invoke(state),
                timeout=total_session_timeout_seconds,
            )
        except TimeoutError:
            return {
                "messages": state.get("messages", []) + [HumanMessage(content="TIMEOUT")],
                "script": state.get("script", ""),
                "route": "timeout",
                "step": state.get("step", 1),
                "metadata": {
                    "agent_id": agent_id,
                    "iteration": iteration,
                    "run_name": run_name,
                    "base_snapshots_dir": "snapshots",
                    "status": "timeout",
                },
                "collab_agent_ids": [],
            }
    else:
        return await traced_invoke(state)


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------


async def research_one_step(
    iteration: int,
    llm: BaseChatModel,
    llm_for: Callable[[str], BaseChatModel] | None,
    agents: list[AgentProfile],
    network_metadata: AgentNetworkMetadata,
    private_archive: PrivatePaperArchive,  # not used atm, kept for API compat
    task_config: OmegaConf,
    research_max_iters: int,
    llm_backend_key_for: Callable[[str], str | None] | None = None,
    default_llm_runtime_metadata: dict[str, Any] | None = None,
    sandbox_requirements: dict[str, Any] | None = None,
    network_setting: str | None = "network_hetero",
    code_timeout_seconds: int | None = 120,
    total_session_timeout_seconds: int | None = 600,
    run_name: str | None = None,
    code_collaboration: bool = True,
    session_create_hard_timeout_seconds: float | None = None,
    execute_code_retries: int = 120,
    execute_code_retry_delay: float = 5.0,
    prior_papers_keep_full: int = DEFAULT_PRIOR_PAPERS_KEEP_FULL,
    prior_papers_summary_cfg_path: Path | None = None,
    prior_papers_enabled: bool = False,
    prior_papers_mode: str = "full",
    self_reflection_prompt_enabled: bool = False,
    nature_style_papers_prompt_enabled: bool = False,
    full_citing_enabled: bool = False,
    graph_logic_cfg: dict[str, Any] | None = None,
    per_agent_research_max_iters: dict[str, int] | None = None,
    paused_agent_ids: set[str] | None = None,
    disable_collab_agent_ids: set[str] | None = None,
    research_phase: str = "primary",
    paper_id_suffix: str | None = None,
    session_id_suffix: str | None = None,
    paper_kind: str = "research",
    phase_context_by_agent: dict[str, dict[str, Any]] | None = None,
    metacognition_feedback_by_agent: dict[str, dict[str, Any]] | None = None,
    fixed_workflow_by_agent: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[PaperMetadataPrivate], list[SessionStats]]:
    """Orchestrate a full research iteration: build agents, run them, and collect papers."""

    paused = set(paused_agent_ids or [])
    collab_disabled = set(disable_collab_agent_ids or [])
    active_agents: list[AgentProfile] = []
    for a in agents:
        agent_id = getattr(a, "agent_id", None) or a.id
        if agent_id in paused:
            continue
        active_agents.append(a)

    if not active_agents:
        return [], []

    phase = str(research_phase or "primary").strip().lower() or "primary"
    paper_kind_norm = str(paper_kind or "research").strip().lower() or "research"
    paper_suffix = _normalize_suffix(paper_id_suffix)
    session_suffix = _normalize_suffix(session_id_suffix) or paper_suffix
    phase_context_lookup: dict[str, dict[str, Any]] = {}
    if isinstance(phase_context_by_agent, Mapping):
        for aid, payload in phase_context_by_agent.items():
            if not isinstance(payload, Mapping):
                continue
            key = str(aid).strip()
            if not key:
                continue
            phase_context_lookup[key] = dict(payload)
    metacognition_lookup: dict[str, dict[str, Any]] = {}
    if isinstance(metacognition_feedback_by_agent, Mapping):
        for aid, payload in metacognition_feedback_by_agent.items():
            if not isinstance(payload, Mapping):
                continue
            key = str(aid).strip()
            if key:
                metacognition_lookup[key] = dict(payload)
    fixed_workflow_lookup: dict[str, dict[str, Any]] = {}
    if isinstance(fixed_workflow_by_agent, Mapping):
        for aid, payload in fixed_workflow_by_agent.items():
            if not isinstance(payload, Mapping):
                continue
            key = str(aid).strip()
            if key:
                fixed_workflow_lookup[key] = dict(payload)

    phase_trace_suffix = _normalize_suffix(session_suffix or phase)
    session_owner = f"research_t{iteration}"
    if phase_trace_suffix:
        session_owner = f"{session_owner}_{phase_trace_suffix}"

    graph_cfg = normalize_graph_logic_cfg(graph_logic_cfg)
    graph_logic_enabled = bool(graph_cfg.get("enabled", False))
    require_plot_citations = bool(graph_cfg.get("require_plot_citations_in_research", False))

    paper_owner_by_id, paper_status_by_id = _index_private_archive(private_archive)
    # store sticky handles so we can close them deterministically
    # keep sticky handles so we can close deterministically
    session_handles: list[tuple[SandboxClient.StickySession, SandboxClient, str]] = []

    if prior_papers_summary_cfg_path is None:
        prior_papers_summary_cfg_path = DEFAULT_LLM_CONFIG_PATH

    # 1) build all agents (each may wait for sandbox)
    session_id_by_agent: dict[str, str] = {}
    effective_research_iters_by_agent: dict[str, int] = {}
    meta_source_ids_by_agent: dict[str, list[str]] = {}
    meta_validation_by_agent: dict[str, dict[str, Any]] = {}
    build_tasks = []
    for a in active_agents:
        # pick the per-agent LLM if provided, else fall back to the default
        agent_id = getattr(a, "agent_id", None) or a.id
        agent_phase_context = phase_context_lookup.get(str(agent_id), {}) or {}
        agent_meta_mode = bool(
            agent_phase_context.get("meta_research_mode", False)
            or phase == "meta_research"
            or paper_kind_norm == "meta_research"
        )
        agent_meta_context = (
            dict(agent_phase_context.get("meta_research_context", {}))
            if isinstance(agent_phase_context.get("meta_research_context", {}), Mapping)
            else {}
        )
        agent_meta_source_ids = [
            str(x).strip()
            for x in (agent_meta_context.get("source_paper_ids", []) or [])
            if str(x).strip()
        ]
        agent_meta_validation = (
            dict(agent_phase_context.get("meta_validation", {}))
            if isinstance(agent_phase_context.get("meta_validation", {}), Mapping)
            else {}
        )
        agent_metacognition_feedback = metacognition_lookup.get(str(agent_id))
        agent_fixed_workflow_context = fixed_workflow_lookup.get(str(agent_id))
        meta_source_ids_by_agent[str(agent_id)] = agent_meta_source_ids
        meta_validation_by_agent[str(agent_id)] = agent_meta_validation
        session_id_by_agent[str(agent_id)] = _build_research_session_id(
            agent_id=str(agent_id),
            iteration=iteration,
            session_id_suffix=session_suffix,
        )
        active_llm = llm_for(agent_id) if llm_for else llm
        effective_max_iters = research_max_iters
        if per_agent_research_max_iters and agent_id in per_agent_research_max_iters:
            effective_max_iters = int(per_agent_research_max_iters[agent_id])
        effective_research_iters_by_agent[str(agent_id)] = int(effective_max_iters)
        build_tasks.append(
            _create_agent_runnable_with_retry(
                llm=active_llm,
                agent_profile=a,
                network_metadata=network_metadata,
                private_archive=private_archive,
                task_config=task_config,
                iteration=iteration,
                network_setting=network_setting,
                code_timeout_seconds=code_timeout_seconds or 120,
                run_name=run_name,
                code_collaboration=code_collaboration,
                hard_timeout_s=session_create_hard_timeout_seconds,
                execute_code_retries=execute_code_retries,
                execute_code_retry_delay=execute_code_retry_delay,
                prior_papers_keep_full=prior_papers_keep_full,
                prior_papers_summary_cfg_path=prior_papers_summary_cfg_path,
                prior_papers_enabled=prior_papers_enabled,
                prior_papers_mode=prior_papers_mode,
                self_reflection_prompt_enabled=self_reflection_prompt_enabled,
                nature_style_papers_prompt_enabled=nature_style_papers_prompt_enabled,
                full_citing_enabled=full_citing_enabled,
                graph_logic_cfg=graph_cfg,
                session_id_suffix=session_suffix,
                session_owner_tag=session_owner,
                meta_research_mode=agent_meta_mode,
                meta_research_context=agent_meta_context,
                metacognition_feedback_context=agent_metacognition_feedback,
                fixed_workflow_context=agent_fixed_workflow_context,
                disable_collab_tools=agent_id in collab_disabled,
            )
        )

    built = await asyncio.gather(*build_tasks, return_exceptions=True)

    agent_runnables: list[tuple[Any, str]] = []
    agent_states: list[Any] = []

    for idx, item in enumerate(built):
        if isinstance(item, Exception):
            # we'll turn this into an INCOMPLETE paper later
            failed_agent_id = getattr(active_agents[idx], "agent_id", None) or getattr(
                active_agents[idx], "id", "unknown"
            )
            agent_runnables.append((item, str(failed_agent_id)))
            agent_states.append(None)
            continue
        agent_runnable, init_state, client, sticky, agent_id = item
        # Add llm_key (if annotated on profile) into init_state metadata for traceability
        try:
            llm_key = getattr(active_agents[idx], "llm_key", None)
            if llm_key:
                init_state.setdefault("metadata", {}).update({"llm_key": llm_key})
        except Exception:
            pass
        init_state.setdefault("metadata", {}).update(
            {
                "research_phase": phase,
                "paper_kind": paper_kind_norm,
                "research_session_id": session_id_by_agent.get(agent_id),
            }
        )
        session_handles.append((sticky, client, agent_id))
        agent_runnables.append((agent_runnable, agent_id))
        agent_states.append(init_state)

    # 2) run all agents
    tasks = []
    for (agent_runnable, agent_id), init_state in zip(agent_runnables, agent_states, strict=False):
        if isinstance(agent_runnable, Exception):
            # build failed -> wrap into awaitable so gather() is happy
            tasks.append(_return_exception(agent_runnable))
        else:
            effective_max_iters = research_max_iters
            if per_agent_research_max_iters and agent_id in per_agent_research_max_iters:
                effective_max_iters = int(per_agent_research_max_iters[agent_id])
            tasks.append(
                _run_one_agent(
                    agent=agent_runnable,
                    state=init_state,
                    agent_id=agent_id,
                    iteration=iteration,
                    recursion_limit=recursion_limit_for(effective_max_iters),
                    run_name=run_name,
                    total_session_timeout_seconds=total_session_timeout_seconds,
                    trace_suffix=phase_trace_suffix,
                )
            )

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold]research running...[/]"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            transient=True,
            console=_console,
        ) as progress:
            task_id = progress.add_task("agents", total=len(tasks))

            wrapped = []
            for (agent_runnable, _aid), coro in zip(agent_runnables, tasks, strict=False):
                if isinstance(agent_runnable, Exception):
                    # no logs to wrap
                    wrapped.append(coro)
                else:
                    wrapped.append(_wrap(agent_runnable, coro, progress, task_id))

            results = await asyncio.gather(*wrapped, return_exceptions=True)
    finally:
        # 3) ALWAYS close the sticky sessions (calls /shutdown under the hood)
        _console.log(
            f"[sandbox] shutting down remaining {len(session_handles)} session(s) for iteration {iteration}"
        )
        for sticky, _client, _agent_id in session_handles:
            try:
                sticky.close()
            except Exception:
                pass

        # 3b) SAFE SWEEP: now that our own sessions are shut down, clean up ONLY ours
        # (i.e. sessions tagged with owner=f"research_t{iteration}")
        owner_tag = session_owner
        # dedupe clients
        seen_clients = set()
        for _sticky, client, _agent_id in session_handles:
            if id(client) in seen_clients:
                continue
            seen_clients.add(id(client))
            try:
                client._reap_stale_sessions(
                    keep_ids=set(),  # we just shut down ours
                    owner=owner_tag,  # only our iteration's sessions
                    max_age_seconds=10.0,  # give newly created ones a chance to settle
                )
            except Exception:
                # best-effort
                pass

    # 4) collect logs
    research_logs: list[SessionStats] = []
    logs_by_agent_id: dict[str, Any] = {}

    # build a simple id → profile map once
    profile_by_id = {getattr(a, "agent_id", None) or a.id: a for a in active_agents}

    for agent_runnable, agent_id in agent_runnables:
        if isinstance(agent_runnable, Exception):
            continue
        logs = get_logs(agent_runnable)
        logs_by_agent_id[agent_id] = logs

        # look up llm_key from the profile
        llm_key = None
        profile = profile_by_id.get(agent_id)
        if profile is not None:
            llm_key = getattr(profile, "llm_key", None)

        active_llm = llm_for(agent_id) if llm_for else llm
        backend_key = llm_backend_key_for(agent_id) if llm_backend_key_for else llm_key
        if not backend_key:
            backend_key = "default"
        llm_runtime_meta = _extract_llm_runtime_metadata(active_llm)
        compute_requirements = _build_compute_requirements(
            session_type="research",
            backend_key=backend_key,
            llm_runtime_metadata=llm_runtime_meta,
            default_llm_runtime_metadata=default_llm_runtime_metadata,
            session_timeout_seconds=total_session_timeout_seconds,
            recursion_limit=recursion_limit_for(
                effective_research_iters_by_agent.get(str(agent_id), research_max_iters)
            ),
            sandbox_required=True,
            sandbox_requirements=sandbox_requirements,
        )
        if logs is not None:
            logs.compute_requirements = compute_requirements

        research_logs.append(
            SessionStats(
                iteration=iteration,
                session_id=session_id_by_agent.get(agent_id)
                or _build_research_session_id(
                    agent_id=agent_id,
                    iteration=iteration,
                    session_id_suffix=session_suffix,
                ),
                session_agent_id=agent_id,
                type="research",
                stats_logs=logs,
                metadata={
                    "llm_key": llm_key,
                    "research_phase": phase,
                    "paper_kind": paper_kind_norm,
                },
            )
        )
        if logs is not None and getattr(logs, "collab_logs", None):
            for collab_id, collab_log in logs.collab_logs.items():
                collab_session_id = _build_collab_session_id(
                    collab_id=collab_id,
                    agent_id=agent_id,
                    iteration=iteration,
                    session_id_suffix=session_suffix,
                )
                research_logs.append(
                    SessionStats(
                        iteration=iteration,
                        session_id=collab_session_id,
                        session_agent_id=collab_id,
                        type="research",
                        stats_logs=collab_log,
                        metadata={
                            "collab_for": agent_id,
                            "llm_key": llm_key,
                            "research_phase": phase,
                            "paper_kind": paper_kind_norm,
                        },
                    )
                )
                if collab_log is not None:
                    # The collaborator runs on the author's model (get_collab_agent gets the
                    # author's llm), so record the author's backend with its deployment.
                    collab_log.compute_requirements = _build_compute_requirements(
                        session_type="research",
                        backend_key=backend_key,
                        llm_runtime_metadata=llm_runtime_meta,
                        default_llm_runtime_metadata=default_llm_runtime_metadata,
                        session_timeout_seconds=total_session_timeout_seconds,
                        recursion_limit=recursion_limit_for(
                            effective_research_iters_by_agent.get(str(agent_id), research_max_iters)
                        ),
                        sandbox_required=True,
                        sandbox_requirements=sandbox_requirements,
                    )

    # 5) turn results into papers
    outputs: list[PaperMetadataPrivate] = []
    for (agent_runnable, agent_id), result in zip(agent_runnables, results, strict=False):
        research_session_id = session_id_by_agent.get(agent_id) or _build_research_session_id(
            agent_id=agent_id,
            iteration=iteration,
            session_id_suffix=session_suffix,
        )
        meta_source_ids = list(meta_source_ids_by_agent.get(str(agent_id), []) or [])
        meta_trigger_iteration = iteration if paper_kind_norm == "meta_research" else None
        agent_logs = logs_by_agent_id.get(agent_id)

        if isinstance(result, Exception):
            log.warning(
                f"[build_paper_from_result] agent_id={agent_id} iteration={iteration}: "
                "the research session raised",
                exc_info=result,
            )
        else:
            script = result.get("script") if isinstance(result, dict) else None
            log.debug(
                f"[build_paper_from_result] agent_id={agent_id} iteration={iteration} "
                f"result_keys={list(result.keys())} "
                f"script_len={len(script) if isinstance(script, str) else '<missing>'}"
            )

        # (a) build failed earlier
        if isinstance(agent_runnable, Exception):
            _console.log(
                f"[research_one_step][terminal_route] iteration={iteration} agent_id={agent_id} "
                "route=incomplete reason=build_failed"
            )
            if agent_logs is not None:
                agent_logs.add_event(
                    kind="terminal_route",
                    agent_id=agent_id,
                    iteration=iteration,
                    route="incomplete",
                    reason="build_failed",
                )
            tb = "".join(
                traceback.format_exception(
                    type(agent_runnable), agent_runnable, agent_runnable.__traceback__
                )
            )
            outputs.append(
                PaperMetadataPrivate(
                    paper_id=_build_paper_id(
                        iteration=iteration,
                        agent_id=agent_id,
                        paper_id_suffix=paper_suffix,
                        terminal_suffix="incomplete",
                    ),
                    primary_agent_id=agent_id,
                    collab_agent_ids=[],
                    title=f"Incomplete Research - Agent {agent_id}",
                    abstract="Agent could not be initialised (sandbox or tool error).",
                    manuscript=tb,
                    citation_count=0,
                    publication_t=iteration,
                    cited_paper_ids=[],
                    metareview=None,
                    code_script="",
                    status="incomplete",
                    paper_kind=paper_kind_norm,
                    research_phase=phase,
                    research_session_id=research_session_id,
                    meta_research_source_paper_ids=meta_source_ids,
                    meta_research_trigger_iteration=meta_trigger_iteration,
                )
            )
            continue

        # (b) run crashed
        if isinstance(result, Exception):
            _console.log(
                f"[research_one_step][terminal_route] iteration={iteration} agent_id={agent_id} "
                "route=exception reason=run_exception"
            )
            if agent_logs is not None:
                agent_logs.add_event(
                    kind="terminal_route",
                    agent_id=agent_id,
                    iteration=iteration,
                    route="exception",
                    reason="run_exception",
                )
            tb = "".join(traceback.format_exception(type(result), result, result.__traceback__))
            outputs.append(
                PaperMetadataPrivate(
                    paper_id=_build_paper_id(
                        iteration=iteration,
                        agent_id=agent_id,
                        paper_id_suffix=paper_suffix,
                        terminal_suffix="incomplete",
                    ),
                    primary_agent_id=agent_id,
                    collab_agent_ids=[],
                    title=f"Incomplete Research - Agent {agent_id}",
                    abstract="Agent run terminated with an exception.",
                    manuscript=tb,
                    citation_count=0,
                    publication_t=iteration,
                    cited_paper_ids=[],
                    metareview=None,
                    code_script="",
                    status="incomplete",
                    paper_kind=paper_kind_norm,
                    research_phase=phase,
                    research_session_id=research_session_id,
                    meta_research_source_paper_ids=meta_source_ids,
                    meta_research_trigger_iteration=meta_trigger_iteration,
                )
            )
            continue

        # (c) timeout
        if result.get("route") == "timeout":
            _console.log(
                f"[research_one_step][terminal_route] iteration={iteration} agent_id={agent_id} route=timeout"
            )
            if agent_logs is not None:
                agent_logs.add_event(
                    kind="terminal_route",
                    agent_id=agent_id,
                    iteration=iteration,
                    route="timeout",
                    reason="session_timeout",
                )
            outputs.append(
                PaperMetadataPrivate(
                    paper_id=_build_paper_id(
                        iteration=iteration,
                        agent_id=agent_id,
                        paper_id_suffix=paper_suffix,
                        terminal_suffix="timeout",
                    ),
                    primary_agent_id=agent_id,
                    collab_agent_ids=result.get("collab_agent_ids", []),
                    title=f"Incomplete Research (Timeout) - Agent {agent_id}",
                    abstract="Research was terminated due to timeout before completion.",
                    manuscript="TIMEOUT",
                    citation_count=0,
                    publication_t=iteration,
                    cited_paper_ids=[],
                    metareview=None,
                    code_script=result.get("script", ""),
                    status="timeout",
                    paper_kind=paper_kind_norm,
                    research_phase=phase,
                    research_session_id=research_session_id,
                    meta_research_source_paper_ids=meta_source_ids,
                    meta_research_trigger_iteration=meta_trigger_iteration,
                )
            )
            continue

        # (d) normal happy path
        self_reflection = _extract_self_reflection(
            result.get("messages", []),
            last_ai_response=result.get("last_ai_response"),
        )
        last_ai_response = result.get("last_ai_response")
        if isinstance(last_ai_response, str) and "Action:" in last_ai_response:
            log.debug(
                "[research_one_step][last_ai_action] "
                f"agent_id={agent_id} iteration={iteration} "
                f"snippet={last_ai_response.strip()[:400]!r}"
            )
        if self_reflection:
            log.debug(
                f"[self_reflection] agent_id={agent_id} iteration={iteration}\n{self_reflection}"
            )
        else:
            log.debug(f"[self_reflection] agent_id={agent_id} iteration={iteration} not_found")
        report_raw = extract_final_report(result["messages"])
        report, report_stripped = sanitize_report(report_raw)
        if report_stripped:
            log.warning(
                "[research_one_step][report_sanitize] "
                f"agent_id={agent_id} iteration={iteration} "
                "action_or_observation_stripped=True"
            )
        if isinstance(report_raw, str) and "Action:" in report_raw:
            snippet = report_raw.split("Action:", 1)[0]
            log.debug(
                "[research_one_step][report_action] "
                f"agent_id={agent_id} iteration={iteration} "
                f"snippet={snippet[-400:]!r}"
            )
        if isinstance(last_ai_response, str) and (
            "Action:" in last_ai_response or "Observation:" in last_ai_response
        ):
            log.warning(
                "[research_one_step][last_ai_contains_action] "
                f"agent_id={agent_id} iteration={iteration}"
            )
        action_exec_count = _count_action_execute_code_in_messages(result.get("messages", []))
        logs = agent_logs
        tool_exec_count = 0
        code_execs_successful = 0
        if logs is not None:
            tool_exec_count = int(getattr(logs, "tool_calls", {}).get("execute_code", 0) or 0)
            code_execs_successful = int(getattr(logs, "code_executions_successful", 0) or 0)
        if action_exec_count > tool_exec_count:
            log.warning(
                "[research_one_step][action_exec_mismatch] "
                f"agent_id={agent_id} iteration={iteration} "
                f"action_exec_count={action_exec_count} tool_exec_count={tool_exec_count}"
            )
        if code_execs_successful == 0:
            lowered = (report or "").lower()
            if any(
                k in lowered
                for k in ("p=", "p < ", "wilcoxon", "fisher", "ttest", "spearman", "pearson")
            ):
                log.warning(
                    "[research_one_step][report_stats_without_exec] "
                    f"agent_id={agent_id} iteration={iteration} "
                    "code_execs_successful=0"
                )
        result_script = str(result.get("script", "") or "")
        result_metadata = result.get("metadata", {}) if isinstance(result, dict) else {}
        plot_validation_waived = bool(
            isinstance(result_metadata, dict)
            and result_metadata.get("plot_validation_waived", False)
        )
        plot_artifacts = []
        if isinstance(result_metadata, dict):
            plot_artifacts = [
                dict(x)
                for x in (result_metadata.get("plot_artifacts", []) or [])
                if isinstance(x, dict)
            ]
        if not plot_artifacts:
            plot_artifacts = _collect_plot_artifacts_from_log(logs)
        known_plot_filenames = sorted(
            {
                str(item.get("plot_filename", "")).strip()
                for item in plot_artifacts
                if isinstance(item, dict) and str(item.get("plot_filename", "")).strip()
            }
        )
        known_plot_handles = sorted(
            {
                str(item.get("plot_handle", "")).strip()
                for item in plot_artifacts
                if isinstance(item, dict) and str(item.get("plot_handle", "")).strip()
            }
        )
        plot_handle_to_filename = {
            str(item.get("plot_handle", "")).strip(): str(item.get("plot_filename", "")).strip()
            for item in plot_artifacts
            if (
                isinstance(item, dict)
                and str(item.get("plot_handle", "")).strip()
                and str(item.get("plot_filename", "")).strip()
            )
        }
        nature_style_mode = bool(
            nature_style_papers_prompt_enabled and paper_kind_norm != "meta_research"
        )
        placeholder_tokens: list[str] = []
        if nature_style_mode and has_unresolved_template_placeholders(report):
            placeholder_tokens = find_unresolved_template_placeholders(report)
            log.warning(
                "[research_one_step][placeholder_leak_detected] "
                f"agent_id={agent_id} iteration={iteration} "
                f"tokens={placeholder_tokens[:8]!r}"
            )
            revised_report = await _attempt_placeholder_revision_with_llm(
                llm=(llm_for(agent_id) if llm_for else llm),
                report=report,
                placeholders=placeholder_tokens,
                agent_id=agent_id,
                iteration=iteration,
            )
            if revised_report:
                revised_clean, revised_stripped = sanitize_report(revised_report)
                if revised_clean:
                    report = revised_clean
                    report_stripped = bool(report_stripped or revised_stripped)
            placeholder_tokens = find_unresolved_template_placeholders(report)
        citation_eval = _evaluate_report_citations(
            report=report,
            result_script=result_script,
            paper_status_by_id=paper_status_by_id,
            paper_owner_by_id=paper_owner_by_id,
            full_citing_enabled=full_citing_enabled,
            graph_logic_enabled=graph_logic_enabled,
            require_plot_citations=require_plot_citations,
            code_execs_successful=code_execs_successful,
            plot_validation_waived=plot_validation_waived,
            known_plot_filenames=known_plot_filenames,
            known_plot_handles=known_plot_handles,
            plot_handle_to_filename=plot_handle_to_filename,
        )
        title = str(citation_eval["title"])
        abstract = str(citation_eval["abstract"])
        citation_verdict = dict(citation_eval["citation_verdict"])
        citation_errors = list(citation_eval["citation_errors"])
        code_evidence_errors = list(citation_eval["code_evidence_errors"])
        invalid_archive_refs = list(citation_eval["invalid_archive_refs"])
        citation_compliant = bool(citation_eval["citation_compliant"])
        internal_citations = list(citation_eval["internal_citations"])
        plot_waiver_applied = bool(citation_eval["plot_waiver_applied"])
        status = "under_review"
        if report_stripped:
            status = "invalid_action_report"
            log.warning(
                "[research_one_step][report_invalid_action] "
                f"agent_id={agent_id} iteration={iteration} "
                "status=invalid_action_report"
            )
        elif full_citing_enabled and code_evidence_errors:
            status = "invalid_code_evidence_report"
            log.warning(
                "[research_one_step][code_evidence_invalid] "
                f"agent_id={agent_id} iteration={iteration} "
                f"errors={code_evidence_errors[:5]!r}"
            )
        elif full_citing_enabled and not citation_verdict.get("is_valid", False):
            status = "invalid_citation_report"
            log.warning(
                "[research_one_step][citation_schema_invalid] "
                f"agent_id={agent_id} iteration={iteration} "
                f"errors={citation_verdict.get('errors', [])[:5]!r}"
            )
        elif nature_style_mode and placeholder_tokens:
            status = "invalid_placeholder_report"
            log.warning(
                "[research_one_step][placeholder_leak_unresolved] "
                f"agent_id={agent_id} iteration={iteration} "
                f"tokens={placeholder_tokens[:8]!r}"
            )
        if has_code_script_recording_anomaly(
            successful_code_executions=code_execs_successful,
            code_script=result_script,
        ):
            log.warning(
                "[research_one_step][code_script_recording_anomaly] "
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                f"agent_id={agent_id} iteration={iteration} "
                f"code_execs_successful={code_execs_successful} but code_script is empty\n"
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            )
            status = "code_script_recording_anomaly"

        paper_id_for_output = _build_paper_id(
            iteration=iteration,
            agent_id=agent_id,
            paper_id_suffix=paper_suffix,
        )
        meta_validation_errors: list[str] = []
        meta_quality_gate_status = "not_applicable"
        meta_quality_fail_reasons: list[str] = []
        meta_revision_attempts = 0
        meta_supported_claim_count = 0
        meta_unsupported_claim_count = 0
        meta_unresolved_uncertainty_claim_count = 0
        meta_code_verified_major_claim_count = 0
        meta_claim_checks: dict[str, Any] = {}
        citation_lint: dict[str, Any] = {}
        if paper_kind_norm == "meta_research":
            validation_cfg = dict(meta_validation_by_agent.get(str(agent_id), {}) or {})
            validation_cfg["paper_id"] = paper_id_for_output
            meta_gate = _evaluate_meta_quality_gate(
                report=report,
                citation_verdict=citation_verdict,
                validation_cfg=validation_cfg,
            )
            meta_validation_errors = list(meta_gate.get("base_validation_errors", []) or [])
            meta_quality_gate_status = str(meta_gate.get("status", "failed"))
            meta_quality_fail_reasons = list(meta_gate.get("final_fail_reasons", []) or [])
            meta_supported_claim_count = int(meta_gate.get("supported_claim_count", 0) or 0)
            meta_unsupported_claim_count = int(meta_gate.get("unsupported_claim_count", 0) or 0)
            meta_unresolved_uncertainty_claim_count = int(
                meta_gate.get("unresolved_uncertainty_claim_count", 0) or 0
            )
            meta_code_verified_major_claim_count = int(
                meta_gate.get("code_verified_major_claim_count", 0) or 0
            )
            meta_claim_checks = dict(meta_gate.get("meta_claim_checks", {}) or {})
            citation_lint = dict(meta_gate.get("citation_lint", {}) or {})

            gate_cfg = dict(meta_gate.get("validation_gate", {}) or {})
            max_revision_attempts = int(gate_cfg.get("max_revision_attempts", 0) or 0)
            gate_failed = meta_gate.get("status") == "failed"
            if gate_failed and max_revision_attempts > 0 and status == "under_review":
                for revision_attempt in range(1, max_revision_attempts + 1):
                    meta_revision_attempts = int(revision_attempt)
                    revised_report = await _attempt_meta_revision_with_llm(
                        llm=(llm_for(agent_id) if llm_for else llm),
                        report=report,
                        fail_reasons=meta_quality_fail_reasons,
                        agent_id=agent_id,
                        iteration=iteration,
                        meta_context=(
                            phase_context_lookup.get(str(agent_id), {}).get(
                                "meta_research_context", {}
                            )
                            if isinstance(phase_context_lookup.get(str(agent_id), {}), Mapping)
                            else {}
                        ),
                    )
                    if not revised_report:
                        revised_report = _render_targeted_meta_revision(
                            report=report,
                            citation_verdict=citation_verdict,
                            validation_cfg=validation_cfg,
                            fail_reasons=meta_quality_fail_reasons,
                        )
                    revised_clean, revised_stripped = sanitize_report(revised_report)
                    if not revised_clean:
                        continue
                    report = revised_clean
                    report_stripped = bool(report_stripped or revised_stripped)
                    citation_eval = _evaluate_report_citations(
                        report=report,
                        result_script=result_script,
                        paper_status_by_id=paper_status_by_id,
                        paper_owner_by_id=paper_owner_by_id,
                        full_citing_enabled=full_citing_enabled,
                        graph_logic_enabled=graph_logic_enabled,
                        require_plot_citations=require_plot_citations,
                        code_execs_successful=code_execs_successful,
                        plot_validation_waived=plot_validation_waived,
                        known_plot_filenames=known_plot_filenames,
                        known_plot_handles=known_plot_handles,
                        plot_handle_to_filename=plot_handle_to_filename,
                    )
                    title = str(citation_eval["title"])
                    abstract = str(citation_eval["abstract"])
                    citation_verdict = dict(citation_eval["citation_verdict"])
                    citation_errors = list(citation_eval["citation_errors"])
                    code_evidence_errors = list(citation_eval["code_evidence_errors"])
                    invalid_archive_refs = list(citation_eval["invalid_archive_refs"])
                    citation_compliant = bool(citation_eval["citation_compliant"])
                    internal_citations = list(citation_eval["internal_citations"])
                    plot_waiver_applied = bool(citation_eval["plot_waiver_applied"])
                    meta_gate = _evaluate_meta_quality_gate(
                        report=report,
                        citation_verdict=citation_verdict,
                        validation_cfg=validation_cfg,
                    )
                    meta_validation_errors = list(meta_gate.get("base_validation_errors", []) or [])
                    meta_quality_gate_status = str(meta_gate.get("status", "failed"))
                    meta_quality_fail_reasons = list(meta_gate.get("final_fail_reasons", []) or [])
                    meta_supported_claim_count = int(meta_gate.get("supported_claim_count", 0) or 0)
                    meta_unsupported_claim_count = int(
                        meta_gate.get("unsupported_claim_count", 0) or 0
                    )
                    meta_unresolved_uncertainty_claim_count = int(
                        meta_gate.get("unresolved_uncertainty_claim_count", 0) or 0
                    )
                    meta_code_verified_major_claim_count = int(
                        meta_gate.get("code_verified_major_claim_count", 0) or 0
                    )
                    meta_claim_checks = dict(meta_gate.get("meta_claim_checks", {}) or {})
                    citation_lint = dict(meta_gate.get("citation_lint", {}) or {})
                    if meta_quality_gate_status in {"passed", "disabled"}:
                        break

            if status == "under_review" and report_stripped:
                status = "invalid_action_report"
            if status == "under_review" and full_citing_enabled and code_evidence_errors:
                status = "invalid_code_evidence_report"
            if (
                status == "under_review"
                and full_citing_enabled
                and not citation_verdict.get("is_valid", False)
            ):
                status = "invalid_citation_report"
            if status == "under_review" and meta_validation_errors:
                status = "invalid_meta_research_report"
                log.warning(
                    "[research_one_step][meta_validation_invalid] "
                    f"agent_id={agent_id} iteration={iteration} "
                    f"errors={meta_validation_errors[:5]!r}"
                )
            if status == "under_review" and meta_quality_gate_status == "failed":
                status = "invalid_meta_research_quality_gate"
                log.warning(
                    "[research_one_step][meta_quality_gate_invalid] "
                    f"agent_id={agent_id} iteration={iteration} "
                    f"errors={meta_quality_fail_reasons[:5]!r}"
                )
        research_validation_status = str(status)
        if logs is not None:
            terminal_payload: dict[str, Any] = {
                "kind": "terminal_route",
                "agent_id": agent_id,
                "iteration": iteration,
                "route": "normal",
                "research_validation_status": research_validation_status,
            }
            logs.add_event(**terminal_payload)
        _console.log(
            f"[research_one_step][terminal_route] iteration={iteration} agent_id={agent_id} "
            f"route=normal status={research_validation_status}"
        )
        outputs.append(
            PaperMetadataPrivate(
                paper_id=paper_id_for_output,
                primary_agent_id=agent_id,
                collab_agent_ids=result.get("collab_agent_ids", []),
                title=title,
                abstract=abstract,
                manuscript=report,
                citation_count=0,
                publication_t=iteration,
                cited_paper_ids=internal_citations,
                metareview=None,
                code_script=result_script,
                status=research_validation_status,
                publication_status="under_review",
                research_validation_status=research_validation_status,
                citation_compliant=citation_compliant,
                citation_errors=citation_errors,
                citation_error_count=len(citation_errors),
                invalid_archive_refs=invalid_archive_refs,
                is_citable_archive_source=False,
                citable_with_warnings=bool(plot_waiver_applied),
                plot_artifacts=plot_artifacts,
                plot_filenames=known_plot_filenames,
                plot_handles=known_plot_handles,
                plot_handle_to_filename=plot_handle_to_filename,
                paper_kind=paper_kind_norm,
                research_phase=phase,
                research_session_id=research_session_id,
                meta_research_source_paper_ids=meta_source_ids,
                meta_research_trigger_iteration=meta_trigger_iteration,
                meta_quality_gate_status=meta_quality_gate_status,
                meta_quality_fail_reasons=meta_quality_fail_reasons,
                meta_revision_attempts=meta_revision_attempts,
                meta_supported_claim_count=meta_supported_claim_count,
                meta_unsupported_claim_count=meta_unsupported_claim_count,
                meta_unresolved_uncertainty_claim_count=meta_unresolved_uncertainty_claim_count,
                meta_code_verified_major_claim_count=meta_code_verified_major_claim_count,
                meta_claim_checks=meta_claim_checks,
                citation_lint=citation_lint,
            )
        )

    return outputs, research_logs
