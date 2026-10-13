import logging
import re
from typing import Any

log = logging.getLogger(__name__)

_ARCHIVE_TAG_RE = re.compile(r"\[Archive:\s*([^\]\n]+?)\s*\]")
_ARXIV_TAG_RE = re.compile(r"\[Arxiv:\s*([^\]\n]+?)\s*\]")
_CODE_TAG_RE = re.compile(r"\[Code:\s*([^\]\n]+?)\s*\]")
_PLOT_TAG_RE = re.compile(r"\[Plot:\s*([^\]\n]+?)\s*\]")

_PAPER_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")
_YEAR_RE = re.compile(r"^\d{4}$")
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-/]*")
_PLOT_FILENAME_TRAILING_PUNCT = ".,;:)"


STRICT_CITATION_SCHEMA_PROMPT_BLOCK = """\
- For each assertion, explicitly cite the evidence source in-line using exactly one of:
  [Archive: paper_id]
  [Arxiv: paper title | author | year]
  [Code: brief description]
  [Plot: plot_token]
- Schema requirements (strict):
  - [Archive: ...] must contain only the paper_id token.
  - [Arxiv: ...] must contain exactly 3 fields separated by "|" in this order:
    paper title | author | year
  - [Code: ...] must contain a brief analysis description of at most 5 words.
  - [Plot: ...] must contain only one plot token (a plot handle is preferred; exact filename is accepted).
- Do not use any other citation tag format.
""".strip()


def _count_words(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def _make_base_tag(tag_type: str, value: str, raw: str, start: int, end: int) -> dict[str, Any]:
    return {
        "type": tag_type,
        "value": value.strip(),
        "raw": raw,
        "start": int(start),
        "end": int(end),
        "valid": False,
        "error": "",
    }


def _normalize_plot_filename_token(value: Any) -> str:
    token = str(value or "").strip()
    if not token:
        return ""
    if len(token) >= 2 and token[0] == token[-1] and token[0] in ("'", '"', "`"):
        token = token[1:-1].strip()
    token = token.replace("\\", "/").strip()
    if "/" in token:
        token = token.rsplit("/", 1)[-1].strip()
    while token and token[-1] in _PLOT_FILENAME_TRAILING_PUNCT:
        token = token[:-1].rstrip()
    return token.strip()


def _validate_archive_tag(tag: dict[str, Any]) -> dict[str, Any]:
    paper_id = str(tag["value"]).strip()
    if not paper_id:
        tag["error"] = "archive tag missing paper_id"
        return tag
    if "|" in paper_id or " " in paper_id:
        tag["error"] = "archive tag must contain only paper_id"
        return tag
    if not _PAPER_ID_RE.match(paper_id):
        tag["error"] = "archive tag paper_id contains unsupported characters"
        return tag
    tag["valid"] = True
    tag["paper_id"] = paper_id
    return tag


def _validate_arxiv_tag(tag: dict[str, Any]) -> dict[str, Any]:
    raw_value = str(tag["value"]).strip()
    parts = [p.strip() for p in raw_value.split("|")]
    if len(parts) != 3:
        tag["error"] = "arxiv tag must be 'paper title | author | year'"
        return tag
    title, author, year = parts
    if not title:
        tag["error"] = "arxiv tag missing paper title"
        return tag
    if not author:
        tag["error"] = "arxiv tag missing author"
        return tag
    if not _YEAR_RE.match(year):
        tag["error"] = "arxiv tag year must be 4 digits"
        return tag
    tag["valid"] = True
    tag["title"] = title
    tag["author"] = author
    tag["year"] = year
    return tag


def _validate_code_tag(tag: dict[str, Any]) -> dict[str, Any]:
    desc = str(tag["value"]).strip()
    if not desc:
        tag["error"] = "code tag missing description"
        return tag
    word_count = _count_words(desc)
    if word_count == 0:
        tag["error"] = "code tag description must include at least 1 word"
        return tag
    if word_count > 5:
        tag["error"] = "code tag description must be <= 5 words"
        return tag
    tag["valid"] = True
    tag["description"] = desc
    tag["word_count"] = word_count
    return tag


def _validate_plot_tag(tag: dict[str, Any]) -> dict[str, Any]:
    plot_token = _normalize_plot_filename_token(tag.get("value", ""))
    if not plot_token:
        tag["error"] = "plot tag missing filename"
        return tag
    if "|" in plot_token or " " in plot_token:
        tag["error"] = "plot tag must contain only filename token"
        return tag
    if not _PAPER_ID_RE.match(plot_token):
        tag["error"] = "plot tag filename contains unsupported characters"
        return tag
    tag["valid"] = True
    tag["plot_token"] = plot_token
    # Backward compatibility for existing callers that expect this field.
    tag["plot_filename"] = plot_token
    return tag


def _build_plot_lookup(
    *,
    known_plot_filenames: list[str] | None,
    known_plot_handles: list[str] | None,
    plot_handle_to_filename: dict[str, str] | None,
) -> dict[str, Any]:
    known_filenames: set[str] = set()
    known_handles: set[str] = set()
    handle_to_filename: dict[str, str] = {}

    for x in known_plot_filenames or []:
        normalized = _normalize_plot_filename_token(x)
        if normalized:
            known_filenames.add(normalized)
    for x in known_plot_handles or []:
        normalized = _normalize_plot_filename_token(x)
        if normalized:
            known_handles.add(normalized)
    for handle_raw, filename_raw in (plot_handle_to_filename or {}).items():
        handle = _normalize_plot_filename_token(handle_raw)
        filename = _normalize_plot_filename_token(filename_raw)
        if not handle or not filename:
            continue
        known_handles.add(handle)
        known_filenames.add(filename)
        handle_to_filename[handle] = filename

    filename_to_handle: dict[str, str] = {}
    for handle, filename in handle_to_filename.items():
        filename_to_handle.setdefault(filename, handle)

    lowercase_handles: dict[str, str] = {}
    lowercase_filenames: dict[str, str] = {}
    for handle in known_handles:
        lowercase_handles.setdefault(handle.lower(), handle)
    for filename in known_filenames:
        lowercase_filenames.setdefault(filename.lower(), filename)

    return {
        "known_filenames": known_filenames,
        "known_handles": known_handles,
        "handle_to_filename": handle_to_filename,
        "filename_to_handle": filename_to_handle,
        "lowercase_handles": lowercase_handles,
        "lowercase_filenames": lowercase_filenames,
    }


def _resolve_plot_token(token: str, lookup: dict[str, Any]) -> dict[str, str]:
    normalized = _normalize_plot_filename_token(token)
    if not normalized:
        return {"token": "", "plot_filename": "", "plot_handle": "", "matched_via": ""}

    known_filenames: set[str] = lookup.get("known_filenames", set())
    known_handles: set[str] = lookup.get("known_handles", set())
    handle_to_filename: dict[str, str] = lookup.get("handle_to_filename", {})
    filename_to_handle: dict[str, str] = lookup.get("filename_to_handle", {})
    lowercase_handles: dict[str, str] = lookup.get("lowercase_handles", {})
    lowercase_filenames: dict[str, str] = lookup.get("lowercase_filenames", {})

    if normalized in handle_to_filename:
        filename = handle_to_filename[normalized]
        return {
            "token": normalized,
            "plot_filename": filename,
            "plot_handle": normalized,
            "matched_via": "handle",
        }
    if normalized in known_filenames:
        return {
            "token": normalized,
            "plot_filename": normalized,
            "plot_handle": filename_to_handle.get(normalized, ""),
            "matched_via": "filename",
        }
    if normalized in known_handles:
        return {
            "token": normalized,
            "plot_filename": handle_to_filename.get(normalized, ""),
            "plot_handle": normalized,
            "matched_via": "handle_only",
        }

    lowered = normalized.lower()
    if lowered in lowercase_handles:
        handle = lowercase_handles[lowered]
        return {
            "token": normalized,
            "plot_filename": handle_to_filename.get(handle, ""),
            "plot_handle": handle,
            "matched_via": "handle_casefold",
        }
    if lowered in lowercase_filenames:
        filename = lowercase_filenames[lowered]
        return {
            "token": normalized,
            "plot_filename": filename,
            "plot_handle": filename_to_handle.get(filename, ""),
            "matched_via": "filename_casefold",
        }

    return {
        "token": normalized,
        "plot_filename": "",
        "plot_handle": "",
        "matched_via": "",
    }


def rewrite_plot_citation_tags(
    text: str,
    *,
    known_plot_filenames: list[str] | None = None,
    known_plot_handles: list[str] | None = None,
    plot_handle_to_filename: dict[str, str] | None = None,
    prefer_plot_handles: bool = True,
) -> dict[str, Any]:
    tags = parse_citation_tags(text or "")
    if not tags:
        return {
            "text": text or "",
            "changed": False,
            "replacements": [],
            "unresolved_tokens": [],
        }

    lookup = _build_plot_lookup(
        known_plot_filenames=known_plot_filenames,
        known_plot_handles=known_plot_handles,
        plot_handle_to_filename=plot_handle_to_filename,
    )

    replacements: list[dict[str, Any]] = []
    unresolved_tokens: list[str] = []
    chunks: list[str] = []
    cursor = 0

    for tag in tags:
        if tag.get("type") != "plot":
            continue
        if not tag.get("valid"):
            continue
        raw = str(tag.get("raw") or "")
        start = int(tag.get("start", 0) or 0)
        end = int(tag.get("end", start) or start)
        token = str(tag.get("plot_token") or tag.get("plot_filename") or "").strip()
        resolved = _resolve_plot_token(token, lookup)
        canonical_token = ""
        if prefer_plot_handles and resolved.get("plot_handle"):
            canonical_token = str(resolved.get("plot_handle") or "").strip()
        if not canonical_token and resolved.get("plot_filename"):
            canonical_token = str(resolved.get("plot_filename") or "").strip()
        if not canonical_token:
            unresolved_tokens.append(token)
            continue
        replacement_raw = f"[Plot: {canonical_token}]"
        if replacement_raw == raw:
            continue
        chunks.append((text or "")[cursor:start])
        chunks.append(replacement_raw)
        cursor = end
        replacements.append(
            {
                "start": start,
                "end": end,
                "from": raw,
                "to": replacement_raw,
                "plot_token": token,
                "plot_filename": resolved.get("plot_filename", ""),
                "plot_handle": resolved.get("plot_handle", ""),
                "matched_via": resolved.get("matched_via", ""),
            }
        )

    if not replacements:
        return {
            "text": text or "",
            "changed": False,
            "replacements": [],
            "unresolved_tokens": unresolved_tokens,
        }

    chunks.append((text or "")[cursor:])
    rewritten = "".join(chunks)
    return {
        "text": rewritten,
        "changed": rewritten != (text or ""),
        "replacements": replacements,
        "unresolved_tokens": unresolved_tokens,
    }


def parse_citation_tags(text: str) -> list[dict[str, Any]]:
    """Extract and validate [Archive]/[Arxiv]/[Code]/[Plot] tags in source-order."""
    if not text:
        return []

    tags: list[dict[str, Any]] = []
    for match in _ARCHIVE_TAG_RE.finditer(text):
        tags.append(
            _validate_archive_tag(
                _make_base_tag(
                    tag_type="archive",
                    value=match.group(1),
                    raw=match.group(0),
                    start=match.start(),
                    end=match.end(),
                )
            )
        )

    for match in _ARXIV_TAG_RE.finditer(text):
        tags.append(
            _validate_arxiv_tag(
                _make_base_tag(
                    tag_type="arxiv",
                    value=match.group(1),
                    raw=match.group(0),
                    start=match.start(),
                    end=match.end(),
                )
            )
        )

    for match in _CODE_TAG_RE.finditer(text):
        tags.append(
            _validate_code_tag(
                _make_base_tag(
                    tag_type="code",
                    value=match.group(1),
                    raw=match.group(0),
                    start=match.start(),
                    end=match.end(),
                )
            )
        )
    for match in _PLOT_TAG_RE.finditer(text):
        tags.append(
            _validate_plot_tag(
                _make_base_tag(
                    tag_type="plot",
                    value=match.group(1),
                    raw=match.group(0),
                    start=match.start(),
                    end=match.end(),
                )
            )
        )

    tags.sort(key=lambda t: t["start"])
    return tags


def validate_citation_tags(
    text: str,
    *,
    require_at_least_one: bool = False,
    known_archive_status_by_id: dict[str, str] | None = None,
    require_accepted_archive_ids: bool = False,
    unknown_archive_ids_are_errors: bool = False,
    require_backed_code_tags: bool = False,
    successful_code_executions_current_iteration: int | None = None,
    code_script: str | None = None,
    known_plot_filenames: list[str] | None = None,
    known_plot_handles: list[str] | None = None,
    plot_handle_to_filename: dict[str, str] | None = None,
    require_known_plot_filenames: bool = False,
    require_known_plot_handles: bool = False,
    require_at_least_one_plot_tag: bool = False,
) -> dict[str, Any]:
    """Validate strict citation schema and optional archive-status constraints.

    Args:
        text: Manuscript or review text to scan for citation tags.
        require_at_least_one: Require at least one citation tag anywhere in the text.
        known_archive_status_by_id: paper_id -> publication status for the archive.
        require_accepted_archive_ids: Require every [Archive: ...] id to be accepted.
        unknown_archive_ids_are_errors: When True, an [Archive: ...] id absent from
            known_archive_status_by_id is a hard error. When False (phase-1 default)
            it is logged under "[citation][unknown_archive_ids]" so the flip rate can
            be measured before papers start being excluded.
        require_backed_code_tags: Require [Code: ...] tags to be backed by a
            successful execute_code run in the current iteration.
        successful_code_executions_current_iteration: Count used by that check.
        code_script: Submitted code, used by that check.
        known_plot_filenames: Plot filenames the run actually produced.
        known_plot_handles: Plot handles the run actually produced.
        plot_handle_to_filename: Mapping used to resolve handles to filenames.
        require_known_plot_filenames: Require [Plot: ...] filenames to be known.
        require_known_plot_handles: Require [Plot: ...] handles to be known.
        require_at_least_one_plot_tag: Require at least one [Plot: ...] tag.

    Returns:
        A verdict dict with ``is_valid``, ``errors``, the parsed ``tags``, and the
        per-category id/handle collections consumed by callers.
    """
    tags = parse_citation_tags(text)
    errors: list[str] = []

    if require_at_least_one and not tags:
        errors.append("at least one citation tag is required")

    invalid_tags = [t for t in tags if not t.get("valid")]
    for tag in invalid_tags:
        err = str(tag.get("error") or "invalid tag")
        errors.append(f"{tag.get('raw')}: {err}")

    status_map = {
        str(k): str(v).strip().lower() for k, v in (known_archive_status_by_id or {}).items()
    }
    # NOTE: no `and status_map` guard. An empty archive is exactly when a fabricated
    # id is most likely, so the check must still run.
    if require_accepted_archive_ids:
        unknown_ids: list[str] = []
        for tag in tags:
            if tag.get("type") != "archive" or not tag.get("valid"):
                continue
            pid = str(tag.get("paper_id"))
            status = status_map.get(pid)
            if status is None:
                msg = f"[Archive: {pid}] references a paper_id not present in the archive"
                unknown_ids.append(pid)
                if unknown_archive_ids_are_errors:
                    errors.append(msg)
            elif status != "accepted":
                errors.append(f"[Archive: {pid}] references non-accepted paper status='{status}'")
        if unknown_ids and not unknown_archive_ids_are_errors:
            # Phase-1 measurement channel. A log line needs no plumbing through the
            # verdict dict, the citation-lint artifact, or the snapshot schema, and is
            # greppable across a run:
            #   grep -c 'citation.unknown_archive_ids' <run log>
            log.warning(
                "[citation][unknown_archive_ids] count=%d ids=%s (phase-1: recorded, not enforced)",
                len(unknown_ids),
                unknown_ids[:10],
            )

    plot_lookup = _build_plot_lookup(
        known_plot_filenames=known_plot_filenames,
        known_plot_handles=known_plot_handles,
        plot_handle_to_filename=plot_handle_to_filename,
    )
    if require_known_plot_filenames or require_known_plot_handles:
        for tag in tags:
            if tag.get("type") != "plot" or not tag.get("valid"):
                continue
            plot_token = str(tag.get("plot_token") or tag.get("plot_filename") or "").strip()
            if not plot_token:
                continue
            resolved = _resolve_plot_token(plot_token, plot_lookup)
            resolved_filename = str(resolved.get("plot_filename", "") or "").strip()
            resolved_handle = str(resolved.get("plot_handle", "") or "").strip()
            tag["plot_token"] = str(resolved.get("token", plot_token) or plot_token).strip()
            if resolved_filename:
                tag["plot_filename"] = resolved_filename
            if resolved_handle:
                tag["plot_handle"] = resolved_handle
            tag["plot_match_type"] = str(resolved.get("matched_via") or "")

            missing_filename = bool(require_known_plot_filenames and not resolved_filename)
            missing_handle = bool(require_known_plot_handles and not resolved_handle)
            if missing_filename or missing_handle:
                errors.append(f"[Plot: {plot_token}] references unknown plot citation token")
    if require_at_least_one_plot_tag:
        plot_tag_count = len([t for t in tags if t.get("type") == "plot" and t.get("valid")])
        if plot_tag_count <= 0:
            errors.append("at least one [Plot: ...] citation tag is required")

    success_count: int | None = None
    if successful_code_executions_current_iteration is not None:
        try:
            success_count = int(successful_code_executions_current_iteration)
        except Exception:
            success_count = 0
    script_chars: int | None = None
    if code_script is not None:
        script_chars = len(str(code_script or "").strip())

    code_tag_count = len([t for t in tags if t.get("type") == "code"])
    code_evidence_errors: list[str] = []
    if code_tag_count > 0:
        if success_count is not None and success_count <= 0:
            code_evidence_errors.append(
                "[Code: ...] tags require at least one successful execute_code run in this iteration"
            )
        if script_chars is not None and script_chars <= 0:
            code_evidence_errors.append(
                "[Code: ...] tags require a non-empty code_script from this iteration"
            )
    if require_backed_code_tags:
        errors.extend(code_evidence_errors)

    valid_tags = [t for t in tags if t.get("valid")]
    valid_archive_ids = [str(t.get("paper_id")) for t in valid_tags if t.get("type") == "archive"]
    valid_plot_filenames = [
        str(t.get("plot_filename"))
        for t in valid_tags
        if t.get("type") == "plot" and str(t.get("plot_filename", "")).strip()
    ]
    valid_plot_handles = [
        str(t.get("plot_handle"))
        for t in valid_tags
        if t.get("type") == "plot" and str(t.get("plot_handle", "")).strip()
    ]
    resolved_plot_citations = [
        {
            "plot_token": str(t.get("plot_token") or ""),
            "plot_filename": str(t.get("plot_filename") or ""),
            "plot_handle": str(t.get("plot_handle") or ""),
            "matched_via": str(t.get("plot_match_type") or ""),
        }
        for t in valid_tags
        if t.get("type") == "plot"
    ]
    is_backed = True
    if code_tag_count > 0 and success_count is not None and script_chars is not None:
        is_backed = success_count > 0 and script_chars > 0

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "tags": tags,
        "valid_tags": valid_tags,
        "valid_archive_ids": valid_archive_ids,
        "valid_plot_filenames": valid_plot_filenames,
        "valid_plot_handles": valid_plot_handles,
        "resolved_plot_citations": resolved_plot_citations,
        "code_evidence": {
            "code_tag_count": int(code_tag_count),
            "successful_code_executions_current_iteration": success_count,
            "code_script_chars": script_chars,
            "is_backed": bool(is_backed),
            "errors": [str(e) for e in code_evidence_errors],
        },
        "counts": {
            "total": len(tags),
            "valid": len(valid_tags),
            "invalid": len(invalid_tags),
            "archive": len([t for t in tags if t.get("type") == "archive"]),
            "arxiv": len([t for t in tags if t.get("type") == "arxiv"]),
            "code": len([t for t in tags if t.get("type") == "code"]),
            "plot": len([t for t in tags if t.get("type") == "plot"]),
        },
    }


def review_citation_schema_nudge(errors: list[str] | None = None) -> str:
    lines = [
        "Your citations must use the exact schema:",
        "[Archive: paper_id]",
        "[Arxiv: paper title | author | year]",
        "[Code: brief description]",
        "[Plot: plot_token]",
        "Where [Code: ...] must be <= 5 words.",
    ]
    for err in (errors or [])[:5]:
        lines.append(f"- {err}")
    return "\n".join(lines)
