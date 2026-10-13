import json
import re
from typing import Any

FORBIDDEN = [
    r"dataset[s]?\s+access\s+(issue|problem)",
    r"\bno\s+(direct\s+)?access\b",
    r"\black of\s+(direct\s+)?access\b",
    r"\bunable to access\b",
    r"\bnot (able|allowed|permitted) to access\b",
    r"\bdid not (access|use)\b",
    r"\b(were|was)\s+(not|never)\s+(directly\s+)?access(ed)?\b",
    r"\bwithout (access|accessing)\b",
    r"\bsince I cannot access\b",
    r"\bonce accessible\b",
    r"\brestricted access\b",
    r"\bhypothesis only\b",
    r"\bgenerated a hypothesis\b",
]


def _negative_access_claims(txt: str) -> bool:
    """Detect phrases indicating lack of dataset access."""
    pats = [
        r"\bno (direct )?access\b",
        r"\black of (direct )?access\b",
        r"\bwithout (direct )?access(ing)?\b",
        r"\b(not|never)\s+(directly\s+)?access(ed|ing)?\b",  # not accessed / never accessed
        r"\bcould\s+not\s+access\b|\bcouldn['’]t\s+access\b",
        r"\bunable\s+to\s+access\b|\bnot\s+able\s+to\s+access\b",
        r"\brestricted\s+access\b|\baccess (unavailable|denied|forbidden)\b",
        r"\baccess (was|were)?\s*not\s*(achieved|obtained|possible)\b",  # <-- covers “was not achieved”
        r"\bdid\s+not\s+(have|obtain|gain)\s+(direct\s+)?access\b",
        r"\bdata (were|was)\s+not\s+accessed\b",
    ]
    return any(re.search(p, txt, re.I) for p in pats)


MIN_CHARS = {"title": 15, "abstract": 120, "manuscript": 800}


def _emptyish(s: str) -> bool:
    """Return True if string is None/empty/very short."""
    return not s or len(s.strip()) < 10


def _boilerplatey(s: str) -> bool:
    """Heuristic to flag generic boilerplate leading text."""
    bad_starts = [
        "it appears that there is an issue",
        "it seems there is an issue",
        "since i cannot access the datasets",
        "given the dataset descriptions",
    ]
    ss = s.strip().lower()
    return any(ss.startswith(bs) for bs in bad_starts)


def _first_json_object(text: str) -> dict[str, Any] | None:
    """Try a few strategies to find & parse a JSON object from free text."""
    # 1) direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 2) ```json ... ``` fences
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # 3) first balanced { ... }
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        break
    return None


def _normalize_payload(p: dict[str, Any]) -> dict[str, Any]:
    """Normalize optional fields and types."""
    out = dict(p)
    # citations can be list[str] or list[dict]; normalize to list[str]
    cits = out.get("citations")
    if cits is None:
        pass
    elif isinstance(cits, list):
        norm = []
        for c in cits:
            if isinstance(c, str):
                norm.append(c)
            elif isinstance(c, dict):
                # try minimal render
                title = c.get("title") or c.get("paper_title") or ""
                rid = c.get("id") or c.get("doi") or c.get("arxiv_id") or ""
                piece = " ".join(str(x) for x in (title, rid) if x) or json.dumps(
                    c, ensure_ascii=False
                )
                norm.append(piece)
            else:
                norm.append(str(c))
        out["citations"] = norm
    else:
        out["citations"] = [str(cits)]
    # datasets_used must be a list of strings
    dus = out.get("datasets_used")
    if isinstance(dus, str):
        out["datasets_used"] = [dus]
    return out


def _validate_strings(payload: dict[str, Any]) -> list[str]:
    """Validate presence, length, boilerplate, and forbidden phrases for text fields."""
    errs: list[str] = []
    for field in ("title", "abstract", "manuscript"):
        if field not in payload:
            errs.append(f"Missing field: {field}")
            continue
        txt = str(payload[field])
        if _emptyish(txt):
            errs.append(f"{field}: empty")
        if len(txt) < MIN_CHARS[field]:
            errs.append(f"{field}: too short ({len(txt)} < {MIN_CHARS[field]})")
        if _boilerplatey(txt):
            errs.append(f"{field}: boilerplate detected")
        for pat in FORBIDDEN:
            if re.search(pat, txt, flags=re.IGNORECASE):
                errs.append(f"{field}: forbidden phrase '{pat}'")

    # Section cues
    manus = str(payload.get("manuscript", ""))
    low = manus.lower()
    if "methods" not in low:
        errs.append("manuscript: must include a 'Methods' section")
    if not any(k in low for k in ["results", "findings", "analysis"]):
        errs.append("manuscript: must include a 'Results'/'Findings'/'Analysis' section")
    return errs


def _validate_datasets(payload: dict[str, Any], allowed_dataset_keys: list[str]) -> list[str]:
    """Validate datasets_used field against allowed keys and deduplicate."""
    errs: list[str] = []
    used = payload.get("datasets_used")
    if not isinstance(used, list) or not all(isinstance(x, str) for x in used):
        errs.append("datasets_used: must be a list of strings")
        return errs
    if not used:
        errs.append("datasets_used: cannot be empty")
        return errs
    if allowed_dataset_keys:
        bad = [x for x in used if x not in allowed_dataset_keys]
        if bad:
            errs.append(f"datasets_used: unknown keys {bad}; allowed: {allowed_dataset_keys}")
    # de-dup preserve order
    seen, out = set(), []
    for x in used:
        if x not in seen:
            seen.add(x)
            out.append(x)
    payload["datasets_used"] = out
    return errs


def validate_paper_payload(
    *,
    payload_or_text: str | dict[str, Any],
    allowed_dataset_keys: list[str],
    allowed_citation_ids: list[str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Validate and normalize a paper payload parsed from model output.

    Args:
        payload_or_text: Dict payload or raw text containing a JSON object.
        allowed_dataset_keys: Valid dataset keys for `datasets_used`.
        allowed_citation_ids: Optional whitelist of citation IDs to cross-check.

    Returns:
        Tuple of (normalized_payload_dict, errors_list). On parse failure, payload is {}.
    """
    # Parse or accept dict
    if isinstance(payload_or_text, dict):
        payload = payload_or_text
    elif isinstance(payload_or_text, str):
        obj = _first_json_object(payload_or_text)
        if obj is None:
            return {}, ["Could not parse JSON object from model output"]
        payload = obj
    else:
        return {}, [f"Unsupported payload type: {type(payload_or_text).__name__}"]

    # Normalize
    payload = _normalize_payload(payload)

    # Required keys basic presence
    errs: list[str] = []

    # Content-level contradiction: if they claim no access, reject
    txt_blobs = " ".join(
        [payload.get("title", ""), payload.get("abstract", ""), payload.get("manuscript", "")]
    )
    if _negative_access_claims(txt_blobs):
        errs.append("forbidden: claims of no/indirect dataset access contradict preflight.")

    # citation cross-check (only if we have allowed evidence IDs) ===
    allowed = set(allowed_citation_ids or [])
    if allowed:
        bad_cites = []
        for c in payload.get("citations") or []:
            s = str(c)
            # Accept if any allowed token is contained in the citation string
            ok = any(tok.lower() in s.lower() for tok in allowed)
            if not ok:
                bad_cites.append(s)
        if bad_cites:
            errs.append(f"citations: {len(bad_cites)} unverified (not seen in tool retrievals)")

    # String/content checks
    errs += _validate_strings(payload)

    # datasets_used checks
    errs += _validate_datasets(payload, allowed_dataset_keys)

    return payload, errs
