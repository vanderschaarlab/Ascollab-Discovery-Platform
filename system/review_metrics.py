import logging
import math
import re
from collections import Counter
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import numpy as np

from stores.paper_archive import PaperArchive
from system.schema import PaperMetadataPrivate
from system.trackers import SessionStatsRegistry

log = logging.getLogger(__name__)

# -----------------------------
# Small helpers
# -----------------------------

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+-]+")
_HYPOTHESIS_CUE_RE = re.compile(
    r"\bwe\s+(?:hypothesi[zs]e|propose|test|investigate|examine|ask|expect)\b|\bhypothes(?:is|es)\b|\bwe\s+aim\s+to\b",
    flags=re.IGNORECASE,
)
_STOPWORDS = {
    "the",
    "and",
    "of",
    "to",
    "in",
    "a",
    "for",
    "on",
    "with",
    "by",
    "an",
    "is",
    "are",
    "we",
    "our",
    "that",
    "this",
    "these",
    "those",
    "as",
    "at",
    "be",
    "from",
    "or",
    "was",
    "were",
    "it",
    "their",
    "its",
    "into",
    "using",
    "use",
    "can",
    "may",
    "might",
    "such",
    "which",
    "have",
    "has",
    "had",
}

# ---------------------------------------------------------------------------
# Metric selection switches (comment out a line to disable a metric)
# ---------------------------------------------------------------------------
# These lists define which metrics are returned under each category. To toggle a
# metric off, comment out its line. The left value is the output key; the right
# value is the context key computed in the corresponding _compute_* function.
SUPPORT_METRICS = [
    ("external_papers_read", "external_papers_read"),
    ("successful_code_executions", "successful_code_executions"),
    ("lines_of_code", "lines_of_code"),
    ("evidence_types_present", "evidence_types_present"),
    ("num_statistical_code_cells", "num_statistical_code_cells"),
    ("evidence_breadth_index", "evidence_breadth_index"),
    ("analytical_depth", "analytical_depth"),
    ("claim_evidence_alignment_max_cosine", "claim_evidence_alignment_max_cosine"),
]

SOUNDNESS_METRICS = [
    ("internal_consistency_score", "internal_consistency_score"),
    ("num_statistical_validity_checks_passed", "num_statistical_validity_checks_passed"),
    ("num_statistical_validity_checks_total", "num_statistical_validity_checks_total"),
    ("diagnostic_validity_pass_rate", "diagnostic_validity_pass_rate"),
]

SIGNIFICANCE_METRICS = [
    ("clinical_biological_relevance", "clinical_biological_relevance"),
    ("clinical_relevance_score", "clinical_relevance_score"),
    ("effect_size_strength", "effect_size_strength"),
    ("distance_to_prior_mean", "distance_to_prior_mean"),
    ("distance_to_prior_min", "distance_to_prior_min"),
    ("distance_to_prior_max", "distance_to_prior_max"),
]

ORIGINALITY_METRICS = [
    ("hypothesis_overlap_jaccard", "hypothesis_overlap_jaccard"),
    ("concept_pair_rarity", "concept_pair_rarity"),
    ("concept_entropy", "concept_entropy"),
    ("semantic_novelty_knn_distance", "semantic_novelty_knn_distance"),
    ("embedding_novelty_percentile", "embedding_novelty_percentile"),
    ("methodological_novelty_flag", "methodological_novelty_flag"),
    ("mechanistic_novelty_flag", "mechanistic_novelty_flag"),
]

# Catalog of all available review metrics by category (output keys).
REVIEW_METRIC_CATALOG: dict[str, list[str]] = {
    "support": [k for (k, _) in SUPPORT_METRICS],
    "soundness": [k for (k, _) in SOUNDNESS_METRICS],
    "significance": [k for (k, _) in SIGNIFICANCE_METRICS],
    "originality": [k for (k, _) in ORIGINALITY_METRICS],
}


def review_metric_catalog() -> dict[str, list[str]]:
    """Return the full catalog of review metric keys by category."""
    return {k: list(v) for k, v in REVIEW_METRIC_CATALOG.items()}


def normalize_metric_allowlist(
    metrics_allowlist: list[str] | None,
) -> tuple[dict[str, set[str]], list[str]]:
    """
    Normalize an allowlist of metric keys into per-category sets.

    Accepted formats:
      - "support.external_papers_read" (explicit metric key)
      - "support" or "support.*" (all metrics in a category)

    If metrics_allowlist is None or empty, all metrics are enabled.

    Returns:
      (allowlist_by_category, unknown_entries)
    """
    unknown: list[str] = []
    catalog = review_metric_catalog()

    if not metrics_allowlist:
        return {cat: set(keys) for cat, keys in catalog.items()}, unknown

    allow: dict[str, set[str]] = {cat: set() for cat in catalog.keys()}
    for raw in metrics_allowlist:
        if not isinstance(raw, str):
            unknown.append(str(raw))
            continue
        entry = raw.strip()
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
        if metric in ("*", "all"):
            allow[category].update(catalog[category])
            continue
        if metric in catalog[category]:
            allow[category].add(metric)
        else:
            unknown.append(entry)
    return allow, unknown


def _metric_allowed(
    allowlist_by_category: dict[str, set[str]] | None,
    category: str,
    metric_key: str,
) -> bool:
    if allowlist_by_category is None:
        return True
    return metric_key in (allowlist_by_category.get(category) or set())


_STAT_CODE_PATTERNS = [
    r"\bscipy\b",
    r"\bstatsmodels\b",
    r"\bpingouin\b",
    r"\blifelines\b",
    r"\banova\b",
    r"\bttest\b",
    r"\bchi[- ]square\b",
    r"\bwilcoxon\b",
    r"\bmann[- ]whitney\b",
    r"\blogistic\b",
    r"\blinear regression\b",
    r"\bcox\b",
    r"\bkaplan[- ]meier\b",
    r"\bhazard ratio\b",
    r"\bconfidence interval\b",
    r"\bfdr\b",
    r"\bbootstrap\b",
    r"\bpermutation test\b",
    r"\bglm\b",
    r"\bmixed[- ]effects\b",
    r"\bauc\b",
    r"\bprecision\b",
    r"\brecall\b",
    r"\bf1\b",
    r"\bp[- ]?value\b",
]

_INTERNAL_CONSISTENCY_POSITIVE: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"95%\s*ci|confidence interval"), 0.07),
    (re.compile(r"sensitivity analysis|robustness check"), 0.08),
    (re.compile(r"adjusted model|multivaria(?:te|ble)"), 0.06),
    (re.compile(r"propensity score"), 0.06),
    (re.compile(r"cross[- ]validation|train[- ]test split"), 0.06),
    (re.compile(r"bootstrap|permutation test"), 0.05),
    (re.compile(r"power analysis|sample size calculation"), 0.10),
    (re.compile(r"multiple testing|fdr|bonferroni"), 0.06),
    (re.compile(r"negative control|positive control"), 0.05),
    (re.compile(r"pre[- ]registered|preregistr"), 0.08),
    (re.compile(r"code availability|data availability|github|zenodo|osf"), 0.05),
]

_INTERNAL_CONSISTENCY_NEGATIVE: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"post[- ]hoc"), 0.06),
    (re.compile(r"exploratory (analysis )?only"), 0.06),
    (re.compile(r"not (adjusted|controlled)"), 0.08),
    (re.compile(r"unadjusted model"), 0.06),
    (re.compile(r"underpowered|insufficient power|small sample size"), 0.08),
    (re.compile(r"uncontrolled confound|residual confound"), 0.07),
    (re.compile(r"inconsistent|contradictory"), 0.07),
]

_STAT_VALIDITY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bvariance inflation factor\b|\bvif\b"),
    re.compile(r"\bheteroscedasticity\b|\bheteroskedasticity\b"),
    re.compile(r"\bresiduals?\b|\bqq[- ]plot\b"),
    re.compile(r"\bnormality\b|\bshapiro[- ]wilks?\b"),
    re.compile(r"\bproportional hazards\b|\bschoenfeld\b"),
    re.compile(r"\bgoodness[- ]of[- ]fit\b|\bhosmer[- ]lemeshow\b"),
    re.compile(r"\bcalibration\b|\bcalibration plot\b"),
    re.compile(r"\boverdispersion\b|\bdispersion parameter\b"),
    re.compile(r"\bautocorrelation\b|\bdurbin[- ]watson\b|\bljung[- ]box\b"),
    re.compile(r"\bbreusch[- ]pagan\b|\bwhite test\b"),
    re.compile(r"\bcollinearity\b|\bcondition number\b"),
    re.compile(r"\bcook'?s distance\b|\bleverage\b|\binfluential point"),
    re.compile(r"\bassumption(?:s| checks?)\b|\bmodel diagnostics?\b"),
]

_STAT_VALIDITY_FAIL_RE = re.compile(
    r"fail(ed)?|violat(ed)?|not\s+(met|satisf(ied|y)|hold)|issue|problematic|"
    r"hetero\S*\s+detected|poor(?:ly)?\s+calibrat|overdispersion|assumption\s+breach"
)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using a lightweight regex boundary."""
    text = (text or "").strip()
    if not text:
        return []
    return _SENT_SPLIT_RE.split(text)


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization for coarse similarity features."""
    return [m.group(0).lower() for m in _WORD_RE.finditer(text or "")]


def _extract_concept_tokens(text: str) -> list[str]:
    """
    Lightweight concept extraction: keep alphanumeric tokens that are not
    stopwords, not purely numeric, and at least 3 characters long. This keeps
    signal-bearing biomedical/technical terms without heavy NLP dependencies.
    """
    tokens: list[str] = []
    for tok in _tokenize(text):
        if tok in _STOPWORDS or len(tok) < 3:
            continue
        if tok.isdigit():
            continue
        tokens.append(tok)
    return tokens


def _cosine_similarity(a: Counter, b: Counter) -> float:
    """Compute cosine similarity between two sparse token frequency Counters."""
    if not a or not b:
        return 0.0
    # dot product
    dot = sum(a[t] * b[t] for t in a.keys() & b.keys())
    if dot == 0:
        return 0.0
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))


def _percentile_rank(values: list[float], value: float) -> float:
    """Return percentile rank in [0, 100] using weak ordering."""
    if not values:
        return 50.0
    sorted_vals = sorted(values)
    less_or_equal = sum(1 for v in sorted_vals if v <= value)
    return float(100.0 * less_or_equal / len(sorted_vals))


def _extract_hypothesis_tokens_from_text(text: str) -> set[str]:
    """
    Extract a lightweight set of hypothesis-related tokens by selecting
    sentences with hypothesis/claim cues and tokenizing them with stopword
    filtering. Falls back to the first couple of sentences if no cues are
    found to avoid returning an empty set.
    """
    text = (text or "").strip()
    if not text:
        return set()

    sentences = _split_sentences(text) or [text]
    selected: list[str] = []
    for sent in sentences:
        if _HYPOTHESIS_CUE_RE.search(sent):
            selected.append(sent)

    if not selected:
        selected = sentences[:2]

    tokens: set[str] = set()
    for sent in selected:
        for tok in _tokenize(sent):
            if len(tok) <= 2 or tok in _STOPWORDS:
                continue
            tokens.add(tok)
    return tokens


def _archive_texts(archive: PaperArchive | None) -> list[str]:
    """Collect manuscript/abstract/title texts from the archive for reference."""
    if (
        archive is None
        or not hasattr(archive, "_entries")
        or not isinstance(archive._entries, dict)
    ):
        return []
    texts: list[str] = []
    try:
        for p in archive._entries.values():
            t = (
                getattr(p, "manuscript", "")
                or getattr(p, "abstract", "")
                or getattr(p, "title", "")
                or ""
            )
            if t:
                texts.append(t)
    except Exception:
        pass
    return texts


def _safe_get_research_stats_for_agent(
    iteration: int,
    primary_agent_id: str,
    session_stats_registry: SessionStatsRegistry,
    research_session_id: str | None = None,
) -> Any | None:
    """
    Robustly grab the research stats object for a paper/agent at an iteration.

    If `research_session_id` is provided, it is attempted first to disambiguate
    multiple same-iteration sessions for one agent (e.g., primary + meta pass).
    Returns the inner stats_log (not the SessionStats wrapper) if possible.
    """
    stats = None
    session_id_hint = str(research_session_id or "").strip()
    if session_id_hint:
        try:
            phase_logs = None
            if hasattr(session_stats_registry, "get_phase_logs"):
                phase_logs = session_stats_registry.get_phase_logs(iteration, "research")
            if phase_logs is None:
                reg = getattr(session_stats_registry, "registry", {}) or {}
                iter_data = reg.get(iteration, {}) or reg.get(str(iteration), {})
                phase_logs = iter_data.get("research", {}) if isinstance(iter_data, Mapping) else {}
            if isinstance(phase_logs, dict):
                stats = phase_logs.get(session_id_hint)
        except Exception:
            stats = None

    if hasattr(session_stats_registry, "get_research_log_for_agent"):
        try:
            if stats is None:
                stats = session_stats_registry.get_research_log_for_agent(
                    iteration=iteration,
                    agent_id=primary_agent_id,
                )
        except Exception:
            stats = None
    if stats is None:
        reg = getattr(session_stats_registry, "registry", {}) or {}
        iter_data = reg.get(iteration, {}) or reg.get(str(iteration), {})
        research_logs = iter_data.get("research", {}) or {}
        if isinstance(research_logs, dict):
            session_id = f"{primary_agent_id}_t{iteration:02d}"
            stats = (
                research_logs.get(session_id)
                or research_logs.get(f"{primary_agent_id}_t{iteration}")
                or research_logs.get(primary_agent_id)
            )

    if stats is None:
        return None

    # unwrap SessionStats wrapper if present
    if hasattr(stats, "stats_logs"):
        return stats.stats_logs

    return stats


# -----------------------------
# Support metrics
# -----------------------------


def _detect_evidence_types(manuscript_text: str) -> list[str]:
    """
    Heuristic detection of evidence types from manuscript text.

    Returns a list subset of:
      ["numerical_data", "statistical_test", "causal_graph_dag",
       "survival_plot_hazard_ratio", "correlation_regression_clustering"]
    """
    text = (manuscript_text or "").lower()

    evidence_types: list[str] = []

    # Numerical data
    if re.search(
        r"\b(p\s*<\s*0\.\d+|95%\s*ci|confidence interval|median|mean|sd\b|standard deviation)", text
    ):
        evidence_types.append("numerical_data")

    # Statistical tests
    if re.search(
        r"\b(t-test|anova|wilcoxon|mann-whitney|log-rank|chi[- ]square|cox model|cox regression)",
        text,
    ):
        evidence_types.append("statistical_test")

    # Causal graphs / DAGs
    if re.search(r"\b(dag|directed acyclic graph|causal graph|causal diagram)\b", text):
        evidence_types.append("causal_graph_dag")

    # Survival plots / hazard ratios. Bare \bhr\b and \bos\b matched "3 hr
    # incubation" and any stray "OS", so both are narrowed to unambiguous forms.
    if re.search(
        r"\bhazard ratio\b|\bhr\s*[=:]\s*\d|\bkaplan[- ]meier\b|\boverall survival\b"
        r"|\bos\.?time\b|\bpfs\b|\bprogression[- ]free",
        text,
    ):
        evidence_types.append("survival_plot_hazard_ratio")

    # Correlation / regression / clustering
    if re.search(
        r"\bcorrelation\b|\bpearson\b|\bspearman\b|\bregression\b|\bcluster(?:ing)?\b|\bk-means\b|\bhierarchical clustering\b",
        text,
    ):
        evidence_types.append("correlation_regression_clustering")

    return evidence_types


def _approx_claim_evidence_alignment(
    manuscript_text: str,
    embedder: Any | None = None,
) -> float:
    """
    Claim-Evidence alignment using sentence embeddings (LLM embedding model).

    Heuristics:
      - Identify claim sentences via intro/claim cue phrases and early position.
      - Identify evidence sentences via numeric/statistical cues, dataset/result language.
      - Embed each selected sentence with the provided embedder (normalized).
      - Return the maximum cosine similarity between any claim/evidence pair.
    """
    sentences = [s.strip() for s in _split_sentences(manuscript_text) if s.strip()]
    if len(sentences) < 2:
        return 0.5  # neutral

    def _score_claim(idx: int, sent: str) -> float:
        s = sent.lower()
        score = 0.0
        if idx <= 1:
            score += 0.4  # early sentences often state claims
        if re.search(r"\bwe (?:propose|introduce|present|develop|demonstrate|show|report)\b", s):
            score += 1.0
        if re.search(r"\bin this (?:paper|study|work|manuscript)\b", s):
            score += 0.6
        if re.search(r"\bour (?:method|approach|model|framework|algorithm)\b", s):
            score += 0.6
        if re.search(r"\bnovel\b|\bnew\b|\bstate[- ]of[- ]the[- ]art\b|\bfirst\b", s):
            score += 0.4
        return score

    evidence_cues = [
        r"\bexperiment",
        r"\bexperiments",
        r"\bdataset",
        r"\bdatasets",
        r"\bvalidation",
        r"\bresult",
        r"\bresults",
        r"\bevaluation",
        r"\bperformance",
        r"\bclinical",
        r"\bpatient",
        r"\btrial",
        r"\bsignificant\b",
        r"\bp\s*<\s*0\.\d+",
        r"\b95%\s*ci\b",
        r"\bconfidence interval\b",
        r"\bn=\d+",
        r"\btable\s*\d+",
        r"\bfigure\s*\d+",
        r"\bauc\b",
        r"\baccuracy\b",
        r"\bprecision\b",
        r"\brecall\b",
        r"\bf1\b",
        r"\bodds ratio\b",
        r"\bhazard ratio\b",
    ]
    evidence_re = re.compile("|".join(evidence_cues))

    def _score_evidence(idx: int, sent: str) -> float:
        s = sent.lower()
        score = 0.0
        if evidence_re.search(s):
            score += 1.0
        if re.search(r"\b\d+(\.\d+)?%\b", s):
            score += 0.4
        if re.search(r"\bdata\b|\bmeasurement\b|\bobservation\b", s):
            score += 0.3
        if idx > 0:
            score += 0.1  # evidence often follows claims
        return score

    scored_claims = sorted(
        [(i, _score_claim(i, s)) for i, s in enumerate(sentences)],
        key=lambda x: x[1],
        reverse=True,
    )
    claim_idxs = [i for i, sc in scored_claims if sc > 0][:3]
    if not claim_idxs:
        claim_idxs = list(range(min(2, len(sentences))))

    scored_evidence = sorted(
        [(i, _score_evidence(i, s)) for i, s in enumerate(sentences) if i not in claim_idxs],
        key=lambda x: x[1],
        reverse=True,
    )
    evidence_idxs = [i for i, sc in scored_evidence if sc > 0][:5]
    if not evidence_idxs:
        evidence_idxs = [i for i in range(len(sentences)) if i not in claim_idxs] or [0]

    claim_sents = [sentences[i] for i in claim_idxs]
    evidence_sents = [sentences[i] for i in evidence_idxs]

    # If we don't have an embedder, fall back to the original bag-of-words proxy.
    if embedder is None or not hasattr(embedder, "embed_items"):
        claim_tokens = _tokenize(" ".join(claim_sents))
        claim_vec = Counter(claim_tokens)
        sims: list[float] = []
        for s in evidence_sents:
            ev_tokens = _tokenize(s)
            ev_vec = Counter(ev_tokens)
            sims.append(_cosine_similarity(claim_vec, ev_vec))
        if not sims:
            return 0.5
        return float(max(0.0, min(1.0, max(sims))))

    try:
        claim_vecs = np.asarray(embedder.embed_items(claim_sents), dtype=np.float32)
        evidence_vecs = np.asarray(embedder.embed_items(evidence_sents), dtype=np.float32)

        # Defensive re-normalisation in case caller embedder does not L2-normalize.
        def _norm(v: np.ndarray) -> np.ndarray:
            return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)

        claim_vecs = _norm(claim_vecs)
        evidence_vecs = _norm(evidence_vecs)

        sims_matrix = claim_vecs @ evidence_vecs.T
        if sims_matrix.size == 0:
            return 0.5

        sim_max = float(np.max(sims_matrix))
        return float(max(0.0, min(1.0, sim_max)))
    except Exception:
        return 0.5


def _compute_support_metrics(
    paper: PaperMetadataPrivate,
    manuscript_text: str,
    stats: Any | None,
    embedder: Any | None,
) -> dict[str, Any]:
    """Compute support/evidence-related metrics for a paper."""

    def _estimate_statistical_code_cells() -> int:
        """
        Heuristic: count executed code cells with statistical/analysis cues.
        Falls back to successful executions if nothing matched.
        """
        if not stats or not hasattr(stats, "timeline"):
            return code_execs_successful

        timeline = getattr(stats, "timeline", []) or []
        stat_re = re.compile("|".join(_STAT_CODE_PATTERNS), flags=re.IGNORECASE)
        count = 0

        for entry in timeline:
            if not isinstance(entry, dict):
                continue
            if entry.get("kind") != "tool_call":
                continue
            if entry.get("tool") != "execute_code":
                continue
            args = entry.get("args") or {}
            code = args.get("code") or ""
            if code and stat_re.search(code):
                count += 1

        if count == 0 and code_execs_successful > 0:
            return code_execs_successful
        return count

    tool_calls = {}
    code_execs_successful = 0

    if stats is not None:
        tool_calls = getattr(stats, "tool_calls", {}) or {}
        if isinstance(tool_calls, Mapping):
            tool_calls = dict(tool_calls)
        code_execs_successful = int(getattr(stats, "code_executions_successful", 0) or 0)

    _LITERATURE_TOOLS = ("arxiv", "search_internal_archive")

    external_papers_read = 0
    for name in _LITERATURE_TOOLS:
        external_papers_read += int(tool_calls.get(name, 0) or 0)

    # Also count literature tool calls made by collab agents
    for clog in (getattr(stats, "collab_logs", None) or {}).values():
        ctool_calls = getattr(clog, "tool_calls", None) or {}
        for name in _LITERATURE_TOOLS:
            external_papers_read += int(ctool_calls.get(name, 0) or 0)

    evidence_types_present = _detect_evidence_types(manuscript_text)
    claim_evidence_alignment_max_cosine = _approx_claim_evidence_alignment(
        manuscript_text,
        embedder=embedder,
    )

    num_statistical_code_cells = _estimate_statistical_code_cells()

    evidence_breadth_index = max(0, external_papers_read) + len(set(evidence_types_present))
    analytical_depth = num_statistical_code_cells

    lines_of_code = int(getattr(stats, "lines_of_code", 0) or 0) if stats is not None else 0

    context = {
        "external_papers_read": int(max(0, external_papers_read)),
        "successful_code_executions": int(max(0, code_execs_successful)),
        "lines_of_code": int(max(0, lines_of_code)),
        "evidence_types_present": sorted(set(evidence_types_present)),
        "num_statistical_code_cells": int(max(0, num_statistical_code_cells)),
        "evidence_breadth_index": evidence_breadth_index,
        "analytical_depth": analytical_depth,
        "claim_evidence_alignment_max_cosine": claim_evidence_alignment_max_cosine,
    }

    return {out_key: context.get(src_key) for (out_key, src_key) in SUPPORT_METRICS}


# -----------------------------
# Soundness metrics
# -----------------------------
def _heuristic_internal_consistency_score(manuscript_text: str) -> float:
    """
    Heuristic:
      - Reward mentions of planned/statistically careful practices.
      - Penalize cues of weak/underpowered or unadjusted analyses.
      - Clamp to [0, 1] for a normalized score.
    """
    text = manuscript_text.lower()

    score = 0.45  # neutral starting point

    for pat, weight in _INTERNAL_CONSISTENCY_POSITIVE:
        if pat.search(text):
            score += weight

    for pat, weight in _INTERNAL_CONSISTENCY_NEGATIVE:
        if pat.search(text):
            score -= weight

    return float(max(0.0, min(1.0, score)))


def _detect_statistical_validity_checks(manuscript_text: str) -> tuple[int, int]:
    """
    Look for textual mentions of statistical diagnostics and treat them as checks.
    Mentions paired with violation/failure cues only increase the total; all
    other mentions count as passed.

    Returns (num_passed, num_total).
    """
    sentences = [s.lower() for s in _split_sentences(manuscript_text)]

    num_passed = 0
    num_total = 0

    for pat in _STAT_VALIDITY_PATTERNS:
        for sent in sentences:
            if not pat.search(sent):
                continue
            num_total += 1

            if _STAT_VALIDITY_FAIL_RE.search(sent):
                break

            num_passed += 1
            break

    return num_passed, num_total


def _compute_soundness_metrics(
    manuscript_text: str,
    stats: Any | None,
) -> dict[str, Any]:
    internal_consistency_score = _heuristic_internal_consistency_score(manuscript_text)
    num_passed, num_total = _detect_statistical_validity_checks(manuscript_text)
    diagnostic_validity_pass_rate = float(num_passed) / float(num_total) if num_total > 0 else 0.0

    context = {
        "internal_consistency_score": internal_consistency_score,
        "num_statistical_validity_checks_passed": int(num_passed),
        "num_statistical_validity_checks_total": int(num_total),
        "diagnostic_validity_pass_rate": diagnostic_validity_pass_rate,
    }

    return {out_key: context.get(src_key) for (out_key, src_key) in SOUNDNESS_METRICS}


# -----------------------------
# Significance metrics
# -----------------------------
def _clinical_biological_relevance(manuscript_text: str) -> float:
    """
    Heuristic: look for clinical / biomarker / outcome language and map to [0, 1].
    """
    text = manuscript_text.lower()

    keywords = [
        "patient",
        "patients",
        "cohort",
        "trial",
        "clinical",
        "therapy",
        "treatment",
        "drug",
        "response",
        "toxicity",
        "overall survival",
        "progression-free survival",
        "pfs",
        "os",
        "biomarker",
        "endpoint",
    ]

    hits = sum(1 for kw in keywords if kw in text)
    if hits == 0:
        return 0.0
    # Saturate reasonably quickly
    score = min(1.0, hits / 8.0)
    return float(score)


def _effect_size_strength(manuscript_text: str) -> float:
    """
    Approximate standardized effect size strength from reported metrics.
    Combines ratio-based measures (HR/OR/RR) and bounded measures (AUC, Cohen's d).
    Returns a 0..1 score, with conservative defaults when nothing is found.
    """
    text = manuscript_text.lower()

    num = r"(-?[0-9]+(?:\.[0-9]+)?)"
    patterns: list[tuple[str, re.Pattern[str]]] = [
        ("ratio", re.compile(rf"hazard ratio[:=]?\s*{num}")),
        ("ratio", re.compile(rf"\bhr[:=]?\s*{num}")),
        ("ratio", re.compile(rf"odds ratio[:=]?\s*{num}")),
        ("ratio", re.compile(rf"\brr[:=]?\s*{num}")),
        ("ratio", re.compile(rf"risk ratio[:=]?\s*{num}")),
        ("ratio", re.compile(rf"\bor\s*[:=]\s*{num}")),
        ("bounded", re.compile(rf"cohen'?s?\s*d[:=]?\s*{num}")),
        ("bounded", re.compile(rf"hedges'?\s*g[:=]?\s*{num}")),
        ("bounded", re.compile(rf"standardized mean difference[:=]?\s*{num}")),
        ("bounded", re.compile(rf"\bauc[:=]?\s*{num}")),
        ("bounded", re.compile(rf"\broc\s*auc[:=]?\s*{num}")),
        ("bounded", re.compile(rf"\br\^2[:=]?\s*{num}")),
        ("bounded", re.compile(rf"\bcramer'?s?\s*v[:=]?\s*{num}")),
        ("bounded", re.compile(rf"\bbeta\s*coefficient[:=]?\s*{num}")),
        ("bounded", re.compile(rf"\beffect size[:=]?\s*{num}")),
        ("coef", re.compile(rf"\bcoef(?:ficient)?[:=]?\s*{num}")),
        ("coef", re.compile(rf"\bbeta[:=]?\s*{num}")),
        ("coef", re.compile(rf"\blog2\s*fold\s*change[:=]?\s*{num}")),
        ("coef", re.compile(rf"\bfold\s*change[:=]?\s*{num}")),
        ("corr", re.compile(rf"\bpearson\s*r[:=]?\s*{num}")),
        ("corr", re.compile(rf"\bspearman\s*r[:=]?\s*{num}")),
        ("corr", re.compile(rf"\br\s*[:=]\s*{num}")),
    ]

    scores: list[float] = []
    for kind, pat in patterns:
        for match in pat.finditer(text):
            try:
                val = float(match.group(1))
                if kind == "ratio" and val <= 0:
                    continue
            except Exception:
                continue

            if kind == "ratio":
                score = (
                    min(abs(math.log(val)), 2.0) / 2.0
                )  # centered at 1.0, saturate at log-distance 2
            elif kind == "corr":
                score = min(abs(val), 1.0)
            elif kind == "coef":
                score = min(abs(val) / 2.0, 1.0)
            else:
                if val > 1.0:
                    score = min(val / 2.0, 1.0)  # Cohen's d style magnitudes
                else:
                    score = min(val, 1.0)  # AUC or other bounded measures
            scores.append(score)

    if not scores:
        return 0.0

    return float(sum(scores) / len(scores))


def _semantic_distances_to_prior(
    paper: PaperMetadataPrivate,
    manuscript_text: str,
    archive: PaperArchive | None,
    k: int = 5,
) -> list[float]:
    """
    Use the archive's semantic querying (if available) to approximate a novelty
    distance to prior literature.

    Consumes the RAW similarity, not the ranking score: with citation/recency
    boosting enabled the ranking score is a min-max-normalised ordering value, so
    the top hit is always 1.0 and the distance is pinned near 0 regardless of
    content. `query_semantic_with_similarity` reports both separately.

    Returns a list of distances (0..1). Empty if the archive is unavailable, or
    if fewer than two neighbours were retrieved (see below).
    """
    if archive is None:
        return []

    query_text = paper.abstract or manuscript_text[:2000]
    if not query_text:
        return []

    try:
        if hasattr(archive, "query_semantic_with_similarity"):
            results = [
                (other, raw_sim)
                for other, _score, raw_sim in archive.query_semantic_with_similarity(query_text, k)
            ]
        elif hasattr(archive, "query_semantic"):
            results = archive.query_semantic(query_text, k)
        else:
            return []
    except Exception:
        return []

    if len(results) < 2:
        # A single neighbour gives no spread to normalise against; report no signal
        # rather than a confidently wrong one (the old code reported distance=1.0,
        # i.e. "maximally novel", which is the inverse of the truth).
        return []

    distances: list[float] = []
    seen_ids: set[str] = set()
    for other, sim in results:
        try:
            pid = getattr(other, "paper_id", None)
            if pid is not None:
                if pid == paper.paper_id or pid in seen_ids:
                    continue
                seen_ids.add(pid)
            if sim is None:
                continue
            sim_f = float(sim)
        except Exception:
            continue

        # Normalise to [0, 1] and convert similarity -> distance
        sim_clamped = max(0.0, min(1.0, sim_f))
        distances.append(1.0 - sim_clamped)

    return distances


def _compute_significance_metrics(
    paper: PaperMetadataPrivate,
    manuscript_text: str,
    archive: PaperArchive | None,
) -> tuple[dict[str, Any], float]:
    """
    Compute significance-related metrics (clinical/biological relevance and effect sizes).
    Returns (metrics_dict, raw_novelty_placeholder) for backward compatibility.
    """
    clinical_relevance = _clinical_biological_relevance(manuscript_text)
    effect_size = _effect_size_strength(manuscript_text)
    # PaperArchive clamps k to max_k (default 5); ask for what we can actually get
    # rather than implying a 10-neighbour estimate we never receive.
    distances = _semantic_distances_to_prior(paper, manuscript_text, archive, k=5)
    distance_mean = float(sum(distances) / len(distances)) if distances else None
    distance_min = float(min(distances)) if distances else None
    distance_max = float(max(distances)) if distances else None

    context = {
        "clinical_biological_relevance": clinical_relevance,
        "clinical_relevance_score": clinical_relevance,
        "effect_size_strength": effect_size,
        "distance_to_prior_mean": distance_mean,
        "distance_to_prior_min": distance_min,
        "distance_to_prior_max": distance_max,
    }

    metrics = {out_key: context.get(src_key) for (out_key, src_key) in SIGNIFICANCE_METRICS}

    # raw_novelty is no longer used for significance but kept for signature compatibility.
    raw_novelty = 0.0

    return metrics, raw_novelty


# -----------------------------
# Originality metrics
# -----------------------------
# A keyword heuristic, not hypothesis extraction (see docs/metrics.md).
def _methodological_novelty_flag(manuscript_text: str) -> bool:
    """
    Heuristic: look for sentences that explicitly claim a new/novel method,
    with both a novelty cue and a methodological target term.
    """
    text = (manuscript_text or "").lower().strip()
    if not text:
        return False

    sentences = _split_sentences(text) or [text]

    novelty_cues = [
        r"\bnovel\b",
        r"\bnew\b",
        r"\boriginal\b",
        r"\bfirst\b",
        r"\bpreviously (?:unreported|unused|unexplored|undescribed)\b",
        r"\bstate[- ]of[- ]the[- ]art\b",
    ]
    method_terms = [
        "method",
        "approach",
        "pipeline",
        "framework",
        "algorithm",
        "model",
        "architecture",
        "strategy",
        "workflow",
        "procedure",
        "loss function",
        "training scheme",
        "objective function",
    ]
    announce_verbs = r"\bwe (?:introduce|propose|develop|present|design|formulate|derive|build)\b"
    first_application = r"\bfirst (?:application|use) of\b"

    def _has(term_list: list[str], s: str) -> bool:
        return any(term in s for term in term_list)

    for sent in sentences:
        s = sent.lower()
        if not _has(method_terms, s):
            continue

        if re.search(announce_verbs, s) and re.search(r"\b(?:new|novel|first)\b", s):
            return True

        if any(re.search(cue, s) for cue in novelty_cues):
            return True

        if re.search(first_application, s):
            return True

    return False


# A keyword heuristic, not hypothesis extraction (see docs/metrics.md).
def _mechanistic_novelty_flag(manuscript_text: str) -> bool:
    """
    Heuristic: mark True if the text mentions 'novel mechanism' / 'previously
    unreported interaction' or similar language.
    """
    text = manuscript_text.lower()
    if re.search(r"\bnovel (mechanism|pathway|interaction)\b", text):
        return True
    if re.search(
        r"\bpreviously unreported\b|\bpreviously undescribed\b|\bnew (mechanistic|pathway)\b", text
    ):
        return True
    return False


def _hypothesis_overlap_jaccard(
    paper: PaperMetadataPrivate,
    archive: PaperArchive | None = None,
) -> float | None:
    """
    Approximate hypothesis overlap by:
      - Extracting hypothesis-like tokens from the paper's manuscript/abstract.
      - Gathering prior papers via explicit citations (if resolvable in the
        archive) or semantic neighbours (top-k).
      - Computing the average Jaccard similarity between the current token set
        and each prior token set.

    Returns a value in [0, 1]; lower implies more novelty.
    Returns None when there is insufficient signal.
    """
    manuscript_text = getattr(paper, "manuscript", "") or ""
    abstract_text = getattr(paper, "abstract", "") or ""
    current_tokens = _extract_hypothesis_tokens_from_text(manuscript_text or abstract_text)
    if not current_tokens:
        return None

    prior_papers: list[Any] = []
    cited_ids: set[str] = set()

    try:
        for citation in getattr(paper, "cited_paper_ids", []) or []:
            if isinstance(citation, dict):
                pid = citation.get("paper_id")
                if pid:
                    cited_ids.add(pid)
    except Exception:
        cited_ids = set()

    if archive is not None:
        try:
            entries = getattr(archive, "_entries", {}) if hasattr(archive, "_entries") else {}
            if isinstance(entries, dict) and entries:
                for prior in entries.values():
                    try:
                        if getattr(prior, "paper_id", None) in cited_ids:
                            prior_papers.append(prior)
                    except Exception:
                        continue
        except Exception:
            pass

        # Always supplement with semantic neighbours; deduplicate by paper_id to avoid double-counting.
        try:
            query_text = abstract_text or manuscript_text[:2000]
            if query_text:
                semantic_candidates = [p for p, _ in archive.query_semantic(query_text, k=5)]
                prior_papers.extend(semantic_candidates)
        except Exception:
            pass

    prior_token_sets: list[set[str]] = []
    seen_prior_ids: set[str] = set()
    for prior in prior_papers:
        pid = getattr(prior, "paper_id", None)
        if pid and pid in seen_prior_ids:
            continue
        if pid:
            seen_prior_ids.add(pid)
        prior_text = (
            getattr(prior, "manuscript", "")
            or getattr(prior, "abstract", "")
            or getattr(prior, "title", "")
            or ""
        )
        tokens = _extract_hypothesis_tokens_from_text(prior_text)
        if tokens:
            prior_token_sets.append(tokens)

    if not prior_token_sets:
        return None

    overlaps: list[float] = []
    for ptokens in prior_token_sets:
        union = current_tokens | ptokens
        if not union:
            continue
        inter = current_tokens & ptokens
        overlaps.append(len(inter) / len(union))

    if not overlaps:
        return None

    score = sum(overlaps) / len(overlaps)
    return float(max(0.0, min(1.0, score)))


def _concept_pair_rarity(concept_tokens: list[str], archive_texts: list[str]) -> float | None:
    """Average rarity of concept pairs based on co-occurrence in archive texts."""
    concept_tokens = [t for t in concept_tokens if len(t) > 3]
    if not concept_tokens or not archive_texts:
        return None
    pair_score = 0.0
    pair_count = 0
    seen_pairs = set()
    limit = min(len(concept_tokens), 50)
    archive_lower = [t.lower() for t in archive_texts]
    for i in range(limit):
        for j in range(i + 1, limit):
            a, b = concept_tokens[i], concept_tokens[j]
            if a == b:
                continue
            key = tuple(sorted((a, b)))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            freq = 0
            for low in archive_lower:
                if a in low and b in low:
                    freq += 1
            pair_count += 1
            pair_score += -math.log((freq + 1) / (len(archive_lower) + 1))
    if pair_count == 0:
        return None
    return float(pair_score / pair_count)


def _concept_entropy(concept_tokens: list[str]) -> float | None:
    """Shannon entropy over concept token distribution."""
    if not concept_tokens:
        return None
    from collections import Counter

    counts = Counter(concept_tokens)
    total = sum(counts.values())
    if total == 0:
        return None
    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * math.log(max(p, 1e-12))
    return float(entropy)


def _compute_originality_metrics(
    paper: PaperMetadataPrivate,
    manuscript_text: str,
    archive: PaperArchive | None = None,
    embedder: Any | None = None,
    archive_texts_cache: list[str] | None = None,
) -> dict[str, Any]:
    """
    Compose originality metrics:
      - hypothesis_overlap_jaccard: lower -> more novel hypotheses
      - semantic_novelty_knn_distance: mean (1 - cosine) to top-k archive neighbors
      - concept_pair_rarity: average -log co-occurrence frequency of concept pairs
      - concept_entropy: Shannon entropy over concept tokens
    """
    semantic_distances = _semantic_distances_to_prior(paper, manuscript_text, archive, k=10)
    semantic_novelty_knn_distance = (
        float(sum(semantic_distances) / len(semantic_distances)) if semantic_distances else None
    )

    hypothesis_overlap = _hypothesis_overlap_jaccard(paper, archive=archive)

    paper_text = (
        manuscript_text or getattr(paper, "abstract", "") or getattr(paper, "title", "") or ""
    )
    archive_texts = (
        archive_texts_cache if archive_texts_cache is not None else _archive_texts(archive)
    )

    concept_tokens = _extract_concept_tokens(paper_text)
    unique_concepts = list(dict.fromkeys(concept_tokens))  # preserve order, drop repeats for rarity

    concept_pair_rarity = (
        _concept_pair_rarity(unique_concepts, archive_texts) if archive_texts else None
    )
    concept_entropy = _concept_entropy(concept_tokens)
    methodological_flag = _methodological_novelty_flag(manuscript_text)
    mechanistic_flag = _mechanistic_novelty_flag(manuscript_text)

    context = {
        "semantic_novelty_knn_distance": semantic_novelty_knn_distance,
        "hypothesis_overlap_jaccard": hypothesis_overlap,
        "embedding_novelty_percentile": None,  # populated in batch post-processing
        "methodological_novelty_flag": methodological_flag,
        "mechanistic_novelty_flag": mechanistic_flag,
    }
    if concept_pair_rarity is not None:
        context["concept_pair_rarity"] = concept_pair_rarity
    if concept_entropy is not None:
        context["concept_entropy"] = concept_entropy

    return {out_key: context.get(src_key) for (out_key, src_key) in ORIGINALITY_METRICS}


# -----------------------------
# Main entry point
# -----------------------------


def build_review_metrics_for_papers(
    iteration: int,
    candidate_papers: list[PaperMetadataPrivate],
    session_stats_registry: SessionStatsRegistry,
    archive: PaperArchive | None = None,
    embedder: Any | None = None,
    metrics_allowlist: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Build a metrics dict for each paper in an iteration, to be passed into
    the reviewer as `paper_metrics_by_id`.

    The returned structure is:

    {
      "<paper_id>": {
        "support": {
          "external_papers_read": int,
          "successful_code_executions": int,
          "lines_of_code": int,
          "evidence_types_present": List[str],
          "num_statistical_code_cells": int,
          "evidence_breadth_index": int,
          "analytical_depth": int,
          "claim_evidence_alignment_max_cosine": float | None,
        },
        "soundness": {
          "internal_consistency_score": float,
          "num_statistical_validity_checks_passed": int,
          "num_statistical_validity_checks_total": int,
          "diagnostic_validity_pass_rate": float,
        },
        "significance": {
          "distance_to_prior_mean": float | None,
          "distance_to_prior_min": float | None,
          "distance_to_prior_max": float | None,
          "clinical_biological_relevance": float,
          "clinical_relevance_score": float,
          "effect_size_strength": float,
        },
        "originality": {
          "hypothesis_overlap_jaccard": float | None,
          "semantic_novelty_knn_distance": float | None,
          "embedding_novelty_percentile": float | None,
          "concept_pair_rarity": float | None,
          "concept_entropy": float | None,
          "methodological_novelty_flag": bool,
          "mechanistic_novelty_flag": bool,
        },
      },
      ...
    }

    The implementation is modular: to add/remove metrics, extend or prune the
    group-specific _compute_* functions above.
    """
    metrics_by_id: dict[str, dict[str, Any]] = {}

    allowlist_by_category: dict[str, set[str]] | None = None
    if metrics_allowlist is not None:
        allowlist_by_category, unknown = normalize_metric_allowlist(metrics_allowlist)
        if unknown:
            log.warning(f"[review][metrics] unknown metrics in allowlist: {sorted(set(unknown))}")

    # Prefer an explicitly provided embedder, otherwise reuse the archive's embedder if present.
    embedder = embedder or getattr(archive, "embedder", None)
    archive_texts_cache = _archive_texts(archive)
    if not archive_texts_cache:
        archive_texts_cache = [
            (
                getattr(p, "manuscript", "")
                or getattr(p, "abstract", "")
                or getattr(p, "title", "")
                or ""
            )
            for p in candidate_papers
            if getattr(p, "manuscript", None)
            or getattr(p, "abstract", None)
            or getattr(p, "title", None)
        ]

    for paper in candidate_papers:
        pid = paper.paper_id
        manuscript_text = getattr(paper, "manuscript", "") or ""

        agent_ids = [paper.primary_agent_id] + list(getattr(paper, "collab_agent_ids", []) or [])
        paper_session_id = str(getattr(paper, "research_session_id", "") or "").strip() or None
        stats_list = []
        missing_ids = []
        primary_stats = _safe_get_research_stats_for_agent(
            iteration=iteration,
            primary_agent_id=paper.primary_agent_id,
            session_stats_registry=session_stats_registry,
            research_session_id=paper_session_id,
        )
        for idx, agent_id in enumerate(agent_ids):
            if idx == 0:
                agent_stats = primary_stats
            else:
                # A collaborator's work on this paper is its log inside the author's session, not
                # its own research session (which produced its own paper).
                agent_stats = (getattr(primary_stats, "collab_logs", None) or {}).get(agent_id)
            if agent_stats is None:
                missing_ids.append(agent_id)
            else:
                stats_list.append((agent_id, agent_stats))
        if stats_list:
            breakdown = []
            for agent_id, s in stats_list:
                breakdown.append(
                    {
                        "agent_id": agent_id,
                        "code_execs": int(getattr(s, "code_executions", 0) or 0),
                        "code_execs_successful": int(
                            getattr(s, "code_executions_successful", 0) or 0
                        ),
                        "timeline_len": len(getattr(s, "timeline", []) or []),
                    }
                )
            log.debug(f"[review][metrics] stats breakdown paper_id={pid} agents={breakdown}")

        stats = stats_list[0][1] if stats_list else None
        if len(stats_list) > 1:
            merged_tool_calls: dict[str, int] = {}
            merged_timeline: list[dict[str, Any]] = []
            code_execs = 0
            code_execs_successful = 0
            code_execs_failed = 0
            merged_lines_of_code = 0
            for _agent_id, s in stats_list:
                tool_calls = getattr(s, "tool_calls", {}) or {}
                if isinstance(tool_calls, Mapping):
                    for k, v in tool_calls.items():
                        merged_tool_calls[k] = merged_tool_calls.get(k, 0) + int(v or 0)
                timeline = getattr(s, "timeline", None)
                if isinstance(timeline, list):
                    merged_timeline.extend(timeline)
                code_execs += int(getattr(s, "code_executions", 0) or 0)
                code_execs_successful += int(getattr(s, "code_executions_successful", 0) or 0)
                code_execs_failed += int(getattr(s, "code_executions_failed", 0) or 0)
                merged_lines_of_code += int(getattr(s, "lines_of_code", 0) or 0)
            stats = SimpleNamespace(
                tool_calls=merged_tool_calls,
                timeline=merged_timeline,
                code_executions=code_execs,
                code_executions_successful=code_execs_successful,
                code_executions_failed=code_execs_failed,
                lines_of_code=merged_lines_of_code,
            )

        if missing_ids:
            reg = getattr(session_stats_registry, "registry", {}) or {}
            iter_data = reg.get(iteration, {})
            research_logs = iter_data.get("research", {}) or {}
            sample_keys = list(research_logs.keys())[:12]
            log.debug(
                "[review][metrics] missing research stats "
                f"paper_id={pid} missing_agent_ids={missing_ids} "
                f"research_session_id={paper_session_id} "
                f"available_research_keys_count={len(research_logs)} "
                f"available_research_keys_sample={sample_keys}"
            )
        if stats is None:
            log.debug(
                "[review][metrics] no research stats resolved "
                f"paper_id={pid} primary_agent_id={paper.primary_agent_id} "
                f"research_session_id={paper_session_id} "
                f"collab_agent_ids={getattr(paper, 'collab_agent_ids', [])}"
            )
        else:
            timeline = getattr(stats, "timeline", None)
            tool_calls = getattr(stats, "tool_calls", {}) or {}
            code_execs_successful = int(getattr(stats, "code_executions_successful", 0) or 0)
            code_execs_failed = int(getattr(stats, "code_executions_failed", 0) or 0)
            code_execs = int(getattr(stats, "code_executions", 0) or 0)
            timeline_len = len(timeline) if isinstance(timeline, list) else 0
            tool_call_keys = list(tool_calls.keys())[:10] if isinstance(tool_calls, dict) else []
            timeline_exec_calls = 0
            timeline_exec_results = 0
            if isinstance(timeline, list):
                for ev in timeline:
                    if not isinstance(ev, dict):
                        continue
                    if ev.get("kind") == "tool_call" and ev.get("tool") == "execute_code":
                        timeline_exec_calls += 1
                    if ev.get("kind") == "tool_result_execute_code":
                        timeline_exec_results += 1
            log.debug(
                "[review][metrics] research stats found "
                f"paper_id={pid} primary_agent_id={paper.primary_agent_id} "
                f"collab_agent_ids={getattr(paper, 'collab_agent_ids', [])} "
                f"code_execs={code_execs} code_execs_successful={code_execs_successful} "
                f"code_execs_failed={code_execs_failed} timeline_len={timeline_len} "
                f"timeline_exec_calls={timeline_exec_calls} timeline_exec_results={timeline_exec_results} "
                f"tool_calls_sample={tool_call_keys}"
            )

        support = None
        if allowlist_by_category is None or allowlist_by_category.get("support"):
            support = _compute_support_metrics(paper, manuscript_text, stats, embedder=embedder)
            if allowlist_by_category is not None:
                support = {
                    k: v
                    for k, v in support.items()
                    if k in allowlist_by_category.get("support", set())
                }
        if stats is not None:
            try:
                code_execs_successful = int(getattr(stats, "code_executions_successful", 0) or 0)
            except Exception:
                code_execs_successful = 0
            if (
                support is not None
                and support.get("analytical_depth", 0) == 0
                and code_execs_successful > 0
            ):
                timeline = getattr(stats, "timeline", None)
                timeline_kinds = []
                if isinstance(timeline, list):
                    timeline_kinds = [ev.get("kind") for ev in timeline if isinstance(ev, dict)]
                timeline_kinds = [k for k in timeline_kinds if k][:12]
                log.debug(
                    "[review][metrics] analytical_depth zero despite code executions "
                    f"paper_id={pid} primary_agent_id={paper.primary_agent_id} "
                    f"code_execs_successful={code_execs_successful} "
                    f"timeline_kinds_sample={timeline_kinds}"
                )
        soundness = None
        if allowlist_by_category is None or allowlist_by_category.get("soundness"):
            soundness = _compute_soundness_metrics(manuscript_text, stats)
            if allowlist_by_category is not None:
                soundness = {
                    k: v
                    for k, v in soundness.items()
                    if k in allowlist_by_category.get("soundness", set())
                }

        significance = None
        if allowlist_by_category is None or allowlist_by_category.get("significance"):
            significance, _ = _compute_significance_metrics(paper, manuscript_text, archive)
            if allowlist_by_category is not None:
                significance = {
                    k: v
                    for k, v in significance.items()
                    if k in allowlist_by_category.get("significance", set())
                }

        originality = None
        if allowlist_by_category is None or allowlist_by_category.get("originality"):
            originality = _compute_originality_metrics(
                paper=paper,
                manuscript_text=manuscript_text,
                archive=archive,
                embedder=embedder,
                archive_texts_cache=archive_texts_cache,
            )
            if allowlist_by_category is not None:
                originality = {
                    k: v
                    for k, v in originality.items()
                    if k in allowlist_by_category.get("originality", set())
                }

        metric_block: dict[str, Any] = {}
        if support is not None:
            metric_block["support"] = support
        if soundness is not None:
            metric_block["soundness"] = soundness
        if significance is not None:
            metric_block["significance"] = significance
        if originality is not None:
            metric_block["originality"] = originality

        metrics_by_id[pid] = metric_block

    # Batch-level normalization to improve separation and reduce score collapse.
    orig_semantic_vals: list[float] = []
    for metrics in metrics_by_id.values():
        orig_sem = (metrics.get("originality", {}) or {}).get("semantic_novelty_knn_distance")
        if isinstance(orig_sem, (int, float)):
            orig_semantic_vals.append(float(orig_sem))

    for metrics in metrics_by_id.values():
        significance = metrics.get("significance", {}) or {}
        originality = metrics.get("originality", {}) or {}

        if significance and originality:
            sig_mean = significance.get("distance_to_prior_mean")
            if sig_mean is None and isinstance(
                originality.get("semantic_novelty_knn_distance"), (int, float)
            ):
                # Backfill significance novelty distance from originality distance when needed.
                if _metric_allowed(allowlist_by_category, "significance", "distance_to_prior_mean"):
                    sig_mean = float(originality["semantic_novelty_knn_distance"])
                    significance["distance_to_prior_mean"] = sig_mean
                if _metric_allowed(allowlist_by_category, "significance", "distance_to_prior_min"):
                    significance["distance_to_prior_min"] = sig_mean
                if _metric_allowed(allowlist_by_category, "significance", "distance_to_prior_max"):
                    significance["distance_to_prior_max"] = sig_mean

        if originality:
            sem_dist = originality.get("semantic_novelty_knn_distance")
            if _metric_allowed(
                allowlist_by_category, "originality", "embedding_novelty_percentile"
            ):
                if isinstance(sem_dist, (int, float)) and orig_semantic_vals:
                    originality["embedding_novelty_percentile"] = _percentile_rank(
                        orig_semantic_vals, float(sem_dist)
                    )
                else:
                    originality["embedding_novelty_percentile"] = None

        if significance:
            metrics["significance"] = significance
        if originality:
            metrics["originality"] = originality

    return metrics_by_id
