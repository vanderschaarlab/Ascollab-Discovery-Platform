import hashlib
import json
import math
import re
import statistics
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

DEFAULT_GRAPH_LOGIC_CFG: dict[str, Any] = {
    "enabled": False,
    "require_plot_citations_in_research": True,
    "plot_only_rewrite_cap": 2,
    "allow_plot_validation_waiver_after_cap": True,
    "summary": {
        "backend": "code_stats_llm",
        "fallback_order": [
            "code_stats_llm",
            "agent_sidecar",
            "heuristic_fallback_only",
        ],
        "allow_vision_fallback": False,
        "timeout_seconds": 90,
        "max_retries": 2,
        "include_descriptive_stats": True,
        "include_comparative_stats": True,
        "max_series_per_plot": 12,
    },
}


_NUM_RE = re.compile(r"(?<![A-Za-z0-9_])[+-]?(?:\d+\.\d+|\d+)(?:[eE][+-]?\d+)?")
_LABEL_CALL_RE = re.compile(
    r"\b(?:plt|ax)\.(?:set_)?(?P<kind>xlabel|ylabel|title)\(\s*(?:label\s*=\s*)?['\"](?P<value>[^'\"]{1,200})['\"]",
    flags=re.IGNORECASE,
)
_LIMIT_CALL_RE = re.compile(
    r"\b(?:plt|ax)\.(?:set_)?(?P<kind>xlim|ylim)\(\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\)",
    flags=re.IGNORECASE,
)
_SERIES_LABEL_RE = re.compile(r"label\s*=\s*['\"]([^'\"]{1,120})['\"]", flags=re.IGNORECASE)
_P_VALUE_RE = re.compile(r"\bp\s*[<=>]\s*([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)", flags=re.IGNORECASE)


def _deepcopy_dict(obj: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in obj.items():
        if isinstance(v, Mapping):
            out[str(k)] = _deepcopy_dict(v)
        elif isinstance(v, list):
            out[str(k)] = list(v)
        else:
            out[str(k)] = v
    return out


def normalize_graph_logic_cfg(cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    base = _deepcopy_dict(DEFAULT_GRAPH_LOGIC_CFG)
    incoming = _deepcopy_dict(cfg or {})
    base["enabled"] = bool(incoming.get("enabled", base["enabled"]))
    base["require_plot_citations_in_research"] = bool(
        incoming.get(
            "require_plot_citations_in_research",
            base["require_plot_citations_in_research"],
        )
    )
    try:
        rewrite_cap = int(
            incoming.get(
                "plot_only_rewrite_cap",
                base.get("plot_only_rewrite_cap", 2),
            )
        )
    except Exception:
        rewrite_cap = int(base.get("plot_only_rewrite_cap", 2))
    base["plot_only_rewrite_cap"] = max(0, rewrite_cap)
    base["allow_plot_validation_waiver_after_cap"] = bool(
        incoming.get(
            "allow_plot_validation_waiver_after_cap",
            base.get("allow_plot_validation_waiver_after_cap", True),
        )
    )
    summary_in = (
        incoming.get("summary", {}) if isinstance(incoming.get("summary", {}), Mapping) else {}
    )
    summary = _deepcopy_dict(base.get("summary", {}))
    summary.update({str(k): v for k, v in summary_in.items()})
    backend = str(summary.get("backend", "code_stats_llm") or "code_stats_llm").strip().lower()
    if not backend:
        backend = "code_stats_llm"
    summary["backend"] = backend
    fallbacks = summary.get("fallback_order")
    if not isinstance(fallbacks, list) or not fallbacks:
        fallbacks = [backend]
    cleaned = [str(x).strip().lower() for x in fallbacks if str(x).strip()]
    if backend not in cleaned:
        cleaned.insert(0, backend)
    if "heuristic_fallback_only" not in cleaned:
        cleaned.append("heuristic_fallback_only")
    summary["fallback_order"] = cleaned
    summary["allow_vision_fallback"] = bool(summary.get("allow_vision_fallback", False))
    summary["include_descriptive_stats"] = bool(summary.get("include_descriptive_stats", True))
    summary["include_comparative_stats"] = bool(summary.get("include_comparative_stats", True))
    try:
        summary["max_series_per_plot"] = max(1, int(summary.get("max_series_per_plot", 12)))
    except Exception:
        summary["max_series_per_plot"] = 12
    base["summary"] = summary
    return base


def is_graph_logic_enabled(cfg: Mapping[str, Any] | None) -> bool:
    return bool(normalize_graph_logic_cfg(cfg).get("enabled", False))


def _coerce_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        f = float(v)
        if math.isfinite(f):
            return f
    except Exception:
        return None
    return None


def _extract_numeric_values(text: str, limit: int = 2000) -> list[float]:
    if not text:
        return []
    vals: list[float] = []
    for m in _NUM_RE.finditer(text):
        f = _coerce_float(m.group(0))
        if f is None:
            continue
        vals.append(f)
        if len(vals) >= limit:
            break
    return vals


def _series_stats(name: str, values: Sequence[float]) -> dict[str, Any]:
    vals = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if not vals:
        return {
            "name": name,
            "n": 0,
            "missing_n": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
            "q1": None,
            "q3": None,
            "iqr": None,
        }
    vals_sorted = sorted(vals)
    q1_idx = max(0, int((len(vals_sorted) - 1) * 0.25))
    q3_idx = max(0, int((len(vals_sorted) - 1) * 0.75))
    q1 = vals_sorted[q1_idx]
    q3 = vals_sorted[q3_idx]
    try:
        std_val = statistics.pstdev(vals_sorted) if len(vals_sorted) > 1 else 0.0
    except Exception:
        std_val = 0.0
    return {
        "name": name,
        "n": len(vals_sorted),
        "missing_n": 0,
        "min": min(vals_sorted),
        "max": max(vals_sorted),
        "mean": statistics.fmean(vals_sorted),
        "median": statistics.median(vals_sorted),
        "std": std_val,
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
    }


def _extract_code_hints(code: str, *, max_series_per_plot: int = 12) -> dict[str, Any]:
    out: dict[str, Any] = {
        "title": "",
        "x_label": "",
        "y_label": "",
        "x_range": {"min": None, "max": None},
        "y_range": {"min": None, "max": None},
        "plot_type": "other",
        "series_names": [],
    }
    if not code:
        return out

    for m in _LABEL_CALL_RE.finditer(code):
        kind = str(m.group("kind") or "").lower()
        value = str(m.group("value") or "").strip()
        if not value:
            continue
        if kind == "title":
            out["title"] = value
        elif kind == "xlabel":
            out["x_label"] = value
        elif kind == "ylabel":
            out["y_label"] = value

    for m in _LIMIT_CALL_RE.finditer(code):
        kind = str(m.group("kind") or "").lower()
        mn = _coerce_float(m.group(2))
        mx = _coerce_float(m.group(3))
        if kind == "xlim":
            out["x_range"] = {"min": mn, "max": mx}
        elif kind == "ylim":
            out["y_range"] = {"min": mn, "max": mx}

    types = [
        ("kaplan_meier", ("kaplan", "kmf", "survival")),
        ("heatmap", ("heatmap", "imshow", "pcolor")),
        ("scatter", ("scatter(",)),
        ("histogram", ("hist(",)),
        ("box", ("boxplot(", "violinplot(")),
        ("bar", ("bar(",)),
        ("line", ("plot(", "lineplot(")),
    ]
    lowered = code.lower()
    for ptype, keys in types:
        if any(k in lowered for k in keys):
            out["plot_type"] = ptype
            break

    labels = []
    for m in _SERIES_LABEL_RE.finditer(code):
        label = str(m.group(1) or "").strip()
        if label and label not in labels:
            labels.append(label)
    out["series_names"] = labels[: max(1, int(max_series_per_plot))]
    return out


def _stats_from_texts(
    *,
    stdout: str,
    stderr: str,
    result_text: str,
    series_names: list[str],
    max_series_per_plot: int,
    include_descriptive_stats: bool,
    include_comparative_stats: bool,
) -> dict[str, Any]:
    values = []
    values.extend(_extract_numeric_values(stdout))
    values.extend(_extract_numeric_values(result_text))
    # Ignore stderr values unless no other values were found.
    if not values:
        values.extend(_extract_numeric_values(stderr))
    values = values[:5000]

    names = list(series_names[:max_series_per_plot]) if series_names else []
    if not names:
        names = ["series_1"]
    n_series = min(max_series_per_plot, max(1, len(names)))
    names = names[:n_series]

    chunks: list[list[float]] = [[] for _ in range(n_series)]
    if values:
        if len(values) < n_series:
            chunks[0] = values
        else:
            per = max(1, len(values) // n_series)
            idx = 0
            for i in range(n_series):
                if i == n_series - 1:
                    chunks[i] = values[idx:]
                else:
                    chunks[i] = values[idx : idx + per]
                idx += per

    computed_series = [_series_stats(names[i], chunks[i]) for i in range(n_series)]
    series = list(computed_series) if include_descriptive_stats else []
    comparisons: list[dict[str, Any]] = []
    if include_comparative_stats and len(computed_series) >= 2:
        a = computed_series[0]
        b = computed_series[1]
        am = _coerce_float(a.get("mean"))
        bm = _coerce_float(b.get("mean"))
        asd = _coerce_float(a.get("std"))
        bsd = _coerce_float(b.get("std"))
        n_a = int(a.get("n", 0) or 0)
        n_b = int(b.get("n", 0) or 0)
        pooled = None
        if asd is not None and bsd is not None and n_a > 1 and n_b > 1:
            pooled_den = max(1.0, float(n_a + n_b - 2))
            pooled_var = ((n_a - 1) * (asd**2) + (n_b - 1) * (bsd**2)) / pooled_den
            pooled = math.sqrt(max(0.0, pooled_var))
        effect = None
        if pooled and pooled > 0 and am is not None and bm is not None:
            effect = (am - bm) / pooled
        p_val = None
        for src in (stdout, result_text, stderr):
            pm = _P_VALUE_RE.search(src or "")
            if pm:
                p_val = _coerce_float(pm.group(1))
                if p_val is not None:
                    break
        comparisons.append(
            {
                "lhs": str(a.get("name") or "series_1"),
                "rhs": str(b.get("name") or "series_2"),
                "delta_mean": None if am is None or bm is None else (am - bm),
                "effect_size": effect,
                "p_value": p_val,
            }
        )

    annotations = {
        "error_bars": any(
            k in (stdout + result_text).lower() for k in ("error bar", "stderr", "std")
        ),
        "confidence_intervals": any(
            k in (stdout + result_text).lower() for k in ("confidence interval", "ci", "95%")
        ),
        "p_value_text": _P_VALUE_RE.search(stdout + "\n" + result_text + "\n" + stderr) is not None,
    }

    return {
        "series": series,
        "comparisons": comparisons,
        "annotations_detected": annotations,
    }


def _completeness_score(summary: Mapping[str, Any]) -> float:
    stats = (
        summary.get("statistics", {}) if isinstance(summary.get("statistics", {}), Mapping) else {}
    )
    axes = summary.get("axes", {}) if isinstance(summary.get("axes", {}), Mapping) else {}
    checks = [
        bool(summary.get("title")),
        bool(summary.get("plot_type") and summary.get("plot_type") != "other"),
        bool(axes.get("x_label")),
        bool(axes.get("y_label")),
        bool(
            (axes.get("x_range") or {}).get("min") is not None
            or (axes.get("x_range") or {}).get("max") is not None
        ),
        bool(
            (axes.get("y_range") or {}).get("min") is not None
            or (axes.get("y_range") or {}).get("max") is not None
        ),
        bool(stats.get("series")),
        bool(summary.get("point_of_plot")),
        bool(summary.get("interpretation_guidance")),
    ]
    return round(float(sum(1 for c in checks if c)) / float(len(checks)), 3)


def _build_fallback_summary(
    *,
    plot_filename: str,
    plot_path: str,
    backend: str,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    warning_list = list(warnings or [])
    return {
        "plot_filename": plot_filename,
        "plot_path": plot_path,
        "summary_version": "v1",
        "generation_backend": backend,
        "plot_type": "other",
        "axes": {
            "x_label": "",
            "y_label": "",
            "x_range": {"min": None, "max": None},
            "y_range": {"min": None, "max": None},
        },
        "statistics": {
            "series": [],
            "comparisons": [],
            "annotations_detected": {
                "error_bars": False,
                "confidence_intervals": False,
                "p_value_text": False,
            },
            "stats_completeness": 0.0,
        },
        "title": "",
        "point_of_plot": "Plot generated; detailed statistical context unavailable.",
        "statistical_takeaway": "No reliable summary statistics were extracted for this figure.",
        "interpretation_guidance": "Use accompanying code/output for quantitative interpretation.",
        "confidence": 0.2,
        "warnings": warning_list or ["fallback_summary_used"],
    }


def build_fallback_plot_summary(
    *,
    plot_filename: str,
    plot_path: str,
    backend: str = "heuristic_fallback_only",
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    return _build_fallback_summary(
        plot_filename=plot_filename,
        plot_path=plot_path,
        backend=backend,
        warnings=warnings,
    )


def _build_code_stats_summary(
    *,
    plot_filename: str,
    plot_path: str,
    code: str,
    resp: Mapping[str, Any] | None,
    summary_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    max_series_per_plot = int(summary_cfg.get("max_series_per_plot", 12) or 12)
    hints = _extract_code_hints(code or "", max_series_per_plot=max_series_per_plot)
    stdout = str((resp or {}).get("stdout") or "")
    stderr = str((resp or {}).get("stderr") or "")
    result_text = str((resp or {}).get("result") or "")
    stats = _stats_from_texts(
        stdout=stdout,
        stderr=stderr,
        result_text=result_text,
        series_names=list(hints.get("series_names") or []),
        max_series_per_plot=max_series_per_plot,
        include_descriptive_stats=bool(summary_cfg.get("include_descriptive_stats", True)),
        include_comparative_stats=bool(summary_cfg.get("include_comparative_stats", True)),
    )
    series = list(stats.get("series") or [])
    comparisons = list(stats.get("comparisons") or [])
    lead = series[0] if series else {}
    mean_txt = ""
    if lead and lead.get("mean") is not None:
        mean_txt = f" (mean={float(lead['mean']):.3g})"
    title = str(hints.get("title") or "").strip()
    plot_type = str(hints.get("plot_type") or "other")
    point_of_plot = f"Primary signal appears in {plot_type} plot{mean_txt}.".strip()
    statistical_takeaway = "Summary statistics extracted from execution outputs."
    if comparisons and comparisons[0].get("delta_mean") is not None:
        delta = float(comparisons[0]["delta_mean"])
        statistical_takeaway = (
            f"First comparison delta_mean={delta:.3g}; "
            "interpret with code-level context and sample size."
        )
    summary: dict[str, Any] = {
        "plot_filename": plot_filename,
        "plot_path": plot_path,
        "summary_version": "v1",
        "generation_backend": "code_stats_llm",
        "plot_type": plot_type,
        "axes": {
            "x_label": str(hints.get("x_label") or ""),
            "y_label": str(hints.get("y_label") or ""),
            "x_range": dict(hints.get("x_range") or {"min": None, "max": None}),
            "y_range": dict(hints.get("y_range") or {"min": None, "max": None}),
        },
        "statistics": stats,
        "title": title,
        "point_of_plot": point_of_plot,
        "statistical_takeaway": statistical_takeaway,
        "interpretation_guidance": "Validate key claims against the associated code cell and printed outputs.",
        "confidence": 0.75 if series else 0.45,
        "warnings": [],
    }
    descriptive_enabled = bool(summary_cfg.get("include_descriptive_stats", True))
    comparative_enabled = bool(summary_cfg.get("include_comparative_stats", True))
    summary["statistics"]["stats_completeness"] = _completeness_score(summary)
    if not title:
        summary["warnings"].append("title_not_detected")
    if descriptive_enabled and not series:
        summary["warnings"].append("no_numeric_stats_detected")
    if not descriptive_enabled:
        summary["warnings"].append("descriptive_stats_disabled_by_config")
    if not comparative_enabled:
        summary["warnings"].append("comparative_stats_disabled_by_config")
    return summary


def _sidecar_candidate(resp: Mapping[str, Any] | None, image_index: int) -> dict[str, Any] | None:
    if not isinstance(resp, Mapping):
        return None
    for key in ("plot_summary_data", "plot_summaries", "plot_summary"):
        payload = resp.get(key)
        if isinstance(payload, Mapping):
            return dict(payload)
        if isinstance(payload, list):
            if len(payload) >= image_index and isinstance(payload[image_index - 1], Mapping):
                return dict(payload[image_index - 1])
            for row in payload:
                if isinstance(row, Mapping) and int(row.get("image_index", -1) or -1) == int(
                    image_index
                ):
                    return dict(row)
    return None


def _build_sidecar_summary(
    *,
    plot_filename: str,
    plot_path: str,
    image_index: int,
    resp: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    row = _sidecar_candidate(resp, image_index)
    if not row:
        return None
    summary = _build_fallback_summary(
        plot_filename=plot_filename,
        plot_path=plot_path,
        backend="agent_sidecar",
    )
    summary["generation_backend"] = "agent_sidecar"
    for key in (
        "plot_type",
        "axes",
        "statistics",
        "title",
        "point_of_plot",
        "statistical_takeaway",
        "interpretation_guidance",
        "confidence",
        "warnings",
    ):
        if key in row:
            summary[key] = row[key]
    if not isinstance(summary.get("statistics"), Mapping):
        summary["statistics"] = {}
    stats = dict(summary.get("statistics", {}))
    stats["stats_completeness"] = _completeness_score(summary)
    summary["statistics"] = stats
    return summary


def generate_plot_summary_for_image(
    *,
    plot_filename: str,
    plot_path: str,
    image_index: int,
    code: str | None,
    resp: Mapping[str, Any] | None,
    graph_logic_cfg: Mapping[str, Any] | None,
) -> dict[str, Any]:
    cfg = normalize_graph_logic_cfg(graph_logic_cfg)
    summary_cfg = cfg.get("summary", {}) if isinstance(cfg.get("summary", {}), Mapping) else {}
    fallback_order = list(summary_cfg.get("fallback_order") or [])
    allow_vision_fallback = bool(summary_cfg.get("allow_vision_fallback", False))
    if not allow_vision_fallback:
        fallback_order = [b for b in fallback_order if b != "vision_llm"]
        if "heuristic_fallback_only" not in fallback_order:
            fallback_order.append("heuristic_fallback_only")
    warnings: list[str] = []

    for backend in fallback_order:
        b = str(backend or "").strip().lower()
        if not b:
            continue
        if b == "code_stats_llm":
            try:
                return _build_code_stats_summary(
                    plot_filename=plot_filename,
                    plot_path=plot_path,
                    code=str(code or ""),
                    resp=resp,
                    summary_cfg=summary_cfg,
                )
            except Exception as e:
                warnings.append(f"code_stats_llm_failed:{type(e).__name__}:{e}")
                continue
        if b == "agent_sidecar":
            try:
                sidecar = _build_sidecar_summary(
                    plot_filename=plot_filename,
                    plot_path=plot_path,
                    image_index=image_index,
                    resp=resp,
                )
                if sidecar:
                    sidecar.setdefault("warnings", [])
                    sidecar["warnings"] = list(sidecar.get("warnings") or []) + warnings
                    return sidecar
                warnings.append("agent_sidecar_missing")
            except Exception as e:
                warnings.append(f"agent_sidecar_failed:{type(e).__name__}:{e}")
            continue
        if b == "vision_llm":
            warnings.append("vision_llm_not_implemented")
            continue
        if b == "heuristic_fallback_only":
            return _build_fallback_summary(
                plot_filename=plot_filename,
                plot_path=plot_path,
                backend="heuristic_fallback_only",
                warnings=warnings or ["heuristic_fallback_only"],
            )

    return _build_fallback_summary(
        plot_filename=plot_filename,
        plot_path=plot_path,
        backend="heuristic_fallback_only",
        warnings=warnings or ["fallback_chain_exhausted"],
    )


def write_plot_summary_artifact(
    *,
    iteration_dir: Path,
    plot_filename: str,
    summary: Mapping[str, Any],
) -> str:
    summaries_dir = iteration_dir / "plots" / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    out_path = summaries_dir / f"{plot_filename}.json"
    out_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_path.relative_to(iteration_dir).as_posix()


def build_statistics_digest(summary: Mapping[str, Any]) -> dict[str, Any]:
    stats = (
        summary.get("statistics", {}) if isinstance(summary.get("statistics", {}), Mapping) else {}
    )
    series = list(stats.get("series") or [])
    comparisons = list(stats.get("comparisons") or [])
    first = series[0] if series else {}
    digest: dict[str, Any] = {
        "series_count": len(series),
        "first_series": {
            "name": first.get("name"),
            "n": first.get("n"),
            "mean": first.get("mean"),
            "std": first.get("std"),
            "min": first.get("min"),
            "max": first.get("max"),
        }
        if first
        else None,
        "comparison_count": len(comparisons),
        "first_comparison": comparisons[0] if comparisons else None,
        "stats_completeness": stats.get("stats_completeness"),
    }
    return digest


def build_plot_handle(plot_filename: str) -> str:
    token = str(plot_filename or "").strip()
    if not token:
        return ""
    digest = hashlib.sha1(token.encode("utf-8")).hexdigest()[:8]
    return f"plot_{digest}"


def build_plot_citation_token(plot_token: str) -> str:
    tok = str(plot_token or "").strip()
    if not tok:
        return ""
    return f"[Plot: {tok}]"


def build_plot_artifact_record(
    *,
    plot_filename: str,
    plot_path: str,
    summary_path: str,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    digest = build_statistics_digest(summary)
    plot_handle = build_plot_handle(plot_filename)
    return {
        "plot_filename": plot_filename,
        "plot_handle": plot_handle,
        "plot_citation": build_plot_citation_token(plot_handle),
        "plot_filename_citation": build_plot_citation_token(plot_filename),
        "plot_path": plot_path,
        "summary_path": summary_path,
        "summary_excerpt": str(
            summary.get("statistical_takeaway") or summary.get("point_of_plot") or ""
        )[:280],
        "statistics_digest": digest,
        "generation_backend": str(summary.get("generation_backend") or "unknown"),
    }


def average(values: Iterable[float | None]) -> float | None:
    nums = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if not nums:
        return None
    return float(sum(nums) / len(nums))
