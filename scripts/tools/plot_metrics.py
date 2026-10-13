"""
Plot per-agent metric trajectories and Pareto fronts from snapshot metrics.jsonl files.
Outputs interactive Plotly HTML with hover details from private archives.

Usage:
    python scripts/tools/plot_metrics.py --snapshots-root snapshots --run-name <run> \
        --out-dir plots \
        --pareto-x significance.effect_size_strength \
        --pareto-y originality.semantic_novelty_knn_distance

Assumptions:
  - Each snapshot iteration directory contains a metrics.jsonl written by create_snapshot
    with fields: iteration, paper_id, primary_agent_id, paper_title, metrics{group->{metric:value}}.
  - private_archive_data.json exists for hover details (optional).
  - plotly is installed.
"""

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import plotly.io as pio


def _flatten_metrics(metrics: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Flatten nested metric groups into dot-separated keys."""
    flat: dict[str, float] = {}
    for group, vals in (metrics or {}).items():
        if not isinstance(vals, dict):
            continue
        for k, v in vals.items():
            key = f"{group}.{k}"
            try:
                flat[key] = float(v)
            except Exception:
                # skip non-numeric
                continue
    return flat


def load_metrics(snapshots_root: Path, run_name: str) -> list[dict[str, Any]]:
    """Load all metrics.jsonl entries for a run."""
    records: list[dict[str, Any]] = []
    # glob manually to retain iteration number
    for metrics_path in sorted((snapshots_root / run_name).glob("iteration_*/metrics.jsonl")):
        try:
            iteration = int(metrics_path.parent.name.split("_")[-1])
        except Exception:
            iteration = None
        with metrics_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                entry["iteration"] = entry.get("iteration", iteration)
                entry["flat_metrics"] = _flatten_metrics(entry.get("metrics", {}))
                records.append(entry)
    return records


def load_private_archives(snapshots_root: Path, run_name: str) -> dict[int, dict[str, Any]]:
    """Load private archive snapshots keyed by iteration (best-effort)."""
    archives: dict[int, dict[str, Any]] = {}
    for archive_path in sorted(
        (snapshots_root / run_name).glob("iteration_*/private_archive_data.json")
    ):
        try:
            iteration = int(archive_path.parent.name.split("_")[-1])
        except Exception:
            continue
        try:
            with archive_path.open("r", encoding="utf-8") as f:
                archives[iteration] = json.load(f)
        except Exception:
            continue
    return archives


def _build_private_index(
    private_archives: dict[int, dict[str, Any]],
) -> dict[int, dict[str, dict[str, Any]]]:
    """Index private archive data by iteration and paper_id."""

    def _iter_papers(papers_obj: Any):
        if not papers_obj:
            return
        if isinstance(papers_obj, list):
            for item in papers_obj:
                # Some snapshots store papers as list-of-lists.
                if isinstance(item, list):
                    for nested in item:
                        if isinstance(nested, dict):
                            yield nested
                elif isinstance(item, dict):
                    yield item
        elif isinstance(papers_obj, dict):
            yield papers_obj

    indexed: dict[int, dict[str, dict[str, Any]]] = {}
    for iteration, data in private_archives.items():
        papers = {}
        for paper in _iter_papers(data.get("papers")):
            paper_id = paper.get("paper_id")
            if paper_id:
                papers[paper_id] = paper
        indexed[iteration] = {
            "papers": papers,
            "reviews": dict(data.get("reviews_by_paper") or {}),
        }
    return indexed


def _select_private_snapshot(
    private_index: dict[int, dict[str, dict[str, Any]]],
    iteration: int | None,
) -> dict[str, dict[str, Any]] | None:
    if not private_index:
        return None
    if iteration is not None and iteration in private_index:
        return private_index[iteration]
    if iteration is not None:
        earlier = [it for it in private_index.keys() if it <= iteration]
        if earlier:
            return private_index[max(earlier)]
    return private_index[max(private_index.keys())]


def _get_paper_status(
    record: dict[str, Any],
    private_index: dict[int, dict[str, dict[str, Any]]],
) -> str | None:
    paper_id = record.get("paper_id")
    iteration = record.get("iteration")
    snapshot = _select_private_snapshot(private_index, iteration)
    if not snapshot:
        return None
    paper = snapshot["papers"].get(paper_id) if snapshot.get("papers") else None
    if not paper:
        return None
    status = paper.get("status")
    return str(status).lower() if status is not None else None


def _escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _clean_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).replace("\r", " ").replace("\n", " ")
    return _escape_html(text)


def _wrap_text(text: Any, width: int = 90) -> str:
    cleaned = _clean_text(text)
    if not cleaned:
        return ""
    return textwrap.fill(cleaned, width=width, break_long_words=True, break_on_hyphens=False)


def _format_metric_value(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.4g}"
    return _clean_text(value)


def _format_scores(scores: dict[str, Any]) -> str:
    if not scores:
        return "no scores"
    parts = []
    for key in sorted(scores.keys()):
        parts.append(f"{key}={scores.get(key)}")
    return ", ".join(parts)


def _label(text: str) -> str:
    return f"== {text.upper()} =="


def _format_reviews(
    reviews: list[dict[str, Any]],
    max_reviews: int = 2,
    include_text: bool = True,
    max_chars: int | None = None,
    wrap_width: int = 100,
) -> str:
    if not reviews:
        return "Reviews: none"
    chunks = []
    for entry in reviews[:max_reviews]:
        reviewer = _clean_text(entry.get("reviewer_id") or "unknown")
        scores = _format_scores(entry.get("scores") or {})
        review_text = _wrap_text(entry.get("review_text", ""), wrap_width) if include_text else ""
        if review_text:
            chunks.append(f"{reviewer} ({scores}): {review_text}")
        else:
            chunks.append(f"{reviewer} ({scores})")
    if len(reviews) > max_reviews:
        chunks.append(f"... +{len(reviews) - max_reviews} more")
    combined = f"{_label('Reviews')}: " + " | ".join(chunks)
    if max_chars is not None and len(combined) > max_chars:
        trimmed = combined[: max_chars - 3].rstrip()
        return trimmed + "..."
    return combined


def _build_hover_text(
    record: dict[str, Any],
    metric_name: str,
    metric_value: float,
    private_index: dict[int, dict[str, dict[str, Any]]],
    extra_metrics: list[tuple[str, Any]] | None = None,
    hover_max_chars: int | None = None,
    hover_max_lines: int | None = None,
    hover_wrap_width: int | None = None,
) -> str:
    iteration = record.get("iteration")
    paper_id = record.get("paper_id") or "unknown"
    record.get("primary_agent_id") or "unknown"
    title = record.get("paper_title")

    base_parts = [f"{_label('Paper ID')}: {_clean_text(paper_id)}"]
    wrap_width = hover_wrap_width or 100
    if title:
        base_parts.append(f"{_label('Title')}: {_wrap_text(title, wrap_width)}")
    # Intentionally omit metric values to prioritize review text.
    metric_parts: list[str] = []

    meta_parts: list[str] = []
    meta_text_part: str | None = None
    paper_text_part: str | None = None
    reviews_part: str | None = None
    reviews_summary_part: str | None = None
    review_entries: list[dict[str, Any]] = []

    snapshot = _select_private_snapshot(private_index, iteration)
    if snapshot:
        paper = snapshot["papers"].get(paper_id) if snapshot.get("papers") else None
        if paper:
            status = paper.get("status")
            paper.get("citation_count")
            collabs = paper.get("collab_agent_ids") or []
            if status:
                meta_parts.append(f"{_label('Status')}: {_clean_text(status)}")
            if collabs:
                meta_parts.append(
                    f"{_label('Collaborators')}: {_clean_text(', '.join(map(str, collabs)))}"
                )
            metareview = paper.get("metareview") or {}
            manuscript = paper.get("manuscript")
            if metareview:
                decision = metareview.get("decision")
                overall = metareview.get("overall_score")
                rank = metareview.get("rank")
                meta_text = metareview.get("meta_review_text")
                meta_bits = []
                if decision:
                    meta_bits.append(f"decision={decision}")
                if overall is not None:
                    meta_bits.append(f"score={overall}")
                if rank:
                    meta_bits.append(f"rank={rank}")
                if meta_bits:
                    meta_parts.append(
                        f"{_label('Metareview')}: " + _clean_text("; ".join(meta_bits))
                    )
                if meta_text:
                    meta_text_part = f"{_label('Metareview text')}: " + _wrap_text(
                        meta_text, wrap_width
                    )
            if manuscript:
                paper_text_part = f"{_label('Paper text')}: " + _wrap_text(manuscript, wrap_width)
        if snapshot.get("reviews"):
            review_entries = snapshot["reviews"].get(paper_id) or []
        reviews_part = _format_reviews(review_entries, include_text=True, wrap_width=wrap_width)
        reviews_summary_part = _format_reviews(
            review_entries, include_text=False, wrap_width=wrap_width
        )

    def _join_wrapped(parts: list[str]) -> str:
        wrapped_parts = []
        for part in parts:
            wrapped = _wrap_text(part, wrap_width)
            if wrapped:
                wrapped_parts.append(wrapped.replace("\n", "<br>"))
        return "<br>".join(wrapped_parts)

    pre_review_parts: list[str] = []
    pre_review_parts.extend(base_parts)
    pre_review_parts.extend(metric_parts)
    pre_review_parts.extend(meta_parts)
    if meta_text_part:
        pre_review_parts.append(meta_text_part)
    if paper_text_part:
        pre_review_parts.append(paper_text_part)

    pre_hover = _join_wrapped(pre_review_parts)
    if not reviews_part:
        return pre_hover

    if not hover_max_chars:
        return _join_wrapped(pre_review_parts + [reviews_part])

    remaining = max(0, hover_max_chars - len(pre_hover) - 4)
    if remaining <= 0:
        return pre_hover

    trimmed_reviews = _format_reviews(
        review_entries,
        include_text=True,
        max_chars=remaining,
        wrap_width=wrap_width,
    )
    hover = _join_wrapped(pre_review_parts + [trimmed_reviews])
    if len(hover) > hover_max_chars:
        # Last resort: drop review text, keep scores.
        hover = _join_wrapped(pre_review_parts + [reviews_summary_part or "Reviews: none"])
    if hover_max_lines:
        lines = hover.split("<br>")
        if len(lines) > hover_max_lines:
            hover = "<br>".join(lines[:hover_max_lines]) + "<br>..."
    return hover


def plot_metric_lines(
    records: list[dict[str, Any]],
    out_dir: Path,
    private_index: dict[int, dict[str, dict[str, Any]]],
    hover_max_chars: int | None = None,
    hover_max_lines: int | None = None,
    hover_wrap_width: int | None = None,
) -> None:
    """Plot per-metric line plots (iteration on x, one line per research agent)."""
    # Collect all metric keys
    metric_keys = set()
    for r in records:
        metric_keys.update(r.get("flat_metrics", {}).keys())

    out_dir.mkdir(parents=True, exist_ok=True)

    for metric in sorted(metric_keys):
        series: dict[str, list[dict[str, Any]]] = {}
        for r in records:
            agent = r.get("primary_agent_id") or "unknown"
            val = r["flat_metrics"].get(metric)
            it = r.get("iteration")
            if val is None or it is None:
                continue
            status = _get_paper_status(r, private_index)
            symbol = "x" if status == "rejected" else "circle"
            hover = _build_hover_text(
                r,
                metric,
                val,
                private_index,
                hover_max_chars=hover_max_chars,
                hover_max_lines=hover_max_lines,
                hover_wrap_width=hover_wrap_width,
            )
            series.setdefault(agent, []).append(
                {"iteration": it, "value": val, "hover": hover, "symbol": symbol}
            )

        if not series:
            continue

        fig = go.Figure()
        for agent, points in series.items():
            points_sorted = sorted(points, key=lambda x: x["iteration"])
            xs = [p["iteration"] for p in points_sorted]
            ys = [p["value"] for p in points_sorted]
            hovers = [p["hover"] for p in points_sorted]
            symbols = [p["symbol"] for p in points_sorted]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines+markers",
                    name=str(agent),
                    customdata=hovers,
                    hovertemplate="%{customdata}<extra></extra>",
                    marker={"size": 10, "symbol": symbols},
                )
            )

        fig.update_layout(
            title=metric,
            xaxis_title="Iteration",
            yaxis_title=metric,
            hovermode="closest",
            hoverdistance=80,
            spikedistance=80,
            legend={
                "yanchor": "middle",
                "y": 0.5,
                "xanchor": "left",
                "x": 1.02,
                "borderwidth": 0,
            },
            margin={"l": 60, "r": 240, "t": 60, "b": 60},
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        pio.write_html(
            fig,
            out_dir / f"{metric.replace('.', '_')}.html",
            include_plotlyjs="cdn",
            full_html=True,
        )


def pareto_front(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """
    Compute Pareto frontier for maximization in both dimensions.
    Returns points on the frontier sorted by x descending.
    """
    if not points:
        return []
    pts = sorted(points, key=lambda p: (p[0], p[1]), reverse=True)
    frontier = []
    best_y = float("-inf")
    for x, y in pts:
        if y >= best_y:
            frontier.append((x, y))
            best_y = y
    return frontier


def plot_pareto(
    records: list[dict[str, Any]],
    out_dir: Path,
    metric_x: str,
    metric_y: str,
    iteration: int | None = None,
    private_index: dict[int, dict[str, dict[str, Any]]] | None = None,
    hover_max_chars: int | None = None,
    hover_max_lines: int | None = None,
    hover_wrap_width: int | None = None,
) -> None:
    """
    Scatter plot of metric_y vs metric_x with Pareto frontier overlay.
    If iteration is None, uses latest iteration found.
    """
    if not records:
        return
    iters = [r.get("iteration") for r in records if r.get("iteration") is not None]
    target_iter = iteration if iteration is not None else (max(iters) if iters else None)
    if target_iter is None:
        return

    points = []
    for r in records:
        if r.get("iteration") != target_iter:
            continue
        fx = r["flat_metrics"].get(metric_x)
        fy = r["flat_metrics"].get(metric_y)
        if fx is None or fy is None:
            continue
        hover = _build_hover_text(
            r,
            metric_x,
            fx,
            private_index or {},
            extra_metrics=[(metric_y, fy)],
            hover_max_chars=hover_max_chars,
            hover_max_lines=hover_max_lines,
            hover_wrap_width=hover_wrap_width,
        )
        status = _get_paper_status(r, private_index or {})
        symbol = "x" if status == "rejected" else "circle"
        points.append({"x": fx, "y": fy, "hover": hover, "symbol": symbol})

    if not points:
        return

    frontier = pareto_front([(p["x"], p["y"]) for p in points])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[p["x"] for p in points],
            y=[p["y"] for p in points],
            mode="markers",
            marker={
                "color": "gray",
                "opacity": 0.6,
                "size": 10,
                "symbol": [p["symbol"] for p in points],
            },
            customdata=[p["hover"] for p in points],
            hovertemplate="%{customdata}<extra></extra>",
            name="Papers",
        )
    )

    if frontier:
        fx, fy = zip(*frontier, strict=False)
        fig.add_trace(
            go.Scatter(
                x=fx,
                y=fy,
                mode="lines",
                line={"color": "red", "width": 2},
                name="Pareto frontier",
            )
        )

    fig.update_layout(
        title=f"Pareto view (iteration {target_iter})",
        xaxis_title=metric_x,
        yaxis_title=metric_y,
        hovermode="closest",
        hoverdistance=50,
        spikedistance=50,
        legend={
            "yanchor": "middle",
            "y": 0.5,
            "xanchor": "left",
            "x": 1.02,
            "borderwidth": 0,
        },
        margin={"l": 60, "r": 240, "t": 60, "b": 60},
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    pio.write_html(
        fig,
        out_dir / f"pareto_{metric_x.replace('.', '_')}_{metric_y.replace('.', '_')}.html",
        include_plotlyjs="cdn",
        full_html=True,
    )


def main():
    ap = argparse.ArgumentParser(description="Plot metrics over time and Pareto fronts.")
    ap.add_argument(
        "--snapshots-root", type=Path, default=Path("snapshots"), help="Root snapshots dir"
    )
    ap.add_argument("--run-name", required=True, help="Run name under snapshots/")
    ap.add_argument(
        "--out-dir", type=Path, default=Path("plots"), help="Output directory for plots"
    )
    ap.add_argument(
        "--pareto-x",
        default="significance.effect_size_strength",
        help="Metric key for Pareto X axis",
    )
    ap.add_argument(
        "--pareto-y",
        default="originality.semantic_novelty_knn_distance",
        help="Metric key for Pareto Y axis",
    )
    ap.add_argument(
        "--pareto-iteration",
        type=int,
        default=None,
        help="Iteration to plot Pareto (default: latest)",
    )
    ap.add_argument(
        "--hover-max-chars",
        type=int,
        default=6000,
        help="Max hover text length; reduce to avoid broken hovers, set 0 to disable.",
    )
    ap.add_argument(
        "--hover-max-lines",
        type=int,
        default=40,
        help="Max hover line count; reduce to avoid off-screen tooltips, set 0 to disable.",
    )
    ap.add_argument(
        "--hover-wrap-width",
        type=int,
        default=200,
        help="Hover text wrap width; increase to make tooltip wider.",
    )
    args = ap.parse_args()

    records = load_metrics(args.snapshots_root, args.run_name)
    if not records:
        print(
            "No metrics found. Ensure metrics.jsonl exist under snapshots/<run-name>/iteration_*/"
        )
        return

    private_archives = load_private_archives(args.snapshots_root, args.run_name)
    private_index = _build_private_index(private_archives)

    base_out = args.out_dir / args.run_name
    hover_max = args.hover_max_chars if args.hover_max_chars and args.hover_max_chars > 0 else None
    hover_lines = (
        args.hover_max_lines if args.hover_max_lines and args.hover_max_lines > 0 else None
    )
    hover_wrap = (
        args.hover_wrap_width if args.hover_wrap_width and args.hover_wrap_width > 0 else None
    )
    plot_metric_lines(
        records,
        base_out / "lines",
        private_index,
        hover_max_chars=hover_max,
        hover_max_lines=hover_lines,
        hover_wrap_width=hover_wrap,
    )
    plot_pareto(
        records,
        base_out / "pareto",
        args.pareto_x,
        args.pareto_y,
        iteration=args.pareto_iteration,
        private_index=private_index,
        hover_max_chars=hover_max,
        hover_max_lines=hover_lines,
        hover_wrap_width=hover_wrap,
    )
    print(f"Plots written to {base_out}")


if __name__ == "__main__":
    main()
