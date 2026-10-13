"""Collect papers from experiment snapshots in ranker-compatible format.

This utility reads the private archive (private_archive_data.json) from
experiment snapshots and converts papers into the dict format that
run_ranker() expects.  It is intentionally separate from the core ranker
functionality.

Usage::

    from ranker.snapshot_import import collect_papers, to_ranker_format

    papers = collect_papers("snapshots/my_run", iterations="last")
    result = run_ranker(
        paper_a=to_ranker_format(papers[0]),
        paper_b=to_ranker_format(papers[1]),
        llm=llm,
    )
"""

import json
import logging
import re
from collections import deque
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - lightweight helpers do not need YAML
    yaml = None

logger = logging.getLogger(__name__)

_RANKER_KEYS = {"title", "abstract", "manuscript", "code"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def collect_papers(
    snapshot_dirs: str | Path | list[str | Path],
    iterations: str | list[int] = "last",
    cumulative: bool = True,
    include_rejected: bool = False,
    exclude_timeout: bool = True,
    paper_kind_filter: str = "all",
) -> list[dict[str, Any]]:
    """Collect papers from experiment snapshot(s) in ranker-compatible format.

    Reads from the private archive (private_archive_data.json), which contains
    all papers (accepted + rejected + timed-out) plus per-paper reviews.

    Args:
        snapshot_dirs: Path(s) to snapshot run directory(ies).
            e.g. "snapshots/ascollab_01_kirc_ascollab_seed_1_260101_1200"
        iterations: Which iterations to collect papers from.
            - "last": only the highest-numbered iteration (default)
            - "all": all iterations
            - [1, 3, 5]: specific iteration numbers
        cumulative: If True (default), include all papers up to and
            including the selected iteration (publication_t <= N) rather
            than only papers from that exact iteration (publication_t == N).
            Raises ValueError when used with a multi-element iteration list.
        include_rejected: If True, include rejected papers in addition to
            accepted papers. Default False (backward-compatible: accepted only).
        exclude_timeout: If True (default), remove timed-out papers.
            A paper is timed out if its paper_id ends with '_timeout' OR
            its manuscript field equals 'TIMEOUT'.
        paper_kind_filter: One of ``all``, ``standard_only``, ``meta_only``.
            When set, filter by ``paper_kind`` (``research`` vs ``meta_research``).

    Returns:
        List of paper dicts. Each dict contains all original snapshot fields
        plus provenance fields (source_snapshot, source_snapshot_dir,
        source_iteration, code) and a reviews list.
    """
    normalized_kind_filter = _normalize_paper_kind_filter(paper_kind_filter)
    if isinstance(snapshot_dirs, (str, Path)):
        snapshot_dirs = [snapshot_dirs]

    all_papers: list[dict[str, Any]] = []

    for snap_dir in snapshot_dirs:
        snap_path = Path(snap_dir)
        if not snap_path.is_dir():
            raise FileNotFoundError(f"Snapshot directory not found: {snap_path}")

        iter_nums = _discover_iterations(snap_path)
        if not iter_nums:
            raise FileNotFoundError(f"No iteration_N directories found in: {snap_path}")

        read_iter = _select_read_iteration(
            snap_path,
            iterations=iterations,
            available_iters=iter_nums,
            cumulative=cumulative,
        )
        run_name = _read_snapshot_metadata(snap_path, read_iter)
        papers, reviews_by_paper = _read_papers(
            snap_path,
            read_iter,
            include_rejected=include_rejected,
            exclude_timeout=exclude_timeout,
            paper_kind_filter=normalized_kind_filter,
        )

        # Status filter: accepted-only by default
        papers = _filter_by_status(papers, include_rejected)

        # Timeout filter: exclude timed-out papers by default
        papers = _filter_timeout(papers, exclude_timeout)

        papers = _filter_by_paper_kind(papers, normalized_kind_filter)

        # Iteration filter (existing logic, unchanged)
        filtered = _filter_papers(papers, iterations, iter_nums, snap_path, cumulative)

        for p in filtered:
            all_papers.append(_convert_paper(p, run_name, snap_path, reviews_by_paper))

    return all_papers


def to_ranker_format(paper: dict[str, Any]) -> dict[str, str]:
    """Strip provenance, return only the fields run_ranker() expects.

    Args:
        paper: A dict returned by collect_papers().

    Returns:
        Dict with keys: title, abstract, manuscript, code.
    """
    return {k: paper[k] for k in _RANKER_KEYS}


def paper_uid(paper: dict[str, Any]) -> str:
    """Key that identifies a paper across runs: ``<source_snapshot_dir>::<paper_id>``.

    Paper IDs (``paper_t<NN>_<agent_id>``) are unique within a run but repeat across runs, so any
    lookup over papers pooled from several snapshots must use this key.

    Args:
        paper: Paper dict as returned by :func:`collect_papers`.

    Returns:
        The run-qualified paper key.
    """
    return f"{paper.get('source_snapshot_dir', '')}::{paper['paper_id']}"


def collect_citable_papers(
    snapshot_dirs: str | Path | list[str | Path],
    iterations: str | list[int] = "last",
) -> list[dict[str, Any]]:
    """Every accepted paper of the runs up to the iteration read: the papers citations can name.

    Citations point to accepted (archive) papers of the same run. A ranker run that compares only
    a selection of papers still needs all of them to resolve each paper's citation chain.

    Args:
        snapshot_dirs: Path(s) to snapshot run directory(ies).
        iterations: The ranker's iteration setting. A list reads up to its largest iteration;
            ``"all"`` and ``"last"`` read up to the last one.

    Returns:
        Accepted, non-timed-out papers, cumulative up to that iteration.
    """
    read: str | list[int] = "last"
    if not isinstance(iterations, str):  # a list, tuple or Hydra ListConfig of iterations
        values = [int(i) for i in iterations]
        if values:
            read = [max(values)]
    return collect_papers(
        snapshot_dirs,
        iterations=read,
        cumulative=True,
        include_rejected=False,
        exclude_timeout=True,
    )


def build_citation_chains(
    papers: list[dict[str, Any]],
    citable: list[dict[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Resolve the transitive citation chain for every paper.

    Uses BFS to follow each paper's ``cited_paper_ids`` references
    transitively, collecting all ancestor papers (direct citations +
    their citations + ...). Citations are resolved within the citing paper's own run
    (``source_snapshot_dir``), so papers pooled from several runs never pick up another run's
    paper with the same ID.

    Args:
        papers: Paper dicts as returned by collect_papers(). Must include
            cited_paper_ids, paper_id, publication_t and source_snapshot_dir fields.
        citable: Papers that citations may resolve to, typically
            :func:`collect_citable_papers` for the same runs. Defaults to ``papers``; pass it when
            ``papers`` is only a selection, or ancestors outside the selection are lost.

    Returns:
        Dict mapping :func:`paper_uid` to a list of ancestor paper dicts, sorted
        by (publication_t, paper_id) ascending (oldest ancestors first).
        Papers with no citations map to an empty list.
    """
    if not papers:
        return {}

    # Lookup per run: source_snapshot_dir -> paper_id -> paper dict
    by_run: dict[str, dict[str, dict[str, Any]]] = {}
    for p in list(citable or []) + list(papers):
        by_run.setdefault(p.get("source_snapshot_dir", ""), {}).setdefault(p["paper_id"], p)

    result: dict[str, list[dict[str, Any]]] = {}

    for paper in papers:
        pid = paper["paper_id"]
        lookup = by_run.get(paper.get("source_snapshot_dir", ""), {})
        visited: set = {pid}  # exclude self from own chain
        ancestors: list[dict[str, Any]] = []

        # BFS queue: start from direct citations
        queue: deque = deque()
        for ref in paper.get("cited_paper_ids", []):
            cited_id = ref["paper_id"] if isinstance(ref, dict) else ref
            if cited_id not in visited:
                queue.append(cited_id)
                visited.add(cited_id)

        while queue:
            current_id = queue.popleft()
            if current_id not in lookup:
                logger.debug(
                    "Paper %s cites %s which is not in its run's papers; skipping.",
                    pid,
                    current_id,
                )
                continue

            current_paper = lookup[current_id]
            ancestors.append(current_paper)

            # Enqueue this paper's own citations
            for ref in current_paper.get("cited_paper_ids", []):
                next_id = ref["paper_id"] if isinstance(ref, dict) else ref
                if next_id not in visited:
                    queue.append(next_id)
                    visited.add(next_id)

        # Sort by (publication_t, paper_id) ascending
        ancestors.sort(key=lambda p: (p["publication_t"], p["paper_id"]))
        result[paper_uid(paper)] = ancestors

    return result


def collect_metrics(
    papers: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Extract per-paper heuristic quality metrics from snapshot iteration directories.

    Each paper's metrics live in ``<source_snapshot_dir>/iteration_<source_iteration>/metrics.jsonl``.
    Papers are grouped by ``(source_snapshot_dir, source_iteration)`` so each
    metrics file is read at most once.

    Args:
        papers: Paper dicts as returned by :func:`collect_papers`, each
            containing ``source_snapshot_dir`` and ``source_iteration``.

    Returns:
        Dict mapping :func:`paper_uid` to the full metrics record (the parsed
        JSONL line).  Papers whose ``metrics.jsonl`` is missing or that
        have no entry in the file are silently omitted.
    """
    # Group papers by (source_snapshot_dir, source_iteration)
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for paper in papers:
        key = (paper["source_snapshot_dir"], paper["source_iteration"])
        groups.setdefault(key, []).append(paper)

    result: dict[str, dict[str, Any]] = {}

    for (snap_dir, iteration), group_papers in groups.items():
        metrics_path = Path(snap_dir) / f"iteration_{iteration}" / "metrics.jsonl"
        if not metrics_path.exists():
            logger.debug("metrics.jsonl not found: %s", metrics_path)
            continue

        # Parse JSONL into a lookup keyed by paper_id
        metrics_by_id: dict[str, dict[str, Any]] = {}
        with open(metrics_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                metrics_by_id[record["paper_id"]] = record

        # Look up metrics for each paper in the group
        for paper in group_papers:
            pid = paper["paper_id"]
            if pid in metrics_by_id:
                result[paper_uid(paper)] = metrics_by_id[pid]

    return result


def collect_dataset_descriptions(
    papers: list[dict[str, Any]],
) -> dict[str, dict[str, str]]:
    """Build per-snapshot dataset descriptions from exp_config.yaml + DATASETS registry.

    Groups papers by ``source_snapshot_dir``, reads each snapshot's
    ``exp_config.yaml`` once, extracts ``discovery_dataset.dataset_keys``,
    and looks up descriptions from ``env_setup.datasets.DATASETS``.

    Args:
        papers: Paper dicts as returned by :func:`collect_papers`, each
            containing ``source_snapshot_dir``.

    Returns:
        Dict mapping ``source_snapshot_dir`` (str) to ``{dataset_key: description_text}``.
        Papers whose snapshot lacks ``exp_config.yaml`` or whose keys are not in
        ``DATASETS`` are logged as warnings and return an empty dict for that snapshot.
    """
    from env_setup.datasets import DATASETS

    if not papers:
        return {}

    # Group papers by source_snapshot_dir
    snapshot_dirs: dict[str, list[dict[str, Any]]] = {}
    for paper in papers:
        snap_dir = paper.get("source_snapshot_dir", "")
        if snap_dir:
            snapshot_dirs.setdefault(snap_dir, []).append(paper)

    result: dict[str, dict[str, str]] = {}

    for snap_dir in snapshot_dirs:
        snap_path = Path(snap_dir)
        dataset_keys = _read_exp_config_dataset_keys(snap_path)

        if dataset_keys is None:
            logger.warning(
                "No exp_config.yaml found for snapshot %s; dataset descriptions will be empty.",
                snap_dir,
            )
            result[snap_dir] = {}
            continue

        descriptions: dict[str, str] = {}
        for key in dataset_keys:
            if key in DATASETS:
                descriptions[key] = DATASETS[key]
            else:
                logger.warning(
                    "Dataset key %r from snapshot %s not found in DATASETS registry; skipping.",
                    key,
                    snap_dir,
                )

        result[snap_dir] = descriptions

    return result


def _read_exp_config_dataset_keys(snap_path: Path) -> list[str] | None:
    """Read exp_config.yaml and extract discovery_dataset.dataset_keys.

    Prefers the latest readable paper iteration, then falls back to the newest
    iteration containing an ``exp_config.yaml``. Dataset descriptions only need
    run configuration, so they should also work for lightweight snapshot
    fixtures that do not contain archived papers.

    Args:
        snap_path: Path to the snapshot run directory.

    Returns:
        List of dataset key strings, or None if the config cannot be read.
    """
    iter_nums = _discover_iterations(snap_path)
    if not iter_nums:
        return None

    preferred_iters: list[int] = []
    latest_iter = _latest_readable_iteration(snap_path)
    if latest_iter is not None:
        preferred_iters.append(latest_iter)
    preferred_iters.extend(
        iteration for iteration in reversed(iter_nums) if iteration not in preferred_iters
    )

    config_path = None
    for iteration in preferred_iters:
        candidate = snap_path / f"iteration_{iteration}" / "exp_config.yaml"
        if candidate.exists():
            config_path = candidate
            break

    if config_path is None:
        return None

    try:
        if yaml is None:
            logger.warning("PyYAML is not installed; cannot parse %s", config_path)
            return None
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except Exception as exc:
        logger.warning("Failed to parse exp_config.yaml at %s: %s", config_path, exc)
        return None

    if not isinstance(config, dict):
        return None

    discovery_dataset = config.get("discovery_dataset")
    if not isinstance(discovery_dataset, dict):
        return None

    dataset_keys = discovery_dataset.get("dataset_keys")
    if not isinstance(dataset_keys, list):
        return None

    return dataset_keys


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_ITER_RE = re.compile(r"^iteration_(\d+)$")


def _normalize_paper_kind_filter(raw_value: Any) -> str:
    value = str(raw_value or "all").strip().lower()
    if value in {"all", "standard_only", "meta_only"}:
        return value
    raise ValueError(
        f"Invalid paper_kind_filter={raw_value!r}. Expected one of: all | standard_only | meta_only."
    )


def _discover_iterations(snap_path: Path) -> list[int]:
    """Find iteration_N/ subdirectories, return sorted list of ints."""
    if not snap_path.exists() or not snap_path.is_dir():
        return []

    nums = []
    for child in snap_path.iterdir():
        if child.is_dir():
            m = _ITER_RE.match(child.name)
            if m:
                nums.append(int(m.group(1)))
    nums.sort()
    return nums


def _iteration_has_metadata(snap_path: Path, iteration: int) -> bool:
    return (snap_path / f"iteration_{iteration}" / "snapshot_metadata.json").exists()


def _iteration_has_paper_data(snap_path: Path, iteration: int) -> bool:
    iter_dir = snap_path / f"iteration_{iteration}"
    return (iter_dir / "private_archive_data.json").exists() or (
        iter_dir / "ranker_pools" / "manifest.json"
    ).exists()


def _discover_readable_iterations(snap_path: Path) -> list[int]:
    """Return iteration numbers that have the files needed for paper collection."""
    return [
        iteration
        for iteration in _discover_iterations(snap_path)
        if _iteration_has_metadata(snap_path, iteration)
        and _iteration_has_paper_data(snap_path, iteration)
    ]


def _latest_readable_iteration(snap_path: Path) -> int | None:
    readable_iters = _discover_readable_iterations(snap_path)
    if not readable_iters:
        return None
    return readable_iters[-1]


def _select_read_iteration(
    snap_path: Path,
    *,
    iterations: str | list[int],
    available_iters: list[int],
    cumulative: bool,
) -> int:
    """Choose the snapshot iteration to read before applying paper filters.

    Snapshot archives are cumulative, so an explicit cutoff should read from
    that cutoff rather than from the highest-numbered directory. This also
    avoids treating partially written iteration directories as completed data.
    """
    readable_iters = _discover_readable_iterations(snap_path)

    if isinstance(iterations, str) and iterations in {"all", "last"}:
        if not readable_iters:
            raise FileNotFoundError(
                f"No readable iteration_N directories found in: {snap_path}. "
                "A readable iteration must contain snapshot_metadata.json and "
                "private_archive_data.json or ranker_pools/manifest.json."
            )
        return readable_iters[-1]

    requested = set(iterations)
    available = set(available_iters)
    missing = requested - available
    if missing:
        raise ValueError(
            f"Iterations {sorted(missing)} not found in snapshot {snap_path}. "
            f"Available iterations: {sorted(available)}"
        )

    if cumulative and len(requested) != 1:
        raise ValueError(
            f"cumulative=True is not supported with a multi-element iteration "
            f"list {sorted(requested)}. Use a single iteration cutoff "
            f"(e.g. iterations=[{max(requested)}]) or set cumulative=False."
        )

    read_iter = max(requested)
    if read_iter not in readable_iters:
        raise FileNotFoundError(
            f"Selected iteration {read_iter} in snapshot {snap_path} is not readable. "
            "It must contain snapshot_metadata.json and private_archive_data.json "
            "or ranker_pools/manifest.json."
        )
    return read_iter


def _read_snapshot_metadata(snap_path: Path, iteration: int) -> str:
    """Read run_name from snapshot_metadata.json of the given iteration."""
    meta_path = snap_path / f"iteration_{iteration}" / "snapshot_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    with open(meta_path) as f:
        meta = json.load(f)
    return meta["run_name"]


def _read_private_archive_reviews(
    snap_path: Path,
    iteration: int,
) -> dict[str, list[dict[str, Any]]]:
    archive_path = snap_path / f"iteration_{iteration}" / "private_archive_data.json"
    if not archive_path.exists():
        return {}
    with open(archive_path) as f:
        data = json.load(f)
    reviews_by_paper = data.get("reviews_by_paper", {})
    return reviews_by_paper if isinstance(reviews_by_paper, dict) else {}


def _read_ranker_pool_papers(
    *,
    snap_path: Path,
    iteration: int,
    include_rejected: bool,
    exclude_timeout: bool,
    paper_kind_filter: str,
) -> list[dict[str, Any]] | None:
    pool_dir = snap_path / f"iteration_{iteration}" / "ranker_pools"
    manifest_path = pool_dir / "manifest.json"
    if not manifest_path.exists():
        return None

    try:
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as exc:
        logger.warning("Failed to read ranker_pools manifest at %s: %s", manifest_path, exc)
        return None

    filters = manifest.get("filters", {}) if isinstance(manifest, dict) else {}
    manifest_include_rejected = bool(filters.get("include_rejected", False))
    manifest_exclude_timeout = bool(filters.get("exclude_timeout", True))
    if manifest_include_rejected != bool(include_rejected) or manifest_exclude_timeout != bool(
        exclude_timeout
    ):
        logger.info(
            "ranker_pools filters mismatch for %s (requested include_rejected=%s exclude_timeout=%s; "
            "manifest include_rejected=%s exclude_timeout=%s). Falling back to private archive.",
            snap_path,
            include_rejected,
            exclude_timeout,
            manifest_include_rejected,
            manifest_exclude_timeout,
        )
        return None

    file_map = {
        "all": "papers_all.jsonl",
        "standard_only": "papers_standard.jsonl",
        "meta_only": "papers_meta.jsonl",
    }
    file_name = file_map.get(paper_kind_filter, "papers_all.jsonl")
    pool_path = pool_dir / file_name
    if not pool_path.exists():
        logger.warning(
            "ranker_pools file missing (%s); falling back to private archive.", pool_path
        )
        return None

    rows: list[dict[str, Any]] = []
    with open(pool_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception as exc:
                logger.warning("Invalid JSONL row in %s: %s", pool_path, exc)
                continue
            if isinstance(record, dict):
                rows.append(record)
    return rows


def _read_papers(
    snap_path: Path,
    iteration: int,
    *,
    include_rejected: bool,
    exclude_timeout: bool,
    paper_kind_filter: str,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Read papers and reviews from ranker_pools (preferred) or private archive.

    The private archive has a nested ``papers`` list (list of lists, grouped
    by agent). This function flattens it into a single flat list.

    Returns:
        (flat_papers_list, reviews_by_paper_dict)
    """
    pool_rows = _read_ranker_pool_papers(
        snap_path=snap_path,
        iteration=iteration,
        include_rejected=include_rejected,
        exclude_timeout=exclude_timeout,
        paper_kind_filter=paper_kind_filter,
    )
    reviews_by_paper = _read_private_archive_reviews(snap_path, iteration)
    if pool_rows is not None:
        return pool_rows, reviews_by_paper

    archive_path = snap_path / f"iteration_{iteration}" / "private_archive_data.json"
    if not archive_path.exists():
        raise FileNotFoundError(f"Private archive file not found: {archive_path}")
    with open(archive_path) as f:
        data = json.load(f)

    nested_papers = data["papers"]
    flat_papers: list[dict[str, Any]] = []
    for group in nested_papers:
        if isinstance(group, list):
            flat_papers.extend(group)
        else:
            flat_papers.append(group)

    return flat_papers, reviews_by_paper


def _filter_by_status(
    papers: list[dict[str, Any]],
    include_rejected: bool,
) -> list[dict[str, Any]]:
    """Filter papers by acceptance status.

    Args:
        papers: Flat list of papers.
        include_rejected: If False, keep only accepted papers.
            If True, keep both accepted and rejected.
    """
    if include_rejected:
        return papers
    return [p for p in papers if p.get("status") == "accepted"]


def _is_timeout(paper: dict[str, Any]) -> bool:
    """Return True if a paper is timed out.

    Timeout detection: paper_id ends with '_timeout' OR manuscript == 'TIMEOUT'.
    """
    pid = paper.get("paper_id", "")
    manuscript = paper.get("manuscript", "")
    return pid.endswith("_timeout") or manuscript == "TIMEOUT"


def _filter_timeout(
    papers: list[dict[str, Any]],
    exclude_timeout: bool,
) -> list[dict[str, Any]]:
    """Remove timed-out papers if exclude_timeout is True."""
    if not exclude_timeout:
        return papers
    return [p for p in papers if not _is_timeout(p)]


def _filter_by_paper_kind(
    papers: list[dict[str, Any]],
    paper_kind_filter: str,
) -> list[dict[str, Any]]:
    if paper_kind_filter == "all":
        return papers
    if paper_kind_filter == "standard_only":
        return [
            p
            for p in papers
            if str(p.get("paper_kind", "research") or "research").strip().lower() != "meta_research"
        ]
    if paper_kind_filter == "meta_only":
        return [
            p
            for p in papers
            if str(p.get("paper_kind", "research") or "research").strip().lower() == "meta_research"
        ]
    raise ValueError(
        f"Invalid paper_kind_filter={paper_kind_filter!r}. Expected one of: all | standard_only | meta_only."
    )


def _filter_papers(
    papers: list[dict[str, Any]],
    iterations: str | list[int],
    available_iters: list[int],
    snap_path: Path,
    cumulative: bool = True,
) -> list[dict[str, Any]]:
    """Filter papers by publication_t according to the iterations spec."""
    if iterations == "all":
        return papers

    if iterations == "last":
        # Find the max publication_t actually present in the papers
        if not papers:
            return []
        max_t = max(p["publication_t"] for p in papers)
        if cumulative:
            return [p for p in papers if p["publication_t"] <= max_t]
        return [p for p in papers if p["publication_t"] == max_t]

    # Explicit list of iteration numbers
    requested = set(iterations)
    available = set(available_iters)
    missing = requested - available
    if missing:
        raise ValueError(
            f"Iterations {sorted(missing)} not found in snapshot {snap_path}. "
            f"Available iterations: {sorted(available)}"
        )

    if cumulative:
        if len(requested) != 1:
            raise ValueError(
                f"cumulative=True is not supported with a multi-element iteration "
                f"list {sorted(requested)}. Use a single iteration cutoff "
                f"(e.g. iterations=[{max(requested)}]) or set cumulative=False."
            )
        cutoff = next(iter(requested))
        return [p for p in papers if p["publication_t"] <= cutoff]

    return [p for p in papers if p["publication_t"] in requested]


def _convert_paper(
    paper: dict[str, Any],
    run_name: str,
    snap_dir: str | Path = "",
    reviews_by_paper: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Map a snapshot paper dict to the output format with provenance.

    Copies ALL original fields from the paper, then adds/overrides provenance
    fields and attaches reviews.
    """
    title = paper.get("title", "")
    manuscript = paper.get("manuscript", "")

    if not title:
        logger.warning("Paper %s has empty/missing title", paper.get("paper_id"))
    if not manuscript:
        logger.warning("Paper %s has empty/missing manuscript", paper.get("paper_id"))

    # Copy ALL original fields
    result = dict(paper)

    # Add provenance fields
    result["source_snapshot"] = run_name
    result["source_snapshot_dir"] = str(snap_dir)
    result["source_iteration"] = paper["publication_t"]

    # Ranker compatibility: map code_script to code
    result["code"] = paper.get("code_script") or ""

    # Attach reviews
    pid = paper["paper_id"]
    result["reviews"] = reviews_by_paper.get(pid, []) if reviews_by_paper else []

    return result
