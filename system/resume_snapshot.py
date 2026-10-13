from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from system.bootstrap_iteration import load_pool

_ITER_RE = re.compile(r"^iteration_(\d+)$")


@dataclass
class ResumeSnapshotState:
    run_dir: Path
    iteration_dir: Path
    restored_iteration: int
    run_name: str
    agents: list[Any]
    public_papers: list[Any]
    private_papers: list[Any]
    reviews_by_paper: dict[str, list[dict[str, Any]]]
    review_assignments: dict[str, dict[str, Any]]
    notepad_entries: dict[str, list[dict[str, Any]]]
    session_stats_payload: dict[str, Any]
    registry_settings: dict[str, Any]
    archive_settings: dict[str, Any]
    private_archive_settings: dict[str, Any]


def _safe_load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _safe_load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    return bool(value)


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def resolve_resume_source(
    snapshot_path: str | Path,
    *,
    project_root: Path,
) -> tuple[Path, int | None]:
    """Resolve resume source into a run directory and optional iteration cap.

    Returns:
        (run_dir, max_iteration)
        - max_iteration is None when a run directory is provided.
        - max_iteration is the iteration number when an iteration directory is provided.
    """
    raw = Path(str(snapshot_path))
    path = raw if raw.is_absolute() else (project_root / raw)
    path = path.expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Resume snapshot path not found: {path}")
    if not path.is_dir():
        raise ValueError(f"Resume snapshot path must be a directory: {path}")

    m = _ITER_RE.match(path.name)
    if m:
        run_dir = path.parent
        if not run_dir.exists() or not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory does not exist for iteration path: {run_dir}")
        return run_dir, int(m.group(1))

    return path, None


def discover_iteration_dirs(run_dir: Path) -> list[tuple[int, Path]]:
    """Discover iteration directories in a run dir as sorted (iteration, path) tuples."""
    discovered: list[tuple[int, Path]] = []
    for child in run_dir.iterdir():
        if not child.is_dir():
            continue
        m = _ITER_RE.match(child.name)
        if not m:
            continue
        discovered.append((int(m.group(1)), child))
    discovered.sort(key=lambda x: x[0])
    return discovered


def is_complete_iteration(iteration_dir: Path) -> tuple[bool, str]:
    """Return whether an iteration directory is complete for resume purposes."""
    if not iteration_dir.exists() or not iteration_dir.is_dir():
        return False, "not_a_directory"

    m = _ITER_RE.match(iteration_dir.name)
    if not m:
        return False, "invalid_iteration_dir_name"
    expected_iteration = int(m.group(1))

    meta_path = iteration_dir / "snapshot_metadata.json"
    if not meta_path.exists():
        return False, "missing_snapshot_metadata"
    try:
        meta = _safe_load_json(meta_path)
    except Exception as e:
        return False, f"invalid_snapshot_metadata:{e}"

    meta_iteration = _as_int(meta.get("iteration", -1), -1)
    if meta_iteration != expected_iteration:
        return False, f"snapshot_metadata_iteration_mismatch:{meta_iteration}!={expected_iteration}"

    if not (iteration_dir / "agent_registry_data.pkl").exists():
        return False, "missing_agent_registry_data"

    has_archive = (iteration_dir / "archive_data.pkl").exists() or (
        iteration_dir / "archive_data.json"
    ).exists()
    if not has_archive:
        return False, "missing_archive_data"

    has_private_archive = (iteration_dir / "private_archive_data.pkl").exists() or (
        iteration_dir / "private_archive_data.json"
    ).exists()
    if not has_private_archive:
        return False, "missing_private_archive_data"

    return True, "ok"


def find_last_complete_iteration(
    run_dir: Path, *, max_iteration: int | None = None
) -> tuple[int, Path]:
    """Find the highest complete iteration in a run directory."""
    discovered = discover_iteration_dirs(run_dir)
    if not discovered:
        raise FileNotFoundError(
            f"No iteration_<N> directories found under run directory: {run_dir}"
        )

    for iteration, iteration_dir in reversed(discovered):
        if max_iteration is not None and iteration > int(max_iteration):
            continue
        complete, _reason = is_complete_iteration(iteration_dir)
        if complete:
            return iteration, iteration_dir

    if max_iteration is None:
        raise FileNotFoundError(f"No complete iteration found in run directory: {run_dir}")
    raise FileNotFoundError(
        f"No complete iteration <= {int(max_iteration)} found in run directory: {run_dir}"
    )


def archive_incomplete_tail(run_dir: Path, *, restored_iteration: int) -> list[Path]:
    """Rename incomplete iteration dirs greater than restored_iteration for cleanliness."""
    archived_paths: list[Path] = []
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    for iteration, iteration_dir in discover_iteration_dirs(run_dir):
        if iteration <= int(restored_iteration):
            continue
        complete, _reason = is_complete_iteration(iteration_dir)
        if complete:
            continue
        base_name = f"{iteration_dir.name}__incomplete_{ts}"
        target = run_dir / base_name
        idx = 1
        while target.exists():
            target = run_dir / f"{base_name}_{idx:02d}"
            idx += 1
        iteration_dir.rename(target)
        archived_paths.append(target)
    return archived_paths


def _load_blob_with_fallback(
    iteration_dir: Path, *, pkl_name: str, json_name: str
) -> dict[str, Any]:
    pkl_path = iteration_dir / pkl_name
    json_path = iteration_dir / json_name
    if pkl_path.exists():
        data = _safe_load_pickle(pkl_path)
        return data if isinstance(data, dict) else {}
    if json_path.exists():
        return _safe_load_json(json_path)
    return {}


def _extract_run_name(iteration_dir: Path) -> str:
    meta_path = iteration_dir / "snapshot_metadata.json"
    if meta_path.exists():
        try:
            meta = _safe_load_json(meta_path)
            run_name = str(meta.get("run_name", "") or "").strip()
            if run_name:
                return run_name
        except Exception:
            pass
    return iteration_dir.parent.name


def load_iteration_exp_config(iteration_dir: Path) -> DictConfig:
    cfg_path = iteration_dir / "exp_config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing exp_config.yaml at {cfg_path}")
    loaded = OmegaConf.load(cfg_path)
    if not isinstance(loaded, DictConfig):
        loaded = OmegaConf.create(loaded)
    if not isinstance(loaded, DictConfig):
        raise ValueError(f"Failed to load exp config as DictConfig from {cfg_path}")
    return loaded


def load_full_state_from_iteration(iteration_dir: Path) -> ResumeSnapshotState:
    complete, reason = is_complete_iteration(iteration_dir)
    if not complete:
        raise ValueError(
            f"Cannot load resume state from incomplete iteration dir {iteration_dir}: {reason}"
        )

    m = _ITER_RE.match(iteration_dir.name)
    assert m is not None
    restored_iteration = int(m.group(1))
    run_dir = iteration_dir.parent

    agents, public_papers, private_papers, reviews_by_paper, notepad_entries = load_pool(
        iteration_dir
    )

    session_stats_payload: dict[str, Any] = {}
    stats_path = iteration_dir / "session_stats_data.pkl"
    if stats_path.exists():
        loaded = _safe_load_pickle(stats_path)
        if isinstance(loaded, dict):
            session_stats_payload = loaded

    registry_blob = _load_blob_with_fallback(
        iteration_dir, pkl_name="agent_registry_data.pkl", json_name="agent_registry_data.json"
    )
    archive_blob = _load_blob_with_fallback(
        iteration_dir, pkl_name="archive_data.pkl", json_name="archive_data.json"
    )
    private_blob = _load_blob_with_fallback(
        iteration_dir, pkl_name="private_archive_data.pkl", json_name="private_archive_data.json"
    )

    registry_settings = {
        "with_citation_boost": _as_bool(registry_blob.get("with_citation_boost", False), False),
        "w_sim": _as_float(registry_blob.get("w_sim", 0.8), 0.8),
        "w_cit": _as_float(registry_blob.get("w_cit", 0.2), 0.2),
        "max_k": _as_int(registry_blob.get("max_k", 5), 5),
    }
    archive_settings = {
        "with_citation_boost": _as_bool(archive_blob.get("with_citation_boost", False), False),
        "with_recency_boost": _as_bool(archive_blob.get("with_recency_boost", False), False),
        "w_citation": _as_float(archive_blob.get("w_citation", 0.1), 0.1),
        "w_recency": _as_float(archive_blob.get("w_recency", 0.1), 0.1),
        "max_k": _as_int(archive_blob.get("max_k", 5), 5),
    }
    private_archive_settings = {
        "with_citation_boost": _as_bool(private_blob.get("with_citation_boost", False), False),
        "with_recency_boost": _as_bool(private_blob.get("with_recency_boost", False), False),
        "w_citation": _as_float(private_blob.get("w_citation", 0.1), 0.1),
        "w_recency": _as_float(private_blob.get("w_recency", 0.1), 0.1),
        "max_k": _as_int(private_blob.get("max_k", 5), 5),
    }

    return ResumeSnapshotState(
        run_dir=run_dir,
        iteration_dir=iteration_dir,
        restored_iteration=restored_iteration,
        run_name=_extract_run_name(iteration_dir),
        agents=list(agents or []),
        public_papers=list(public_papers or []),
        private_papers=list(private_papers or []),
        reviews_by_paper=dict(reviews_by_paper or {}),
        review_assignments=dict(private_blob.get("review_assignments") or {}),
        notepad_entries=dict(notepad_entries or {}),
        session_stats_payload=session_stats_payload,
        registry_settings=registry_settings,
        archive_settings=archive_settings,
        private_archive_settings=private_archive_settings,
    )
