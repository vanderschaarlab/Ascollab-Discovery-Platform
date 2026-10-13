from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _clean_text(value: Any, *, fallback: str = "") -> str:
    return " ".join(str(value if value is not None else fallback).split()).strip()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


class NotepadStore:
    """Thread-safe store for private per-agent notepad ideas."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        max_entries_per_agent: int = 200,
        max_entry_chars: int = 2000,
        base_snapshots_dir: str = "snapshots",
    ) -> None:
        self.enabled = bool(enabled)
        self.max_entries_per_agent = max(1, _safe_int(max_entries_per_agent, 200))
        self.max_entry_chars = max(64, _safe_int(max_entry_chars, 2000))
        self.base_snapshots_dir = str(base_snapshots_dir or "snapshots")
        self._entries_by_agent: dict[str, list[dict[str, Any]]] = {}
        self._notepad_paths_by_agent: dict[str, str] = {}
        self._lock = threading.RLock()
        self._seq = 0

    def configure(
        self,
        *,
        enabled: bool | None = None,
        max_entries_per_agent: int | None = None,
        max_entry_chars: int | None = None,
        base_snapshots_dir: str | None = None,
    ) -> None:
        with self._lock:
            if enabled is not None:
                self.enabled = bool(enabled)
            if max_entries_per_agent is not None:
                self.max_entries_per_agent = max(1, _safe_int(max_entries_per_agent, 200))
            if max_entry_chars is not None:
                self.max_entry_chars = max(64, _safe_int(max_entry_chars, 2000))
            if base_snapshots_dir is not None:
                self.base_snapshots_dir = str(base_snapshots_dir or "snapshots")

    def _next_id(self, *, agent_id: str, iteration: int) -> str:
        self._seq += 1
        safe_agent = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in agent_id)
        return f"idea_t{int(iteration):02d}_{safe_agent}_{self._seq:06d}"

    def _notepad_path(self, *, run_name: str, agent_id: str) -> Path:
        return (
            Path(self.base_snapshots_dir)
            / str(run_name or "unnamed_experiment")
            / "notepads"
            / str(agent_id)
            / "notepad.jsonl"
        )

    def _trim_entries(self, agent_id: str) -> None:
        rows = self._entries_by_agent.get(agent_id, [])
        if len(rows) > self.max_entries_per_agent:
            self._entries_by_agent[agent_id] = rows[-self.max_entries_per_agent :]

    def append_idea(
        self,
        *,
        agent_id: str,
        iteration: int,
        run_name: str,
        title: str,
        idea_text: str,
        deferred_reason: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if not self.enabled:
                raise RuntimeError("notepad feature is disabled")

            clean_agent_id = _clean_text(agent_id)
            if not clean_agent_id:
                raise ValueError("agent_id is required")

            clean_title = _clean_text(title, fallback="Untitled deferred idea")
            clean_idea_text = str(idea_text or "").strip()
            if not clean_idea_text:
                raise ValueError("idea_text is required")
            if len(clean_idea_text) > self.max_entry_chars:
                clean_idea_text = clean_idea_text[: self.max_entry_chars].rstrip()

            clean_deferred_reason = str(deferred_reason or "").strip()
            if len(clean_deferred_reason) > self.max_entry_chars:
                clean_deferred_reason = clean_deferred_reason[: self.max_entry_chars].rstrip()

            clean_tags = []
            for tag in tags or []:
                tok = _clean_text(tag)
                if tok:
                    clean_tags.append(tok)
            clean_tags = clean_tags[:20]

            ts = _utc_now_iso()
            idea_id = self._next_id(agent_id=clean_agent_id, iteration=int(iteration))
            entry = {
                "timestamp": ts,
                "iteration": int(iteration),
                "run_name": str(run_name or "unnamed_experiment"),
                "agent_id": clean_agent_id,
                "idea_id": idea_id,
                "title": clean_title,
                "idea_text": clean_idea_text,
                "deferred_reason": clean_deferred_reason,
                "tags": clean_tags,
                "visibility": "private",
            }
            hash_payload = json.dumps(
                {
                    "iteration": entry["iteration"],
                    "run_name": entry["run_name"],
                    "agent_id": entry["agent_id"],
                    "idea_id": entry["idea_id"],
                    "title": entry["title"],
                    "idea_text": entry["idea_text"],
                    "deferred_reason": entry["deferred_reason"],
                    "tags": entry["tags"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            entry["content_hash"] = hashlib.sha256(hash_payload.encode("utf-8")).hexdigest()

            path = self._notepad_path(run_name=entry["run_name"], agent_id=clean_agent_id)
            entry["notepad_path"] = str(path.as_posix())
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            rows = list(self._entries_by_agent.get(clean_agent_id, []))
            rows.append(dict(entry))
            self._entries_by_agent[clean_agent_id] = rows
            self._trim_entries(clean_agent_id)
            self._notepad_paths_by_agent[clean_agent_id] = str(path.as_posix())

            return {
                "idea_id": idea_id,
                "agent_id": clean_agent_id,
                "notepad_path": str(path.as_posix()),
                "entry": dict(entry),
                "append_status": "ok",
            }

    def get_entries(self, agent_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(x) for x in (self._entries_by_agent.get(str(agent_id), []) or [])]

    def get_notepad_path(self, agent_id: str) -> str | None:
        with self._lock:
            path = self._notepad_paths_by_agent.get(str(agent_id))
            return str(path) if path else None

    def snapshot_data(self) -> dict[str, Any]:
        with self._lock:
            return {
                "entries_by_agent": {
                    str(agent_id): [dict(x) for x in rows]
                    for agent_id, rows in self._entries_by_agent.items()
                },
                "notepad_file_paths": dict(self._notepad_paths_by_agent),
                "config": {
                    "enabled": bool(self.enabled),
                    "max_entries_per_agent": int(self.max_entries_per_agent),
                    "max_entry_chars": int(self.max_entry_chars),
                    "base_snapshots_dir": str(self.base_snapshots_dir),
                },
            }

    def load_snapshot_data(self, payload: Mapping[str, Any] | None) -> None:
        with self._lock:
            raw = dict(payload or {}) if isinstance(payload, Mapping) else {}
            incoming = raw.get("entries_by_agent", {}) or {}
            merged: dict[str, list[dict[str, Any]]] = {}
            if isinstance(incoming, Mapping):
                for agent_id, rows in incoming.items():
                    clean_agent_id = _clean_text(agent_id)
                    if not clean_agent_id:
                        continue
                    acc: list[dict[str, Any]] = []
                    for row in rows or []:
                        if not isinstance(row, Mapping):
                            continue
                        entry = {
                            "timestamp": str(row.get("timestamp", "") or ""),
                            "iteration": _safe_int(row.get("iteration", 0), 0),
                            "run_name": str(row.get("run_name", "") or ""),
                            "agent_id": clean_agent_id,
                            "idea_id": str(row.get("idea_id", "") or ""),
                            "title": str(row.get("title", "") or ""),
                            "idea_text": str(row.get("idea_text", "") or ""),
                            "deferred_reason": str(row.get("deferred_reason", "") or ""),
                            "tags": [
                                str(t).strip()
                                for t in (row.get("tags", []) or [])
                                if str(t).strip()
                            ][:20],
                            "visibility": "private",
                            "notepad_path": str(row.get("notepad_path", "") or ""),
                            "content_hash": str(row.get("content_hash", "") or ""),
                        }
                        if not entry["idea_id"] or not entry["idea_text"]:
                            continue
                        acc.append(entry)
                    acc.sort(
                        key=lambda x: (
                            _safe_int(x.get("iteration", 0), 0),
                            str(x.get("idea_id", "")),
                        )
                    )
                    if acc:
                        merged[clean_agent_id] = acc[-self.max_entries_per_agent :]
            self._entries_by_agent = merged

            raw_paths = raw.get("notepad_file_paths", {}) or {}
            paths: dict[str, str] = {}
            if isinstance(raw_paths, Mapping):
                for agent_id, path in raw_paths.items():
                    aid = _clean_text(agent_id)
                    p = str(path or "").strip()
                    if aid and p:
                        paths[aid] = p
            self._notepad_paths_by_agent = paths
