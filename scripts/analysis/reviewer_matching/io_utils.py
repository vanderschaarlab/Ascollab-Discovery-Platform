"""Shared JSON/YAML helpers for the reviewer-matching ablation scripts."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any


def to_plain(value: Any) -> Any:
    """Convert dataclasses and nested containers into JSON-serializable values."""
    if dataclasses.is_dataclass(value):
        # Avoid dataclasses.asdict(): it reconstructs mapping subclasses
        # (e.g., defaultdict) via type(obj)(items), which fails for
        # defaultdict because first arg must be a callable default_factory.
        return {
            field.name: to_plain(getattr(value, field.name, None))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {str(k): to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_plain(v) for v in value]
    return value


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, payload: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(to_plain(payload), f, indent=2, ensure_ascii=False)


def write_jsonl(path: str | Path, rows: list[Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(to_plain(row), ensure_ascii=False) + "\n")
