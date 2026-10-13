"""End-to-end harness: run every experiment condition through ``system.main`` with fakes.

Each condition from ``scripts/experiments/*/run.sh`` runs for two rounds with at most five agents
(model mix scaled down, keeping the same backends), using the scripted LLM, embedder and sandbox in
``tests/e2e/fakes.py``. The recorded fixture holds:

- ``exit_code`` and, on failure, the last error line;
- ``llm_calls``: number of calls per kind, as ``tests/e2e/fakes.py`` classifies them
  (``persona_behavior``, ``persona_expertise``, ``research``, ``collaborator``, ``reviewer``,
  ``metareviewer``, ``feedback_synthesizer``, or ``other`` for any unrecognised call);
- ``prompts``: for each kind, a representative system prompt the pipeline sent (the
  lexicographically smallest, so it does not depend on call order), plus ``distinct_prompts``;
- ``accepted``: number of papers in the public archive (``archive_data.json``) after each round,
  i.e. all papers accepted so far that the archive keeps (Markovian memory drops older ones);
- ``snapshot``: per round, every snapshot file with the structure (keys and types, not values) of
  its JSON content.

Regenerate with ``E2E_UPDATE=1 pytest tests/e2e``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from tests.golden.harness import collect_conditions
from tests.isolation import subprocess_env

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
MAX_AGENTS = 5
ROUNDS = 2
BACKENDS = ("gpt4o", "gpt5", "deepseek", "llama")


def _override_value(overrides: list[str], key: str) -> str | None:
    value = None
    for item in overrides:
        k, _, v = item.lstrip("+").partition("=")
        if k == key:
            value = v
    return value


def _scaled_counts(counts: dict[str, int], n: int) -> dict[str, int]:
    """Scale a model mix to n agents, keeping every non-zero backend while n allows."""
    total = sum(counts.values())
    if total <= n:
        return dict(counts)
    nonzero = [b for b in BACKENDS if counts.get(b, 0) > 0]
    out = dict.fromkeys(BACKENDS, 0)
    for b in nonzero[:n]:
        out[b] = 1
    remaining = n - sum(out.values())
    shares = sorted(nonzero, key=lambda b: -counts[b])
    i = 0
    while remaining > 0 and shares:
        out[shares[i % len(shares)]] += 1
        remaining -= 1
        i += 1
    return out


def sized_overrides(overrides: list[str], run_dir: Path) -> list[str]:
    """Shrink a condition to a short fake run without changing what it manipulates."""
    n_cond = _override_value(overrides, "network.num_agents")
    n = min(int(n_cond), MAX_AGENTS) if n_cond else MAX_AGENTS
    counts = {b: int(_override_value(overrides, f"llm_backends.counts.{b}") or 0) for b in BACKENDS}
    if sum(counts.values()) == 0:
        counts = {"gpt4o": 0, "gpt5": n, "deepseek": 0, "llama": 0}
    counts = _scaled_counts(counts, n)
    sized = [
        *overrides,
        f"network.num_agents={n}",
        f"network.iterations={ROUNDS}",
        "research.research_max_iters=12",
        "data.preflight=false",
        f"hydra.run.dir={run_dir / 'hydra'}",
        *[f"llm_backends.counts.{b}={c}" for b, c in counts.items()],
    ]
    k = int(_override_value(overrides, "review.num_reviewers_per_paper") or _default_k())
    self_review = (_override_value(overrides, "review.allow_self_review") or "").lower() == "true"
    if not self_review:
        sized.append(f"review.num_reviewers_per_paper={min(k, max(1, n - 1))}")
    return sized


def _default_k() -> int:
    from omegaconf import OmegaConf

    return int(
        OmegaConf.load(REPO_ROOT / "sys_config" / "ascollab.yaml").review.num_reviewers_per_paper
    )


def _schema(obj: Any, depth: int = 0) -> Any:
    if depth > 6:
        return "…"
    if isinstance(obj, dict):
        return {
            str(k): _schema(v, depth + 1) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(obj, list):
        return [_schema(obj[0], depth + 1)] if obj else []
    return type(obj).__name__


def _file_schema(path: Path) -> Any:
    if path.suffix == ".json":
        try:
            return _schema(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            return f"unreadable json: {type(exc).__name__}"
    if path.suffix == ".jsonl":
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return {"jsonl_first_row": _schema(json.loads(lines[0])) if lines else None}
    return path.suffix.lstrip(".") or "file"


def _snapshot_structure(root: Path) -> dict[str, Any]:
    runs = sorted((root / "snapshots").glob("*"))
    if not runs:
        return {}
    out: dict[str, Any] = {}
    for it_dir in sorted(p for p in runs[0].iterdir() if p.is_dir()):
        files = {}
        for f in sorted(it_dir.rglob("*")):
            if f.is_file():
                rel = str(f.relative_to(it_dir))
                if rel.startswith("timelines/") or rel.startswith("ranker_pools/"):
                    rel = rel.split("/")[0] + "/<" + f.suffix.lstrip(".") + ">"
                files.setdefault(rel, _file_schema(f))
        out[it_dir.name] = files
    return out


def _accepted(root: Path) -> list[int]:
    runs = sorted((root / "snapshots").glob("*"))
    if not runs:
        return []
    counts = []
    for it_dir in sorted(
        p for p in runs[0].iterdir() if p.is_dir() and p.name.startswith("iteration_")
    ):
        data = it_dir / "archive_data.json"
        try:
            doc = json.loads(data.read_text(encoding="utf-8"))
            papers = doc.get("papers", doc) if isinstance(doc, dict) else doc
            counts.append(len(papers))
        except Exception:
            counts.append(-1)
    return counts


def run_condition(overrides: list[str], workdir: Path) -> dict[str, Any]:
    workdir.mkdir(parents=True, exist_ok=True)
    capture = workdir / "capture.json"
    env = subprocess_env(PYTHONPATH=str(REPO_ROOT), E2E_CAPTURE=str(capture))
    proc = subprocess.run(
        [sys.executable, "-m", "tests.e2e.fake_pipeline", *sized_overrides(overrides, workdir)],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    calls = json.loads(capture.read_text(encoding="utf-8")) if capture.exists() else []
    # Agents run concurrently, so call order varies between runs; record an order-independent
    # representative prompt per kind (the lexicographically smallest) and how many distinct
    # prompts each kind had.
    by_kind: dict[str, set[str]] = {}
    for call in calls:
        by_kind.setdefault(call["kind"], set()).add(call["system"])
    prompts = {kind: min(texts) for kind, texts in by_kind.items()}
    distinct = {kind: len(texts) for kind, texts in sorted(by_kind.items())}
    fixture: dict[str, Any] = {
        "exit_code": proc.returncode,
        "llm_calls": dict(sorted(Counter(c["kind"] for c in calls).items())),
        "prompts": prompts,
        "distinct_prompts": distinct,
        "accepted": _accepted(workdir),
        "snapshot": _snapshot_structure(workdir),
    }
    if proc.returncode != 0:
        lines = [ln for ln in (proc.stderr or "").splitlines() if ln.strip()]
        errors = [ln for ln in lines if "Error" in ln or "Exception" in ln]
        fixture["error"] = (errors or lines or ["unknown"])[-1][:300]
    return fixture


def conditions() -> dict[str, dict[str, Any]]:
    return collect_conditions()


def dump(fixture: dict[str, Any]) -> str:
    return json.dumps(fixture, indent=1, sort_keys=True) + "\n"
