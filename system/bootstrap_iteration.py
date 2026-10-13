import hashlib
import json
import pickle
import random
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from system.schema import AgentProfile, PaperMetadataPrivate, PaperMetaReview


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _agent_from_dict(data: Mapping[str, Any]) -> AgentProfile:
    return AgentProfile(
        agent_id=str(data.get("agent_id", "")),
        behavior=str(data.get("behavior", "")),
        expertise=str(data.get("expertise", "")),
        expertise_topics=list(data.get("expertise_topics", []) or []),
        citation_count=int(data.get("citation_count", 0) or 0),
        num_accepted_papers=int(data.get("num_accepted_papers", 0) or 0),
        metric_weights=data.get("metric_weights"),
        reputation=data.get("reputation"),
        pause_state=data.get("pause_state"),
        learning_state=data.get("learning_state"),
    )


def _paper_from_dict(data: Mapping[str, Any]) -> PaperMetadataPrivate:
    metareview = data.get("metareview")
    if isinstance(metareview, dict) and metareview.get("paper_id"):
        try:
            metareview = PaperMetaReview(**metareview)
        except Exception:
            pass
    return PaperMetadataPrivate(
        paper_id=str(data.get("paper_id", "")),
        primary_agent_id=str(data.get("primary_agent_id", "")),
        collab_agent_ids=list(data.get("collab_agent_ids", []) or []),
        title=str(data.get("title", "")),
        abstract=str(data.get("abstract", "")),
        manuscript=str(data.get("manuscript", "")),
        citation_count=int(data.get("citation_count", 0) or 0),
        publication_t=int(data.get("publication_t", 0) or 0),
        cited_paper_ids=list(data.get("cited_paper_ids", []) or []),
        code_script=data.get("code_script"),
        metareview=metareview,
        review_metrics=dict(data.get("review_metrics", {}) or {})
        if isinstance(data.get("review_metrics", {}), Mapping)
        else {},
        status=str(data.get("status", "under_review")),
        publication_status=str(data.get("publication_status", data.get("status", "under_review"))),
        research_validation_status=str(
            data.get("research_validation_status", data.get("status", "under_review"))
        ),
        citation_compliant=bool(data.get("citation_compliant", True)),
        citation_errors=[str(x) for x in (data.get("citation_errors", []) or [])],
        citation_error_count=int(data.get("citation_error_count", 0) or 0),
        invalid_archive_refs=[str(x) for x in (data.get("invalid_archive_refs", []) or [])],
        is_citable_archive_source=bool(data.get("is_citable_archive_source", False)),
        citable_with_warnings=bool(data.get("citable_with_warnings", False)),
        plot_artifacts=[
            dict(x) for x in (data.get("plot_artifacts", []) or []) if isinstance(x, Mapping)
        ],
        plot_filenames=[str(x) for x in (data.get("plot_filenames", []) or []) if str(x).strip()],
        plot_handles=[str(x) for x in (data.get("plot_handles", []) or []) if str(x).strip()],
        plot_handle_to_filename={
            str(k): str(v)
            for k, v in (data.get("plot_handle_to_filename", {}) or {}).items()
            if str(k).strip() and str(v).strip()
        },
    )


def _flatten_papers(items: Any) -> list[Mapping[str, Any]]:
    flat: list[Mapping[str, Any]] = []
    for item in items or []:
        if isinstance(item, list):
            for sub in item:
                if isinstance(sub, Mapping):
                    flat.append(sub)
        elif isinstance(item, Mapping):
            flat.append(item)
    return flat


def _normalize_notepad_entries(raw: Any) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(raw, Mapping):
        return {}
    rows = raw.get("entries_by_agent", raw)
    if not isinstance(rows, Mapping):
        return {}
    out: dict[str, list[dict[str, Any]]] = {}
    for agent_id, entries in rows.items():
        aid = str(agent_id or "").strip()
        if not aid:
            continue
        bucket: list[dict[str, Any]] = []
        for entry in entries or []:
            if not isinstance(entry, Mapping):
                continue
            idea_id = str(entry.get("idea_id", "") or "").strip()
            idea_text = str(entry.get("idea_text", "") or "").strip()
            if not idea_id or not idea_text:
                continue
            bucket.append(dict(entry))
        if bucket:
            out[aid] = bucket
    return out


def _load_pool_from_snapshot_dir(
    path: Path,
) -> tuple[
    list[AgentProfile],
    list[PaperMetadataPrivate],
    list[PaperMetadataPrivate],
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
]:
    agent_path = path / "agent_registry_data.pkl"
    archive_json = path / "archive_data.json"
    private_json = path / "private_archive_data.json"
    archive_pkl = path / "archive_data.pkl"
    private_pkl = path / "private_archive_data.pkl"
    notepad_json = path / "notepad_data.json"
    notepad_pkl = path / "notepad_data.pkl"

    if not agent_path.exists():
        raise FileNotFoundError(f"Missing agent registry pool at {agent_path}")

    agent_blob = _load_pickle(agent_path)
    agents_raw = agent_blob.get("agents", []) if isinstance(agent_blob, dict) else agent_blob
    agents: list[AgentProfile] = []
    for a in agents_raw or []:
        if isinstance(a, AgentProfile):
            agents.append(a)
        elif isinstance(a, Mapping):
            agents.append(_agent_from_dict(a))

    archive_raw: dict[str, Any] = {}
    if archive_pkl.exists():
        archive_raw = _load_pickle(archive_pkl)
    elif archive_json.exists():
        archive_raw = _load_json(archive_json)

    private_raw: dict[str, Any] = {}
    if private_pkl.exists():
        private_raw = _load_pickle(private_pkl)
    elif private_json.exists():
        private_raw = _load_json(private_json)

    public_papers = [_paper_from_dict(p) for p in _flatten_papers(archive_raw.get("papers"))]
    private_papers = [_paper_from_dict(p) for p in _flatten_papers(private_raw.get("papers"))]
    reviews_by_paper = private_raw.get("reviews_by_paper") or {}
    notepad_raw: dict[str, Any] = {}
    if notepad_pkl.exists():
        notepad_raw = _load_pickle(notepad_pkl) or {}
    elif notepad_json.exists():
        notepad_raw = _load_json(notepad_json) or {}
    notepad_entries = _normalize_notepad_entries(notepad_raw)

    return agents, public_papers, private_papers, reviews_by_paper, notepad_entries


def _load_pool_from_json(
    path: Path,
) -> tuple[
    list[AgentProfile],
    list[PaperMetadataPrivate],
    list[PaperMetadataPrivate],
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
]:
    raw = _load_json(path)
    agents = [_agent_from_dict(a) for a in (raw.get("agents") or [])]
    public_papers = [_paper_from_dict(p) for p in (raw.get("public_papers") or [])]
    private_papers = [_paper_from_dict(p) for p in (raw.get("private_papers") or [])]
    reviews_by_paper = raw.get("reviews_by_paper") or {}
    notepad_entries = _normalize_notepad_entries(
        raw.get("notepad_data", raw.get("notepad_entries", {}))
    )
    return agents, public_papers, private_papers, reviews_by_paper, notepad_entries


def load_pool(
    path: Path,
) -> tuple[
    list[AgentProfile],
    list[PaperMetadataPrivate],
    list[PaperMetadataPrivate],
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
]:
    if path.is_dir():
        return _load_pool_from_snapshot_dir(path)
    if path.is_file() and path.suffix.lower() == ".json":
        return _load_pool_from_json(path)
    raise ValueError(f"Unsupported pool path: {path}")


def load_pool_many(
    paths: Iterable[Path],
) -> tuple[
    list[AgentProfile],
    list[PaperMetadataPrivate],
    list[PaperMetadataPrivate],
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
]:
    agents: dict[str, AgentProfile] = {}
    public_by_id: dict[str, PaperMetadataPrivate] = {}
    private_by_id: dict[str, PaperMetadataPrivate] = {}
    reviews_by_paper: dict[str, list[dict[str, Any]]] = {}
    notepad_by_agent: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        a, pub, priv, reviews, notepad_entries = load_pool(path)
        for agent in a:
            if agent.agent_id and agent.agent_id not in agents:
                agents[agent.agent_id] = agent
        for paper in pub:
            if paper.paper_id and paper.paper_id not in public_by_id:
                public_by_id[paper.paper_id] = paper
        for paper in priv:
            if paper.paper_id and paper.paper_id not in private_by_id:
                private_by_id[paper.paper_id] = paper
        for paper_id, entries in (reviews or {}).items():
            if paper_id and paper_id not in reviews_by_paper:
                reviews_by_paper[paper_id] = list(entries or [])
        for agent_id, entries in (notepad_entries or {}).items():
            aid = str(agent_id or "").strip()
            if not aid:
                continue
            bucket = notepad_by_agent.setdefault(aid, [])
            seen_ids = {
                str(x.get("idea_id", "") or "").strip() for x in bucket if isinstance(x, Mapping)
            }
            for entry in entries or []:
                if not isinstance(entry, Mapping):
                    continue
                idea_id = str(entry.get("idea_id", "") or "").strip()
                if not idea_id or idea_id in seen_ids:
                    continue
                bucket.append(dict(entry))
                seen_ids.add(idea_id)
    return (
        list(agents.values()),
        list(public_by_id.values()),
        list(private_by_id.values()),
        reviews_by_paper,
        notepad_by_agent,
    )


def _select_agents(
    agents: list[AgentProfile],
    *,
    num_agents: int,
    seed: int,
) -> list[AgentProfile]:
    by_id = {a.agent_id: a for a in agents if a.agent_id}
    agent_ids = sorted(by_id.keys())
    if num_agents > len(agent_ids):
        raise ValueError(f"Requested {num_agents} agents but pool only has {len(agent_ids)}")
    ranked = []
    for aid in agent_ids:
        key = f"{seed}|{aid}".encode()
        ranked.append((hashlib.sha256(key).hexdigest(), aid))
    ranked.sort(key=lambda x: x[0])
    chosen_ids = [aid for _, aid in ranked[:num_agents]]
    return [by_id[aid] for aid in chosen_ids]


def _select_papers(
    papers: Iterable[PaperMetadataPrivate],
    *,
    selected_ids: list[str],
    max_per_agent: int | None,
    rng: random.Random,
) -> list[PaperMetadataPrivate]:
    selected: list[PaperMetadataPrivate] = []
    selected_set = set(selected_ids)
    grouped: dict[str, list[PaperMetadataPrivate]] = {aid: [] for aid in selected_ids}
    for p in papers:
        if p.primary_agent_id in selected_set:
            grouped[p.primary_agent_id].append(p)
    for agent_id in selected_ids:
        items = sorted(grouped.get(agent_id, []), key=lambda p: p.paper_id)
        if max_per_agent is not None and len(items) > max_per_agent:
            items = rng.sample(items, k=max_per_agent)
        selected.extend(items)
    return selected


def build_bootstrap_selection(
    *,
    pool_paths: list[Path],
    num_agents: int,
    seed: int,
    max_public_papers_per_agent: int | None,
    max_private_papers_per_agent: int | None,
) -> tuple[
    list[AgentProfile],
    list[PaperMetadataPrivate],
    list[PaperMetadataPrivate],
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
]:
    rng = random.Random(seed)
    agents, public_pool, private_pool, reviews_by_paper, notepad_by_agent = load_pool_many(
        pool_paths
    )
    selected_agents = _select_agents(agents, num_agents=num_agents, seed=seed)
    selected_ids = [a.agent_id for a in selected_agents]

    public_selected = _select_papers(
        public_pool,
        selected_ids=selected_ids,
        max_per_agent=max_public_papers_per_agent,
        rng=rng,
    )
    private_selected = _select_papers(
        private_pool,
        selected_ids=selected_ids,
        max_per_agent=max_private_papers_per_agent,
        rng=rng,
    )

    selected_paper_ids = {p.paper_id for p in private_selected if p.paper_id}
    reviews_selected = {
        paper_id: list(entries or [])
        for paper_id, entries in (reviews_by_paper or {}).items()
        if paper_id in selected_paper_ids
    }
    notepad_selected = {
        aid: list(notepad_by_agent.get(aid, []) or [])
        for aid in selected_ids
        if aid in notepad_by_agent
    }

    return selected_agents, public_selected, private_selected, reviews_selected, notepad_selected
