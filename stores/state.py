from collections.abc import Mapping
from typing import Any

from langchain_core.embeddings.embeddings import Embeddings

from stores.agent_registry import AgentRegistry
from stores.notepad_store import NotepadStore
from stores.paper_archive import PaperArchive
from stores.private_paper_archive import PrivatePaperArchive
from system.schema import AgentNetworkMetadata

_archive: PaperArchive | None = None
_private_archive: PrivatePaperArchive | None = None
_registry: AgentRegistry | None = None
_network_metadata: AgentNetworkMetadata | None = None
_notepad_store: NotepadStore | None = None
_seq: int = 0  # bump after each write


def retrieval_settings(
    memory_cfg: Mapping[str, Any] | None,
    *,
    saved_archive: Mapping[str, Any] | None = None,
    saved_registry: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Ranking settings for the public archive and the agent registry.

    Values come from the ``memory`` config block. On resume, settings saved in the snapshot
    (``saved_archive`` / ``saved_registry``) win, so a resumed run keeps ranking the way it started.

    Args:
        memory_cfg: The ``memory`` config block (``max_k``, ``archive``, ``registry``).
        saved_archive: Archive settings saved in a resumed snapshot, if any.
        saved_registry: Registry settings saved in a resumed snapshot, if any.

    Returns:
        ``(archive_settings, registry_settings)``, keyed by the ``PaperArchive`` /
        ``AgentRegistry`` attribute names that snapshots save (``with_citation_boost``,
        ``w_citation``, ``w_sim``, ``max_k``, ...). ``system.main`` maps them onto
        ``init_state``'s prefixed keywords (``archive_with_citation_boost``,
        ``registry_w_sim``, ...).
    """
    memory_cfg = memory_cfg or {}
    archive_cfg = memory_cfg.get("archive", {}) or {}
    registry_cfg = memory_cfg.get("registry", {}) or {}
    max_k = int(memory_cfg.get("max_k", 5))
    archive = {
        "with_citation_boost": bool(archive_cfg.get("citation_boost", True)),
        "with_recency_boost": bool(archive_cfg.get("recency_boost", False)),
        "w_citation": float(archive_cfg.get("w_citation", 0.1)),
        "w_recency": float(archive_cfg.get("w_recency", 0.1)),
        "max_k": max_k,
    }
    registry = {
        "with_citation_boost": bool(registry_cfg.get("citation_boost", True)),
        "w_sim": float(registry_cfg.get("w_sim", 0.8)),
        "w_cit": float(registry_cfg.get("w_cit", 0.2)),
        "max_k": max_k,
    }
    archive.update(dict(saved_archive or {}))
    registry.update(dict(saved_registry or {}))
    return archive, registry


def init_state(
    *,
    embedder: Embeddings,
    max_k: int | None = 5,
    archive_with_citation_boost: bool | None = False,
    archive_with_recency_boost: bool | None = False,
    archive_w_citation: float | None = 0.1,
    archive_w_recency: float | None = 0.1,
    registry_with_citation_boost: bool | None = False,
    registry_w_sim: float | None = 0.8,
    registry_w_cit: float | None = 0.2,
    notepad_enabled: bool | None = False,
    notepad_max_entries_per_agent: int | None = 200,
    notepad_max_entry_chars: int | None = 2000,
    notepad_base_snapshots_dir: str | None = "snapshots",
) -> None:
    """Initialize global singleton stores for archives, registry, and metadata."""
    global _archive, _private_archive, _registry, _network_metadata, _code_memory, _notepad_store
    if _archive is None:
        _archive = PaperArchive(
            embedder=embedder,
            items=None,  # Start with None
            with_citation_boost=archive_with_citation_boost,
            with_recency_boost=archive_with_recency_boost,
            w_citation=archive_w_citation,
            w_recency=archive_w_recency,
            max_k=max_k,
        )
    if _private_archive is None:
        _private_archive = PrivatePaperArchive(items=None)
    if _registry is None:
        _registry = AgentRegistry(
            embedder=embedder,
            agents=None,  # Start with None
            with_citation_boost=registry_with_citation_boost,
            w_sim=registry_w_sim,
            w_cit=registry_w_cit,
            max_k=max_k,
        )
    if _network_metadata is None:
        _network_metadata = AgentNetworkMetadata()
    if _notepad_store is None:
        _notepad_store = NotepadStore(
            enabled=bool(notepad_enabled),
            max_entries_per_agent=int(notepad_max_entries_per_agent or 200),
            max_entry_chars=int(notepad_max_entry_chars or 2000),
            base_snapshots_dir=str(notepad_base_snapshots_dir or "snapshots"),
        )
    else:
        _notepad_store.configure(
            enabled=bool(notepad_enabled),
            max_entries_per_agent=int(notepad_max_entries_per_agent or 200),
            max_entry_chars=int(notepad_max_entry_chars or 2000),
            base_snapshots_dir=str(notepad_base_snapshots_dir or "snapshots"),
        )


def archive() -> PaperArchive:
    """Return the singleton public PaperArchive."""
    assert _archive is not None, "state.init_state() not called yet"
    return _archive


def private_archive() -> PrivatePaperArchive:
    """Return the singleton private PaperArchive."""
    assert _private_archive is not None, "state.init_state() not called yet"
    return _private_archive


def registry() -> AgentRegistry:
    """Return the singleton AgentRegistry."""
    assert _registry is not None, "state.init_state() not called yet"
    return _registry


def network_metadata() -> AgentNetworkMetadata:
    """Return the singleton AgentNetworkMetadata."""
    assert _network_metadata is not None, "state.init_state() not called yet"
    return _network_metadata


def notepad_store() -> NotepadStore:
    """Return the singleton NotepadStore."""
    assert _notepad_store is not None, "state.init_state() not called yet"
    return _notepad_store


def bump_seq() -> int:
    """Increment and return a monotonically increasing sequence number."""
    global _seq
    _seq += 1
    return _seq


def current_seq() -> int:
    """Return the current sequence number without incrementing."""
    return _seq


def update_network_metadata() -> AgentNetworkMetadata:
    """Recompute aggregate network metadata from registry and archive."""
    assert _registry is not None and _archive is not None and _network_metadata is not None, (
        "state.init_state() not called yet"
    )
    agent_profiles = []
    if _registry.vector_store is not None:
        for doc_id in _registry.vector_store.index_to_docstore_id.values():
            try:
                doc = _registry.vector_store.docstore.search(doc_id)
                if doc and hasattr(doc, "metadata"):
                    agent_profile = _registry._convert_doc_to_profile(doc)
                    agent_profiles.append(agent_profile)
            except Exception:
                continue

    papers = list(_archive._entries.values())

    if agent_profiles:
        agent_citations = [agent.citation_count for agent in agent_profiles]
        agent_papers = [agent.num_accepted_papers for agent in agent_profiles]

        _network_metadata.num_agents = len(agent_profiles)
        _network_metadata.min_agent_citation_count = min(agent_citations)
        _network_metadata.max_agent_citation_count = max(agent_citations)
        _network_metadata.avg_agent_citation_count = int(
            sum(agent_citations) / len(agent_citations)
        )
        _network_metadata.min_agent_accepted_papers = min(agent_papers)
        _network_metadata.max_agent_accepted_papers = max(agent_papers)
        _network_metadata.avg_agent_accepted_papers = int(sum(agent_papers) / len(agent_papers))
    else:
        _network_metadata.num_agents = 0
        _network_metadata.min_agent_citation_count = 0
        _network_metadata.max_agent_citation_count = 0
        _network_metadata.avg_agent_citation_count = 0
        _network_metadata.min_agent_accepted_papers = 0
        _network_metadata.max_agent_accepted_papers = 0
        _network_metadata.avg_agent_accepted_papers = 0

    if papers:
        paper_citations = [paper.citation_count for paper in papers]

        _network_metadata.num_papers = len(papers)
        _network_metadata.min_paper_citation_count = min(paper_citations)
        _network_metadata.max_paper_citation_count = max(paper_citations)
        _network_metadata.avg_paper_citation_count = int(
            sum(paper_citations) / len(paper_citations)
        )
    else:
        _network_metadata.num_papers = 0
        _network_metadata.min_paper_citation_count = 0
        _network_metadata.max_paper_citation_count = 0
        _network_metadata.avg_paper_citation_count = 0

    return _network_metadata
