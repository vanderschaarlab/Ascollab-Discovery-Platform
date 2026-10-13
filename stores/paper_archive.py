import faiss
import numpy as np
from langchain_core.embeddings.embeddings import Embeddings

from stores.embedder import Embedder
from system.schema import PaperMetadata


class PaperArchive:
    def __init__(
        self,
        embedder: Embeddings,
        items: list[PaperMetadata] | None = None,
        *,
        with_citation_boost: bool | None = False,
        with_recency_boost: bool | None = False,
        w_citation: float | None = 0.1,
        w_recency: float | None = 0.1,
        max_k: int | None = 5,
    ) -> None:
        """FAISS-backed archive of papers with optional citation/recency boosting.

        Args:
            embedder: Embeddings implementation used for indexing/search.
            items: Optional initial list of papers to add.
            with_citation_boost: Whether to boost scores using citation counts.
            with_recency_boost: Whether to boost scores using publication time.
            w_citation: Weight for citation boost.
            w_recency: Weight for recency boost.
            max_k: Maximum number of results to return per query.
        """
        self.embedder = Embedder(embedder)
        self._index: faiss.Index | None = None
        self._entries: dict[int, PaperMetadata] = {}

        # Boosting configuration
        self.with_citation_boost = with_citation_boost
        self.with_recency_boost = with_recency_boost
        self.w_citation = w_citation
        self.w_recency = w_recency
        self.max_k = max_k

        self._next_faiss_id = 0
        self._max_citations_seen = 0
        self._current_time = 0
        if items:
            self.add(items)

    def advance_time(self, new_time: int) -> None:
        """Advance the archive's notion of current time (for recency boosts)."""
        if new_time < self._current_time:
            raise ValueError(f"Cannot go backwards in time: {new_time} < {self._current_time}")
        self._current_time = new_time

    @staticmethod
    def _minmax(x: list[float]) -> list[float]:
        if not x:
            return []
        min_x, max_x = min(x), max(x)
        if min_x == max_x:
            return [0.0] * len(x)
        return [(v - min_x) / (max_x - min_x) for v in x]

    def add(self, items: PaperMetadata | list[PaperMetadata]) -> None:
        """Add one or many papers to the archive and index their abstracts."""
        if not isinstance(items, list):
            items = [items]
        if not items:
            return

        abstracts = [item.abstract for item in items]
        vecs = self.embedder.embed_items(abstracts)  # Returns np.ndarray

        if self._index is None:
            base_index = faiss.IndexFlatIP(vecs.shape[1])
            self._index = faiss.IndexIDMap(base_index)

        assert vecs.shape[1] == self._index.d, "Embedding dimension mismatch."
        ids = np.empty(len(items), dtype="int64")

        for i, item in enumerate(items):
            fid = self._next_faiss_id
            self._next_faiss_id += 1
            ids[i] = fid

            self._entries[fid] = item

            if item.citation_count > self._max_citations_seen:
                self._max_citations_seen = item.citation_count
            if item.publication_t > self._current_time:
                self._current_time = item.publication_t

        self._index.add_with_ids(vecs.astype("float32"), ids)

    def update_entry(self, primary_agent_id: str, paper_id: str, new_citation_count: int) -> None:
        """Update citation count for a specific paper if present."""
        found = False
        for fid, paper in self._entries.items():
            if paper.primary_agent_id == primary_agent_id and paper.paper_id == paper_id:
                if paper.citation_count > new_citation_count:
                    raise ValueError("New citation count cannot be less than the current count.")
                paper.citation_count = new_citation_count
                if new_citation_count > self._max_citations_seen:
                    self._max_citations_seen = new_citation_count
                found = True
                break

        if not found:
            raise ValueError(
                f"Paper not found: primary_agent_id={primary_agent_id}, paper_id={paper_id}"
            )

    def prune_older_than(self, min_publication_t: int) -> list[str]:
        """Remove entries with ``publication_t < min_publication_t`` (Markovian memory, experiment 08).

        Removes matching entries from both ``_entries`` and the FAISS ``IndexIDMap``
        (via ``remove_ids``), keeping the internal counters consistent:
        ``_next_faiss_id`` is never rewound (ids must never be reused), ``_current_time``
        is left unchanged (pruning only drops older entries), and
        ``_max_citations_seen`` is recomputed over the survivors. Semantic search,
        recency/citation boosting, and ``get_most_recent_by_agent`` behave correctly on
        the pruned archive, including the empty-archive case.

        Args:
            min_publication_t: Keep only entries with ``publication_t >= min_publication_t``.

        Returns:
            The ``paper_id``s of the removed entries.
        """
        if self._index is None or not self._entries:
            return []

        victims = [
            (fid, paper)
            for fid, paper in list(self._entries.items())
            if paper.publication_t < min_publication_t
        ]
        if not victims:
            return []

        victim_fids = np.array([fid for fid, _ in victims], dtype="int64")
        self._index.remove_ids(victim_fids)

        removed_ids: list[str] = []
        for fid, paper in victims:
            removed_ids.append(paper.paper_id)
            del self._entries[fid]

        # When the archive becomes empty, drop back to the lazy-init state (index None)
        # so query_semantic's empty guard fires and faiss is never searched with k=0.
        # A later add() rebuilds the index; _next_faiss_id is not rewound so ids never
        # collide.
        if not self._entries:
            self._index = None

        # Recompute the citation ceiling over the survivors (do not rewind time/ids).
        self._max_citations_seen = max(
            (p.citation_count for p in self._entries.values()), default=0
        )
        return removed_ids

    def get_most_recent_by_agent(
        self, primary_agent_id: str, k: int | None = 5
    ) -> list[PaperMetadata]:
        """Return up to k most recent papers for a given primary agent."""
        papers = [
            paper for paper in self._entries.values() if paper.primary_agent_id == primary_agent_id
        ]
        papers.sort(key=lambda p: p.publication_t, reverse=True)
        return papers[:k]

    def query_semantic(
        self, query_text: str, k: int | None = 3
    ) -> list[tuple[PaperMetadata, float]]:
        """Semantic search for similar papers with optional boosting.

        Args:
            query_text: Natural-language query.
            k: Desired number of results.

        Returns:
            List of (PaperMetadata, ranking score) tuples. With boosting enabled
            the score is a min-max-normalised ranking value, not a similarity;
            use :meth:`query_semantic_with_similarity` if you need the raw
            inner product.
        """
        return [(paper, score) for paper, score, _sim in self._search_scored(query_text, k)]

    def query_semantic_with_similarity(
        self, query_text: str, k: int | None = 3
    ) -> list[tuple[PaperMetadata, float, float]]:
        """Semantic search returning (paper, ranking_score, raw_similarity).

        Args:
            query_text: Natural-language query.
            k: Desired number of results.

        Returns:
            List of (PaperMetadata, ranking_score, raw_similarity) triples.
            ``ranking_score`` reflects citation/recency boosts and is only
            meaningful for ordering. ``raw_similarity`` is the unmodified inner
            product and is the value metric code should consume.
        """
        return self._search_scored(query_text, k)

    def _search_scored(
        self, query_text: str, k: int | None
    ) -> list[tuple[PaperMetadata, float, float]]:
        """Run the FAISS query and apply boosting, keeping the raw similarity.

        Args:
            query_text: Natural-language query.
            k: Desired number of results; ``None`` falls back to 3.

        Returns:
            List of (PaperMetadata, ranking_score, raw_similarity) triples,
            ordered best-first.
        """
        if self._index is None:
            return []
        k = int(k) if k is not None else 3
        if k <= 0:
            return []
        max_k = int(self.max_k) if self.max_k is not None else k
        k = min(k, max_k)

        k_ = min(
            k * 2 if (self.with_citation_boost or self.with_recency_boost) else k,
            self._index.ntotal,
        )

        query_vec = self.embedder.embed_query(query_text)
        query_vec = query_vec.reshape(1, -1).astype("float32")

        distances, indices = self._index.search(query_vec, k_)
        distances, indices = distances[0], indices[0]

        # Filter once on the FAISS sentinel so every parallel list stays aligned.
        # A retrieved neighbour may legitimately have a negative inner product, so
        # `sim >= 0` is NOT a validity test.
        kept = [
            (int(fid), float(sim)) for sim, fid in zip(distances, indices, strict=True) if fid != -1
        ]
        if not kept:
            return []

        valid_fids = [fid for fid, _ in kept]
        sims = [sim for _, sim in kept]

        if not (self.with_citation_boost or self.with_recency_boost):
            return [(self._entries[fid], sim, sim) for fid, sim in kept][:k]

        cit_counts_raw = [self._entries[fid].citation_count for fid in valid_fids]
        cit_counts = [
            np.log1p(c) / np.log1p(max(self._max_citations_seen, 1)) for c in cit_counts_raw
        ]

        recency_raw = [self._entries[fid].publication_t for fid in valid_fids]
        if self._current_time > 0:
            recency = [r / self._current_time for r in recency_raw]
        else:
            recency = [1.0] * len(recency_raw)

        sims_norm = self._minmax(sims)
        cit_norm = self._minmax(cit_counts) if self.with_citation_boost else [0.0] * len(valid_fids)
        rec_norm = self._minmax(recency) if self.with_recency_boost else [0.0] * len(valid_fids)

        combined = [
            s + self.w_citation * c + self.w_recency * r
            for s, c, r in zip(sims_norm, cit_norm, rec_norm, strict=True)
        ]
        ranked = sorted(
            zip(valid_fids, combined, sims, strict=True),
            key=lambda x: x[1],
            reverse=True,
        )[:k]

        return [(self._entries[fid], score, sim) for fid, score, sim in ranked]
