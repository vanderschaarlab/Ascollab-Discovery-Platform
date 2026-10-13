import math

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.embeddings.embeddings import Embeddings

from system.schema import AgentProfile


class AgentRegistry:
    def __init__(
        self,
        embedder: Embeddings,
        agents: list[AgentProfile] | None = None,
        *,
        with_citation_boost: bool | None = False,
        w_sim: float | None = 0.8,
        w_cit: float | None = 0.2,
        max_k: int | None = 5,
    ):
        """In-memory agent registry backed by a FAISS vector store.

        Args:
            embedder: Embedding model used for similarity search.
            agents: Optional initial list of AgentProfile objects to insert.
            with_citation_boost: Whether to boost rankings using citation counts.
            w_sim: Weight for similarity when boosting.
            w_cit: Weight for citation count when boosting.
            max_k: Maximum number of search results to return.
        """
        self._max_citations_seen = 0
        self.embedder = embedder
        self.vector_store = None

        # Boosting configuration
        self.with_citation_boost = with_citation_boost
        self.w_sim = w_sim
        self.w_cit = w_cit
        self.max_k = max_k

        if agents is not None:
            self.insert(agents)

    def insert(self, agents: AgentProfile | list[AgentProfile]) -> None:
        """Insert one or more agents into the registry, replacing by ID."""
        if not isinstance(agents, list):
            agents = [agents]

        docs = []
        ids = []
        for agent in agents:
            expertise_text = (
                agent.expertise
                if isinstance(agent.expertise, str)
                else ", ".join(agent.expertise or [])
            )
            docs.append(
                Document(
                    page_content=expertise_text,
                    metadata={
                        "agent_id": agent.agent_id,
                        "agent_behavior": agent.behavior,
                        "agent_expertise": agent.expertise,
                        "agent_expertise_topics": agent.expertise_topics,
                        "citation_count": agent.citation_count,
                        "num_accepted_papers": agent.num_accepted_papers,
                        "metric_weights": getattr(agent, "metric_weights", None),
                        "reputation": getattr(agent, "reputation", None),
                        "pause_state": getattr(agent, "pause_state", None),
                        "learning_state": getattr(agent, "learning_state", None),
                    },
                )
            )
            ids.append(agent.agent_id)

        if agents:
            self._max_citations_seen = max(
                self._max_citations_seen, max(agent.citation_count for agent in agents)
            )

        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(
                docs, embedding=self.embedder, distance_strategy=DistanceStrategy.COSINE, ids=ids
            )
        else:
            existing_ids = []
            for agent_id in ids:
                try:
                    self.vector_store.delete([agent_id])
                    existing_ids.append(agent_id)
                except (ValueError, KeyError):
                    pass

            self.vector_store.add_documents(docs, ids=ids)

    def get_agent(self, agent_id: str) -> tuple[bool, AgentProfile]:
        """Retrieve an agent profile by ID."""
        if self.vector_store is None:
            raise KeyError(f"Agent {agent_id} not found")

        try:
            doc = self.vector_store.docstore.search(agent_id)
            if doc is None or isinstance(doc, str):
                raise KeyError(f"Agent {agent_id} not found")
            return True, self._convert_doc_to_profile(doc)
        except Exception:
            return False, None

    def update(
        self,
        agent_id: str,
        *,
        new_expertise: str | None = None,
        new_citations: int | None = None,
        new_accepted_papers: int | None = None,
        new_metric_weights: dict | None = None,
        new_reputation: dict | None = None,
        new_pause_state: dict | None = None,
        new_learning_state: dict | None = None,
    ) -> None:
        """Update an agent's expertise, citation count, or accepted paper count."""
        if self.vector_store is None:
            raise KeyError(f"Agent {agent_id} not found")

        try:
            doc = self.vector_store.docstore.search(agent_id)
        except Exception as exc:
            raise KeyError(f"Agent {agent_id} not found") from exc
        if doc is None or isinstance(doc, str):
            raise KeyError(f"Agent {agent_id} not found")

        if new_expertise is not None:
            doc.page_content = new_expertise
            doc.metadata["agent_expertise"] = new_expertise
        if new_citations is not None:
            if new_citations < 0:
                raise ValueError("Citation count cannot be negative.")
            if new_citations < doc.metadata["citation_count"]:
                raise ValueError("New citation count cannot be less than the current count.")
            doc.metadata["citation_count"] = new_citations
            self._max_citations_seen = max(self._max_citations_seen, new_citations)
        if new_accepted_papers is not None:
            if new_accepted_papers < 0:
                raise ValueError("Number of accepted papers cannot be negative.")
            if new_accepted_papers < doc.metadata["num_accepted_papers"]:
                raise ValueError("New accepted papers count cannot be less than the current count.")
            doc.metadata["num_accepted_papers"] = new_accepted_papers
        if new_reputation is not None:
            doc.metadata["reputation"] = new_reputation
        if new_pause_state is not None:
            doc.metadata["pause_state"] = new_pause_state
        if new_metric_weights is not None:
            doc.metadata["metric_weights"] = new_metric_weights
        if new_learning_state is not None:
            doc.metadata["learning_state"] = new_learning_state

        self.vector_store.delete([agent_id])
        self.vector_store.add_documents([doc], ids=[agent_id])

    @staticmethod
    def _to_similarity(cosine_distance: float) -> float:
        """Convert cosine distance to similarity score in [0, 1]."""
        s = 1.0 - cosine_distance
        return max(0.0, min(1.0, s))

    @staticmethod
    def _minmax(xs: list[float]) -> list[float]:
        if not xs:
            return xs
        mn, mx = min(xs), max(xs)
        if mx - mn < 1e-6:
            return [0.0 for _ in xs]
        return [(x - mn) / (mx - mn) for x in xs]

    @staticmethod
    def _convert_doc_to_profile(doc: Document) -> AgentProfile:
        metadata = doc.metadata
        return AgentProfile(
            agent_id=metadata["agent_id"],
            behavior=metadata["agent_behavior"],
            expertise=metadata["agent_expertise"],
            expertise_topics=metadata.get("agent_expertise_topics"),
            citation_count=metadata["citation_count"],
            num_accepted_papers=metadata["num_accepted_papers"],
            metric_weights=metadata.get("metric_weights"),
            reputation=metadata.get("reputation"),
            pause_state=metadata.get("pause_state"),
            learning_state=metadata.get("learning_state"),
        )

    def get_all_agents(self) -> list[AgentProfile]:
        """Return all stored agent profiles."""
        if self.vector_store is None:
            return []

        all_docs = []
        for doc_id in self.vector_store.index_to_docstore_id.values():
            doc = self.vector_store.docstore.search(doc_id)
            if doc is not None and not isinstance(doc, str):
                all_docs.append(doc)

        profiles = []
        for doc in all_docs:
            profile = self._convert_doc_to_profile(doc)
            profiles.append(profile)

        return profiles

    def query(self, query_text: str, *, k: int | None = 3) -> list[tuple[AgentProfile, float]]:
        """Query agents by text similarity with optional citation boosting."""
        if self.vector_store is None or k <= 0:
            return []

        k = min(k, self.max_k)

        k_ = k if not self.with_citation_boost else k * 2
        candidates = self.vector_store.similarity_search_with_score(query_text, k=k_)

        if not self.with_citation_boost:
            results = []
            for doc, dist in candidates[:k]:
                profile = self._convert_doc_to_profile(doc)
                results.append((profile, self._to_similarity(dist)))
            return results

        sims = [self._to_similarity(dist) for _, dist in candidates]
        cit_counts_raw = [int(doc.metadata["citation_count"]) for doc, _ in candidates]
        # Before any agent is cited the ceiling is 0; guard it as PaperArchive does.
        cit_ceiling = math.log1p(max(self._max_citations_seen, 1))
        cit_counts = [math.log1p(c) / cit_ceiling for c in cit_counts_raw]

        sims_norm = self._minmax(sims)
        cit_norm = self._minmax(cit_counts)

        combined = [
            self.w_sim * s + self.w_cit * c for s, c in zip(sims_norm, cit_norm, strict=False)
        ]
        reranked = sorted(
            [(doc, score) for (doc, _), score in zip(candidates, combined, strict=False)],
            key=lambda x: x[1],
            reverse=True,
        )[:k]

        results = []
        for doc, score in reranked:
            profile = self._convert_doc_to_profile(doc)
            results.append((profile, score))
        return results
