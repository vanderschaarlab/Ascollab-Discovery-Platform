from typing import Any

from system.schema import PaperMetadataPrivate


class PrivatePaperArchive:
    def __init__(
        self,
        items: list[PaperMetadataPrivate] | None = None,
    ) -> None:
        """In-memory archive for private papers and their reviews."""
        self._entries: dict[str, list[PaperMetadataPrivate]] = {}
        # Store final reviewer texts + scores per paper_id
        # Shape: {paper_id: [ {iteration, reviewer_id, review_text, scores}, ... ]}
        self._reviews: dict[str, list[dict[str, Any]]] = {}
        # Optional experiment metadata keyed by assignment record id.
        self._review_assignments: dict[str, dict[str, Any]] = {}

        if items:
            self.add(items)

    def add(
        self,
        items: PaperMetadataPrivate | list[PaperMetadataPrivate],
        *,
        on_duplicate: str = "skip",
    ) -> list[str]:
        """Add private papers, enforcing unique paper_id across agents.

        Args:
            items: One paper or a list of papers to add.
            on_duplicate: "skip" (default) records the duplicate and continues so a
                single collision cannot drop the rest of the batch; "raise" restores
                the previous strict behaviour.

        Returns:
            The list of paper_ids that were skipped as duplicates.

        Raises:
            ValueError: If a citation count is negative, or a duplicate is found
                while on_duplicate="raise".
        """
        if not isinstance(items, list):
            items = [items]

        known_ids = {
            existing.paper_id
            for agent_papers in self._entries.values()
            for existing in agent_papers
        }
        skipped: list[str] = []

        for item in items:
            if not item.paper_id or not item.paper_id.strip():
                continue  # Skip empty paper_id
            if item.citation_count < 0:
                raise ValueError(f"Citation count cannot be negative: {item.citation_count}")

            if item.paper_id in known_ids:
                if on_duplicate == "raise":
                    raise ValueError(
                        f"Paper with paper_id '{item.paper_id}' already exists in the archive"
                    )
                skipped.append(item.paper_id)
                continue

            known_ids.add(item.paper_id)
            self._entries.setdefault(item.primary_agent_id, []).append(item)

        return skipped

    def update_entry(self, primary_agent_id: str, paper_id: str, new_citation_count: int) -> None:
        """Update citation count for a paper belonging to a primary agent."""
        if new_citation_count < 0:
            raise ValueError("Citation count cannot be negative.")

        found = False
        for paper in self._entries.get(primary_agent_id, []):
            if paper.paper_id == paper_id:
                if new_citation_count < paper.citation_count:
                    raise ValueError("New citation count cannot be less than the current count.")
                paper.citation_count = new_citation_count
                found = True
                break

        if not found:
            raise ValueError(
                f"Paper not found: primary_agent_id={primary_agent_id}, paper_id={paper_id}"
            )

    def add_reviews_for_iteration(
        self,
        iteration: int,
        reviews_by_paper: dict[str, list[dict[str, Any]]],
    ) -> None:
        """Persist final reviewer outputs for this iteration.

        Args:
            iteration: Current iteration index (e.g., t18).
            reviews_by_paper: Mapping paper_id -> list of review dicts.
        """
        # Backwards-compat: if older pickles don’t have _reviews, create it
        if not hasattr(self, "_reviews") or self._reviews is None:
            self._reviews = {}

        for paper_id, entries in (reviews_by_paper or {}).items():
            if not paper_id:
                continue

            bucket = self._reviews.setdefault(paper_id, [])
            for e in entries or []:
                bucket.append(
                    {
                        "iteration": iteration,
                        "reviewer_id": e.get("reviewer_id"),
                        "review_text": e.get("review_text", ""),
                        "scores": dict(e.get("scores") or {}),
                    }
                )

    @property
    def reviews_by_paper(self) -> dict[str, list[dict[str, Any]]]:
        """
        Read-only view of stored reviews:
        {paper_id: [ {iteration, reviewer_id, review_text, scores}, ... ]}
        """
        return self._reviews

    def set_reviews(self, reviews_by_paper: dict[str, list[dict[str, Any]]]) -> None:
        """Replace stored reviews with the provided mapping."""
        self._reviews = {
            pid: list(entries or []) for pid, entries in (reviews_by_paper or {}).items()
        }

    @property
    def review_assignments(self) -> dict[str, dict[str, Any]]:
        """Read-only view of persisted reviewer-assignment metadata."""
        if not hasattr(self, "_review_assignments") or self._review_assignments is None:
            self._review_assignments = {}
        return self._review_assignments

    def set_review_assignments(self, assignments: dict[str, Any]) -> None:
        """Replace stored reviewer-assignment metadata."""
        self._review_assignments = {
            str(record_id): dict(record or {})
            for record_id, record in (assignments or {}).items()
            if record_id
        }

    def add_review_assignment_record(self, record_id: str, record: dict[str, Any]) -> None:
        """Persist one reviewer-assignment metadata record."""
        if not record_id:
            return
        if not hasattr(self, "_review_assignments") or self._review_assignments is None:
            self._review_assignments = {}
        self._review_assignments[str(record_id)] = dict(record or {})

    def get_by_primary_agent(
        self, primary_agent_id: str, k: int | None = 3
    ) -> list[PaperMetadataPrivate]:
        """Return up to k papers for a given primary agent, sorted by publication time."""
        if primary_agent_id not in self._entries:
            return []
        papers = sorted(
            self._entries[primary_agent_id], key=lambda p: p.publication_t, reverse=False
        )
        if k is None:
            return papers
        return papers[:k]
