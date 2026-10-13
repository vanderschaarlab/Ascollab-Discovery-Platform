"""Regression: a malformed citation tag must not silently discard a review."""

import asyncio
import logging
import time

import pytest
from langchain_core.messages import AIMessage

from system import review_one_step as ros


def _review_body(citation: str) -> str:
    return f"""
# Review Text
1. Summary
Strong work supported by prior results {citation}.
2. Motivation
Clear.
3. Claims and Evidence
Reasonable.
4. Soundness of Methodology
Mostly sound.
5. Relation to Prior Knowledge
Adequate.
6. Novelty and Significance
Moderate novelty.
7. Other Comments
None.

# Quantitative Ratings
- Support: 3
- Soundness: 3
- Significance: 3
- Originality: 3
- Overall Recommendation: 3
""".strip()


VALID_REVIEW = _review_body("[Archive: paper_01]")
INVALID_REVIEW = _review_body("[Archive: paper_01 extra junk here]")


class _ScriptedLLM:
    """Async chat model returning queued contents, recording call windows."""

    def __init__(self, *contents, delay: float = 0.0):
        self.contents = list(contents)
        self.delay = delay
        self.windows: list[tuple[float, float]] = []

    async def ainvoke(self, messages, **_):
        started = time.perf_counter()
        if self.delay:
            await asyncio.sleep(self.delay)
        self.windows.append((started, time.perf_counter()))
        content = self.contents.pop(0) if self.contents else self.contents_default
        return AIMessage(content=content)

    contents_default = INVALID_REVIEW


# ---------------------------------------------------------------------------
# The shared helper
# ---------------------------------------------------------------------------


def test_retry_helper_accepts_a_repaired_review():
    parsed, ok = asyncio.run(
        ros._retry_review_for_citation_schema(
            llm_model=_ScriptedLLM(VALID_REVIEW),
            review_text=INVALID_REVIEW,
            citation_errors=["bad tag"],
            stage_label="test",
            require_assertion_justification=True,
        )
    )

    assert ok is True
    assert "Strong work" in parsed["Review Text"]


def test_retry_helper_rejects_a_still_broken_review():
    parsed, ok = asyncio.run(
        ros._retry_review_for_citation_schema(
            llm_model=_ScriptedLLM(INVALID_REVIEW),
            review_text=INVALID_REVIEW,
            citation_errors=["bad tag"],
            stage_label="test",
            require_assertion_justification=True,
        )
    )

    assert ok is False
    assert parsed is not None


def test_retry_helper_validates_the_stored_section_not_the_raw_response():
    """Tags placed outside the 'Review Text' section must not rescue a retry."""
    sneaky = _review_body("(no tag here)") + "\n\n[Archive: paper_01]"

    _parsed, ok = asyncio.run(
        ros._retry_review_for_citation_schema(
            llm_model=_ScriptedLLM(sneaky),
            review_text=INVALID_REVIEW,
            citation_errors=["bad tag"],
            stage_label="test",
            require_assertion_justification=True,
        )
    )

    assert ok is False


# ---------------------------------------------------------------------------
# The review_one_step path
# ---------------------------------------------------------------------------


class _StubRegistry:
    def __init__(self, reviewer_ids):
        self.reviewer_ids = list(reviewer_ids)

    def get_all_agents(self):
        from system.schema import AgentProfile

        return [
            AgentProfile(
                agent_id=rid,
                behavior="",
                expertise="genomics",
                expertise_topics=[],
                citation_count=0,
                num_accepted_papers=0,
            )
            for rid in self.reviewer_ids
        ]

    def get_agent(self, agent_id):
        for agent in self.get_all_agents():
            if agent.agent_id == agent_id:
                return True, agent
        return False, None


def _paper(pid="paper_99", author="author_01"):
    from system.schema import PaperMetadataPrivate

    return PaperMetadataPrivate(
        paper_id=pid,
        primary_agent_id=author,
        collab_agent_ids=[],
        title="T",
        abstract="A",
        manuscript="M",
        citation_count=0,
        publication_t=1,
        cited_paper_ids=[],
        code_script="",
    )


def _install_reviewer_stubs(monkeypatch, first_pass_by_reviewer):
    """Stub the reviewer agent machinery so only the citation path is exercised."""
    monkeypatch.setattr(ros, "get_reviewer_agent", lambda *a, **k: object())
    monkeypatch.setattr(ros, "get_logs", lambda _agent: None)

    async def fake_run_one_reviewer(agent, state, reviewer_id, paper_id, iteration, **kwargs):
        from agents.review.rev_agent_utils import parse_review_text

        parsed, _ok = parse_review_text(first_pass_by_reviewer[reviewer_id])
        return {"route": "final", "parsed_review": parsed}

    monkeypatch.setattr(ros, "run_one_reviewer", fake_run_one_reviewer)


def _run_review(monkeypatch, reviewer_ids, first_pass, retry_llm):
    from stores.private_paper_archive import PrivatePaperArchive

    _install_reviewer_stubs(monkeypatch, first_pass)
    paper = _paper()

    return asyncio.run(
        ros.review_one_step(
            iteration=1,
            llm=retry_llm,
            llm_for=None,
            candidate_papers=[paper],
            network_metadata=None,
            private_archive=PrivatePaperArchive(),
            agent_registry=_StubRegistry(reviewer_ids),
            agent_registry_search_tool=None,
            task_config=type("Cfg", (), {"collab_agent_tools": {}})(),
            num_reviewers_per_paper=len(reviewer_ids),
            review_max_iterations=2,
            reviewer_assignment_strategy="random",
            require_assertion_justification=True,
        )
    )


def test_a_repaired_review_is_kept(monkeypatch):
    reviews_by_paper, _logs = _run_review(
        monkeypatch,
        ["rev_01"],
        {"rev_01": INVALID_REVIEW},
        _ScriptedLLM(VALID_REVIEW),
    )

    assert len(reviews_by_paper["paper_99"]) == 1
    assert reviews_by_paper["paper_99"][0].reviewer_agent_id == "rev_01"


def test_a_review_failing_twice_is_dropped_loudly(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    reviews_by_paper, _logs = _run_review(
        monkeypatch,
        ["rev_01"],
        {"rev_01": INVALID_REVIEW},
        _ScriptedLLM(INVALID_REVIEW),
    )

    assert reviews_by_paper == {}
    out = caplog.text
    assert "queued for schema-nudge retry" in out
    assert "failed citation schema after retry" in out


def test_retries_run_concurrently(monkeypatch):
    """Three failures cost one round-trip, not three."""
    reviewers = ["rev_01", "rev_02", "rev_03"]
    llm = _ScriptedLLM(*([VALID_REVIEW] * 3), delay=0.2)

    started = time.perf_counter()
    reviews_by_paper, _logs = _run_review(
        monkeypatch, reviewers, dict.fromkeys(reviewers, INVALID_REVIEW), llm
    )
    elapsed = time.perf_counter() - started

    assert len(reviews_by_paper["paper_99"]) == 3
    assert len(llm.windows) == 3
    assert elapsed < 0.5, "retries appear to have been serialised"
    # Every retry window overlaps the first one.
    first_start, _first_end = llm.windows[0]
    assert all(start < first_start + 0.2 for start, _end in llm.windows)


def test_valid_reviews_never_enter_the_retry_path(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    llm = _ScriptedLLM()

    reviews_by_paper, _logs = _run_review(monkeypatch, ["rev_01"], {"rev_01": VALID_REVIEW}, llm)

    assert len(reviews_by_paper["paper_99"]) == 1
    assert llm.windows == []
    assert "schema-nudge retry" not in caplog.text


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
