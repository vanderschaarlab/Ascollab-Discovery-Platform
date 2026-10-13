from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage

from system.schema import (
    AgentNetworkMetadata,
    AgentProfile,
    PaperMetadata,
    PaperMetadataPrivate,
    PaperMetaReview,
    PaperReview,
)
from tests.isolation import remove_tracing_variables


def _redact_test_secrets(value: Any) -> Any:
    """Return a copy safe to show in pytest failure output."""
    sensitive_key_parts = (
        "api_key",
        "apikey",
        "api-key",
        "secret",
        "password",
        "token",
    )
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            key_text = str(key).lower()
            if any(part in key_text for part in sensitive_key_parts):
                redacted[key] = "***REDACTED***"
            else:
                redacted[key] = _redact_test_secrets(item)
        return redacted
    if isinstance(value, list):
        return [_redact_test_secrets(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_test_secrets(item) for item in value)
    return value


def pytest_configure(config: pytest.Config) -> None:
    """Prevent OmegaConf assertion introspection from printing secrets."""
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf
    except Exception:
        return

    if getattr(DictConfig, "_ascollab_redacted_repr", False):
        return

    def _redacted_repr(self: Any) -> str:
        try:
            container = OmegaConf.to_container(self, resolve=False)
            return repr(_redact_test_secrets(container))
        except Exception:
            return f"<{type(self).__name__} ***REDACTED***>"

    DictConfig.__repr__ = _redacted_repr
    ListConfig.__repr__ = _redacted_repr
    DictConfig._ascollab_redacted_repr = True
    ListConfig._ascollab_redacted_repr = True


@pytest.fixture(autouse=True)
def _no_langsmith_tracing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every test without LangChain/LangSmith tracing variables (see tests/isolation.py)."""
    remove_tracing_variables(monkeypatch)


class DeterministicEmbeddings(Embeddings):
    """Small deterministic embedding model for tests that need vector stores."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        digest = hashlib.sha256((text or "").encode("utf-8")).digest()
        values = [digest[i] / 255.0 for i in range(self.dim)]
        norm = sum(v * v for v in values) ** 0.5 or 1.0
        return [v / norm for v in values]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class QueuedChatModel:
    """Minimal invoke-compatible fake chat model returning queued AI messages."""

    def __init__(self, *contents: str, usage: dict[str, int] | None = None) -> None:
        self.contents = list(contents)
        self.usage = usage or {}
        self.calls: list[list[Any]] = []

    def invoke(self, messages: list[Any], **_: Any) -> AIMessage:
        self.calls.append(list(messages))
        if not self.contents:
            raise AssertionError("QueuedChatModel invoked more times than expected")
        msg = AIMessage(content=self.contents.pop(0))
        msg.usage_metadata = dict(self.usage)
        return msg


@pytest.fixture
def fake_embeddings() -> DeterministicEmbeddings:
    return DeterministicEmbeddings()


@pytest.fixture
def agent_profile_factory() -> Callable[..., AgentProfile]:
    def _make(**overrides: Any) -> AgentProfile:
        data = {
            "agent_id": "agent_01",
            "behavior": "Careful and empirical.",
            "expertise": "cancer genomics and statistics",
            "expertise_topics": ["genomics", "statistics"],
            "citation_count": 4,
            "num_accepted_papers": 2,
        }
        data.update(overrides)
        return AgentProfile(**data)

    return _make


@pytest.fixture
def paper_factory() -> Callable[..., PaperMetadata]:
    def _make(**overrides: Any) -> PaperMetadata:
        data = {
            "paper_id": "paper_01",
            "primary_agent_id": "agent_01",
            "collab_agent_ids": [],
            "title": "A reproducible genomics study",
            "abstract": "We analyze expression signatures with validation.",
            "manuscript": "Methods include validation. Results report confidence intervals.",
            "citation_count": 1,
            "publication_t": 1,
            "cited_paper_ids": [],
        }
        data.update(overrides)
        return PaperMetadata(**data)

    return _make


@pytest.fixture
def private_paper_factory() -> Callable[..., PaperMetadataPrivate]:
    def _make(**overrides: Any) -> PaperMetadataPrivate:
        data = {
            "paper_id": "paper_01",
            "primary_agent_id": "agent_01",
            "collab_agent_ids": [],
            "title": "A reproducible genomics study",
            "abstract": "We analyze expression signatures with validation.",
            "manuscript": "Methods include validation. Results report confidence intervals.",
            "citation_count": 1,
            "publication_t": 1,
            "cited_paper_ids": [],
            "code_script": "print('ok')",
        }
        data.update(overrides)
        return PaperMetadataPrivate(**data)

    return _make


@pytest.fixture
def review_factory() -> Callable[..., PaperReview]:
    def _make(**overrides: Any) -> PaperReview:
        data = {
            "paper_id": "paper_01",
            "reviewer_agent_id": "reviewer_01",
            "review_text": "Clear evidence and appropriate validation.",
            "score": {"Overall Recommendation": 0.8},
        }
        data.update(overrides)
        return PaperReview(**data)

    return _make


@pytest.fixture
def metareview_factory() -> Callable[..., PaperMetaReview]:
    def _make(**overrides: Any) -> PaperMetaReview:
        data = {
            "paper_id": "paper_01",
            "meta_review_text": "Strong paper.",
            "overall_score": 0.8,
            "rank": "1/2",
            "justification": "Best supported submission.",
            "decision": "accept",
        }
        data.update(overrides)
        return PaperMetaReview(**data)

    return _make


@pytest.fixture
def network_metadata() -> AgentNetworkMetadata:
    return AgentNetworkMetadata(num_agents=2, num_papers=3)
