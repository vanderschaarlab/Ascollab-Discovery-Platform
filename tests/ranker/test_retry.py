"""Tests for LLM retry logic and nudge-on-bad-parse behavior."""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from ranker.ranker_agent import _invoke_with_retry
from ranker.ranker_core import run_ranker
from ranker.ranker_schemas import RankerResult


def _make_ai_msg(content: str) -> AIMessage:
    """Create an AIMessage with usage metadata for testing."""
    msg = AIMessage(content=content)
    msg.usage_metadata = {
        "input_tokens": 10,
        "output_tokens": 20,
        "total_tokens": 30,
    }
    return msg


class _FakeAIMessage:
    """Minimal AI message mock for _invoke_with_retry tests only.

    Not usable with LangGraph state (which requires real BaseMessage objects),
    but fine for testing the retry function in isolation.
    """

    def __init__(self, content: str):
        self.content = content
        self.usage_metadata = {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
        }


class TestRetryLogic:
    """Test _invoke_with_retry with mocked LLM."""

    def test_retry_on_rate_limit(self):
        from openai import RateLimitError

        mock_llm = MagicMock()
        # Fail twice, then succeed
        err = RateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(status_code=429, headers={}),
            body=None,
        )
        mock_llm.invoke.side_effect = [err, err, _FakeAIMessage("ok")]

        result = _invoke_with_retry(
            mock_llm,
            "prompt",
            {"max_retries": 3, "base_delay_seconds": 0.01, "max_delay_seconds": 0.1},
        )
        assert result.content == "ok"
        assert mock_llm.invoke.call_count == 3

    def test_retry_exhausted(self):
        from openai import RateLimitError

        mock_llm = MagicMock()
        err = RateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(status_code=429, headers={}),
            body=None,
        )
        mock_llm.invoke.side_effect = err

        with pytest.raises(RateLimitError):
            _invoke_with_retry(
                mock_llm,
                "prompt",
                {"max_retries": 2, "base_delay_seconds": 0.01, "max_delay_seconds": 0.1},
            )
        assert mock_llm.invoke.call_count == 3  # initial + 2 retries


class TestNudgeOnBadParse:
    """Test that the graph nudges on unparseable output then succeeds."""

    def test_nudge_on_bad_parse(self):
        # First call: returns unparseable text
        # Second call (after nudge): returns valid verdict
        call_count = 0

        def mock_invoke(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_ai_msg("I'm thinking about this...")
            return _make_ai_msg("Verdict:\n- Winner: A\n- Justification: Paper A is clearly better")

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = mock_invoke

        result = run_ranker(
            paper_a={"title": "A", "abstract": "A", "manuscript": "Good paper"},
            paper_b={"title": "B", "abstract": "B", "manuscript": "Bad paper"},
            llm=mock_llm,
            max_iterations=5,
            max_parse_retries=2,
        )

        assert isinstance(result, RankerResult)
        assert result.score == 1
        assert result.turns >= 2

    def test_graph_with_immediate_verdict(self):
        """Test that the graph works when LLM returns a valid verdict immediately."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_ai_msg(
            "Verdict:\n- Winner: B\n- Justification: Paper B is superior in methodology"
        )

        result = run_ranker(
            paper_a={"title": "A", "abstract": "A", "manuscript": "Paper A text"},
            paper_b={"title": "B", "abstract": "B", "manuscript": "Paper B text"},
            llm=mock_llm,
            max_iterations=5,
        )

        assert isinstance(result, RankerResult)
        assert result.score == 0
        assert result.turns >= 1

    def test_graph_with_tool_use_then_verdict(self):
        """Test that the graph handles tool use followed by a verdict."""
        call_count = 0

        def mock_invoke(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_ai_msg(
                    'Let me search arXiv.\nAction: {"tool": "arxiv", "args": {"query": "gene expression"}}'
                )
            return _make_ai_msg(
                "Verdict:\n- Winner: A\n- Justification: Based on literature, Paper A is stronger"
            )

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = mock_invoke

        # Create a mock arxiv tool
        mock_tool = MagicMock()
        mock_tool.invoke.return_value = "Found 3 relevant papers about gene expression..."

        result = run_ranker(
            paper_a={"title": "A", "abstract": "A", "manuscript": "Good paper"},
            paper_b={"title": "B", "abstract": "B", "manuscript": "Bad paper"},
            llm=mock_llm,
            tools_cfg={"arxiv": {"enabled": True}},
            max_iterations=5,
        )

        assert isinstance(result, RankerResult)
        assert result.score in (0, 1)
