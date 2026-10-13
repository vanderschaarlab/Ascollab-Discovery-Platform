"""Tests for ranker logging & observability trace."""

import dataclasses
import json
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from ranker.ranker_core import run_ranker
from ranker.ranker_schemas import RankerTrace

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ai_msg(content: str) -> AIMessage:
    """Create an AIMessage with usage metadata for testing."""
    msg = AIMessage(content=content)
    msg.usage_metadata = {
        "input_tokens": 10,
        "output_tokens": 20,
        "total_tokens": 30,
    }
    return msg


_VERDICT_A = "Verdict:\n- Winner: A\n- Justification: Paper A is clearly better"
_VERDICT_B = "Verdict:\n- Winner: B\n- Justification: Paper B is superior in methodology"

_PAPER_A = {"title": "A", "abstract": "A", "manuscript": "Good paper"}
_PAPER_B = {"title": "B", "abstract": "B", "manuscript": "Bad paper"}


def _immediate_verdict_result(winner="A"):
    """Run ranker with a mocked LLM that returns an immediate verdict."""
    text = _VERDICT_A if winner == "A" else _VERDICT_B
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = _make_ai_msg(text)
    return run_ranker(
        paper_a=_PAPER_A,
        paper_b=_PAPER_B,
        llm=mock_llm,
        max_iterations=5,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTracePresent:
    """1. test_trace_present_on_result"""

    def test_trace_present_on_result(self):
        result = _immediate_verdict_result()
        assert result.trace is not None
        assert isinstance(result.trace, RankerTrace)
        assert isinstance(result.trace.ranker_id, str) and len(result.trace.ranker_id) > 0
        assert result.trace.duration_seconds > 0
        assert isinstance(result.trace.events, list) and len(result.trace.events) > 0


class TestTraceEventsStructure:
    """2. test_trace_events_structure"""

    def test_trace_events_structure(self):
        result = _immediate_verdict_result()
        events = result.trace.events
        for event in events:
            assert "seq" in event
            assert "ts" in event
            assert "kind" in event

        # seq values monotonically increasing
        seqs = [e["seq"] for e in events]
        assert seqs == sorted(seqs)
        assert len(seqs) == len(set(seqs))  # all unique


class TestTraceLlmCallEvent:
    """3. test_trace_has_llm_call_event"""

    def test_trace_has_llm_call_event(self):
        result = _immediate_verdict_result()
        llm_events = [e for e in result.trace.events if e["kind"] == "llm_call"]
        assert len(llm_events) >= 1
        ev = llm_events[0]
        for key in (
            "turn",
            "input_messages",
            "output_text",
            "input_tokens",
            "output_tokens",
            "duration_seconds",
        ):
            assert key in ev, f"Missing key {key!r} in llm_call event"


class TestTraceVerdictParseEvent:
    """4. test_trace_has_verdict_parse_event"""

    def test_trace_has_verdict_parse_event(self):
        result = _immediate_verdict_result()
        vp_events = [e for e in result.trace.events if e["kind"] == "verdict_parse"]
        assert len(vp_events) >= 1
        ev = vp_events[0]
        assert ev["valid"] is True
        assert ev["outcome"] in ("A", "B")


class TestTraceRecordsToolCall:
    """5. test_trace_records_tool_call"""

    def test_trace_records_tool_call(self):
        call_count = 0

        def mock_invoke(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_ai_msg(
                    'Let me search arXiv.\nAction: {"tool": "arxiv", "args": '
                    '{"query": "HIF pathway activity and survival in clear cell renal carcinoma"}}'
                )
            return _make_ai_msg(_VERDICT_A)

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = mock_invoke

        # Create a mock arxiv tool
        mock_tool = MagicMock()
        mock_tool.invoke.return_value = "Found 3 relevant papers about gene expression..."

        result = run_ranker(
            paper_a=_PAPER_A,
            paper_b=_PAPER_B,
            llm=mock_llm,
            tools_cfg={"arxiv": {"enabled": True}},
            max_iterations=5,
        )

        tc_events = [e for e in result.trace.events if e["kind"] == "tool_call"]
        assert len(tc_events) >= 1
        ev = tc_events[0]
        for key in ("tool", "args", "result", "error"):
            assert key in ev, f"Missing key {key!r} in tool_call event"


class TestTraceRecordsNudgeRetries:
    """6. test_trace_records_nudge_retries"""

    def test_trace_records_nudge_retries(self):
        call_count = 0

        def mock_invoke(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_ai_msg("I'm thinking about this...")
            return _make_ai_msg(_VERDICT_A)

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = mock_invoke

        result = run_ranker(
            paper_a=_PAPER_A,
            paper_b=_PAPER_B,
            llm=mock_llm,
            max_iterations=5,
            max_parse_retries=2,
        )

        assert result.trace.summary["nudge_retries"] >= 1
        vp_events = [e for e in result.trace.events if e["kind"] == "verdict_parse"]
        assert any(e["valid"] is False for e in vp_events)


class TestTraceSummaryCounts:
    """7. test_trace_summary_counts"""

    def test_trace_summary_counts(self):
        result = _immediate_verdict_result()
        summary = result.trace.summary
        for key in ("llm_calls", "tool_calls", "input_tokens", "output_tokens"):
            assert key in summary, f"Missing key {key!r} in summary"
        assert summary["llm_calls"] >= 1


class TestCustomRankerId:
    """8. test_custom_ranker_id"""

    def test_custom_ranker_id(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_ai_msg(_VERDICT_A)

        result = run_ranker(
            paper_a=_PAPER_A,
            paper_b=_PAPER_B,
            llm=mock_llm,
            max_iterations=5,
            ranker_id="my_test_id",
        )

        assert result.trace.ranker_id == "my_test_id"


class TestTraceSerializesToJson:
    """9. test_trace_serializes_to_json"""

    def test_trace_serializes_to_json(self):
        result = _immediate_verdict_result()
        # Should not raise
        serialized = json.dumps(dataclasses.asdict(result.trace))
        assert isinstance(serialized, str)
        # Verify it round-trips
        data = json.loads(serialized)
        assert data["ranker_id"] == result.trace.ranker_id


@pytest.mark.integration
class TestTraceWithRealLlm:
    """10. test_trace_with_real_llm"""

    def test_trace_with_real_llm(self, llm, sample_papers):
        paper_a, paper_b = sample_papers
        result = run_ranker(
            paper_a=paper_a,
            paper_b=paper_b,
            llm=llm,
            max_iterations=5,
        )

        assert result.trace is not None
        llm_events = [e for e in result.trace.events if e["kind"] == "llm_call"]
        assert len(llm_events) >= 1
        assert result.trace.summary["llm_calls"] == result.turns
        assert result.trace.paper_a_title == paper_a["title"]
        assert result.trace.paper_b_title == paper_b["title"]
