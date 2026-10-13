"""Regression tests for parse-retry context and verdict validity.

Every nudge retry must show the model its own rejected attempt, both
inside ``check_verdict`` and on the way back out to ``think``.
An exhausted retry budget must surface as ``valid_result=False`` /
``score=None`` rather than defaulting the score to 0 (a free Paper B win).
"""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from ranker.ranker_agent import build_ranker_agent
from ranker.ranker_core import run_ranker
from ranker.ranker_schemas import RankerResult


def _make_ai_msg(content: str) -> AIMessage:
    """Create an AIMessage with usage metadata for testing."""
    msg = AIMessage(content=content)
    msg.usage_metadata = {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}
    return msg


class _RecordingLLM:
    """Fake LLM recording every rendered prompt; always returns junk."""

    def __init__(self, reply_prefix: str = "unparseable reply"):
        self.prompts: list[list] = []
        self.reply_prefix = reply_prefix

    def invoke(self, prompt_val):
        """Record the rendered prompt and return an unparseable response."""
        self.prompts.append(list(prompt_val.to_messages()))
        return _make_ai_msg(f"{self.reply_prefix} #{len(self.prompts)}")


def _texts(messages) -> list[str]:
    """Return the string contents of a rendered prompt."""
    return [str(getattr(m, "content", "")) for m in messages]


class TestRetryPromptGrowsWithFailedAttempts:
    """Inner loop: retry N+1 must contain retry N's rejected reply."""

    def test_second_retry_sees_first_failed_reply(self):
        llm = _RecordingLLM()
        agent = build_ranker_agent(llm=llm, system_prompt="sys", max_parse_retries=2)

        agent.invoke(
            {
                "messages": [_make_ai_msg("user")],
                "step": 0,
                "last_ai_response": "no verdict here",
                "parsed_result": None,
                "valid_result": None,
            }
        )

        # prompts[0] is the `think` call; prompts[1] and [2] are the two nudges.
        assert len(llm.prompts) >= 3
        first_retry, second_retry = llm.prompts[1], llm.prompts[2]
        assert len(second_retry) > len(first_retry)
        assert "unparseable reply #2" in "\n".join(_texts(second_retry))

    def test_failure_return_carries_messages_to_the_outer_loop(self):
        llm = _RecordingLLM()
        # max_iterations=1 keeps the graph from looping forever on junk output.
        agent = build_ranker_agent(
            llm=llm, system_prompt="sys", max_iterations=1, max_parse_retries=2
        )

        final_state = agent.invoke(
            {
                "messages": [_make_ai_msg("user")],
                "step": 0,
                "last_ai_response": "",
                "parsed_result": None,
                "valid_result": None,
            }
        )

        assert final_state["valid_result"] is False
        # The rejected exchange is persisted in graph state, so a return trip to
        # `think` no longer restarts blind.
        joined = "\n".join(_texts(final_state["messages"]))
        assert "unparseable reply" in joined


class TestUnparseableVerdictIsNotAWin:
    """run_ranker must not silently score an unparseable verdict as 0."""

    def test_exhausted_retries_produce_none_score(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_ai_msg("I cannot decide.")

        result = run_ranker(
            paper_a={"title": "A", "abstract": "A", "manuscript": "Paper A"},
            paper_b={"title": "B", "abstract": "B", "manuscript": "Paper B"},
            llm=mock_llm,
            max_iterations=2,
            max_parse_retries=1,
        )

        assert isinstance(result, RankerResult)
        assert result.valid_result is False
        assert result.score is None

    def test_valid_verdict_still_reports_valid(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_ai_msg(
            "Verdict:\n- Winner: A\n- Justification: Paper A is stronger"
        )

        result = run_ranker(
            paper_a={"title": "A", "abstract": "A", "manuscript": "Paper A"},
            paper_b={"title": "B", "abstract": "B", "manuscript": "Paper B"},
            llm=mock_llm,
            max_iterations=2,
        )

        assert result.valid_result is True
        assert result.score == 1


class TestTraceOutcomeLabels:
    """A parse with no score must not be labelled with an outcome."""

    def test_no_outcome_label_when_score_missing(self):
        llm = _RecordingLLM()
        agent = build_ranker_agent(
            llm=llm,
            system_prompt="sys",
            max_iterations=1,
            max_parse_retries=1,
            # A parser that "succeeds structurally" but yields no score.
            parse_verdict_fn=lambda text: ({"justification": "hmm"}, False),
        )

        agent.invoke(
            {
                "messages": [_make_ai_msg("user")],
                "step": 0,
                "last_ai_response": "",
                "parsed_result": None,
                "valid_result": None,
            }
        )

        parses = [e for e in agent._ranker_tracker["events"] if e["kind"] == "verdict_parse"]
        assert parses
        assert all(e["outcome"] is None for e in parses)
