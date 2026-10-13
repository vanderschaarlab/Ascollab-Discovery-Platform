"""End-to-end graph tests without tools -- uses a real LLM."""

import pytest

from ranker.ranker_core import run_ranker
from ranker.ranker_schemas import RankerResult


@pytest.mark.integration
class TestGraphNoTools:
    """Core 'does it work' tests with a real LLM and no tools."""

    def test_ranker_returns_valid_result(self, llm, sample_papers):
        paper_a, paper_b = sample_papers
        result = run_ranker(
            paper_a=paper_a,
            paper_b=paper_b,
            llm=llm,
            max_iterations=5,
        )
        assert isinstance(result, RankerResult)
        assert result.score in (0, 1)
        assert isinstance(result.justification, str)
        assert len(result.justification) > 0
        assert result.turns >= 1

    def test_ranker_picks_obvious_winner(self, llm, sample_papers):
        paper_a, paper_b = sample_papers
        result = run_ranker(
            paper_a=paper_a,
            paper_b=paper_b,
            llm=llm,
            max_iterations=5,
        )
        # Paper A is clearly better (proper science vs vague hand-waving)
        assert result.score == 1

    def test_ranker_respects_max_iterations(self, llm, sample_papers):
        paper_a, paper_b = sample_papers
        result = run_ranker(
            paper_a=paper_a,
            paper_b=paper_b,
            llm=llm,
            max_iterations=2,
        )
        assert result.turns <= 4  # max_iterations + possible parse retries

    def test_ranker_custom_criteria(self, llm, sample_papers):
        paper_a, paper_b = sample_papers
        result = run_ranker(
            paper_a=paper_a,
            paper_b=paper_b,
            llm=llm,
            criteria_prompt="Only compare based on writing clarity and specificity.",
            max_iterations=5,
        )
        assert isinstance(result, RankerResult)
        assert result.score in (0, 1)
