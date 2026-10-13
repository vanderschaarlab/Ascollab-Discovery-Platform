"""End-to-end graph tests with arXiv tool -- uses a real LLM + live arXiv API."""

import pytest

from ranker.ranker_core import run_ranker
from ranker.ranker_schemas import RankerResult


@pytest.mark.integration
class TestGraphArxiv:
    """Tests with arXiv tool enabled."""

    def test_ranker_with_arxiv(self, llm, sample_papers):
        paper_a, paper_b = sample_papers
        result = run_ranker(
            paper_a=paper_a,
            paper_b=paper_b,
            llm=llm,
            tools_cfg={"arxiv": {"enabled": True, "top_k_results": 3, "load_max_docs": 3}},
            max_iterations=8,
        )
        assert isinstance(result, RankerResult)
        assert result.score in (0, 1)
        assert isinstance(result.justification, str)
        assert len(result.justification) > 0
        # The LLM *may* use arXiv (not forced), so just check result is valid
