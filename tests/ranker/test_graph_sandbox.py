"""End-to-end graph tests with sandbox tool -- uses a real LLM + live sandbox server."""

import pytest

from ranker.ranker_core import run_ranker
from ranker.ranker_schemas import RankerResult


@pytest.mark.integration
@pytest.mark.sandbox
class TestGraphSandbox:
    """Tests with sandbox tool enabled. Requires a running sandbox server."""

    def test_ranker_with_sandbox(self, llm, sandbox_client):
        paper_a = {
            "title": "Efficient matrix factorization for gene expression",
            "abstract": (
                "We propose a non-negative matrix factorization approach for "
                "analyzing gene expression data, achieving 15% faster convergence."
            ),
            "manuscript": (
                "Methods: We implemented NMF using the multiplicative update rule.\n"
                "Code:\n```python\nimport numpy as np\n"
                "def nmf(V, k, iters=100):\n"
                "    W = np.random.rand(V.shape[0], k)\n"
                "    H = np.random.rand(k, V.shape[1])\n"
                "    for _ in range(iters):\n"
                "        H *= (W.T @ V) / (W.T @ W @ H + 1e-10)\n"
                "        W *= (V @ H.T) / (W @ H @ H.T + 1e-10)\n"
                "    return W, H\n```"
            ),
        }
        paper_b = {
            "title": "Gene analysis thoughts",
            "abstract": "We looked at genes and found things.",
            "manuscript": "We ran some code. It worked. Trust us.",
        }

        result = run_ranker(
            paper_a=paper_a,
            paper_b=paper_b,
            llm=llm,
            tools_cfg={
                "sandbox": {
                    "enabled": True,
                    "code_timeout_seconds": 60,
                },
            },
            sandbox_client=sandbox_client,
            max_iterations=8,
        )
        assert isinstance(result, RankerResult)
        assert result.score in (0, 1)
        assert isinstance(result.justification, str)
        assert len(result.justification) > 0
