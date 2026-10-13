"""Run ``ranker.run_sample_and_rank_paired`` with a scripted judge. Invoked by tests/e2e.

    python -m tests.e2e.fake_ranker <hydra overrides...>

The judge prefers paper A whenever the pair's text hash is even, so verdicts are deterministic
and both outcomes occur.
"""

from __future__ import annotations

import hashlib
import os
import sys

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class ScriptedJudge(BaseChatModel):
    """Answers every comparison with a verdict in the format ranker.ranker_utils parses."""

    @property
    def _llm_type(self) -> str:
        return "scripted-judge"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        text = "\n".join(str(m.content) for m in messages)
        winner = "A" if int(hashlib.sha256(text.encode()).hexdigest(), 16) % 2 == 0 else "B"
        verdict = f"Verdict:\n- Winner: {winner}\n- Justification: scripted judge."
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=verdict))])

    def bind_tools(self, tools, **kwargs):
        return self


def main() -> None:
    import ranker.run_sample_and_rank_paired as paired

    paired.build_llm_pool = lambda endpoints, keys=None, **_: {
        k: ScriptedJudge() for k in keys or []
    }
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_dir = os.path.join(repo, "sys_config")
    sys.argv = [
        "ranker.run_sample_and_rank_paired",
        *("--config-path", config_dir, "--config-name", "sample_and_rank_paired"),
        *sys.argv[1:],
    ]
    paired.main()


if __name__ == "__main__":
    main()
