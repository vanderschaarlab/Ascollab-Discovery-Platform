"""Scripted stand-ins for the LLMs, embedder and sandbox, used to run the pipeline end to end.

The fake chat model recognises each kind of call by its prompt and answers in the format that
call's parser expects. Every call is appended to ``CAPTURE`` so tests can record what prompts the
pipeline actually sent.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
from typing import Any

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

CAPTURE: list[dict[str, Any]] = []

BEHAVIOR_TEXT = (
    "Methodical and collaborative; extends existing ideas with careful, well-controlled analyses "
    "and engages constructively with peers."
)
EXPERTISE_TEXT = (
    "<expertise>Survival analysis, differential expression and pathway enrichment for cancer "
    "genomics cohorts.</expertise>"
)
CODE = "import pandas as pd\nprint('rows', 42)\n"

FINAL_REPORT = """Final Report:
# Title
Synthetic association between a pathway score and survival

# Summary
A synthetic report produced by the end-to-end test harness.

# Research Question
Does a pathway activity score associate with overall survival in this cohort?

# Hypothesis and Key Findings
We hypothesized that higher pathway activity predicts shorter survival.
Key findings:
1. The score separated survival groups in a Cox model [Code: M1].

# Results and Analysis
The hazard ratio was above one in the synthetic analysis [Code: M1].

# Methods
Data sources: the provided datasets. Analysis: Cox proportional hazards model.

# Formal Methods
## M1, M2, ... Method Runs
- M1
  - Data slice/input: all samples.
  - Procedure/model/test: Cox model.
  - Used for claims: C1.

# References
1. [Arxiv: Synthetic reference | Doe | 2024]
"""

REVIEW_TEXT = """Final Review:
# Review Text
1. Summary
The report tests a pathway score against survival.
2. Motivation
Reasonable.
3. Claims and Evidence
Supported by one analysis.
4. Soundness of Methodology
Standard Cox model.
5. Relation to Prior Knowledge
Briefly discussed.
6. Novelty and Significance
Moderate.
7. Other Comments
None.

# Quantitative Ratings
- Support: 3
- Soundness: 3
- Significance: 3
- Originality: 2
- Overall Recommendation: 3
Overall=3 because the evidence is a single supporting analysis.
"""


def paper_score(paper_id: str) -> float:
    """Meta-review score of a paper: a stable hash of its ID, in [0.05, 0.95].

    Round 1 runs its meta-review tournaments concurrently, in executor threads, so a score that
    depended on call order would make which paper is accepted (and so the fixtures) vary between
    runs. sha256, unlike Python's salted ``hash()``, gives the same score in every process. The
    e2e paper IDs all get different scores, so the best-scored papers of a round are accepted
    whatever the tournament order, and the others are rejected.
    """
    digest = hashlib.sha256(paper_id.encode("utf-8")).digest()
    return round(0.05 + 0.9 * int.from_bytes(digest[:4], "big") / 0xFFFFFFFF, 4)


def _text(messages: list[BaseMessage]) -> tuple[str, str]:
    system = "\n".join(str(m.content) for m in messages if m.type == "system")
    rest = "\n".join(str(m.content) for m in messages if m.type != "system")
    return system, rest


def _kind(system: str, rest: str) -> str:
    if "generate a single epistemic researcher profile" in system:
        return "persona_behavior"
    if "You are to generate a domain expertise description" in system:
        return "persona_expertise"
    if "Final Review:" in system:
        return "reviewer"
    if "Final Feedback:" in system:
        return "collaborator"
    if "synthesize previous-round scientific review evidence" in system:
        return "feedback_synthesizer"
    if (
        "Paper ID" in system + rest
        and "Decision" in system + rest
        and "Final Report:" not in system
    ):
        return "metareviewer"
    if "Final Report:" in system:
        return "research"
    return "other"


SYNTHESIZER_JSON = json.dumps(
    {
        "summary": "Synthetic feedback: strengthen controls and robustness checks.",
        "axes": {
            axis: {
                "strengths": [f"clear {axis}"],
                "weaknesses": [f"limited {axis} evidence"],
                "actions": [f"add one {axis} check"],
            }
            for axis in ("originality", "significance", "soundness", "support")
        },
    }
)


def _action(tool: str, args: dict[str, Any]) -> str:
    return f"Thought: next step.\nAction: {json.dumps({'tool': tool, 'args': args})}"


def _research_step(system: str, rest: str) -> str:
    """Scripted research session: collaborate (when offered), run code once, then report.

    With E2E_RESEARCH_FIRST_ACTION set (an action's JSON), every session starts with that action.
    """
    first = os.environ.get("E2E_RESEARCH_FIRST_ACTION")
    if first and first not in rest:
        return f"Thought: first, look up prior work.\nAction: {first}"
    did = {
        t: f'"tool": "{t}"' in rest
        for t in ("establish_collaboration", "communicate_with_collaborator", "execute_code")
    }
    assigned = re.search(r"You must collaborate with (\S+?) before executing code", system)
    # Agent ids start at agent_01. The agent's own id is not in its prompt, so agent_01 tries to
    # collaborate with itself, which exercises that path too.
    collaborator = assigned.group(1) if assigned else "agent_01"
    if assigned or "establish_collaboration" in system:
        if not did["establish_collaboration"]:
            return _action("establish_collaboration", {"agent_id": collaborator})
        if not did["communicate_with_collaborator"]:
            return _action(
                "communicate_with_collaborator",
                {"agent_id": collaborator, "message": "Could you sanity-check my analysis plan?"},
            )
    if not did["execute_code"]:
        return _action("execute_code", {"code": CODE})
    return FINAL_REPORT


def respond(messages: list[BaseMessage]) -> tuple[str, str]:
    system, rest = _text(messages)
    kind = _kind(system, rest)
    if kind == "persona_behavior":
        return kind, BEHAVIOR_TEXT
    if kind == "persona_expertise":
        return kind, EXPERTISE_TEXT
    if kind == "reviewer":
        return kind, REVIEW_TEXT
    if kind == "collaborator":
        return kind, "Final Feedback: Add a sensitivity analysis and report effect sizes."
    if kind == "feedback_synthesizer":
        # E2E_SYNTHESIZER_REPLY lets a test make every feedback-synthesizer reply unparseable.
        return kind, os.environ.get("E2E_SYNTHESIZER_REPLY", SYNTHESIZER_JSON)
    if kind == "metareviewer":
        ids = list(dict.fromkeys(re.findall(r"Paper ID:\s*([A-Za-z0-9_\-\.]+)", rest)))
        blocks = []
        for rank, pid in enumerate(ids, start=1):
            blocks.append(
                f"Paper ID: {pid}\n1. Summary\nSynthetic.\n2. Assessment\nSynthetic.\n"
                f"3. Decision\n- Score: {paper_score(pid)}\n- Rank: {rank}\n"
                f"- Justification: synthetic ranking.\n"
            )
        return kind, "\n".join(blocks) or "Paper ID: none\n3. Decision\n- Score: 0.5\n- Rank: 1\n"
    if kind == "research":
        return kind, _research_step(system, rest)
    return kind, "OK"


class ScriptedChatModel(BaseChatModel):
    """Chat model that answers each pipeline call in the format its parser expects."""

    backend: str = "fake"

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        kind, text = respond(list(messages))
        system, _ = _text(list(messages))
        human = next((str(m.content) for m in messages if m.type == "human"), "")
        CAPTURE.append({"backend": self.backend, "kind": kind, "system": system, "human": human})
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def bind_tools(self, tools, **kwargs):  # text-based agents do not need real tool binding
        return self


def fake_embedder() -> DeterministicFakeEmbedding:
    return DeterministicFakeEmbedding(size=64)


class FakeSandboxClient:
    """In-memory replacement for env_setup.sandbox_utils.SandboxClient."""

    _ids = itertools.count(1)

    def __init__(self, *args, **kwargs):
        self.sessions: set[str] = set()

    def get_server_stats(self, path: str = "/base/stats"):
        return {"max_sessions": 64, "active_sessions": len(self.sessions)}

    def create_session(self, *args, **kwargs) -> str:
        sid = f"fake-session-{next(self._ids)}"
        self.sessions.add(sid)
        return sid

    def list_sessions(self, path: str = "/base/sessions"):
        return [{"session_id": s} for s in sorted(self.sessions)]

    def run(self, session_id: str, code: str, *args, **kwargs) -> dict[str, Any]:
        return {
            "status": "success",
            "stdout": "rows 42\n",
            "stderr": "",
            "result": None,
            "images": [],
        }

    def restart(self, session_id: str, *args, **kwargs):
        return {"status": "ok"}

    def shutdown(self, session_id: str, *args, **kwargs):
        self.sessions.discard(session_id)
        return {"status": "ok"}

    def shutdown_all_sessions(self, *args, **kwargs):
        self.sessions.clear()
        return {"status": "ok"}

    def reap_stuck_sessions(self, *args, **kwargs):
        return {"reaped": 0}

    def _reap_stale_sessions(self, *args, **kwargs):
        return {"reaped": 0}

    def sticky_session(self, session_id: str | None = None, **create_kwargs):
        from env_setup.sandbox_utils import StickySession

        return StickySession(self, session_id=session_id, create_kwargs=create_kwargs)
