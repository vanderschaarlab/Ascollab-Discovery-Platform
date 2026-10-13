"""Data schemas for the LLM-judge ranker."""

from dataclasses import dataclass, field
from typing import Annotated, Any, Optional

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


@dataclass
class RankerTrace:
    """Full observability trace of a single ranker invocation."""

    ranker_id: str
    started_at: float
    ended_at: float
    duration_seconds: float
    paper_a_title: str
    paper_b_title: str
    events: list[dict[str, Any]]
    summary: dict[str, Any]


@dataclass
class RankerResult:
    """Output of a single pairwise ranking invocation.

    Attributes:
        score: Mode-dependent binary outcome, or None when no verdict could be
            parsed (see valid_result).
            - comparison mode: 1 if paper A wins, 0 if paper B wins.
            - hypothesis_similarity mode: 0 if same hypothesis, 1 if different.
        justification: Free-text reasoning for the decision.
        raw_response: Full final LLM output (for debugging/auditing).
        tool_calls: Tool usage stats, e.g. {"arxiv": 2, "sandbox": 1}.
        turns: Number of think-act iterations used.
        input_tokens: Total input tokens consumed.
        output_tokens: Total output tokens consumed.
        trace: Optional observability trace of the ranker invocation.
        valid_result: False when every parse retry was exhausted without a
            parseable verdict. Consumers MUST NOT treat score as a decision
            when this is False.
    """

    score: int | None
    justification: str
    raw_response: str
    tool_calls: dict[str, int] = field(default_factory=dict)
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    trace: Optional["RankerTrace"] = None
    valid_result: bool = True


class RankerState(TypedDict):
    """State container for the ranker agent LangGraph.

    Attributes:
        messages: Conversation history tracked by LangGraph.
        step: Current iteration counter.
        last_ai_response: Raw text of the most recent model output.
        parsed_result: Parsed verdict dict {"score": 0|1, "justification": "..."}.
        valid_result: Whether parsing succeeded.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    step: int
    last_ai_response: str
    parsed_result: dict[str, Any] | None
    valid_result: bool | None
