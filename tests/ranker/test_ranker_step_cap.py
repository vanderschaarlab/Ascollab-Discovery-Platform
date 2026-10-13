"""The judge's step limit holds: a judge that never gives a verdict ends, it does not loop.

``max_iterations`` caps tool use: once reached, the next reply is checked for a verdict instead of
running another tool. A failed verdict check always uses up a step, also with
``max_parse_retries=0``. Each test uses a low recursion limit, so a loop fails fast with
``GraphRecursionError`` instead of making thousands of model calls.
"""

from langchain_core.messages import AIMessage, HumanMessage

from ranker.ranker_agent import build_ranker_agent

ACTION = 'Action: {"tool": "arxiv", "args": {"query": "kidney cancer"}}'
VERDICT = "Verdict:\nWinner: A\nJustification: A is better supported."
RECURSION = {"recursion_limit": 60}


class _ScriptedLLM:
    """Returns the scripted replies in order, then repeats the last one."""

    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.calls = 0

    def invoke(self, prompt_val):
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        msg = AIMessage(content=reply)
        msg.usage_metadata = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
        return msg


def _run(llm, **kwargs):
    tools = {"arxiv": {"object": _EchoTool(), "tool_signature": "arxiv(query)"}}
    agent = build_ranker_agent(llm=llm, system_prompt="sys", tools=tools, **kwargs)
    state = agent.invoke(
        {
            "messages": [HumanMessage(content="Compare the papers.")],
            "step": 0,
            "last_ai_response": "",
            "parsed_result": None,
            "valid_result": None,
        },
        RECURSION,
    )
    return state, agent._ranker_tracker


class _EchoTool:
    def __init__(self):
        self.calls = 0

    def invoke(self, args):
        self.calls += 1
        return f"results for {args}"


def test_a_judge_that_only_calls_tools_stops_at_the_limit():
    llm = _ScriptedLLM(ACTION)
    state, tracker = _run(llm, max_iterations=3, max_parse_retries=1)

    assert state["valid_result"] is False
    assert tracker["tool_calls"] == {"arxiv": 3}
    assert llm.calls <= 3 + 1 + 1  # three tool turns, the capped turn, one nudge


def test_no_parse_retries_still_ends():
    llm = _ScriptedLLM("I am still thinking.")
    state, _ = _run(llm, max_iterations=4, max_parse_retries=0)

    assert state["valid_result"] is False
    assert llm.calls == 4


def test_a_verdict_within_the_limit_is_unchanged():
    llm = _ScriptedLLM(ACTION, ACTION, VERDICT)
    state, tracker = _run(llm, max_iterations=10, max_parse_retries=2)

    assert state["valid_result"] is True
    assert state["parsed_result"]["score"] == 1
    assert tracker["tool_calls"] == {"arxiv": 2}
    assert llm.calls == 3


def test_a_verdict_on_the_capped_turn_is_accepted():
    llm = _ScriptedLLM(ACTION, ACTION, VERDICT)
    state, tracker = _run(llm, max_iterations=2, max_parse_retries=0)

    assert state["valid_result"] is True
    assert tracker["tool_calls"] == {"arxiv": 2}
