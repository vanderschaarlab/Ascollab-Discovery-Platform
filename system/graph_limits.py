"""Step limits of the research and reviewer graphs.

A session's step limit is LangGraph's recursion limit, which counts node runs. Each step (an
action, or an attempt at the final report or review) runs two nodes: ``think``, then ``act`` or
the final check. LangGraph raises ``GraphRecursionError`` when a run reaches the limit, even if
its last node would have ended it, so a session must finish within M - 1 steps; otherwise it
stops with that error after step M.
"""


def recursion_limit_for(max_steps: int) -> int:
    """Return the recursion limit that allows ``max_steps`` steps.

    Args:
        max_steps: Steps allowed in one session (``research.research_max_iters`` or
            ``review.review_max_iterations``).

    Returns:
        The LangGraph recursion limit, two node runs per step.
    """
    return 2 * int(max_steps)
