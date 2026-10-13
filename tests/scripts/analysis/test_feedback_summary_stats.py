"""feedback_summary's research totals: collaborator logs are counted apart from sessions."""

from scripts.analysis import feedback_summary
from system.trackers import ResearchStatsLog


def _log(tokens: int, arxiv: int) -> ResearchStatsLog:
    log = ResearchStatsLog()
    log.total_tokens = tokens
    log.tool_calls = {"arxiv": arxiv}
    return log


def test_collaborator_logs_add_work_but_not_sessions():
    research = {
        "agent_01_t02": _log(100, 1),
        "agent_02_t02": _log(200, 2),
        # agent_02 helped agent_01: its work inside agent_01's session
        "agent_02_t02_collab_for_agent_01": _log(30, 3),
    }
    stats = {"registry": {2: {"research": research}}}

    totals = feedback_summary._summarize_research_stats(stats, 2)

    assert totals["sessions"] == 2
    assert totals["collab_sessions"] == 1
    assert totals["total_tokens"] == 330
    assert totals["tool_calls"] == {"arxiv": 6}
