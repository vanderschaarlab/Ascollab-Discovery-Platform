from system.trackers import (
    MetaReviewStatsLog,
    ResearchStatsLog,
    SessionStats,
    SessionStatsRegistry,
    attach_logs,
    get_logs,
)


def _make_research_log(successful_execs: int) -> ResearchStatsLog:
    log = ResearchStatsLog()
    log.code_executions_successful = int(successful_execs)
    return log


def test_registry_is_canonical_on_session_id_for_research_logs():
    registry = SessionStatsRegistry()
    primary_log = _make_research_log(1)
    collab_log = _make_research_log(2)
    registry.add_log(
        [
            SessionStats(
                iteration=2,
                session_id="agent_06_t02",
                session_agent_id="agent_06",
                type="research",
                stats_logs=primary_log,
            ),
            SessionStats(
                iteration=2,
                session_id="agent_06_t02_collab_for_agent_07",
                session_agent_id="agent_06",
                type="research",
                stats_logs=collab_log,
            ),
        ]
    )

    research_logs = registry.get_phase_logs(2, "research")
    assert "agent_06_t02" in research_logs
    assert "agent_06_t02_collab_for_agent_07" in research_logs
    assert research_logs["agent_06_t02"] is primary_log
    assert research_logs["agent_06_t02_collab_for_agent_07"] is collab_log
    assert registry.get_research_log_for_agent(2, "agent_06") is primary_log


def test_get_research_log_for_agent_supports_legacy_agent_key():
    registry = SessionStatsRegistry()
    legacy_log = _make_research_log(3)

    # Backward-compatible read path for snapshots written before canonical session_id keying.
    registry.registry[2]["research"]["agent_08"] = legacy_log

    assert registry.get_research_log_for_agent(2, "agent_08") is legacy_log


def test_registry_accepts_metareview_logs():
    registry = SessionStatsRegistry()
    metareview_log = MetaReviewStatsLog()
    registry.add_log(
        [
            SessionStats(
                iteration=5,
                session_id="metareview_reviewer_01_t05_tour000",
                session_agent_id="reviewer_01",
                type="metareview",
                stats_logs=metareview_log,
            )
        ]
    )

    phase_logs = registry.get_phase_logs(5, "metareview")
    assert "metareview_reviewer_01_t05_tour000" in phase_logs
    assert phase_logs["metareview_reviewer_01_t05_tour000"] is metareview_log


def test_attach_logs_supports_metareview_type():
    class _DummyAgent:
        pass

    marker = _DummyAgent()
    log = attach_logs(marker, type="metareview")
    assert isinstance(log, MetaReviewStatsLog)
    assert get_logs(marker) is log
