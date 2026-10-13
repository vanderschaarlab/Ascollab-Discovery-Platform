"""Research-session limits resolved from the research config."""

from system.main import (
    _resolve_research_admission_limit,
    _resolve_session_create_hard_timeout_seconds,
)


def test_resolve_session_create_hard_timeout_prefers_seconds_key():
    resolved = _resolve_session_create_hard_timeout_seconds(
        {
            "session_create_hard_timeout_seconds": 2400,
            "session_create_hard_timeout": 600,
        }
    )
    assert resolved == 2400.0


def test_resolve_session_create_hard_timeout_supports_legacy_key():
    resolved = _resolve_session_create_hard_timeout_seconds({"session_create_hard_timeout": 1800})
    assert resolved == 1800.0


def test_resolve_research_admission_limit_applies_cap():
    admission, reserve, override = _resolve_research_admission_limit(
        max_sessions=40,
        network_num_agents=16,
        max_concurrent_research_agents=8,
    )
    assert reserve == 5
    assert override == 8
    assert admission == 8


def test_resolve_research_admission_limit_has_safe_minimum():
    admission, reserve, override = _resolve_research_admission_limit(
        max_sessions=4,
        network_num_agents=16,
        max_concurrent_research_agents=None,
    )
    assert reserve == 4
    assert override is None
    assert admission == 1
