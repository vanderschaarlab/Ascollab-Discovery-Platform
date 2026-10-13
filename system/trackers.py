import contextvars
import time
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from weakref import WeakKeyDictionary


@dataclass
class ResearchStatsLog:
    """Per-session metrics for research agents (code, tools, tokens, timeline).

    Attributes:
        started_at: Start timestamp.
        ended_at: End timestamp (set on finish).
        duration: Total duration in seconds.
        turns: Number of model turns.
        invalid_actions: Count of invalid actions parsed.
        input_tokens: Token usage for inputs.
        output_tokens: Token usage for outputs.
        total_tokens: Total token usage.
        tool_calls: Counts per tool name.
        tool_inputs: Raw tool input payloads per tool.
        code_executions: Number of code executions attempted.
        code_executions_successful: Successful code executions.
        code_executions_failed: Failed code executions.
        lines_of_code: Aggregate lines of code executed.
        images_created: Number of images produced.
        collab_established: Collaboration sessions established.
        collab_interactions: Interactions with collaborators.
        timeline: Ordered list of event dicts.
    """

    started_at: float = field(default_factory=time.time)
    ended_at: float = 0.0
    duration: float = field(init=False)
    turns: int = 0
    invalid_actions: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    tool_calls: defaultdict = field(default_factory=lambda: defaultdict(int))
    # raw inputs per tool
    tool_inputs: defaultdict = field(default_factory=lambda: defaultdict(list))
    # structured per-tool events (e.g., stage telemetry)
    tool_events: defaultdict = field(default_factory=lambda: defaultdict(list))
    code_executions: int = 0
    code_executions_successful: int = 0
    code_executions_failed: int = 0
    lines_of_code: int = 0
    images_created: int = 0
    collab_established: int = 0
    collab_interactions: int = 0
    timeline: list[dict[str, Any]] = field(default_factory=list)
    collab_logs: dict[str, "ResearchStatsLog"] = field(default_factory=dict)
    compute_requirements: dict[str, Any] = field(default_factory=dict)

    def add_event(self, *, kind: str, **payload: Any) -> None:
        """Append a structured, time-ordered event to this agent's timeline.

        Args:
            kind: Event kind (e.g., "think", "tool_call").
            **payload: Additional JSON-serialisable fields.
        """
        self.timeline.append(
            {
                "ts": time.time(),
                "kind": kind,
                **payload,
            }
        )

    def finish(self):
        self.ended_at = time.time()
        self.duration = self.ended_at - self.started_at


@dataclass
class ReviewStatsLog:
    """Per-session metrics for review agents (tokens, tool calls, timeline)."""

    started_at: float = field(default_factory=time.time)
    ended_at: float = 0.0
    duration: float = field(init=False)
    turns: int = 0
    invalid_actions: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    tool_calls: defaultdict = field(default_factory=lambda: defaultdict(int))
    tool_inputs: defaultdict = field(default_factory=lambda: defaultdict(list))
    tool_events: defaultdict = field(default_factory=lambda: defaultdict(list))
    timeline: list[dict[str, Any]] = field(default_factory=list)
    compute_requirements: dict[str, Any] = field(default_factory=dict)

    def add_event(self, *, kind: str, **payload: Any) -> None:
        """Append a structured event to the reviewer's timeline."""
        self.timeline.append(
            {
                "ts": time.time(),
                "kind": kind,
                **payload,
            }
        )

    def finish(self):
        self.ended_at = time.time()
        self.duration = self.ended_at - self.started_at


@dataclass
class MetaReviewStatsLog:
    """Per-session metrics for metareview tournaments."""

    started_at: float = field(default_factory=time.time)
    ended_at: float = 0.0
    duration: float = field(init=False)
    turns: int = 0
    invalid_actions: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    tool_calls: defaultdict = field(default_factory=lambda: defaultdict(int))
    tool_inputs: defaultdict = field(default_factory=lambda: defaultdict(list))
    tool_events: defaultdict = field(default_factory=lambda: defaultdict(list))
    timeline: list[dict[str, Any]] = field(default_factory=list)
    compute_requirements: dict[str, Any] = field(default_factory=dict)

    def add_event(self, *, kind: str, **payload: Any) -> None:
        """Append a structured event to the metareview timeline."""
        self.timeline.append(
            {
                "ts": time.time(),
                "kind": kind,
                **payload,
            }
        )

    def finish(self):
        self.ended_at = time.time()
        self.duration = self.ended_at - self.started_at


@dataclass
class SessionStats:
    """Wrapper tying a stats log to a session/iteration."""

    iteration: int
    session_id: str
    session_agent_id: str
    type: str
    stats_logs: ResearchStatsLog | ReviewStatsLog | MetaReviewStatsLog
    metadata: dict[str, Any] | None = None


class SessionStatsRegistry:
    def __init__(self):
        self.registry = defaultdict(lambda: defaultdict(dict))

    def add_log(self, log_tuples: list[SessionStats]):
        """Insert SessionStats entries keyed by iteration, type, and session_id."""
        for entry in log_tuples:
            if entry.type == "research":
                assert isinstance(entry.stats_logs, ResearchStatsLog)
            elif entry.type == "review":
                assert isinstance(entry.stats_logs, ReviewStatsLog)
            elif entry.type == "metareview":
                assert isinstance(entry.stats_logs, MetaReviewStatsLog)
            key = str(entry.session_id or "").strip()
            if not key:
                continue
            self.registry[entry.iteration][entry.type][key] = entry.stats_logs

    def get_phase_logs(self, iteration: int, phase: str) -> dict[str, Any]:
        """Return logs for a phase in an iteration, supporting int/str iteration keys."""
        iter_data = self.registry.get(iteration)
        if iter_data is None:
            iter_data = self.registry.get(str(iteration))
        if not isinstance(iter_data, dict):
            return {}
        logs = iter_data.get(phase, {})
        if not isinstance(logs, dict):
            return {}
        return logs

    def get_research_log_for_agent(self, iteration: int, agent_id: str) -> Any:
        """
        Resolve the primary research log for an agent in an iteration.

        Preferred canonical keys are session_id forms:
        - {agent_id}_t{iteration:02d}
        - {agent_id}_t{iteration}

        Falls back to legacy agent-id key for backward compatibility with
        snapshots written before canonical session-id keying.
        """
        research_logs = self.get_phase_logs(iteration, "research")
        if not research_logs:
            return None

        canonical_ids = (
            f"{agent_id}_t{int(iteration):02d}",
            f"{agent_id}_t{int(iteration)}",
        )
        for session_id in canonical_ids:
            if session_id in research_logs:
                return research_logs[session_id]

        return research_logs.get(agent_id)


def _to_defaultdict_int(raw: Any) -> defaultdict:
    out = defaultdict(int)
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            try:
                out[str(key)] = int(value)
            except Exception:
                out[str(key)] = 0
    return out


def _to_defaultdict_list(raw: Any) -> defaultdict:
    out = defaultdict(list)
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            if isinstance(value, list):
                out[str(key)] = list(value)
            elif value is None:
                out[str(key)] = []
            else:
                out[str(key)] = [value]
    return out


def _new_stats_log(
    phase: str, log_type: str | None
) -> ResearchStatsLog | ReviewStatsLog | MetaReviewStatsLog:
    normalized_type = str(log_type or "").strip()
    normalized_phase = str(phase or "").strip().lower()
    if normalized_type == "ResearchStatsLog" or normalized_phase == "research":
        return ResearchStatsLog()
    if normalized_type == "ReviewStatsLog" or normalized_phase == "review":
        return ReviewStatsLog()
    if normalized_type == "MetaReviewStatsLog" or normalized_phase == "metareview":
        return MetaReviewStatsLog()
    return ResearchStatsLog()


def _hydrate_stats_log(
    phase: str,
    payload: Mapping[str, Any],
) -> ResearchStatsLog | ReviewStatsLog | MetaReviewStatsLog:
    log_type = payload.get("log_type") if isinstance(payload, Mapping) else None
    log_obj = _new_stats_log(phase, log_type)

    # Restore persisted scalar/list fields while keeping dataclass defaults for missing keys.
    for key, value in dict(payload or {}).items():
        if key in {"tool_calls", "tool_inputs", "tool_events", "log_type", "session_agent_id"}:
            continue
        try:
            setattr(log_obj, key, value)
        except Exception:
            continue

    if isinstance(payload, Mapping):
        log_obj.tool_calls = _to_defaultdict_int(payload.get("tool_calls"))
        log_obj.tool_inputs = _to_defaultdict_list(payload.get("tool_inputs"))
        log_obj.tool_events = _to_defaultdict_list(payload.get("tool_events"))

    return log_obj


def session_stats_registry_from_snapshot_payload(
    payload: Mapping[str, Any] | None,
) -> SessionStatsRegistry:
    """Hydrate a SessionStatsRegistry from snapshot payload data."""
    registry = SessionStatsRegistry()
    raw = dict(payload or {}) if isinstance(payload, Mapping) else {}
    raw_registry = raw.get("registry", {}) or {}
    if not isinstance(raw_registry, Mapping):
        return registry

    for iteration_key, iteration_data in raw_registry.items():
        if not isinstance(iteration_data, Mapping):
            continue
        try:
            iteration = int(iteration_key)
        except Exception:
            continue

        for phase, phase_sessions in iteration_data.items():
            phase_name = str(phase or "").strip().lower()
            if phase_name not in {"research", "review", "metareview"}:
                continue
            if not isinstance(phase_sessions, Mapping):
                continue

            for session_id, log_payload in phase_sessions.items():
                if not isinstance(log_payload, Mapping):
                    continue
                sid = str(session_id or "").strip()
                if not sid:
                    continue
                registry.registry[iteration][phase_name][sid] = _hydrate_stats_log(
                    phase_name,
                    log_payload,
                )

    return registry


def count_loc(code: str) -> int:
    """Count non-empty, non-comment lines of code."""
    return sum(1 for line in code.splitlines() if line.strip() and not line.strip().startswith("#"))


_metrics_for_agents = WeakKeyDictionary()


def attach_logs(
    agent, type: str = "research"
) -> ResearchStatsLog | ReviewStatsLog | MetaReviewStatsLog:
    """Create and attach a stats log to an agent."""
    if type == "research":
        log = ResearchStatsLog()
    elif type == "review":
        log = ReviewStatsLog()
    elif type == "metareview":
        log = MetaReviewStatsLog()
    else:
        raise ValueError("Unknown type. Use 'research', 'review', or 'metareview'.")
    _metrics_for_agents[agent] = log
    return log


def get_logs(agent) -> ResearchStatsLog | ReviewStatsLog | MetaReviewStatsLog | None:
    """Retrieve the stats log attached to an agent, if any."""
    return _metrics_for_agents.get(agent, None)


# Which agent's log is "current" on this task
_CURRENT_LOG = contextvars.ContextVar("_CURRENT_LOG", default=None)


def record_tool_query(tool_name: str, query: str) -> None:
    """Called by tools to log raw inputs (e.g., arxiv queries) against the current agent."""
    log = _CURRENT_LOG.get()
    if log is None:
        return
    # Ensure the field exists
    if not hasattr(log, "tool_inputs"):
        log.tool_inputs = defaultdict(list)  # type: ignore[attr-defined]
    log.tool_inputs[tool_name].append(str(query))


def record_tool_event(tool_name: str, stage: str, payload: dict[str, Any] | None = None) -> None:
    """Attach structured tool-stage telemetry to the current agent log."""
    log = _CURRENT_LOG.get()
    if log is None:
        return
    if not hasattr(log, "tool_events"):
        log.tool_events = defaultdict(list)  # type: ignore[attr-defined]
    evt = {
        "ts": time.time(),
        "stage": str(stage),
        "payload": payload or {},
    }
    log.tool_events[tool_name].append(evt)
