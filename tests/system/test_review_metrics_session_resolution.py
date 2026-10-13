from system.review_metrics import build_review_metrics_for_papers
from system.schema import PaperMetadataPrivate
from system.trackers import ResearchStatsLog, SessionStats, SessionStatsRegistry


def _paper(*, paper_id: str, research_session_id: str | None) -> PaperMetadataPrivate:
    return PaperMetadataPrivate(
        paper_id=paper_id,
        primary_agent_id="agent_01",
        collab_agent_ids=[],
        title="T",
        abstract="A",
        manuscript="We hypothesize and report p < 0.05.",
        citation_count=0,
        publication_t=2,
        cited_paper_ids=[],
        code_script="print('ok')",
        research_session_id=research_session_id,
    )


def _stats(*, lines_of_code: int, code_execs_successful: int) -> ResearchStatsLog:
    log = ResearchStatsLog()
    log.lines_of_code = lines_of_code
    log.code_executions_successful = code_execs_successful
    log.code_executions = code_execs_successful
    return log


def test_metrics_prefers_research_session_id_when_present():
    reg = SessionStatsRegistry()
    reg.add_log(
        [
            SessionStats(
                iteration=2,
                session_id="agent_01_t02",
                session_agent_id="agent_01",
                type="research",
                stats_logs=_stats(lines_of_code=10, code_execs_successful=1),
            ),
            SessionStats(
                iteration=2,
                session_id="agent_01_t02_meta",
                session_agent_id="agent_01",
                type="research",
                stats_logs=_stats(lines_of_code=77, code_execs_successful=3),
            ),
        ]
    )
    paper = _paper(paper_id="paper_t02_agent_01_meta", research_session_id="agent_01_t02_meta")
    metrics = build_review_metrics_for_papers(
        iteration=2,
        candidate_papers=[paper],
        session_stats_registry=reg,
        archive=None,
    )
    support = metrics[paper.paper_id]["support"]
    assert support["lines_of_code"] == 77
    assert support["successful_code_executions"] == 3


def test_metrics_falls_back_to_agent_lookup_without_session_id():
    reg = SessionStatsRegistry()
    reg.add_log(
        [
            SessionStats(
                iteration=2,
                session_id="agent_01_t02",
                session_agent_id="agent_01",
                type="research",
                stats_logs=_stats(lines_of_code=11, code_execs_successful=2),
            ),
            SessionStats(
                iteration=2,
                session_id="agent_01_t02_meta",
                session_agent_id="agent_01",
                type="research",
                stats_logs=_stats(lines_of_code=66, code_execs_successful=5),
            ),
        ]
    )
    paper = _paper(paper_id="paper_t02_agent_01", research_session_id=None)
    metrics = build_review_metrics_for_papers(
        iteration=2,
        candidate_papers=[paper],
        session_stats_registry=reg,
        archive=None,
    )
    support = metrics[paper.paper_id]["support"]
    assert support["lines_of_code"] == 11
    assert support["successful_code_executions"] == 2
