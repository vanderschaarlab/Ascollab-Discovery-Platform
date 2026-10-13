import json

import pytest
from omegaconf import OmegaConf

from stores.private_paper_archive import PrivatePaperArchive
from system.metacognition_feedback import (
    build_metacognition_feedback_for_agents,
    collect_feedback_source_records,
    format_metrics_only_feedback,
    group_records_by_agent,
    load_snapshot_metrics,
    make_feedback_synthesizer_user_message,
    make_metacognition_feedback_metadata,
    merge_metacognition_feedback_learning_state,
    normalize_metacognition_feedback_cfg,
    parse_feedback_synthesizer_output,
    resolve_feedback_source_iteration,
)
from system.review_metrics import review_metric_catalog
from system.schema import AgentProfile, PaperMetadataPrivate, PaperMetaReview


def _paper(
    *,
    paper_id="paper_t01_agent_01",
    agent_id="agent_01",
    publication_t=1,
    status="accepted",
    collabs=None,
    metrics=None,
):
    default_metrics = {
        "support": {"successful_code_executions": 2, "lines_of_code": 12},
        "soundness": {"internal_consistency_score": 0.4},
        "significance": {"effect_size_strength": 0.2},
        "originality": {"embedding_novelty_percentile": 72.0},
    }
    return PaperMetadataPrivate(
        paper_id=paper_id,
        primary_agent_id=agent_id,
        collab_agent_ids=list(collabs or []),
        title="A paper",
        abstract="An abstract",
        manuscript="A manuscript with evidence.",
        citation_count=0,
        publication_t=publication_t,
        cited_paper_ids=[],
        metareview=PaperMetaReview(
            paper_id=paper_id,
            meta_review_text="Good support, weak novelty.",
            overall_score=3.5,
            rank="1/1",
            justification="solid",
            decision="accept",
        ),
        review_metrics=metrics if metrics is not None else default_metrics,
        status=status,
        publication_status=status,
    )


def _agent(agent_id="agent_01", learning_state=None):
    return AgentProfile(
        agent_id=agent_id,
        behavior="careful",
        expertise="omics",
        expertise_topics=["omics"],
        citation_count=0,
        num_accepted_papers=0,
        learning_state=learning_state,
    )


def test_missing_config_normalizes_to_disabled():
    cfg = normalize_metacognition_feedback_cfg(None)
    assert cfg["enabled"] is False
    assert cfg["mode"] == "none"


def test_base_config_exposes_disabled_metacognition_defaults():
    cfg = OmegaConf.load("sys_config/ascollab.yaml")
    assert cfg.research.metacognition_feedback.enabled is False
    assert cfg.research.metacognition_feedback.mode == "none"


def test_string_booleans_normalize_without_enabling_disabled_runs():
    cfg = normalize_metacognition_feedback_cfg({"enabled": "false", "mode": "metrics_only"})
    assert cfg["enabled"] is False
    assert cfg["mode"] == "none"


def test_source_iteration_restored_resolves_only_with_resume_context():
    cfg = normalize_metacognition_feedback_cfg(
        {"enabled": True, "mode": "metrics_only", "source_iteration": "restored"}
    )
    assert resolve_feedback_source_iteration(10, 9, cfg) == 9
    assert resolve_feedback_source_iteration(10, None, cfg) is None


def test_collect_source_records_filters_iteration_status_and_timeout():
    archive = PrivatePaperArchive(
        [
            _paper(paper_id="paper_t01_agent_01", publication_t=1, status="accepted"),
            _paper(paper_id="paper_t02_agent_01", publication_t=2, status="accepted"),
            _paper(
                paper_id="paper_t01_agent_02",
                agent_id="agent_02",
                publication_t=1,
                status="rejected",
            ),
            _paper(
                paper_id="paper_t01_agent_03_timeout",
                agent_id="agent_03",
                publication_t=1,
                status="timeout",
            ),
        ]
    )
    archive.add_reviews_for_iteration(
        1,
        {
            "paper_t01_agent_01": [
                {
                    "reviewer_id": "r1",
                    "review_text": "x" * 2000,
                    "scores": {"Overall Recommendation": 4},
                }
            ]
        },
    )

    records = collect_feedback_source_records(
        archive,
        1,
        include_rejected=False,
        include_timeout=False,
        max_chars_per_review=20,
    )

    assert [r["paper_id"] for r in records] == ["paper_t01_agent_01"]
    assert records[0]["reviews"][0]["review_text"].endswith("...")
    assert len(records[0]["reviews"][0]["review_text"]) <= 23


def test_group_records_by_agent_can_include_collaborators():
    records = [
        {
            "paper_id": "paper_a",
            "primary_agent_id": "agent_01",
            "collab_agent_ids": ["agent_02"],
        }
    ]
    primary_only = group_records_by_agent(records, include_collaborator_papers=False)
    with_collabs = group_records_by_agent(records, include_collaborator_papers=True)
    assert set(primary_only) == {"agent_01"}
    assert set(with_collabs) == {"agent_01", "agent_02"}


def test_load_snapshot_metrics_from_jsonl(tmp_path):
    iter_dir = tmp_path / "iteration_1"
    iter_dir.mkdir()
    (iter_dir / "metrics.jsonl").write_text(
        json.dumps({"paper_id": "paper_a", "metrics": {"support": {"lines_of_code": 3}}}) + "\n",
        encoding="utf-8",
    )
    assert load_snapshot_metrics(iter_dir) == {"paper_a": {"support": {"lines_of_code": 3}}}


def test_metrics_only_feedback_is_deterministic_and_no_llm_required():
    records = [
        {
            "paper_id": "paper_a",
            "publication_t": 1,
            "metrics": {"support": {"successful_code_executions": 0}},
        }
    ]
    cfg = normalize_metacognition_feedback_cfg({"enabled": True, "mode": "metrics_only"})
    one = format_metrics_only_feedback("agent_01", records, review_metric_catalog(), cfg)
    two = format_metrics_only_feedback("agent_01", records, review_metric_catalog(), cfg)
    assert one == two
    assert one["raw_metrics"]["paper_a"]["support"]["successful_code_executions"] == 0
    assert "metric_definitions" in one


def test_metrics_allowlist_filters_scores_and_definitions():
    records = [
        {
            "paper_id": "paper_a",
            "publication_t": 1,
            "metrics": {
                "support": {"successful_code_executions": 1, "lines_of_code": 99},
                "soundness": {"internal_consistency_score": 0.4},
            },
        }
    ]
    cfg = normalize_metacognition_feedback_cfg(
        {
            "enabled": True,
            "mode": "metrics_only",
            "metrics_allowlist": ["support.successful_code_executions"],
        }
    )
    payload = format_metrics_only_feedback("agent_01", records, review_metric_catalog(), cfg)
    assert payload["raw_metrics"] == {"paper_a": {"support": {"successful_code_executions": 1}}}
    assert payload["metric_definitions"]["support"] == {
        "successful_code_executions": "Successful analysis code executions."
    }
    assert payload["metric_definitions"]["soundness"] == {}

    prompt = make_feedback_synthesizer_user_message(
        "agent_01",
        records,
        include_metrics=True,
        cfg=cfg,
    )
    assert "successful_code_executions" in prompt
    assert "lines_of_code" not in prompt
    assert "internal_consistency_score" not in prompt


def test_axes_filter_metric_categories_shown_to_feedback():
    records = [
        {
            "paper_id": "paper_a",
            "publication_t": 1,
            "metrics": {
                "support": {"successful_code_executions": 1},
                "soundness": {"internal_consistency_score": 0.4},
            },
        }
    ]
    cfg = normalize_metacognition_feedback_cfg(
        {"enabled": True, "mode": "metrics_only", "axes": ["support"]}
    )
    payload = format_metrics_only_feedback("agent_01", records, review_metric_catalog(), cfg)
    assert set(payload["axes"]) == {"support"}
    assert payload["raw_metrics"] == {"paper_a": {"support": {"successful_code_executions": 1}}}
    assert set(payload["metric_definitions"]) == {"support"}


def test_build_disabled_returns_without_touching_raising_llm():
    class RaisingLLM:
        def invoke(self, _prompt):
            raise AssertionError("LLM should not be called")

    archive = PrivatePaperArchive([_paper()])
    result = build_metacognition_feedback_for_agents(
        agents=[_agent()],
        private_archive=archive,
        iteration=2,
        restored_iteration=None,
        cfg={"enabled": False, "mode": "meta_review_only"},
        llm=RaisingLLM(),
    )
    assert result == {}


def test_parse_feedback_synthesizer_output_validates_required_shape():
    text = json.dumps(
        {
            "summary": "Improve evidence.",
            "axes": {
                axis: {"strengths": [], "weaknesses": [], "actions": ["do x"]}
                for axis in ("originality", "significance", "soundness", "support")
            },
        }
    )
    parsed = parse_feedback_synthesizer_output(text)
    assert parsed["summary"] == "Improve evidence."

    with pytest.raises(ValueError):
        parse_feedback_synthesizer_output('{"summary": "missing axes"}')


def test_learning_state_merge_preserves_existing_keys():
    agent = _agent(
        learning_state={
            "policy_learning": {"active": True},
            "other_state": {"x": 1},
        }
    )
    merged = merge_metacognition_feedback_learning_state(agent, {"summary": "s"})
    assert merged["policy_learning"] == {"active": True}
    assert merged["other_state"] == {"x": 1}
    assert merged["metacognition_feedback"]["active"]["summary"] == "s"


def test_learning_state_merge_does_not_persist_feedback_llm_usage():
    agent = _agent()
    merged = merge_metacognition_feedback_learning_state(
        agent,
        {"summary": "s", "_llm_usage": {"total_tokens": 12}},
    )
    assert "_llm_usage" not in merged["metacognition_feedback"]["active"]


def test_build_metrics_only_uses_snapshot_metrics_fallback(tmp_path):
    archive = PrivatePaperArchive([_paper(metrics={})])
    iter_dir = tmp_path / "iteration_1"
    iter_dir.mkdir()
    (iter_dir / "metrics.jsonl").write_text(
        json.dumps(
            {
                "paper_id": "paper_t01_agent_01",
                "metrics": {"support": {"successful_code_executions": 4}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    feedback = build_metacognition_feedback_for_agents(
        agents=[_agent()],
        private_archive=archive,
        iteration=2,
        restored_iteration=None,
        cfg={"enabled": True, "mode": "metrics_only", "source_iteration": "previous"},
        source_snapshot_dir=iter_dir,
    )
    assert (
        feedback["agent_01"]["raw_metrics"]["paper_t01_agent_01"]["support"][
            "successful_code_executions"
        ]
        == 4
    )


def test_strict_metrics_only_requires_some_metric_source():
    archive = PrivatePaperArchive([_paper(metrics={})])

    with pytest.raises(ValueError, match="no internal review metrics"):
        build_metacognition_feedback_for_agents(
            agents=[_agent()],
            private_archive=archive,
            iteration=2,
            restored_iteration=None,
            cfg={"enabled": True, "mode": "metrics_only", "source_iteration": "previous"},
        )


def test_strict_source_requires_records_for_each_active_agent():
    archive = PrivatePaperArchive([_paper(agent_id="agent_01")])

    with pytest.raises(ValueError, match="agent_02"):
        build_metacognition_feedback_for_agents(
            agents=[_agent("agent_01"), _agent("agent_02")],
            private_archive=archive,
            iteration=2,
            restored_iteration=None,
            cfg={"enabled": True, "mode": "metrics_only", "source_iteration": "previous"},
        )


def test_non_strict_missing_metrics_are_reported_as_payload_warnings():
    archive = PrivatePaperArchive([_paper(metrics={})])

    feedback = build_metacognition_feedback_for_agents(
        agents=[_agent()],
        private_archive=archive,
        iteration=2,
        restored_iteration=None,
        cfg={
            "enabled": True,
            "mode": "metrics_only",
            "source_iteration": "previous",
            "strict_source_required": False,
            "best_effort": True,
        },
    )

    assert feedback["agent_01"]["raw_metrics"]["paper_t01_agent_01"] == {}
    assert "no internal review metrics" in " ".join(feedback["agent_01"]["_warnings"])


def test_best_effort_requires_non_strict_source_mode():
    archive = PrivatePaperArchive([_paper(metrics={})])

    with pytest.raises(ValueError, match="no internal review metrics"):
        build_metacognition_feedback_for_agents(
            agents=[_agent()],
            private_archive=archive,
            iteration=2,
            restored_iteration=None,
            cfg={
                "enabled": True,
                "mode": "metrics_only",
                "source_iteration": "previous",
                "best_effort": True,
            },
        )


class _ScriptedFeedbackLLM:
    """Returns the given replies in order (the last one repeats)."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0

    def invoke(self, messages, **kwargs):
        from langchain_core.messages import AIMessage

        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        usage = {
            "input_tokens": 100,
            "output_tokens": 10,
            "total_tokens": 110,
            "output_token_details": {"reasoning": 4},
        }
        return AIMessage(content=reply, usage_metadata=usage)


_VALID_FEEDBACK = json.dumps(
    {
        "summary": "Strengthen controls.",
        "axes": {
            axis: {"strengths": ["a"], "weaknesses": ["b"], "actions": ["c"]}
            for axis in ("originality", "significance", "soundness", "support")
        },
    }
)


def _build_with(llm, skipped_agents=None, **cfg_overrides):
    cfg = {
        "enabled": True,
        "mode": "meta_review_only",
        "source_iteration": "previous",
        **cfg_overrides,
    }
    return build_metacognition_feedback_for_agents(
        agents=[_agent()],
        private_archive=PrivatePaperArchive([_paper()]),
        iteration=2,
        restored_iteration=None,
        cfg=cfg,
        llm=llm,
        **({} if skipped_agents is None else {"skipped_agents": skipped_agents}),
    )


def test_unparseable_synthesizer_reply_is_retried_once():
    llm = _ScriptedFeedbackLLM("OK", _VALID_FEEDBACK)

    feedback = _build_with(llm)

    assert llm.calls == 2
    assert feedback["agent_01"]["summary"] == "Strengthen controls."


def test_best_effort_skips_an_agent_whose_reply_never_parses(caplog):
    llm = _ScriptedFeedbackLLM("OK")

    with caplog.at_level("WARNING", logger="system.metacognition_feedback"):
        feedback = _build_with(llm, strict_source_required=False, best_effort=True)

    assert llm.calls == 2
    assert feedback == {}
    assert "no feedback for agent_id=agent_01" in caplog.text


def test_strict_run_stops_when_the_reply_never_parses():
    llm = _ScriptedFeedbackLLM("OK")

    with pytest.raises(ValueError, match="after 2 attempts"):
        _build_with(llm)
    assert llm.calls == 2


def test_valid_reply_is_used_without_retry():
    llm = _ScriptedFeedbackLLM(_VALID_FEEDBACK)

    _build_with(llm)

    assert llm.calls == 1


def test_token_usage_of_a_retried_reply_counts_both_attempts():
    llm = _ScriptedFeedbackLLM("OK", _VALID_FEEDBACK)

    feedback = _build_with(llm)
    metadata = make_metacognition_feedback_metadata(
        iteration=2, source_iteration=1, mode="meta_review_only", feedback_by_agent=feedback
    )

    assert metadata["llm_usage_by_agent"]["agent_01"] == {
        "input_tokens": 200,
        "output_tokens": 20,
        "total_tokens": 220,
        "output_token_details": {"reasoning": 8},
    }


def test_best_effort_skip_is_recorded_and_replaces_previous_feedback():
    llm = _ScriptedFeedbackLLM("OK")
    skipped: dict = {}

    feedback = _build_with(llm, skipped, strict_source_required=False, best_effort=True)
    metadata = make_metacognition_feedback_metadata(
        iteration=2,
        source_iteration=1,
        mode="meta_review_only",
        feedback_by_agent=feedback,
        skipped_agents=skipped,
    )
    previous = {"metacognition_feedback": {"active": {"summary": "Old."}, "history": []}}
    learning_state = merge_metacognition_feedback_learning_state(
        _agent(learning_state=previous), skipped["agent_01"]
    )

    assert feedback == {}
    assert metadata["feedback_by_agent"] == {}
    assert "unparseable twice" in metadata["warnings_by_agent"]["agent_01"][0]
    assert metadata["llm_usage_by_agent"]["agent_01"]["total_tokens"] == 220
    active = learning_state["metacognition_feedback"]["active"]
    assert "summary" not in active
    assert active["source_iteration"] == 1
    assert "unparseable twice" in active["skipped"]
    assert "_llm_usage" not in active
