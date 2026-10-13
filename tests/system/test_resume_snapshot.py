import json
import pickle
from pathlib import Path

from omegaconf import OmegaConf

from system.resume_snapshot import (
    archive_incomplete_tail,
    find_last_complete_iteration,
    is_complete_iteration,
    load_full_state_from_iteration,
    load_iteration_exp_config,
    resolve_resume_source,
)
from system.trackers import (
    MetaReviewStatsLog,
    ResearchStatsLog,
    ReviewStatsLog,
    session_stats_registry_from_snapshot_payload,
)


def _paper_dict(*, paper_id: str, primary_agent_id: str, publication_t: int) -> dict:
    return {
        "paper_id": paper_id,
        "primary_agent_id": primary_agent_id,
        "collab_agent_ids": [],
        "title": f"title_{paper_id}",
        "abstract": "abstract",
        "manuscript": "manuscript",
        "citation_count": 0,
        "publication_t": int(publication_t),
        "cited_paper_ids": [],
    }


def _write_complete_iteration(
    run_dir: Path,
    *,
    iteration: int,
    run_name: str = "run_a",
    agent_count: int = 1,
    public_papers: list[dict] | None = None,
    private_papers: list[dict] | None = None,
    reviews_by_paper: dict | None = None,
    review_assignments: dict | None = None,
    notepad_entries: dict | None = None,
    session_stats_payload: dict | None = None,
) -> Path:
    iter_dir = run_dir / f"iteration_{iteration}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    (iter_dir / "snapshot_metadata.json").write_text(
        json.dumps({"iteration": int(iteration), "run_name": str(run_name)}),
        encoding="utf-8",
    )

    agents = []
    for idx in range(agent_count):
        aid = f"agent_{idx + 1:02d}"
        agents.append(
            {
                "agent_id": aid,
                "behavior": "curious",
                "expertise": "omics",
                "expertise_topics": ["omics"],
                "citation_count": 1,
                "num_accepted_papers": 0,
            }
        )

    with (iter_dir / "agent_registry_data.pkl").open("wb") as f:
        pickle.dump(
            {
                "agents": agents,
                "with_citation_boost": True,
                "w_sim": 0.7,
                "w_cit": 0.3,
                "max_k": 9,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    with (iter_dir / "archive_data.pkl").open("wb") as f:
        pickle.dump(
            {
                "papers": list(public_papers or []),
                "with_citation_boost": True,
                "with_recency_boost": True,
                "w_citation": 0.25,
                "w_recency": 0.15,
                "max_k": 11,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    with (iter_dir / "private_archive_data.pkl").open("wb") as f:
        pickle.dump(
            {
                "papers": list(private_papers or []),
                "reviews_by_paper": dict(reviews_by_paper or {}),
                "review_assignments": dict(review_assignments or {}),
                "with_citation_boost": False,
                "with_recency_boost": True,
                "w_citation": 0.05,
                "w_recency": 0.2,
                "max_k": 13,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    with (iter_dir / "notepad_data.pkl").open("wb") as f:
        pickle.dump(
            {
                "entries_by_agent": dict(notepad_entries or {}),
                "notepad_file_paths": {},
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    with (iter_dir / "session_stats_data.pkl").open("wb") as f:
        pickle.dump(
            dict(session_stats_payload or {"registry": {}}), f, protocol=pickle.HIGHEST_PROTOCOL
        )

    OmegaConf.save(
        config=OmegaConf.create(
            {
                "run_name": str(run_name),
                "network": {
                    "num_agents": int(agent_count),
                    "iterations": int(iteration + 2),
                    "setting": "network_hetero",
                },
                "bootstrap": {"enabled": False},
                "resume": {"enabled": False},
            }
        ),
        f=str(iter_dir / "exp_config.yaml"),
    )

    return iter_dir


def _write_incomplete_iteration(run_dir: Path, *, iteration: int) -> Path:
    iter_dir = run_dir / f"iteration_{iteration}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    # Missing snapshot_metadata/core files by design.
    return iter_dir


def test_find_last_complete_iteration_skips_incomplete_tail(tmp_path):
    run_dir = tmp_path / "snapshots" / "run_tail"
    _write_complete_iteration(run_dir, iteration=1)
    _write_complete_iteration(run_dir, iteration=2)
    _write_incomplete_iteration(run_dir, iteration=3)

    restored_iteration, restored_dir = find_last_complete_iteration(run_dir)
    assert restored_iteration == 2
    assert restored_dir.name == "iteration_2"


def test_metacognition_artifact_not_required_for_complete_iteration(tmp_path):
    run_dir = tmp_path / "snapshots" / "run_no_feedback_artifact"
    iter_dir = _write_complete_iteration(run_dir, iteration=1)

    complete, reason = is_complete_iteration(iter_dir)

    assert complete is True
    assert reason == "ok"
    assert not (iter_dir / "metacognition_feedback.json").exists()


def test_iteration_path_caps_search_range(tmp_path):
    run_dir = tmp_path / "snapshots" / "run_cap"
    iter_1 = _write_complete_iteration(run_dir, iteration=1)
    _write_incomplete_iteration(run_dir, iteration=2)
    _write_complete_iteration(run_dir, iteration=3)

    resolved_run_dir, cap = resolve_resume_source(
        iter_1.parent / "iteration_2", project_root=tmp_path
    )
    assert resolved_run_dir == run_dir
    assert cap == 2

    capped_iteration, capped_dir = find_last_complete_iteration(run_dir, max_iteration=cap)
    assert capped_iteration == 1
    assert capped_dir.name == "iteration_1"

    uncapped_iteration, uncapped_dir = find_last_complete_iteration(run_dir)
    assert uncapped_iteration == 3
    assert uncapped_dir.name == "iteration_3"


def test_archive_incomplete_tail_renames_dirs(tmp_path):
    run_dir = tmp_path / "snapshots" / "run_archive_tail"
    _write_complete_iteration(run_dir, iteration=1)
    _write_complete_iteration(run_dir, iteration=2)
    _write_incomplete_iteration(run_dir, iteration=3)
    _write_incomplete_iteration(run_dir, iteration=4)
    _write_complete_iteration(run_dir, iteration=5)

    archived = archive_incomplete_tail(run_dir, restored_iteration=2)
    archived_names = {p.name for p in archived}
    assert len(archived) == 2
    assert all(
        name.startswith(("iteration_3__incomplete_", "iteration_4__incomplete_"))
        for name in archived_names
    )
    assert not (run_dir / "iteration_3").exists()
    assert not (run_dir / "iteration_4").exists()
    assert (run_dir / "iteration_5").exists()


def test_load_full_state_restores_agents_archives_reviews_notepad(tmp_path):
    run_dir = tmp_path / "snapshots" / "run_restore"
    notepad_entries = {
        "agent_01": [
            {
                "idea_id": "idea_1",
                "idea_text": "test idea",
                "iteration": 4,
            }
        ]
    }
    session_stats_payload = {
        "registry": {
            4: {
                "research": {
                    "agent_01_t04": {
                        "log_type": "ResearchStatsLog",
                        "turns": 3,
                        "tool_calls": {"arxiv": 2},
                    }
                }
            }
        }
    }
    iteration_dir = _write_complete_iteration(
        run_dir,
        iteration=4,
        run_name="run_restore",
        agent_count=1,
        public_papers=[
            _paper_dict(paper_id="paper_pub", primary_agent_id="agent_01", publication_t=4)
        ],
        private_papers=[
            _paper_dict(paper_id="paper_priv", primary_agent_id="agent_01", publication_t=4)
        ],
        reviews_by_paper={
            "paper_priv": [{"reviewer_id": "rev_1", "review_text": "ok", "scores": {"s": 4.0}}]
        },
        review_assignments={"iteration_4_random_b": {"assignments": {"paper_priv": ["rev_1"]}}},
        notepad_entries=notepad_entries,
        session_stats_payload=session_stats_payload,
    )

    restored_cfg = load_iteration_exp_config(iteration_dir)
    assert int(restored_cfg.network.num_agents) == 1
    assert int(restored_cfg.network.iterations) == 6

    state = load_full_state_from_iteration(iteration_dir)
    assert state.run_name == "run_restore"
    assert state.restored_iteration == 4
    assert len(state.agents) == 1
    assert state.agents[0].agent_id == "agent_01"
    assert len(state.public_papers) == 1
    assert state.public_papers[0].paper_id == "paper_pub"
    assert len(state.private_papers) == 1
    assert state.private_papers[0].paper_id == "paper_priv"
    assert "paper_priv" in state.reviews_by_paper
    assert state.review_assignments["iteration_4_random_b"]["assignments"]["paper_priv"] == [
        "rev_1"
    ]
    assert state.notepad_entries == notepad_entries
    assert state.session_stats_payload == session_stats_payload
    assert state.registry_settings["with_citation_boost"] is True
    assert float(state.registry_settings["w_sim"]) == 0.7
    assert int(state.registry_settings["max_k"]) == 9
    assert state.archive_settings["with_recency_boost"] is True
    assert int(state.archive_settings["max_k"]) == 11
    assert state.private_archive_settings["with_recency_boost"] is True
    assert int(state.private_archive_settings["max_k"]) == 13


def test_restore_session_stats_registry_roundtrip():
    payload = {
        "registry": {
            7: {
                "research": {
                    "agent_01_t07": {
                        "log_type": "ResearchStatsLog",
                        "turns": 4,
                        "tool_calls": {"arxiv": 2},
                        "tool_inputs": {"arxiv": ["query a"]},
                        "tool_events": {"arxiv": [{"stage": "start", "payload": {"k": 1}}]},
                    }
                },
                "review": {
                    "reviewer_01_t07": {
                        "log_type": "ReviewStatsLog",
                        "turns": 2,
                        "tool_calls": {"search_agent_registry": 1},
                    }
                },
                "metareview": {
                    "metareview_reviewer_01_t07_tour000": {
                        "log_type": "MetaReviewStatsLog",
                        "turns": 1,
                    }
                },
            }
        }
    }

    registry = session_stats_registry_from_snapshot_payload(payload)
    research_log = registry.get_phase_logs(7, "research")["agent_01_t07"]
    review_log = registry.get_phase_logs(7, "review")["reviewer_01_t07"]
    metareview_log = registry.get_phase_logs(7, "metareview")["metareview_reviewer_01_t07_tour000"]

    assert isinstance(research_log, ResearchStatsLog)
    assert isinstance(review_log, ReviewStatsLog)
    assert isinstance(metareview_log, MetaReviewStatsLog)
    assert research_log.turns == 4
    assert research_log.tool_calls["arxiv"] == 2
    assert research_log.tool_inputs["arxiv"] == ["query a"]
    assert research_log.tool_events["arxiv"][0]["stage"] == "start"
    assert review_log.turns == 2
    assert review_log.tool_calls["search_agent_registry"] == 1
    assert metareview_log.turns == 1
