import json
import pickle

from system.bootstrap_iteration import build_bootstrap_selection


def test_build_bootstrap_selection_loads_notepad_entries(tmp_path):
    snapshot_dir = tmp_path / "snapshots" / "pool_run" / "iteration_1"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    agent_registry_data = {
        "agents": [
            {
                "agent_id": "agent_01",
                "behavior": "careful",
                "expertise": "biology",
                "expertise_topics": ["genomics"],
                "citation_count": 0,
                "num_accepted_papers": 0,
            },
            {
                "agent_id": "agent_02",
                "behavior": "bold",
                "expertise": "statistics",
                "expertise_topics": ["causal inference"],
                "citation_count": 0,
                "num_accepted_papers": 0,
            },
        ]
    }
    with (snapshot_dir / "agent_registry_data.pkl").open("wb") as f:
        pickle.dump(agent_registry_data, f)

    (snapshot_dir / "archive_data.json").write_text(json.dumps({"papers": []}), encoding="utf-8")
    (snapshot_dir / "private_archive_data.json").write_text(
        json.dumps({"papers": [], "reviews_by_paper": {}}),
        encoding="utf-8",
    )
    (snapshot_dir / "notepad_data.json").write_text(
        json.dumps(
            {
                "entries_by_agent": {
                    "agent_01": [
                        {
                            "timestamp": "2026-03-01T00:00:00Z",
                            "iteration": 2,
                            "run_name": "old_run",
                            "agent_id": "agent_01",
                            "idea_id": "idea_a1",
                            "title": "A1 idea",
                            "idea_text": "Deferred A1 idea text.",
                            "tags": ["tag1"],
                        }
                    ],
                    "agent_02": [
                        {
                            "timestamp": "2026-03-01T00:00:00Z",
                            "iteration": 3,
                            "run_name": "old_run",
                            "agent_id": "agent_02",
                            "idea_id": "idea_a2",
                            "title": "A2 idea",
                            "idea_text": "Deferred A2 idea text.",
                            "tags": ["tag2"],
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    (
        selected_agents,
        public_papers,
        private_papers,
        reviews_selected,
        notepad_selected,
    ) = build_bootstrap_selection(
        pool_paths=[snapshot_dir],
        num_agents=2,
        seed=7,
        max_public_papers_per_agent=None,
        max_private_papers_per_agent=None,
    )

    assert len(selected_agents) == 2
    assert public_papers == []
    assert private_papers == []
    assert reviews_selected == {}
    assert set(notepad_selected.keys()) == {"agent_01", "agent_02"}
    assert notepad_selected["agent_01"][0]["idea_id"] == "idea_a1"
    assert notepad_selected["agent_02"][0]["idea_id"] == "idea_a2"
