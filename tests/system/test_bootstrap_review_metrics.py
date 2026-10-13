import json
import pickle

from system.bootstrap_iteration import load_pool


def test_load_pool_restores_private_paper_review_metrics(tmp_path):
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
            }
        ]
    }
    with (snapshot_dir / "agent_registry_data.pkl").open("wb") as f:
        pickle.dump(agent_registry_data, f)

    (snapshot_dir / "archive_data.json").write_text(json.dumps({"papers": []}), encoding="utf-8")
    private_payload = {
        "papers": [
            {
                "paper_id": "paper_001",
                "primary_agent_id": "agent_01",
                "collab_agent_ids": [],
                "title": "Paper 001",
                "abstract": "Test abstract.",
                "manuscript": "Test manuscript.",
                "citation_count": 0,
                "publication_t": 1,
                "cited_paper_ids": [],
                "status": "accepted",
                "publication_status": "accepted",
                "research_validation_status": "under_review",
                "review_metrics": {
                    "originality": {
                        "embedding_novelty_percentile": 88.0,
                    }
                },
            }
        ],
        "reviews_by_paper": {},
    }
    (snapshot_dir / "private_archive_data.json").write_text(
        json.dumps(private_payload),
        encoding="utf-8",
    )
    (snapshot_dir / "notepad_data.json").write_text(
        json.dumps({"entries_by_agent": {}}),
        encoding="utf-8",
    )

    _agents, _public_papers, private_papers, _reviews, _notepad = load_pool(snapshot_dir)
    assert len(private_papers) == 1
    metrics = private_papers[0].review_metrics
    assert isinstance(metrics, dict)
    assert metrics["originality"]["embedding_novelty_percentile"] == 88.0
