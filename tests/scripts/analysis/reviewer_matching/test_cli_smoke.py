import json

from scripts.analysis.reviewer_matching import aggregate_autoranker_scores, prepare_cohorts


def test_aggregate_autoranker_scores_cli_writes_scores(tmp_path, monkeypatch):
    comparisons = tmp_path / "comparisons.json"
    manifest = tmp_path / "manifest.json"
    output = tmp_path / "scores.json"
    comparisons.write_text(
        json.dumps(
            [
                {"status": "completed", "paper_a_id": "p1", "paper_b_id": "p2", "outcome": "A"},
                {"status": "completed", "paper_a_id": "p2", "paper_b_id": "p1", "outcome": "B"},
            ]
        ),
        encoding="utf-8",
    )
    manifest.write_text(json.dumps({"candidate_paper_ids": ["p1", "p2"]}), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "aggregate_autoranker_scores",
            "--comparisons",
            str(comparisons),
            "--manifest",
            str(manifest),
            "--output",
            str(output),
        ],
    )

    aggregate_autoranker_scores.main()

    scores = json.loads(output.read_text(encoding="utf-8"))
    assert scores["scoring_method"] == "win_rate"
    assert [row["paper_id"] for row in scores["paper_scores"]] == ["p1", "p2"]


def test_prepare_cohorts_cli_uses_candidate_pool_and_writes_manifest(tmp_path, monkeypatch):
    output_dir = tmp_path / "manifests"
    papers = [
        {"paper_id": f"p{i}", "publication_t": 5, "title": "t", "abstract": "a", "manuscript": "m"}
        for i in range(3)
    ]
    monkeypatch.setattr(
        prepare_cohorts, "load_candidate_pool", lambda snapshot_dirs, max_iteration: papers
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_cohorts",
            "--dataset",
            "tcga",
            "--snapshot-dirs",
            "snap_a",
            "--target-iterations",
            "5",
            "--cohorts-per-iteration",
            "1",
            "--cohort-size",
            "2",
            "--output-dir",
            str(output_dir),
        ],
    )

    prepare_cohorts.main()

    files = list(output_dir.glob("*.json"))
    assert len(files) == 1
    manifest = json.loads(files[0].read_text(encoding="utf-8"))
    assert manifest["dataset"] == "tcga"
    assert manifest["cohort_size"] == 2
