"""End-to-end: a collaborator's sub-log records the model it ran on, the author's."""

from __future__ import annotations

import pickle

from tests.e2e import harness


def test_collaborator_sub_log_records_the_authors_backend(tmp_path):
    fixture = harness.run_condition(
        harness.conditions()["01__ascollab__kirc"]["overrides"], tmp_path
    )
    assert fixture["exit_code"] == 0, fixture.get("error")
    (run_dir,) = sorted((tmp_path / "snapshots").glob("*"))
    with (run_dir / f"iteration_{harness.ROUNDS}" / "session_stats_data.pkl").open("rb") as f:
        registry = pickle.load(f)["registry"]

    def backend(entry: dict) -> str:
        return entry["compute_requirements"]["llm"]["backend_key"]

    checked = differing = 0
    for iteration, logs in registry.items():
        research = logs.get("research", {})
        for session_id, entry in research.items():
            if "_collab_for_" not in session_id:
                continue
            # "<collab>_tNN_collab_for_<author>"; only the key says whose log it is.
            prefix, author = session_id.split("_collab_for_", 1)
            collab_id = prefix.rsplit("_t", 1)[0]
            author_entry = research[f"{author}_t{int(iteration):02d}"]
            assert backend(entry) == backend(author_entry), session_id
            checked += 1
            own_entry = research.get(f"{collab_id}_t{int(iteration):02d}")
            differing += own_entry is not None and backend(own_entry) != backend(author_entry)
    assert checked
    # The run mixes backends, so some collaborators' own backend differs from the author's.
    assert differing
