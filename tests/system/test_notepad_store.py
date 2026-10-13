import json
from pathlib import Path

from stores.notepad_store import NotepadStore


def test_notepad_store_append_and_snapshot_roundtrip(tmp_path):
    base_dir = tmp_path / "snapshots"
    store = NotepadStore(
        enabled=True,
        max_entries_per_agent=50,
        max_entry_chars=500,
        base_snapshots_dir=str(base_dir),
    )
    row = store.append_idea(
        agent_id="agent_1",
        iteration=3,
        run_name="run_alpha",
        title="Try subgroup stability",
        idea_text="Check whether the effect persists in low-baseline subgroup.",
        deferred_reason="Need more runtime budget this iteration.",
        tags=["subgroup", "robustness"],
    )

    notepad_path = Path(row["notepad_path"])
    assert notepad_path.exists()
    assert notepad_path.name == "notepad.jsonl"
    lines = notepad_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    saved = json.loads(lines[0])
    assert saved["idea_id"] == row["idea_id"]
    assert saved["agent_id"] == "agent_1"

    payload = store.snapshot_data()
    restored = NotepadStore(
        enabled=True,
        max_entries_per_agent=50,
        max_entry_chars=500,
        base_snapshots_dir=str(base_dir),
    )
    restored.load_snapshot_data(payload)
    restored_rows = restored.get_entries("agent_1")
    assert len(restored_rows) == 1
    assert restored_rows[0]["idea_id"] == row["idea_id"]
