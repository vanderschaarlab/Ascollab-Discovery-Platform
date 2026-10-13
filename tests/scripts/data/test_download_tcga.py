"""download_tcga keeps files with a recorded checksum and replaces the others (no network)."""

import hashlib

from scripts.data import download_tcga


def test_keeps_matching_files_and_replaces_the_others(tmp_path, monkeypatch):
    files = [rel for _, rel in download_tcga.files_for("kirc")]
    good, wrong, unrecorded, missing = files
    fresh = b"downloaded"
    monkeypatch.setattr(
        download_tcga,
        "load_known_checksums",
        lambda: {rel: [hashlib.sha256(fresh).hexdigest()] for rel in (good, wrong, missing)},
    )
    downloaded = []

    def fake_download(url, dest):
        downloaded.append(dest.relative_to(tmp_path).as_posix())
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(fresh)

    monkeypatch.setattr(download_tcga, "download", fake_download)
    for rel, content in ((good, fresh), (wrong, b"old"), (unrecorded, b"old")):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(content)

    assert download_tcga.main(["--cohorts", "kirc", "--dest", str(tmp_path)]) == 0

    assert downloaded == [wrong, unrecorded, missing]
    assert all((tmp_path / rel).read_bytes() == fresh for rel in files)
