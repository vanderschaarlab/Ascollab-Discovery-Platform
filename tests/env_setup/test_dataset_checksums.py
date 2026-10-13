"""TCGA checksums: the download script's layout matches them and preflight flags a changed file."""

import hashlib
from pathlib import Path

from env_setup.preflight_data import _checksum_error, load_known_checksums, preflight_data
from scripts.data.download_tcga import COHORTS, files_for


def test_every_downloaded_file_has_a_checksum_and_a_description_path():
    known = load_known_checksums()
    rels = {rel for cohort in COHORTS for _, rel in files_for(cohort)}
    assert rels == set(known)
    descriptions = (Path(__file__).resolve().parents[2] / "env_setup" / "datasets.py").read_text()
    for rel in rels:
        assert f'"sandbox/datasets/{rel}"' in descriptions, rel


def test_checksum_mismatch_is_named_and_match_passes(tmp_path):
    path = tmp_path / "KIRC_NeurIPS" / "TCGA.KIRC.sampleMap_RPPA.gz"
    path.parent.mkdir()
    path.write_bytes(b"not the Xena file")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    known = {"KIRC_NeurIPS/TCGA.KIRC.sampleMap_RPPA.gz": {"0" * 64}}
    assert "differs" in _checksum_error(path, sha, tmp_path, known)
    assert (
        _checksum_error(path, sha, tmp_path, {"KIRC_NeurIPS/TCGA.KIRC.sampleMap_RPPA.gz": {sha}})
        is None
    )
    assert _checksum_error(path, sha, tmp_path, {}) is None


def test_preflight_fails_on_a_changed_tcga_file(tmp_path):
    root = tmp_path  # descriptions give repo-relative paths: <root>/sandbox/datasets/...
    target = root / "sandbox" / "datasets" / "KIRC_NeurIPS" / "survival_FKIRC_survival.txt"
    target.parent.mkdir(parents=True)
    target.write_text("sample\tOS\nTCGA-XX-0001-01\t1\n")
    report = preflight_data(
        dataset_keys=["tcga_kirc_neurips_survival"],
        dataset_root=root,
        save_manifest=tmp_path / "manifest.json",
    )
    assert report["ok"] is False
    assert any("survival_FKIRC_survival.txt: SHA-256 differs" in e for e in report["errors"])


SURVIVAL = "KIRC_NeurIPS/survival_FKIRC_survival.txt"


def _changed_survival_file(folder: Path) -> Path:
    path = folder / "survival_FKIRC_survival.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("sample\tOS\nTCGA-XX-0001-01\t1\n")
    return path


def _survival_report(root: Path, tmp_path: Path) -> dict:
    return preflight_data(
        dataset_keys=["tcga_kirc_neurips_survival"],
        dataset_root=root,
        save_manifest=tmp_path / "manifest.json",
    )


def test_preflight_checks_a_dataset_folder_symlinked_from_elsewhere(tmp_path):
    # Resolving the symlink first put the file outside sandbox/datasets, so the check was skipped.
    _changed_survival_file(tmp_path / "storage" / "kirc")
    root = tmp_path / "repo"
    (root / "sandbox" / "datasets").mkdir(parents=True)
    (root / "sandbox" / "datasets" / "KIRC_NeurIPS").symlink_to(tmp_path / "storage" / "kirc")

    report = _survival_report(root, tmp_path)

    assert report["ok"] is False
    assert any(f"{SURVIVAL}: SHA-256 differs" in e for e in report["errors"])


def test_preflight_checks_files_through_a_symlinked_datasets_folder(tmp_path):
    # docs/data.md suggests symlinking sandbox/datasets to where the data are.
    path = _changed_survival_file(tmp_path / "storage" / "KIRC_NeurIPS")
    root = tmp_path / "repo"
    (root / "sandbox").mkdir(parents=True)
    (root / "sandbox" / "datasets").symlink_to(tmp_path / "storage")

    report = _survival_report(root, tmp_path)

    assert any(f"{SURVIVAL}: SHA-256 differs" in e for e in report["errors"])
    linked = root / "sandbox" / "datasets" / SURVIVAL
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert _checksum_error(linked, sha, root, {SURVIVAL: {sha}}) is None


def test_preflight_checks_files_found_in_a_dataset_folder(tmp_path, monkeypatch):
    # A description may name a folder; the files probed inside it are checked too.
    rppa = tmp_path / "sandbox" / "datasets" / "KIRC_NeurIPS" / "TCGA.KIRC.sampleMap_RPPA.gz"
    rppa.parent.mkdir(parents=True)
    rppa.write_bytes(b"not the Xena file")
    (tmp_path / "folder_datasets_for_test.py").write_text(
        'DATASETS = {"kirc_folder": {"path": "sandbox/datasets/KIRC_NeurIPS"}}\n'
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    report = preflight_data(
        dataset_module="folder_datasets_for_test",
        dataset_keys=["kirc_folder"],
        dataset_root=tmp_path,
        save_manifest=tmp_path / "manifest.json",
    )

    assert report["datasets"]["kirc_folder"]["paths"] == [str(rppa)]
    assert report["ok"] is False
    assert any(
        "KIRC_NeurIPS/TCGA.KIRC.sampleMap_RPPA.gz: SHA-256 differs" in e for e in report["errors"]
    )
