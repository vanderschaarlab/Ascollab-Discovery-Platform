"""Dataset preflight defaults: where dataset keys and the dataset root come from."""

import subprocess
import sys
from pathlib import Path

import yaml

from env_setup.preflight_data import preflight_data
from tests.isolation import subprocess_env

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_default_config_supplies_the_default_bundles_dataset_keys(tmp_path, monkeypatch):
    # sys_config/ascollab.yaml lists no dataset keys (they are in the bundles), so a default
    # that pointed there always reported "No dataset keys".
    monkeypatch.chdir(REPO_ROOT)
    kirc = yaml.safe_load((REPO_ROOT / "sys_config/dataset/kirc.yaml").read_text())

    report = preflight_data(dataset_root=tmp_path, save_manifest=tmp_path / "manifest.json")

    assert report["keys_source"] == "base_cfg"
    assert list(report["datasets"]) == kirc["dataset_keys"]
    assert not any("No dataset keys" in e for e in report["errors"])


def test_cli_help_names_every_dataset_root_source():
    out = subprocess.run(
        [sys.executable, "-m", "env_setup.preflight_data", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    help_text = " ".join(out.split())
    for source in ("$DATASET_ROOT", "$DATA_ROOT", "data.root", "datasets_root", "paths.data_root"):
        assert source in help_text


def test_cli_exits_1_when_the_data_check_fails(tmp_path):
    # An unknown dataset key fails the check on every machine, whatever data it has.
    config = tmp_path / "bundle.yaml"
    config.write_text("dataset_keys: [no_such_dataset]\n", encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "env_setup.preflight_data",
            "--config",
            str(config),
            "--manifest",
            str(tmp_path / "manifest.json"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=subprocess_env(),
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "DATA_AVAILABLE: False" in proc.stdout
