from pathlib import Path

import yaml

from env_setup.datasets import DATASETS


def test_all_experiment_dataset_keys_exist_in_dataset_registry():
    missing: dict[str, list[str]] = {}
    for cfg_path in Path("sys_config").glob("*.yaml"):
        payload = yaml.safe_load(cfg_path.read_text()) or {}
        dataset_cfg = payload.get("discovery_dataset") or {}
        for key in dataset_cfg.get("dataset_keys") or []:
            if key not in DATASETS:
                missing.setdefault(str(cfg_path), []).append(key)

    assert missing == {}


def test_dataset_descriptions_are_non_empty_strings():
    assert DATASETS
    assert all(isinstance(key, str) and key for key in DATASETS)
    assert all(isinstance(desc, str) and desc.strip() for desc in DATASETS.values())
