"""Experiment 03's AutoRanker config: the judge and parallelism are set only when given."""

import shutil
from pathlib import Path

from scripts.analysis.reviewer_matching.run_autoranker_for_cohort import _compose_ranker_config

_SYS_CONFIG = Path(__file__).resolve().parents[4] / "sys_config"


def _templates_only(directory, names):
    """copytree filter: leave out endpoint files (they may hold keys), keep the templates."""
    if Path(directory).name in ("llm_api", "embedder_api"):
        return [n for n in names if n != "default_to_complete.yaml"]
    return []


def test_defaults_unless_judge_or_workers_given(tmp_path, monkeypatch):
    shutil.copytree(_SYS_CONFIG, tmp_path / "sys_config", ignore=_templates_only)
    template = tmp_path / "sys_config" / "llm_api" / "default_to_complete.yaml"
    shutil.copy(template, template.with_name("gpt-4o.yaml"))
    monkeypatch.chdir(tmp_path)

    default = _compose_ranker_config("sample_and_rank_non_paired")
    assert default.ranker.llm_backend == "gpt5-high"
    assert default.parallelism.max_workers == 4

    cfg = _compose_ranker_config("sample_and_rank_non_paired", judge="gpt4o", max_workers=8)
    assert cfg.ranker.llm_backend == "gpt4o"
    assert cfg.parallelism.max_workers == 8
    assert "gpt4o" in cfg.endpoints  # the judge's endpoint file is composed
