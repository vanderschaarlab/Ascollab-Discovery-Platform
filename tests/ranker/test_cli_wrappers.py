import pytest
from omegaconf import OmegaConf

import ranker.run_ranker as run_ranker_cli
import ranker.run_sample_and_rank_non_paired as non_paired_cli
import ranker.run_sample_and_rank_paired as paired_cli
from ranker.ranker_schemas import RankerResult


def _run_ranker_cli(monkeypatch, score, mode="comparison"):
    calls = {}
    monkeypatch.setattr(run_ranker_cli, "setup_file_logging_hydra_safe", lambda: None)
    monkeypatch.setattr(
        run_ranker_cli, "build_llm_pool", lambda endpoints, keys=None, **_: {"fake": object()}
    )

    def fake_run_ranker(**kwargs):
        calls.update(kwargs)
        return RankerResult(score=score, justification="A wins", raw_response="raw", turns=2)

    monkeypatch.setattr(run_ranker_cli, "run_ranker", fake_run_ranker)
    cfg = OmegaConf.create(
        {
            "endpoints": {},
            "paper_a": {"title": "A", "abstract": "a", "manuscript": "ma"},
            "paper_b": {"title": "B", "abstract": "b", "manuscript": "mb"},
            "ranker": {
                "llm_backend": "fake",
                "tools": {"arxiv": {"enabled": False}, "sandbox": {"enabled": False}},
                "mode": mode,
                "max_iterations": 4,
                "llm_retry": {},
                "output_parsing": {"max_parse_retries": 1},
            },
        }
    )

    run_ranker_cli.main.__wrapped__(cfg)
    return calls


def test_run_ranker_cli_invokes_core_with_cli_papers(monkeypatch, capsys):
    calls = _run_ranker_cli(monkeypatch, score=1)

    assert calls["paper_a"]["title"] == "A"
    assert calls["max_iterations"] == 4
    assert calls["max_parse_retries"] == 1
    assert "Outcome: Paper A" in capsys.readouterr().out


@pytest.mark.parametrize("mode", ["comparison", "hypothesis_similarity"])
def test_run_ranker_cli_reports_a_missing_verdict(monkeypatch, capsys, mode):
    _run_ranker_cli(monkeypatch, score=None, mode=mode)

    assert "Outcome: No verdict (score=None)" in capsys.readouterr().out


def test_paired_cli_passes_composed_config_to_runner(monkeypatch):
    calls = {}
    monkeypatch.setattr(paired_cli, "setup_file_logging_hydra_safe", lambda: None)
    monkeypatch.setattr(
        paired_cli, "build_llm_pool", lambda endpoints, keys=None, **_: {"fake": object()}
    )
    monkeypatch.setattr(
        paired_cli.HydraConfig, "get", lambda: (_ for _ in ()).throw(RuntimeError())
    )
    monkeypatch.setattr(
        paired_cli, "run_sample_and_rank_paired", lambda **kwargs: calls.update(kwargs)
    )
    cfg = OmegaConf.create(
        {
            "endpoints": {},
            "run_name": "run",
            "experiments": {"experiment_a": {}, "experiment_b": {}},
            "sampling": {"n_pairs": 1},
            "parallelism": {"max_workers": 1},
            "paper_selection": {"top_n": None},
            "ranker": {
                "llm_backend": "fake",
                "tools": {"sandbox": {"enabled": False}},
            },
        }
    )

    paired_cli.main.__wrapped__(cfg)

    assert calls["run_id"] == "run"
    assert calls["output_dir"] is None
    assert calls["ranker_cfg"]["llm_backend"] == "fake"


def test_non_paired_cli_passes_composed_config_to_runner(monkeypatch):
    calls = {}
    monkeypatch.setattr(non_paired_cli, "setup_file_logging_hydra_safe", lambda: None)
    monkeypatch.setattr(
        non_paired_cli, "build_llm_pool", lambda endpoints, keys=None, **_: {"fake": object()}
    )
    monkeypatch.setattr(
        non_paired_cli.HydraConfig, "get", lambda: (_ for _ in ()).throw(RuntimeError())
    )
    monkeypatch.setattr(
        non_paired_cli,
        "run_sample_and_rank_non_paired",
        lambda **kwargs: calls.update(kwargs),
    )
    cfg = OmegaConf.create(
        {
            "endpoints": {},
            "run_name": "run_np",
            "paper_pool": {"snapshot_dirs": []},
            "sampling": {"n_pairs": 1},
            "parallelism": {"max_workers": 1},
            "paper_selection": {"top_n": None},
            "ranker": {
                "llm_backend": "fake",
                "tools": {"sandbox": {"enabled": False}},
            },
        }
    )

    non_paired_cli.main.__wrapped__(cfg)

    assert calls["run_id"] == "run_np"
    assert calls["paper_pool_cfg"] == {"snapshot_dirs": []}


# A filled-in endpoint (building a client sends no request) and the unfilled template.
_FILLED = {
    "azure_deployment": "gpt-5",
    "azure_endpoint": "https://example.invalid",
    "api_key": "not-a-key",
    "api_version": "2024-12-01-preview",
}
_TEMPLATE = {"azure_deployment": None, "azure_endpoint": None, "api_key": None}


@pytest.mark.parametrize(
    ("cli", "runner"),
    [
        (paired_cli, "run_sample_and_rank_paired"),
        (non_paired_cli, "run_sample_and_rank_non_paired"),
        (run_ranker_cli, "run_ranker"),
    ],
)
def test_cli_builds_only_its_judge(monkeypatch, cli, runner):
    # An unfilled default.yaml (llm_primary) must not stop a ranker whose judge is gpt5-high.
    calls = {}
    monkeypatch.setattr(cli, "setup_file_logging_hydra_safe", lambda: None)
    if hasattr(cli, "HydraConfig"):
        monkeypatch.setattr(cli.HydraConfig, "get", lambda: (_ for _ in ()).throw(RuntimeError()))

    def fake_runner(**kwargs):
        calls.update(kwargs)
        return RankerResult(score=1, justification="A wins", raw_response="raw", turns=1)

    monkeypatch.setattr(cli, runner, fake_runner)
    cfg = OmegaConf.create(
        {
            "endpoints": {"llm_primary": _TEMPLATE, "gpt5-high": _FILLED},
            "run_name": "run",
            "experiments": {"experiment_a": {}, "experiment_b": {}},
            "paper_pool": {"snapshot_dirs": []},
            "paper_a": {"title": "A", "abstract": "a", "manuscript": "ma"},
            "paper_b": {"title": "B", "abstract": "b", "manuscript": "mb"},
            "sampling": {"n_pairs": 1},
            "parallelism": {"max_workers": 1},
            "paper_selection": {"top_n": None},
            "ranker": {"llm_backend": "gpt5-high", "tools": {"sandbox": {"enabled": False}}},
        }
    )

    cli.main.__wrapped__(cfg)

    assert calls["llm"] is not None
