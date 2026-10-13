"""Release defaults, experiments 07-10 as the scripts run them, and the review_norm group."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from tests.golden.harness import SENSITIVE_KEYS, collect_conditions

_CFG_DIR = str(Path(__file__).resolve().parents[2] / "sys_config")

_NORM_A_TEXT = (
    "prioritize papers that are potentially groundbreaking even if it is not strongly supported"
)
_NORM_B_TEXT = "prioritize papers with sound empirical support"


def _compose(name, overrides=None):
    with initialize_config_dir(config_dir=_CFG_DIR, version_base=None):
        return compose(config_name=name, overrides=overrides or [])


def test_release_defaults():
    """The base config holds the paper's settings (docs/running.md, Main settings); experiments
    override them."""
    cfg = _compose("ascollab")
    assert cfg.exp_name == "ascollab"
    assert cfg.SEED == 1
    assert list(cfg.discovery_dataset.dataset_keys) == [
        "tcga_kirc_neurips_hiseqv2",
        "tcga_kirc_neurips_rppa",
        "tcga_kirc_neurips_merged_reals",
        "tcga_kirc_neurips_survival",
    ]
    assert cfg.data.preflight_strict is True
    # Paper hyperparameters: N, T, M, K, L.
    assert cfg.network.num_agents == 16
    assert cfg.network.iterations == 40
    assert cfg.research.research_max_iters == 40
    assert cfg.review.num_reviewers_per_paper == 2
    assert cfg.metareview.tournament_size == 4
    assert cfg.review.reviewer_assignment_strategy == "expertise_matching"
    # Release defaults: the full ASCollab system.
    assert cfg.network.setting == "network_hetero"
    assert cfg.network.enable_collaboration_tools is True
    assert cfg.network.enable_social_memory_tools is True
    assert dict(cfg.llm_backends.counts) == {"gpt4o": 6, "gpt5": 5, "deepseek": 5, "llama": 0}
    assert cfg.research.code_collaboration is True
    assert cfg.research.prior_papers.enabled is False
    assert cfg.research.full_citing is False  # needs prior_papers, so off too
    assert cfg.research.code_timeout_seconds == 900
    assert cfg.research.total_session_timeout_seconds == 720
    assert cfg.research.max_concurrent_research_agents == 10
    assert cfg.research.self_reflection_prompt.enabled is False
    assert cfg.research.nature_style_papers_prompt.enabled is False
    assert cfg.research.metacognition_feedback.enabled is False
    assert cfg.graph_logic.enabled is False
    assert cfg.bootstrap.enabled is False
    assert list(cfg.bootstrap.pool_path) == []
    assert cfg.reputation.enabled is False
    assert cfg.reputation_pause.mode == "none"
    # The manipulations of experiments 07-10 are off by default.
    assert cfg.metareview.reviewerless.enabled is False
    assert cfg.network.social_memory.mode == "cumulative"
    assert cfg.review.norm_prompt.enabled is False


def _flat(node, prefix=""):
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k not in SENSITIVE_KEYS:
                out.update(_flat(v, f"{prefix}.{k}" if prefix else str(k)))
        return out
    return {prefix: node}


def _flat_config(cfg) -> dict:
    """A composed config as {dotted key: value}, without endpoints or credentials.

    Plain dicts lose the redacting DictConfig repr from tests/conftest.py, so a failing test run
    with ``-vv -l`` would print them; leaving these keys out keeps API keys out of that output.
    """
    return _flat(OmegaConf.to_container(cfg, resolve=False))


@pytest.fixture(scope="module")
def followup_configs():
    """Resolved configs of the 07-10 conditions, composed from the scripts' DRY_RUN output."""
    conditions = collect_conditions()
    out = {}
    for fid, cond in conditions.items():
        if fid.split("__")[0] in ("rb", "07", "08", "09", "10"):
            cfg = _compose(cond["config_name"], cond["overrides"])
            out[fid] = _flat_config(cfg)
    return out


def test_flattened_configs_hold_no_endpoint_or_key():
    cfg = _compose("ascollab", ["+endpoints.fake_backend.api_key=FAKE-KEY-123"])
    flat = _flat_config(cfg)
    assert "FAKE-KEY-123" not in repr(flat)
    assert not [k for k in flat if k.startswith("endpoints.") or "api_key" in k]
    assert flat["review.num_reviewers_per_paper"] == 2


def _diff_from_baseline(configs, fid):
    base, cond = configs["rb__baseline__kirc"], configs[fid]
    keys = (set(base) | set(cond)) - {"exp_name"}
    return {k: cond.get(k) for k in keys if base.get(k) != cond.get(k)}


def test_shared_baseline_condition(followup_configs):
    cfg = followup_configs["rb__baseline__kirc"]
    assert cfg["metareview.reviewerless.enabled"] is False
    assert cfg["network.social_memory.mode"] == "cumulative"
    assert cfg["review.norm_prompt.enabled"] is False
    assert cfg["review.num_reviewers_per_paper"] == 2
    assert cfg["review.reviewer_assignment_condition_label"] == "n_reviewers=2"
    assert cfg["network.iterations"] == 20
    assert cfg["research.prior_papers.enabled"] is True
    assert cfg["research.prior_papers.mode"] == "full"
    assert cfg["research.full_citing"] is True


def test_no_reviewers_changes_only_reviewerless(followup_configs):
    # The no-reviewers arm drops the (inert) reviewer-count override and label.
    assert _diff_from_baseline(followup_configs, "07__no_reviewers__kirc") == {
        "metareview.reviewerless.enabled": True,
        "review.reviewer_assignment_condition_label": None,
    }


def test_markovian_memory_changes_only_social_memory(followup_configs):
    assert _diff_from_baseline(followup_configs, "08__markovian_memory__kirc") == {
        "network.social_memory.mode": "markovian"
    }


def test_reviewer_norms_change_only_the_norm(followup_configs):
    for norm, text in (("norm_a_novelty", _NORM_A_TEXT), ("norm_b_soundness", _NORM_B_TEXT)):
        assert _diff_from_baseline(followup_configs, f"09__{norm}__kirc") == {
            "review.norm_prompt.enabled": True,
            "review.norm_prompt.label": norm,
            "review.norm_prompt.text": text,
        }


def test_reviewer_count_changes_only_k_and_its_label(followup_configs):
    for n in (1, 3, 4):
        assert _diff_from_baseline(followup_configs, f"10__n{n}__kirc") == {
            "review.num_reviewers_per_paper": n,
            "review.reviewer_assignment_condition_label": f"n_reviewers={n}",
        }


def test_review_norm_group_standalone_on_base_config():
    a = _compose("ascollab", ["+review_norm@review.norm_prompt=norm_a_novelty"]).review.norm_prompt
    assert a.enabled is True and a.label == "norm_a_novelty" and a.text == _NORM_A_TEXT

    b = _compose(
        "ascollab", ["+review_norm@review.norm_prompt=norm_b_soundness"]
    ).review.norm_prompt
    assert b.enabled is True and b.label == "norm_b_soundness" and b.text == _NORM_B_TEXT

    none = _compose("ascollab", ["+review_norm@review.norm_prompt=none"]).review.norm_prompt
    assert none.enabled is False and none.label is None and none.text is None
