from system.plot_summary_artifacts import (
    generate_plot_summary_for_image,
    normalize_graph_logic_cfg,
)


def test_normalize_graph_logic_cfg_defaults_to_non_vision_backend():
    cfg = normalize_graph_logic_cfg({"enabled": True})
    assert cfg["enabled"] is True
    assert cfg["plot_only_rewrite_cap"] == 2
    assert cfg["allow_plot_validation_waiver_after_cap"] is True
    assert cfg["summary"]["backend"] == "code_stats_llm"
    assert cfg["summary"]["fallback_order"][0] == "code_stats_llm"


def test_generate_plot_summary_for_image_uses_code_stats_llm_by_default():
    summary = generate_plot_summary_for_image(
        plot_filename="agent_01_sid-sid_job-job_cell-1_step-1_img-01.png",
        plot_path="plots/a.png",
        image_index=1,
        code="import matplotlib.pyplot as plt\nplt.plot([1,2,3],[2,4,6])\nplt.title('Trend')\n",
        resp={"stdout": "mean=4 std=2 p=0.01"},
        graph_logic_cfg={"enabled": True},
    )
    assert summary["generation_backend"] == "code_stats_llm"
    assert summary["plot_type"] in {
        "line",
        "bar",
        "scatter",
        "box",
        "histogram",
        "heatmap",
        "kaplan_meier",
        "other",
    }
    assert "statistics" in summary
    assert "stats_completeness" in summary["statistics"]


def test_generate_plot_summary_for_image_respects_heuristic_fallback():
    summary = generate_plot_summary_for_image(
        plot_filename="agent_01_sid-sid_job-job_cell-1_step-1_img-01.png",
        plot_path="plots/a.png",
        image_index=1,
        code="",
        resp={},
        graph_logic_cfg={
            "enabled": True,
            "summary": {
                "backend": "heuristic_fallback_only",
                "fallback_order": ["heuristic_fallback_only"],
            },
        },
    )
    assert summary["generation_backend"] == "heuristic_fallback_only"
    assert "warnings" in summary


def test_generate_plot_summary_for_image_uses_agent_sidecar_when_configured():
    summary = generate_plot_summary_for_image(
        plot_filename="agent_01_sid-sid_job-job_cell-1_step-1_img-01.png",
        plot_path="plots/a.png",
        image_index=1,
        code="",
        resp={
            "plot_summary_data": {
                "plot_type": "line",
                "title": "From Sidecar",
                "statistics": {
                    "series": [{"name": "a", "n": 3, "mean": 2.0}],
                    "comparisons": [],
                },
            }
        },
        graph_logic_cfg={
            "enabled": True,
            "summary": {
                "backend": "agent_sidecar",
                "fallback_order": ["agent_sidecar", "heuristic_fallback_only"],
            },
        },
    )
    assert summary["generation_backend"] == "agent_sidecar"
    assert summary["title"] == "From Sidecar"
    assert "stats_completeness" in summary["statistics"]


def test_generate_plot_summary_for_image_vision_is_not_used_when_not_allowed():
    summary = generate_plot_summary_for_image(
        plot_filename="agent_01_sid-sid_job-job_cell-1_step-1_img-01.png",
        plot_path="plots/a.png",
        image_index=1,
        code="import matplotlib.pyplot as plt\nplt.plot([1,2],[3,4])\n",
        resp={"stdout": "mean=3.5"},
        graph_logic_cfg={
            "enabled": True,
            "summary": {
                "backend": "vision_llm",
                "fallback_order": ["vision_llm", "code_stats_llm", "heuristic_fallback_only"],
                "allow_vision_fallback": False,
            },
        },
    )
    assert summary["generation_backend"] == "code_stats_llm"


def test_generate_plot_summary_for_image_vision_path_falls_back_with_warning():
    summary = generate_plot_summary_for_image(
        plot_filename="agent_01_sid-sid_job-job_cell-1_step-1_img-01.png",
        plot_path="plots/a.png",
        image_index=1,
        code="",
        resp={},
        graph_logic_cfg={
            "enabled": True,
            "summary": {
                "backend": "vision_llm",
                "fallback_order": ["vision_llm", "heuristic_fallback_only"],
                "allow_vision_fallback": True,
            },
        },
    )
    assert summary["generation_backend"] == "heuristic_fallback_only"
    warnings = list(summary.get("warnings") or [])
    assert any("vision_llm_not_implemented" in w for w in warnings)


def test_generate_plot_summary_for_image_respects_stats_toggles():
    summary = generate_plot_summary_for_image(
        plot_filename="agent_01_sid-sid_job-job_cell-1_step-1_img-01.png",
        plot_path="plots/a.png",
        image_index=1,
        code="import matplotlib.pyplot as plt\nplt.plot([1,2],[3,4],label='a')\nplt.plot([1,2],[5,6],label='b')\n",
        resp={"stdout": "a 3 4 b 5 6 p=0.04"},
        graph_logic_cfg={
            "enabled": True,
            "summary": {
                "backend": "code_stats_llm",
                "include_descriptive_stats": False,
                "include_comparative_stats": False,
            },
        },
    )
    stats = summary["statistics"]
    assert stats["series"] == []
    assert stats["comparisons"] == []
