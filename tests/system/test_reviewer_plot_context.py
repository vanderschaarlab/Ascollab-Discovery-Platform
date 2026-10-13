from agents.review.rev_agent_utils import make_reviewer_user_message


def test_make_reviewer_user_message_includes_plot_artifacts_block():
    msg = make_reviewer_user_message(
        report_text="Report text",
        code_report="print('code')",
        metrics={"support": {"external_papers_read": 2}},
        plot_artifacts=[
            {
                "plot_citation": "[Plot: plot_ab12cd34]",
                "plot_handle": "plot_ab12cd34",
                "plot_filename": "agent_01_sid-sid_job-job_cell-2_step-5_img-01.png",
                "plot_path": "plots/a.png",
                "summary_path": "plots/summaries/agent_01_sid-sid_job-job_cell-2_step-5_img-01.png.json",
                "summary_excerpt": "Treatment outperforms control.",
                "generation_backend": "code_stats_llm",
                "statistics_digest": {"series_count": 2, "comparison_count": 1},
            }
        ],
    )
    assert "PLOT ARTEFACTS" in msg
    assert "[Plot: plot_ab12cd34]" in msg
    assert "plot_ab12cd34" in msg
    assert "agent_01_sid-sid_job-job_cell-2_step-5_img-01.png" in msg
    assert "statistics_digest" in msg


def test_make_reviewer_user_message_omits_plot_artifacts_block_when_empty():
    msg = make_reviewer_user_message(
        report_text="Report text",
        code_report="print('code')",
        metrics={"support": {"external_papers_read": 2}},
        plot_artifacts=[],
    )
    assert "PLOT ARTEFACTS" not in msg
