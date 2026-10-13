from agents.review.rev_agent_utils import make_reviewer_user_message


def test_reviewer_message_includes_meta_research_context_block():
    msg = make_reviewer_user_message(
        report_text="report",
        code_report=None,
        metrics=None,
        plot_artifacts=None,
        meta_research_context={
            "paper_type": "meta_research synthesis paper",
            "expectations": [
                "Evaluate breadth and integration quality.",
                "Evaluate contradiction handling quality.",
            ],
            "source_paper_ids": ["paper_a", "paper_b"],
        },
    )
    assert "Meta-Research Reviewer Context" in msg
    assert "meta_research synthesis paper" in msg
    assert "paper_a" in msg
    assert "Evaluate breadth and integration quality." in msg
