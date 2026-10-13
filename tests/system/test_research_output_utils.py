from langchain_core.messages import AIMessage

from system.research_output_utils import (
    extract_final_report,
    find_unresolved_template_placeholders,
    has_unresolved_template_placeholders,
    sanitize_report,
)


def test_extract_final_report_strips_action_lines():
    msg = AIMessage(
        content=(
            "Final Report: Title\n"
            "Some result text.\n"
            "Action: {'tool': 'execute_code', 'args': {'code': 'print(1)'}}\n"
            "Observation: <the result of the action, do not write this yourself>\n"
        )
    )
    report = extract_final_report([msg])
    assert "Action:" not in report
    assert "Observation:" not in report
    assert "Some result text." in report


def test_sanitize_report_flags_and_strips():
    raw = "Line 1\nAction: {'tool': 'x'}\nLine 2\nObservation: y\n"
    cleaned, stripped = sanitize_report(raw)
    assert stripped is True
    assert "Action:" not in cleaned
    assert "Observation:" not in cleaned
    assert cleaned == "Line 1\nLine 2"


def test_detect_unresolved_template_placeholders():
    text = (
        "# Methods\n"
        "Datasets: <dataset_key_1>, <dataset_key_2>\n"
        "Traceability: C<id> -> M<id>\n"
        "Reference: [Archive: <paper_id>]\n"
    )
    tokens = find_unresolved_template_placeholders(text)
    assert "<dataset_key_1>" in tokens
    assert "<dataset_key_2>" in tokens
    assert "<paper_id>" in tokens
    assert "C<id>" in tokens
    assert "M<id>" in tokens
    assert has_unresolved_template_placeholders(text) is True


def test_ignore_normal_statistical_text_for_placeholder_detection():
    text = "Observed p < 0.05 with stable effect size across controls."
    tokens = find_unresolved_template_placeholders(text)
    assert tokens == []
    assert has_unresolved_template_placeholders(text) is False
