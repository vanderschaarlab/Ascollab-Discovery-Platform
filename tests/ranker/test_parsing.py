"""Tests for verdict and action parsing."""

from ranker.ranker_utils import parse_action, parse_verdict

# ---------------------------------------------------------------------------
# Verdict parsing tests
# ---------------------------------------------------------------------------


class TestParseVerdict:
    """Test multi-strategy verdict parsing."""

    def test_clean_format_a(self):
        text = "Verdict:\n- Winner: A\n- Justification: Paper A is stronger"
        result, valid = parse_verdict(text)
        assert valid is True
        assert result["score"] == 1
        assert "Paper A is stronger" in result["justification"]

    def test_clean_format_b(self):
        text = "Verdict:\n- Winner: B\n- Justification: Paper B has better methodology"
        result, valid = parse_verdict(text)
        assert valid is True
        assert result["score"] == 0

    def test_messy_whitespace(self):
        text = "  verdict :\n  -  winner:  A \n- justification: reasons here"
        result, valid = parse_verdict(text)
        assert valid is True
        assert result["score"] == 1

    def test_markdown_headers(self):
        text = "## Verdict\n- Winner: B\n- Justification: Better evidence"
        result, valid = parse_verdict(text)
        assert valid is True
        assert result["score"] == 0

    def test_inline_winner_only(self):
        text = "After analysis, Winner: A. Because of stronger methodology."
        result, valid = parse_verdict(text)
        assert valid is True
        assert result["score"] == 1

    def test_natural_language_paper_a_is_better(self):
        text = "Paper A is better because it has stronger evidence and methodology."
        result, valid = parse_verdict(text)
        assert valid is True
        assert result["score"] == 1

    def test_natural_language_paper_b_wins(self):
        text = "After careful review, Paper B wins due to superior analysis."
        result, valid = parse_verdict(text)
        assert valid is True
        assert result["score"] == 0

    def test_no_verdict_at_all(self):
        text = "I need more information to make a decision."
        result, valid = parse_verdict(text)
        assert valid is False

    def test_invalid_winner(self):
        text = "Verdict:\n- Winner: C"
        result, valid = parse_verdict(text)
        assert valid is False

    def test_action_block_not_verdict(self):
        text = 'Action: {"tool": "arxiv", "args": {"query": "test"}}'
        result, valid = parse_verdict(text)
        assert valid is False

    def test_empty_string(self):
        result, valid = parse_verdict("")
        assert valid is False

    def test_none_input(self):
        result, valid = parse_verdict(None)
        assert valid is False

    def test_justification_extraction(self):
        text = (
            "Some reasoning here.\n\n"
            "Verdict:\n"
            "- Winner: A\n"
            "- Justification: Paper A demonstrates superior methodology "
            "with proper statistical controls."
        )
        result, valid = parse_verdict(text)
        assert valid is True
        assert result["score"] == 1
        assert "statistical controls" in result["justification"]

    def test_choose_pattern(self):
        text = "After analysis, I choose A as the better paper."
        result, valid = parse_verdict(text)
        assert valid is True
        assert result["score"] == 1


# ---------------------------------------------------------------------------
# Action parsing tests
# ---------------------------------------------------------------------------


class TestParseAction:
    """Test action block extraction from LLM output."""

    def test_valid_action(self):
        text = 'Action: {"tool": "arxiv", "args": {"query": "test"}}'
        result = parse_action(text)
        assert result is not None
        assert result["tool"] == "arxiv"
        assert result["args"]["query"] == "test"

    def test_no_action_present(self):
        text = "Just some reasoning text with no action block."
        result = parse_action(text)
        assert result is None

    def test_malformed_json_fallback(self):
        text = "Action: {'tool': 'arxiv', 'args': {'query': 'test'}}"
        result = parse_action(text)
        assert result is not None
        assert result["tool"] == "arxiv"

    def test_action_in_code_block(self):
        text = '```\nAction: {"tool": "sandbox", "args": {"code": "print(1)"}}\n```'
        result = parse_action(text)
        assert result is not None
        assert result["tool"] == "sandbox"

    def test_action_with_surrounding_text(self):
        text = (
            "I should search for this paper.\n"
            'Action: {"tool": "arxiv", "args": {"query": "gene expression KIRC"}}\n'
            "Let me wait for results."
        )
        result = parse_action(text)
        assert result is not None
        assert result["tool"] == "arxiv"

    def test_completely_invalid(self):
        text = "Action: not json at all"
        result = parse_action(text)
        assert result is None

    def test_verdict_not_parsed_as_action(self):
        text = "Verdict:\n- Winner: A\n- Justification: reasons"
        result = parse_action(text)
        assert result is None
