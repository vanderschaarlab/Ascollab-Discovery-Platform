from agents.review.reviewer_agent import parse_action


def test_reviewer_parse_action_accepts_python_literal_payload():
    action = parse_action("Action: {'tool': 'arxiv', 'args': {'query': 'cancer genomics'}}")

    assert action == {"tool": "arxiv", "args": {"query": "cancer genomics"}}


def test_reviewer_parse_action_rejects_values_that_are_not_actions():
    # Mismatched curly/ASCII quotes that literal_eval reads as a set of one string.
    assert parse_action("Action: {“tool': 'arxiv', 'args': {'query': 'cancer'}”}") is None
    assert parse_action("Action: {'tool': 'arxiv', 'args': None}") is None


def test_reviewer_parse_action_keeps_string_args():
    action = parse_action('Action: {"tool": "arxiv", "args": "cancer genomics"}')
    assert action == {"tool": "arxiv", "args": "cancer genomics"}


def test_reviewer_parse_action_rejects_a_tool_that_is_not_a_string():
    assert parse_action('Action: {"tool": ["arxiv"], "args": {"query": "cancer"}}') is None
