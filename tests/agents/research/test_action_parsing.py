import pytest

from agents.research.collab_agent import parse_action as collab_parse_action
from agents.research.research_team import (
    _normalize_action_block,
    _repair_unescaped_quotes,
    _unusable_args,
    parse_action,
)

# Fixture prose and code come from the gpt-5 research agent in run
# outputs/2026-07-24/14-54-38, which lost 10 consecutive steps to parse failures.
# The run log truncates each block at 500 characters, so the captured fragments are
# closed off here to form complete blocks; the quoting hazards are verbatim.
CAPTURED_PROSE = (
    "Acknowledged. I will implement the refined OXPHOS module analysis focusing on "
    "nuclear-encoded subunits in class 5 with top-k=300, compute submodule "
    "(I/II/III/IV/V) scores, bootstrap CIs (1,000), and an empirical mean-matched "
    "null (2D bins by global mean and detection rate; 2,000 sets). I will also "
    "quantify specificity with an approximate Cohen’s d vs “others,” generate a "
    "gene-level heatmap of μ_c − μ_global for the module."
)

CAPTURED_CODE = (
    "import numpy as np\\nimport re\\n\\n"
    "# Assumes X, y, genes are in memory\\n"
    'oxphos_patterns = {\\n    \\"complex_I\\": [re.compile(r\\"^NDUF\\")],\\n'
    '    \\"complex_II\\": [re.compile(r\\"^SDH[ABCD]\\")],\\n}'
)


def _action(payload: str) -> str:
    return f"Action: {payload}"


def test_parse_action_with_embedded_report_text():
    text = (
        "Final Report: something\n"
        "Action: {'tool': 'execute_code', 'args': {'code': \"print('ok')\"}}\n"
        "Observation: <the result of the action>\n"
    )
    act = parse_action(text)
    assert isinstance(act, dict)
    assert act.get("tool") == "execute_code"


def test_parse_action_invalid_returns_marker():
    text = "Action: {not valid json"
    act = parse_action(text)
    assert isinstance(act, dict)
    assert act.get("tool") == "__invalid_action__"


# --- _normalize_action_block ------------------------------------------
def test_normalize_leaves_smart_quotes_inside_prose_alone():
    # Rewriting ’ to ' here would terminate the single-quoted string early, which
    # is what previously made the normalized fallback useless.
    block = "{'tool': 'communicate_with_collaborator', 'args': {'message': 'Cohen’s d vs “others,” ok'}}"

    normalized = _normalize_action_block(block)

    assert "Cohen’s" in normalized
    assert "“others,”" in normalized


def test_normalize_rewrites_smart_quotes_used_as_delimiters():
    block = "{‘tool’: ‘execute_code’, ‘args’: {‘code’: ‘print(1)’}}"

    assert parse_action(_action(block)) == {
        "tool": "execute_code",
        "args": {"code": "print(1)"},
    }


def test_normalize_strips_trailing_commas():
    block = "{'tool': 'arxiv', 'args': {'query': 'renal',},}"

    assert parse_action(_action(block)) == {"tool": "arxiv", "args": {"query": "renal"}}


# --- _repair_unescaped_quotes -----------------------------------------
def test_repair_escapes_apostrophe_inside_single_quoted_string():
    repaired = _repair_unescaped_quotes("{'message': 'the cell's state'}")

    assert repaired == "{'message': 'the cell\\'s state'}"


def test_repair_leaves_well_formed_blocks_untouched():
    block = "{'tool': 'arxiv', 'args': {'query': 'renal, carcinoma'}}"

    assert _repair_unescaped_quotes(block) == block


def test_repair_preserves_existing_escapes():
    block = "{'code': 'print(\\\"hi\\\")'}"

    assert _repair_unescaped_quotes(block) == block


# --- parse_action end to end ------------------------------------------
def test_parse_action_recovers_prose_with_unescaped_apostrophe():
    block = (
        "{'tool': 'communicate_with_collaborator', 'args': "
        "{'agent_id': 'agent_02', 'message': 'we can't rule out doublets'}}"
    )

    parsed = parse_action(_action(block))

    assert parsed["tool"] == "communicate_with_collaborator"
    assert parsed["args"]["message"] == "we can't rule out doublets"


def test_parse_action_handles_captured_prose_block():
    block = (
        "{'tool': 'communicate_with_collaborator', 'args': "
        f"{{'agent_id': 'agent_02', 'message': '{CAPTURED_PROSE}'}}}}"
    )

    parsed = parse_action(_action(block))

    assert parsed["tool"] == "communicate_with_collaborator"
    assert "Cohen’s d" in parsed["args"]["message"]


def test_parse_action_handles_captured_code_block():
    block = f"{{'tool': 'execute_code', 'args': {{'code': '{CAPTURED_CODE}'}}}}"

    parsed = parse_action(_action(block))

    assert parsed["tool"] == "execute_code"
    assert 'complex_I": [re.compile(r"^NDUF")]' in parsed["args"]["code"]


def test_parse_action_accepts_strict_json():
    parsed = parse_action(_action('{"tool": "arxiv", "args": {"query": "renal"}}'))

    assert parsed == {"tool": "arxiv", "args": {"query": "renal"}}


# --- mixed straight and curly quotes (regression) ---------------------
def test_parse_action_accepts_smart_delimiters_mixed_with_ascii_strings():
    # Smart quotes as delimiters for some strings, ASCII for others.
    block = '{“tool”: “execute_code”, "args": {"code": "print(\'ok\')"}}'

    assert parse_action(_action(block)) == {
        "tool": "execute_code",
        "args": {"code": "print('ok')"},
    }


def test_parse_action_accepts_smart_single_delimiters_with_ascii_code():
    block = "{‘tool’: ‘execute_code’, ‘args’: {‘code’: \"df['x'].mean()\"}}"

    assert parse_action(_action(block)) == {
        "tool": "execute_code",
        "args": {"code": "df['x'].mean()"},
    }


def test_parse_action_keeps_smart_quotes_in_ascii_delimited_prose():
    block = (
        '{"tool": "communicate_with_collaborator", "args": '
        '{"agent_id": "agent_02", "message": "Cohen’s d vs “others” holds",},}'
    )

    parsed = parse_action(_action(block))

    assert parsed["args"]["message"] == "Cohen’s d vs “others” holds"


@pytest.mark.parametrize(
    "text",
    [
        # Mismatched curly/ASCII quotes that literal_eval reads as a set of one string.
        "Action: {“tool': 'execute_code', 'args': {'code': 'print(1)'}”}",
        # A valid tool whose args literal_eval reads as a set.
        'Action: {"tool": "execute_code", "args": {\'code”: \'print(1)’}}',
        # The marker fallback passes bare values through.
        'Action: "execute_code"',
        "Action: 42",
        "Action: [1, 2]",
        "Action: {'tool': 'execute_code', 'args': None}",
    ],
    ids=["top-level-set", "args-set", "bare-string", "bare-int", "bare-list", "args-none"],
)
def test_parse_action_returns_the_invalid_marker_for_non_action_values(text):
    act = parse_action(text)
    assert act is not None
    assert act["tool"] == "__invalid_action__"


def test_parse_action_keeps_an_action_without_args():
    assert parse_action('Action: {"tool": "search_agent_registry"}') == {
        "tool": "search_agent_registry"
    }


def test_parse_action_keeps_string_args_for_generic_tools():
    # LangChain passes a string to the tool's single input, as the source allowed.
    assert parse_action('Action: {"tool": "arxiv", "args": "kidney cancer"}') == {
        "tool": "arxiv",
        "args": "kidney cancer",
    }


def test_string_args_are_unusable_only_for_tools_that_read_named_fields():
    assert _unusable_args("execute_code", "print(1)")
    assert _unusable_args("establish_collaboration", "agent_02")
    assert not _unusable_args("arxiv", "kidney cancer")
    assert not _unusable_args("execute_code", {"code": "print(1)"})


@pytest.mark.parametrize(
    "text",
    [
        "Action: {“tool': 'arxiv', 'args': {'query': 'kidney'}”}",
        "Action: {'tool': 'arxiv', 'args': None}",
    ],
    ids=["top-level-set", "args-none"],
)
def test_collab_parse_action_returns_the_invalid_marker_for_non_action_values(text):
    act = collab_parse_action(text)
    assert act is not None and act["tool"] == "__invalid_action__"


def test_collab_parse_action_keeps_actions():
    assert collab_parse_action('Action: {"tool": "arxiv", "args": "kidney cancer"}') == {
        "tool": "arxiv",
        "args": "kidney cancer",
    }
    assert collab_parse_action('Action: {"tool": "arxiv", "args": {"query": "kidney"}}') == {
        "tool": "arxiv",
        "args": {"query": "kidney"},
    }


def test_an_action_whose_tool_is_not_a_string_is_invalid():
    text = 'Action: {"tool": ["arxiv"], "args": {"query": "kidney"}}'
    assert parse_action(text)["tool"] == "__invalid_action__"
    assert collab_parse_action(text)["tool"] == "__invalid_action__"
