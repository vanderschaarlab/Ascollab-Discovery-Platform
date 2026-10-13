import pytest

from agents.research.collab_agent import parse_action as collab_parse_action
from agents.research.research_team import _extract_code_block, parse_action


def test_parse_action_accepts_fenced_json_and_trailing_commas():
    action = parse_action(
        """```json
Action: {"tool": "execute_code", "args": {"code": "print(1)",},}
```"""
    )

    assert action == {"tool": "execute_code", "args": {"code": "print(1)"}}


def test_parse_action_stops_before_observation_marker():
    action = parse_action(
        "Action: {'tool': 'search_agent_registry', 'args': {'query_text': 'rna'}}\n"
        "Observation: should not be parsed"
    )

    assert action["tool"] == "search_agent_registry"


def test_parse_action_with_unbalanced_payload_returns_invalid_marker():
    action = parse_action("Action: {'tool': 'execute_code', 'args': {'code': 'print(1)'}")

    assert action["tool"] == "__invalid_action__"
    assert "raw" in action["args"]


def test_extract_code_block_returns_first_python_block():
    assert _extract_code_block("text\n```python\nprint('ok')\n```\n") == "print('ok')"
    assert _extract_code_block("no code") is None


# A real GPT-5 final report (live smoke test, 2026-10-13) had "Cox with interaction: ..."; the
# substring "action:" made the fallback read the rest of the report as a malformed action.
REPORT_WITH_INTERACTION = (
    "Final Report: # Title\nmTORC1 activity and survival\n\n# Empirical Evidence\n"
    "- Cox with interaction: EPAS1_z protective (p=4.6e-5), EPAS1xpS6 not significant (p=0.78).\n"
)


@pytest.mark.parametrize("parse", [parse_action, collab_parse_action], ids=["research", "collab"])
@pytest.mark.parametrize("word", ["interaction", "fraction", "transaction"])
def test_a_word_ending_in_action_is_not_an_action_header(parse, word):
    assert parse(REPORT_WITH_INTERACTION.replace("interaction", word)) is None


@pytest.mark.parametrize("parse", [parse_action, collab_parse_action], ids=["research", "collab"])
def test_an_action_after_such_a_word_is_still_found(parse):
    action = parse(
        "Cox with interaction: not significant.\n"
        'Action: {"tool": "search_internal_archive", "args": {"query_text": "EPAS1"}}'
    )

    assert action == {"tool": "search_internal_archive", "args": {"query_text": "EPAS1"}}


@pytest.mark.parametrize("parse", [parse_action, collab_parse_action], ids=["research", "collab"])
def test_an_unparseable_block_after_a_real_header_is_still_invalid(parse):
    action = parse("Action: run the survival model next\nObservation: none")

    assert action["tool"] == "__invalid_action__"
