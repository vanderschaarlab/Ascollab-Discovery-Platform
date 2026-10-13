"""Expertise replies are read from the <expertise> tags the prompt asks for."""

from agents.create_agents import _parse_expertise_response

# The shape the primary model (GPT-5) gave in the live smoke tests: tags on their own lines.
OWN_LINES = (
    "<expertise>\n"
    "Computational oncologist working on TCGA KIRC.\n"
    "\n"
    "Core knowledge\n"
    "- VHL/HIF signaling and survival modeling.\n"
    "</expertise>"
)
EXPECTED = (
    "- Computational oncologist working on TCGA KIRC.\n"
    "- Core knowledge\n"
    "- VHL/HIF signaling and survival modeling."
)


def test_tags_on_their_own_lines_are_unchanged():
    assert _parse_expertise_response(OWN_LINES) == EXPECTED


def test_a_reply_without_tags_is_unchanged():
    assert _parse_expertise_response("Expert in survival analysis.\n- Cox models") == (
        "- Expert in survival analysis.\n- Cox models"
    )


def test_a_one_line_tagged_reply_loses_its_tags():
    reply = "<expertise>Survival analysis for cancer genomics cohorts.</expertise>"

    assert _parse_expertise_response(reply) == "- Survival analysis for cancer genomics cohorts."


def test_an_opening_tag_on_the_first_content_line_keeps_that_line():
    reply = "<expertise>- Expert in pathway analysis\n- Survival modeling\n</expertise>"

    assert _parse_expertise_response(reply) == "- Expert in pathway analysis\n- Survival modeling"


def test_text_outside_the_tags_is_dropped():
    reply = "Here is the description:\n" + OWN_LINES + "\nLet me know if you need changes."

    assert _parse_expertise_response(reply) == EXPECTED


def test_a_cut_off_reply_keeps_the_text_after_the_opening_tag():
    reply = "<Expertise>Expert in pathway analysis\n- Survival modeling"

    assert _parse_expertise_response(reply) == "- Expert in pathway analysis\n- Survival modeling"
