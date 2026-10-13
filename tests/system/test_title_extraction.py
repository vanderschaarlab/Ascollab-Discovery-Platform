"""Paper titles are found in every form research agents write them."""

import re

import pytest

from agents.research.primary_agent_utils import make_primary_agent_system_prompt
from system.research_output_utils import extract_title_and_abstract
from system.schema import AgentNetworkMetadata, AgentProfile

BODY = (
    "# Summary\nA short summary.\n\n"
    "# Research Question\nDoes X predict survival?\n\n"
    "# Hypothesis and Key Findings\nX predicts survival.\n\n"
    "# Methods\nCox model.\n"
)
TITLE = "Pathway activity predicts survival in renal carcinoma"


@pytest.mark.parametrize(
    "header",
    [
        f"# Title\n{TITLE}\n\n",
        f"# Title: {TITLE}\n\n",
        f"# Title {TITLE}\n\n",
        f"## Title\n{TITLE}\n\n",
        f"# TITLE: {TITLE}\n\n",
        f"Title: {TITLE}\n\n",
        f"**Title:** {TITLE}\n\n",
        f"**Title**: {TITLE}\n\n",
        f"# **Title**\n{TITLE}\n\n",
        f"# **Title:** {TITLE}\n\n",
        f"# Title: **{TITLE}**\n\n",
        f"# Title - {TITLE}\n\n",
        f"# Title of the study\n{TITLE}\n\n",
        f"## Title and Abstract\n{TITLE}\n\n",
        f"# {TITLE}\n\n",
        f"# **{TITLE}**\n\n",
        f"# Titles of prior work\n\n# {TITLE}\n\n",
        f"```markdown\n# Title\n{TITLE}\n```\n\n",
    ],
    ids=[
        "heading-then-line",
        "heading-colon",
        "heading-inline",
        "level-2-heading",
        "upper-case",
        "label",
        "bold-label",
        "bold-closed-before-colon",
        "bold-heading-then-line",
        "bold-heading-colon",
        "bold-inline-title",
        "heading-dash",
        "title-of-heading-then-line",
        "title-and-heading-then-line",
        "first-level-1-heading",
        "bold-first-level-1-heading",
        "titles-heading-skipped-by-fallback",
        "markdown-fence",
    ],
)
def test_title_forms(header):
    title, abstract = extract_title_and_abstract(header + BODY)
    assert title == TITLE
    assert abstract == "Does X predict survival?\n\nX predicts survival."


def test_report_sections_are_not_mistaken_for_a_title():
    title, _ = extract_title_and_abstract(BODY)
    assert title == ""


def test_title_word_inside_a_heading_is_not_a_title():
    """'# Titles of ...' is neither a title heading nor the fallback title."""
    title, _ = extract_title_and_abstract("# Titles of prior work\nA list.\n\n" + BODY)
    assert title == ""


@pytest.mark.parametrize(
    "report",
    [
        "```python\n# Load data\ndf = load()\n```\n\n" + BODY,
        BODY + "\n```python\n# title for the plot\nplt.title('x')\n```\n",
        BODY + "\n```python\n# Title\nplt.title('x')\n```\n",
        "```python\ntitle: str = 'abc'\n```\n\n" + BODY,
        "~~~\n# Load data\n~~~\n\n" + BODY,
    ],
    ids=["comment-fallback", "comment-title-words", "comment-title", "code-label", "tilde-fence"],
)
def test_code_blocks_never_give_a_title(report):
    title, abstract = extract_title_and_abstract(report)
    assert title == ""
    assert abstract == "Does X predict survival?\n\nX predicts survival."


def test_code_block_does_not_hide_the_real_title():
    report = f"# Title\n{TITLE}\n\n```python\n# Title for the plot\n```\n\n" + BODY
    assert extract_title_and_abstract(report)[0] == TITLE


def test_whole_report_in_a_bare_fence_is_read_as_text():
    report = f"```\n# Title\n{TITLE}\n\n{BODY}```"
    assert extract_title_and_abstract(report)[0] == TITLE


def test_title_words_without_a_following_line_or_capital_are_not_a_title():
    assert extract_title_and_abstract("# Title of the study\n\n" + BODY)[0] == ""
    assert extract_title_and_abstract("## Title and Abstract\n\n" + BODY)[0] == ""


def test_title_line_in_a_body_section_is_not_the_title():
    report = BODY + "\n# Results and Analysis\nTitle: Kaplan Meier plot\n## Title: Hazard ratios\n"
    assert extract_title_and_abstract(report)[0] == ""


def test_level_1_title_heading_counts_anywhere():
    """The source form: a '# Title' section after other sections still gives the title."""
    assert extract_title_and_abstract(BODY + f"\n# Title\n{TITLE}\n")[0] == TITLE


def test_first_title_source_wins():
    report = f"Title: {TITLE}\n\n# Title\nA later title\n\n" + BODY
    assert extract_title_and_abstract(report)[0] == TITLE


def _rendered_template_headings() -> set[str]:
    profile = AgentProfile(
        agent_id="agent_01",
        behavior="careful",
        expertise="omics",
        expertise_topics=["omics"],
        citation_count=0,
        num_accepted_papers=0,
    )
    headings: set[str] = set()
    for mode in (
        {},
        {"nature_style_papers_prompt_enabled": True},
        {"meta_research_mode": True, "meta_research_context": {}},
    ):
        prompt = make_primary_agent_system_prompt(
            agent_profile=profile,
            agent_network_metadata=AgentNetworkMetadata(num_agents=4),
            iteration=1,
            tools={},
            **mode,
        )
        headings.update(re.findall(r"(?m)^#\s+(.+?)\s*$", prompt))
    return headings - {"Title"}


def test_template_section_headings_are_never_the_title():
    headings = _rendered_template_headings()
    assert {"Rationale/Mechanism", "Data and Code Availability", "Coherent Synthesis"} <= headings
    for heading in sorted(headings):
        assert extract_title_and_abstract(f"# {heading}\nText.\n")[0] == "", heading


def test_template_report_without_a_title_gets_no_title():
    """Section headings, from the templates or not, never win after the first section."""
    report = (
        "# Summary\nS.\n\n# Research Question\nQ?\n\n# Hypothesis and Key Findings\nH.\n\n"
        "# Rationale/Mechanism\nR.\n\n# Empirical Evidence\nE.\n\n# Kaplan-Meier analysis\nK.\n"
    )
    assert extract_title_and_abstract(report) == ("", "Q?\n\nH.")


def test_empty_report():
    assert extract_title_and_abstract("") == ("", "")


@pytest.mark.parametrize(
    "report",
    [
        f"```\n# Title\n{TITLE}\n\n{BODY}",  # cut off before the closing fence
        f"```text\n# Title\n{TITLE}\n\n{BODY}```",
        f"```\n# Title\n{TITLE}\n\n{BODY}```\nLet me know if you need changes.",
        f"```\n# Title: {TITLE}\n\n{BODY}```\n",
    ],
    ids=["unclosed-fence", "text-fence", "fence-then-text", "fenced-inline-heading"],
)
def test_title_inside_an_untagged_or_text_fence_is_found(report):
    # The first pass reads these fences as code; with no title found, they are read as text
    # for the explicit title forms, as the source read every line.
    assert extract_title_and_abstract(report)[0] == TITLE


def test_bold_markers_under_a_title_heading_are_dropped():
    assert extract_title_and_abstract(f"# Title\n**{TITLE}**\n\n" + BODY)[0] == TITLE


def test_title_label_in_an_untagged_fence_is_not_a_title():
    # The second pass reads fences as the source did: title headings only.
    assert extract_title_and_abstract("```\ntitle: str = 'abc'\n```\n\n" + BODY)[0] == ""
