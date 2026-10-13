"""Tests for prompt construction."""

from ranker.ranker_utils import (
    DEFAULT_SECTION_NOTES,
    build_section_notes_block,
    make_ranker_system_prompt,
    make_ranker_user_message,
)


class TestSystemPrompt:
    """Test system prompt assembly."""

    def test_system_prompt_no_tools(self):
        criteria = "Compare on novelty and rigor."
        prompt = make_ranker_system_prompt(criteria, tool_signatures=None)

        assert "Verdict:" in prompt
        assert "Winner:" in prompt
        assert "Compare on novelty and rigor" in prompt
        # No tool signatures
        assert "arxiv" not in prompt
        assert "sandbox" not in prompt
        assert "Action:" not in prompt

    def test_system_prompt_with_tools(self):
        criteria = "Compare on novelty and rigor."
        sigs = {
            "arxiv": '- arxiv: args = { "query": str } (Search arXiv)',
            "sandbox": '- sandbox: args = { "code": str } (Execute Python code)',
        }
        prompt = make_ranker_system_prompt(criteria, tool_signatures=sigs)

        assert "Verdict:" in prompt
        assert "arxiv" in prompt
        assert "sandbox" in prompt
        assert "Action:" in prompt

    def test_criteria_injection(self):
        custom = "CUSTOM_CRITERIA_XYZ: Evaluate based on statistical power only."
        prompt = make_ranker_system_prompt(custom)
        assert "CUSTOM_CRITERIA_XYZ" in prompt

    def test_empty_tool_signatures_no_tool_instructions(self):
        criteria = "Compare papers."
        prompt = make_ranker_system_prompt(criteria, tool_signatures={})
        assert "Action:" not in prompt

    def test_section_notes_block_inserted(self):
        criteria = "Compare on novelty."
        prompt = make_ranker_system_prompt(criteria, section_notes_block="SECTION_NOTES_MARKER")
        assert "SECTION_NOTES_MARKER" in prompt
        # Comes after criteria text
        criteria_pos = prompt.index("Compare on novelty")
        marker_pos = prompt.index("SECTION_NOTES_MARKER")
        assert marker_pos > criteria_pos
        # Comes before verdict format
        verdict_pos = prompt.index("You may take multiple reasoning steps")
        assert marker_pos < verdict_pos

    def test_section_notes_block_none(self):
        criteria = "Compare on novelty."
        prompt_with_none = make_ranker_system_prompt(criteria, section_notes_block=None)
        prompt_baseline = make_ranker_system_prompt(criteria)
        assert prompt_with_none == prompt_baseline

    def test_section_notes_block_empty_string(self):
        criteria = "Compare on novelty."
        prompt_empty = make_ranker_system_prompt(criteria, section_notes_block="")
        prompt_baseline = make_ranker_system_prompt(criteria)
        assert prompt_empty == prompt_baseline


class TestBuildSectionNotesBlock:
    """Tests for build_section_notes_block()."""

    def test_all_flags_true_default_notes(self):
        block = build_section_notes_block(
            include_code=True,
            include_dataset_description=True,
            include_cited_papers=True,
        )
        # Preamble present
        assert DEFAULT_SECTION_NOTES["preamble"] in block
        # All four section notes present
        assert DEFAULT_SECTION_NOTES["manuscript"] in block
        assert DEFAULT_SECTION_NOTES["include_dataset_description"] in block
        assert DEFAULT_SECTION_NOTES["include_code"] in block
        assert DEFAULT_SECTION_NOTES["include_cited_papers"] in block
        # Order: dataset description < manuscript < code < cited papers
        dd_pos = block.index(DEFAULT_SECTION_NOTES["include_dataset_description"])
        ms_pos = block.index(DEFAULT_SECTION_NOTES["manuscript"])
        cd_pos = block.index(DEFAULT_SECTION_NOTES["include_code"])
        cp_pos = block.index(DEFAULT_SECTION_NOTES["include_cited_papers"])
        assert dd_pos < ms_pos < cd_pos < cp_pos

    def test_all_flags_false(self):
        block = build_section_notes_block(
            include_code=False,
            include_dataset_description=False,
            include_cited_papers=False,
        )
        assert DEFAULT_SECTION_NOTES["preamble"] in block
        assert DEFAULT_SECTION_NOTES["manuscript"] in block
        # No mention of code, dataset, or cited papers
        assert DEFAULT_SECTION_NOTES["include_code"] not in block
        assert DEFAULT_SECTION_NOTES["include_dataset_description"] not in block
        assert DEFAULT_SECTION_NOTES["include_cited_papers"] not in block

    def test_single_flag_code_only(self):
        block = build_section_notes_block(
            include_code=True,
            include_dataset_description=False,
            include_cited_papers=False,
        )
        assert DEFAULT_SECTION_NOTES["preamble"] in block
        assert DEFAULT_SECTION_NOTES["manuscript"] in block
        assert DEFAULT_SECTION_NOTES["include_code"] in block
        assert DEFAULT_SECTION_NOTES["include_dataset_description"] not in block
        assert DEFAULT_SECTION_NOTES["include_cited_papers"] not in block

    def test_single_flag_dataset_only(self):
        block = build_section_notes_block(
            include_code=False,
            include_dataset_description=True,
            include_cited_papers=False,
        )
        assert DEFAULT_SECTION_NOTES["preamble"] in block
        assert DEFAULT_SECTION_NOTES["include_dataset_description"] in block
        assert DEFAULT_SECTION_NOTES["manuscript"] in block
        assert DEFAULT_SECTION_NOTES["include_code"] not in block
        assert DEFAULT_SECTION_NOTES["include_cited_papers"] not in block

    def test_single_flag_cited_papers_only(self):
        block = build_section_notes_block(
            include_code=False,
            include_dataset_description=False,
            include_cited_papers=True,
        )
        assert DEFAULT_SECTION_NOTES["preamble"] in block
        assert DEFAULT_SECTION_NOTES["manuscript"] in block
        assert DEFAULT_SECTION_NOTES["include_cited_papers"] in block
        assert DEFAULT_SECTION_NOTES["include_code"] not in block
        assert DEFAULT_SECTION_NOTES["include_dataset_description"] not in block

    def test_custom_section_notes_override(self):
        block = build_section_notes_block(
            include_code=True,
            include_dataset_description=False,
            include_cited_papers=False,
            section_notes={"include_code": "CUSTOM_CODE_NOTE"},
        )
        assert "CUSTOM_CODE_NOTE" in block
        assert DEFAULT_SECTION_NOTES["include_code"] not in block

    def test_custom_section_notes_partial_override(self):
        custom_preamble = "CUSTOM_PREAMBLE_TEXT"
        block = build_section_notes_block(
            include_code=True,
            include_dataset_description=True,
            include_cited_papers=True,
            section_notes={"preamble": custom_preamble},
        )
        assert custom_preamble in block
        assert DEFAULT_SECTION_NOTES["preamble"] not in block
        # All other notes use defaults
        assert DEFAULT_SECTION_NOTES["manuscript"] in block
        assert DEFAULT_SECTION_NOTES["include_code"] in block
        assert DEFAULT_SECTION_NOTES["include_dataset_description"] in block
        assert DEFAULT_SECTION_NOTES["include_cited_papers"] in block

    def test_section_order_matches_user_message(self):
        block = build_section_notes_block(
            include_code=True,
            include_dataset_description=True,
            include_cited_papers=True,
        )
        dd_pos = block.index(DEFAULT_SECTION_NOTES["include_dataset_description"])
        ms_pos = block.index(DEFAULT_SECTION_NOTES["manuscript"])
        cd_pos = block.index(DEFAULT_SECTION_NOTES["include_code"])
        cp_pos = block.index(DEFAULT_SECTION_NOTES["include_cited_papers"])
        assert dd_pos < ms_pos < cd_pos < cp_pos


class TestSystemPromptWithSectionNotes:
    """Integration tests: make_ranker_system_prompt with real build_section_notes_block output."""

    def test_full_prompt_all_sections(self):
        notes = build_section_notes_block(
            include_code=True,
            include_dataset_description=True,
            include_cited_papers=True,
        )
        prompt = make_ranker_system_prompt("Compare papers.", section_notes_block=notes)
        # All four section descriptions present
        assert DEFAULT_SECTION_NOTES["manuscript"] in prompt
        assert DEFAULT_SECTION_NOTES["include_dataset_description"] in prompt
        assert DEFAULT_SECTION_NOTES["include_code"] in prompt
        assert DEFAULT_SECTION_NOTES["include_cited_papers"] in prompt
        # Standard verdict format still present
        assert "Verdict:" in prompt
        assert "Winner:" in prompt

    def test_full_prompt_code_disabled(self):
        notes = build_section_notes_block(
            include_code=False,
            include_dataset_description=True,
            include_cited_papers=True,
        )
        prompt = make_ranker_system_prompt("Compare papers.", section_notes_block=notes)
        assert DEFAULT_SECTION_NOTES["include_code"] not in prompt
        assert DEFAULT_SECTION_NOTES["include_dataset_description"] in prompt
        assert DEFAULT_SECTION_NOTES["manuscript"] in prompt
        assert DEFAULT_SECTION_NOTES["include_cited_papers"] in prompt

    def test_full_prompt_no_optional_sections(self):
        notes = build_section_notes_block(
            include_code=False,
            include_dataset_description=False,
            include_cited_papers=False,
        )
        prompt = make_ranker_system_prompt("Compare papers.", section_notes_block=notes)
        assert DEFAULT_SECTION_NOTES["manuscript"] in prompt
        assert DEFAULT_SECTION_NOTES["include_code"] not in prompt
        assert DEFAULT_SECTION_NOTES["include_dataset_description"] not in prompt
        assert DEFAULT_SECTION_NOTES["include_cited_papers"] not in prompt


class TestUserMessage:
    """Test user message formatting."""

    def test_user_message_basic(self):
        paper_a = {
            "title": "Title A",
            "abstract": "Abstract A",
            "manuscript": "Manuscript A",
        }
        paper_b = {
            "title": "Title B",
            "abstract": "Abstract B",
            "manuscript": "Manuscript B",
        }
        msg = make_ranker_user_message(paper_a, paper_b)

        assert "# Paper A" in msg
        assert "# Paper B" in msg
        assert "Title A" in msg
        assert "Title B" in msg
        assert "Abstract A" in msg
        assert "Abstract B" in msg
        assert "Manuscript A" in msg
        assert "Manuscript B" in msg

    def test_user_message_with_code(self):
        paper_a = {
            "title": "Title A",
            "abstract": "Abstract A",
            "manuscript": "Manuscript A",
            "code": "print('hello')",
        }
        paper_b = {
            "title": "Title B",
            "abstract": "Abstract B",
            "manuscript": "Manuscript B",
        }
        msg = make_ranker_user_message(paper_a, paper_b)

        assert "## Code" in msg
        assert "print('hello')" in msg
        # Paper B has no code -- should say "No code was provided"
        paper_b_start = msg.index("# Paper B")
        paper_b_section = msg[paper_b_start:]
        assert "No code was provided for this paper." in paper_b_section

    def test_labels_present(self):
        paper_a = {"title": "A", "abstract": "A", "manuscript": "A"}
        paper_b = {"title": "B", "abstract": "B", "manuscript": "B"}
        msg = make_ranker_user_message(paper_a, paper_b)
        assert "# Paper A" in msg
        assert "# Paper B" in msg
        assert "---" in msg
