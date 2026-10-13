"""Tests for the customize-ranker-input feature.

Covers:
- Feature 1: Cited Papers section shown even when paper has no citations
- Feature 2: Configurable cited-papers sort order (newest_first / oldest_first)
- Feature 3: Configurable paper separator
"""

import pytest

from ranker.ranker_utils import make_ranker_user_message

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def paper_a():
    return {
        "title": "Analysis of Gene Expression",
        "abstract": "We analyzed gene expression data.",
        "manuscript": "Introduction: Gene expression analysis...",
        "code": "import pandas as pd\ndf = pd.read_csv('data.csv')",
    }


@pytest.fixture
def paper_b():
    return {
        "title": "Theoretical Framework",
        "abstract": "A theoretical analysis.",
        "manuscript": "Introduction: We present a theoretical framework...",
        "code": "",
    }


@pytest.fixture
def cp_old():
    """Cited paper with publication_t=1 (oldest)."""
    return {
        "title": "Foundational Study",
        "abstract": "We established the foundation.",
        "manuscript": "This is the foundational manuscript.",
        "code": "print('foundation')",
        "paper_id": "paper_t01_agent_02",
        "publication_t": 1,
    }


@pytest.fixture
def cp_mid():
    """Cited paper with publication_t=2 (middle)."""
    return {
        "title": "Early Work",
        "abstract": "Early exploration of the field.",
        "manuscript": "This is the early work manuscript.",
        "code": "",
        "paper_id": "paper_t02_agent_05",
        "publication_t": 2,
    }


@pytest.fixture
def cp_new():
    """Cited paper with publication_t=3 (newest)."""
    return {
        "title": "Intermediate Results",
        "abstract": "Building on prior work.",
        "manuscript": "This is the intermediate manuscript.",
        "code": "x = compute()",
        "paper_id": "paper_t03_agent_01",
        "publication_t": 3,
    }


def _paper_a_section(msg):
    """Extract Paper A's portion of the user message (before the separator)."""
    # Find where Paper B starts
    idx = msg.index("# Paper B")
    return msg[:idx]


# ---------------------------------------------------------------------------
# Feature 1: Cited Papers section shown when paper has no citations
# ---------------------------------------------------------------------------


class TestEmptyCitedPapers:
    def test_empty_list_shows_no_citations_message(self, paper_a, paper_b):
        """Empty cited_papers_a=[] produces the 'does not cite' message."""
        msg = make_ranker_user_message(
            paper_a,
            paper_b,
            include_cited_papers=True,
            cited_papers_a=[],
        )
        section_a = _paper_a_section(msg)
        assert "## Cited Papers" in section_a
        assert "This paper does not cite any other papers." in section_a

    def test_none_citations_shows_no_citations_message(self, paper_a, paper_b):
        """cited_papers_a=None produces the 'does not cite' message."""
        msg = make_ranker_user_message(
            paper_a,
            paper_b,
            include_cited_papers=True,
            cited_papers_a=None,
        )
        section_a = _paper_a_section(msg)
        assert "## Cited Papers" in section_a
        assert "This paper does not cite any other papers." in section_a

    def test_non_empty_citations_no_empty_message(self, paper_a, paper_b, cp_old):
        """Non-empty cited papers don't show the 'does not cite' message."""
        msg = make_ranker_user_message(
            paper_a,
            paper_b,
            include_cited_papers=True,
            cited_papers_a=[cp_old],
        )
        section_a = _paper_a_section(msg)
        assert "## Cited Papers" in section_a
        assert "This paper does not cite any other papers." not in section_a
        assert "1 previously published" in section_a

    def test_section_omitted_when_disabled(self, paper_a, paper_b):
        """include_cited_papers=False omits the section entirely."""
        msg = make_ranker_user_message(
            paper_a,
            paper_b,
            include_cited_papers=False,
            cited_papers_a=[],
        )
        assert "## Cited Papers" not in msg

    def test_both_papers_show_no_citations(self, paper_a, paper_b):
        """Both papers independently show the 'does not cite' message."""
        msg = make_ranker_user_message(
            paper_a,
            paper_b,
            include_cited_papers=True,
            cited_papers_a=[],
            cited_papers_b=[],
        )
        assert msg.count("This paper does not cite any other papers.") == 2


# ---------------------------------------------------------------------------
# Feature 2: Cited-papers sort order
# ---------------------------------------------------------------------------


class TestCitedPapersOrder:
    def test_default_order_is_newest_first(
        self,
        paper_a,
        paper_b,
        cp_old,
        cp_mid,
        cp_new,
    ):
        """Default (newest_first): cp_new title appears before cp_old title."""
        msg = make_ranker_user_message(
            paper_a,
            paper_b,
            include_cited_papers=True,
            # Input is oldest-first (as build_citation_chains returns)
            cited_papers_a=[cp_old, cp_mid, cp_new],
        )
        section_a = _paper_a_section(msg)
        assert "listed newest first" in section_a
        # cp_new should come first, cp_old last
        pos_new = section_a.index(cp_new["title"])
        pos_mid = section_a.index(cp_mid["title"])
        pos_old = section_a.index(cp_old["title"])
        assert pos_new < pos_mid < pos_old

    def test_oldest_first_order(
        self,
        paper_a,
        paper_b,
        cp_old,
        cp_mid,
        cp_new,
    ):
        """oldest_first: cp_old title appears before cp_new title."""
        msg = make_ranker_user_message(
            paper_a,
            paper_b,
            include_cited_papers=True,
            cited_papers_a=[cp_old, cp_mid, cp_new],
            cited_papers_order="oldest_first",
        )
        section_a = _paper_a_section(msg)
        assert "listed oldest first" in section_a
        pos_old = section_a.index(cp_old["title"])
        pos_mid = section_a.index(cp_mid["title"])
        pos_new = section_a.index(cp_new["title"])
        assert pos_old < pos_mid < pos_new

    def test_order_does_not_affect_main_paper(
        self,
        paper_a,
        paper_b,
        cp_old,
    ):
        """Changing cited_papers_order doesn't change the Manuscript/Code sections."""
        msg_new = make_ranker_user_message(
            paper_a,
            paper_b,
            include_cited_papers=True,
            cited_papers_a=[cp_old],
            cited_papers_order="newest_first",
        )
        msg_old = make_ranker_user_message(
            paper_a,
            paper_b,
            include_cited_papers=True,
            cited_papers_a=[cp_old],
            cited_papers_order="oldest_first",
        )
        # Extract just manuscript+code section (between ## Manuscript and ## Cited Papers)
        for msg in (msg_new, msg_old):
            section_a = _paper_a_section(msg)
            ms_start = section_a.index("## Manuscript")
            cited_start = section_a.index("## Cited Papers")
            section_a[ms_start:cited_start]
        # Both should have the same manuscript section
        section_a_new = _paper_a_section(msg_new)
        section_a_old = _paper_a_section(msg_old)
        ms_new = section_a_new[
            section_a_new.index("## Manuscript") : section_a_new.index("## Cited Papers")
        ]
        ms_old = section_a_old[
            section_a_old.index("## Manuscript") : section_a_old.index("## Cited Papers")
        ]
        assert ms_new == ms_old

    def test_config_has_cited_papers_order(self, ranker_cfg):
        """Default config contains cited_papers_order set to newest_first."""
        assert ranker_cfg.ranker.cited_papers_order == "newest_first"


# ---------------------------------------------------------------------------
# Feature 3: Configurable paper separator
# ---------------------------------------------------------------------------


class TestPaperSeparator:
    def test_default_separator_is_triple_rule(self, paper_a, paper_b):
        """Default separator is three consecutive --- lines."""
        msg = make_ranker_user_message(paper_a, paper_b)
        assert "---\n---\n---" in msg

    def test_custom_separator(self, paper_a, paper_b):
        """Custom separator replaces the default."""
        msg = make_ranker_user_message(
            paper_a,
            paper_b,
            paper_separator="=== BREAK ===",
        )
        assert "=== BREAK ===" in msg
        assert "---\n---\n---" not in msg

    def test_multiline_custom_separator(self, paper_a, paper_b):
        """Multiline separator is preserved."""
        msg = make_ranker_user_message(
            paper_a,
            paper_b,
            paper_separator="***\n***",
        )
        assert "***\n***" in msg

    def test_separator_stripped(self, paper_a, paper_b):
        """Trailing newline (from YAML block scalar) is stripped."""
        msg = make_ranker_user_message(
            paper_a,
            paper_b,
            paper_separator="---\n---\n---\n",
        )
        # Should produce the same \n\n...\n\n pattern as default
        assert "\n\n---\n---\n---\n\n" in msg
        # No triple-blank-line gap
        assert "\n\n\n---" not in msg

    def test_config_has_paper_separator(self, ranker_cfg):
        """Default config contains paper_separator whose stripped value is the triple rule."""
        sep = ranker_cfg.ranker.paper_separator.strip()
        assert sep == "---\n---\n---"
