"""Prompt builders and output parsers for the ranker agent."""

import ast
import json
import re
from collections.abc import Callable
from typing import Any

# ---------------------------------------------------------------------------
# Section notes defaults
# ---------------------------------------------------------------------------

DEFAULT_SECTION_NOTES: dict[str, str] = {
    "preamble": "Each paper is presented with the following sections:",
    "manuscript": "A Manuscript section containing the paper's title, abstract, and full text.",
    "include_dataset_description": "A Dataset Description section describing the dataset(s) used in the research.",
    "include_code": "A Code section showing the implementation produced by the research agent.",
    "include_cited_papers": "A Cited Papers section listing previously published papers that the research builds upon.",
}


def build_section_notes_block(
    *,
    include_code: bool = True,
    include_dataset_description: bool = True,
    include_cited_papers: bool = True,
    section_notes: dict[str, str] | None = None,
) -> str:
    """Build a bullet-list block describing which sections each paper contains.

    Args:
        include_code: Whether the code section is enabled.
        include_dataset_description: Whether the dataset description section is enabled.
        include_cited_papers: Whether the cited papers section is enabled.
        section_notes: Optional overrides merged on top of DEFAULT_SECTION_NOTES.

    Returns:
        Formatted string with preamble and bullet items.
    """
    notes = {**DEFAULT_SECTION_NOTES}
    if section_notes:
        notes.update(section_notes)

    lines = [notes["preamble"]]

    # Emit in user-message section order:
    # dataset description -> manuscript -> code -> cited papers
    if include_dataset_description:
        lines.append(f"- {notes['include_dataset_description']}")
    lines.append(f"- {notes['manuscript']}")
    if include_code:
        lines.append(f"- {notes['include_code']}")
    if include_cited_papers:
        lines.append(f"- {notes['include_cited_papers']}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# System prompt construction
# ---------------------------------------------------------------------------


def make_ranker_system_prompt(
    criteria_prompt: str,
    tool_signatures: dict[str, str] | None = None,
    section_notes_block: str | None = None,
) -> str:
    """Build the system prompt for the ranker agent.

    Args:
        criteria_prompt: The evaluation criteria text injected into the prompt.
        tool_signatures: Optional mapping of tool name to human-readable
            signature string. When non-empty, tool usage instructions are
            included in the prompt.
        section_notes_block: Optional pre-built block describing the sections
            present in each paper. Inserted between criteria and verdict format.

    Returns:
        Complete system prompt string.
    """
    parts = [
        "You are an expert scientific judge. You will be given two research papers,",
        'labelled "Paper A" and "Paper B". Your task is to determine which paper is',
        "better according to the evaluation criteria below.",
        "",
        criteria_prompt.strip(),
    ]

    # Tool instructions (only if at least one tool is provided)
    if tool_signatures:
        parts.append("")
        parts.append("You have access to the following tools to help inform your judgement.")
        parts.append("You are NOT required to use them -- only use them if you believe they")
        parts.append("would meaningfully improve your comparison.")
        parts.append("")
        for name, sig in tool_signatures.items():
            parts.append(sig)

    # Section notes (describes what sections each paper contains)
    if section_notes_block:
        parts.append("")
        parts.append(section_notes_block)

    parts.append("")
    parts.append("You may take multiple reasoning steps before reaching your verdict.")
    parts.append("When you are ready to deliver your final judgement, use this exact format:")
    parts.append("")
    parts.append("Verdict:")
    parts.append("- Winner: <A or B>")
    parts.append("- Justification: <your reasoning>")
    parts.append("")
    parts.append("Constraints:")
    parts.append("- You MUST eventually produce a Verdict block.")
    parts.append('- The Winner field must be exactly "A" or "B" (nothing else).')
    parts.append("- The Justification must reference specific aspects of both papers.")

    if tool_signatures:
        parts.append("- If using tools, use this format for tool calls:")
        parts.append('  Action: {"tool": "<tool_name>", "args": {<arguments>}}')
        parts.append("- Only output one Action block per response.")
        parts.append("- Never mix Action blocks with Verdict blocks in the same response.")

    return "\n".join(parts)


def make_hypothesis_similarity_system_prompt(
    criteria_prompt: str,
    tool_signatures: dict[str, str] | None = None,
    section_notes_block: str | None = None,
) -> str:
    """Build the system prompt for hypothesis similarity mode.

    Args:
        criteria_prompt: Criteria text describing what counts as "the same
            hypothesis".
        tool_signatures: Optional mapping of tool name to human-readable
            signature string.
        section_notes_block: Optional pre-built block describing the sections
            present in each paper.

    Returns:
        Complete system prompt string.
    """
    parts = [
        "You are an expert scientific judge. You will be given two research papers,",
        'labelled "Paper A" and "Paper B". Your task is to determine whether the two',
        "papers test the same hypothesis or different hypotheses, according to the",
        "criteria below.",
        "",
        criteria_prompt.strip(),
    ]

    # Tool instructions (only if at least one tool is provided)
    if tool_signatures:
        parts.append("")
        parts.append("You have access to the following tools to help inform your judgement.")
        parts.append("You are NOT required to use them -- only use them if you believe they")
        parts.append("would meaningfully improve your comparison.")
        parts.append("")
        for name, sig in tool_signatures.items():
            parts.append(sig)

    # Section notes (describes what sections each paper contains)
    if section_notes_block:
        parts.append("")
        parts.append(section_notes_block)

    parts.append("")
    parts.append("You may take multiple reasoning steps before reaching your verdict.")
    parts.append("When you are ready to deliver your final judgement, use this exact format:")
    parts.append("")
    parts.append("Verdict:")
    parts.append("- Decision: <Same or Different>")
    parts.append("- Justification: <your reasoning>")
    parts.append("")
    parts.append("Constraints:")
    parts.append("- You MUST eventually produce a Verdict block.")
    parts.append('- The Decision field must be exactly "Same" or "Different" (nothing else).')
    parts.append("- The Justification must reference specific aspects of both papers' hypotheses.")

    if tool_signatures:
        parts.append("- If using tools, use this format for tool calls:")
        parts.append('  Action: {"tool": "<tool_name>", "args": {<arguments>}}')
        parts.append("- Only output one Action block per response.")
        parts.append("- Never mix Action blocks with Verdict blocks in the same response.")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# User message construction
# ---------------------------------------------------------------------------


def _format_dataset_description_section(
    dataset_descriptions: dict[str, str] | None,
) -> str:
    """Format the dataset description section for a single paper.

    Args:
        dataset_descriptions: Mapping of dataset key to description text,
            or None if no descriptions are available.

    Returns:
        Formatted section string (without trailing newline).
    """
    parts = ["## Dataset Description"]
    if not dataset_descriptions:
        parts.append("No dataset description available for this paper.")
    else:
        for idx, (key, desc) in enumerate(dataset_descriptions.items(), start=1):
            parts.append(f"Dataset {idx}: {key}")
            parts.append(desc.strip())
            parts.append("")  # blank line between datasets
        # Remove trailing blank line
        if parts and parts[-1] == "":
            parts.pop()
    return "\n".join(parts)


def _format_cited_papers_section(
    cited_papers: list[dict[str, str]],
    *,
    include_code: bool = True,
    cited_papers_order: str = "newest_first",
) -> str:
    """Format the Cited Papers section for a single paper.

    Args:
        cited_papers: Ordered list of ancestor paper dicts (oldest first,
            as returned by build_citation_chains).
        include_code: Whether to include code sub-sections for cited papers
            that have code.
        cited_papers_order: ``"newest_first"`` (default) or ``"oldest_first"``.
            Controls the presentation order and the intro line wording.

    Returns:
        Formatted section string (without trailing newline).
    """
    parts = ["## Cited Papers"]

    if not cited_papers:
        parts.append("")
        parts.append("This paper does not cite any other papers.")
        return "\n".join(parts)

    # Apply ordering: input is always oldest-first from build_citation_chains.
    if cited_papers_order == "newest_first":
        ordered = list(reversed(cited_papers))
        order_label = "newest first"
    else:
        ordered = list(cited_papers)
        order_label = "oldest first"

    n = len(ordered)
    parts.append("")
    parts.append(
        f"This paper's research builds on {n} previously published papers (listed {order_label}):"
    )

    for idx, cp in enumerate(ordered, start=1):
        cp_id = cp.get("paper_id", "unknown")
        parts.append("")
        parts.append(f"### Cited Paper {idx} of {n} ({cp_id})")
        parts.append("")
        parts.append(f"Title: {cp.get('title', 'N/A')}")
        parts.append(f"Abstract: {cp.get('abstract', 'N/A')}")
        parts.append(cp.get("manuscript", "N/A"))

        # Code sub-section: only if include_code=True AND paper has code
        if include_code and cp.get("code"):
            parts.append("")
            parts.append("#### Code")
            parts.append(cp["code"])

    return "\n".join(parts)


def _format_single_paper(
    paper: dict[str, str],
    label: str,
    *,
    include_code: bool = True,
    include_dataset_description: bool = True,
    dataset_descriptions: dict[str, str] | None = None,
    include_cited_papers: bool = True,
    cited_papers: list[dict[str, str]] | None = None,
    cited_papers_order: str = "newest_first",
) -> str:
    """Format a single paper's sections for the user message.

    Sections are emitted in order:
    Dataset Description -> Manuscript -> Code -> Cited Papers.

    Args:
        paper: Dict with keys "title", "abstract", "manuscript", and
            optionally "code".
        label: Paper label, e.g. "Paper A".
        include_code: Whether to include the code section.
        include_dataset_description: Whether to include the dataset section.
        dataset_descriptions: Mapping of dataset key to description text.
        include_cited_papers: Whether to include the cited papers section.
        cited_papers: Ordered list of ancestor paper dicts. None or empty
            list means no cited papers.
        cited_papers_order: ``"newest_first"`` or ``"oldest_first"``.

    Returns:
        Formatted paper string.
    """
    parts = [f"# {label}"]

    # Dataset Description section
    if include_dataset_description:
        parts.append("")
        parts.append(_format_dataset_description_section(dataset_descriptions))

    # Manuscript section
    parts.append("")
    parts.append("## Manuscript")
    parts.append(f"Title: {paper.get('title', 'N/A')}")
    parts.append(f"Abstract: {paper.get('abstract', 'N/A')}")
    parts.append(paper.get("manuscript", "N/A"))

    # Code section
    if include_code:
        parts.append("")
        parts.append("## Code")
        if paper.get("code"):
            parts.append(paper["code"])
        else:
            parts.append("No code was provided for this paper.")

    # Cited Papers section
    if include_cited_papers:
        parts.append("")
        parts.append(
            _format_cited_papers_section(
                cited_papers or [],
                include_code=include_code,
                cited_papers_order=cited_papers_order,
            )
        )

    return "\n".join(parts)


_DEFAULT_PAPER_SEPARATOR = "---\n---\n---"


def make_ranker_user_message(
    paper_a: dict[str, str],
    paper_b: dict[str, str],
    *,
    include_code: bool = True,
    include_dataset_description: bool = True,
    dataset_descriptions_a: dict[str, str] | None = None,
    dataset_descriptions_b: dict[str, str] | None = None,
    include_cited_papers: bool = True,
    cited_papers_a: list[dict[str, str]] | None = None,
    cited_papers_b: list[dict[str, str]] | None = None,
    cited_papers_order: str = "newest_first",
    paper_separator: str | None = None,
) -> str:
    """Build the user message presenting two papers for comparison.

    Args:
        paper_a: Dict with keys "title", "abstract", "manuscript", and
            optionally "code".
        paper_b: Same structure as paper_a.
        include_code: Whether to include the code section for each paper.
        include_dataset_description: Whether to include the dataset
            description section for each paper.
        dataset_descriptions_a: Mapping of dataset key to description text
            for Paper A.
        dataset_descriptions_b: Same for Paper B.
        include_cited_papers: Whether to include cited papers sections.
        cited_papers_a: Ordered list of ancestor paper dicts for Paper A.
        cited_papers_b: Ordered list of ancestor paper dicts for Paper B.
        cited_papers_order: ``"newest_first"`` or ``"oldest_first"``.
        paper_separator: Visual separator between Paper A and Paper B.
            Defaults to a triple horizontal rule (``---\\n---\\n---``).

    Returns:
        Formatted user message string.
    """
    section_a = _format_single_paper(
        paper_a,
        "Paper A",
        include_code=include_code,
        include_dataset_description=include_dataset_description,
        dataset_descriptions=dataset_descriptions_a,
        include_cited_papers=include_cited_papers,
        cited_papers=cited_papers_a,
        cited_papers_order=cited_papers_order,
    )
    section_b = _format_single_paper(
        paper_b,
        "Paper B",
        include_code=include_code,
        include_dataset_description=include_dataset_description,
        dataset_descriptions=dataset_descriptions_b,
        include_cited_papers=include_cited_papers,
        cited_papers=cited_papers_b,
        cited_papers_order=cited_papers_order,
    )

    separator = (paper_separator or _DEFAULT_PAPER_SEPARATOR).strip()
    return f"{section_a}\n\n{separator}\n\n{section_b}"


# ---------------------------------------------------------------------------
# Nudge message
# ---------------------------------------------------------------------------

NUDGE_MESSAGE = (
    "Your previous response could not be parsed as a valid verdict.\n"
    "Please provide your final judgement using this exact format:\n\n"
    "Verdict:\n"
    "- Winner: <A or B>\n"
    "- Justification: <your reasoning>\n\n"
    'The Winner must be exactly "A" or "B".'
)


# ---------------------------------------------------------------------------
# Verdict parsing (multi-strategy)
# ---------------------------------------------------------------------------

# Primary: structured Verdict block with Winner: A|B
_VERDICT_RE = re.compile(
    r"(?i)(?:#*\s*)?verdict\s*:\s*\n\s*-?\s*winner\s*:\s*(A|B)\b",
)

# Fallback 1: Winner: A|B anywhere in text
_WINNER_RE = re.compile(r"(?i)\bwinner\s*:\s*(A|B)\b")

# Fallback 2: natural-language patterns
_NL_PATTERNS = [
    re.compile(r"(?i)\bpaper\s+(A|B)\s+is\s+(?:better|stronger|superior)\b"),
    re.compile(r"(?i)\bpaper\s+(A|B)\s+wins\b"),
    re.compile(r"(?i)\bI\s+choose\s+(A|B)\b"),
    re.compile(r"(?i)\b(A|B)\s+is\s+the\s+(?:better|stronger|superior)\s+paper\b"),
    re.compile(r"(?i)\bprefer\s+paper\s+(A|B)\b"),
    re.compile(
        r"(?i)\bpaper\s+(A|B)\s+(?:is|demonstrates|shows)\s+(?:clearly\s+)?(?:better|stronger|more)\b"
    ),
]

# Justification extraction
_JUSTIFICATION_RE = re.compile(
    r"(?i)-?\s*justification\s*:\s*(.*)",
    re.DOTALL,
)


def _extract_winner(text: str) -> str | None:
    """Extract 'A' or 'B' from text using fallback chain.

    Returns:
        'A' or 'B' if found, else None.
    """
    # Strategy 1: structured Verdict block
    m = _VERDICT_RE.search(text)
    if m:
        return m.group(1).upper()

    # Strategy 2: Winner: X anywhere
    m = _WINNER_RE.search(text)
    if m:
        return m.group(1).upper()

    # Strategy 3: natural-language patterns
    for pat in _NL_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1).upper()

    return None


def _extract_justification(text: str) -> str:
    """Extract justification text, falling back to full response."""
    m = _JUSTIFICATION_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def parse_verdict(text: str) -> tuple[dict[str, Any] | None, bool]:
    """Parse a verdict from LLM output.

    Uses a multi-strategy fallback chain:
    1. Structured Verdict block with Winner/Justification lines.
    2. Standalone Winner: A|B anywhere.
    3. Natural language patterns (e.g. "Paper A is better").

    Args:
        text: Raw LLM response text.

    Returns:
        Tuple of (result_dict, is_valid). result_dict has keys "score"
        (1 for A, 0 for B) and "justification" (str). If parsing fails,
        returns (None, False).
    """
    if not text:
        return None, False

    winner = _extract_winner(text)
    if winner is None:
        return None, False

    score = 1 if winner == "A" else 0
    justification = _extract_justification(text)

    return {"score": score, "justification": justification}, True


# ---------------------------------------------------------------------------
# Hypothesis similarity nudge message
# ---------------------------------------------------------------------------

HYPOTHESIS_SIMILARITY_NUDGE_MESSAGE = (
    "Your previous response could not be parsed as a valid verdict.\n"
    "Please provide your final judgement using this exact format:\n\n"
    "Verdict:\n"
    "- Decision: <Same or Different>\n"
    "- Justification: <your reasoning>\n\n"
    'The Decision must be exactly "Same" or "Different".'
)


# ---------------------------------------------------------------------------
# Hypothesis similarity verdict parsing (multi-strategy)
# ---------------------------------------------------------------------------

# Strategy 1: structured Verdict block with Decision: Same|Different
_HS_VERDICT_RE = re.compile(
    r"(?i)(?:#*\s*)?verdict\s*:\s*\n\s*-?\s*decision\s*:\s*(Same|Different)\b",
)

# Strategy 2: standalone Decision: Same|Different anywhere
_HS_DECISION_RE = re.compile(r"(?i)\bdecision\s*:\s*(Same|Different)\b")

# Strategy 3: natural language patterns
_HS_NL_PATTERNS = [
    re.compile(r"(?i)\b(?:the\s+)?hypotheses\s+are\s+(the\s+same|different)\b"),
    re.compile(r"(?i)\b(?:the\s+)?papers\s+test\s+(the\s+same|different)\s+hypothes"),
    re.compile(r"(?i)\b(same|different)\s+hypothes[ei]s\b"),
    re.compile(r"(?i)\bhypothes[ei]s\s+(?:are|is)\s+(the\s+same|different)\b"),
]


def _extract_hs_decision(text: str) -> str | None:
    """Extract 'same' or 'different' from text using fallback chain.

    Returns:
        Normalized 'same' or 'different' if found, else None.
    """
    # Strategy 1: structured Verdict block
    m = _HS_VERDICT_RE.search(text)
    if m:
        raw = m.group(1).lower()
        return "same" if "same" in raw else "different"

    # Strategy 2: Decision: X anywhere
    m = _HS_DECISION_RE.search(text)
    if m:
        raw = m.group(1).lower()
        return "same" if "same" in raw else "different"

    # Strategy 3: natural language patterns
    for pat in _HS_NL_PATTERNS:
        m = pat.search(text)
        if m:
            raw = m.group(1).lower()
            return "same" if "same" in raw else "different"

    return None


def parse_hypothesis_similarity_verdict(
    text: str,
) -> tuple[dict[str, Any] | None, bool]:
    """Parse a hypothesis similarity verdict from LLM output.

    Uses a multi-strategy fallback chain:
    1. Structured Verdict block with Decision: Same|Different.
    2. Standalone Decision: Same|Different anywhere.
    3. Natural language patterns (e.g. "hypotheses are the same").

    Args:
        text: Raw LLM response text.

    Returns:
        Tuple of (result_dict, is_valid). result_dict has keys "score"
        (0 for Same, 1 for Different) and "justification" (str). If parsing
        fails, returns (None, False).
    """
    if not text:
        return None, False

    decision = _extract_hs_decision(text)
    if decision is None:
        return None, False

    score = 0 if decision == "same" else 1
    justification = _extract_justification(text)

    return {"score": score, "justification": justification}, True


# ---------------------------------------------------------------------------
# Mode-based dispatch helpers
# ---------------------------------------------------------------------------


def get_system_prompt_builder(mode: str) -> Callable:
    """Return the system prompt builder function for the given mode."""
    if mode == "comparison":
        return make_ranker_system_prompt
    if mode == "hypothesis_similarity":
        return make_hypothesis_similarity_system_prompt
    raise ValueError(f"Unknown ranker mode: {mode!r}")


def get_verdict_parser(mode: str) -> tuple[Callable, str]:
    """Return (parse_fn, nudge_message) for the given mode."""
    if mode == "comparison":
        return parse_verdict, NUDGE_MESSAGE
    if mode == "hypothesis_similarity":
        return parse_hypothesis_similarity_verdict, HYPOTHESIS_SIMILARITY_NUDGE_MESSAGE
    raise ValueError(f"Unknown ranker mode: {mode!r}")


# ---------------------------------------------------------------------------
# Action parsing (reimplemented from reviewer agent for independence)
# ---------------------------------------------------------------------------

ACTION_HDR_RE = re.compile(r"(?i)\baction\s*:\s*", re.DOTALL)


def _extract_json_like_block(text: str) -> str | None:
    """Extract the first balanced JSON-like {...} block following an Action: header."""
    text = (text or "").strip()

    # Strip markdown code fences
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1 :]
        if text.endswith("```"):
            text = text[:-3].rstrip()

    m = ACTION_HDR_RE.search(text)
    if not m:
        return None
    rest = text[m.end() :].lstrip()

    brace = 0
    start = None
    in_single = False
    in_double = False
    escape = False
    for i, ch in enumerate(rest):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            continue
        if in_single or in_double:
            continue
        if ch == "{":
            if brace == 0:
                start = i
            brace += 1
        elif ch == "}":
            brace -= 1
            if brace == 0 and start is not None:
                return rest[start : i + 1]
    return None


def parse_action(text: str) -> dict[str, Any] | None:
    """Extract an action dict from a tool call string in model output.

    Tries json.loads() first, then falls back to ast.literal_eval().

    Args:
        text: Raw model response text expected to contain an
            ``Action: { ... }`` block.

    Returns:
        Parsed action dict if found and valid; otherwise None.
    """
    block = _extract_json_like_block(text)
    if not block:
        return None
    try:
        return json.loads(block)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(block)
    except Exception:
        return None
