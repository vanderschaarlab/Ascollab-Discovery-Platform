import ast
import re

from langchain_core.messages import AIMessage, BaseMessage

_ACTION_LINE_RE = re.compile(r"(?im)^\s*(Action|Observation)\s*:\s*.*$")
_ANGLE_PLACEHOLDER_RE = re.compile(r"<[A-Za-z0-9_][^>\n]{0,80}>")
_METHOD_CLAIM_SENTINEL_RE = re.compile(r"(?<!\w)[MC]<id>(?!\w)")


def _strip_action_observation_lines(text: str) -> tuple[str, bool]:
    if not text:
        return "", False
    lines = text.splitlines()
    kept = []
    stripped = False
    for line in lines:
        if _ACTION_LINE_RE.match(line):
            stripped = True
            continue
        kept.append(line)
    return "\n".join(kept).strip(), stripped


def extract_final_report(messages: list[BaseMessage]) -> str:
    """Extract the final report text from a conversation history."""
    final_report_re = re.compile(r"(?im)^Final\s+Report\s*:\s*", re.MULTILINE)
    thought_re = re.compile(r"(?im)^Thought\s*:\s*", re.MULTILINE)
    tool_chatter_re = re.compile(r"(?im)^\s*(Action|Observation)\s*:\s*")

    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            content = msg.content.strip()

            matches = list(final_report_re.finditer(content))
            if matches:
                last_match = matches[-1]
                report_content = content[last_match.end() :].strip()
                if report_content:
                    report_content, _ = _strip_action_observation_lines(report_content)
                    return report_content

            # Skip tool chatter messages that don't contain a final report.
            if tool_chatter_re.match(content):
                continue

            thought_match = thought_re.search(content)
            if thought_match:
                remaining_content = content[thought_match.end() :].strip()
                if remaining_content:
                    remaining_content, _ = _strip_action_observation_lines(remaining_content)
                    return remaining_content
            if content:
                content, _ = _strip_action_observation_lines(content)
                return content

    return ""


def sanitize_report(report: str) -> tuple[str, bool]:
    """Strip Action/Observation lines from report content."""
    return _strip_action_observation_lines(report)


def find_unresolved_template_placeholders(text: str) -> list[str]:
    """Return unresolved template placeholders found in report text.

    Detects placeholder literals commonly used in style templates, such as:
    - angle placeholders: `<dataset_key_1>`, `<paper_id>`, `<id>`
    - method/claim sentinels: `M<id>`, `C<id>`
    """
    if not text:
        return []

    findings: list[str] = []
    seen: set[str] = set()

    for match in _ANGLE_PLACEHOLDER_RE.finditer(text):
        token = match.group(0).strip()
        if token and token not in seen:
            seen.add(token)
            findings.append(token)

    for match in _METHOD_CLAIM_SENTINEL_RE.finditer(text):
        token = match.group(0).strip()
        if token and token not in seen:
            seen.add(token)
            findings.append(token)

    return findings


def has_unresolved_template_placeholders(text: str) -> bool:
    """Boolean helper for unresolved template placeholder detection."""
    return bool(find_unresolved_template_placeholders(text))


def extract_final_script(raw_script: str) -> str:
    """Strip inline image markers from a generated script for brevity."""
    if not raw_script:
        return ""

    image_line_pattern = re.compile(r"^# Image \d+: .+$", re.MULTILINE)
    image_matches = list(image_line_pattern.finditer(raw_script))

    if not image_matches:
        return raw_script.strip()

    lines = raw_script.split("\n")
    result_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]
        if image_line_pattern.match(line):
            image_count = 0

            while i < len(lines) and image_line_pattern.match(lines[i]):
                image_count += 1
                i += 1

            if image_count == 1:
                result_lines.append(
                    "# {Image successfully generated, removed from script for brevity concerns...}"
                )
            else:
                result_lines.append(
                    f"# {{{image_count} images successfully generated, removed from script for brevity concerns...}}"
                )
        else:
            result_lines.append(line)
            i += 1

    return "\n".join(result_lines).strip()


# A title heading at any level, after bold markers are removed: "# Title" (title on the lines
# below), "# Title: <t>" / "# Title - <t>" (with a separator: <t> is in "text"), or
# "# Title <t>" (no separator: <t> is in "words").
_TITLE_HEADING_RE = re.compile(
    r"^(?P<hashes>#{1,6})\s*title"
    r"(?:\s*(?P<sep>:|[-\u2013\u2014](?=\s|$))\s*(?P<text>.*?)|\s+(?P<words>.*?))?\s*$",
    re.IGNORECASE,
)
# "Title: <t>", after bold markers are removed ("**Title:** <t>", "**Title**: <t>").
_TITLE_LABEL_RE = re.compile(r"^title\s*:\s*(.+?)\s*$", re.IGNORECASE)
# A heading starting with the word "Title" or "Titles" is never the fallback title.
_TITLE_WORD_RE = re.compile(r"^titles?\b", re.IGNORECASE)
_HEADING_RE = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<text>.+?)\s*#*\s*$")
# A code fence: ``` or ~~~ (three or more) and an optional info string such as "python".
_FENCE_RE = re.compile(r"^(?P<marker>`{3,}|~{3,})\s*(?P<info>[^\s`]*)")
_MARKDOWN_FENCE_INFO = {"markdown", "md"}
# Fences the second pass reads as report text: untagged, text or markdown (never a language).
_TEXT_FENCE_INFO = _MARKDOWN_FENCE_INFO | {"", "text", "txt", "plain", "plaintext"}
# Section headings of the research report templates in make_primary_agent_system_prompt
# (agents/research/primary_agent_utils.py: default, nature-style and meta-research), plus a
# few common ones. They are never the paper's title. tests/system/test_title_extraction.py
# checks that every level-1 heading of the rendered templates is listed.
_REPORT_SECTIONS = frozenset(
    {
        "summary",
        "abstract",
        "research question",
        "research questions",
        "hypothesis and key findings",
        "hypothesis and key finding",
        "rationale/mechanism",
        "results and analysis",
        "empirical evidence",
        "methods",
        "literature evidence",
        "formal methods",
        "data and code availability",
        "references",
        "discussion",
        "limitations",
        "conclusion",
        "conclusions",
        # meta-research synthesis sections
        "cross-paper themes",
        "claim-evidence table",
        "support and validity checks",
        "contradictions in public archive",
        "contradictions and resolution",
        "coherent synthesis",
        "evidence map",
        "bounded final claims",
        "executable claim checks",
        "reproducibility notes",
    }
)


def _is_report_section(heading_text: str) -> bool:
    return " ".join(heading_text.split()).rstrip(":").strip().lower() in _REPORT_SECTIONS


def _fence_closes(line: str, marker: str) -> bool:
    return line.startswith(marker) and not line.strip(marker[0])


def _wrapping_fence_lines(lines: list[str]) -> set[int]:
    """Return the indices of a bare code fence that wraps the whole report, if any."""
    text_rows = [i for i, line in enumerate(lines) if line.strip()]
    if len(text_rows) < 2:
        return set()
    first, last = text_rows[0], text_rows[-1]
    opening = _FENCE_RE.match(lines[first].strip())
    if opening is None or opening.group("info"):
        return set()
    if not _fence_closes(lines[last].strip(), opening.group("marker")):
        return set()
    return {first, last}


def extract_title_and_abstract(report: str) -> tuple[str, str]:
    """Parse the title and abstract-like sections from a structured report.

    Title sources, in any case and with or without ``**`` bold markers (which never end
    up in the title):

    - a title heading: ``# Title`` / ``# Title:`` with the title on the lines below it,
      or ``# Title: <t>`` / ``# Title - <t>`` (also en and em dash) with the title inline.
      ``# Title <t>`` without a separator is ambiguous (``# Title of the study``,
      ``## Title and Abstract``): ``<t>`` is the title only when it does not start with a
      lowercase letter and no text line follows before the next heading;
      otherwise the heading reads as ``# Title`` and the text below it is the title.
    - a ``Title: <t>`` line outside any title section.
    - failing all of these, the first level-1 heading that is not a report section of
      the research report templates and does not start with the word "Title(s)".

    The front matter is the part of the report before its first report-section heading
    (``# Summary``, ``# Research Question``, ...). ``Title:`` lines, title headings below
    level 1 and the fallback heading count only in the front matter, so a figure title
    in ``# Results and Analysis`` is never the paper title; a level-1 title heading
    counts anywhere. Lines in fenced code blocks (``` or ~~~) never give a title; fences
    tagged ``markdown``/``md``, or a bare fence around the whole report, hold report
    text and are read as such. If that finds no title (e.g. a report wrapped in a
    ```` ```text ```` fence, or cut off before its closing fence), untagged and text fences
    are read as text too, for the title headings only: ``# Title`` with the title below (the
    one form the source read, in fences too) and ``# Title: <t>``; no ``# Title <t>``,
    ``Title: <t>`` line or fallback heading.
    Fences tagged with a language (``python``, ...) stay code. When there are several title
    sources, the first one in the report wins.

    The abstract joins the research-question and hypothesis sections, or falls back to
    the whole report.

    Args:
        report: The final report text (Markdown).

    Returns:
        ``(title, abstract)``; the title is ``""`` when the report has none.
    """
    title, abstract = _parse_title_and_abstract(report, read_fences=False)
    if not title:
        # Without fences the second pass finds nothing the first did not.
        title, _ = _parse_title_and_abstract(report, read_fences=True)
    return title, abstract


def _parse_title_and_abstract(report: str, *, read_fences: bool) -> tuple[str, str]:
    """Parse as `extract_title_and_abstract` describes.

    With ``read_fences`` untagged and text fences are read as report text, and only the title
    headings count (no ``# Title <t>``, no ``Title:`` line and no fallback heading).
    """
    if not report:
        return "", ""

    title = ""
    abstract_parts = []
    fallback = ""
    pending = ""  # the <t> of a "# Title <t>" heading, used if no text follows it
    in_front_matter = True
    seen_text = False  # any text line under the current title heading
    code_fence = ""  # marker of the open code fence, "" outside one
    text_fence = ""  # marker of an open markdown fence, "" outside one

    lines = report.split("\n")
    wrapper = _wrapping_fence_lines(lines)
    current_section = None
    current_content: list[str] = []

    def flush() -> None:
        nonlocal title, pending, seen_text
        if current_section and current_content:
            if current_section == "title":
                title = title or "\n".join(current_content).replace("**", "").strip()
            elif current_section in ["research_question", "hypothesis"]:
                abstract_parts.append("\n".join(current_content).strip())
        if current_section == "title" and not seen_text:
            title = title or pending
        pending = ""
        seen_text = False

    for index, line in enumerate(lines):
        line_stripped = line.strip()

        # Fence lines and code lines never give a title.
        in_code = True
        if code_fence:
            if _fence_closes(line_stripped, code_fence):
                code_fence = ""
        elif index in wrapper:
            pass
        elif text_fence and _fence_closes(line_stripped, text_fence):
            text_fence = ""
        elif fence := _FENCE_RE.match(line_stripped):
            text_infos = _TEXT_FENCE_INFO if read_fences else _MARKDOWN_FENCE_INFO
            if not text_fence and fence.group("info").lower() in text_infos:
                text_fence = fence.group("marker")
            else:
                code_fence = fence.group("marker")
        else:
            in_code = False

        plain = "" if in_code else line_stripped.replace("**", "")
        heading = _HEADING_RE.match(plain)
        if heading and _is_report_section(heading.group("text")):
            in_front_matter = False
        title_heading = _TITLE_HEADING_RE.match(plain)
        if title_heading and len(title_heading.group("hashes")) > 1 and not in_front_matter:
            title_heading = None
        if title_heading and read_fences and title_heading.group("words"):
            title_heading = None

        if title_heading:
            flush()
            inline = (title_heading.group("text") or "").strip()
            words = (title_heading.group("words") or "").strip()
            if inline:
                title = title or inline
                current_section = None
            else:
                current_section = "title"
                if words and not words[0].islower():
                    pending = words
            current_content = []

        elif re.match(r"^#\s*research\s+questions?\s*$", line_stripped, re.IGNORECASE):
            flush()
            current_section = "research_question"
            current_content = []

        elif re.match(r"^#\s*hypothesis\s+and\s+key\s+findings?\s*$", line_stripped, re.IGNORECASE):
            flush()
            current_section = "hypothesis"
            current_content = []

        elif line_stripped.startswith("#"):
            flush()
            if (
                heading
                and len(heading.group("hashes")) == 1
                and in_front_matter
                and not fallback
                and not read_fences
                and not _TITLE_WORD_RE.match(heading.group("text"))
            ):
                fallback = heading.group("text")
            current_section = None
            current_content = []

        else:
            label = None if read_fences else _TITLE_LABEL_RE.match(plain)
            if label and in_front_matter and current_section is None:
                title = title or label.group(1)
            elif current_section == "title" and in_code:
                pass
            elif current_section:
                current_content.append(line)
                if line_stripped:
                    seen_text = True

    flush()
    if not title:
        title = fallback

    abstract = ""
    if abstract_parts:
        non_empty_parts = [part for part in abstract_parts if part]
        if non_empty_parts:
            abstract = "\n\n".join(non_empty_parts)

    if not abstract:
        abstract = report.strip()

    return title, abstract


def extract_internal_citations(report: str) -> list[dict]:
    """Extract internal archive citations from a report body."""
    if not report:
        return []

    citations = []
    seen_paper_ids = set()  # Track seen paper_ids for deduplication

    citation_patterns = [
        re.compile(
            r"-\s*\[\s*internal\s+archive\s*\]\s*(\{[^}]+\})",
            re.MULTILINE | re.DOTALL | re.IGNORECASE,
        ),
    ]

    all_matches = []
    for pattern in citation_patterns:
        matches = pattern.findall(report)
        all_matches.extend(matches)

    for match in all_matches:
        try:
            dict_str = match.strip()
            citation_dict = ast.literal_eval(dict_str)

            if isinstance(citation_dict, dict):
                paper_id = citation_dict.get("paper_id")

                if paper_id and paper_id not in seen_paper_ids:
                    citations.append(citation_dict)
                    seen_paper_ids.add(paper_id)

        except (SyntaxError, ValueError, NameError):
            continue

    return citations
