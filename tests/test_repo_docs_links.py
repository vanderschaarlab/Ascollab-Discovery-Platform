"""Relative links in the Markdown docs point at files in the repository, and at headings that exist."""

import os
import re
from pathlib import Path

from tests.isolation import tracked_files

REPO_ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*#*\s*$")
HTML_ANCHOR = re.compile(r"<a\s+(?:id|name)=\"([^\"]+)\"")


def _tracked() -> set[str]:
    return set(tracked_files(REPO_ROOT))


def _prose_lines(doc: str):
    """Yield (line number, line) outside fenced code blocks."""
    in_code_block = False
    for lineno, line in enumerate((REPO_ROOT / doc).read_text(encoding="utf-8").splitlines(), 1):
        if line.lstrip().startswith("```"):
            in_code_block = not in_code_block
            continue
        if not in_code_block:
            yield lineno, line


def _slug(heading: str) -> str:
    """GitHub's anchor for a heading: link text kept, punctuation dropped, spaces to hyphens."""
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading).strip().lower()
    return re.sub(r"[^\w\- ]", "", text).replace(" ", "-")


def _anchors(doc: str) -> set[str]:
    anchors: set[str] = set()
    seen: dict[str, int] = {}
    for _, line in _prose_lines(doc):
        anchors.update(HTML_ANCHOR.findall(line))
        match = HEADING.match(line)
        if match:
            slug = _slug(match.group(1))
            count = seen.get(slug, 0)
            seen[slug] = count + 1
            anchors.add(slug if count == 0 else f"{slug}-{count}")
    return anchors


def _links():
    """Yield (doc, line number, target) for every relative link in a tracked Markdown file."""
    for doc in sorted(f for f in _tracked() if f.endswith(".md")):
        for lineno, line in _prose_lines(doc):
            for target in LINK.findall(line):
                if not target.startswith(("http://", "https://", "mailto:")):
                    yield doc, lineno, target


def test_relative_markdown_links_resolve_to_tracked_paths():
    tracked = _tracked()
    tracked_dirs = {str(Path(f).parent) for f in tracked} | {
        str(parent) for f in tracked for parent in Path(f).parents
    }
    broken = []
    for doc, lineno, target in _links():
        if target.startswith("#"):
            continue
        path = os.path.normpath(os.path.join(os.path.dirname(doc), target.split("#")[0]))
        if path not in tracked and path not in tracked_dirs:
            broken.append(f"{doc}:{lineno} -> {target}")

    assert broken == []


def test_markdown_link_anchors_match_a_heading():
    tracked = _tracked()
    broken = []
    for doc, lineno, target in _links():
        if "#" not in target:
            continue
        path, anchor = target.split("#", 1)
        path = os.path.normpath(os.path.join(os.path.dirname(doc), path)) if path else doc
        if path.endswith(".md") and path in tracked and anchor not in _anchors(path):
            broken.append(f"{doc}:{lineno} -> {target}")

    assert broken == []


def test_heading_slugs_follow_github():
    assert _slug("Resume a stopped run") == "resume-a-stopped-run"
    assert (
        _slug("`network.*` and [models](models.md): an overview")
        == "network-and-models-an-overview"
    )
    assert _slug("A/B tests (v2)") == "ab-tests-v2"
