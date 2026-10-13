import json
import re
from typing import Any

from env_setup.config_factory import ToolInfo
from system.citation_tags import (
    STRICT_CITATION_SCHEMA_PROMPT_BLOCK,
    validate_citation_tags,
)
from system.citation_tags import (
    review_citation_schema_nudge as _review_citation_schema_nudge,
)
from system.review_metrics import normalize_metric_allowlist, review_metric_catalog
from system.schema import AgentNetworkMetadata, AgentProfile

_METRIC_DESCRIPTIONS: dict[str, str] = {
    # Support
    "support.external_papers_read": "integer count (>= 0); higher is better.",
    "support.successful_code_executions": "integer count (>= 0); higher is better.",
    "support.lines_of_code": "integer count (>= 0); higher often implies more analysis, but quality matters.",
    "support.evidence_types_present": "list/array of modality labels; more distinct types is better.",
    "support.claim_evidence_alignment_max_cosine": "cosine similarity in [0, 1]; higher is better.",
    "support.num_statistical_code_cells": "integer count (>= 0); higher is better.",
    "support.evidence_breadth_index": "aggregate breadth diagnostic; higher is generally better.",
    "support.analytical_depth": "aggregate depth diagnostic; higher is generally better.",
    # Soundness
    "soundness.internal_consistency_score": "0-1; higher is better.",
    "soundness.num_statistical_validity_checks_passed": "integer count (>= 0); higher is better relative to total.",
    "soundness.num_statistical_validity_checks_total": "integer count (>= 0); used to compute pass rate.",
    "soundness.diagnostic_validity_pass_rate": "0-1; higher is better.",
    # Significance
    "significance.distance_to_prior_mean": "distance metric (>= 0); higher implies more novelty, compare within-run.",
    "significance.distance_to_prior_min": "distance metric (>= 0); higher implies more novelty, compare within-run.",
    "significance.distance_to_prior_max": "distance metric (>= 0); higher implies more novelty, compare within-run.",
    "significance.clinical_biological_relevance": "0-1; higher is better.",
    "significance.clinical_relevance_score": "0-1; alias of clinical_biological_relevance; higher is better.",
    "significance.effect_size_strength": "0-1; higher is better.",
    # Originality
    "originality.embedding_novelty_percentile": "0-100; higher is more novel.",
    "originality.hypothesis_overlap_jaccard": "0-1; lower is more novel.",
    "originality.semantic_novelty_knn_distance": ">= 0; higher is more novel.",
    "originality.concept_pair_rarity": "higher indicates rarer concept combinations.",
    "originality.concept_entropy": "higher indicates broader concept mixture.",
    "originality.methodological_novelty_flag": "True/False; True indicates novelty.",
    "originality.mechanistic_novelty_flag": "True/False; True indicates novelty.",
}


def _render_community_norms_block(community_norms: str) -> str:
    """Render the reviewer-norm instruction (experiment 09) as a clearly delimited block.

    The block is injected verbatim into every reviewer prompt (reviewer-level only).
    It does not alter the review output schema, rating scales, or metric guidance.
    """
    return f"""
====================
Community reviewing norms:
{community_norms}
====================
"""


def _build_metrics_guidance(metrics_allowlist: list[str] | None) -> str:
    review_metric_catalog()
    allowlist_by_category, unknown = normalize_metric_allowlist(metrics_allowlist)
    if unknown:
        unknown_txt = f"\nNote: Unknown metrics in allowlist were ignored: {', '.join(sorted(set(unknown)))}\n"
    else:
        unknown_txt = ""

    lines = [
        "You will ALSO be given a JSON object called METRICS for each paper.",
        "This object contains precomputed, quantitative diagnostics derived from:",
        "- tool usage (literature queries, Europe PMC, internal archive),",
        "- sandboxed code execution logs,",
        "- embedding-based similarity/novelty analyses,",
        "- and internal consistency checks.",
        "",
        "You MUST use these METRICS as the primary basis for the numeric ratings",
        "(Support, Soundness, Significance, Originality, Overall Recommendation),",
        "and only adjust based on your qualitative judgement where strongly warranted.",
        "",
        "Only the metrics listed below are enabled for this run. Do not assume",
        "that any other metrics exist. If a category has no metrics, rely on",
        "qualitative judgment and explicitly note the missing signal.",
        "",
    ]
    if unknown_txt:
        lines.append(unknown_txt.rstrip())

    order = [
        ("Support", "support"),
        ("Soundness", "soundness"),
        ("Significance", "significance"),
        ("Originality", "originality"),
    ]
    for label, category in order:
        keys = list(allowlist_by_category.get(category, set()))
        keys.sort()
        if not keys:
            lines.append(f"- {label}: (no metrics enabled)")
            continue
        lines.append(f"- {label} metrics:")
        for key in keys:
            full_key = f"{category}.{key}"
            desc = _METRIC_DESCRIPTIONS.get(full_key, "See METRICS JSON for values.")
            lines.append(f"  - metrics.{full_key}: {desc}")

    lines.extend(
        [
            "",
            "Decision rationale requirement:",
            "- In your overall recommendation justification, explicitly reference the METRICS values",
            "  you used (include key metric names and values, not just qualitative paraphrases).",
            "- If a metric value is null/missing, do not invent a value; rely on available metrics",
            "  and state that a signal was missing.",
        ]
    )

    return "\n".join(lines).strip()


def validate_review_citation_schema(
    text: str,
    *,
    require_assertion_justification: bool,
) -> tuple[bool, list[str]]:
    if not require_assertion_justification:
        return True, []
    verdict = validate_citation_tags(
        text or "",
        require_at_least_one=True,
    )
    errors = [str(err) for err in verdict.get("errors", [])]
    return bool(verdict.get("is_valid", False)), errors


def review_citation_schema_nudge(errors: list[str] | None = None) -> str:
    return _review_citation_schema_nudge(errors=errors)


def make_reviewer_system_prompt(
    agent_profile: AgentProfile,
    agent_network_metadata: AgentNetworkMetadata,
    tools: ToolInfo | None = None,
    require_assertion_justification: bool = False,
    metrics_allowlist: list[str] | None = None,
    enforce_code_review_reference: bool = False,
    community_norms: str | None = None,
    include_metrics_guidance: bool = True,
) -> str:
    """Build the system prompt for a reviewer agent.

    Assembles a structured instruction string for an LLM acting as a scientific
    reviewer. The prompt explains the reviewer's task, the Action/Observation
    protocol, lists any available tools with their signatures, and specifies the
    evaluation criteria, quantitative ratings rubric, and mandatory output
    sections.

    Args:
        agent_profile: Profile describing the reviewer (expertise, behavior, etc.).
        agent_network_metadata: Metadata about the broader research network (e.g., number of agents).
        tools: Optional tool(s) available to the reviewer; if provided, their signatures are included.
        require_assertion_justification: If true, require every assertion to be justified with archive/arxiv/code evidence.
        metrics_allowlist: Metrics the guidance lists; None lists every metric.
        enforce_code_review_reference: If true, require a [Code Review] reference in the review.
        community_norms: Reviewer-norm text injected as its own block (experiment 09); None for none.
        include_metrics_guidance: Whether the prompt describes the metrics and asks the reviewer
            to ground ratings in them. ``system.main`` turns it off when
            ``review.metrics_pipeline`` is none/off/disabled (no metrics are computed).

    Returns:
        str: A complete system prompt string suitable for use as the model's system message.
    """

    weights = getattr(agent_profile, "metric_weights", None) or {}
    weight_parts = []
    for label, key in (
        ("Support", "support"),
        ("Soundness", "soundness"),
        ("Significance", "significance"),
        ("Originality", "originality"),
    ):
        if key in weights:
            try:
                weight_parts.append(f"{label}={float(weights[key]):.2f}")
            except Exception:
                continue
    if not weight_parts and weights:
        weight_parts = [f"{k}={v}" for k, v in weights.items()]
    weighting_text = ""
    if weight_parts:
        weighting_text = f"""
Weighting preference (agent-specific):
- When assigning Support, Soundness, Significance, and Originality (and synthesizing Overall Recommendation), weight categories proportionally as: {", ".join(weight_parts)}.
- You may only deviate from these weightings if the manuscript content overwhelmingly warrants it, and you must note that deviation in the overall justification.
"""

    if include_metrics_guidance:
        metrics_text = f"""{_build_metrics_guidance(metrics_allowlist)}

You may adjust one point up/down from what the metrics alone suggest if your
qualitative reading of the report strongly supports this, but you must always
explain these tensions in the text (especially in sections 3, 4, and 6).

Overall score justification requirement:
- After the Quantitative Ratings section, add 1-2 sentences that explicitly
  justify the Overall Recommendation by citing the metric values you used
  (e.g., "Overall=3 because support.evidence_breadth_index=2, soundness.internal_consistency_score=0.55, originality.hypothesis_overlap_jaccard=0.62").
  Keep it concise and metric-grounded."""
    else:
        metrics_text = """Overall score justification requirement:
- After the Quantitative Ratings section, add 1-2 sentences that explicitly
  justify the Overall Recommendation.
  Keep it concise."""

    prompt = """
You are an expert scientific researcher. Your task is to critically evaluate research reports.
"""

    if community_norms:
        prompt += _render_community_norms_block(community_norms)

    prompt += """
Input:
You will be provided with:
- A report describing the research problem, motivation, methods, results.
- The code used to generate the results (if available).
- Structured plot artefacts (when available), including plot citation tokens, filenames, and statistical digests.

Use the following format for tool use only:
Action: {'tool': <tool name>, 'args': <the arguments to the tool as a json object>}

Final Review: <your final report summarizing your hypothesis>

Constraints:
- Only output Action blocks in tool-use steps; do not mix Action with other content in the same response.
- Never include Action or Observation text in the Final Review.
"""
    if tools:
        prompt += """
You can use the following tools:
"""
        for _, tool in tools.items():
            tool_signature = tool.tool_signature
            prompt += f"""{tool_signature}
"""

    if weighting_text:
        prompt += weighting_text

    if require_assertion_justification:
        prompt += f"""
Assertion justification requirement (strict):
- Every assertion you make must be grounded in one of:
  - findings from the public archive,
  - real-world papers retrieved via the arxiv tool,
  - or code/results you have seen in the report.
- For every assertion, include a citation tag that conforms exactly to this schema:
{STRICT_CITATION_SCHEMA_PROMPT_BLOCK}
- If you cannot ground a statement, do not assert it; instead, state the uncertainty
  and note the missing evidence source.
"""
    if enforce_code_review_reference:
        prompt += """
Code review reference requirement (strict):
- You must explicitly reference the code review summary using the tag [Code Review]
  at least once in the final review text.
"""

    prompt += f"""
Evaluation Criteria:
1. Summary:
Briefly summarize the report (including the main findings, main results, etc. that the report claims to contribute).

2. Motivation:
- What is the specific question and/or problem tackled by the report?

3. Claims and Evidence:
- Are the main claims of the report clearly stated? Are these claims supported by sufficient reasoning, data, or theoretical analysis?

4. Soundness of Methodology:
- Are the methods and/or analyses and/or evaluation metrics appropriate for the problem?

5. Relation to Prior Knowledge:
- How are the key contributions of the report related to the broader scientific literature? Be specific in terms of prior related findings/results/ideas/etc.

6. Novelty and Significance:
- What is the significance of the work? Does it contribute new knowledge and sufficient value to the community?

7. Other Comments
- If you have any other comments or suggestions, please write them here.

# Quantitative Ratings
Use these to summarize your written evaluations. Respond with a numeric rating for each category (integers or 0.5 steps).

- Support: How well are the claims supported by empirical evidence, reasoning, and/or logical consistency with prior knowledge?
4 = Excellent | 3 = Good | 2 = Fair | 1 = Poor

- Soundness: How technically sound and scientifically rigorous is the work?
4 = Excellent | 3 = Good | 2 = Fair | 1 = Poor

- Significance: How much does the work advance knowledge or practice in the field?
4 = Excellent | 3 = Good | 2 = Fair | 1 = Poor

- Originality: How novel are the ideas, methods, or results?
4 = Excellent | 3 = Good | 2 = Fair | 1 = Poor

- Overall Recommendation:
5: Strong accept
4: Accept
3: Weak accept (i.e., leaning towards accept, but could also be rejected)
2: Weak reject (i.e., leaning towards reject, but could also be accepted)
1: Reject

Output Format (mandatory sections):

# Review Text
1. Summary
(Your summary of the report.)
2. Motivation
(Your evaluation of the motivation.)
3. Claims and Evidence
(Your evaluation of the claims and evidence.)
4. Soundness of Methodology
(Your evaluation of the methodology.)
5. Relation to Prior Knowledge
(Your evaluation of the relation to prior knowledge.)
6. Novelty and Significance
(Your evaluation of the novelty and significance.)
7. Other Comments
(Any other comments or suggestions.)

# Quantitative Ratings
- Support: <number>
- Soundness: <number>
- Significance: <number>
- Originality: <number>
- Overall Recommendation: <number>
{metrics_text}

Instructions:
- Respond to all 7 qualitative sections in order (1-7).
- Provide concise, well-structured text under each section.
- Provide numerical ratings for each quantitative category at the end.
"""

    return prompt.strip()


def make_reviewer_user_message(
    report_text: str,
    code_report: str | None = None,
    metrics: dict[str, Any] | None = None,
    plot_artifacts: list[dict[str, Any]] | None = None,
    meta_research_context: dict[str, Any] | None = None,
) -> str:
    """Compose the user message delivered to the reviewer model.

    Packages the research report and (optionally) a code/results report into a
    single string that will be provided as the user message to the LLM.

    Args:
        report_text: The research report content to review.
        code_report: Optional code/results text associated with the report.

    Returns:
        str: A formatted message containing the report (and code section if provided).
    """
    user_message = f"""
Here is the research report to review:
{report_text}
""".strip()
    if code_report:
        user_message += f"""
Here is the code and results used to generate the report (if applicable):
{code_report}
""".rstrip()

    if isinstance(meta_research_context, dict):
        paper_type = str(meta_research_context.get("paper_type", "") or "").strip()
        expectations = [
            str(x).strip()
            for x in (meta_research_context.get("expectations", []) or [])
            if str(x).strip()
        ]
        source_paper_ids = [
            str(x).strip()
            for x in (meta_research_context.get("source_paper_ids", []) or [])
            if str(x).strip()
        ]
        if paper_type or expectations or source_paper_ids:
            user_message += """

Meta-Research Reviewer Context:
""".rstrip()
            if paper_type:
                user_message += f"\n- Paper type: {paper_type}"
            if expectations:
                user_message += "\n- Additional evaluation expectations:"
                for item in expectations:
                    user_message += f"\n  - {item}"
            if source_paper_ids:
                user_message += "\n- Declared source accepted-paper IDs:"
                for pid in source_paper_ids[:40]:
                    user_message += f"\n  - {pid}"

    if metrics:
        metrics_block = json.dumps(metrics, indent=2, ensure_ascii=False)
        user_message += f"""

Here are precomputed quantitative METRICS for this paper.
These are computed from the research phase logs, code, and prior literature.
You MUST USE these metrics as the primary basis when assigning the numeric
quantitative ratings (Support, Soundness, Significance, Originality, Overall Recommendation).

The metrics are provided as a JSON object:

```json
{metrics_block}
```
""".rstrip()
    if plot_artifacts:
        compact_rows: list[dict[str, Any]] = []
        for item in list(plot_artifacts)[:12]:
            if not isinstance(item, dict):
                continue
            digest = (
                item.get("statistics_digest", {})
                if isinstance(item.get("statistics_digest", {}), dict)
                else {}
            )
            compact_rows.append(
                {
                    "plot_citation": str(item.get("plot_citation", "")),
                    "plot_handle": str(item.get("plot_handle", "")),
                    "plot_filename": str(item.get("plot_filename", "")),
                    "plot_path": str(item.get("plot_path", "")),
                    "summary_path": str(item.get("summary_path", "")),
                    "summary_excerpt": str(item.get("summary_excerpt", "")),
                    "generation_backend": str(item.get("generation_backend", "")),
                    "statistics_digest": digest,
                }
            )
        if compact_rows:
            user_message += f"""

Here are structured PLOT ARTEFACTS for figures generated during research.
Use these as evidence for plot interpretation; reference plot citations/tokens in text when relevant.

```json
{json.dumps(compact_rows, indent=2, ensure_ascii=False)}
```
""".rstrip()
    return user_message.strip()


def parse_review_text(text: str) -> tuple[dict[str, Any], bool]:
    """Parse a review into structured text and ratings from LLM output.

    Heuristically extracts the "Review Text" section and attempts to locate the
    quantitative ratings (Support, Soundness, Significance, Originality, Overall
    Recommendation) from a variety of header and formatting variants. Ratings are
    validated against allowed integer ranges.

    Args:
        text: Raw LLM output containing the review.

    Returns:
        Tuple[Dict[str, Any], bool]:
            - parsed: A dictionary with keys:
                * "Review Text": str
                * "Support": Optional[float] in 1..4
                * "Soundness": Optional[float] in 1..4
                * "Significance": Optional[float] in 1..4
                * "Originality": Optional[float] in 1..4
                * "Overall Recommendation": Optional[float] in 1..5
            - ok: True if all numeric ratings are present and the review text is non-empty;
              False otherwise.
    """
    parsed: dict[str, Any] = {
        "Review Text": "",
        "Support": None,
        "Soundness": None,
        "Significance": None,
        "Originality": None,
        "Overall Recommendation": None,
    }

    if not text or not isinstance(text, str):
        return parsed, False

    body = text.strip()

    final_review_patterns = [
        r"(?i)(?:^|\n)\s*#+\s*final\s+reviews?\s*:?\s*\n",
        r"(?i)(?:^|\n)\s*final\s+reviews?\s*:?\s*\n",
    ]

    extracted_content = None
    for pattern in final_review_patterns:
        match = re.search(pattern, body)
        if match:
            extracted_content = body[match.end() :].strip()
            break

    if extracted_content is None:
        review_start_patterns = [
            (r"(?i)(?:^|\n)(\s*#\s*review\s+text.*)", "review_text_header"),
            (r"(?i)(?:^|\n)(\s*1\.\s*summary.*)", "numbered_summary"),
            (r"(?i)(?:^|\n)(\s*summary\s*:.*)", "summary_colon"),
        ]

        for pattern, pattern_name in review_start_patterns:
            match = re.search(pattern, body, re.DOTALL)
            if match:
                start_pos = match.start(1) if match.start(1) > 0 else 0
                extracted_content = body[start_pos:].strip()
                break

    if extracted_content is None:
        thought_pattern = r"(?i)(?:^|\n)\s*thought\s*:\s*([^\n]*)"
        thought_matches = list(re.finditer(thought_pattern, body))

        if thought_matches:
            for match in reversed(thought_matches):
                remaining_text = body[match.end() :].strip()
                action_match = re.search(r"(?i)^\s*action\s*:", remaining_text, re.MULTILINE)
                if not action_match:
                    extracted_content = remaining_text
                    break

    if extracted_content is None:
        obs_matches = list(re.finditer(r"(?i)(?:^|\n)\s*observation\s*:", body))
        if obs_matches:
            last_obs = obs_matches[-1]
            after_obs = body[last_obs.end() :].strip()
            thought_match = re.search(r"(?i)(?:^|\n)\s*thought\s*:[^\n]*\n", after_obs)
            if thought_match:
                extracted_content = after_obs[thought_match.end() :].strip()
            else:
                extracted_content = after_obs

    if extracted_content is None:
        extracted_content = body

    review_text = extracted_content
    patterns = [
        r"^\s*#+\s*Quantitative\s+Ratings?\s*$",  # ### Quantitative Rating(s)
        r"^\s*#+\s*Numerical\s+Ratings?\s*$",  # ### Numerical Rating(s)
        r"^\s*#+\s*Ratings?\s*$",  # ### Rating(s)
        r"^\s*#+\s*Scores?\s*$",  # ### Score(s)
        r"^\s*Quantitative\s+Ratings?\s*:?\s*$",  # Quantitative Rating(s):
        r"^\s*Numerical\s+Ratings?\s*:?\s*$",  # Numerical Rating(s):
        r"^\s*Ratings?\s*:?\s*$",  # Rating(s):
    ]

    lines = extracted_content.split("\n")
    split_index = len(lines)  # Default: include all lines

    for i, line in enumerate(lines):
        for pattern in patterns:
            if re.match(pattern, line, re.IGNORECASE):
                split_index = i
                break
        if split_index < len(lines):  # Found a match
            break

    if split_index == len(lines):  # No header found, look for rating lines
        for i, line in enumerate(lines):
            if re.match(
                r"^\s*[-*]?\s*(Support|Soundness|Significance|Originality|Overall\s+Recommendation)\s*:\s*\d+",
                line,
                re.IGNORECASE,
            ):
                split_index = i
                break

    review_text = "\n".join(lines[:split_index]).strip()

    def _grab_score(label: str, lo: float, hi: float) -> float | None:
        pattern = rf"(?mi)^[ \t>*-]*{re.escape(label)}\s*:\s*([0-9]+(?:\.[0-9]+)?)\b"
        m = re.search(pattern, extracted_content)
        if not m:
            return None
        try:
            val = float(m.group(1))
        except ValueError:
            return None
        if val < lo or val > hi:
            return None
        return val

    support = _grab_score("Support", 1, 4)
    soundness = _grab_score("Soundness", 1, 4)
    significance = _grab_score("Significance", 1, 4)
    originality = _grab_score("Originality", 1, 4)
    overall = _grab_score("Overall Recommendation", 1, 5)

    parsed["Support"] = support
    parsed["Soundness"] = soundness
    parsed["Significance"] = significance
    parsed["Originality"] = originality
    parsed["Overall Recommendation"] = overall

    parsed["Review Text"] = review_text

    has_all_ints = all(v is not None for k, v in parsed.items() if k != "Review Text")
    has_text = bool(parsed["Review Text"].strip())

    return parsed, (has_all_ints and has_text)
