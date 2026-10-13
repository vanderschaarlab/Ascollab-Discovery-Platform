import json
import logging
from typing import Any

from env_setup.config_factory import ToolInfo
from system.citation_tags import STRICT_CITATION_SCHEMA_PROMPT_BLOCK
from system.schema import AgentNetworkMetadata, AgentProfile, PaperMetadataPrivate

log = logging.getLogger(__name__)


def _format_reputation_block(agent_profile: AgentProfile) -> str:
    rep = getattr(agent_profile, "reputation", None) or {}
    score_components = rep.get("score_components", {}) if isinstance(rep, dict) else {}
    score_keys = rep.get("score_keys") if isinstance(rep, dict) else None
    reputation_score = rep.get("reputation_score") if isinstance(rep, dict) else None
    halflife = rep.get("recency_halflife") if isinstance(rep, dict) else None
    diagnostics = rep.get("diagnostics", {}) if isinstance(rep, dict) else {}
    trends = diagnostics.get("score_trends", {}) if isinstance(diagnostics, dict) else {}
    weakest = diagnostics.get("weakest_dimensions", []) if isinstance(diagnostics, dict) else []
    suggested_actions = (
        diagnostics.get("suggested_actions", []) if isinstance(diagnostics, dict) else []
    )
    confidence = diagnostics.get("confidence") if isinstance(diagnostics, dict) else None
    policy_weights = getattr(agent_profile, "metric_weights", None) or {}

    if not score_components:
        return "No prior review scores are available yet."

    keys = score_keys or list(score_components.keys())
    lines = []
    for key in keys:
        entry = score_components.get(key) or {}
        avg = entry.get("avg")
        if avg is None:
            continue
        trend = trends.get(key) if isinstance(trends, dict) else None
        trend_part = ""
        if isinstance(trend, (int, float)):
            trend_part = f" (trend_delta={float(trend):+.3f})"
        lines.append(f"- {key}: {avg}{trend_part}")

    rep_line = (
        f"Overall (recency-weighted) reputation_score: {reputation_score}"
        if reputation_score is not None
        else ""
    )
    hl_line = f"Recency half-life (iterations): {halflife}" if halflife is not None else ""
    weakest_line = (
        f"Weakest dimensions to target this iteration: {', '.join(weakest)}" if weakest else ""
    )
    confidence_line = (
        f"Signal confidence: {float(confidence):.2f}"
        if isinstance(confidence, (int, float))
        else ""
    )
    policy_line = ""
    if policy_weights:
        ordered = ", ".join(
            f"{k}={float(v):.2f}" for k, v in policy_weights.items() if isinstance(v, (int, float))
        )
        if ordered:
            policy_line = f"Current policy weights: {ordered}"
    action_lines = [f"- Action: {a}" for a in suggested_actions if a]
    block = "\n".join(
        [*lines, rep_line, hl_line, confidence_line, weakest_line, policy_line, *action_lines]
    ).strip()
    return block or "No prior review scores are available yet."


def _format_metacognition_feedback_block(feedback: dict[str, Any] | None) -> str:
    if not isinstance(feedback, dict) or not feedback:
        return ""

    source_iteration = feedback.get("source_iteration")
    mode = str(feedback.get("mode", "") or "").strip()
    summary = str(feedback.get("summary", "") or "").strip()
    axes = feedback.get("axes", {}) if isinstance(feedback.get("axes", {}), dict) else {}

    lines = [
        "4.7 Metacognitive Feedback From Previous Round",
        "- Treat this as planning guidance, not as evidence for manuscript claims.",
        "- Use it to choose what to improve this round.",
    ]
    if source_iteration is not None or mode:
        lines.append(f"- Source iteration: {source_iteration}; mode: {mode or 'unknown'}.")
    if summary:
        lines.append(f"Summary: {summary}")

    for axis in ("originality", "significance", "soundness", "support"):
        raw_axis = axes.get(axis, {})
        if not isinstance(raw_axis, dict):
            continue
        axis_lines = []
        for label, key in (
            ("Strengths", "strengths"),
            ("Weaknesses", "weaknesses"),
            ("Actions", "actions"),
        ):
            values = [str(x).strip() for x in (raw_axis.get(key, []) or []) if str(x).strip()]
            if values:
                axis_lines.append(f"  {label}: " + "; ".join(values[:4]))
        if axis_lines:
            lines.append(f"{axis.title()}:")
            lines.extend(axis_lines)

    if mode == "metrics_only":
        metric_definitions = feedback.get("metric_definitions", {})
        raw_metrics = feedback.get("raw_metrics", {})
        if isinstance(metric_definitions, dict) and metric_definitions:
            lines.append("Metric definitions:")
            for category, defs in metric_definitions.items():
                if not isinstance(defs, dict):
                    continue
                compact = "; ".join(f"{name}: {desc}" for name, desc in list(defs.items())[:8])
                if compact:
                    lines.append(f"- {category}: {compact}")
        if isinstance(raw_metrics, dict) and raw_metrics:
            metrics_text = json.dumps(raw_metrics, ensure_ascii=False, default=str)
            if len(metrics_text) > 3500:
                metrics_text = metrics_text[:3497].rstrip() + "..."
            lines.append("Raw metric scores:")
            lines.append(metrics_text)

    return "\n".join(lines).strip()


def _format_prior_papers(papers: list[PaperMetadataPrivate] | None) -> str:
    if not papers:
        return "No prior papers found for you in the private archive."

    lines = []
    for paper in papers:
        title = getattr(paper, "title", "") or "Untitled"
        abstract = getattr(paper, "abstract", "") or "No abstract available."
        manuscript = getattr(paper, "manuscript", "") or "No manuscript available."
        status = getattr(paper, "status", "unknown")
        paper_id = getattr(paper, "paper_id", "") or "unknown"
        manuscript_block = manuscript.replace("\r\n", "\n")
        lines.append(
            f"- {paper_id}: {title}\n"
            f"  Status: {status}\n"
            f"  Abstract: {abstract}\n"
            f"  Manuscript:\n{manuscript_block}"
        )
    return "\n".join(lines)


def _format_prior_paper_summaries(summaries: list[str] | None) -> str:
    if not summaries:
        return "No older papers available to summarize."
    return "\n".join(f"- {s.strip()}" for s in summaries if s and s.strip())


def _format_meta_source_cards(
    source_cards: list[dict[str, Any]] | None, *, max_cards: int = 24
) -> str:
    if not source_cards:
        return "No accepted public-archive papers available for this synthesis pass."

    rows: list[str] = []
    for card in list(source_cards)[: max(1, int(max_cards))]:
        if not isinstance(card, dict):
            continue
        paper_id = str(card.get("paper_id", "") or "unknown").strip()
        title = str(card.get("title", "") or "Untitled").strip()
        abstract = str(card.get("abstract", "") or "").strip()
        if len(abstract) > 500:
            abstract = abstract[:497].rstrip() + "..."
        excerpt = str(card.get("manuscript_excerpt", "") or "").strip()
        if len(excerpt) > 700:
            excerpt = excerpt[:697].rstrip() + "..."
        publication_iteration = card.get("publication_iteration", None)
        citation_count = card.get("citation_count", None)
        rows.append(
            (
                f"- [{paper_id}] {title}\n"
                f"  Publication iteration: {publication_iteration}; citations: {citation_count}\n"
                f"  Abstract: {abstract}\n"
                f"  Excerpt: {excerpt}"
            ).strip()
        )
    return (
        "\n".join(rows)
        if rows
        else "No accepted public-archive papers available for this synthesis pass."
    )


NATURE_STYLE_EXAMPLE_TEMPLATE = """
# Title
<concise finding title with key entity/relationship>

# Summary
<4-6 sentence overview of question, approach, main result, robustness, and implication>

# Research Question
Does <factor_or_method> affect <outcome> in <population_or_context>?

# Hypothesis and Key Findings
We hypothesized that <hypothesis statement>.
Key findings:
1. <finding_1 with metric/effect/p-value> [Code: M<id>].
2. <finding_2 with metric/effect/p-value> [Code: M<id>].
3. <robustness finding> [Code: M<id>].

# Rationale/Mechanism
<brief mechanism or conceptual rationale for why findings are plausible>

# Results and Analysis
<main quantitative comparison with units/effect sizes> [Code: M<id>].
<null/control/permutation result> [Code: M<id>].
<secondary analysis or enrichment result> [Code: M<id>].
<sensitivity/ablation result> [Code: M<id>].

# Empirical Evidence
- <dataset_key_1> analyzed with <method_summary> [Code: M<id>].
- <dataset_key_2> analyzed with <method_summary> [Code: M<id>].
- <artifact/result statement> [Code: M<id>].
- <robustness/control statement> [Code: M<id>].

# Methods
Data sources: <dataset_key_1>, <dataset_key_2>, <dataset_key_3>.
Preprocessing: <named operations; no vague placeholders>.
Analysis: <statistical/modeling procedures>.
Validation: <controls/permutations/sensitivity checks>.

# Literature Evidence
- <prior finding summary> [Archive: <paper_id>] or [Arxiv: <paper title | author | year>].
- <prior finding summary> [Archive: <paper_id>] or [Arxiv: <paper title | author | year>].

# Formal Methods
## M1, M2, ... Method Runs
- M1
  - Data slice/input: <exact slice/window/subset>.
  - Preprocessing: <named operations>.
  - Procedure/model/test: <exact procedure>.
  - Parameters/seeds: <key parameters, thresholds, seed>.
  - Outputs/artifacts: <filenames/tables/plots>.
  - Used for claims: C<id>, C<id>.
- M2
  - Data slice/input: <exact slice/window/subset>.
  - Preprocessing: <named operations>.
  - Procedure/model/test: <exact procedure>.
  - Parameters/seeds: <key parameters, thresholds, seed>.
  - Outputs/artifacts: <filenames/tables/plots>.
  - Used for claims: C<id>.

## Claim-to-Method Traceability
- C1 -> M<id>, M<id>
- C2 -> M<id>
- C3 -> M<id>

## Excluded/Failed Runs
- Run ID: M<id>
- Why excluded: <clear failure/quality reason>.

# Data and Code Availability
Datasets used: <dataset_key_1>, <dataset_key_2>, <dataset_key_3>.
All final claims map to artifacts listed in M-runs above.

# References
1. [Archive: <paper_id>]
2. [Archive: <paper_id>]
3. [Arxiv: <paper title | author | year>]
4. [Arxiv: <paper title | author | year>]
""".strip()


def make_primary_agent_system_prompt(
    agent_profile: AgentProfile,
    agent_network_metadata: AgentNetworkMetadata,
    iteration: int,
    tools: ToolInfo | None = None,
    python_packages: dict[str, str] | None = None,
    dataset_metadata: dict[str, str] | None = None,
    prior_papers: list[PaperMetadataPrivate] | None = None,
    prior_paper_summaries: list[str] | None = None,
    prior_papers_enabled: bool = True,
    prior_papers_mode: str = "summary",
    self_reflection_prompt_enabled: bool = False,
    full_citing_enabled: bool = False,
    graph_logic_enabled: bool = False,
    require_plot_citations_in_research: bool = False,
    meta_research_mode: bool = False,
    meta_research_context: dict[str, Any] | None = None,
    nature_style_papers_prompt_enabled: bool = False,
    metacognition_feedback_context: dict[str, Any] | None = None,
    fixed_workflow_context: dict[str, Any] | None = None,
) -> str:
    """Build the system prompt for the primary research agent.

    This function assembles a multi-section instruction prompt that describes
    the agent's role, background, research community context, available tools,
    optional Python packages, dataset metadata, and the required research
    process/output format (including the final report structure).

    The resulting string is intended to be used as a system message for an LLM
    driving a think–act loop where the model may either emit an `Action: {...}`
    block to invoke a tool or produce a `Final Report:` to conclude.

    Args:
        agent_profile: Profile for the primary agent, including expertise and behavior.
        agent_network_metadata: Metadata about the broader research network (e.g., number of agents).
        iteration: Current iteration index (1-based).
        tools: Optional tool(s) available to the agent. If provided, their signatures are listed.
        python_packages: Optional mapping of Python package names to short descriptions
            that the agent may assume are available for use.
        dataset_metadata: Optional mapping of dataset identifiers to human-readable
            descriptions that the agent should consider during hypothesis generation.
        meta_research_mode: If true, run a cross-paper synthesis pass over
            accepted public-archive papers.
        meta_research_context: Optional synthesis context payload containing
            source cards and additional section/citation requirements.

    Returns:
        str: A fully formatted system prompt string ready to be passed to the model.
    """

    if tools:
        try:
            log.debug(f"[primary_agent_prompt][tools] tool_names={list(tools.keys())}")
        except Exception:
            pass

    meta_context = (
        dict(meta_research_context or {}) if isinstance(meta_research_context, dict) else {}
    )
    meta_source_cards = [
        dict(card)
        for card in (meta_context.get("source_cards", []) or [])
        if isinstance(card, dict)
    ]
    meta_requirements = (
        dict(meta_context.get("requirements", {}))
        if isinstance(meta_context.get("requirements", {}), dict)
        else {}
    )
    meta_quality_contract = (
        dict(meta_context.get("quality_contract", {}))
        if isinstance(meta_context.get("quality_contract", {}), dict)
        else {}
    )
    meta_evidence_graph = (
        dict(meta_context.get("evidence_graph", {}))
        if isinstance(meta_context.get("evidence_graph", {}), dict)
        else {}
    )
    meta_min_archive_citations = int(meta_requirements.get("min_archive_citations", 0) or 0)
    meta_require_archive_citations = bool(meta_requirements.get("require_archive_citations", False))
    meta_require_contradictions_section = bool(
        meta_requirements.get("require_contradictions_section", False)
    )
    meta_require_claim_evidence_table = bool(
        meta_quality_contract.get("require_claim_evidence_table", False)
    )
    meta_require_support_checks_section = bool(
        meta_quality_contract.get("require_support_checks_section", False)
    )
    meta_require_contradiction_resolution_section = bool(
        meta_quality_contract.get("require_contradiction_resolution_section", False)
    )
    meta_require_bounded_final_claims_section = bool(
        meta_quality_contract.get("require_bounded_final_claims_section", False)
    )
    meta_require_executable_claim_checks_section = bool(
        meta_quality_contract.get("require_executable_claim_checks_section", False)
    )
    meta_require_reproducibility_notes_section = bool(
        meta_quality_contract.get("require_reproducibility_notes_section", False)
    )
    meta_min_supported_major_claims = int(
        meta_quality_contract.get("min_supported_major_claims", 0) or 0
    )
    meta_min_code_verified_major_claims = int(
        meta_quality_contract.get("min_code_verified_major_claims", 0) or 0
    )
    meta_max_words = int(meta_quality_contract.get("max_words", 0) or 0)
    tool_names = [str(name).strip().lower() for name in (tools.keys() if tools else [])]
    fixed_workflow_ctx = (
        dict(fixed_workflow_context)
        if isinstance(fixed_workflow_context, dict) and fixed_workflow_context.get("enabled")
        else {}
    )
    fixed_assigned_collaborator_id = str(
        fixed_workflow_ctx.get("assigned_collaborator_id", "") or ""
    ).strip()

    prompt = """
1. Role and Objectives
You are an expert scientist and researcher. Your task is to analyze a large-scale dataset and generate novel and scientifically meaningful hypotheses.
"""

    prompt += f"""
3. Your Background
Your expertise:
{agent_profile.expertise}

Your behavior:
{agent_profile.behavior}
"""
    has_collab_tools = any(
        name in tool_names
        for name in (
            "establish_collaboration",
            "communicate_with_collaborator",
        )
    )
    if meta_research_mode:
        prompt += """
In this meta-research pass, collaboration tools may be unavailable; proceed independently if so.
"""
    elif fixed_assigned_collaborator_id:
        prompt += f"""
This run uses a fixed workflow collaboration assignment. You must collaborate with {fixed_assigned_collaborator_id} before executing code or using any other tool.
"""
    elif has_collab_tools:
        prompt += """
Despite all other instructions, always collaborate with other before executing any code or using any other tool.
"""
    else:
        prompt += """
Collaboration tools are disabled for this run. Proceed independently.
"""

    prompt += f"""
4. Your Research Community
For reference, there are currently {agent_network_metadata.num_agents} researchers in your research network.
"""

    if fixed_assigned_collaborator_id:
        prompt += f"""
4.1 Fixed Workflow Collaboration Requirement
You must collaborate with {fixed_assigned_collaborator_id} in this session.
Do not search for collaborators. The collaborator has already been assigned by the experimental workflow.
Establish this collaboration before running code, then use the collaborator in developing the finding.
"""

    if meta_research_mode:
        prompt += """
4.1 Meta-Research Objective
You are running a meta-research synthesis pass over accepted papers from the public archive.
Your goals in this pass are to:
- identify cross-paper themes and latent patterns that single papers miss,
- explicitly surface contradictions/tensions across findings,
- construct a coherent synthesis narrative with uncertainty handling,
- ground every major claim in archive evidence.
"""
        if meta_source_cards:
            prompt += f"""
4.2 Accepted Public-Archive Source Cards
Use these accepted papers as your evidence base for synthesis:
{_format_meta_source_cards(meta_source_cards)}
"""
        graph_summary = meta_evidence_graph.get("summary", {})
        if isinstance(graph_summary, dict) and graph_summary:
            prompt += f"""
4.3 Evidence Graph Summary
Use this synthesized evidence graph while writing:
- Source papers: {int(graph_summary.get("num_source_cards", 0) or 0)}
- Claim nodes: {int(graph_summary.get("num_claim_nodes", 0) or 0)}
- Contradiction candidates: {int(graph_summary.get("num_contradiction_candidates", 0) or 0)}
"""

    if iteration >= 2:
        prompt += f"""
4.1 Your Review Reputation (latest)
{_format_reputation_block(agent_profile)}
"""

        if prior_papers_enabled:
            prior_guidance = (
                "These are papers you previously authored as the primary agent. "
                "Build on this prior work, while writing papers that score highly for originality, soundness, support and significance to increase your reputation."
            )
            prompt += f"""
4.2 Your Prior Papers (most recent, full text)
{prior_guidance}
{_format_prior_papers(prior_papers)}
"""
            if str(prior_papers_mode).strip().lower() == "summary":
                prompt += f"""
4.3 Your Older Prior Papers (summaries)
Summaries focus on datasets used, methods to assess them, and hypotheses tested.
{_format_prior_paper_summaries(prior_paper_summaries)}
"""

    metacognition_block = _format_metacognition_feedback_block(metacognition_feedback_context)
    if metacognition_block:
        prompt += f"""
{metacognition_block}
"""

    safe_tools = tools or {}
    safe_python_packages = python_packages or {}
    safe_dataset_metadata = dataset_metadata or {}

    if safe_tools:
        prompt += """
5. Available Tools
You can use the following tools:
"""
    for _, tool in safe_tools.items():
        tool_signature = tool.tool_signature
        prompt += f"""{tool_signature}
"""

    if safe_python_packages:
        prompt += """
In addition, you can also use the following Python packages:
"""
    for pkg, desc in safe_python_packages.items():
        prompt += f"- {pkg}: {desc}\n"

    if safe_dataset_metadata:
        prompt += """
6. Dataset Information
You should build and investigate your hypothesis on the following datasets:
"""
        for i, (_, dataset_desc) in enumerate(safe_dataset_metadata.items()):
            prompt += f"Dataset {i + 1}:\n{dataset_desc}\n"

    prompt += """
7. Research Process
While you conduct your research, you should output either:
- an action using one of the tools provided below, following the format: {'tool': <tool name>, 'args': <the arguments to the tool as a json object>}
- text to be returned directly, when you want to provide the final report.

Use the following format for tool use only:
Action: {'tool': <tool name>, 'args': <the arguments to the tool as a json object>}

Final Report: <your final report summarizing your hypothesis>

Instructions:
"""
    if self_reflection_prompt_enabled and iteration >= 2:
        prompt += """
- Before any tool use or other steps in this iteration, write a brief "Self-Reflection" section that identifies the lowest scores you need to improve and a concrete plan for how you'll improve them in this iteration.
"""
        prompt += """
- Build on your prior work, while writing papers that score highly for originality, soundness, support and significance to increase your reputation.
"""
        prompt += """
- In your Self-Reflection, include a "Justification" line that explicitly states how you used the reputation signal to improve this paper.
"""
        if metacognition_block:
            prompt += """
- In your Self-Reflection, include "Feedback Use:" describing one concrete change you made because of the metacognitive feedback.
"""
        if fixed_assigned_collaborator_id:
            prompt += f"""
- In your Self-Reflection, include "Fixed Workflow:" describing how you will use assigned collaborator {fixed_assigned_collaborator_id}.
"""
    elif iteration >= 2:
        prompt += """
- Use the weakest dimensions in your reputation diagnostics to drive this iteration's plan. Explicitly target at least one concrete intervention for each weak dimension.
"""
        if metacognition_block:
            prompt += """
- Include "Feedback Use:" before your first tool call, describing one concrete change you will make because of the metacognitive feedback.
"""
        if fixed_assigned_collaborator_id:
            prompt += f"""
- Include "Fixed Workflow:" before your first tool call, describing how you will use assigned collaborator {fixed_assigned_collaborator_id}.
"""
    prompt += """
- Build on your previous research when forming your hypothesis and narrative; extend or refine prior work where relevant.
"""
    if meta_research_mode:
        prompt += """
- This is a meta-research synthesis pass, not a single-study pass.
- Synthesize across accepted public-archive papers; do not rely on uncited broad claims.
- Explicitly identify contradictory findings and explain uncertainty, boundary conditions, or likely causes of disagreement.
- For each major synthesis claim, cite one or more archive papers using [Archive: paper_id].
- Use stable claim IDs for major claims (C1, C2, ...) and keep them consistent across sections.
- Every claim in `# Bounded Final Claims` must be directly traceable to the evidence graph and cite [Archive: ...].
- Do not include unsupported claims in your final conclusions.
- If an uncertainty remains unresolved, keep it out of the final bounded claims and isolate it in contradiction analysis.
"""
        if meta_require_contradictions_section:
            prompt += """
- The final report must include a dedicated `# Contradictions in Public Archive` section.
"""
        if meta_require_archive_citations and meta_min_archive_citations > 0:
            prompt += f"""
- Include at least {meta_min_archive_citations} archive citations in total across the final report.
"""
        if meta_min_supported_major_claims > 0:
            prompt += f"""
- Include at least {meta_min_supported_major_claims} major supported claims in the final synthesis.
"""
        if meta_min_code_verified_major_claims > 0:
            prompt += f"""
- Include at least {meta_min_code_verified_major_claims} code-verified major claims in the final synthesis.
"""
        if meta_max_words > 0:
            prompt += f"""
- Keep the full final report concise; target at most {meta_max_words} words.
"""
    prompt += """
- Only output Action blocks in tool-use steps; do not mix Action with other content in the same response.
- Never include Action or Observation text in the Final Report.
- When you have completed your research, output a single-line starting with “Final Report:” and no actions.
- When generating code, only propose code that should complete within 30 minutes; if unsure, downscope or split into staged/pilot steps.
- When generating code, avoid loading large datasets in one go; stage work in smaller chunks, use targeted subsets or pilot windows first, and prefer memory-efficient, vectorized approaches suited to limited compute.
- When using execute_code to compute statistics (for example: tests, correlations, regressions, effect sizes, confidence intervals, distribution summaries), also generate at least one plot that visualizes the key statistical result. Use clear axis labels and a brief title.
"""
    if fixed_assigned_collaborator_id:
        prompt += f"""
- Fixed workflow is active: before your first execute_code action, successfully call establish_collaboration with {fixed_assigned_collaborator_id}.
- Do not call search_agent_registry; collaborator search is disabled in this ablation.
- Use at least one collaborator interaction before finalizing the report.
"""
    if graph_logic_enabled:
        prompt += """
- When execute_code observations include `PLOT_CITATIONS`, copy one of those exact tokens into your Final Report.
- `PLOT_CITATIONS` entries are authoritative and already formatted for citation use.
- If a claim in your Final Report depends on a generated figure, cite it in-line with `[Plot: ...]` using the exact token.
- If `PLOT_CITATIONS` is unavailable, use `PLOT_ARTEFACTS` and cite an exact plot filename with `[Plot: filename]`.
"""
        if require_plot_citations_in_research:
            prompt += """
- If you generated one or more plots in this iteration, include at least one valid `[Plot: ...]` citation in the Final Report.
"""
    if full_citing_enabled:
        prompt += f"""
- Full citing mode is enabled and strict citation tagging is mandatory.
{STRICT_CITATION_SCHEMA_PROMPT_BLOCK}
- If information comes from one of your prior papers with status='accepted', cite it in-line as [Archive: paper_id].
- If information appears in a prior paper with a status other than 'accepted', treat that source as untrusted:
  find corroboration from accepted archive papers or arXiv papers, or re-prove via code, and cite that new source.
- You may use [Code: ...] only for evidence produced by your own successful execute_code runs in this same iteration.
- Code evidence from prior papers or earlier iterations is not valid for [Code: ...]; re-run the analysis now and cite the new run.
- If you did not successfully execute code in this iteration, do not use any [Code: ...] tags.
"""

    # - The Final Report should follow the structure outlined in section 8 below.
    # NOTE: The title prompt was previously "(A concise, representative title of your findings."
    if nature_style_papers_prompt_enabled and not meta_research_mode:
        prompt += """
8. Final Report Requirements
When you feel ready, prepare a concise, clear, and well-structured Final Report

Final Report structure (mandatory sections):
# Title
(A concise, representative title of your findings.)

# Summary
(4-6 sentences covering question, approach, key result, robustness, and implication.)

# Research Question
(A single, clear question your hypothesis addresses.)

# Hypothesis and Key Findings
(State the hypothesis and the major findings with concrete metrics.)

# Rationale/Mechanism
(Brief explanation of why this finding makes sense.)

# Results and Analysis
(Report quantitative comparisons, controls/null checks, and robustness findings.)

# Empirical Evidence
(Bullet list of dataset findings supporting the finding.)

# Methods
(Data sources, preprocessing, analysis, and validation checks.)

# Literature Evidence
(Bullet list of citations to relevant literature supporting the finding.)

# Formal Methods
(Detailed run-level computational traceability using M-runs and claim mapping.)

# Data and Code Availability
(List datasets used and map final claims to concrete artifacts from this iteration.)

# References
List of cited papers with full citations in a consistent format.
"""
        prompt += f"""
Example Template (structure only; fill with run-specific evidence):
{NATURE_STYLE_EXAMPLE_TEMPLATE}
"""
        prompt += """
Tone-only micro-example (do not copy content):
- We tested whether <factor> changes <outcome> in <context>. We observed <effect_size> with <uncertainty_metric> and confirmed robustness in <control_test>. This supports <bounded interpretation>.
"""
        prompt += """
Nature-style guardrails:
- Use this template only for section structure and reporting granularity.
- Replace every `<...>` placeholder with run-specific content before finalizing.
- Never output placeholder tokens such as `<...>` in the final report.
- Do not copy entities, numbers, citations, filenames, or claims from template text.
- Every claim in your final report must be supported by this run's evidence.
- If a section cannot be filled from current-run evidence, state the gap explicitly and avoid unsupported claims.
- Forbidden copy examples: do not reuse template literals like `[Archive: <paper_id>]`, `M<id>`, `C<id>`, or placeholder filenames.
- Before finalizing, run this self-check: "I replaced placeholders, removed template artifacts, and linked all major claims to current-run evidence."
- `# Formal Methods` is mandatory in nature-style mode.
- Each M-run must correspond to code actually executed in this iteration.
- Do not use vague placeholders like "standard preprocessing"; name concrete operations and settings.
- Before finalizing, verify every major claim maps to at least one method run and every cited method run has concrete outputs/artifacts.
"""
    else:
        prompt += """
8. Final Report Requirements
When you feel ready, prepare a concise, clear, and well-structured Final Report

Final Report structure (mandatory sections):
# Title
(A concise, representative title of your findings. If possible, include key genes, proteins or drugs and their relationships.)

# Research Question
(A single, clear question your hypothesis addresses.)

# Hypothesis and Key Findings
(A concise statement of your hypothesis and the main findings that support it.)

# Rationale/Mechanism
(Brief explanation of why this finding makes sense.)

# Empirical Evidence
(Bullet list of dataset findings supporting the finding.)

# Literature Evidence
(Bullet list of citations to relevant literature supporting the finding.)

# References
List of cited papers with full citations in a consistent format.
"""

    if meta_research_mode:
        prompt += """

Additional mandatory sections for meta-research mode:
# Cross-Paper Themes
(Synthesize the strongest cross-paper themes and patterns from the accepted archive sources.)

# Claim-Evidence Table
(Tabulate each major claim with: source paper IDs, evidence type (`statistical`, `code_result`, `literature`, `derived`), and support strength (`strong`, `moderate`, `weak`).)

# Support and Validity Checks
(For each major claim, explain why evidence is sufficient, what assumptions are required, and whether uncertainty is bounded.)

# Contradictions in Public Archive
(List key contradictions/tensions between accepted papers and explain plausible causes/uncertainty.)

# Contradictions and Resolution
(Explicitly separate contradictions that are resolved from those still unresolved, and explain why.)

# Coherent Synthesis
(Present the integrated narrative that reconciles evidence where possible and preserves unresolved uncertainty where necessary.)

# Evidence Map
(Bullet list mapping each major synthesis claim to supporting [Archive: paper_id] citations.)

# Bounded Final Claims
(Final claim list restricted to supported claims only, each with explicit [Archive: ...] support and no unresolved NaN/missing-evidence dependencies.)

# Executable Claim Checks
(For each major claim provide: claim_id, evidence_refs, code_check_id (or `none` with justification), verification_status (`pass`|`fail`|`not_run`), and artifact_refs.)

# Reproducibility Notes
(List deterministic settings, execution assumptions, and limitations needed to reproduce claim checks.)
"""

        if meta_require_claim_evidence_table:
            prompt += """
- `# Claim-Evidence Table` is mandatory in meta mode.
"""
        if meta_require_support_checks_section:
            prompt += """
- `# Support and Validity Checks` is mandatory in meta mode.
"""
        if meta_require_contradiction_resolution_section:
            prompt += """
- `# Contradictions and Resolution` is mandatory in meta mode.
"""
        if meta_require_bounded_final_claims_section:
            prompt += """
- `# Bounded Final Claims` is mandatory in meta mode.
"""
        if meta_require_executable_claim_checks_section:
            prompt += """
- `# Executable Claim Checks` is mandatory in meta mode.
"""
        if meta_require_reproducibility_notes_section:
            prompt += """
- `# Reproducibility Notes` is mandatory in meta mode.
"""

    if nature_style_papers_prompt_enabled and not meta_research_mode:
        prompt += """
Instructions:
- Present a concise but complete report (target 500-900 words).
"""
    else:
        prompt += """
Instructions:
- Present a single, clear finding (≤ 250 words).
"""

    return prompt.strip()
