import logging
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage

from system.llm_retry import call_with_content_filter_retry
from system.schema import AgentProfile

log = logging.getLogger(__name__)

_DEFAULT_BEHAVIOR_FALLBACK = "- You are an expert scientific researcher"
_DEFAULT_EXPERTISE_FALLBACK = "- You are an expert in multi-omics analysis"
_CONTENT_FILTER_MAX_RETRIES = 3
_CONTENT_FILTER_RETRY_BASE_SECONDS = 1.5


def _parse_profile(text: str) -> str:
    """Normalize a generated profile string.

    Strips surrounding whitespace and removes a single leading and trailing
    quote character if present (to handle LLM outputs that wrap content
    in quotes).

    Args:
        text: Raw profile text from the model.

    Returns:
        str: Cleaned profile text with outer quotes removed if detected.
    """
    s = text.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    return s


def _extract_tagged(text: str, tag: str) -> str:
    """Return the text inside the first ``<tag>...</tag>`` pair (case-insensitive).

    With an opening tag but no closing one (a reply cut off), the text after the opening tag;
    with no opening tag, the whole text.

    Args:
        text: Model reply.
        tag: Tag name, without angle brackets.

    Returns:
        str: The tagged content, stripped.
    """
    closed = re.search(rf"<{tag}\s*>(.*?)</{tag}\s*>", text, re.IGNORECASE | re.DOTALL)
    if closed:
        return closed.group(1).strip()
    opened = re.search(rf"<{tag}\s*>(.*)", text, re.IGNORECASE | re.DOTALL)
    return opened.group(1).strip() if opened else text.strip()


def _parse_expertise_response(text: str) -> str:
    """Convert the expertise reply into a bullet list when appropriate.

    Takes the text inside the ``<expertise>`` tags the prompt asks for (the whole reply when
    there are none), splits it into lines, preserves leading bullets ('- ', '• ', '* ')
    if already present, otherwise converts non-empty lines (that aren't XML-ish
    tags like `<...>` or `> ...`) into '- ' bullet lines. Falls back to the
    extracted text if no bullet-like structure emerges.

    Args:
        text: Raw expertise description emitted by the model.

    Returns:
        str: A normalized, possibly bullet-pointed expertise block.
    """
    text = _extract_tagged(text, "expertise")
    expertise_content = text

    lines = expertise_content.split("\n")
    bullets = []
    for line in lines:
        line = line.strip()
        if line and (line.startswith("- ") or line.startswith("• ") or line.startswith("* ")):
            bullets.append(line)
        elif line and not line.startswith("<") and not line.startswith(">"):
            bullets.append(f"- {line}")

    return "\n".join(bullets) if bullets else expertise_content


def _extract_message_text(content: Any) -> str:
    """Convert model response content into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        return "\n".join(p for p in parts if p).strip()
    return str(content)


def _extract_generation_text(generation: Any) -> str:
    """Extract text from a LangChain generation object."""
    if hasattr(generation, "message"):
        return _extract_message_text(getattr(generation.message, "content", ""))
    return _extract_message_text(getattr(generation, "text", ""))


def _extract_batch_texts(result: Any, expected_count: int, fallback_text: str) -> list[str]:
    """Extract N generation texts from a LangChain ChatResult-like object."""
    generations = getattr(result, "generations", None) or []
    texts: list[str] = []
    for idx in range(expected_count):
        try:
            generation = generations[idx][0]
        except Exception:
            log.warning(
                "[init_agents_profiles] Missing generation for index %d; using fallback text.",
                idx,
            )
            texts.append(fallback_text)
            continue
        text = _extract_generation_text(generation).strip()
        texts.append(text or fallback_text)
    return texts


def _generate_single_prompt_with_retry(
    llm: BaseChatModel,
    prompt_messages: list[SystemMessage],
    *,
    stage_label: str,
    prompt_index: int,
    fallback_text: str,
) -> str:
    """Generate one prompt with retries, then fall back to deterministic text."""
    try:
        result = call_with_content_filter_retry(
            lambda: llm.generate([prompt_messages]),
            stage_label=f"init_agents_profiles.{stage_label}.prompt_{prompt_index}",
            logger=log,
            max_retries=_CONTENT_FILTER_MAX_RETRIES,
            base_delay_seconds=_CONTENT_FILTER_RETRY_BASE_SECONDS,
        )
        return _extract_batch_texts(result, expected_count=1, fallback_text=fallback_text)[0]
    except Exception as exc:
        log.error(
            "[init_agents_profiles] Prompt %d failed during %s after retries (%s). Using fallback text.",
            prompt_index,
            stage_label,
            type(exc).__name__,
        )
        return fallback_text


def _generate_prompts_with_retry(
    llm: BaseChatModel,
    prompts: list[list[SystemMessage]],
    *,
    stage_label: str,
    fallback_text: str,
) -> list[str]:
    """Generate a batch with retries, then degrade to per-prompt retries if needed."""
    try:
        result = call_with_content_filter_retry(
            lambda: llm.generate(prompts),
            stage_label=f"init_agents_profiles.{stage_label}.batch",
            logger=log,
            max_retries=_CONTENT_FILTER_MAX_RETRIES,
            base_delay_seconds=_CONTENT_FILTER_RETRY_BASE_SECONDS,
        )
        return _extract_batch_texts(
            result, expected_count=len(prompts), fallback_text=fallback_text
        )
    except Exception as exc:
        log.warning(
            "[init_agents_profiles] Batch generation failed during %s after retries (%s). "
            "Falling back to per-agent generation.",
            stage_label,
            type(exc).__name__,
        )

    return [
        _generate_single_prompt_with_retry(
            llm,
            prompt_messages,
            stage_label=stage_label,
            prompt_index=i,
            fallback_text=fallback_text,
        )
        for i, prompt_messages in enumerate(prompts, start=1)
    ]


behavior_system_prompt_base = """
You are to generate a single epistemic researcher profile.

Each persona must capture how the researcher behaves and thinks across six stances:
1. Ideas – Refining and extending existing ideas ↔ generating brand new ones.
2. Collaboration – Independence ↔ collaboration.
3. Scope – Broad exploration ↔ deep exploitation of a problem.
4. Evaluation – Critical scrutiny ↔ constructive engagement.
5. Literature – Reliance on existing literature ↔ intuition with minimal reference to prior work.
6. Resources – Maximal use of resources and depth ↔ lean, minimalist approaches.
"""


def create_behavior_system_prompt() -> str:
    """Build the system prompt used to elicit a researcher behavior persona.

    The prompt instructs the model to generate a single epistemic researcher
    profile across six behavioral stances (ideas, collaboration, scope,
    evaluation, literature reliance, and resource usage).

    Returns:
        str: The behavior system prompt text.
    """
    return behavior_system_prompt_base


def create_expertise_system_prompt(dataset_metadata: dict[str, str]) -> str:
    """Build the system prompt used to elicit domain expertise text.

    Includes a dataset context section enumerating the provided datasets and
    asks the model to produce a focused expertise description explaining how
    the researcher would leverage the datasets to generate findings. The
    desired output is enclosed in `<expertise>...</expertise>` tags.

    Args:
        dataset_metadata: Mapping of dataset name to description.

    Returns:
        str: The expertise-generation system prompt including dataset context.
    """
    dataset_context = "\nDataset context available for research:\n"
    for i, (dataset_name, description) in enumerate(dataset_metadata.items(), 1):
        dataset_context += f"Dataset {i}: {dataset_name}\n{description}\n\n"

    prompt = f"""
You are to generate a domain expertise description for a researcher with the following specific technical expertise areas:.

{dataset_context}

The expertise should describe what domain knowledge and technical skills this researcher possesses in these areas, specifically focused on how they would generate novel research findings using the available datasets.
Format your response as:
<expertise>
...
</expertise>
"""
    return prompt


def init_agents_profiles(
    llm: BaseChatModel,
    dataset_metadata: dict[str, str],
    n: int,
    network_setting: str | None = "network_hetero",
    agent_id_prefix: str = "agent",
) -> list[AgentProfile]:
    """Initialize N agent profiles, either homogeneous or LLM-generated heterogeneous.

    For homogeneous settings (`network_homo` or `indep_homo`), returns `n` identical
    baseline profiles. Otherwise, queries the LLM twice (per agent): once to
    generate a behavior persona and once to generate dataset-aware expertise,
    then normalizes and assembles them into `AgentProfile` objects.

    Azure content-filter failures are retried with exponential backoff. If a
    batch keeps failing due to content filtering, generation degrades to
    per-agent retries, and finally falls back to deterministic default profile
    text for only the still-failing prompts.

    Args:
        llm: Chat model used to generate behavior and expertise content.
        dataset_metadata: Mapping of dataset names to descriptions used to
            condition the expertise prompt.
        n: Number of agent profiles to create.
        network_setting: One of {'network_hetero', 'network_homo', 'indep_homo'}.
            Homogeneous settings bypass LLM generation.

    Returns:
        List[AgentProfile]: A list of constructed agent profiles.
    """

    if network_setting in ["network_homo", "indep_homo"]:
        profiles: list[AgentProfile] = []

        for i in range(n):
            profile = AgentProfile(
                agent_id=f"{agent_id_prefix}_{i + 1:02d}",
                behavior=_DEFAULT_BEHAVIOR_FALLBACK,
                expertise=_DEFAULT_EXPERTISE_FALLBACK,
                expertise_topics=["multi-omics analysis"],
                citation_count=0,
                num_accepted_papers=0,
            )

            profiles.append(profile)
        return profiles

    behavior_prompts: list[list[SystemMessage]] = []
    for _ in range(n):
        behavior_prompt = create_behavior_system_prompt()
        behavior_prompts.append([SystemMessage(content=behavior_prompt)])
    behavior_texts = _generate_prompts_with_retry(
        llm,
        behavior_prompts,
        stage_label="behavior",
        fallback_text=_DEFAULT_BEHAVIOR_FALLBACK,
    )

    expertise_prompts: list[list[SystemMessage]] = []

    for _ in range(n):
        expertise_system_prompt = create_expertise_system_prompt(dataset_metadata)
        expertise_prompts.append([SystemMessage(content=expertise_system_prompt)])
    expertise_texts = _generate_prompts_with_retry(
        llm,
        expertise_prompts,
        stage_label="expertise",
        fallback_text=_DEFAULT_EXPERTISE_FALLBACK,
    )

    profiles = []

    for i in range(n):
        behavior_content = behavior_texts[i]
        expertise_raw_content = expertise_texts[i]
        expertise_content = (
            _parse_expertise_response(expertise_raw_content) or _DEFAULT_EXPERTISE_FALLBACK
        )
        agent_expertise_topics = expertise_content

        profile = AgentProfile(
            agent_id=f"{agent_id_prefix}_{i + 1:02d}",
            behavior=_parse_profile(behavior_content),
            expertise=_parse_profile(expertise_content),
            expertise_topics=agent_expertise_topics,
            citation_count=0,
            num_accepted_papers=0,
        )

        profiles.append(profile)

    return profiles
