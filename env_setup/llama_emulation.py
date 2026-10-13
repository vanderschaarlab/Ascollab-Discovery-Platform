from typing import Any


def is_llama_backend(llm: Any) -> bool:
    """
    Heuristic: True if the Azure deployment name (AzureChatOpenAI) or, for an
    OpenAI-compatible endpoint, the model name (ChatOpenAI) hints "llama".

    AzureChatOpenAI keeps the deployment in ``deployment_name``; ``azure_deployment`` is only
    its constructor alias, and is read as well for other client objects.
    """
    dep = (
        getattr(llm, "deployment_name", None)
        or getattr(llm, "azure_deployment", None)
        or getattr(llm, "model_name", None)
        or ""
    )
    return "llama" in str(dep).lower()


def make_llama_auto_tools_prompt(
    base_system_prompt: str, valid_agent_ids: list[str] | None = None
) -> str:
    """
    Augment your system prompt so Llama decides whether to act:
      - If an ACTION is needed -> emit strict `Action: {...}` (no fences)
      - Otherwise -> emit `Thought: ...`
    Adds a rule to constrain valid agent_ids when present.
    """
    ids_rule = ""
    if valid_agent_ids:
        ids_rule = (
            "\nRULE: When calling establish_collaboration or communicate_with_collaborator, "
            f"you MUST use one of these agent_ids: {', '.join(valid_agent_ids)}. "
            "Do not invent new IDs."
        )
    return (
        base_system_prompt + ids_rule + "\n\n== TOOL DECISION PROTOCOL ==\n"
        "When an ACTION is needed, respond ONLY with:\n"
        'Action: {"tool": "<tool_name>", "args": { /* JSON args */ }}\n'
        "When NO tool is needed, respond ONLY with:\n"
        "Thought: <one brief sentence>\n"
        "Never mix Thought and Action in the same turn. "
        "Never use code fences around the JSON.\n"
        "Make sure your any JSON is well-formed and complete."
    )
