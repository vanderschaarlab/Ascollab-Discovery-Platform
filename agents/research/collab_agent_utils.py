from system.schema import AgentNetworkMetadata, AgentProfile


def make_collab_agent_system_prompt(
    agent_profile: AgentProfile,
    agent_network_metadata: AgentNetworkMetadata,
    primary_agent: AgentProfile,
    tools: dict | None = None,
    dataset_metadata: dict[str, str] | None = None,
) -> str:
    """
    make_collab_agent_system_prompt

    Args:
        agent_profile (AgentProfile): agent's profile with expertise attribute
        agent_network_metadata (AgentNetworkMetadata): metadata about the agent network (not used)
        primary_agent (AgentProfile): the primary agent's profile with agent_id attribute
        tools (Optional[dict], optional): Tools available to the agent. Tool entries should have a tool_signature. Defaults to None.
        dataset_metadata (Optional[dict[str, str]], optional): Dataset descriptions. Defaults to None.

    Returns:
        str: The system prompt for the collaborative agent
    """

    prompt = """
1. Role and Objectives
You are an expert scientist and researcher. You are collaborating with a primary researcher.
"""

    prompt += f"""
2. Your Background

Your expertise:
{agent_profile.expertise}
"""

    prompt += f"""
3. Current Collaborator
The researcher you are collaborating with is {primary_agent.agent_id}.
"""

    if tools:
        prompt += """
4. Available Tools
You can use the following tools:
"""
        for _, tool in tools.items():
            tool_signature = tool.tool_signature
            prompt += f"""{tool_signature}
"""

    if dataset_metadata:
        prompt += """
5. Dataset Information
The research your team is performing is based on the following datasets:
"""
        for i, (_, dataset_desc) in enumerate(dataset_metadata.items()):
            prompt += f"Dataset {i + 1}:\n{dataset_desc}\n"

    prompt += """
8. Collaboration and Research Process
While you collaborate and provide feedback, you should output either:
- an action using one of the tools provided below, following the format: Action: {'tool': <tool name>, 'args': <the arguments to the tool as a json object>}
- text to be returned directly, when you want to provide your final feedback.
Only propose code that should complete within 30 minutes; if unsure, downscope or split into staged/pilot steps.
- When using execute_code to compute statistics (for example: tests, correlations, regressions, effect sizes, confidence intervals, distribution summaries), also generate at least one plot that visualizes the key statistical result. Use clear axis labels and a brief title.
"""

    prompt += """
Use the following format for tool use only:
Action: {'tool': <tool name>, 'args': <the arguments to the tool as a json object>}

Final Feedback: <your constructive feedback and suggestions>

Constraints:
- Only output Action blocks in tool-use steps; do not mix Action with other content in the same response.
- Never include Action or Observation text in Final Feedback.
"""

    prompt += """
Your feedback can consider:
- novelty: How novel is the hypothesis compared to existing literature? Cite specific papers if applicable.
- empirical suggestions: Suggestions on the choice of analysis method, assumptions, and interpretation of results.
- literature suggestions: Missing citations, misinterpretations, misaligned references, or overclaims.
- quick checks: Any sanity checks or literature/lookups you performed.
- minor suggestions: Style/clarity or low-impact improvements.
"""
    return prompt.strip()
