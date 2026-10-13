import json

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from stores import state
from system.schema import AgentProfile, PaperMetadataPrivate  # ,


def archive_query_tool(query_text: str, k: int = 3) -> str:
    """Query the internal PaperArchive for semantically similar papers.

    Looks up the global `state.archive()` object, verifies that it has entries,
    and performs a semantic search for the provided `query_text`. Returns a
    lightweight summary of the top-`k` matches (title and abstract).

    Args:
        query_text: Natural-language query describing the topic of interest.
        k: Maximum number of paper results to include.

    Returns:
        str | dict: A human-readable message if the archive is empty, otherwise
        a dictionary with a `"results"` list of `{title, abstract}` items.
    """
    archive = state.archive()

    # Check if the archive is empty
    if not hasattr(archive, "_entries") or not archive._entries:
        return "There are no papers in the archive right now"

    results = archive.query_semantic(query_text, k)
    output = {
        "results": [
            {
                "paper_id": getattr(paper, "paper_id", None),
                "title": paper.title,
                "abstract": paper.abstract,
            }
            for paper, score in results
        ]
    }

    return output


def registry_query_tool(query_text: str, k: int = 3) -> str:
    """Query the AgentRegistry for researchers relevant to a topic.

    Uses the global `state.registry()` to search for agents whose expertise
    matches the `query_text`. If the registry or its vector index is empty,
    returns a friendly message.

    Args:
        query_text: Natural-language query describing desired expertise.
        k: Maximum number of researcher profiles to return.

    Returns a dict:
      {"results": [{"agent_id": "...", "expertise": "...", "expertise_topics": [...],
                    "citation_count": int, "num_accepted_papers": int, "score": float}, ...]}
    or a friendly string if empty.
    """
    registry = state.registry()

    # Check if the registry is empty
    if (
        not hasattr(registry, "vector_store")
        or registry.vector_store is None
        or not hasattr(registry.vector_store, "index_to_docstore_id")
        or not registry.vector_store.index_to_docstore_id
    ):
        return "Observation: AGENT_SEARCH_RESULTS = []"

    try:
        results = registry.query(query_text, k=k)  # List[Tuple[AgentProfile, score]]
    except Exception:
        return "Observation: AGENT_SEARCH_RESULTS = []"

    output = {
        "results": [
            {
                "agent_id": getattr(profile, "agent_id", None),
                "expertise": getattr(profile, "expertise", None),
                "expertise_topics": getattr(profile, "expertise_topics", []),
                "citation_count": getattr(profile, "citation_count", None),
                "num_accepted_papers": getattr(profile, "num_accepted_papers", None),
                "score": float(score) if isinstance(score, (int, float)) else None,
            }
            for (profile, score) in results
        ][:k]
    }
    return f"Observation: AGENT_SEARCH_RESULTS = {json.dumps(output, ensure_ascii=False)}"


def establish_connection(agent_id: str) -> tuple[bool, AgentProfile, list[PaperMetadataPrivate]]:
    """Establish a collaboration context with a given agent.

    Looks up the agent in the global AgentRegistry and, if found, retrieves the
    most recent private papers associated with that agent to seed collaboration.

    Args:
        agent_id: The unique ID of the agent to connect with.

    Returns:
        Tuple[bool, AgentProfile, List[PaperMetadataPrivate]]:
            - A boolean indicating if the agent was found.
            - The AgentProfile of the collaborator (or None if not found).
            - A list of up to 3 recent PaperMetadataPrivate objects associated
              with the agent (empty if not found).
    """
    registry = state.registry()
    private_archive = state.private_archive()
    exists, agent_profile = registry.get_agent(agent_id)

    if not exists or agent_profile is None:
        return False, None, []

    papers = private_archive.get_by_primary_agent(agent_id, k=3) or []
    return exists, agent_profile, papers


def send_communication(agent_id: str, message: str) -> dict[str, str]:
    """Back the ``communicate_with_collaborator`` tool's schema; not called during runs.

    The research team intercepts this tool and passes the message to its collaborator agent
    itself (``agents/research/research_team.py``), so a run never reaches this function. Called
    directly, it echoes the request.

    Args:
        agent_id: The recipient collaborator's agent ID.
        message: The message content to send.

    Returns:
        dict[str, str]: The request, with ``status`` and ``delivery`` fields.
    """
    return {
        "status": "ok",
        "agent_id": agent_id,
        "message": message,
        "delivery": "queued",
    }


##########################################
#### Structured Tools for Agent Use ######
##########################################


class ArchiveQueryArgs(BaseModel):
    """Arguments for querying the internal paper archive."""

    query_text: str = Field(description="The search query text.")
    k: int | None = Field(default=3, description="The maximum number of results")


class RegistryQueryArgs(BaseModel):
    """Arguments for querying the agent registry."""

    query_text: str = Field(description="The search query text")
    k: int | None = Field(default=3, description="The maximum number of results")


class EstablishCollaborationArgs(BaseModel):
    """Arguments for establishing a collaboration with an agent."""

    agent_id: str = Field(..., description="Collaaator id")


class CommunicateArgs(BaseModel):
    """Arguments for sending a message to a collaborator."""

    agent_id: str = Field(..., description="Collaborator id")
    message: str = Field(..., description="Message")


# Create the structured tools
paper_search_tool = StructuredTool.from_function(
    func=archive_query_tool,
    name="search_internal_archive",
    description="Search for papers in the internal archive.",
    args_schema=ArchiveQueryArgs,
)

agent_search_tool = StructuredTool.from_function(
    func=registry_query_tool,
    name="search_agent_registry",
    description="Search for researchers by expertise.",
    args_schema=RegistryQueryArgs,
)

establish_collaboration_tool = StructuredTool.from_function(
    func=establish_connection,
    name="establish_collaboration",
    description="Establish a collaboration with another agent. ",
    args_schema=EstablishCollaborationArgs,
)

communicate_tool = StructuredTool.from_function(
    func=send_communication,
    name="communicate_with_collaborator",
    description="Send a message to a collaborator agent.",
    args_schema=CommunicateArgs,
)
