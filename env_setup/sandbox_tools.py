def _emit_arxiv_code(params: dict) -> str:
    """Generate source code for a simple arXiv search tool and its signature.

    This factory function embeds the provided configuration `params` directly
    into a Python source string that uses `langchain_community.utilities.ArxivAPIWrapper`
    to query arXiv. It returns two multi-line strings:
      1) `code`: a complete, importable module snippet defining `search_arxiv(...)`
         with a pre-configured wrapper singleton.
      2) `signature`: a lightweight function signature/docstring snippet suitable
         for prompt/tool descriptions.

    Args:
        params: Keyword arguments to initialize `ArxivAPIWrapper` (e.g., `top_k`,
            `load_max_docs`, `doc_content_chars_max`, etc.). These are baked
            into the generated source at creation time.

    Returns:
        tuple[str, str]: A pair `(code, signature)` where:
            - `code` is the full Python implementation as a string.
            - `signature` is a concise function signature + docstring string.
    """
    code = f'''
from typing import Optional
from langchain_community.utilities.arxiv import ArxivAPIWrapper

# Resolved hyperparameters baked at generation time
_ARXIV_CFG = {params!r}

# Wrapper instance (module-level singleton)
_arxiv_wrapper = ArxivAPIWrapper(**_ARXIV_CFG)

def search_arxiv(query: str) -> str:
    """
    Search arXiv with query and return a text summary.

    Args:
        query: arXiv query string. Examples:
            - "challenges in long-context transformers"
            - "reinforcement learning from human feedback survey"
    Returns:
        str: Human-readable summary of the top results retrieved from arXiv.
             (Titles, authors, and summary.)
    """
    return _arxiv_wrapper.run(query)
'''

    signature = '''
def search_arxiv(query: str) -> str:
    """
    Search arXiv with query and return a text summary.
    Args:
        query: arXiv query string. Examples:
            - "challenges in long-context transformers"
            - "reinforcement learning from human feedback survey"
    Returns:
        str: Human-readable summary of the top results retrieved from arXiv.
    """
'''
    return code, signature


def _emit_search_agent_registry_code(params: dict) -> tuple[str, str]:
    code = '''
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
    if (not hasattr(registry, 'vector_store') or
        registry.vector_store is None or
        not hasattr(registry.vector_store, 'index_to_docstore_id') or
        not registry.vector_store.index_to_docstore_id):
        return "There are no researchers in the registry right now!"

    try:
        results = registry.query(query_text, k=k)  # List[Tuple[AgentProfile, score]]
    except Exception:
        return "There are no researchers in the registry right now!"

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
    return output
'''
    sig = '''
def registry_query_tool(query_text: str, k: int = 3) -> str:
    """
    Search AgentRegistry and return candidates.
    Use the 'id' field from these results as agent_id when calling establish_collaboration.
    """
'''
    return code, sig
