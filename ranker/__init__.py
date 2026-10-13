"""LLM-judge ranker: pairwise paper comparison via an autonomous LangGraph agent.

Public API::

    from ranker import run_ranker, RankerResult

    result = run_ranker(
        paper_a={"title": "...", "abstract": "...", "manuscript": "..."},
        paper_b={"title": "...", "abstract": "...", "manuscript": "..."},
        llm=llm_instance,
    )
"""

try:
    from .ranker_core import run_ranker
    from .ranker_schemas import RankerResult, RankerTrace
except ModuleNotFoundError as exc:  # pragma: no cover - optional-dependency lightweight imports
    if exc.name != "langchain_core":
        raise
    RankerResult = None  # type: ignore[assignment]
    RankerTrace = None  # type: ignore[assignment]

    def run_ranker(*_args, _exc=exc, **_kwargs):  # type: ignore[no-redef]
        raise ModuleNotFoundError(
            "run_ranker requires optional ranker dependencies, including langchain_core."
        ) from _exc


__all__ = ["RankerResult", "RankerTrace", "run_ranker"]
