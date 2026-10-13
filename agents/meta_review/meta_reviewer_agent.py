import logging
from typing import Annotated, Any, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from agents.meta_review.meta_rev_agent_utils import (
    make_metareviewer_system_prompt,
    parse_metareview,
)
from system.llm_retry import invoke_with_content_filter_retry
from system.trackers import attach_logs, get_logs

log = logging.getLogger(__name__)


def _accumulate_usage(log_obj: Any, ai: Any) -> None:
    if not log_obj:
        return
    usage = getattr(ai, "usage_metadata", {}) or {}
    log_obj.input_tokens += int(usage.get("input_tokens", 0) or 0)
    log_obj.output_tokens += int(usage.get("output_tokens", 0) or 0)
    log_obj.total_tokens += int(usage.get("total_tokens", 0) or 0)


class MetaReviewerState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    metareview_text: str | None
    parsed_reviews: dict[str, dict[str, Any]] | None
    valid_metareview: bool | None
    num_papers: int | None
    metadata: dict[str, Any] | None


def get_metareviewer_agent(
    llm: BaseChatModel,
    review_free: bool = False,
) -> Any:
    """
    get_metareviewer_agent

    Args:
        llm (BaseChatModel): language model for the metareviewer agent
        review_free: When True (the reviewerless ablation, experiment 07), build the system
            prompt without any mention of reviews.

    Returns:
        Any: the metareviewer agent as a anggraph.graph.StateGraph
    """

    system_prompt = make_metareviewer_system_prompt(review_free=review_free)

    def metareview_node(state: MetaReviewerState) -> MetaReviewerState:
        """
        metareview_node

        Args:
            state (MetaReviewerState): current state of the metareviewer agent

        Returns:
            MetaReviewerState: updated state after metareviewing
        """
        m = get_logs(agent)
        response = invoke_with_content_filter_retry(
            llm,
            [
                SystemMessage(content=system_prompt),
                *state["messages"],
            ],
            stage_label="meta_reviewer.initial",
            logger=log,
        )
        _accumulate_usage(m, response)
        if m:
            m.turns += 1
        msgs_delta = [response]

        metareview = response.content or ""
        num_papers = state.get("num_papers", 0)
        parsed_metareview, is_valid = parse_metareview(metareview, num_papers)
        if m:
            meta = state.get("metadata", {}) or {}
            m.add_event(
                kind="think",
                tournament_id=meta.get("tournament_id"),
                iteration=meta.get("iteration"),
                step=1,
                valid_metareview=bool(is_valid),
            )

        if not is_valid:
            nudge = HumanMessage(content="Please try again.")
            retry = invoke_with_content_filter_retry(
                llm,
                [
                    SystemMessage(content=system_prompt),
                    *state["messages"],
                    *msgs_delta,
                    nudge,
                ],
                stage_label="meta_reviewer.nudge",
                logger=log,
            )
            _accumulate_usage(m, retry)
            if m:
                m.turns += 1

            msgs_delta.append(retry)
            metareview = retry.content or ""
            parsed_metareview, is_valid = parse_metareview(metareview, num_papers)
            if m:
                meta = state.get("metadata", {}) or {}
                m.add_event(
                    kind="retry",
                    tournament_id=meta.get("tournament_id"),
                    iteration=meta.get("iteration"),
                    step=2,
                    valid_metareview=bool(is_valid),
                )

        return {
            **state,
            "messages": msgs_delta,
            "metareview_text": metareview,
            "parsed_reviews": parsed_metareview,
            "valid_metareview": is_valid,
        }

    graph = StateGraph(MetaReviewerState)
    graph.add_node("meta_review", metareview_node)
    graph.set_entry_point("meta_review")
    graph.add_edge("meta_review", END)

    agent = graph.compile()
    attach_logs(agent, type="metareview")
    return agent
