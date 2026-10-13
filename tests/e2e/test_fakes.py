"""The scripted meta-reviewer gives each paper the same score whatever the call order.

Round 1 of an e2e run meta-reviews two tournaments concurrently, so the order of the fake's calls
varies between runs; the accepted paper, and so the fixtures, must not.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agents.meta_review.meta_rev_agent_utils import (
    make_metareviewer_system_prompt,
    make_metareviewer_user_message,
    parse_metareview,
)
from tests.e2e import fakes, harness


def _scores(papers) -> dict[str, float]:
    messages = [
        SystemMessage(content=make_metareviewer_system_prompt()),
        HumanMessage(content=make_metareviewer_user_message(papers, reviews={})),
    ]
    kind, text = fakes.respond(messages)
    assert kind == "metareviewer"
    parsed, valid = parse_metareview(text, len(papers))
    assert valid
    return {pid: row["score"] for pid, row in parsed.items()}


def test_metareview_scores_do_not_depend_on_call_order(private_paper_factory):
    papers = [private_paper_factory(paper_id=f"paper_t01_agent_0{i}") for i in range(1, 6)]
    big, small = papers[:4], papers[4:]  # round 1: tournaments of 4 and 1 papers

    big_first = {**_scores(big), **_scores(small)}
    small_first = {**_scores(small), **_scores(big)}

    assert big_first == small_first
    assert big_first == {p.paper_id: fakes.paper_score(p.paper_id) for p in papers}


def test_e2e_paper_ids_get_distinct_scores():
    ids = [
        f"paper_t{t:02d}_agent_{a:02d}"
        for t in range(1, 4)
        for a in range(1, harness.MAX_AGENTS + 1)
    ]
    scores = [fakes.paper_score(pid) for pid in ids]

    assert len(set(scores)) == len(ids)  # no ties, so no order-dependent tie-break
    assert all(0 < s < 1 for s in scores)
