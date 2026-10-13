import asyncio

from system.metareview_one_step import metareview_one_step


def test_metareview_one_step_returns_reviews_and_logs_tuple_for_empty_input():
    reviews, logs = asyncio.run(
        metareview_one_step(
            iteration=1,
            llm=object(),
            llm_for=None,
            candidate_papers=[],
            reviews_by_paper={},
            tournament_size=2,
        )
    )

    assert reviews == []
    assert logs == []
