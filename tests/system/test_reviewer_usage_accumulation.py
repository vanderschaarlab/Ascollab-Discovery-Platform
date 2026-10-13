from agents.review.reviewer_agent import _accumulate_usage
from system.trackers import ReviewStatsLog


class _WithUsage:
    def __init__(self, usage):
        self.usage_metadata = usage


class _WithoutUsage:
    pass


def test_accumulate_usage_handles_sparse_provider_metadata():
    log = ReviewStatsLog()
    _accumulate_usage(log, _WithUsage({"input_tokens": 9}))
    _accumulate_usage(log, _WithUsage({"output_tokens": 4}))
    _accumulate_usage(log, _WithUsage(None))
    _accumulate_usage(log, _WithoutUsage())

    assert log.input_tokens == 9
    assert log.output_tokens == 4
    assert log.total_tokens == 0
