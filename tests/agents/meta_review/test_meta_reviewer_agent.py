from langchain_core.messages import AIMessage

from agents.meta_review.meta_reviewer_agent import _accumulate_usage


def test_meta_reviewer_accumulate_usage_handles_missing_metadata():
    class Log:
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0

    log = Log()
    _accumulate_usage(log, AIMessage(content="no usage"))

    assert log.input_tokens == 0
    assert log.output_tokens == 0
    assert log.total_tokens == 0


def test_meta_reviewer_accumulate_usage_adds_provider_counts():
    class Log:
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0

    msg = AIMessage(content="usage")
    msg.usage_metadata = {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7}
    log = Log()

    _accumulate_usage(log, msg)

    assert (log.input_tokens, log.output_tokens, log.total_tokens) == (3, 4, 7)
