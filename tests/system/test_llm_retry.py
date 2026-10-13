from unittest.mock import MagicMock

import pytest

import system.llm_retry as llm_retry


def _openai_error(cls, status_code):
    """Construct an openai APIStatusError subclass with a fake response."""
    return cls(
        message=cls.__name__,
        response=MagicMock(status_code=status_code, headers={}),
        body=None,
    )


@pytest.mark.parametrize(
    ("name", "status"),
    [
        ("BadRequestError", 400),
        ("AuthenticationError", 401),
        ("PermissionDeniedError", 403),
        ("NotFoundError", 404),
        ("ConflictError", 409),
        ("UnprocessableEntityError", 422),
    ],
)
def test_client_errors_are_not_retried(name, status):
    """openai.APIError is the base of the whole hierarchy, 4xx included."""
    import openai

    exc = _openai_error(getattr(openai, name), status)

    assert llm_retry._is_transient_retryable_error(exc) is False


def test_transient_errors_are_retried():
    import openai

    assert llm_retry._is_transient_retryable_error(_openai_error(openai.RateLimitError, 429))
    assert llm_retry._is_transient_retryable_error(_openai_error(openai.InternalServerError, 500))
    assert llm_retry._is_transient_retryable_error(openai.APITimeoutError(request=MagicMock()))
    assert llm_retry._is_transient_retryable_error(openai.APIConnectionError(request=MagicMock()))
    # 408 Request Timeout is worth another attempt even though it is a 4xx.
    assert llm_retry._is_transient_retryable_error(_openai_error(openai.APIStatusError, 408))


def test_non_retryable_api_error_is_raised_after_one_attempt(monkeypatch):
    import openai

    monkeypatch.setattr(llm_retry.time, "sleep", lambda _seconds: None)
    calls = {"count": 0}

    def failing():
        calls["count"] += 1
        raise _openai_error(openai.BadRequestError, 400)

    with pytest.raises(openai.BadRequestError):
        llm_retry.call_with_content_filter_retry(failing, stage_label="test", max_retries=3)

    assert calls["count"] == 1


def test_retry_delay_respects_max_and_jitter(monkeypatch):
    monkeypatch.setattr(llm_retry.random, "uniform", lambda _lo, _hi: 0.5)

    assert llm_retry.retry_delay_seconds(3, base_delay_seconds=2, max_delay_seconds=5) == 5


def test_content_filter_detection_checks_body_and_response_text():
    class Response:
        text = "blocked by content management policy"

    exc = Exception("request failed")
    exc.response = Response()

    assert llm_retry.is_azure_content_filter_error(exc) is True


def test_call_with_content_filter_retry_retries_then_returns(monkeypatch):
    calls = {"count": 0}

    def flaky():
        calls["count"] += 1
        if calls["count"] == 1:
            raise Exception("content_filter")
        return "ok"

    monkeypatch.setattr(llm_retry.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(llm_retry, "retry_delay_seconds", lambda *args, **kwargs: 0)

    assert (
        llm_retry.call_with_content_filter_retry(flaky, stage_label="test", max_retries=1) == "ok"
    )
    assert calls["count"] == 2


def test_call_with_content_filter_retry_does_not_retry_non_retryable(monkeypatch):
    monkeypatch.setattr(llm_retry.time, "sleep", lambda _seconds: None)

    with pytest.raises(ValueError, match="bad input"):
        llm_retry.call_with_content_filter_retry(
            lambda: (_ for _ in ()).throw(ValueError("bad input")),
            stage_label="test",
            max_retries=3,
        )
