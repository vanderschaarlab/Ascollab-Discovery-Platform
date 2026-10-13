import asyncio
import json
import logging
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from openai import APIConnectionError, APIStatusError, InternalServerError, RateLimitError

DEFAULT_CONTENT_FILTER_MAX_RETRIES = 3
DEFAULT_CONTENT_FILTER_RETRY_BASE_SECONDS = 1.5
DEFAULT_RETRY_MAX_DELAY_SECONDS = 30.0

_T = TypeVar("_T")


@dataclass(frozen=True)
class RetryPolicy:
    """Retry policy for LLM calls.

    max_retries counts retries after the initial attempt.
    """

    max_retries: int = DEFAULT_CONTENT_FILTER_MAX_RETRIES
    base_delay_seconds: float = DEFAULT_CONTENT_FILTER_RETRY_BASE_SECONDS
    max_delay_seconds: float | None = DEFAULT_RETRY_MAX_DELAY_SECONDS


def _coerce_retry_policy(
    *,
    max_retries: int,
    base_delay_seconds: float,
    max_delay_seconds: float | None,
) -> RetryPolicy:
    try:
        retries = max(0, int(max_retries))
    except Exception:
        retries = DEFAULT_CONTENT_FILTER_MAX_RETRIES

    try:
        base_delay = float(base_delay_seconds)
    except Exception:
        base_delay = DEFAULT_CONTENT_FILTER_RETRY_BASE_SECONDS
    base_delay = max(0.0, base_delay)

    if max_delay_seconds is None:
        max_delay = None
    else:
        try:
            max_delay = float(max_delay_seconds)
        except Exception:
            max_delay = DEFAULT_RETRY_MAX_DELAY_SECONDS
        max_delay = max(0.0, max_delay)

    return RetryPolicy(
        max_retries=retries,
        base_delay_seconds=base_delay,
        max_delay_seconds=max_delay,
    )


def _error_text(exc: Exception) -> str:
    parts = [str(exc)]

    body = getattr(exc, "body", None)
    if body:
        if isinstance(body, (dict, list)):
            try:
                parts.append(json.dumps(body, sort_keys=True))
            except Exception:
                parts.append(str(body))
        else:
            parts.append(str(body))

    response = getattr(exc, "response", None)
    if response is not None:
        text = getattr(response, "text", None)
        if text:
            parts.append(str(text))

    return " ".join(parts).lower()


def is_azure_content_filter_error(exc: Exception) -> bool:
    """Return True when an exception likely represents Azure/OpenAI content filtering."""
    msg = _error_text(exc)
    if "azure has not provided the response due to a content filter being triggered" in msg:
        return True

    content_filter_markers = (
        "content filter",
        "content_filter",
        "content management policy",
        "responsibleai",
    )
    return any(marker in msg for marker in content_filter_markers)


def retry_delay_seconds(
    attempt: int,
    *,
    base_delay_seconds: float = DEFAULT_CONTENT_FILTER_RETRY_BASE_SECONDS,
    max_delay_seconds: float | None = None,
) -> float:
    """Compute jittered exponential backoff delay for retry attempt (1-indexed)."""
    base = max(0.0, base_delay_seconds) * (2 ** max(0, attempt - 1))
    jitter = random.uniform(0.0, max(0.0, base_delay_seconds))
    delay = base + jitter
    if max_delay_seconds is not None:
        delay = min(delay, max(0.0, max_delay_seconds))
    return delay


# 408 Request Timeout and 429 Too Many Requests are worth another attempt.
# 409 Conflict is deliberately NOT included: it signals a state conflict that a
# retry will reproduce.
_RETRYABLE_STATUS_CODES = frozenset({408, 429})


def _is_transient_retryable_error(exc: Exception) -> bool:
    """Return True only for errors that could plausibly succeed on a retry.

    `openai.APIError` is the base of the WHOLE hierarchy, including 400/401/404,
    so it must not be used as the retry predicate.

    Args:
        exc: The exception raised by the LLM call.

    Returns:
        True when a retry is worth attempting.
    """
    if isinstance(exc, (RateLimitError, APIConnectionError, InternalServerError)):
        return True
    if isinstance(exc, APIStatusError):
        status = getattr(exc, "status_code", None)
        if isinstance(status, int):
            return status >= 500 or status in _RETRYABLE_STATUS_CODES
    return False


def _classify_retryable_error(
    exc: Exception,
    *,
    retry_on_content_filter: bool,
    retry_on_transient: bool,
) -> str | None:
    if retry_on_content_filter and is_azure_content_filter_error(exc):
        return "Azure content filter"
    if retry_on_transient and _is_transient_retryable_error(exc):
        return type(exc).__name__
    return None


def _resolve_retry_cfg(
    llm: Any,
    retry_cfg: Mapping[str, Any] | None = None,
) -> RetryPolicy:
    cfg = dict(retry_cfg or {})
    return _coerce_retry_policy(
        max_retries=cfg.get("max_retries", getattr(llm, "max_retries", 3)),
        base_delay_seconds=cfg.get("base_delay_seconds", 2.0),
        max_delay_seconds=cfg.get("max_delay_seconds", DEFAULT_RETRY_MAX_DELAY_SECONDS),
    )


def call_with_content_filter_retry(
    fn: Callable[[], _T],
    *,
    stage_label: str,
    logger: logging.Logger | None = None,
    max_retries: int = DEFAULT_CONTENT_FILTER_MAX_RETRIES,
    base_delay_seconds: float = DEFAULT_CONTENT_FILTER_RETRY_BASE_SECONDS,
    max_delay_seconds: float | None = None,
) -> _T:
    """Run a sync callable with retries for content-filter and transient API errors."""
    log = logger or logging.getLogger(__name__)
    policy = _coerce_retry_policy(
        max_retries=max_retries,
        base_delay_seconds=base_delay_seconds,
        max_delay_seconds=max_delay_seconds,
    )
    max_attempts = policy.max_retries + 1
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            reason = _classify_retryable_error(
                exc,
                retry_on_content_filter=True,
                retry_on_transient=True,
            )
            if reason is None:
                raise

            last_exc = exc
            if attempt == max_attempts:
                break

            wait_s = retry_delay_seconds(
                attempt,
                base_delay_seconds=policy.base_delay_seconds,
                max_delay_seconds=policy.max_delay_seconds,
            )
            log.warning(
                "[llm_retry] %s hit during %s (attempt %d/%d). Retrying in %.1fs.",
                reason,
                stage_label,
                attempt,
                max_attempts,
                wait_s,
            )
            time.sleep(wait_s)

    log.error(
        "[llm_retry] Retryable LLM error persisted during %s after %d attempts.",
        stage_label,
        max_attempts,
    )
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"[llm_retry] Unexpected retry failure in {stage_label}")


def invoke_with_content_filter_retry(
    llm: BaseChatModel,
    prompt: Any,
    *,
    stage_label: str,
    logger: logging.Logger | None = None,
    max_retries: int = DEFAULT_CONTENT_FILTER_MAX_RETRIES,
    base_delay_seconds: float = DEFAULT_CONTENT_FILTER_RETRY_BASE_SECONDS,
    max_delay_seconds: float | None = None,
) -> Any:
    """Invoke a chat model with retries for content-filter and transient API errors."""
    return call_with_content_filter_retry(
        lambda: llm.invoke(prompt),
        stage_label=stage_label,
        logger=logger,
        max_retries=max_retries,
        base_delay_seconds=base_delay_seconds,
        max_delay_seconds=max_delay_seconds,
    )


async def ainvoke_with_content_filter_retry(
    llm: BaseChatModel,
    prompt: Any,
    *,
    stage_label: str,
    logger: logging.Logger | None = None,
    max_retries: int = DEFAULT_CONTENT_FILTER_MAX_RETRIES,
    base_delay_seconds: float = DEFAULT_CONTENT_FILTER_RETRY_BASE_SECONDS,
    max_delay_seconds: float | None = None,
) -> Any:
    """Async invoke with retries for content-filter and transient API errors."""
    log = logger or logging.getLogger(__name__)
    policy = _coerce_retry_policy(
        max_retries=max_retries,
        base_delay_seconds=base_delay_seconds,
        max_delay_seconds=max_delay_seconds,
    )
    max_attempts = policy.max_retries + 1
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return await llm.ainvoke(prompt)
        except Exception as exc:
            reason = _classify_retryable_error(
                exc,
                retry_on_content_filter=True,
                retry_on_transient=True,
            )
            if reason is None:
                raise

            last_exc = exc
            if attempt == max_attempts:
                break

            wait_s = retry_delay_seconds(
                attempt,
                base_delay_seconds=policy.base_delay_seconds,
                max_delay_seconds=policy.max_delay_seconds,
            )
            log.warning(
                "[llm_retry] %s hit during %s (attempt %d/%d). Retrying in %.1fs.",
                reason,
                stage_label,
                attempt,
                max_attempts,
                wait_s,
            )
            await asyncio.sleep(wait_s)

    log.error(
        "[llm_retry] Retryable LLM error persisted during %s after %d attempts.",
        stage_label,
        max_attempts,
    )
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"[llm_retry] Unexpected retry failure in {stage_label}")


def invoke_with_retry(
    *,
    llm: Any,
    prompt_val: Any,
    logger: logging.Logger | None = None,
    context: str = "main-pipeline",
    retry_cfg: Mapping[str, Any] | None = None,
) -> Any:
    """Invoke `llm.invoke` with transient-error retry logs."""
    log = logger or logging.getLogger(__name__)
    policy = _resolve_retry_cfg(llm, retry_cfg)
    max_attempts = policy.max_retries + 1

    for attempt in range(1, max_attempts + 1):
        try:
            return llm.invoke(prompt_val)
        except Exception as exc:
            reason = _classify_retryable_error(
                exc,
                retry_on_content_filter=False,
                retry_on_transient=True,
            )
            if reason is None:
                raise
            if attempt == max_attempts:
                raise

            delay = retry_delay_seconds(
                attempt,
                base_delay_seconds=policy.base_delay_seconds,
                max_delay_seconds=policy.max_delay_seconds,
            )
            log.warning(
                "[retry][%s] LLM call failed (%s), retrying in %.1fs (attempt %d/%d)",
                context,
                reason,
                delay,
                attempt,
                max_attempts,
            )
            time.sleep(delay)


async def ainvoke_with_retry(
    *,
    llm: Any,
    prompt_val: Any,
    logger: logging.Logger | None = None,
    context: str = "main-pipeline",
    retry_cfg: Mapping[str, Any] | None = None,
) -> Any:
    """Invoke `llm.ainvoke` with transient-error retry logs."""
    log = logger or logging.getLogger(__name__)
    policy = _resolve_retry_cfg(llm, retry_cfg)
    max_attempts = policy.max_retries + 1

    for attempt in range(1, max_attempts + 1):
        try:
            return await llm.ainvoke(prompt_val)
        except Exception as exc:
            reason = _classify_retryable_error(
                exc,
                retry_on_content_filter=False,
                retry_on_transient=True,
            )
            if reason is None:
                raise
            if attempt == max_attempts:
                raise

            delay = retry_delay_seconds(
                attempt,
                base_delay_seconds=policy.base_delay_seconds,
                max_delay_seconds=policy.max_delay_seconds,
            )
            log.warning(
                "[retry][%s] LLM call failed (%s), retrying in %.1fs (attempt %d/%d)",
                context,
                reason,
                delay,
                attempt,
                max_attempts,
            )
            await asyncio.sleep(delay)
