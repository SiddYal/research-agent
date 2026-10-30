"""Retry-with-backoff for every external call (LLM, embeddings, search, fetch).

Only transient failures are retried; a 400 or an auth error fails immediately.
"""

import httpx
import tenacity

from research_agent.logging import get_logger

log = get_logger(__name__)

_TRANSIENT: list[type[BaseException]] = [
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
    TimeoutError,
    ConnectionError,
]

try:
    import openai

    _TRANSIENT += [
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.RateLimitError,
        openai.InternalServerError,
    ]
except ImportError:  # pragma: no cover
    pass

try:
    import anthropic

    _TRANSIENT += [
        anthropic.APIConnectionError,
        anthropic.APITimeoutError,
        anthropic.RateLimitError,
        anthropic.InternalServerError,
    ]
except ImportError:  # pragma: no cover
    pass


class TransientError(Exception):
    """Raise to force a retry for a failure that isn't a known transient type."""


TRANSIENT_ERRORS = tuple(_TRANSIENT) + (TransientError,)


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (408, 425, 429) or exc.response.status_code >= 500
    return isinstance(exc, TRANSIENT_ERRORS)


def _log_retry(state: tenacity.RetryCallState) -> None:
    exc = state.outcome.exception() if state.outcome else None
    log.warning(
        "retrying_external_call",
        fn=getattr(state.fn, "__qualname__", str(state.fn)),
        attempt=state.attempt_number,
        error=repr(exc),
    )


def with_retry(max_attempts: int = 4, max_wait_s: float = 20.0):
    return tenacity.retry(
        retry=tenacity.retry_if_exception(_is_transient),
        stop=tenacity.stop_after_attempt(max_attempts),
        wait=tenacity.wait_exponential_jitter(initial=1, max=max_wait_s),
        before_sleep=_log_retry,
        reraise=True,
    )
