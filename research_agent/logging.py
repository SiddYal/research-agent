"""Structured application logging.

Every log line carries whatever is bound in structlog's contextvars, so the API
middleware binds `request_id` and the job runner binds `job_id`; an error can be
traced back to the request that caused it from the logs alone.
"""

import logging
import sys

import structlog

_configured = False


def configure_logging(level: str = "INFO", json: bool = False) -> None:
    global _configured
    if _configured:
        return
    renderer = structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer()
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level.upper())),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
    # Quiet chatty third-party loggers that go through stdlib logging.
    for name in ("httpx", "httpcore", "trafilatura", "primp", "ddgs"):
        logging.getLogger(name).setLevel(logging.WARNING)
    _configured = True


def get_logger(name: str | None = None):
    return structlog.get_logger(name)
