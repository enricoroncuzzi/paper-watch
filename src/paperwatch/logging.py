"""Structured JSON logging: one JSON object per line, easy to search in Azure Monitor."""

# Absolute import: this is the standard library module, not this file.
import logging
import sys
from typing import Any, TextIO, cast

import structlog


class _StderrAtCallTime:
    """A file that always writes to the current `sys.stderr`.

    A logger keeps the stream it was built with, and a test runner (pytest's capture, Click's
    `CliRunner`) replaces and then closes `sys.stderr` between runs. A logger built earlier would
    write to that closed file and raise ValueError on the next log line, so the stream is
    resolved on every write instead.
    """

    def write(self, text: str) -> int:
        return sys.stderr.write(text)

    def flush(self) -> None:
        sys.stderr.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(sys.stderr, name)


class DropQueryString(logging.Filter):
    """Keep uvicorn's access line but not the query string it carries.

    The corpus table's search is `GET /items?q=...`, and the API never logs a question or a search
    itself; the one place that text would reach the server's journal is the access record, which
    carries the whole target. The method, path and status are what an access log is for, so the
    target is cut at the `?` rather than the line being dropped.

    uvicorn emits one record per request with five arguments - client, method, target with its
    query string, HTTP version and status - and the default formatter interpolates them, so
    rewriting the arguments is enough.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if isinstance(args, tuple) and len(args) == 5 and isinstance(args[2], str):
            record.args = (args[0], args[1], args[2].split("?", 1)[0], args[3], args[4])
        return True


def configure_logging(level: str = "INFO") -> None:
    # Progress and logs go to stderr, so a piped answer (`paperwatch ask q > answer.txt`) stays
    # just the answer. This covers the stdlib logging that httpx and SQLAlchemy use, and structlog
    # itself, whose default logger would otherwise print to stdout, inside the answer.
    sink = _StderrAtCallTime()
    logging.basicConfig(format="%(message)s", level=level, stream=cast(TextIO, sink))
    access = logging.getLogger("uvicorn.access")
    # Added once: every command configures logging, and a filter per call would accumulate.
    if not any(isinstance(existing, DropQueryString) for existing in access.filters):
        access.addFilter(DropQueryString())
    structlog.configure(
        # Resolves stderr on every write, for the reason in `_StderrAtCallTime`.
        logger_factory=structlog.PrintLoggerFactory(file=cast(TextIO, sink)),
        # Each log call flows through these steps in order, building one event dict.
        processors=[
            # Adds request-scoped context bound with structlog.contextvars (e.g. run_id).
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            # UTC ISO timestamps, so logs from different machines line up.
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            # Last step: turn the event dict into a JSON line.
            structlog.processors.JSONRenderer(),
        ],
        # Drops calls below the level cheaply, before any processor runs.
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level)),
    )
