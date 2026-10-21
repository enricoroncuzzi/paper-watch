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


def configure_logging(level: str = "INFO") -> None:
    # Progress and logs go to stderr, so a piped answer (`paperwatch ask q > answer.txt`) stays
    # just the answer. This covers the stdlib logging that httpx and SQLAlchemy use, and structlog
    # itself, whose default logger would otherwise print to stdout, inside the answer.
    sink = _StderrAtCallTime()
    logging.basicConfig(format="%(message)s", level=level, stream=cast(TextIO, sink))
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
