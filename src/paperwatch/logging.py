"""Structured JSON logging: one JSON object per line, easy to search in Azure Monitor."""

# Absolute import: this is the standard library module, not this file.
import logging

import structlog


def configure_logging(level: str = "INFO") -> None:
    # Route stdlib logging (used by libraries like httpx and SQLAlchemy) to stderr at this level.
    logging.basicConfig(format="%(message)s", level=level)
    structlog.configure(
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
