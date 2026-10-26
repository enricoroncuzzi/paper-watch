"""The access log keeps the request line but not the search text.

The journal is the one place a query string survives a request: the app never logs a question, but
uvicorn's access line carries the full target, and `/items?q=...` is the corpus search. These tests
emit a record in uvicorn's own shape, so a change to the format (or a filter that stops being
installed) fails here rather than quietly writing search text to the server's journal.
"""

import logging

import pytest

from paperwatch.logging import configure_logging


def _emit_access(path: str) -> str:
    """The line uvicorn's access logger produces for one request, after configuration."""
    configure_logging("INFO")
    logger = logging.getLogger("uvicorn.access")
    # uvicorn's own log config sets the access logger's level; the root logger's is the test
    # runner's, so the level is set here rather than inherited.
    logger.setLevel(logging.INFO)
    captured: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    handler = Capture()
    # On the logger itself, so the test does not depend on what the root logger carries: pytest's
    # own handlers are there, and a test that silenced the process would otherwise pass silently.
    logger.addHandler(handler)
    try:
        # The record format is uvicorn's own: client, method, full target, version, status. They go
        # in as separate arguments, which is how uvicorn emits them and how logging stores them.
        logger.info(
            '%s - "%s %s HTTP/%s" %d',
            "203.0.113.7:51000",
            "GET",
            path,
            "1.1",
            200,
        )
    finally:
        logger.removeHandler(handler)
    [record] = captured
    return record.getMessage()


def test_the_access_log_drops_the_query_string() -> None:
    line = _emit_access("/items?q=private+search+text&limit=25")
    assert line == '203.0.113.7:51000 - "GET /items HTTP/1.1" 200'
    assert "private" not in line


def test_the_access_log_keeps_a_path_that_has_no_query_string() -> None:
    assert _emit_access("/threads") == '203.0.113.7:51000 - "GET /threads HTTP/1.1" 200'


def test_the_filter_is_installed_once(monkeypatch: pytest.MonkeyPatch) -> None:
    # `serve` and every other command call configure_logging, and a test suite calls it many times;
    # a filter added per call would grow the logger's list on every invocation.
    configure_logging("INFO")
    configure_logging("INFO")
    logger = logging.getLogger("uvicorn.access")
    assert sum(1 for f in logger.filters if type(f).__name__ == "DropQueryString") == 1
