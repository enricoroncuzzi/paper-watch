"""Pacing a finished SSE frame stream: heartbeats while it is silent, a deadline that ends it.

The producer is a real thread and the clock is fake, so a test can place 46 seconds of silence in
a few milliseconds and watch the deadline fire without waiting two minutes.
"""

import threading
import time
from collections.abc import Iterator

import pytest

from paperwatch.api.pacing import paced

PING = ": ping\n\n"
DEADLINE = 'event: error\ndata: {"code": "deadline"}\n\n'


class Clock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _paced(
    events: Iterator[str],
    clock: Clock,
    cancel: threading.Event,
    *,
    heartbeat_s: float = 15,
    deadline_s: float = 120,
) -> Iterator[str]:
    return paced(
        events,
        cancel=cancel,
        clock=clock,
        heartbeat_s=heartbeat_s,
        deadline_s=deadline_s,
        heartbeat=lambda: PING,
        deadline_error=lambda: DEADLINE,
    )


def _silent(cancel: threading.Event, started: threading.Event, stopped: threading.Event):
    # One frame first, so a test can start the generator (the producer thread starts on the first
    # next()) and only then move its fake clock past the heartbeat interval.
    started.set()
    yield "start\n\n"
    try:
        while not cancel.is_set():
            time.sleep(0.002)
        yield ""  # pragma: no cover - reached only when the consumer closed the generator
    finally:
        stopped.set()


def _frames(frames: list[str], cancel: threading.Event, stopped: threading.Event):
    try:
        yield from frames
        while not cancel.is_set():
            time.sleep(0.002)
        yield ""  # pragma: no cover
    finally:
        stopped.set()


def test_a_silent_answer_gets_heartbeats() -> None:
    cancel, started, stopped = threading.Event(), threading.Event(), threading.Event()
    clock = Clock()
    generator = _paced(_silent(cancel, started, stopped), clock, cancel)
    assert next(generator) == "start\n\n"
    assert started.wait(1)
    clock.now = 16
    assert next(generator) == PING
    clock.now = 31
    assert next(generator) == PING
    clock.now = 46
    assert next(generator) == PING
    generator.close()
    assert stopped.wait(1)


def test_no_heartbeat_while_frames_flow() -> None:
    # A long answer that keeps writing must not look silent: every frame resets the heartbeat.
    cancel = threading.Event()

    def events() -> Iterator[str]:
        for index in range(4):
            yield f"frame-{index}\n\n"
            time.sleep(0.005)

    clock = Clock()
    generator = _paced(events(), clock, cancel)
    received = []
    for _ in range(4):
        # Time passes between frames, but never more than the heartbeat interval since the last.
        clock.now += 10
        received.append(next(generator))
    assert received == [f"frame-{index}\n\n" for index in range(4)]
    generator.close()


def test_the_deadline_ends_the_stream_with_an_error_event() -> None:
    cancel, started, stopped = threading.Event(), threading.Event(), threading.Event()
    clock = Clock()
    generator = _paced(_silent(cancel, started, stopped), clock, cancel)
    assert next(generator) == "start\n\n"
    clock.now = 121
    frames = list(generator)
    assert frames == [DEADLINE]
    assert cancel.is_set()
    assert stopped.wait(1)


def test_closing_the_consumer_stops_the_producer() -> None:
    cancel, started, stopped = threading.Event(), threading.Event(), threading.Event()
    clock = Clock()
    generator = _paced(_silent(cancel, started, stopped), clock, cancel)
    assert next(generator) == "start\n\n"
    assert started.wait(1)
    generator.close()
    assert cancel.is_set()
    assert stopped.wait(1), "the producer thread did not stop after the consumer closed"


def test_a_producer_error_reaches_the_consumer() -> None:
    cancel = threading.Event()

    def events() -> Iterator[str]:
        yield "one\n\n"
        raise RuntimeError("the graph broke")

    clock = Clock()
    generator = _paced(events(), clock, cancel)
    assert next(generator) == "one\n\n"
    with pytest.raises(RuntimeError, match="the graph broke"):
        next(generator)


def test_closing_the_consumer_waits_for_the_producer() -> None:
    # N5: close joins the producer. The producer takes 100 ms to stop after the cancel, so a
    # missing join returns before `stopped` is set.
    cancel, stopped = threading.Event(), threading.Event()

    def events() -> Iterator[str]:
        yield "start\n\n"
        while not cancel.is_set():
            time.sleep(0.002)
        time.sleep(0.1)
        stopped.set()

    generator = _paced(events(), Clock(), cancel)
    assert next(generator) == "start\n\n"
    generator.close()
    assert stopped.is_set()
