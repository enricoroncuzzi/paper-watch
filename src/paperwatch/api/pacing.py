"""Pace a finished SSE frame stream: a heartbeat while it is silent, a deadline that ends it.

`run_question` produces frames only when the model or a tool has something to say, and a free model
can think for a minute. A client that sees nothing cannot tell thinking from a dead connection, and
iOS drops a long silent stream, so this wrapper sends an SSE comment line (`: ping`) every 15 s of
silence and ends the answer with a `deadline` error at 120 s, setting the cancel event so the model
call in flight stops.

The producer runs in a thread because the frame generator blocks in `next()`; the consumer polls a
queue with a short real timeout so an injected clock still decides when a heartbeat or the deadline
is due. Closing the consumer sets the cancel event and joins the producer, on the worker thread
`_ClosingStream` already uses for exactly this reason.
"""

import queue
import threading
from collections.abc import Callable, Generator, Iterator

# A poll interval, not a delay: it bounds how late a heartbeat can be, and the consumer yields as
# soon as a frame arrives. Short enough to be invisible, long enough not to spin a core.
POLL_S = 0.01
# A producer that does not stop on cancel is broken; joining forever would hang the request that
# is trying to close it.
JOIN_TIMEOUT_S = 5.0

_DONE = object()
_EMPTY = object()


def paced(
    events: Iterator[str],
    *,
    cancel: threading.Event,
    clock: Callable[[], float],
    heartbeat_s: float,
    deadline_s: float,
    heartbeat: Callable[[], str],
    deadline_error: Callable[[], str],
) -> Generator[str, None, None]:
    frames: queue.Queue[object] = queue.Queue()
    errors: list[BaseException] = []

    def produce() -> None:
        try:
            for frame in events:
                frames.put(frame)
        except BaseException as exc:  # noqa: BLE001 - re-raised in the consumer
            errors.append(exc)
        finally:
            frames.put(_DONE)

    producer = threading.Thread(target=produce, name="sse-pacer", daemon=True)
    producer.start()
    started = last_frame = clock()
    try:
        while True:
            # No per-frame cancel check: Starlette's TestClient reports http.disconnect as soon as
            # the request body is read, so checking here would truncate a stream that a real
            # client is still reading. `_ClosingStream` closes this generator on a hang-up, which
            # runs the finally below (cancel and join) at once.
            now = clock()
            if now - started >= deadline_s:
                # Set the flag before yielding: the model call in flight hears it at its next token,
                # so the answer stops at the deadline instead of after it.
                cancel.set()
                yield deadline_error()
                return
            # A short poll: the loop reads the injected clock for the heartbeat and the deadline,
            # so a plain timeout is enough. Two answers at most, so the wake-ups are negligible.
            try:
                item = frames.get(timeout=POLL_S)
            except queue.Empty:
                item = _EMPTY
            if item is _DONE:
                if errors:
                    raise errors[0]
                return
            if item is not _EMPTY:
                last_frame = clock()
                yield item  # type: ignore[misc]
                continue
            if clock() - last_frame >= heartbeat_s:
                last_frame = clock()
                yield heartbeat()
    finally:
        cancel.set()
        producer.join(timeout=JOIN_TIMEOUT_S)
