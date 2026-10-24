"""The embedding model in a child process: one JSON line in, one out, and an idle exit.

The child is the fake worker in `tests/helpers/`, so these tests exercise the protocol, the
timeout, the respawn and the idle reaper without loading the real 180 MB model. The clock is
injected, so "ten idle minutes" takes no real time.
"""

import json
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from paperwatch.config import Settings
from paperwatch.llm.embed_worker import serve
from paperwatch.llm.embedder import (
    FakeEmbedder,
    WorkerEmbedder,
    clip_for_embedding,
)

FAKE = Path(__file__).resolve().parents[1] / "helpers" / "fake_embed_worker.py"


class Clock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def workers() -> Iterator[Callable[..., WorkerEmbedder]]:
    created: list[WorkerEmbedder] = []

    def build(**kwargs: object) -> WorkerEmbedder:
        worker = WorkerEmbedder([sys.executable, str(FAKE)], **kwargs)  # type: ignore[arg-type]
        created.append(worker)
        return worker

    yield build
    for worker in created:
        worker.stop()


def test_embeds_through_the_child_and_matches_the_fake_embedder(
    workers: Callable[..., WorkerEmbedder],
) -> None:
    worker = workers()
    assert worker.embed(["hello world"]) == FakeEmbedder().embed(["hello world"])
    assert worker.alive()


def test_prewarm_returns_at_once_and_starts_the_child(
    workers: Callable[..., WorkerEmbedder],
) -> None:
    worker = workers()
    started = time.monotonic()
    worker.prewarm()
    assert time.monotonic() - started < 1.0, "prewarm must not wait for the model to load"
    deadline = time.monotonic() + 20
    while not worker.alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert worker.alive()


def test_queries_are_clipped_to_600_bytes(workers: Callable[..., WorkerEmbedder]) -> None:
    worker = workers()
    text = "x" * 2_000
    assert worker.embed([text]) == FakeEmbedder().embed([clip_for_embedding(text, 600)])


def test_a_hung_child_times_out_and_is_killed(
    workers: Callable[..., WorkerEmbedder], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_WORKER_MODE", "hang")
    worker = workers(timeout_s=0.5)
    with pytest.raises(Exception, match="timed out"):
        worker.embed(["hello"])
    assert not worker.alive(), "a timed-out child must be killed, not left to answer later"


def test_a_dead_child_raises_embedder_unavailable_and_the_next_call_respawns(
    workers: Callable[..., WorkerEmbedder], monkeypatch: pytest.MonkeyPatch
) -> None:
    from paperwatch.llm.embedder import EmbedderUnavailable

    monkeypatch.setenv("FAKE_WORKER_MODE", "crash")
    worker = workers()
    with pytest.raises(EmbedderUnavailable):
        worker.embed(["hello"])
    assert not worker.alive()
    # The next question starts a fresh child, so one crash does not take search down for the day.
    monkeypatch.delenv("FAKE_WORKER_MODE")
    assert worker.embed(["hello"]) == FakeEmbedder().embed(["hello"])


def test_the_child_exits_after_idle_but_never_while_an_answer_is_open(
    workers: Callable[..., WorkerEmbedder],
) -> None:
    clock = Clock()
    worker = workers(idle_s=10, clock=clock)
    worker.embed(["hello"])
    assert worker.alive()
    clock.now += 9
    worker.reap_once()
    assert worker.alive()
    with worker.answer_open():
        clock.now += 20
        worker.reap_once()
        assert worker.alive(), "an open answer keeps the model alive"
    clock.now += 11
    worker.reap_once()
    assert not worker.alive()


def test_a_true_should_stop_kills_the_child_at_the_next_reap(
    workers: Callable[..., WorkerEmbedder],
) -> None:
    stop = False
    worker = workers(should_stop=lambda: stop)
    worker.embed(["hello"])
    assert worker.alive()
    stop = True
    worker.reap_once()
    assert not worker.alive()


@pytest.mark.skipif(sys.platform != "linux", reason="oom_score_adj is a Linux interface")
def test_the_worker_raises_its_own_oom_score(workers: Callable[..., WorkerEmbedder]) -> None:
    # The API side must be the one the kernel picks when the shared slice runs out of memory.
    worker = workers()
    assert worker.embed(["__oom__"])[0][0] == 1000


@pytest.mark.skipif(sys.platform != "linux", reason="the RSS figures come from /proc")
def test_the_child_memory_is_the_real_proc_figure(
    workers: Callable[..., WorkerEmbedder],
) -> None:
    # The default reader against a real child and a real /proc: the child holds the interpreter and
    # the fake model, so its anonymous memory is a plausible figure rather than None.
    worker = workers()
    worker.embed(["hello"])
    rss = worker.child_rss_mb()
    assert rss is not None and 5 < rss < 1_000


def test_no_child_holds_no_memory(workers: Callable[..., WorkerEmbedder]) -> None:
    # Nothing was spawned yet: the gate gets None and counts the child as zero.
    assert workers().child_rss_mb() is None


def test_a_dead_child_holds_no_memory(workers: Callable[..., WorkerEmbedder]) -> None:
    worker = workers()
    worker.embed(["hello"])
    worker.stop()
    assert worker.child_rss_mb() is None


def test_a_crashed_child_holds_no_memory(
    workers: Callable[..., WorkerEmbedder], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The child died on its own and the worker noticed: there is no process left whose memory the
    # gate could count.
    from paperwatch.llm.embedder import EmbedderUnavailable

    monkeypatch.setenv("FAKE_WORKER_MODE", "crash")
    worker = workers(read_status=lambda pid: "RssAnon:\t 225280 kB\n")
    with pytest.raises(EmbedderUnavailable):
        worker.embed(["hello"])
    assert worker.child_rss_mb() is None


def test_a_finished_child_holds_no_memory(workers: Callable[..., WorkerEmbedder]) -> None:
    # A child that died while its Popen is still held (poll has not run yet) is gone: the gate must
    # count nothing rather than the figure of a pid that no longer holds any memory.
    worker = workers(read_status=lambda pid: "RssAnon:\t 225280 kB\n")
    worker.embed(["hello"])
    process = worker._process  # type: ignore[attr-defined]
    assert process is not None
    process.kill()
    process.wait()
    assert worker.child_rss_mb() is None


def test_an_unreadable_child_status_holds_no_memory(
    workers: Callable[..., WorkerEmbedder],
) -> None:
    # A /proc read that fails says nothing about the child's memory, so it counts as zero rather
    # than as a reason not to trust the reading: the gate must still be able to say "short".
    worker = workers(read_status=lambda pid: None)
    worker.embed(["hello"])
    assert worker.child_rss_mb() is None


def test_the_child_reports_its_own_anonymous_memory(
    workers: Callable[..., WorkerEmbedder],
) -> None:
    # RssAnon is the memory killing the child frees; RssFile is page cache the kernel already
    # counts as reclaimable, so counting it would make the gate overestimate what a kill buys.
    seen: list[int] = []
    status = "Name:\tpython\nVmRSS:\t  300000 kB\nRssAnon:\t  225280 kB\nRssFile:\t   47104 kB\n"

    def read_status(pid: int) -> str | None:
        seen.append(pid)
        return status

    worker = workers(read_status=read_status)
    worker.embed(["hello"])
    assert worker.child_rss_mb() == 220.0
    assert seen and seen[-1] == worker._process.pid  # type: ignore[attr-defined]


def test_serve_reports_an_error_line_without_dying() -> None:
    # A bad request is a line the parent can read, not a crash: the API turns it into a text-only
    # search for that call and keeps the child for the next one.
    import io

    class Broken:
        def embed(self, texts: list[str]) -> list[list[float]]:
            raise ValueError("bad text")

    inp = io.StringIO(json.dumps({"texts": ["x"]}) + "\n")
    out = io.StringIO()
    serve(Broken(), inp, out)  # type: ignore[arg-type]
    assert json.loads(out.getvalue()) == {"error": "ValueError"}


def test_a_failing_stop_check_keeps_the_reaper_alive(
    workers: Callable[..., WorkerEmbedder],
) -> None:
    # A database error in the mode check is already handled by the gate, but a callback that raises
    # must not kill the reaper thread: the child would then never exit when it goes idle.
    def broken() -> bool:
        raise RuntimeError("the mode check is down")

    worker = workers(should_stop=broken, idle_s=10, clock=Clock())
    worker.embed(["hello"])
    worker.reap_once()  # must not raise
    assert worker.alive()


def test_raise_oom_score_writes_the_highest_value(tmp_path: Path) -> None:
    # The write itself, against a file instead of /proc, so it is testable on every platform.
    from paperwatch.llm.embed_worker import raise_oom_score

    target = tmp_path / "oom_score_adj"
    raise_oom_score(target)
    assert target.read_text() == "1000"


def test_main_raises_the_oom_score_before_serving(monkeypatch: pytest.MonkeyPatch) -> None:
    # E1: the call in `main`, not just the helper, is what a real child relies on.
    from paperwatch.llm import embed_worker

    order: list[str] = []
    monkeypatch.setattr(embed_worker, "raise_oom_score", lambda: order.append("oom"))
    monkeypatch.setattr(embed_worker, "get_settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr(embed_worker, "make_local_model", lambda settings: object())
    monkeypatch.setattr(embed_worker, "LocalEmbedder", lambda model, batch: object())
    monkeypatch.setattr(embed_worker, "serve", lambda *args: order.append("serve"))
    embed_worker.main()
    assert order == ["oom", "serve"]
