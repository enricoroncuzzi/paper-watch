"""A stand-in embedding child for the WorkerEmbedder tests.

It speaks the worker protocol through the real `serve()` loop and the deterministic `FakeEmbedder`,
so a test exercises the framing, the timeout and the respawn without loading a model. Two
environment values reproduce the failures a real child can have: `FAKE_WORKER_MODE=hang` never
answers, and `crash` exits on the first request. The `__oom__` probe reports the child's own
`oom_score_adj`, which is how the Linux-only test proves the child volunteers to the OOM killer.
"""

import os
import sys
import time
from pathlib import Path

from paperwatch.llm.embed_worker import serve
from paperwatch.llm.embedder import FakeEmbedder


class ModeAwareEmbedder:
    def __init__(self) -> None:
        self._inner = FakeEmbedder()

    def embed(self, texts: list[str]) -> list[list[float]]:
        mode = os.environ.get("FAKE_WORKER_MODE")
        if mode == "hang":
            time.sleep(600)
        if mode == "crash":
            # SystemExit is not an Exception, so `serve()` lets the process die as a real crash.
            raise SystemExit(1)
        if texts == ["__oom__"]:
            try:
                value = float(Path("/proc/self/oom_score_adj").read_text().strip())
            except OSError:
                value = -1.0
            return [[value]]
        return self._inner.embed(texts)


def main() -> None:
    serve(ModeAwareEmbedder(), sys.stdin, sys.stdout)


if __name__ == "__main__":
    main()
