"""The embedding-model child process: one JSON request line in, one response line out.

The API holds no model. A worker starts when a question arrives, loads the model, answers embeds,
and exits when it is idle or when the ingestion needs the memory. This module is the child's side:
`serve` is the loop, which takes any `Embedder` so a test can run it with the deterministic fake,
and `main` loads the real local model.

The child raises its own OOM score first, so when the shared memory slice runs out the kernel picks
the API's model rather than the ingestion.
"""

import contextlib
import json
import sys
from pathlib import Path
from typing import TextIO

from paperwatch.config import get_settings
from paperwatch.llm.embedder import Embedder, LocalEmbedder, make_local_model

# The kernel's range, highest first: this process volunteers to be killed before anything else in
# the slice. Writing it is Linux-only, and an unprivileged process may only raise its own score.
OOM_SCORE_ADJ_PATH = Path("/proc/self/oom_score_adj")
OOM_SCORE_ADJ_MAX = "1000"


def serve(embedder: Embedder, inp: TextIO, out: TextIO) -> None:
    """Answer one request per input line until EOF.

    A bad request or a model error becomes an `{"error": "<ExceptionType>"}` line, never a crash:
    the parent turns it into a keyword-only search for that call, and the child stays for the next
    one. The exception's message is not sent, because a model error can echo the question.
    """
    for line in inp:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            texts = request["texts"]
            if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
                raise ValueError("texts must be a list of strings")
            response: dict[str, object] = {"vectors": embedder.embed(texts)}
        except Exception as exc:  # noqa: BLE001 - every failure is a line the parent can read
            response = {"error": type(exc).__name__}
        out.write(json.dumps(response) + "\n")
        out.flush()


def raise_oom_score(path: Path = OOM_SCORE_ADJ_PATH) -> None:
    """Volunteer this process to the OOM killer. A Mac has no /proc; a sandbox may refuse."""
    with contextlib.suppress(OSError):
        path.write_text(OOM_SCORE_ADJ_MAX)


def main() -> None:
    raise_oom_score()
    settings = get_settings()
    embedder = LocalEmbedder(make_local_model(settings), settings.local_embedding_batch)
    serve(embedder, sys.stdin, sys.stdout)


if __name__ == "__main__":
    main()
