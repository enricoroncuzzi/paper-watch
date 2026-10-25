#!/usr/bin/env python3
"""The soak's SSE test server: a heartbeat stream on 127.0.0.1:8443, for the tunnel soak only.

The 72-hour soak in docs/deploy/vps.md needs a stream that keeps flowing through the public
tunnel for five minutes at a time, so that a tunnel break shows up as a gap between frames and not
only as a refused connection. It is not part of the product: nothing reads it, and it is published
by a second Funnel mapping (and a second systemd unit) for the duration of the soak only.

Standard library only on purpose. The soak runs from the operator's machine and from a fresh
account on the server, and neither should need an install for it.
"""

import argparse
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

# The probe's own default (deploy/vps/tunnel-soak.sh --heartbeat): one frame every fifteen seconds
# is frequent enough to time a restart's recovery to well inside the two-minute rule.
HEARTBEAT_S = 15.0


class StreamHandler(BaseHTTPRequestHandler):
    """A never-ending SSE response: one data frame, then a heartbeat on the same interval."""

    heartbeat = HEARTBEAT_S

    def do_GET(self) -> None:  # http.server's spelling, not ours
        if self.path.split("?", 1)[0] != "/events":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        tick = 0
        try:
            while True:
                self.wfile.write(f'data: {{"tick": {tick}}}\n\n'.encode())
                # Unbuffered would still risk a partial frame across the tunnel: flush the whole
                # frame before the heartbeat sleep so curl reads one frame per interval.
                self.wfile.flush()
                tick += 1
                time.sleep(self.heartbeat)
        except (BrokenPipeError, ConnectionResetError):
            # The probe's --max-time ends the stream from its side, and that is the normal exit.
            return

    def log_message(self, *_args: Any) -> None:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The Paper Watch soak's SSE test stream.")
    # Loopback only: the tunnel's own Funnel mapping publishes it for the soak.
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8443)
    parser.add_argument("--heartbeat", type=float, default=HEARTBEAT_S)
    args = parser.parse_args(argv)

    handler = type("SoakStreamHandler", (StreamHandler,), {"heartbeat": args.heartbeat})
    server = ThreadingHTTPServer((args.host, args.port), handler)
    host, port = server.server_address[0], server.server_address[1]
    # Port 0 binds an ephemeral port; printing the real one is how the tests (and a person who
    # wants the stream somewhere else) learn where it went.
    print(f"listening on http://{host}:{port}/events every {args.heartbeat:g}s", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
