/**
 * The server-sent-events parser for `POST /ask`.
 *
 * The API sends one `event:` line and one `data:` line per frame, `: ping` comment lines while it
 * thinks, and nothing else. A chunk boundary can fall anywhere, including inside `data:` or the
 * blank line that ends a frame, so parsing has to keep a buffer between reads.
 */
export type SseEvent = { event: string; data: unknown };

function parseFrame(frame: string): SseEvent | null {
  let event = "message";
  const data: string[] = [];
  for (const rawLine of frame.split("\n")) {
    // A frame can end with \r\n; the \r belongs to the line ending, not the value.
    const line = rawLine.endsWith("\r") ? rawLine.slice(0, -1) : rawLine;
    if (line === "" || line.startsWith(":")) continue;
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") event = value;
    if (field === "data") data.push(value);
  }
  // A comment-only frame (the `: ping` heartbeat) has no data and yields no event.
  if (data.length === 0) return null;
  try {
    return { event, data: JSON.parse(data.join("\n")) };
  } catch {
    return null;
  }
}

/**
 * Yields one event per frame. `onBytes` runs for every received chunk before parsing, so the
 * caller's watchdog resets on a `: ping` comment even though that frame yields nothing.
 */
export async function* parseSse(
  body: ReadableStream<Uint8Array>,
  onBytes?: () => void,
): AsyncGenerator<SseEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      onBytes?.();
      // A frame can end with CRLF, and a chunk can split between the \r and the \n; normalising
      // the accumulated buffer on every read handles both without a second delimiter scan.
      buffer = (buffer + decoder.decode(value, { stream: true })).replace(/\r\n/g, "\n");
      let boundary = buffer.indexOf("\n\n");
      while (boundary !== -1) {
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const event = parseFrame(frame);
        if (event) yield event;
        boundary = buffer.indexOf("\n\n");
      }
    }
  } finally {
    reader.releaseLock();
  }
}
