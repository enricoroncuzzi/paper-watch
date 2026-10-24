import { describe, expect, it } from "vitest";

import { parseSse, type SseEvent } from "./sse";

function streamOf(chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
}

async function collect(
  body: ReadableStream<Uint8Array>,
  onBytes?: () => void,
): Promise<SseEvent[]> {
  const events: SseEvent[] = [];
  for await (const event of parseSse(body, onBytes)) events.push(event);
  return events;
}

describe("the SSE parser", () => {
  it("handles frames split across chunks, inside data and inside the blank line", async () => {
    const events = await collect(
      streamOf(["event: tok", 'en\ndata: {"te', 'xt":"hi"}\n', "\n"]),
    );
    expect(events).toEqual([{ event: "token", data: { text: "hi" } }]);
  });

  it("ignores comments and joins multiple data lines", async () => {
    const events = await collect(
      streamOf([": ping\n\n", 'event: answer\ndata: {"a":1,\ndata: "b":2}\n\n']),
    );
    expect(events).toEqual([{ event: "answer", data: { a: 1, b: 2 } }]);
  });

  it("accepts CRLF line endings", async () => {
    const events = await collect(streamOf(['event: token\r\ndata: {"text":"x"}\r\n\r\n']));
    expect(events).toEqual([{ event: "token", data: { text: "x" } }]);
  });

  it("reports every received chunk, including a heartbeat", async () => {
    let chunks = 0;
    await collect(streamOf([": ping\n\n", "event: token\ndata: {}\n\n"]), () => {
      chunks += 1;
    });
    // The heartbeat frame is the second chunk: the watchdog resets on it even with no event.
    expect(chunks).toBe(2);
  });

  it("skips a frame whose data is not JSON", async () => {
    const events = await collect(streamOf(["event: token\ndata: not-json\n\n"]));
    expect(events).toEqual([]);
  });
});
