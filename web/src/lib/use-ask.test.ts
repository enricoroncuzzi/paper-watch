import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  askReducer,
  AskController,
  footnoteNumbers,
  INITIAL_ASK_STATE,
  type AskState,
} from "./use-ask";

/** A stream the test feeds by hand, the way the network would. */
function controllable() {
  const encoder = new TextEncoder();
  let inner!: ReadableStreamDefaultController<Uint8Array>;
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      inner = controller;
    },
  });
  return {
    body,
    push: (text: string) => inner.enqueue(encoder.encode(text)),
    close: () => inner.close(),
  };
}

function frame(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

function controllerFor(body: ReadableStream<Uint8Array>, status = 200) {
  const fetchImpl = vi.fn().mockResolvedValue(new Response(body, { status }));
  return { fetchImpl, controller: new AskController({ fetchImpl, watchdogMs: 45_000 }) };
}

const ANSWER = {
  text: "Coding agents are measured on real repository bugs [S1].",
  search_mode: "hybrid",
  supported: true,
  citations: [
    {
      id: "S1",
      item_id: "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
      title: "SWE-bench Verified",
      url: "https://arxiv.org/abs/2601.00001",
      tier: "peer_reviewed",
      basis: "venue",
    },
  ],
  flags: [],
};

describe("the ask reducer", () => {
  it("replaces the streamed text with the checked answer", () => {
    let state: AskState = askReducer(INITIAL_ASK_STATE, { type: "start", question: "q" });
    state = askReducer(state, { type: "event", event: "token", data: { text: "draft words" } });
    expect(state.text).toBe("draft words");
    state = askReducer(state, { type: "event", event: "answer", data: ANSWER });
    expect(state.status).toBe("done");
    expect(state.text).toBe(ANSWER.text);
    expect(state.answer?.citations).toHaveLength(1);
  });

  it("keeps one entry per source alias", () => {
    let state: AskState = askReducer(INITIAL_ASK_STATE, { type: "start", question: "q" });
    const source = ANSWER.citations[0]!;
    state = askReducer(state, { type: "event", event: "source", data: source });
    state = askReducer(state, { type: "event", event: "source", data: source });
    expect(state.sources).toHaveLength(1);
  });
});

describe("footnote numbers", () => {
  it("follows the citation order, not the alias number", () => {
    // The API lists only the cited sources, in first-cited order: [S2] then [S1] means 1 and 2.
    const numbers = footnoteNumbers([ANSWER.citations[0]!, { ...ANSWER.citations[0]!, id: "S2" }]);
    expect(numbers.get("S1")).toBe(1);
    expect(numbers.get("S2")).toBe(2);
  });
});

describe("the ask controller", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("a silent stream becomes dropped after 45 seconds", async () => {
    const stream = controllable();
    const { controller } = controllerFor(stream.body);
    const run = controller.ask("q");
    await vi.advanceTimersByTimeAsync(0);
    expect(controller.getState().status).toBe("streaming");

    await vi.advanceTimersByTimeAsync(45_000);
    expect(controller.getState().status).toBe("dropped");
    stream.close();
    await run;
  });

  it("ping comment lines keep the watchdog alive", async () => {
    const stream = controllable();
    const { controller } = controllerFor(stream.body);
    const run = controller.ask("q");
    await vi.advanceTimersByTimeAsync(0);

    // Two heartbeats 30 s apart: without them the 45 s watchdog would have fired.
    await vi.advanceTimersByTimeAsync(30_000);
    stream.push(": ping\n\n");
    await vi.advanceTimersByTimeAsync(30_000);
    stream.push(": ping\n\n");
    await vi.advanceTimersByTimeAsync(30_000);
    expect(controller.getState().status).toBe("streaming");

    stream.push(frame("answer", ANSWER));
    stream.close();
    await run;
    expect(controller.getState().status).toBe("done");
  });

  it("a stream that ends without an answer is dropped", async () => {
    const stream = controllable();
    const { controller } = controllerFor(stream.body);
    const run = controller.ask("q");
    await vi.advanceTimersByTimeAsync(0);
    stream.push(frame("token", { text: "half an answer" }));
    stream.close();
    await run;
    expect(controller.getState().status).toBe("dropped");
    expect(controller.getState().answer).toBeNull();
  });

  it("abort returns to idle and cancels the request", async () => {
    const stream = controllable();
    const { fetchImpl, controller } = controllerFor(stream.body);
    const run = controller.ask("q");
    await vi.advanceTimersByTimeAsync(0);
    const init = fetchImpl.mock.calls[0]![1] as RequestInit;

    controller.abort();
    expect(controller.getState().status).toBe("idle");
    expect((init.signal as AbortSignal).aborted).toBe(true);
    stream.close();
    await run;
  });

  it("the shipped watchdog drops at 45 seconds and aborts the request", async () => {
    const stream = controllable();
    const fetchImpl = vi.fn().mockResolvedValue(new Response(stream.body));
    // No watchdogMs: this pins the default the app ships with, not the test's own value.
    const controller = new AskController({ fetchImpl });
    const run = controller.ask("q");
    await vi.advanceTimersByTimeAsync(0);
    await vi.advanceTimersByTimeAsync(44_999);
    expect(controller.getState().status).toBe("streaming");

    await vi.advanceTimersByTimeAsync(1);
    expect(controller.getState().status).toBe("dropped");
    const init = fetchImpl.mock.calls[0]![1] as RequestInit;
    expect((init.signal as AbortSignal).aborted).toBe(true);
    stream.close();
    await run;
  });

  it("a request that never answers is dropped, not left streaming", async () => {
    // The tunnel stalls before the response headers: the watchdog must cover that wait too.
    const fetchImpl = vi.fn(() => new Promise<Response>(() => undefined));
    const controller = new AskController({ fetchImpl });
    void controller.ask("q");
    await vi.advanceTimersByTimeAsync(45_000);
    expect(controller.getState().status).toBe("dropped");
  });

  it("a second ask keeps its own watchdog", async () => {
    // The first request has no byte yet, so aborting it rejects its fetch; that request's finally
    // must leave the second request's watchdog armed.
    const second = controllable();
    const fetchImpl = vi
      .fn()
      .mockImplementationOnce(
        (_url: string, init: RequestInit) =>
          new Promise<Response>((_resolve, reject) => {
            init.signal?.addEventListener("abort", () =>
              reject(new DOMException("The operation was aborted.", "AbortError")),
            );
          }),
      )
      .mockResolvedValueOnce(new Response(second.body));
    const controller = new AskController({ fetchImpl, watchdogMs: 45_000 });

    const firstRun = controller.ask("first");
    await vi.advanceTimersByTimeAsync(0);
    const secondRun = controller.ask("second");
    // The aborted request settles while the second one is the active request.
    await vi.advanceTimersByTimeAsync(0);
    await firstRun;

    await vi.advanceTimersByTimeAsync(45_000);
    expect(controller.getState().status).toBe("dropped");
    expect(controller.getState().question).toBe("second");

    second.close();
    await secondRun;
  });

  it("a finished answer leaves no watchdog timer behind", async () => {
    const stream = controllable();
    const { controller } = controllerFor(stream.body);
    const run = controller.ask("q");
    await vi.advanceTimersByTimeAsync(0);
    stream.push(frame("answer", ANSWER));
    stream.close();
    await run;
    expect(controller.getState().status).toBe("done");
    expect(vi.getTimerCount()).toBe(0);
  });

  it("a late byte after abort does not re-arm the watchdog", async () => {
    const stream = controllable();
    const { controller } = controllerFor(stream.body);
    const run = controller.ask("q");
    await vi.advanceTimersByTimeAsync(0);

    controller.abort();
    // A connection can deliver one more buffered chunk after the abort; it must not arm a timer
    // that nothing will ever clear.
    stream.push(": ping\n\n");
    stream.close();
    await run;
    expect(vi.getTimerCount()).toBe(0);
  });

  it("abort clears the sheet instead of keeping a partial answer", async () => {
    const stream = controllable();
    const { controller } = controllerFor(stream.body);
    const run = controller.ask("q");
    await vi.advanceTimersByTimeAsync(0);
    stream.push(frame("token", { text: "partial unchecked" }));
    await vi.advanceTimersByTimeAsync(0);
    expect(controller.getState().text).toBe("partial unchecked");

    controller.abort();
    const state = controller.getState();
    expect(state.question).toBe("");
    expect(state.text).toBe("");
    expect(state.status).toBe("idle");
    stream.close();
    await run;
  });

  it("a proxied 502 reads as the API being unreachable", async () => {
    const fetchImpl = vi
      .fn()
      .mockResolvedValue(Response.json({ kind: "unavailable" }, { status: 502 }));
    const controller = new AskController({ fetchImpl });
    await controller.ask("q");
    expect(controller.getState().error?.code).toBe("unavailable");
  });

  it("a bare 502 with no JSON body still reads as unreachable", async () => {
    // A gateway error from Vercel itself is HTML, not our proxy's {kind}: the status must decide.
    const fetchImpl = vi.fn().mockResolvedValue(new Response("bad gateway", { status: 502 }));
    const controller = new AskController({ fetchImpl });
    await controller.ask("q");
    expect(controller.getState().error?.code).toBe("unavailable");
  });

  it("a 429 becomes a rate_limited error with the wait", async () => {
    const fetchImpl = vi.fn().mockResolvedValue(
      Response.json(
        { kind: "rate_limited", retryAfterS: 12 },
        { status: 429, headers: { "retry-after": "12" } },
      ),
    );
    const controller = new AskController({ fetchImpl });
    await controller.ask("q");
    const error = controller.getState().error;
    expect(controller.getState().status).toBe("error");
    expect(error?.code).toBe("rate_limited");
    expect(error?.retryAfterS).toBe(12);
  });
});
