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

/** A response whose body read fails when the request is aborted, as a real fetch's does. */
function abortsOnAbort(init: RequestInit, status: number): Response {
  const signal = init.signal;
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      signal?.addEventListener("abort", () =>
        controller.error(new DOMException("The operation was aborted.", "AbortError")),
      );
    },
  });
  return new Response(body, { status });
}

function controllerFor(body: ReadableStream<Uint8Array>, status = 200) {
  const fetchImpl = vi.fn().mockResolvedValue(new Response(body, { status }));
  return { fetchImpl, controller: new AskController({ fetchImpl, watchdogMs: 45_000 }) };
}

const THREAD = "55555555-5555-4555-8555-555555555555";

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
    let state: AskState = askReducer(INITIAL_ASK_STATE, {
      type: "start",
      question: "q",
      at: "2026-11-04T09:00:00Z",
      threadId: null,
    });
    state = askReducer(state, { type: "event", event: "token", data: { text: "draft words" } });
    expect(state.text).toBe("draft words");
    state = askReducer(state, { type: "event", event: "answer", data: ANSWER });
    expect(state.status).toBe("done");
    expect(state.text).toBe(ANSWER.text);
    expect(state.answer?.citations).toHaveLength(1);
  });

  it("keeps one entry per source alias", () => {
    let state: AskState = askReducer(INITIAL_ASK_STATE, {
      type: "start",
      question: "q",
      at: "2026-11-04T09:00:00Z",
      threadId: null,
    });
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

  it("a superseded 503 whose body read is aborted leaves the newer request streaming", async () => {
    // `response.json().catch(() => ({}))` swallows the abort and used to fall through to the error
    // branch, which wrote "error" on the request that had just replaced this one.
    const second = controllable();
    const fetchImpl = vi
      .fn()
      .mockImplementationOnce((_url: string, init: RequestInit) =>
        Promise.resolve(abortsOnAbort(init, 503)),
      )
      .mockResolvedValueOnce(new Response(second.body));
    const controller = new AskController({ fetchImpl, watchdogMs: 45_000 });

    const firstRun = controller.ask("first");
    await vi.advanceTimersByTimeAsync(0);
    const secondRun = controller.ask("second");
    await vi.advanceTimersByTimeAsync(0);
    await firstRun;
    expect(controller.getState().status).toBe("streaming");
    expect(controller.getState().question).toBe("second");

    second.push(frame("answer", ANSWER));
    second.close();
    await secondRun;
    expect(controller.getState().status).toBe("done");
  });

  it("a superseded stream that ends leaves the newer request streaming", async () => {
    // The aborted request's stream can still end (a queued frame, a proxy flush). Its `dropped`
    // dispatch must not land on the request that replaced it.
    const first = controllable();
    const second = controllable();
    const fetchImpl = vi
      .fn()
      .mockResolvedValueOnce(new Response(first.body))
      .mockResolvedValueOnce(new Response(second.body));
    const controller = new AskController({ fetchImpl, watchdogMs: 45_000 });

    const firstRun = controller.ask("first");
    await vi.advanceTimersByTimeAsync(0);
    const secondRun = controller.ask("second");
    await vi.advanceTimersByTimeAsync(0);
    first.close();
    await firstRun;
    expect(controller.getState().status).toBe("streaming");
    expect(controller.getState().question).toBe("second");

    second.push(frame("token", { text: "the newer answer" }));
    await vi.advanceTimersByTimeAsync(0);
    expect(controller.getState().text).toBe("the newer answer");
    second.close();
    await secondRun;
  });

  it("a superseded response that arrives late does not touch the newer request", async () => {
    // Response headers can arrive after the newer ask aborted this one.
    let resolveFirst!: (response: Response) => void;
    const firstResponse = new Promise<Response>((resolve) => {
      resolveFirst = resolve;
    });
    const second = controllable();
    const fetchImpl = vi
      .fn()
      .mockImplementationOnce(() => firstResponse)
      .mockResolvedValueOnce(new Response(second.body));
    const controller = new AskController({ fetchImpl, watchdogMs: 45_000 });

    const firstRun = controller.ask("first");
    await vi.advanceTimersByTimeAsync(0);
    const secondRun = controller.ask("second");
    await vi.advanceTimersByTimeAsync(0);
    resolveFirst(Response.json({ kind: "rate_limited" }, { status: 429 }));
    await vi.advanceTimersByTimeAsync(0);
    await firstRun;
    expect(controller.getState().status).toBe("streaming");
    expect(controller.getState().error).toBeNull();

    second.push(frame("answer", ANSWER));
    second.close();
    await secondRun;
    expect(controller.getState().status).toBe("done");
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

  it("sends the item id when a card opened the sheet", async () => {
    const stream = controllable();
    const { fetchImpl, controller } = controllerFor(stream.body);
    const run = controller.ask("what did they measure?", {
      itemId: "33333333-3333-4333-8333-333333333301",
    });
    await vi.advanceTimersByTimeAsync(0);
    const init = fetchImpl.mock.calls[0]![1] as RequestInit;
    // The API registers this item as S1 before the graph starts, so only the id travels.
    expect(JSON.parse(String(init.body))).toEqual({
      question: "what did they measure?",
      item_id: "33333333-3333-4333-8333-333333333301",
    });
    stream.push(frame("answer", ANSWER));
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

function turn(n: number, question: string, answer: unknown = ANSWER) {
  return {
    n,
    question,
    answer,
    search_mode: "hybrid",
    created_at: `2026-11-04T0${n}:00:00Z`,
  };
}

/** A fetch that answers by path, so one test can drive both the stream and the thread read. */
function routedFetch(routes: Record<string, () => Response | Promise<Response>>) {
  return vi.fn(async (url: string | URL | Request) => {
    const path = String(url);
    const handler = routes[path];
    if (!handler) throw new Error(`unexpected fetch: ${path}`);
    return handler();
  });
}

describe("the ask controller with threads", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("a follow-up sends the thread id the first answer created", async () => {
    const first = controllable();
    const second = controllable();
    const fetchImpl = vi
      .fn()
      .mockResolvedValueOnce(new Response(first.body))
      .mockResolvedValueOnce(new Response(second.body));
    const controller = new AskController({ fetchImpl, watchdogMs: 45_000 });

    const firstRun = controller.ask("q1");
    await vi.advanceTimersByTimeAsync(0);
    first.push(frame("answer", ANSWER));
    first.push(frame("thread", { id: THREAD, turn: 1 }));
    first.close();
    await firstRun;
    expect(controller.getState().threadId).toBe(THREAD);
    expect(controller.getState().turnCount).toBe(1);

    const secondRun = controller.ask("q2");
    await vi.advanceTimersByTimeAsync(0);
    const init = fetchImpl.mock.calls[1]![1] as RequestInit;
    expect(JSON.parse(String(init.body))).toEqual({ question: "q2", thread_id: THREAD });
    second.close();
    await secondRun;
  });

  it("a follow-up keeps the finished turn above it", async () => {
    const fetchImpl = vi.fn();
    const controller = new AskController({ fetchImpl, watchdogMs: 45_000 });
    const first = controllable();
    fetchImpl.mockResolvedValueOnce(new Response(first.body));
    const firstRun = controller.ask("q1");
    await vi.advanceTimersByTimeAsync(0);
    first.push(frame("answer", ANSWER));
    first.push(frame("thread", { id: THREAD, turn: 1 }));
    first.close();
    await firstRun;

    const second = controllable();
    fetchImpl.mockResolvedValueOnce(new Response(second.body));
    const secondRun = controller.ask("q2");
    await vi.advanceTimersByTimeAsync(0);
    const state = controller.getState();
    expect(state.turns).toHaveLength(1);
    expect(state.turns[0]).toMatchObject({ n: 1, question: "q1", answer: ANSWER });
    expect(state.question).toBe("q2");
    expect(state.threadId).toBe(THREAD);
    second.close();
    await secondRun;
  });

  it("the peer-reviewed chip sends the filter as tiers", async () => {
    const stream = controllable();
    const { fetchImpl, controller } = controllerFor(stream.body);
    const run = controller.ask("how do they measure it?", {
      threadId: THREAD,
      tiers: ["peer_reviewed"],
    });
    await vi.advanceTimersByTimeAsync(0);
    const init = fetchImpl.mock.calls[0]![1] as RequestInit;
    expect(JSON.parse(String(init.body))).toEqual({
      question: "how do they measure it?",
      thread_id: THREAD,
      tiers: ["peer_reviewed"],
    });
    stream.close();
    await run;
  });

  it("opening a recent thread shows its last turn and keeps the earlier ones above it", async () => {
    const detail = {
      id: THREAD,
      title: "q1",
      turns: [turn(1, "q1"), turn(2, "q2"), turn(3, "q3")],
    };
    const fetchImpl = routedFetch({ [`/api/threads/${THREAD}`]: () => Response.json(detail) });
    const controller = new AskController({ fetchImpl, watchdogMs: 45_000 });
    await controller.openThread(THREAD);
    const state = controller.getState();
    expect(state.threadId).toBe(THREAD);
    expect(state.turnCount).toBe(3);
    expect(state.turns.map((one) => one.n)).toEqual([1, 2]);
    expect(state.question).toBe("q3");
    expect(state.status).toBe("done");
    expect(state.answer).toEqual(ANSWER);
  });

  it("recovering adopts the answer the server finished after the stream died", async () => {
    // The follow-up's stream died on the phone, but the server kept going and stored the turn:
    // the re-fetch is what turns the interrupted state back into an answer.
    let turns = [turn(1, "q1")];
    const fetchImpl = routedFetch({
      [`/api/threads/${THREAD}`]: () => Response.json({ id: THREAD, title: "q1", turns }),
    });
    const controller = new AskController({ fetchImpl, watchdogMs: 45_000 });
    await controller.openThread(THREAD);

    const stalled = controllable();
    fetchImpl.mockImplementationOnce(async () => new Response(stalled.body));
    const run = controller.ask("q2");
    await vi.advanceTimersByTimeAsync(0);
    turns = [...turns, turn(2, "q2")];
    await controller.recover();
    stalled.close();
    await run;
    const state = controller.getState();
    expect(state.status).toBe("done");
    expect(state.question).toBe("q2");
    expect(state.answer).toEqual(ANSWER);
    expect(state.interrupted).toBe(false);
    expect(state.turns.map((one) => one.n)).toEqual([1]);
  });

  it("recovering marks the question interrupted when the thread has no turn for it", async () => {
    let turns = [turn(1, "q1")];
    const fetchImpl = routedFetch({
      [`/api/threads/${THREAD}`]: () => Response.json({ id: THREAD, title: "q1", turns }),
    });
    const controller = new AskController({ fetchImpl, watchdogMs: 45_000 });
    await controller.openThread(THREAD);

    const stalled = controllable();
    fetchImpl.mockImplementationOnce(async () => new Response(stalled.body));
    const run = controller.ask("q2");
    await vi.advanceTimersByTimeAsync(0);
    await controller.recover();
    stalled.close();
    await run;
    const state = controller.getState();
    expect(state.status).toBe("dropped");
    expect(state.interrupted).toBe(true);
    expect(state.question).toBe("q2");
    expect(state.turns.map((one) => one.n)).toEqual([1]);
  });

  it("recovering without a thread does nothing", async () => {
    const fetchImpl = vi.fn();
    const controller = new AskController({ fetchImpl, watchdogMs: 45_000 });
    await controller.recover();
    expect(fetchImpl).not.toHaveBeenCalled();
  });
});
