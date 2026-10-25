import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  askReducer,
  AskController,
  footnoteNumbers,
  INITIAL_ASK_STATE,
  type AnswerPayload,
  type AskErrorCode,
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

const ANSWER: AnswerPayload = {
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
      filter: "all tiers",
      tiers: null,
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
      filter: "all tiers",
      tiers: null,
    });
    const source = ANSWER.citations[0]!;
    state = askReducer(state, { type: "event", event: "source", data: source });
    state = askReducer(state, { type: "event", event: "source", data: source });
    expect(state.sources).toHaveLength(1);
  });

  it("a finished answer clears an interruption", () => {
    // A thread read can mark a question interrupted just as the answer arrives; the answer is
    // what the server checked, so the interrupted line goes.
    let state: AskState = askReducer(INITIAL_ASK_STATE, {
      type: "start",
      question: "q",
      at: "2026-11-04T09:00:00Z",
      threadId: "55555555-5555-4555-8555-555555555555",
      filter: "all tiers",
      tiers: null,
    });
    state = askReducer(state, { type: "recover", turns: [] });
    expect(state.interrupted).toBe(true);
    state = askReducer(state, { type: "event", event: "answer", data: ANSWER });
    expect(state.interrupted).toBe(false);
    expect(state.status).toBe("done");
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

  it("a frame from a superseded stream does not land on the newer request", async () => {
    // The old request's reader can still deliver one buffered frame after it was replaced. Both
    // the check after the response and the check inside the loop are what keep it out.
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

    first.push(frame("token", { text: "the older request" }));
    first.push(frame("answer", ANSWER));
    await vi.advanceTimersByTimeAsync(0);
    const state = controller.getState();
    expect(state.text).toBe("");
    expect(state.answer).toBeNull();
    expect(state.question).toBe("second");
    expect(state.status).toBe("streaming");

    first.close();
    await firstRun;
    expect(controller.getState().status).toBe("streaming");
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

function turn(n: number, question: string, answer: AnswerPayload = ANSWER) {
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

  it("a third turn keeps the conversation in order", async () => {
    const fetchImpl = vi.fn();
    const controller = new AskController({ fetchImpl, watchdogMs: 45_000 });
    for (const n of [1, 2]) {
      const stream = controllable();
      fetchImpl.mockResolvedValueOnce(new Response(stream.body));
      const run = controller.ask(`q${n}`);
      await vi.advanceTimersByTimeAsync(0);
      stream.push(frame("answer", ANSWER));
      stream.push(frame("thread", { id: THREAD, turn: n }));
      stream.close();
      await run;
    }
    const third = controllable();
    fetchImpl.mockResolvedValueOnce(new Response(third.body));
    const thirdRun = controller.ask("q3");
    await vi.advanceTimersByTimeAsync(0);
    expect(controller.getState().turns.map((one) => one.n)).toEqual([1, 2]);
    expect(controller.getState().turnCount).toBe(2);
    third.close();
    await thirdRun;
  });

  it("an answer the server could not save still moves above the next question", async () => {
    // No `thread` event: the write failed. The reader saw the answer, so it stays in the sheet.
    const fetchImpl = vi.fn();
    const controller = new AskController({ fetchImpl, watchdogMs: 45_000 });
    const first = controllable();
    fetchImpl.mockResolvedValueOnce(new Response(first.body));
    const firstRun = controller.ask("q1");
    await vi.advanceTimersByTimeAsync(0);
    first.push(frame("answer", ANSWER));
    first.close();
    await firstRun;
    expect(controller.getState().turn).toBeNull();

    const second = controllable();
    fetchImpl.mockResolvedValueOnce(new Response(second.body));
    const secondRun = controller.ask("q2");
    await vi.advanceTimersByTimeAsync(0);
    const state = controller.getState();
    expect(state.turns).toHaveLength(1);
    expect(state.turns[0]).toMatchObject({ n: null, question: "q1", answer: ANSWER });
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
    expect(controller.getState().filter).toBe("peer-reviewed only");
    expect(controller.getState().tiers).toEqual(["peer_reviewed"]);
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
    // A stored turn was asked before the sheet could know its filter, so it prints none.
    expect(state.filter).toBeNull();
    expect(state.expected).toBe(4);
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
    expect(state.turnCount).toBe(2);
  });

  it("coming back to the page leaves a healthy stream running", async () => {
    // A tab switch is not a dead connection: the thread has no turn for this question yet, so the
    // stream must keep running and its answer must arrive normally.
    const fetchImpl = routedFetch({
      [`/api/threads/${THREAD}`]: () =>
        Response.json({ id: THREAD, title: "q1", turns: [turn(1, "q1")] }),
    });
    const controller = new AskController({ fetchImpl, watchdogMs: 45_000 });
    await controller.openThread(THREAD);

    const live = controllable();
    fetchImpl.mockImplementationOnce(async () => new Response(live.body));
    const run = controller.ask("q2");
    await vi.advanceTimersByTimeAsync(0);
    await controller.recover();
    expect(controller.getState().status).toBe("streaming");
    expect(controller.getState().interrupted).toBe(false);

    live.push(frame("token", { text: "still streaming" }));
    await vi.advanceTimersByTimeAsync(0);
    expect(controller.getState().text).toBe("still streaming");
    live.push(frame("answer", ANSWER));
    live.close();
    await run;
    expect(controller.getState().status).toBe("done");
  });

  it("recovering marks the question interrupted when the stream already ended", async () => {
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
    stalled.close();
    await run;
    expect(controller.getState().status).toBe("dropped");

    await controller.recover();
    const state = controller.getState();
    expect(state.status).toBe("dropped");
    expect(state.interrupted).toBe(true);
    expect(state.question).toBe("q2");
    expect(state.turns.map((one) => one.n)).toEqual([1]);
  });

  it("a repeated question does not adopt the older answer", async () => {
    // The peer-reviewed chip asks the same text again, so the text alone cannot identify the turn:
    // the thread's turn 1 is the same question, and adopting it would show the unfiltered answer.
    const turns = [turn(1, "the same question")];
    const fetchImpl = routedFetch({
      [`/api/threads/${THREAD}`]: () => Response.json({ id: THREAD, title: "q", turns }),
    });
    const controller = new AskController({ fetchImpl, watchdogMs: 45_000 });
    await controller.openThread(THREAD);

    const stalled = controllable();
    fetchImpl.mockImplementationOnce(async () => new Response(stalled.body));
    const run = controller.ask("the same question", { tiers: ["peer_reviewed"] });
    await vi.advanceTimersByTimeAsync(0);
    stalled.close();
    await run;

    await controller.recover();
    const state = controller.getState();
    expect(state.status).toBe("dropped");
    expect(state.interrupted).toBe(true);
    expect(state.answer).toBeNull();
    expect(state.turnCount).toBe(1);
    // The re-ask still knows which filter to send again.
    expect(state.tiers).toEqual(["peer_reviewed"]);
  });

  it("a late thread read does not touch a newer request", async () => {
    let release!: () => void;
    const gated = new Promise<void>((resolve) => {
      release = resolve;
    });
    const live = controllable();
    let reads = 0;
    const fetchImpl = routedFetch({
      "/api/ask": () => new Response(live.body),
      [`/api/threads/${THREAD}`]: async () => {
        reads += 1;
        if (reads > 1) await gated;
        return Response.json({ id: THREAD, title: "q1", turns: [turn(1, "q1")] });
      },
    });
    const controller = new AskController({ fetchImpl, watchdogMs: 45_000 });
    await controller.openThread(THREAD);

    const read = controller.recover();
    const run = controller.ask("q3");
    await vi.advanceTimersByTimeAsync(0);
    release();
    await read;
    const state = controller.getState();
    expect(state.status).toBe("streaming");
    expect(state.question).toBe("q3");
    expect(state.interrupted).toBe(false);

    live.push(frame("answer", ANSWER));
    live.close();
    await run;
    expect(controller.getState().status).toBe("done");
    expect(controller.getState().interrupted).toBe(false);
  });

  it("opening a thread ignores a read that lands after another request", async () => {
    let release!: () => void;
    const gated = new Promise<void>((resolve) => {
      release = resolve;
    });
    const live = controllable();
    const fetchImpl = routedFetch({
      "/api/ask": () => new Response(live.body),
      [`/api/threads/${THREAD}`]: async () => {
        await gated;
        return Response.json({ id: THREAD, title: "q1", turns: [turn(1, "q1")] });
      },
    });
    const controller = new AskController({ fetchImpl, watchdogMs: 45_000 });
    const opening = controller.openThread(THREAD);

    const run = controller.ask("q2");
    await vi.advanceTimersByTimeAsync(0);
    release();
    await opening;
    expect(controller.getState().question).toBe("q2");
    expect(controller.getState().status).toBe("streaming");
    live.close();
    await run;
  });

  it("recovering without a thread does nothing", async () => {
    const fetchImpl = vi.fn();
    const controller = new AskController({ fetchImpl, watchdogMs: 45_000 });
    await controller.recover();
    expect(fetchImpl).not.toHaveBeenCalled();
  });

  it("coming back to an opened conversation does not duplicate its last turn", async () => {
    // An iPhone locks and unlocks on every glance, and each return used to move the shown turn
    // into the earlier ones as well. A finished sheet has nothing to recover.
    const fetchImpl = routedFetch({
      [`/api/threads/${THREAD}`]: () =>
        Response.json({
          id: THREAD,
          title: "q1",
          turns: [turn(1, "q1"), turn(2, "q2"), turn(3, "q3")],
        }),
    });
    const controller = new AskController({ fetchImpl, watchdogMs: 45_000 });
    await controller.openThread(THREAD);

    await controller.recover();
    await controller.recover();
    const state = controller.getState();
    expect(state.status).toBe("done");
    expect(state.turns.map((one) => one.n)).toEqual([1, 2]);
    expect(state.question).toBe("q3");
    expect(state.turnCount).toBe(3);
    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });

  it("coming back after an adopted answer does not duplicate it either", async () => {
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
    expect(controller.getState().status).toBe("done");

    const calls = fetchImpl.mock.calls.length;
    await controller.recover();
    await controller.recover();
    const state = controller.getState();
    expect(state.turns.map((one) => one.n)).toEqual([1]);
    expect(state.question).toBe("q2");
    expect(fetchImpl.mock.calls.length).toBe(calls);
  });

  it.each([
    "daily_cap",
    "budget",
    "model_unavailable",
    "internal",
    "deadline",
    "rate_limited",
    "busy",
    "unauthorized",
    "unavailable",
  ] as const)("a %s error survives a return to the page", async (code: AskErrorCode) => {
    // A refused follow-up is final, not interrupted: telling the reader to ask again while the
    // daily cap is spent, or the server is busy, loses the copy and the matching items that say
    // why. The return does not even read the thread.
    const fetchImpl = routedFetch({
      [`/api/threads/${THREAD}`]: () =>
        Response.json({ id: THREAD, title: "q1", turns: [turn(1, "q1")] }),
    });
    const controller = new AskController({ fetchImpl, watchdogMs: 45_000 });
    await controller.openThread(THREAD);

    const stream = controllable();
    fetchImpl.mockImplementationOnce(async () => new Response(stream.body));
    const run = controller.ask("q2");
    await vi.advanceTimersByTimeAsync(0);
    stream.push(frame("error", { code, message: "refused" }));
    stream.close();
    await run;
    expect(controller.getState().status).toBe("error");

    const calls = fetchImpl.mock.calls.length;
    await controller.recover();
    const state = controller.getState();
    expect(state.status).toBe("error");
    expect(state.error?.code).toBe(code);
    expect(state.interrupted).toBe(false);
    expect(fetchImpl.mock.calls.length).toBe(calls);
  });

  it("a stored turn for a live request replaces the stream and stops it", async () => {
    // The server finished the answer after the phone stopped hearing it: adopting it is what
    // makes stopping the stream safe, and no later frame may overwrite the checked answer.
    let turns = [turn(1, "q1")];
    const fetchImpl = routedFetch({
      [`/api/threads/${THREAD}`]: () => Response.json({ id: THREAD, title: "q1", turns }),
    });
    const controller = new AskController({ fetchImpl, watchdogMs: 45_000 });
    await controller.openThread(THREAD);

    const live = controllable();
    let signal: AbortSignal | null = null;
    fetchImpl.mockImplementationOnce(async (_url: string | URL | Request, init?: RequestInit) => {
      signal = init?.signal ?? null;
      return new Response(live.body);
    });
    const run = controller.ask("q2");
    await vi.advanceTimersByTimeAsync(0);
    turns = [...turns, turn(2, "q2")];
    await controller.recover();
    expect(signal).not.toBeNull();
    expect(signal!.aborted).toBe(true);
    expect(controller.getState().status).toBe("done");

    live.push(frame("token", { text: "late words from the stream" }));
    await vi.advanceTimersByTimeAsync(0);
    expect(controller.getState().text).toBe(ANSWER.text);
    live.close();
    await run;
    expect(controller.getState().status).toBe("done");
  });

  it("adopting a turn takes the thread's turns, not the ones the sheet held", () => {
    // The controller only adopts when the sheet's count lines up, but the contract is the
    // thread's: the turns the server stored are what a reader sees above the adopted answer.
    const held: AskState = {
      ...INITIAL_ASK_STATE,
      status: "dropped",
      question: "q4",
      askedAt: "2026-11-04T04:00:00Z",
      threadId: "55555555-5555-4555-8555-555555555555",
      turnCount: 3,
      expected: 4,
      turns: [
        { id: "saved-1", n: 1, question: "q1", answer: ANSWER, askedAt: "2026-11-04T01:00:00Z" },
      ],
    };
    const adopted = askReducer(held, {
      type: "adopted",
      turns: [turn(1, "q1"), turn(2, "q2"), turn(3, "q3"), turn(4, "q4")],
    });
    expect(adopted.turns.map((one) => one.n)).toEqual([1, 2, 3]);
    expect(adopted.turn).toBe(4);
    expect(adopted.turnCount).toBe(4);
    expect(adopted.status).toBe("done");
    expect(adopted.interrupted).toBe(false);
  });

  it("adopting a turn with a gap in the numbers counts by the highest one", () => {
    // Turn 2's payload no longer validates, so the route leaves it out. The adopted turn is still
    // the third, and the chip and the next request both have to say three.
    const held: AskState = {
      ...INITIAL_ASK_STATE,
      status: "dropped",
      question: "q3",
      askedAt: "2026-11-04T03:00:00Z",
      threadId: "55555555-5555-4555-8555-555555555555",
      turnCount: 2,
      expected: 3,
    };
    const adopted = askReducer(held, { type: "adopted", turns: [turn(1, "q1"), turn(3, "q3")] });
    expect(adopted.turn).toBe(3);
    expect(adopted.turnCount).toBe(3);
    expect(adopted.turns.map((one) => one.n)).toEqual([1]);
  });

  it("a recovered conversation counts its turns by number after a skipped turn", async () => {
    // Turn 2's payload no longer validates, so the route leaves it out; the next question is still
    // numbered by the highest stored n.
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
    stalled.close();
    await run;
    expect(controller.getState().status).toBe("dropped");

    turns = [turn(1, "q1"), turn(3, "q3")];
    await controller.recover();
    const state = controller.getState();
    expect(state.status).toBe("dropped");
    expect(state.interrupted).toBe(true);
    expect(state.turnCount).toBe(3);
    expect(state.turns.map((one) => one.n)).toEqual([1, 3]);
  });

  it("a thread read that lands after the sheet closed leaves it closed", async () => {
    // The reader closed the sheet while a recovery read was in flight: the read must not turn the
    // cleared sheet into a dropped one.
    let release!: () => void;
    const gated = new Promise<void>((resolve) => {
      release = resolve;
    });
    let reads = 0;
    const stalled = controllable();
    const fetchImpl = routedFetch({
      "/api/ask": () => new Response(stalled.body),
      [`/api/threads/${THREAD}`]: async () => {
        reads += 1;
        if (reads > 1) await gated;
        return Response.json({ id: THREAD, title: "q1", turns: [turn(1, "q1")] });
      },
    });
    const controller = new AskController({ fetchImpl, watchdogMs: 45_000 });
    await controller.openThread(THREAD);
    const run = controller.ask("q2");
    await vi.advanceTimersByTimeAsync(0);
    stalled.close();
    await run;
    expect(controller.getState().status).toBe("dropped");

    const read = controller.recover();
    controller.abort();
    release();
    await read;
    expect(controller.getState().status).toBe("idle");
    expect(controller.getState().question).toBe("");
  });

  it("closing the sheet while a conversation loads leaves the sheet closed", async () => {
    let release!: () => void;
    const gated = new Promise<void>((resolve) => {
      release = resolve;
    });
    const fetchImpl = routedFetch({
      [`/api/threads/${THREAD}`]: async () => {
        await gated;
        return Response.json({ id: THREAD, title: "q1", turns: [turn(1, "q1")] });
      },
    });
    const controller = new AskController({ fetchImpl, watchdogMs: 45_000 });
    const opening = controller.openThread(THREAD);
    controller.abort();
    release();
    await opening;
    expect(controller.getState().threadId).toBeNull();
    expect(controller.getState().status).toBe("idle");
  });

  it("opening a second conversation wins over a read still in flight", async () => {
    const other = "77777777-7777-4777-8777-777777777777";
    let release!: () => void;
    const gated = new Promise<void>((resolve) => {
      release = resolve;
    });
    const fetchImpl = routedFetch({
      [`/api/threads/${THREAD}`]: async () => {
        await gated;
        return Response.json({ id: THREAD, title: "q1", turns: [turn(1, "q1")] });
      },
      [`/api/threads/${other}`]: () =>
        Response.json({ id: other, title: "other", turns: [turn(1, "another question")] }),
    });
    const controller = new AskController({ fetchImpl, watchdogMs: 45_000 });
    const first = controller.openThread(THREAD);
    await vi.advanceTimersByTimeAsync(0);
    await controller.openThread(other);
    release();
    await first;
    expect(controller.getState().threadId).toBe(other);
    expect(controller.getState().question).toBe("another question");
  });

  it("a stored turn with this number but other words is not adopted", async () => {
    // Another tab or device appended a turn while this one was open: the number matches, the words
    // do not, so it is not this request's answer and the sheet must say the request was interrupted.
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
    turns = [turn(1, "q1"), turn(2, "a question another tab asked")];
    stalled.close();
    await run;
    expect(controller.getState().status).toBe("dropped");

    await controller.recover();
    const state = controller.getState();
    expect(state.status).toBe("dropped");
    expect(state.interrupted).toBe(true);
    expect(state.answer).toBeNull();
    expect(state.question).toBe("q2");
  });

  it("a first question with no thread reads nothing when the page returns", async () => {
    // The thread id arrives with the first answer, so there is nothing to read yet: a return must
    // not ask the API for the thread of a conversation it has not stored.
    const stream = controllable();
    const fetchImpl = vi.fn().mockResolvedValue(new Response(stream.body));
    const controller = new AskController({ fetchImpl, watchdogMs: 45_000 });
    const run = controller.ask("q1");
    await vi.advanceTimersByTimeAsync(0);
    await controller.recover();
    expect(fetchImpl).toHaveBeenCalledTimes(1);
    stream.close();
    await run;
  });

  it("a refusal that lands while the thread read is in flight stays a refusal", async () => {
    // The read started while the stream was healthy; the server then refused the answer. That
    // outcome is already on screen, so the late read must not turn it into an interruption.
    let release!: () => void;
    const gated = new Promise<void>((resolve) => {
      release = resolve;
    });
    let reads = 0;
    const stream = controllable();
    const fetchImpl = routedFetch({
      "/api/ask": () => new Response(stream.body),
      [`/api/threads/${THREAD}`]: async () => {
        reads += 1;
        if (reads > 1) await gated;
        return Response.json({ id: THREAD, title: "q1", turns: [turn(1, "q1")] });
      },
    });
    const controller = new AskController({ fetchImpl, watchdogMs: 45_000 });
    await controller.openThread(THREAD);

    const run = controller.ask("q2");
    await vi.advanceTimersByTimeAsync(0);
    const read = controller.recover();
    await vi.advanceTimersByTimeAsync(0);
    stream.push(frame("error", { code: "daily_cap", message: "cap" }));
    stream.close();
    await run;
    expect(controller.getState().status).toBe("error");

    release();
    await read;
    const state = controller.getState();
    expect(state.status).toBe("error");
    expect(state.error?.code).toBe("daily_cap");
    expect(state.interrupted).toBe(false);
  });

  it("an answer that lands while the thread read is in flight keeps its turn count", async () => {
    // Same race with a finished answer: the read was taken before the server stored this turn, so
    // applying it would put the chip back to the count the conversation had before the answer.
    let release!: () => void;
    const gated = new Promise<void>((resolve) => {
      release = resolve;
    });
    let reads = 0;
    const stream = controllable();
    const fetchImpl = routedFetch({
      "/api/ask": () => new Response(stream.body),
      [`/api/threads/${THREAD}`]: async () => {
        reads += 1;
        if (reads > 1) await gated;
        return Response.json({ id: THREAD, title: "q1", turns: [turn(1, "q1")] });
      },
    });
    const controller = new AskController({ fetchImpl, watchdogMs: 45_000 });
    await controller.openThread(THREAD);

    const run = controller.ask("q2");
    await vi.advanceTimersByTimeAsync(0);
    const read = controller.recover();
    await vi.advanceTimersByTimeAsync(0);
    stream.push(frame("answer", ANSWER));
    stream.push(frame("thread", { id: THREAD, turn: 2 }));
    stream.close();
    await run;
    expect(controller.getState().status).toBe("done");

    release();
    await read;
    const state = controller.getState();
    expect(state.status).toBe("done");
    expect(state.turnCount).toBe(2);
    expect(state.turn).toBe(2);
    expect(state.answer).toEqual(ANSWER);
  });

  it("a conversation with an unreadable turn still counts its turns by number", async () => {
    // The route skips a stored turn whose payload no longer validates; the next request is
    // numbered by the highest stored n, not by how many readable turns came back.
    const fetchImpl = routedFetch({
      [`/api/threads/${THREAD}`]: () =>
        Response.json({ id: THREAD, title: "q1", turns: [turn(1, "q1"), turn(3, "q3")] }),
    });
    const controller = new AskController({ fetchImpl, watchdogMs: 45_000 });
    await controller.openThread(THREAD);
    const state = controller.getState();
    expect(state.turnCount).toBe(3);
    expect(state.expected).toBe(4);
    expect(state.turns.map((one) => one.n)).toEqual([1]);
    expect(state.question).toBe("q3");
  });
});
