import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// The route is tested without a Next server: the session guard and the credential are the two
// collaborators, and everything else is the fetch contract with the API.
vi.mock("@/auth", () => ({
  requireOwnerApi: vi.fn(async () => ({ githubId: "4242" })),
}));
vi.mock("@/lib/api-token", () => ({ apiToken: vi.fn(async () => "test-token") }));

import { POST } from "./route";

function askRequest(body: unknown, signal?: AbortSignal): Request {
  return new Request("http://localhost/api/ask", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
}

function emptyStream(): ReadableStream<Uint8Array> {
  return new ReadableStream({
    start(controller) {
      controller.close();
    },
  });
}

describe("the ask proxy", () => {
  beforeEach(() => {
    vi.stubEnv("PW_API_URL", "http://api.test");
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.unstubAllEnvs();
    vi.useRealTimers();
  });

  it("forwards the request's signal, so a browser hang-up cancels the answer", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        new Response(emptyStream(), { status: 200, headers: { "content-type": "text/event-stream" } }),
      );
    vi.stubGlobal("fetch", fetchMock);
    const requestAbort = new AbortController();

    const response = await POST(askRequest({ question: "q" }, requestAbort.signal));
    expect(response.status).toBe(200);
    expect(response.headers.get("content-type")).toBe("text/event-stream");

    const init = fetchMock.mock.calls[0]![1] as RequestInit;
    const signal = init.signal as AbortSignal;
    expect(signal.aborted).toBe(false);
    requestAbort.abort();
    expect(signal.aborted).toBe(true);
  });

  it("gives up on a stalled header wait after 30 seconds", async () => {
    vi.useFakeTimers();
    vi.stubGlobal(
      "fetch",
      vi.fn(
        (_url: string, init: RequestInit) =>
          new Promise<Response>((_resolve, reject) => {
            (init.signal as AbortSignal).addEventListener("abort", () =>
              reject(new DOMException("aborted", "AbortError")),
            );
          }),
      ),
    );

    const pending = POST(askRequest({ question: "q" }));
    await vi.advanceTimersByTimeAsync(30_000);
    const response = await pending;
    expect(response.status).toBe(503);
    expect(await response.json()).toEqual({ kind: "unavailable" });
  });

  it("refuses a body over 16 KB before parsing it", async () => {
    vi.stubGlobal("fetch", vi.fn());
    const response = await POST(askRequest({ question: "x".repeat(20_000) }));
    expect(response.status).toBe(413);
  });

  it("keeps the API's busy detail and retry wait", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        Response.json({ detail: "busy" }, { status: 429, headers: { "retry-after": "7" } }),
      ),
    );
    const response = await POST(askRequest({ question: "q" }));
    expect(response.status).toBe(429);
    expect(await response.json()).toEqual({ kind: "busy", retryAfterS: 7 });
  });

  it.each([
    [401, "unauthorized"],
    [403, "unauthorized"],
    [400, "error"],
    [413, "error"],
    [415, "error"],
    [500, "unavailable"],
    [502, "unavailable"],
  ])("passes a %i refusal through as %s", async (status, kind) => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(Response.json({ detail: "refused" }, { status })),
    );
    const response = await POST(askRequest({ question: "q" }));
    expect(response.status).toBe(status);
    expect(await response.json()).toEqual({ kind });
  });

  it("takes the wait from a 429 body when there is no Retry-After", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        Response.json({ detail: "rate_limited", retry_after_s: 12 }, { status: 429 }),
      ),
    );
    const response = await POST(askRequest({ question: "q" }));
    expect(response.status).toBe(429);
    expect(await response.json()).toEqual({ kind: "rate_limited", retryAfterS: 12 });
  });

  it("refuses an empty question with 400", async () => {
    vi.stubGlobal("fetch", vi.fn());
    const response = await POST(askRequest({ question: "   " }));
    expect(response.status).toBe(400);
  });

  it("forwards only the question", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response("data: {}\n\n", { headers: { "content-type": "text/event-stream" } }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const response = await POST(
      askRequest({ question: "q", thread_id: "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", tiers: [1] }),
    );
    expect(response.status).toBe(200);
    const init = fetchMock.mock.calls[0]![1] as RequestInit;
    expect(JSON.parse(init.body as string)).toEqual({ question: "q" });
  });
});
