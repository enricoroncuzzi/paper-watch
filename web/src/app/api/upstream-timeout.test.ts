import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// Every proxy call to the API has a bound. A tunnel that accepts the connection and then stalls
// would otherwise hold the Vercel function open to its maxDuration and leave the page waiting
// with no refusal to show. The reads give up after ten seconds; only a draft keeps waiting, for as
// long as the dialog itself does.
vi.mock("@/auth", () => ({
  requireOwnerApi: vi.fn(async () => ({ githubId: "4242" })),
}));
vi.mock("@/lib/api-token", () => ({ apiToken: vi.fn(async () => "test-token") }));

import { GET as corpus } from "./corpus/route";
import { POST as createDraft } from "./drafts/route";
import { POST as regenerate } from "./drafts/[id]/regenerate/route";
import { POST as recordEvent } from "./suggestions/[id]/events/route";

const ITEM = "44444444-4444-4444-8444-444444444401";
const READ_BOUND_MS = 10_000;
const LONG_BOUND_MS = 280_000;

function context(id: string): { params: Promise<{ id: string }> } {
  return { params: Promise.resolve({ id }) };
}

function corpusRequest(): Promise<Response> {
  return corpus(new Request("http://localhost/api/corpus?limit=10"));
}

function draftRequest(): Promise<Response> {
  return createDraft(
    new Request("http://localhost/api/drafts", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ item_id: ITEM }),
    }),
  );
}

function regenerateRequest(): Promise<Response> {
  return regenerate(
    new Request(`http://localhost/api/drafts/${ITEM}/regenerate`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: "{}",
    }),
    context(ITEM),
  );
}

function eventRequest(): Promise<Response> {
  return recordEvent(
    // sendBeacon posts text/plain.
    new Request(`http://localhost/api/suggestions/${ITEM}/events`, {
      method: "POST",
      headers: { "content-type": "text/plain;charset=UTF-8" },
      body: JSON.stringify({ event: "opened" }),
    }),
    context(ITEM),
  );
}

/** A promise that settles only in response to something, so "still waiting" can be asserted. */
function settled(pending: Promise<unknown>): () => boolean {
  let done = false;
  void pending.then(
    () => (done = true),
    () => (done = true),
  );
  return () => done;
}

describe("the upstream timeout", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.stubEnv("PW_API_URL", "http://api.test");
    // A tunnel that accepts the connection and then stalls: it rejects only on the proxy's abort.
    vi.stubGlobal(
      "fetch",
      vi.fn(
        (_url: string, init: RequestInit) =>
          new Promise<Response>((_resolve, reject) => {
            // No signal at all means no bound was put on this call: it must stay pending, so a
            // test that expects a refusal can only pass once the proxy added one.
            init.signal?.addEventListener("abort", () =>
              reject(new DOMException("aborted", "AbortError")),
            );
          }),
      ),
    );
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
    vi.unstubAllEnvs();
  });

  it("gives up on a stalled corpus read after ten seconds", async () => {
    const pending = corpusRequest();
    await vi.advanceTimersByTimeAsync(READ_BOUND_MS);
    const response = await pending;
    expect(response.status).toBe(503);
    expect(await response.json()).toEqual({ kind: "unavailable" });
  });

  it("bounds the beacon's own fetch, which nobody is waiting for", async () => {
    const pending = eventRequest();
    await vi.advanceTimersByTimeAsync(READ_BOUND_MS);
    const response = await pending;
    expect(response.status).toBe(503);
    expect(await response.json()).toEqual({ kind: "unavailable" });
  });

  it.each([
    ["a new draft", draftRequest],
    ["a regenerated draft", regenerateRequest],
  ])("%s keeps waiting past the read bound, up to the dialog's own wait", async (_name, call) => {
    const pending = call();
    const done = settled(pending);
    // A draft spends a model call; the read bound must not cut it off.
    await vi.advanceTimersByTimeAsync(READ_BOUND_MS);
    expect(done()).toBe(false);

    await vi.advanceTimersByTimeAsync(LONG_BOUND_MS - READ_BOUND_MS + 1);
    const response = await pending;
    expect(response.status).toBe(503);
    expect(await response.json()).toEqual({ kind: "unavailable" });
  });
});
