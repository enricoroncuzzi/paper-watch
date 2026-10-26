import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// Every route handler checks the session itself, not only the proxy in front of the page: the proxy
// has been bypassed before. These call each handler with a guard that refuses, so deleting the
// guard from a route fails here even though a browser would still be redirected.
vi.mock("@/auth", () => ({
  requireOwnerApi: vi.fn(async () => new Response(null, { status: 401 })),
}));

import { requireOwnerApi } from "@/auth";

import { POST as ask } from "./ask/route";
import { GET as corpus } from "./corpus/route";
import { POST as createDraft } from "./drafts/route";
import { GET as draft } from "./drafts/[id]/route";
import { POST as regenerate } from "./drafts/[id]/regenerate/route";
import { POST as recordEvent } from "./suggestions/[id]/events/route";
import { GET as threads } from "./threads/route";
import { GET as thread } from "./threads/[id]/route";

const ID = "44444444-4444-4444-8444-444444444401";

function context(id: string): { params: Promise<{ id: string }> } {
  return { params: Promise.resolve({ id }) };
}

function get(url: string): Request {
  return new Request(url, { method: "GET" });
}

function post(url: string, body: unknown): Request {
  return new Request(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
}

const CALLS: [string, () => Promise<Response>][] = [
  ["POST /api/ask", () => ask(post("http://localhost/api/ask", { question: "q" }))],
  ["POST /api/drafts", () => createDraft(post("http://localhost/api/drafts", { item_id: ID }))],
  ["GET /api/drafts/{id}", () => draft(get(`http://localhost/api/drafts/${ID}`), context(ID))],
  [
    "POST /api/drafts/{id}/regenerate",
    () => regenerate(post(`http://localhost/api/drafts/${ID}/regenerate`, {}), context(ID)),
  ],
  [
    "POST /api/suggestions/{id}/events",
    () =>
      recordEvent(
        post(`http://localhost/api/suggestions/${ID}/events`, { event: "opened" }),
        context(ID),
      ),
  ],
  ["GET /api/corpus", () => corpus(get("http://localhost/api/corpus?limit=10"))],
  ["GET /api/threads", () => threads()],
  ["GET /api/threads/{id}", () => thread(get(`http://localhost/api/threads/${ID}`), context(ID))],
];

describe("the owner guard on every route handler", () => {
  beforeEach(() => {
    vi.stubEnv("PW_API_URL", "http://api.test");
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.unstubAllEnvs();
  });

  it.each(CALLS)("%s refuses an anonymous call before any upstream read", async (_name, call) => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    const response = await call();
    expect(response.status).toBe(401);
    expect(fetchMock).not.toHaveBeenCalled();
    expect(requireOwnerApi).toHaveBeenCalled();
  });
});
