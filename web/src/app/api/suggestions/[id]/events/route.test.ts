import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// The route is tested without a Next server: the session guard and the credential are the two
// collaborators, and everything else is the fetch contract with the API.
vi.mock("@/auth", () => ({
  requireOwnerApi: vi.fn(async () => ({ githubId: "4242" })),
}));
vi.mock("@/lib/api-token", () => ({ apiToken: vi.fn(async () => "test-token") }));

import { requireOwnerApi } from "@/auth";
import { POST } from "./route";

const ITEM = "33333333-3333-4333-8333-333333333301";

function eventRequest(body: unknown): Request {
  return new Request(`http://localhost/api/suggestions/${ITEM}/events`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
}

function context(id: string): { params: Promise<{ id: string }> } {
  return { params: Promise.resolve({ id }) };
}

describe("the suggestion event proxy", () => {
  beforeEach(() => {
    vi.stubEnv("PW_API_URL", "http://api.test");
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.unstubAllEnvs();
  });

  it("forwards an opening to the API and answers no content", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(null, { status: 204 }));
    vi.stubGlobal("fetch", fetchMock);

    const response = await POST(eventRequest({ event: "opened" }), context(ITEM));
    expect(response.status).toBe(204);

    const [url, init] = fetchMock.mock.calls[0]! as [URL, RequestInit];
    expect(url.toString()).toBe(`http://api.test/suggestions/${ITEM}/events`);
    expect((init.headers as Record<string, string>).authorization).toBe("Bearer test-token");
    expect(JSON.parse(String(init.body))).toEqual({ event: "opened" });
  });

  it("refuses an anonymous caller before reading the body", async () => {
    vi.mocked(requireOwnerApi).mockResolvedValueOnce(
      new Response(JSON.stringify({ kind: "unauthorized" }), { status: 401 }),
    );
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    const response = await POST(eventRequest({ event: "dismissed" }), context(ITEM));
    expect(response.status).toBe(401);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("refuses an event the contract does not define", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    const response = await POST(eventRequest({ event: "shown" }), context(ITEM));
    expect(response.status).toBe(400);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("keeps a 404 for an item the API does not admit", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(null, { status: 404 })));
    const response = await POST(eventRequest({ event: "dismissed" }), context(ITEM));
    expect(response.status).toBe(404);
  });
});
