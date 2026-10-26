import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// The route is tested without a Next server: the session guard and the credential are the two
// collaborators, and everything else is the fetch contract with the API.
vi.mock("@/auth", () => ({
  requireOwnerApi: vi.fn(async () => ({ githubId: "4242" })),
}));
vi.mock("@/lib/api-token", () => ({ apiToken: vi.fn(async () => "test-token") }));

import { GET } from "./route";

function corpusRequest(search: string): Request {
  return new Request(`http://localhost/api/corpus${search}`, { method: "GET" });
}

function page(): Response {
  return Response.json({ items: [], next_cursor: null, total: 0 });
}

function forwardedUrl(fetchMock: ReturnType<typeof vi.fn>): URL {
  return new URL(String(fetchMock.mock.calls[0]![0]));
}

describe("the corpus proxy", () => {
  beforeEach(() => {
    vi.stubEnv("PW_API_URL", "http://api.test");
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.unstubAllEnvs();
  });

  it("forwards a since day as it came", async () => {
    const fetchMock = vi.fn().mockResolvedValue(page());
    vi.stubGlobal("fetch", fetchMock);

    const response = await GET(corpusRequest("?since=2026-11-01&limit=10"));
    expect(response.status).toBe(200);
    const url = forwardedUrl(fetchMock);
    expect(url.pathname).toBe("/items");
    expect(url.searchParams.get("since")).toBe("2026-11-01");
    expect(url.searchParams.get("limit")).toBe("10");
  });

  it("leaves since off a request that did not set it", async () => {
    const fetchMock = vi.fn().mockResolvedValue(page());
    vi.stubGlobal("fetch", fetchMock);

    await GET(corpusRequest(""));
    expect(forwardedUrl(fetchMock).searchParams.has("since")).toBe(false);
  });

  it.each([
    "01-11-2026",
    "2026-11",
    "2026-11-01T00:00:00Z",
    "2026-02-30",
    "yesterday",
    " 2026-11-01",
  ])("refuses %s before spending a round trip", async (since) => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    const response = await GET(corpusRequest(`?since=${encodeURIComponent(since)}`));
    expect(response.status).toBe(400);
    expect(await response.json()).toEqual({ kind: "error" });
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
