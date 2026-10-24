import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { apiGet } from "./api";

function respond(status: number, body: unknown, headers: Record<string, string> = {}) {
  return new Response(JSON.stringify(body), { status, headers });
}

describe("apiGet", () => {
  beforeEach(() => {
    // The e2e credential is the one a unit test can mint without Vercel or a secret.
    vi.stubEnv("PW_API_URL", "http://api.test");
    vi.stubEnv("PW_API_AUTH", "e2e");
    vi.stubEnv("PW_E2E_SESSION", "1");
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
  });

  it("returns the parsed body and sends the bearer token", async () => {
    const fetchMock = vi.fn().mockResolvedValue(respond(200, { runs: [] }));
    vi.stubGlobal("fetch", fetchMock);

    const result = await apiGet<{ runs: unknown[] }>("/runs");

    expect(result).toEqual({ ok: true, data: { runs: [] } });
    const [, init] = fetchMock.mock.calls[0]!;
    expect(init.headers.authorization).toBe("Bearer e2e");
    expect(init.cache).toBe("no-store");
  });

  it("maps 401, 429, 503 and 404 to section errors", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(respond(401, { detail: "unauthorized" })));
    expect(await apiGet("/runs")).toEqual({ ok: false, kind: "unauthorized" });

    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(respond(429, { detail: "rate_limited" }, { "retry-after": "12" })),
    );
    expect(await apiGet("/runs")).toEqual({ ok: false, kind: "rate_limited", retryAfterS: 12 });

    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(respond(503, { status: "unavailable" })));
    expect(await apiGet("/runs")).toEqual({ ok: false, kind: "unavailable" });

    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(respond(404, { detail: "not found" })));
    expect(await apiGet("/runs/1")).toEqual({ ok: false, kind: "not_found" });
  });

  it("maps a timeout and a refused connection to unavailable", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockRejectedValue(
          new DOMException("The operation was aborted due to timeout", "TimeoutError"),
        ),
    );
    expect(await apiGet("/runs")).toEqual({ ok: false, kind: "unavailable" });

    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("fetch failed")));
    expect(await apiGet("/runs")).toEqual({ ok: false, kind: "unavailable" });
  });

  it("treats a missing credential as unauthorized, not as a crash", async () => {
    vi.stubEnv("PW_API_AUTH", "hmac");
    vi.stubEnv("PW_API_HMAC_SECRET", "");
    expect(await apiGet("/runs")).toEqual({ ok: false, kind: "unauthorized" });
  });
});
