import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// The routes are tested without a Next server: the session guard and the credential are the two
// collaborators, and everything else is the fetch contract with the API.
vi.mock("@/auth", () => ({
  requireOwnerApi: vi.fn(async () => ({ githubId: "4242" })),
  // The discovery below imports every route module, including the Auth.js handler route, which
  // reads `handlers` from this module; an empty object gives it no mutating export to call.
  handlers: {},
}));
vi.mock("@/lib/api-token", () => ({ apiToken: vi.fn(async () => "test-token") }));

import { POST as ask } from "./ask/route";
import { POST as createDraft } from "./drafts/route";
import { POST as regenerate } from "./drafts/[id]/regenerate/route";
import { POST as recordEvent } from "./suggestions/[id]/events/route";

const ITEM = "44444444-4444-4444-8444-444444444401";

interface Proxied {
  name: string;
  /** The route's own success status, so a request the guard let through is told from a refusal. */
  ok: number;
  call: (headers: Record<string, string>) => Promise<Response>;
}

function context(id: string): { params: Promise<{ id: string }> } {
  return { params: Promise.resolve({ id }) };
}

function jsonRequest(url: string, body: unknown, headers: Record<string, string>): Request {
  return new Request(url, {
    method: "POST",
    headers: { ...headers, "content-type": "application/json" },
    body: JSON.stringify(body),
  });
}

const ROUTES: Proxied[] = [
  {
    name: "ask",
    ok: 200,
    call: (headers) => ask(jsonRequest("http://localhost/api/ask", { question: "q" }, headers)),
  },
  {
    name: "drafts",
    ok: 201,
    call: (headers) =>
      createDraft(jsonRequest("http://localhost/api/drafts", { item_id: ITEM }, headers)),
  },
  {
    name: "drafts/[id]/regenerate",
    ok: 201,
    call: (headers) =>
      regenerate(
        jsonRequest(`http://localhost/api/drafts/${ITEM}/regenerate`, {}, headers),
        context(ITEM),
      ),
  },
  {
    name: "suggestions/[id]/events",
    ok: 204,
    call: (headers) =>
      recordEvent(
        // sendBeacon posts text/plain, so the guard must not require JSON to let a beacon through.
        new Request(`http://localhost/api/suggestions/${ITEM}/events`, {
          method: "POST",
          headers: { ...headers, "content-type": "text/plain;charset=UTF-8" },
          body: JSON.stringify({ event: "opened" }),
        }),
        context(ITEM),
      ),
  },
];

function upstream(route: Proxied): Response {
  if (route.name === "ask") {
    return new Response("data: {}\n\n", { headers: { "content-type": "text/event-stream" } });
  }
  if (route.name === "drafts" || route.name === "drafts/[id]/regenerate") {
    return Response.json({ id: ITEM }, { status: 201 });
  }
  return new Response(null, { status: 204 });
}

describe("the cross-site guard on every mutating proxy", () => {
  beforeEach(() => {
    vi.stubEnv("PW_API_URL", "http://api.test");
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.unstubAllEnvs();
  });

  it.each(ROUTES)("$name refuses a cross-site request before any upstream call", async (route) => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    const response = await route.call({
      "sec-fetch-site": "cross-site",
      origin: "https://evil.example",
    });
    expect(response.status).toBe(403);
    expect(await response.json()).toEqual({ kind: "forbidden" });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it.each(ROUTES)("$name refuses a same-site request, which is another origin", async (route) => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    const response = await route.call({ "sec-fetch-site": "same-site" });
    expect(response.status).toBe(403);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it.each(ROUTES)("$name lets a same-origin request through", async (route) => {
    const fetchMock = vi.fn().mockResolvedValue(upstream(route));
    vi.stubGlobal("fetch", fetchMock);

    const response = await route.call({ "sec-fetch-site": "same-origin" });
    expect(response.status).toBe(route.ok);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it.each(ROUTES)("$name falls back to the Origin host without Sec-Fetch-Site", async (route) => {
    const fetchMock = vi.fn().mockResolvedValue(upstream(route));
    vi.stubGlobal("fetch", fetchMock);

    const allowed = await route.call({ origin: "http://localhost" });
    expect(allowed.status).toBe(route.ok);

    const refused = await route.call({ origin: "https://evil.example" });
    expect(refused.status).toBe(403);
    // Only the allowed one reached the API.
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it.each(ROUTES)("$name refuses a request with neither header", async (route) => {
    // The page's own fetch and sendBeacon calls send them in every supported browser, and these
    // routes are not a curl API: a request without either has no origin to trust.
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    const response = await route.call({});
    expect(response.status).toBe(403);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

// Every mutating handler in this directory, discovered rather than listed: a route added later is
// covered by the same call without anyone remembering to add it here, and the discovery is the
// route tree itself (the modules Next runs), not a scan of their source.
const ROUTE_MODULES = import.meta.glob<Record<string, unknown>>("./**/route.ts", { eager: true });
const MUTATING = ["POST", "PUT", "PATCH", "DELETE"];

interface MutatingRoute {
  name: string;
  handler: (request: Request, context: unknown) => Promise<Response>;
}

function mutatingRoutes(): MutatingRoute[] {
  const found: MutatingRoute[] = [];
  for (const [path, module] of Object.entries(ROUTE_MODULES)) {
    for (const method of MUTATING) {
      const handler = module[method];
      if (typeof handler === "function") {
        found.push({
          name: `${method} ${path}`,
          handler: handler as MutatingRoute["handler"],
        });
      }
    }
  }
  return found;
}

describe("the guard's coverage", () => {
  it("discovers the mutating handlers it is meant to check", () => {
    // A discovery over nothing would pass silently, and the guard would look covered while it is
    // not; there are four mutating routes today.
    expect(mutatingRoutes().length).toBeGreaterThanOrEqual(4);
  });

  it("every mutating handler refuses a cross-site request, whichever route it is", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    vi.stubEnv("PW_API_URL", "http://api.test");
    try {
      for (const route of mutatingRoutes()) {
        const request = new Request("http://localhost/api/whatever", {
          method: route.name.split(" ")[0],
          headers: {
            "sec-fetch-site": "cross-site",
            origin: "https://evil.example",
            "content-type": "application/json",
          },
          body: JSON.stringify({ question: "q", item_id: ITEM, event: "opened" }),
        });
        const response = await route.handler(request, context(ITEM));
        expect(response.status, route.name).toBe(403);
      }
      expect(fetchMock).not.toHaveBeenCalled();
    } finally {
      vi.unstubAllGlobals();
      vi.unstubAllEnvs();
    }
  });
});
