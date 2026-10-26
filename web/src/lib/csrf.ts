import "server-only";

/**
 * The refusal a mutating proxy returns for a request another origin made.
 *
 * Every mutating route here is authenticated by a cookie, so a page on another origin could
 * otherwise make the browser send it and act as the owner. `Sec-Fetch-Site` is the browser's own
 * word on the request's origin; a client that does not send it (an old browser) is checked against
 * `Origin` instead. A caller with neither header is not a browser riding the session cookie
 * (curl, the e2e runner's request context), so it is let through and the session check decides.
 *
 * The body is not required to be `application/json`: `navigator.sendBeacon` sends `text/plain`
 * on purpose, and requiring JSON would refuse the impression the events route exists for.
 */
export function crossSiteRefusal(request: Request): Response | null {
  const site = request.headers.get("sec-fetch-site");
  if (site) return site === "same-origin" ? null : refused();
  const origin = request.headers.get("origin");
  if (!origin) return null;
  // The host header is what the request arrived for; `request.url` is the fallback a test or a
  // runtime without one builds.
  const host = request.headers.get("host") ?? new URL(request.url).host;
  let originHost: string;
  try {
    originHost = new URL(origin).host;
  } catch {
    // An unparseable Origin is not the same origin as anything.
    return refused();
  }
  return originHost.toLowerCase() === host.toLowerCase() ? null : refused();
}

function refused(): Response {
  return Response.json(
    { kind: "forbidden" },
    { status: 403, headers: { "cache-control": "no-store" } },
  );
}
