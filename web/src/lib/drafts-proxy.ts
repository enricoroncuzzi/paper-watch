import "server-only";

import { READ_TIMEOUT_MS } from "@/lib/api";
import { apiToken } from "@/lib/api-token";

export const NO_STORE = { "cache-control": "no-store" } as const;

/**
 * How long a draft may wait. A draft spends a model call, and the dialog and the function both
 * wait 300 s; this bound sits just under them, so the reader gets a refusal to show instead of a
 * function that dies with nothing to render.
 */
export const DRAFT_TIMEOUT_MS = 280_000;

/** A refusal the page can act on, in the shape the dialog and the draft cards read. */
export function refused(kind: string, status: number, retryAfterS?: number): Response {
  return Response.json({ kind, retryAfterS }, { status, headers: NO_STORE });
}

/**
 * One call from a route handler to the API. The API's address and credential stay on this server,
 * and a missing base, a missing token and a refused connection all become the same refusals the
 * page already knows, so no route repeats that plumbing.
 *
 * Every call is bounded, its body included: a tunnel that accepts the connection and then stalls
 * would otherwise hold the function open to its maxDuration. The default is the read bound, since
 * a read is a page load; only a draft passes the long one.
 */
export async function upstreamFetch(
  path: string,
  init: RequestInit,
  timeoutMs: number = READ_TIMEOUT_MS,
): Promise<Response> {
  const base = process.env.PW_API_URL;
  if (!base) return refused("unavailable", 503);
  let token: string;
  try {
    token = await apiToken();
  } catch {
    return refused("unauthorized", 401);
  }
  // An explicit controller rather than `AbortSignal.timeout`: a test can advance its clock.
  const timeout = new AbortController();
  const timer = setTimeout(() => timeout.abort(), timeoutMs);
  try {
    const upstream = await fetch(new URL(path, base), {
      ...init,
      headers: {
        authorization: `Bearer ${token}`,
        ...(init.headers as Record<string, string> | undefined),
      },
      signal: init.signal ? AbortSignal.any([init.signal, timeout.signal]) : timeout.signal,
      cache: "no-store",
    });
    // `fetch` resolves on the headers; the body is read here, under the same bound, so a tunnel
    // that sends headers and then stalls cannot hold the function open either.
    const body = await upstream.arrayBuffer();
    return new Response(body, { status: upstream.status, headers: upstream.headers });
  } catch {
    // A timeout, a stalled socket and a refused connection are the same state to the page.
    return refused("unavailable", 503);
  } finally {
    clearTimeout(timer);
  }
}

/**
 * An upstream failure as the browser's own response.
 *
 * A 503 keeps the API's `{code, message, retry_at}` untouched: the dialog shows the refusal
 * sentence as the API wrote it. A 429 becomes the same `{kind, retryAfterS}` shape the ask proxy
 * uses, and everything else a kind the page already maps to copy.
 */
export async function failureFor(upstream: Response): Promise<Response> {
  const body = (await upstream.json().catch(() => null)) as {
    detail?: unknown;
    retry_after_s?: unknown;
  } | null;
  if (upstream.status === 503) {
    return Response.json(body ?? { kind: "unavailable" }, { status: 503, headers: NO_STORE });
  }
  if (upstream.status === 429) {
    const retryAfter = Number(upstream.headers.get("retry-after") ?? body?.retry_after_s);
    return refused(
      typeof body?.detail === "string" ? body.detail : "rate_limited",
      429,
      Number.isFinite(retryAfter) ? retryAfter : undefined,
    );
  }
  const kind =
    upstream.status === 401 || upstream.status === 403
      ? "unauthorized"
      : upstream.status === 404
        ? "not_found"
        : upstream.status >= 500
          ? "unavailable"
          : "error";
  return refused(kind, upstream.status);
}

/** A successful upstream body, forwarded unchanged with the status the page expects. */
export async function forwardJson(upstream: Response, status: number): Promise<Response> {
  const body = await upstream.json().catch(() => null);
  if (body === null) return refused("unavailable", 502);
  return Response.json(body, { status, headers: NO_STORE });
}
