import "server-only";

import { apiToken } from "@/lib/api-token";

export const NO_STORE = { "cache-control": "no-store" } as const;

/** A refusal the page can act on, in the shape the dialog and the draft cards read. */
export function refused(kind: string, status: number, retryAfterS?: number): Response {
  return Response.json({ kind, retryAfterS }, { status, headers: NO_STORE });
}

/**
 * One call from a route handler to the API. The API's address and credential stay on this server,
 * and a missing base, a missing token and a refused connection all become the same refusals the
 * page already knows, so no route repeats that plumbing.
 */
export async function upstreamFetch(path: string, init: RequestInit): Promise<Response> {
  const base = process.env.PW_API_URL;
  if (!base) return refused("unavailable", 503);
  let token: string;
  try {
    token = await apiToken();
  } catch {
    return refused("unauthorized", 401);
  }
  try {
    return await fetch(new URL(path, base), {
      ...init,
      headers: {
        authorization: `Bearer ${token}`,
        ...(init.headers as Record<string, string> | undefined),
      },
      cache: "no-store",
    });
  } catch {
    return refused("unavailable", 503);
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
