import { z } from "zod";

import { requireOwnerApi } from "@/auth";
import { apiToken } from "@/lib/api-token";
import { crossSiteRefusal } from "@/lib/csrf";

// A streamed answer can run to the API's 120 s deadline, so Vercel must allow the function to live
// longer than that; the proxy itself adds no buffering.
export const runtime = "nodejs";
// The API caps a request body at 16 KB; the proxy refuses the same size before parsing.
const BODY_MAX_BYTES = 16 * 1024;
// A tunnel can stall before the response headers. The sheet's watchdog covers the whole wait; this
// bound keeps the function from holding a request open for its full maxDuration.
const HEADERS_TIMEOUT_MS = 30_000;
export const maxDuration = 300;
export const dynamic = "force-dynamic";

const AskBody = z.object({
  question: z.string().trim().min(1).max(1000),
  item_id: z.string().uuid().optional(),
  // A follow-up continues a stored thread; the API refuses an unknown id before its stream starts.
  thread_id: z.string().uuid().optional(),
  // The peer-reviewed chip re-asks with this filter; the API's own bound is one to three.
  tiers: z
    .array(z.enum(["peer_reviewed", "vetted_preprint", "lab_blog"]))
    .max(3)
    .optional(),
});

function refused(kind: string, status: number, retryAfterS?: number): Response {
  return Response.json({ kind, retryAfterS }, { status, headers: { "cache-control": "no-store" } });
}

/**
 * The only path from the browser to `POST /ask`. The session is checked here, not only in the
 * proxy; the API's address and credential stay on this server; the stream is passed through
 * unbuffered so tokens reach the sheet as they are written.
 */
export async function POST(request: Request): Promise<Response> {
  const session = await requireOwnerApi();
  if (session instanceof Response) return session;
  const crossSite = crossSiteRefusal(request);
  if (crossSite) return crossSite;

  const declared = Number(request.headers.get("content-length") ?? 0);
  if (declared > BODY_MAX_BYTES) return refused("error", 413);
  const raw = await request.text();
  if (new TextEncoder().encode(raw).length > BODY_MAX_BYTES) return refused("error", 413);
  let payload: unknown;
  try {
    payload = JSON.parse(raw);
  } catch {
    return refused("error", 400);
  }
  const body = AskBody.safeParse(payload);
  if (!body.success) return refused("error", 400);

  const base = process.env.PW_API_URL;
  if (!base) return refused("unavailable", 503);

  let token: string;
  try {
    token = await apiToken();
  } catch {
    return refused("unauthorized", 401);
  }

  // Only the headers phase is bounded: once the body streams, the API's own 120 s deadline owns
  // the answer, while the browser hanging up still cancels it as the CLI's Ctrl-C does.
  const headers = new AbortController();
  const timer = setTimeout(() => headers.abort(), HEADERS_TIMEOUT_MS);
  let upstream: Response;
  try {
    upstream = await fetch(new URL("/ask", base), {
      method: "POST",
      headers: { authorization: `Bearer ${token}`, "content-type": "application/json" },
      body: JSON.stringify(body.data),
      signal: AbortSignal.any([request.signal, headers.signal]),
      cache: "no-store",
    });
  } catch {
    return refused("unavailable", 503);
  } finally {
    clearTimeout(timer);
  }

  if (upstream.status !== 200 || !upstream.body) {
    const payload = (await upstream.json().catch(() => ({}))) as {
      detail?: string;
      retry_after_s?: number;
    };
    const retryAfter = Number(upstream.headers.get("retry-after") ?? payload.retry_after_s);
    // The gate's declared bodies: `RateLimited` carries `detail` and `retry_after_s`, `Refusal` a
    // `detail` string. A 401/403 means the credential, a 5xx the API, anything else our own bug.
    const kind =
      upstream.status === 429
        ? (payload.detail ?? "rate_limited")
        : upstream.status === 401 || upstream.status === 403
          ? "unauthorized"
          : upstream.status >= 500
            ? "unavailable"
            : "error";
    return refused(kind, upstream.status, Number.isFinite(retryAfter) ? retryAfter : undefined);
  }

  return new Response(upstream.body, {
    headers: {
      "content-type": "text/event-stream",
      "cache-control": "no-cache, no-transform",
      "x-accel-buffering": "no",
    },
  });
}
