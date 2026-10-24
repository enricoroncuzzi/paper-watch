import { z } from "zod";

import { requireOwnerApi } from "@/auth";
import { apiToken } from "@/lib/api-token";

// A streamed answer can run to the API's 120 s deadline, so Vercel must allow the function to live
// longer than that; the proxy itself adds no buffering.
export const runtime = "nodejs";
export const maxDuration = 300;
export const dynamic = "force-dynamic";

const AskBody = z.object({
  question: z.string().trim().min(1).max(1000),
  // Slices 4 and 6 send these; the fields exist now so the contract does not change twice.
  tiers: z.array(z.string()).max(3).optional(),
  profiles: z.array(z.string()).max(8).optional(),
  item_id: z.string().uuid().optional(),
  thread_id: z.string().uuid().optional(),
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

  const body = AskBody.safeParse(await request.json().catch(() => null));
  if (!body.success) return refused("error", 400);

  const base = process.env.PW_API_URL;
  if (!base) return refused("unavailable", 503);

  let token: string;
  try {
    token = await apiToken();
  } catch {
    return refused("unauthorized", 401);
  }

  let upstream: Response;
  try {
    upstream = await fetch(new URL("/ask", base), {
      method: "POST",
      headers: { authorization: `Bearer ${token}`, "content-type": "application/json" },
      body: JSON.stringify(body.data),
      // The browser hanging up cancels the answer upstream too, as the CLI's Ctrl-C does.
      signal: request.signal,
      cache: "no-store",
    });
  } catch {
    return refused("unavailable", 503);
  }

  if (upstream.status !== 200 || !upstream.body) {
    const payload = (await upstream.json().catch(() => ({}))) as {
      detail?: string;
      retry_after_s?: number;
    };
    const retryAfter = Number(upstream.headers.get("retry-after") ?? payload.retry_after_s);
    // `detail` carries `rate_limited` or `busy`, which the sheet phrases differently.
    const kind = upstream.status === 429 ? (payload.detail ?? "rate_limited") : "unavailable";
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
