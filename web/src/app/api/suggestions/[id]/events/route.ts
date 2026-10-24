import { z } from "zod";

import { requireOwnerApi } from "@/auth";
import { apiToken } from "@/lib/api-token";

// An event body is a few bytes; anything larger is not a suggestion event.
const BODY_MAX_BYTES = 1024;
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

const EventBody = z.object({ event: z.enum(["opened", "dismissed"]) });

/**
 * The only path from the browser to `POST /suggestions/{item_id}/events`. The API's address and
 * credential stay on this server, and the session is checked here rather than only in the proxy.
 * A beacon posts "opened" and never reads the answer, so the reply is empty either way.
 */
export async function POST(
  request: Request,
  context: { params: Promise<{ id: string }> },
): Promise<Response> {
  const session = await requireOwnerApi();
  if (session instanceof Response) return session;

  const { id } = await context.params;
  if (!UUID.test(id)) {
    return Response.json({ kind: "not_found" }, { status: 404, headers: { "cache-control": "no-store" } });
  }

  const raw = await request.text();
  if (new TextEncoder().encode(raw).length > BODY_MAX_BYTES) {
    return Response.json({ kind: "error" }, { status: 413, headers: { "cache-control": "no-store" } });
  }
  let payload: unknown;
  try {
    payload = JSON.parse(raw);
  } catch {
    return Response.json({ kind: "error" }, { status: 400, headers: { "cache-control": "no-store" } });
  }
  const body = EventBody.safeParse(payload);
  if (!body.success) {
    return Response.json({ kind: "error" }, { status: 400, headers: { "cache-control": "no-store" } });
  }

  const base = process.env.PW_API_URL;
  if (!base) {
    return Response.json({ kind: "unavailable" }, { status: 503, headers: { "cache-control": "no-store" } });
  }

  let token: string;
  try {
    token = await apiToken();
  } catch {
    return Response.json({ kind: "unauthorized" }, { status: 401, headers: { "cache-control": "no-store" } });
  }

  let upstream: Response;
  try {
    upstream = await fetch(new URL(`/suggestions/${id}/events`, base), {
      method: "POST",
      headers: { authorization: `Bearer ${token}`, "content-type": "application/json" },
      body: JSON.stringify(body.data),
      cache: "no-store",
    });
  } catch {
    return Response.json({ kind: "unavailable" }, { status: 503, headers: { "cache-control": "no-store" } });
  }

  if (!upstream.ok) {
    // A 404 is the item, not this page, so it is passed on; anything else is our own failure.
    const status = upstream.status === 404 ? 404 : 502;
    return Response.json({ kind: "error" }, { status, headers: { "cache-control": "no-store" } });
  }
  return new Response(null, { status: 204, headers: { "cache-control": "no-store" } });
}
