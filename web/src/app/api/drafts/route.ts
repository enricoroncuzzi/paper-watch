import { z } from "zod";

import { requireOwnerApi } from "@/auth";
import { failureFor, forwardJson, NO_STORE, refused, upstreamFetch } from "@/lib/drafts-proxy";

export const runtime = "nodejs";
// The API's draft call has its own timeout, but a slow tunnel can hold this function for longer;
// the dialog waits to the same 300 s before it gives up.
export const maxDuration = 300;
export const dynamic = "force-dynamic";

// The API caps a request body at 16 KB; 2,000 characters of point of view plus JSON stays inside.
const BODY_MAX_BYTES = 16 * 1024;
const DraftBody = z.object({
  item_id: z.string().uuid(),
  pov: z.string().max(2000).default(""),
  language: z.string().min(2).max(8).default("en"),
});

/**
 * The only path from the browser to `POST /drafts`. Generating saves on the API, so this proxy
 * forwards the 201 body as it is; the point of view never leaves this server for anywhere but the
 * API, which stores it with the draft.
 */
export async function POST(request: Request): Promise<Response> {
  const session = await requireOwnerApi();
  if (session instanceof Response) return session;

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
  const body = DraftBody.safeParse(payload);
  if (!body.success) return refused("error", 400);

  const upstream = await upstreamFetch("/drafts", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body.data),
  });
  if (upstream.status !== 201) return failureFor(upstream);
  return forwardJson(upstream, 201);
}
