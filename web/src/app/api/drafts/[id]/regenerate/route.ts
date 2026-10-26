import { requireOwnerApi } from "@/auth";
import { crossSiteRefusal } from "@/lib/csrf";
import {
  DRAFT_TIMEOUT_MS,
  failureFor,
  forwardJson,
  refused,
  upstreamFetch,
} from "@/lib/drafts-proxy";

export const runtime = "nodejs";
export const maxDuration = 300;
export const dynamic = "force-dynamic";

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

/**
 * Regenerate one saved draft from its stored inputs and point of view. The old draft stays on the
 * API; the 201 body is the new one, which the page refreshes to show above it.
 */
export async function POST(
  request: Request,
  context: { params: Promise<{ id: string }> },
): Promise<Response> {
  const session = await requireOwnerApi();
  if (session instanceof Response) return session;
  const crossSite = crossSiteRefusal(request);
  if (crossSite) return crossSite;

  const { id } = await context.params;
  if (!UUID.test(id)) return refused("not_found", 404);

  const upstream = await upstreamFetch(
    `/drafts/${id}/regenerate`,
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: "{}",
    },
    DRAFT_TIMEOUT_MS,
  );
  if (upstream.status !== 201) return failureFor(upstream);
  return forwardJson(upstream, 201);
}
