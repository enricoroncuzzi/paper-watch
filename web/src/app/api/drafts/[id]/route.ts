import { requireOwnerApi } from "@/auth";
import { failureFor, forwardJson, refused, upstreamFetch } from "@/lib/drafts-proxy";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

/**
 * The one draft's detail, so "Open" can show the point of view the list deliberately leaves out.
 * The API owns the shape; this handler only carries the session and the credential.
 */
export async function GET(
  _request: Request,
  context: { params: Promise<{ id: string }> },
): Promise<Response> {
  const session = await requireOwnerApi();
  if (session instanceof Response) return session;

  const { id } = await context.params;
  if (!UUID.test(id)) return refused("not_found", 404);

  const upstream = await upstreamFetch(`/drafts/${id}`, { method: "GET" });
  if (upstream.status !== 200) return failureFor(upstream);
  return forwardJson(upstream, 200);
}
