import { requireOwnerApi } from "@/auth";
import { apiGet } from "@/lib/api";
import type { components } from "@/lib/api-types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

function failed(kind: string, status: number): Response {
  return Response.json({ kind }, { status, headers: { "cache-control": "no-store" } });
}

/**
 * One conversation's finished turns, for the sheet to re-render after a dropped stream or a return
 * to a frozen tab, and for the Recent menu to open. Reads only, so it spends no model call.
 */
export async function GET(
  _request: Request,
  context: { params: Promise<{ id: string }> },
): Promise<Response> {
  const session = await requireOwnerApi();
  if (session instanceof Response) return session;

  const { id } = await context.params;
  if (!UUID.test(id)) return failed("not_found", 404);

  const result = await apiGet<components["schemas"]["ThreadDetailData"]>(
    `/threads/${encodeURIComponent(id)}`,
  );
  if (!result.ok) {
    const status =
      result.kind === "not_found"
        ? 404
        : result.kind === "unauthorized"
          ? 401
          : result.kind === "unavailable"
            ? 503
            : result.kind === "rate_limited"
              ? 429
              : 502;
    return failed(result.kind, status);
  }
  return Response.json(result.data, { headers: { "cache-control": "no-store" } });
}
