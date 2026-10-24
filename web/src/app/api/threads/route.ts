import { requireOwnerApi } from "@/auth";
import { apiGet } from "@/lib/api";
import type { components } from "@/lib/api-types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export type ThreadSummary = components["schemas"]["ThreadSummaryData"];

/** The API's refusal as the browser's own: the menu only needs the kind, never the detail. */
function failed(kind: string, status: number): Response {
  return Response.json({ kind }, { status, headers: { "cache-control": "no-store" } });
}

/**
 * The Recent menu's list. It goes through this handler, not straight to the API, because the API's
 * address and credential live only on this server; the menu is read when it opens, so it is never
 * cached.
 */
export async function GET(): Promise<Response> {
  const session = await requireOwnerApi();
  if (session instanceof Response) return session;

  const result = await apiGet<components["schemas"]["ThreadsData"]>("/threads");
  if (!result.ok) {
    const status =
      result.kind === "unauthorized"
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
