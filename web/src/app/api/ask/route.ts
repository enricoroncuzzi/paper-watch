import { requireOwnerApi } from "@/auth";

// The streaming proxy is filled in with the reading sheet; the session gate is the route's own,
// never middleware's alone.
export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST() {
  const session = await requireOwnerApi();
  if (session instanceof Response) return session;
  return Response.json({ kind: "unavailable" }, { status: 503 });
}
