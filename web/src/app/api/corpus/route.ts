import { requireOwnerApi } from "@/auth";
import { failureFor, forwardJson, upstreamFetch } from "@/lib/drafts-proxy";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// The API's own bounds. Refusing here keeps a bad query from spending a round trip on it.
const QUERY_MAX_CHARS = 200;
const PROFILE_MAX_CHARS = 64;
const CURSOR_MAX_CHARS = 512;
const LIMIT_MAX = 50;
const TIERS = new Set(["peer_reviewed", "vetted_preprint", "lab_blog"]);
const SOURCES = new Set(["arxiv", "feed"]);

/**
 * The corpus table's read path. The browser never holds the API token, so every filter, search and
 * next page is one call here, forwarded as it came; the API decides what a page is.
 */
export async function GET(request: Request): Promise<Response> {
  const session = await requireOwnerApi();
  if (session instanceof Response) return session;

  const raw = new URL(request.url).searchParams;
  const query = new URLSearchParams();
  const q = raw.get("q")?.trim() ?? "";
  if (q.length > QUERY_MAX_CHARS) return Response.json({ kind: "error" }, { status: 400 });
  if (q) query.set("q", q);

  const tier = raw.get("tier");
  if (tier) {
    if (!TIERS.has(tier)) return Response.json({ kind: "error" }, { status: 400 });
    query.set("tier", tier);
  }
  const source = raw.get("source");
  if (source) {
    if (!SOURCES.has(source)) return Response.json({ kind: "error" }, { status: 400 });
    query.set("source", source);
  }
  const profile = raw.get("profile");
  if (profile) {
    if (profile.length > PROFILE_MAX_CHARS) {
      return Response.json({ kind: "error" }, { status: 400 });
    }
    query.set("profile", profile);
  }
  const cursor = raw.get("cursor");
  if (cursor) {
    if (cursor.length > CURSOR_MAX_CHARS) return Response.json({ kind: "error" }, { status: 400 });
    query.set("cursor", cursor);
  }
  const limit = Number(raw.get("limit") ?? 25);
  if (!Number.isInteger(limit) || limit < 1 || limit > LIMIT_MAX) {
    return Response.json({ kind: "error" }, { status: 400 });
  }
  query.set("limit", String(limit));

  const upstream = await upstreamFetch(`/items?${query}`, { method: "GET" });
  if (upstream.status !== 200) return failureFor(upstream);
  return forwardJson(upstream, 200);
}
