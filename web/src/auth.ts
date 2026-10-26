import "server-only";

import NextAuth, { type Session } from "next-auth";
import { redirect } from "next/navigation";

import { buildAuthConfig } from "@/auth.config";

export { buildAuthConfig, E2E_PROVIDER_ID, GITHUB_PROVIDER_ID } from "@/auth.config";

export const { handlers, auth, signIn, signOut } = NextAuth(buildAuthConfig(process.env));

/**
 * The owner's session, or null. Every server component and route handler calls this itself:
 * middleware only saves an anonymous page load from rendering, and has been bypassed before.
 */
export async function ownerSession(): Promise<Session | null> {
  const session = await auth();
  const allowed = process.env.ALLOWED_GITHUB_ID;
  if (!allowed || !session?.githubId || session.githubId !== allowed) return null;
  return session;
}

/** For pages: an anonymous visit is sent to the sign-in page. */
export async function requireOwner(): Promise<Session> {
  const session = await ownerSession();
  if (!session) redirect("/signin");
  return session;
}

/**
 * For route handlers: an anonymous call gets a 401 body it can return, not a redirect the browser
 * would follow. One function cannot do both, which is why there are two.
 */
export async function requireOwnerApi(): Promise<Session | Response> {
  const session = await ownerSession();
  if (session) return session;
  return new Response(JSON.stringify({ kind: "unauthorized" }), {
    status: 401,
    headers: { "content-type": "application/json", "cache-control": "no-store" },
  });
}
