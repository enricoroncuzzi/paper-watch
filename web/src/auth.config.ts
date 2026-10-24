import type { NextAuthConfig } from "next-auth";
import type { CredentialsConfig } from "next-auth/providers/credentials";
import GitHub from "next-auth/providers/github";

import { e2eSessionAllowed, type Env } from "@/lib/e2e";

export const GITHUB_PROVIDER_ID = "github";
export const E2E_PROVIDER_ID = "e2e";

/**
 * The allowed account is compared by its numeric GitHub id, never by the login: a login can be
 * renamed and released, an id cannot.
 */
export function isAllowedGithubProfile(profile: unknown, allowedId: string | undefined): boolean {
  if (!allowedId || !profile || typeof profile !== "object") return false;
  const id = (profile as { id?: unknown }).id;
  if (id === undefined || id === null) return false;
  return String(id) === allowedId;
}

/**
 * The Auth.js config. JWT sessions, no database: there is one owner and nothing to persist.
 * The test provider is added only when `e2eSessionAllowed` says so, so a production build cannot
 * contain it even by mistake.
 */
export function buildAuthConfig(env: Env): NextAuthConfig {
  const providers: NextAuthConfig["providers"] = [
    GitHub({
      clientId: env.AUTH_GITHUB_ID,
      clientSecret: env.AUTH_GITHUB_SECRET,
    }),
  ];
  const e2eAllowed = e2eSessionAllowed(env);
  if (e2eAllowed) {
    // Written out rather than `Credentials({...})`, whose id is fixed to `credentials`: the test
    // signs in by this provider's own id, so it cannot be confused with a password provider.
    const testProvider: CredentialsConfig = {
      id: E2E_PROVIDER_ID,
      name: "Test owner",
      type: "credentials",
      credentials: {},
      // The test session has no password: the flag and the Vercel checks above are the gate.
      authorize: () => ({ id: env.ALLOWED_GITHUB_ID ?? "1", name: "Test owner" }),
    };
    providers.push(testProvider);
  }

  return {
    providers,
    // One owner on one page: a week is long enough to stay signed in and short enough that a
    // stolen cookie ages out. Errors land on the styled sign-in page, not Auth.js's default page.
    session: { strategy: "jwt", maxAge: 7 * 24 * 60 * 60 },
    pages: { signIn: "/signin", error: "/signin" },
    callbacks: {
      signIn({ profile }) {
        if (!profile) return e2eAllowed;
        return isAllowedGithubProfile(profile, env.ALLOWED_GITHUB_ID);
      },
      jwt({ token, profile, user }) {
        if (profile && typeof profile === "object" && "id" in profile) {
          token.githubId = String((profile as { id: unknown }).id);
        } else if (e2eAllowed && user?.id) {
          token.githubId = String(user.id);
        }
        return token;
      },
      session({ session, token }) {
        return { ...session, githubId: token.githubId };
      },
    },
  };
}
