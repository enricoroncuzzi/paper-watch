import "server-only";

import { getVercelOidcToken } from "@vercel/oidc";
import { SignJWT } from "jose";

import { e2eSessionAllowed, type Env } from "@/lib/e2e";

/** The audience the API's HmacVerifier requires; a token for another audience is refused. */
export const HMAC_AUDIENCE = "paperwatch-api";
/** The verifier refuses a token that lives longer than this, so the client never mints one. */
export const HMAC_TTL_S = 60;
/** The token the fake API accepts. It exists only when `e2eSessionAllowed` is true. */
export const E2E_TOKEN = "e2e";

/**
 * The credential the page presents to the API. Only this server-side module ever holds one:
 * the browser sees neither the token nor the API's address.
 *
 * `PW_API_AUTH` chooses the mode the API is configured for; on Vercel production the default is
 * the Vercel OIDC token, which the API verifies against Vercel's keys and pins to the project.
 */
export async function apiToken(env: Env = process.env): Promise<string> {
  if (env.PW_API_AUTH === "e2e") {
    if (!e2eSessionAllowed(env)) {
      throw new Error("the e2e API token exists only outside Vercel");
    }
    return E2E_TOKEN;
  }
  if (env.PW_API_AUTH === "hmac") {
    const secret = env.PW_API_HMAC_SECRET;
    if (!secret) throw new Error("PW_API_AUTH=hmac needs PW_API_HMAC_SECRET");
    const issued = Math.floor(Date.now() / 1000);
    return new SignJWT({ sub: env.ALLOWED_GITHUB_ID ?? "owner" })
      .setProtectedHeader({ alg: "HS256" })
      .setAudience(HMAC_AUDIENCE)
      .setIssuedAt(issued)
      .setExpirationTime(issued + HMAC_TTL_S)
      // Single use: the verifier remembers the id until the token expires.
      .setJti(crypto.randomUUID().replaceAll("-", ""))
      .sign(new TextEncoder().encode(secret));
  }
  if (env.PW_API_AUTH === "oidc" || env.VERCEL_ENV === "production") {
    // The default audience (the team URL), deliberately: the API pins issuer, audience and the
    // exact project plus production subject, and the subject is what confines this token.
    return getVercelOidcToken();
  }
  throw new Error("no API credential is configured");
}
