import { decodeJwt } from "jose";
import { describe, expect, it } from "vitest";

import { apiToken, HMAC_AUDIENCE, HMAC_TTL_S } from "./api-token";

describe("the API token", () => {
  it("is refused on Vercel even with the e2e flag", async () => {
    await expect(
      apiToken({ PW_E2E_SESSION: "1", PW_API_AUTH: "e2e", VERCEL_ENV: "production" }),
    ).rejects.toThrow(/outside Vercel/);
    await expect(
      apiToken({ PW_E2E_SESSION: "1", PW_API_AUTH: "e2e", VERCEL: "1" }),
    ).rejects.toThrow(/outside Vercel/);
  });

  it("returns the fake API's token for the test session", async () => {
    await expect(apiToken({ PW_E2E_SESSION: "1", PW_API_AUTH: "e2e" })).resolves.toBe("e2e");
  });

  it("mints a short single-use HS256 token for the hmac mode", async () => {
    const token = await apiToken({
      PW_API_AUTH: "hmac",
      PW_API_HMAC_SECRET: "a-test-secret-that-is-long-enough",
      ALLOWED_GITHUB_ID: "4242",
    });
    const claims = decodeJwt(token);
    expect(claims.aud).toBe(HMAC_AUDIENCE);
    expect(claims.sub).toBe("4242");
    expect(claims.exp! - claims.iat!).toBe(HMAC_TTL_S);
    expect(typeof claims.jti).toBe("string");
  });

  it("throws when no mode is configured", async () => {
    await expect(apiToken({})).rejects.toThrow(/no API credential/);
  });
});
