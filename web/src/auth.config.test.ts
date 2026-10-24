import { describe, expect, it } from "vitest";

import { buildAuthConfig, E2E_PROVIDER_ID, isAllowedGithubProfile } from "./auth.config";

const OWNER_ID = "4242";

describe("the auth config", () => {
  it("has no e2e provider on Vercel", () => {
    const config = buildAuthConfig({
      PW_E2E_SESSION: "1",
      VERCEL_ENV: "production",
      ALLOWED_GITHUB_ID: OWNER_ID,
    });
    expect(config.providers.map((provider) => (provider as { id: string }).id)).toEqual(["github"]);
  });

  it("adds the e2e provider only with the flag off Vercel", () => {
    const config = buildAuthConfig({ PW_E2E_SESSION: "1", ALLOWED_GITHUB_ID: OWNER_ID });
    expect(config.providers.map((provider) => (provider as { id: string }).id)).toEqual([
      "github",
      E2E_PROVIDER_ID,
    ]);
    expect(buildAuthConfig({ ALLOWED_GITHUB_ID: OWNER_ID }).providers).toHaveLength(1);
  });

  it("sign-in accepts only the allowed numeric id", () => {
    // The login is the same and the id is not: a renamed or recycled login must not sign in.
    expect(isAllowedGithubProfile({ id: 9999, login: "owner" }, OWNER_ID)).toBe(false);
    expect(isAllowedGithubProfile({ id: 4242, login: "someone-else" }, OWNER_ID)).toBe(true);
    expect(isAllowedGithubProfile({ login: "owner" }, OWNER_ID)).toBe(false);
    expect(isAllowedGithubProfile({ id: 4242 }, undefined)).toBe(false);
  });
});
