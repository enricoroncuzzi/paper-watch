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

  it("sends sign-in errors to the styled page and keeps the session to a week", () => {
    const config = buildAuthConfig({ ALLOWED_GITHUB_ID: OWNER_ID });
    expect(config.pages?.error).toBe("/signin");
    expect(config.session?.maxAge).toBe(7 * 24 * 60 * 60);
  });

  it("sign-in accepts only the allowed numeric id", () => {
    // The login is the same and the id is not: a renamed or recycled login must not sign in.
    expect(isAllowedGithubProfile({ id: 9999, login: "owner" }, OWNER_ID)).toBe(false);
    expect(isAllowedGithubProfile({ id: 4242, login: "someone-else" }, OWNER_ID)).toBe(true);
    expect(isAllowedGithubProfile({ login: "owner" }, OWNER_ID)).toBe(false);
    expect(isAllowedGithubProfile({ id: 4242 }, undefined)).toBe(false);
    // A prefix or a substring is not the id: 14242 and 42 must both be refused.
    expect(isAllowedGithubProfile({ id: 14242 }, OWNER_ID)).toBe(false);
    expect(isAllowedGithubProfile({ id: 42 }, OWNER_ID)).toBe(false);
  });

  it("the sign-in callback refuses every profile but the owner", async () => {
    const config = buildAuthConfig({ PW_E2E_SESSION: "1", ALLOWED_GITHUB_ID: OWNER_ID });
    const signIn = config.callbacks!.signIn!;

    expect(signIn({ profile: { id: 4242, login: "owner" } } as never)).toBe(true);
    expect(signIn({ profile: { id: 9999, login: "owner" } } as never)).toBe(false);
    expect(signIn({ profile: { id: 14242, login: "owner" } } as never)).toBe(false);
    // No profile is the credentials provider: allowed only when the e2e flag allows it.
    expect(signIn({ profile: undefined } as never)).toBe(true);
    const production = buildAuthConfig({ ALLOWED_GITHUB_ID: OWNER_ID });
    expect(production.callbacks!.signIn!({ profile: undefined } as never)).toBe(false);
  });

  it("the jwt stores the numeric id, never the login", async () => {
    const config = buildAuthConfig({ PW_E2E_SESSION: "1", ALLOWED_GITHUB_ID: OWNER_ID });
    const jwt = config.callbacks!.jwt!;

    const fromGithub = await jwt({
      token: {},
      profile: { id: 4242, login: "renamed-owner" },
      user: { id: "9999" },
      account: null,
      isNewUser: false,
      trigger: "signIn",
    } as never);
    expect(fromGithub!.githubId).toBe("4242");

    // The credentials provider has no profile; the e2e user's id is used, and only with the flag.
    const fromE2e = await jwt({
      token: {},
      profile: undefined,
      user: { id: "4242" },
      account: null,
      isNewUser: false,
      trigger: "signIn",
    } as never);
    expect(fromE2e!.githubId).toBe("4242");

    const session = await config.callbacks!.session!({
      session: { user: {}, expires: "" },
      token: { githubId: "4242" },
      user: { id: "4242" },
    } as never);
    expect((session as { githubId?: string }).githubId).toBe("4242");
  });
});
