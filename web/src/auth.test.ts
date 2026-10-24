import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// The session guard is what every page and route handler leans on, so it is tested against a
// stubbed Auth.js session instead of a browser: the owner id, another id, a missing id and an
// unset ALLOWED_GITHUB_ID must all behave.
const currentSession = vi.hoisted(() => ({ value: null as { githubId?: string } | null }));

vi.mock("next-auth", () => ({
  default: () => ({
    handlers: {},
    auth: async () => currentSession.value,
    signIn: vi.fn(),
    signOut: vi.fn(),
  }),
}));
vi.mock("next/navigation", () => ({
  redirect: (url: string) => {
    throw new Error(`NEXT_REDIRECT:${url}`);
  },
}));

import { ownerSession, requireOwner, requireOwnerApi } from "./auth";

describe("the owner guard", () => {
  beforeEach(() => {
    currentSession.value = { githubId: "4242" };
    vi.stubEnv("ALLOWED_GITHUB_ID", "4242");
  });

  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("accepts only the allowed numeric id", async () => {
    await expect(ownerSession()).resolves.toMatchObject({ githubId: "4242" });

    currentSession.value = { githubId: "9999" };
    await expect(ownerSession()).resolves.toBeNull();

    currentSession.value = {};
    await expect(ownerSession()).resolves.toBeNull();

    currentSession.value = null;
    await expect(ownerSession()).resolves.toBeNull();
  });

  it("fails closed when ALLOWED_GITHUB_ID is unset", async () => {
    vi.stubEnv("ALLOWED_GITHUB_ID", "");
    await expect(ownerSession()).resolves.toBeNull();
  });

  it("a route handler gets a 401 body, not a redirect", async () => {
    currentSession.value = null;
    const refused = await requireOwnerApi();
    expect(refused).toBeInstanceOf(Response);
    expect((refused as Response).status).toBe(401);
    expect(await (refused as Response).json()).toEqual({ kind: "unauthorized" });

    currentSession.value = { githubId: "4242" };
    const allowed = await requireOwnerApi();
    expect(allowed).not.toBeInstanceOf(Response);
  });

  it("a page is redirected to sign in", async () => {
    currentSession.value = null;
    await expect(requireOwner()).rejects.toThrow("NEXT_REDIRECT:/signin");
  });
});
