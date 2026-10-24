import { describe, expect, it } from "vitest";

import { e2eSessionAllowed } from "./e2e";

// The test session must be impossible to reach on Vercel, whatever else is set.
describe("the e2e guard", () => {
  it("allows the test session only off Vercel", () => {
    expect(e2eSessionAllowed({ PW_E2E_SESSION: "1" })).toBe(true);
    expect(e2eSessionAllowed({ PW_E2E_SESSION: "1", VERCEL_ENV: "production" })).toBe(false);
    expect(e2eSessionAllowed({ PW_E2E_SESSION: "1", VERCEL_ENV: "preview" })).toBe(false);
    expect(e2eSessionAllowed({ PW_E2E_SESSION: "1", VERCEL: "1" })).toBe(false);
  });

  it("refuses without the flag", () => {
    expect(e2eSessionAllowed({})).toBe(false);
    expect(e2eSessionAllowed({ PW_E2E_SESSION: "true" })).toBe(false);
  });
});
