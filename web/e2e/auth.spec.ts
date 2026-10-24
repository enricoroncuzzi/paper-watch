import { expect, test } from "@playwright/test";

import { signIn } from "./helpers";

// The session gate, end to end: an anonymous browser never renders the page, and the test-only
// provider reaches it only because the server runs with PW_E2E_SESSION=1 off Vercel.

test("an anonymous visit lands on the sign-in page", async ({ page }) => {
  await page.goto("/");
  await expect(page).toHaveURL(/\/signin$/);
  await expect(page.getByRole("heading", { name: "Paper Watch" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Sign in with GitHub" })).toBeVisible();
  await expect(page.getByText("A private reading room for the AI-research corpus.")).toBeVisible();
});

test("the e2e session signs in and reaches the page", async ({ page }) => {
  await page.goto("/signin");
  await page.getByRole("button", { name: "Sign in as the test owner" }).click();
  await expect(page).toHaveURL(/\/$/);
  await expect(page.getByTestId("today-strip")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Runs" })).toBeVisible();
});

test("a refused account lands on the styled sign-in page", async ({ page }) => {
  await page.goto("/signin?error=AccessDenied");
  await expect(page.getByTestId("signin-refused")).toContainText(
    "That GitHub account is not the owner of this page.",
  );
  await expect(page.getByRole("button", { name: "Sign in with GitHub" })).toBeVisible();
});

test("any other sign-in failure does not blame the account", async ({ page }) => {
  await page.goto("/signin?error=Configuration");
  const notice = page.getByTestId("signin-refused");
  await expect(notice).toContainText("Sign-in did not complete.");
  await expect(notice).not.toContainText("not the owner");
});

test("the page sends its security headers", async ({ page }) => {
  await signIn(page);
  for (const path of ["/signin", "/"]) {
    const response = await page.request.get(path);
    expect(response.status()).toBe(200);
    expect(response.headers()["x-frame-options"]).toBe("DENY");
    expect(response.headers()["referrer-policy"]).toBe("no-referrer");
    expect(response.headers()["x-content-type-options"]).toBe("nosniff");
    expect(response.headers()["permissions-policy"]).toBe(
      "camera=(), microphone=(), geolocation=()",
    );
    expect(response.headers()["x-powered-by"]).toBeUndefined();
  }
});

test("an anonymous document request is redirected even with a bypass header", async ({
  request,
}) => {
  // Next 16 no longer honours this header, so this pins the proxy's redirect for a document
  // request, not the per-page guard: that one is unit-tested in src/app/guards.test.ts.
  const headers = { "x-middleware-subrequest": "middleware" };
  for (const path of ["/", "/runs/11111111-1111-4111-8111-111111111111"]) {
    const response = await request.get(path, { headers, maxRedirects: 0 });
    expect([307, 401]).toContain(response.status());
    if (response.status() === 307) {
      expect(response.headers()["location"]).toContain("/signin");
    }
  }
});

test("an api route without a session is 401", async ({ request }) => {
  const response = await request.post("/api/ask", { data: { question: "anything" } });
  expect(response.status()).toBe(401);
  expect(await response.json()).toEqual({ kind: "unauthorized" });
});
