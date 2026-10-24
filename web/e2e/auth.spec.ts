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

test("a known proxy bypass header still lands on sign-in", async ({ request }) => {
  // The per-page guard is the real check; the proxy is only a courtesy. The header that once
  // skipped Next's proxy must not open either page.
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
