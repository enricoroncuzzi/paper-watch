import { expect, type APIRequestContext, type Page } from "@playwright/test";

/** The fake API's scenario switch, so a test picks the state the page renders. */
export async function setScenario(request: APIRequestContext, name: string): Promise<void> {
  const response = await request.post("http://127.0.0.1:8787/__scenario", {
    data: { name },
    headers: { authorization: "Bearer e2e" },
  });
  expect(response.ok()).toBe(true);
}

/** Sign in through the test-only provider. Every test gets a fresh browser context. */
export async function signIn(page: Page): Promise<void> {
  await page.goto("/signin");
  await page.getByRole("button", { name: "Sign in as the test owner" }).click();
  await expect(page).toHaveURL(/\/$/);
}
