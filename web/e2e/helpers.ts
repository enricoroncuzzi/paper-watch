import { expect, type APIRequestContext, type Page } from "@playwright/test";

import { API_URL } from "./servers";

/**
 * The fake API's scenario switch, so a test picks the state the page renders. `keepThreads` keeps
 * the conversations stored so far, so a test can ask, then change how the next stream behaves.
 */
export async function setScenario(
  request: APIRequestContext,
  name: string,
  options: { keepThreads?: boolean } = {},
): Promise<void> {
  const response = await request.post(`${API_URL}/__scenario`, {
    data: { name, ...options },
    headers: { authorization: "Bearer e2e" },
  });
  expect(response.ok()).toBe(true);
}

/** What the page last sent to `POST /ask`: the thread a follow-up continues and its filters. */
export async function lastAsk(request: APIRequestContext): Promise<Record<string, unknown>> {
  const response = await request.get(`${API_URL}/__last_ask`, {
    headers: { authorization: "Bearer e2e" },
  });
  expect(response.ok()).toBe(true);
  return (await response.json()) as Record<string, unknown>;
}

/** Sign in through the test-only provider. Every test gets a fresh browser context. */
export async function signIn(page: Page): Promise<void> {
  await page.goto("/signin");
  await page.getByRole("button", { name: "Sign in as the test owner" }).click();
  await expect(page).toHaveURL(/\/$/);
}
