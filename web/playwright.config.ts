import { defineConfig, devices } from "@playwright/test";

import { API_URL, BASE_URL, FAKE_API_PORT, PORT, REUSE_EXISTING_SERVER } from "./e2e/servers";

// The e2e suite runs against the fake API only: it never reaches the real server, holds no key
// and calls no model. One worker, because the fake API keeps a single global scenario.
// CI builds once in its own step and sets this, so Playwright does not build a second time; a local
// run builds first, so the e2e suite always tests the source it just changed.
const SERVER_COMMAND = process.env.PW_E2E_SKIP_BUILD
  ? `npm run start -- --port ${PORT}`
  : `npm run build && npm run start -- --port ${PORT}`;

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  workers: 1,
  forbidOnly: !!process.env.CI,
  retries: 0,
  reporter: process.env.CI ? [["list"], ["html", { open: "never" }]] : "list",
  use: {
    baseURL: BASE_URL,
    trace: "retain-on-failure",
  },
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
    { name: "webkit", use: { ...devices["Desktop Safari"] } },
    { name: "webkit-iphone", use: { ...devices["iPhone 15"] } },
  ],
  webServer: [
    {
      command: "node fake-api/server.mjs",
      url: `${API_URL}/healthz`,
      // Off unless asked for: a fixed port may hold another checkout's build, and a run that
      // attached to it would report the wrong code as green.
      reuseExistingServer: REUSE_EXISTING_SERVER,
      env: { FAKE_API_PORT: String(FAKE_API_PORT) },
      stdout: "pipe",
    },
    {
      command: SERVER_COMMAND,
      url: `${BASE_URL}/signin`,
      reuseExistingServer: REUSE_EXISTING_SERVER,
      timeout: 300_000,
      // The test-only session and the fake API are the only credentials a test has.
      env: {
        PW_E2E_SESSION: "1",
        PW_API_URL: API_URL,
        PW_API_AUTH: "e2e",
        // The page polls for a pending summary every 20 s; the suite cannot wait that long.
        PW_SUMMARY_REFRESH_MS: "400",
        AUTH_SECRET: "a-fixed-test-secret-that-is-long-enough",
        AUTH_TRUST_HOST: "true",
        ALLOWED_GITHUB_ID: "1",
      },
    },
  ],
});
