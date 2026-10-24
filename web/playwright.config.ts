import { defineConfig, devices } from "@playwright/test";

// The e2e suite runs against the fake API only: it never reaches the real server, holds no key
// and calls no model. One worker, because the fake API keeps a single global scenario.
const PORT = 3100;
const BASE_URL = `http://127.0.0.1:${PORT}`;

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
      url: "http://127.0.0.1:8787/healthz",
      reuseExistingServer: !process.env.CI,
      stdout: "pipe",
    },
    {
      command: `npm run build && npm run start -- --port ${PORT}`,
      url: `${BASE_URL}/signin`,
      reuseExistingServer: !process.env.CI,
      timeout: 300_000,
      // The test-only session and the fake API are the only credentials a test has.
      env: {
        PW_E2E_SESSION: "1",
        PW_API_URL: "http://127.0.0.1:8787",
        PW_API_AUTH: "e2e",
        AUTH_SECRET: "a-fixed-test-secret-that-is-long-enough",
        AUTH_TRUST_HOST: "true",
        ALLOWED_GITHUB_ID: "1",
      },
    },
  ],
});
