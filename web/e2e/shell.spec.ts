import { expect, test } from "@playwright/test";

import { setScenario, signIn } from "./helpers";

// The page shell: the today strip, Runs, the states each section can be in, the theme and the
// palette. The mockup is the visual reference; these tests pin behaviour and the phone layout.

test.beforeEach(async ({ page, request }) => {
  await setScenario(request, "normal");
  await signIn(page);
});

// A first visit has no cookie: the palette must still be the light one on a light OS, and the dark
// one on a dark OS. Every other test sets a theme, so only this pair can catch a missing default.
test.describe("a first visit with no theme cookie on a light OS", () => {
  test.use({ colorScheme: "light" });

  test("gets the light palette", async ({ page }) => {
    await page.goto("/signin");
    const paper = await page.evaluate(() =>
      getComputedStyle(document.documentElement).getPropertyValue("--paper").trim(),
    );
    expect(paper).toBe("#f6f5ef");
  });
});

test.describe("a first visit with no theme cookie on a dark OS", () => {
  test.use({ colorScheme: "dark" });

  test("gets the dark palette", async ({ page }) => {
    await page.goto("/signin");
    const paper = await page.evaluate(() =>
      getComputedStyle(document.documentElement).getPropertyValue("--paper").trim(),
    );
    expect(paper).toBe("#12140f");
  });
});

test("the today strip shows the last run", async ({ page }) => {
  const strip = page.getByTestId("today-strip");
  await expect(strip).toContainText("healthy");
  await expect(strip).toContainText("Europe/Rome");
  await expect(page.getByTestId("run-day")).toHaveText("Today");
  await expect(strip).toContainText("9 stored");
  await expect(strip).toContainText("41 fetched");
  await expect(strip).toContainText("Runs");
});

test("an unhealthy run shows its first problem", async ({ page, request }) => {
  await setScenario(request, "unhealthy");
  await page.reload();
  const strip = page.getByTestId("today-strip");
  await expect(strip).toContainText("unhealthy");
  await expect(strip).toContainText("arxiv fetched 0 items");
});

test("a waiting follow-up shows the next pass", async ({ page, request }) => {
  await setScenario(request, "followup");
  await page.reload();
  const strip = page.getByTestId("today-strip");
  await expect(strip).toContainText("follow-up pass 3 of 40");
  await expect(strip).toContainText("12 papers pending");
});

test("keyword-only mode is announced in the strip", async ({ page, request }) => {
  await setScenario(request, "text_only");
  await page.reload();
  await expect(page.getByTestId("today-strip")).toContainText(
    "keyword search while the corpus updates",
  );
});

test("runs open a run detail with dead letters", async ({ page }) => {
  const runs = page.getByTestId("runs-section");
  await runs.getByRole("link").first().click();
  await expect(page).toHaveURL(/\/runs\/11111111-1111-4111-8111-111111111111$/);
  await expect(page.getByTestId("run-detail")).toContainText("read timeout after 20 s");
  await expect(page.getByTestId("run-detail")).toContainText("2601.00001");
  await expect(page.getByTestId("run-detail")).toContainText("window");
});

test("each section shows its loading empty and error states", async ({ page, request }) => {
  await setScenario(request, "slow");
  await page.goto("/", { waitUntil: "commit" });
  await expect(page.getByTestId("today-loading")).toBeVisible();
  await expect(page.getByTestId("today-strip")).toBeVisible();

  await setScenario(request, "empty");
  await page.goto("/");
  await expect(page.getByTestId("runs-section")).toContainText("No runs recorded");
  await expect(page.getByTestId("today-strip")).toContainText("No run recorded yet");

  await setScenario(request, "error");
  await page.goto("/");
  await expect(page.getByTestId("today-error")).toContainText(
    "The API is unreachable right now. The ingestion is unaffected.",
  );
  await expect(page.getByTestId("runs-error")).toContainText(
    "The API is unreachable right now. The ingestion is unaffected.",
  );
});

test("a run from an earlier day says so in amber", async ({ page, request }) => {
  await setScenario(request, "yesterday");
  await page.reload();

  const day = page.getByTestId("run-day");
  await expect(day).toHaveText("Yesterday");
  await expect(day).toHaveClass(/text-warn/);
});

test("the run detail heading names the run's Rome day before its time", async ({ page, request }) => {
  const detail = "/runs/11111111-1111-4111-8111-111111111111";
  await page.goto(detail);
  await expect(page.getByTestId("run-detail").getByRole("heading", { level: 1 })).toHaveText(
    /^Today \d{2}:\d{2} Europe\/Rome$/,
  );

  await setScenario(request, "yesterday");
  await page.goto(detail);
  await expect(page.getByTestId("run-detail").getByRole("heading", { level: 1 })).toHaveText(
    /^Yesterday \d{2}:\d{2} Europe\/Rome$/,
  );
});

test("a link to a malformed run id says nothing here", async ({ page }) => {
  const response = await page.goto("/runs/not-a-uuid");
  expect(response?.status()).toBe(404);
  await expect(page.getByRole("heading", { name: "Nothing here" })).toBeVisible();
});

test("the theme toggle persists across a reload", async ({ page }) => {
  const toggle = page.getByRole("button", { name: "Toggle theme" });
  await toggle.click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await page.reload();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await toggle.click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
});

test("command k opens the palette and navigates", async ({ page }) => {
  await page.keyboard.press("ControlOrMeta+k");
  const palette = page.getByRole("dialog");
  await expect(palette).toBeVisible();
  await palette.getByRole("option", { name: /Runs/ }).click();
  await expect(page).toHaveURL(/#runs$/);
  await expect(page.getByRole("heading", { name: "Runs" })).toBeVisible();
});

test("from a run detail the masthead and the palette return to Runs", async ({ page }) => {
  const detail = "/runs/11111111-1111-4111-8111-111111111111";
  await page.goto(detail);
  await page.getByRole("navigation", { name: "Sections" }).getByRole("link", { name: "Runs" }).click();
  await expect(page).toHaveURL(/\/#runs$/);
  await expect(page.getByTestId("runs-section")).toBeVisible();

  await page.goto(detail);
  await expect(page.getByTestId("run-detail")).toBeVisible();
  await page.keyboard.press("ControlOrMeta+k");
  await page.getByRole("dialog").getByRole("option", { name: /Runs/ }).click();
  await expect(page).toHaveURL(/\/#runs$/);
  await expect(page.getByTestId("runs-section")).toBeVisible();
});

test("the phone layout has no horizontal scroll and 44px targets", async ({ page }, testInfo) => {
  // The 44 px rule is the phone layout's; on a desktop viewport the masthead follows the mockup.
  test.skip(!testInfo.project.name.includes("iphone"), "checked on the iPhone project");
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - window.innerWidth,
  );
  expect(overflow).toBeLessThanOrEqual(0);

  // Only the controls a thumb can hit: a control hidden at this width has no box.
  const boxes = await page.locator("header button, header a").evaluateAll((elements) =>
    elements
      .filter((element) => element.getClientRects().length > 0)
      .map((element) => {
        const box = element.getBoundingClientRect();
        return { width: box.width, height: box.height };
      }),
  );
  expect(boxes.length).toBeGreaterThan(0);
  for (const box of boxes) {
    expect(box.height).toBeGreaterThanOrEqual(44);
    expect(box.width).toBeGreaterThanOrEqual(44);
  }
});
