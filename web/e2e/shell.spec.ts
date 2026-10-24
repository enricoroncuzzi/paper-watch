import { expect, test } from "@playwright/test";

import { setScenario, signIn } from "./helpers";

// The page shell: the today strip, Runs, the states each section can be in, the theme and the
// palette. The mockup is the visual reference; these tests pin behaviour and the phone layout.

test.beforeEach(async ({ page, request }) => {
  await setScenario(request, "normal");
  await signIn(page);
});

test("the today strip shows the last run", async ({ page }) => {
  const strip = page.getByTestId("today-strip");
  await expect(strip).toContainText("healthy");
  await expect(strip).toContainText("06:00");
  await expect(strip).toContainText("Europe/Rome");
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
  await expect(strip).toContainText("10:30");
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
