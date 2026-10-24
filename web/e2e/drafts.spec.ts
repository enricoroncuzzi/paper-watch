import { expect, test, type Page } from "@playwright/test";

import { setScenario, signIn } from "./helpers";

// The point-of-view dialog and the Drafts section, against the fake API only. The mockup's `#pov`
// dialog and `.draft` cards are the visual reference; these tests pin the behaviour the design
// text decides.

const LEAD_TITLE = "Replay Buffers Beat Summaries for Long-Running Coding Agents";
const LEAD_ID = "33333333-3333-4333-8333-333333333301";
const CREATED_BODY = "It cut tokens by 41% on 40 tasks, which is worth a look before adopting it.";
const POV = "Replaying tool results is the part I want to test on our own harness.";

test.beforeEach(async ({ page, request }) => {
  await setScenario(request, "normal");
  await signIn(page);
});

async function openDialog(page: Page) {
  await page
    .getByTestId("lead-pick")
    .getByRole("button", { name: `LinkedIn post preview: ${LEAD_TITLE}` })
    .click();
  const dialog = page.getByTestId("pov-dialog");
  await expect(dialog).toBeVisible();
  return dialog;
}

async function generate(page: Page, pov = POV) {
  const dialog = await openDialog(page);
  await dialog.getByTestId("pov-input").fill(pov);
  await dialog.getByRole("button", { name: "Generate draft" }).click();
  await expect(dialog.getByTestId("pov-preview")).toBeVisible();
  return dialog;
}

test("generating a draft saves it and shows its check notes", async ({ page }) => {
  const dialog = await openDialog(page);
  // The dialog carries the item and the same 30-second context the card shows.
  await expect(dialog).toContainText(LEAD_TITLE);
  await expect(dialog).toContainText("replay buffer of raw tool observations");

  const posted = page.waitForRequest(
    (request) => request.url().endsWith("/api/drafts") && request.method() === "POST",
  );
  await dialog.getByTestId("pov-input").fill(POV);
  await dialog.getByRole("button", { name: "Generate draft" }).click();
  expect((await posted).postDataJSON()).toEqual({ item_id: LEAD_ID, pov: POV, language: "en" });

  await expect(dialog.getByTestId("pov-preview")).toBeVisible();
  await expect(dialog.getByTestId("pov-saved")).toContainText("Saved to Drafts");
  await expect(dialog.getByTestId("draft-check-ok")).toContainText("All checks passed");
  // There is no separate save step: generating saved it already.
  await expect(dialog.getByRole("button", { name: "Save to Drafts" })).toHaveCount(0);

  await dialog.getByRole("button", { name: "Close", exact: true }).click();
  await expect(dialog).toBeHidden();
  // The new draft is first in Drafts, and the drafted item has left the picks.
  const drafts = page.getByTestId("drafts-section");
  await expect(drafts.getByTestId("draft-card").first()).toContainText(CREATED_BODY);
  await expect(drafts.getByTestId("draft-card").first()).toContainText(LEAD_TITLE);
  await expect(page.getByTestId("picks-section")).not.toContainText(LEAD_TITLE);
});

test("the button says it is writing until the draft arrives", async ({ page, request }) => {
  await setScenario(request, "draft_slow");
  await page.reload();

  const dialog = await openDialog(page);
  await dialog.getByTestId("pov-input").fill(POV);
  await dialog.getByRole("button", { name: "Generate draft" }).click();
  // The API writes one call and checks it; the button says so instead of looking stuck.
  await expect(
    dialog.getByRole("button", { name: "Writing and checking the draft..." }),
  ).toBeVisible();
  await expect(dialog.getByTestId("pov-preview")).toBeVisible({ timeout: 10_000 });
});

test("a draft with flags lists them numbered", async ({ page, request }) => {
  await setScenario(request, "draft_flags");
  await page.reload();

  const dialog = await generate(page, "I would test this on our own harness.");

  const notes = dialog.getByTestId("draft-checks");
  await expect(notes.getByTestId("draft-check-note")).toHaveCount(2);
  await expect(notes).toContainText("1. Numbers");
  await expect(notes).toContainText("2. Links");
  // The server's kind stays server-side: a reader sees words, never an enum. "number" and "link"
  // are ordinary English here, so the check looks for the forms only the database uses.
  const text = await notes.innerText();
  for (const raw of [
    "first_person",
    "preprint_fact",
    "early_notice",
    "long_copy",
    "peer_reviewed",
    "vetted_preprint",
  ]) {
    expect(text).not.toContain(raw);
  }
});

test("copy for linkedin puts the plain text on the clipboard", async ({ page, browserName }) => {
  test.skip(browserName !== "chromium", "clipboard permissions are granted on Chromium");
  await page.context().grantPermissions(["clipboard-read", "clipboard-write"]);

  const dialog = await generate(page);
  await dialog.getByRole("button", { name: /^Copy for LinkedIn:/ }).click();

  const copied = await page.evaluate(() => navigator.clipboard.readText());
  expect(copied).toBe(
    "A replay buffer is the part worth copying.\n\n" +
      `${CREATED_BODY}\n\n` +
      "#agents #memory\n\n" +
      "https://arxiv.org/abs/2610.04412",
  );
});

test("regenerate adds a new draft above the old one", async ({ page }) => {
  const drafts = page.getByTestId("drafts-section");
  await expect(drafts.getByTestId("draft-card")).toHaveCount(2);

  const regenerated = page.waitForRequest(
    (request) => /\/api\/drafts\/.+\/regenerate$/.test(request.url()) && request.method() === "POST",
  );
  await drafts
    .getByTestId("draft-card")
    .first()
    .getByRole("button", { name: /^Regenerate:/ })
    .click();
  await regenerated;

  await expect(drafts.getByTestId("draft-card")).toHaveCount(3);
  await expect(drafts.getByTestId("draft-card").first()).toContainText("Regenerated");
});

test("an empty point of view is allowed", async ({ page }) => {
  const posted = page.waitForRequest(
    (request) => request.url().endsWith("/api/drafts") && request.method() === "POST",
  );
  const dialog = await openDialog(page);
  await dialog.getByRole("button", { name: "Generate draft" }).click();

  expect((await posted).postDataJSON()).toMatchObject({ item_id: LEAD_ID, pov: "" });
  await expect(dialog.getByTestId("pov-preview")).toBeVisible();
});

test("the privacy refusal message is shown as is", async ({ page, request }) => {
  await setScenario(request, "draft_refused");
  await page.reload();

  const dialog = await openDialog(page);
  await dialog.getByTestId("pov-input").fill(POV);
  await dialog.getByRole("button", { name: "Generate draft" }).click();

  await expect(dialog.getByTestId("pov-error")).toContainText(
    "No model that refuses training on inputs answered, so the draft was not generated.",
  );
  await expect(dialog.getByTestId("pov-preview")).toHaveCount(0);
});

test("the daily cap names its reset time", async ({ page, request }) => {
  await setScenario(request, "draft_cap");
  await page.reload();

  const dialog = await openDialog(page);
  await dialog.getByRole("button", { name: "Generate draft" }).click();

  const error = dialog.getByTestId("pov-error");
  await expect(error).toContainText("The daily model cap is spent");
  await expect(error).toContainText("It resets at");
  await expect(error).toContainText("Europe/Rome");
});

test("a rising item says the draft will note it is early", async ({ page }) => {
  const rising = page.getByTestId("rising-section").getByTestId("rising-card").first();
  await expect(rising).toContainText("This item is early work; the draft will say so.");
  await expect(rising.getByRole("button", { name: /^LinkedIn post preview:/ })).toBeVisible();
});

test("the Drafts link stops below the masthead", async ({ page }) => {
  // The masthead is sticky: an anchor that puts the section under it hides the heading and its
  // rule. The header and scroll-padding-top share one custom property, and this asserts they meet.
  await page.getByRole("navigation", { name: "Sections" }).getByRole("link", { name: "Drafts" }).click();
  await expect
    .poll(async () => {
      const header = await page.locator("header").first().boundingBox();
      const section = await page.locator("#drafts").boundingBox();
      if (!header || !section) return -1;
      return Math.round(section.y - (header.y + header.height));
    })
    .toBeGreaterThanOrEqual(0);
});

test("closing the dialog returns focus to the button that opened it", async ({ page }) => {
  await openDialog(page);
  await page.keyboard.press("Escape");
  await expect(page.getByTestId("pov-dialog")).toBeHidden();
  const opener = page
    .getByTestId("lead-pick")
    .getByRole("button", { name: `LinkedIn post preview: ${LEAD_TITLE}` });
  expect(await opener.evaluate((element) => element === document.activeElement)).toBe(true);
});

test("a body keeps one paragraph per blank line", async ({ page }) => {
  const cards = page.getByTestId("drafts-section").getByTestId("draft-card");
  // The second fixture draft's body has two paragraphs; the card renders them as two paragraphs.
  await expect(cards.nth(1).getByTestId("draft-paragraph")).toHaveCount(2);
});

test("the dialog is full screen on the phone", async ({ page }, testInfo) => {
  test.skip(!testInfo.project.name.includes("iphone"), "checked on the iPhone project");

  const dialog = await openDialog(page);
  const box = await dialog.boundingBox();
  const viewport = page.viewportSize();
  expect(box).not.toBeNull();
  expect(viewport).not.toBeNull();
  if (box && viewport) {
    expect(box.width).toBeGreaterThanOrEqual(viewport.width - 1);
    expect(box.height).toBeGreaterThanOrEqual(viewport.height - 1);
  }
});
