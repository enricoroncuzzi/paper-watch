import { expect, test, type Page } from "@playwright/test";

import { lastAsk, setScenario, signIn } from "./helpers";

// Ask, end to end against the fake API: the pill, the sheet, the streamed trace and answer, the
// citation superscripts, every error state and the phone layout.

async function openSheet(page: Page): Promise<void> {
  await page.getByRole("button", { name: /Ask the corpus/ }).click();
  await expect(page.getByRole("dialog")).toBeVisible();
}

async function ask(page: Page, question: string): Promise<void> {
  const sheet = page.getByRole("dialog");
  await sheet.getByRole("textbox", { name: "Your question" }).fill(question);
  await sheet.getByRole("button", { name: "Send the question" }).click();
  await expect(sheet.getByTestId("asked-question")).toHaveText(question);
}

test.beforeEach(async ({ page, request }) => {
  await setScenario(request, "normal");
  await signIn(page);
});

test("asking streams tools, tokens and a checked answer", async ({ page }) => {
  await openSheet(page);
  await ask(page, "What do coding agents measure?");

  const sheet = page.getByRole("dialog");
  await expect(sheet.getByTestId("trace")).toContainText("search_corpus");
  await expect(sheet.getByTestId("answer")).toContainText(
    "Coding agents are measured on real repository bugs",
  );
  // A citation becomes a superscript only in the checked answer.
  await expect(sheet.getByTestId("answer").locator("sup")).toHaveCount(1);
  await expect(sheet.getByTestId("sources")).toContainText("SWE-bench Verified");
  await expect(sheet.getByTestId("sources")).toContainText("peer_reviewed");
  await expect(sheet.getByTestId("sources")).toContainText("arxiv.org");
});

test("citation numbers follow the footnote order, not the alias", async ({ page, request }) => {
  await setScenario(request, "citations");
  await openSheet(page);
  await ask(page, "What do agents measure and remember?");

  const sheet = page.getByRole("dialog");
  // The answer cites [S2] first, so its superscript is 1 and [S1] is 2.
  await expect(sheet.getByTestId("answer").locator("sup")).toHaveText(["1", "2"]);
  const footnotes = sheet.getByTestId("sources").locator("li");
  await expect(footnotes.nth(0)).toContainText("Memory for long-horizon agents");
  await expect(footnotes.nth(1)).toContainText("SWE-bench Verified");
});

test("an unchecked answer says so", async ({ page, request }) => {
  await setScenario(request, "flagged");
  await openSheet(page);
  await ask(page, "Did coding agents improve 40%?");

  const note = page.getByTestId("check-note");
  await expect(note).toContainText("Some sentences are not backed by a retrieved source.");
  await expect(note).toContainText("uncited claim");
  await expect(note).toContainText("No retrieved source states this number.");
});

test("the daily cap shows the reset time and the matching items", async ({ page, request }) => {
  await setScenario(request, "daily_cap");
  await openSheet(page);
  await ask(page, "What do coding agents measure?");

  const error = page.getByTestId("ask-error");
  await expect(error).toContainText("The daily model limit is spent.");
  await expect(error).toContainText("02:00 Europe/Rome");
  await expect(error).toContainText("Matching items");
  await expect(error).toContainText("SWE-bench Verified");
  // The item is listed once, inside the error box; the footnote list stays out of an error.
  await expect(
    page.getByText("SWE-bench Verified: coding agents on real repository bugs"),
  ).toHaveCount(1);
});

test("keyword-only answers show the notice", async ({ page, request }) => {
  await setScenario(request, "text_only");
  await openSheet(page);
  await ask(page, "What do agents remember?");

  await expect(page.getByTestId("notice")).toContainText(
    "Semantic search is paused while the corpus updates",
  );
  await expect(page.getByTestId("answer")).toContainText("Long-horizon agents plan better");
});

test("a dropped first question shows the interrupted line and keeps its thread", async ({
  page,
  request,
}) => {
  await setScenario(request, "drop");
  await openSheet(page);
  await ask(page, "What do coding agents measure?");

  // The server named the thread before the first token, so the re-fetch reads it back even though
  // no turn was stored: the sheet knows the question was interrupted, not merely that bytes stopped.
  const sheet = page.getByRole("dialog");
  await expect(sheet.getByTestId("interrupted")).toContainText(
    "That answer was interrupted. Ask it again.",
  );
  await expect(sheet.getByTestId("turns-chip")).toHaveCount(0);
  // The interrupted block carries the retry, so the question is not dumped into the composer too.
  await expect(sheet.getByTestId("composer")).toHaveValue("");

  await setScenario(request, "normal", { keepThreads: true });
  await sheet.getByRole("button", { name: "Ask it again" }).click();
  await expect(sheet.getByTestId("answer").locator("sup")).toHaveCount(1);
  await expect(sheet.getByTestId("interrupted")).toHaveCount(0);
  // The retry continues the thread the early event named instead of starting a second one.
  const sent = await lastAsk(request);
  expect(typeof sent.thread_id).toBe("string");
});

test("an aborted first question leaves no listed thread", async ({ page, request }) => {
  await setScenario(request, "slow");
  await openSheet(page);
  await ask(page, "What would an aborted question measure?");
  const sheet = page.getByRole("dialog");
  await expect(sheet.getByTestId("caret")).toBeVisible();
  await sheet.getByRole("button", { name: "Close the sheet" }).click();
  await expect(sheet).toHaveCount(0);

  // The empty row the early event created is not a conversation: Recent lists only finished turns.
  await openSheet(page);
  await sheet.getByRole("button", { name: "Recent", exact: true }).click();
  const list = sheet.getByTestId("recent-threads");
  await expect(list).toContainText("What do recent papers say about memory");
  await expect(list).not.toContainText("What would an aborted question measure?");
});

test("a rate-limited ask shows the wait", async ({ page, request }) => {
  await setScenario(request, "refused");
  await openSheet(page);
  await ask(page, "What do coding agents measure?");

  await expect(page.getByTestId("ask-error")).toContainText(
    "Too many questions right now. Try again in 30s.",
  );
});

test("the composer is empty while streaming and keeps what the reader types", async ({
  page,
  request,
}) => {
  await setScenario(request, "slow");
  await openSheet(page);
  await ask(page, "What do coding agents measure?");

  const composer = page.getByTestId("composer");
  // Sent, not yet answered: the composer holds nothing while the slow stream runs.
  await expect(composer).toHaveValue("");

  const typed = "A different follow-up I am typing";
  await composer.fill(typed);
  await expect(page.getByTestId("answer").locator("sup")).toHaveCount(1, { timeout: 15_000 });
  // The finished answer must not wipe what the reader typed while it streamed.
  await expect(composer).toHaveValue(typed);
});

test("a failed ask does not overwrite typing that came after it", async ({ page }) => {
  // Hold the answer long enough to type, then refuse it: the restore must not replace the draft.
  await page.route("**/api/ask", async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 500));
    await route.fulfill({
      status: 429,
      contentType: "application/json",
      body: JSON.stringify({ kind: "rate_limited", retryAfterS: 30 }),
    });
  });
  await openSheet(page);
  const composer = page.getByTestId("composer");
  await composer.fill("first question");
  await page.getByRole("dialog").getByRole("button", { name: "Send the question" }).click();
  await composer.fill("newer typing");

  await expect(page.getByTestId("ask-error")).toContainText("Too many questions right now");
  await expect(composer).toHaveValue("newer typing");
});

test("a failed ask puts the question back in the composer", async ({ page, request }) => {
  await setScenario(request, "refused");
  await openSheet(page);
  await ask(page, "What do coding agents measure?");

  await expect(page.getByTestId("ask-error")).toContainText("Too many questions right now");
  await expect(page.getByTestId("composer")).toHaveValue("What do coding agents measure?");
});

test("a slow answer keeps the caret while heartbeats arrive", async ({ page, request }) => {
  await setScenario(request, "slow");
  await openSheet(page);
  await ask(page, "What do coding agents measure?");

  const sheet = page.getByRole("dialog");
  await expect(sheet.getByTestId("caret")).toBeVisible();
  // Still streaming: the citation is literal text in the draft, not a link yet.
  await expect(sheet.getByTestId("answer").locator("sup")).toHaveCount(0);

  // The answer arrives after the heartbeats: the caret goes and the superscript appears.
  await expect(sheet.getByTestId("answer").locator("sup")).toHaveCount(1, { timeout: 15_000 });
  await expect(sheet.getByTestId("caret")).toHaveCount(0);
});

test("command enter opens the sheet and sends", async ({ page }) => {
  await page.keyboard.press("ControlOrMeta+Enter");
  const sheet = page.getByRole("dialog");
  await expect(sheet).toBeVisible();
  await sheet
    .getByRole("textbox", { name: "Your question" })
    .fill("What do coding agents measure?");
  await page.keyboard.press("ControlOrMeta+Enter");
  await expect(sheet.getByTestId("answer")).toContainText("Coding agents are measured");
});

test("the ask bar keeps its send circle at the right end on every width", async ({ page }) => {
  const pill = page.getByRole("button", { name: /Ask the corpus/ });
  const send = pill.locator("span").last();
  const pillBox = await pill.boundingBox();
  const sendBox = await send.boundingBox();
  expect(pillBox).not.toBeNull();
  expect(sendBox).not.toBeNull();
  // The circle ends at the bar's right padding, not after the label (the hint is hidden on touch).
  const gap = pillBox!.x + pillBox!.width - (sendBox!.x + sendBox!.width);
  expect(gap).toBeLessThanOrEqual(20);
});

test("the sheet is full screen on the phone and the composer sits above the safe area", async ({
  page,
}, testInfo) => {
  test.skip(!testInfo.project.name.includes("iphone"), "checked on the iPhone project");
  await openSheet(page);
  await ask(page, "What do coding agents measure?");

  const viewport = page.viewportSize();
  expect(viewport).not.toBeNull();
  const sheet = await page.getByRole("dialog").boundingBox();
  expect(sheet).not.toBeNull();
  expect(sheet!.width).toBeGreaterThanOrEqual(viewport!.width - 1);
  expect(sheet!.height).toBeGreaterThanOrEqual(viewport!.height - 1);

  const composer = page.getByTestId("composer");
  await expect(composer).toHaveCSS("font-size", "16px");
  const box = await composer.boundingBox();
  expect(box).not.toBeNull();
  // Above the safe area and inside the screen: the composer is the last thing above the home bar.
  expect(box!.y + box!.height).toBeLessThanOrEqual(viewport!.height);
  expect(box!.y + box!.height).toBeGreaterThanOrEqual(viewport!.height - 160);
});
