import { expect, test, type Page } from "@playwright/test";

import { lastAsk, setScenario, signIn } from "./helpers";

// Follow-ups, end to end against the fake API: the conversation the sheet holds, the thread's
// turns read back after a dropped stream, the Recent menu and the phone layout.

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

/** One finished turn: the answer and the `thread` event that says it was stored. */
async function firstTurn(page: Page, question = "What do coding agents measure?"): Promise<void> {
  await ask(page, question);
  const sheet = page.getByRole("dialog");
  await expect(sheet.getByTestId("answer").locator("sup")).toHaveCount(1);
  await expect(sheet.getByTestId("turns-chip")).toHaveText("1 turn");
}

test.beforeEach(async ({ page, request }) => {
  await setScenario(request, "normal");
  await signIn(page);
});

test("a follow-up shows both turns and keeps the citation numbers", async ({ page }) => {
  await openSheet(page);
  await firstTurn(page);

  await ask(page, "And how was it measured?");
  const sheet = page.getByRole("dialog");
  // The first turn stays above the follow-up, with its own question, answer and footnotes.
  await expect(sheet.getByTestId("turn-1")).toContainText("What do coding agents measure?");
  await expect(sheet.getByTestId("turn-1").getByTestId("past-answer")).toContainText(
    "Coding agents are measured",
  );
  await expect(sheet.getByTestId("turn-1").getByTestId("past-sources")).toContainText(
    "SWE-bench Verified",
  );
  // The follow-up's own answer numbers its citations from 1 again, as its own footnote list.
  await expect(sheet.getByTestId("answer").locator("sup")).toHaveText(["1"]);
  await expect(sheet.getByTestId("past-answer").locator("sup")).toHaveText(["1"]);
  await expect(sheet.getByTestId("turns-chip")).toHaveText("2 turns");
  // The composer now invites a follow-up, and the sheet remembers which thread it is on.
  await expect(sheet.getByTestId("composer")).toHaveAttribute(
    "placeholder",
    "Ask a follow-up...",
  );
});

test("the peer-reviewed chip re-asks with the filter", async ({ page, request }) => {
  await openSheet(page);
  await firstTurn(page, "How do coding agents get evaluated?");

  const sheet = page.getByRole("dialog");
  await sheet.getByRole("button", { name: "Only peer-reviewed sources" }).click();
  await expect(sheet.getByTestId("turns-chip")).toHaveText("2 turns");

  const sent = await lastAsk(request);
  expect(sent).toMatchObject({
    question: "How do coding agents get evaluated?",
    tiers: ["peer_reviewed"],
  });
  // The chip continues the conversation rather than starting a new one.
  expect(typeof sent.thread_id).toBe("string");
});

test("a follow-up whose stream dies shows the finished turns and the interrupted line", async ({
  page,
  request,
}) => {
  await openSheet(page);
  await firstTurn(page);
  await setScenario(request, "thread_drop", { keepThreads: true });

  await ask(page, "What would break this approach?");
  const sheet = page.getByRole("dialog");
  // The server never finished that turn, so the sheet says so instead of spinning.
  await expect(sheet.getByTestId("interrupted")).toContainText(
    "That answer was interrupted. Ask it again.",
  );
  // The turn it did finish is still there, read back from the thread.
  await expect(sheet.getByTestId("turn-1")).toContainText("What do coding agents measure?");
  await expect(sheet.getByTestId("turns-chip")).toHaveText("1 turn");

  // "Ask it again" sends the same question, and the server's answer arrives this time.
  await setScenario(request, "normal", { keepThreads: true });
  await sheet.getByRole("button", { name: "Ask it again" }).click();
  await expect(sheet.getByTestId("answer").locator("sup")).toHaveCount(1);
  await expect(sheet.getByTestId("interrupted")).toHaveCount(0);
});

test("returning to a backgrounded tab shows the finished turns", async ({ page, request }) => {
  await openSheet(page);
  await firstTurn(page);
  // A frozen tab: the stream is slow, so it is still open when the page comes back.
  await setScenario(request, "slow", { keepThreads: true });

  await ask(page, "What would break this approach?");
  const sheet = page.getByRole("dialog");
  await expect(sheet.getByTestId("caret")).toBeVisible();
  // The phone returns to the page while the answer is still streaming: the sheet reads the thread
  // back instead of waiting for a watchdog that a frozen tab never fired.
  await page.evaluate(() => document.dispatchEvent(new Event("visibilitychange")));

  await expect(sheet.getByTestId("interrupted")).toContainText(
    "That answer was interrupted. Ask it again.",
  );
  await expect(sheet.getByTestId("turn-1")).toContainText("What do coding agents measure?");
});

test("recent threads reopen a conversation", async ({ page }) => {
  await openSheet(page);
  const sheet = page.getByRole("dialog");

  await sheet.getByRole("button", { name: "Recent" }).click();
  const menu = sheet.getByRole("menu", { name: "Recent conversations" });
  await expect(menu).toBeVisible();
  const entry = menu.getByRole("menuitem", {
    name: /What do recent papers say about memory for long-running coding agents\?, 3 turns/,
  });
  await expect(entry).toBeVisible();
  await entry.click();

  // The stored conversation opens with its last turn as the current one and the others above it.
  await expect(sheet.getByTestId("turns-chip")).toHaveText("3 turns");
  await expect(sheet.getByTestId("asked-question")).toHaveText("What are the limitations?");
  await expect(sheet.getByTestId("turn-1")).toContainText(
    "What do recent papers say about memory for long-running coding agents?",
  );
  await expect(sheet.getByTestId("turn-2")).toContainText("How was it measured?");
  await expect(sheet.getByTestId("composer")).toHaveAttribute(
    "placeholder",
    "Ask a follow-up...",
  );
});

test("escape closes the recent menu before it closes the sheet", async ({ page }) => {
  await openSheet(page);
  const sheet = page.getByRole("dialog");
  const recent = sheet.getByRole("button", { name: "Recent" });
  await recent.click();
  await expect(sheet.getByRole("menu", { name: "Recent conversations" })).toBeVisible();

  await page.keyboard.press("Escape");
  await expect(sheet.getByRole("menu", { name: "Recent conversations" })).toHaveCount(0);
  await expect(sheet).toBeVisible();
  // Focus goes back to the control that opened the list.
  await expect(recent).toBeFocused();

  await page.keyboard.press("Escape");
  await expect(sheet).toHaveCount(0);
});

test("the conversation keeps its chips and composer on the phone", async ({ page }, testInfo) => {
  test.skip(!testInfo.project.name.includes("iphone"), "checked on the iPhone project");
  await openSheet(page);
  await firstTurn(page);
  await ask(page, "And how was it measured?");

  const viewport = page.viewportSize();
  expect(viewport).not.toBeNull();
  const sheet = page.getByRole("dialog");
  const box = await sheet.boundingBox();
  expect(box).not.toBeNull();
  expect(box!.width).toBeGreaterThanOrEqual(viewport!.width - 1);
  expect(box!.height).toBeGreaterThanOrEqual(viewport!.height - 1);

  // The three fixed chips and the composer are visible without scrolling, above the safe area.
  for (const label of ["Only peer-reviewed sources", "What are the limitations?", "How was it measured?"]) {
    await expect(sheet.getByRole("button", { name: label })).toBeVisible();
  }
  const composer = await sheet.getByTestId("composer").boundingBox();
  expect(composer).not.toBeNull();
  expect(composer!.y + composer!.height).toBeLessThanOrEqual(viewport!.height);
  expect(composer!.y + composer!.height).toBeGreaterThanOrEqual(viewport!.height - 200);
  // The current answer is the one on screen: the earlier turn is above it, not in its place.
  await expect(sheet.getByTestId("asked-question")).toHaveText("And how was it measured?");
});
