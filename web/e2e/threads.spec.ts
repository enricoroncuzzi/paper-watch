import { expect, test, type Page } from "@playwright/test";

import { lastAsk, setScenario, signIn } from "./helpers";

// Follow-ups, end to end against the fake API: the conversation the sheet holds, the thread's
// turns read back after a dropped stream, the Recent menu and the phone layout.

/** The checked answer the normal stream sends, reused by the routed thread read below. */
const ANSWER = {
  text: "Coding agents are measured on real repository bugs [S1].",
  search_mode: "hybrid",
  supported: true,
  citations: [
    {
      id: "S1",
      item_id: "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
      title: "SWE-bench Verified: coding agents on real repository bugs",
      url: "https://arxiv.org/abs/2601.00001",
      tier: "peer_reviewed",
      basis: "venue",
    },
  ],
  flags: [],
};

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

test("coming back to the page leaves a healthy stream running", async ({ page, request }) => {
  await openSheet(page);
  await firstTurn(page);
  // A tab switch is not a dead connection: the stream is slow, so it is still open on return.
  await setScenario(request, "slow", { keepThreads: true });

  await ask(page, "What would break this approach?");
  const sheet = page.getByRole("dialog");
  await expect(sheet.getByTestId("caret")).toBeVisible();
  await page.evaluate(() => document.dispatchEvent(new Event("visibilitychange")));

  // Still streaming after the return, and the answer arrives instead of being killed.
  await expect(sheet.getByTestId("caret")).toBeVisible();
  await expect(sheet.getByTestId("interrupted")).toHaveCount(0);
  await expect(sheet.getByTestId("answer").locator("sup")).toHaveCount(1, { timeout: 15_000 });
  await expect(sheet.getByTestId("turns-chip")).toHaveText("2 turns");
});

test("a return to the page adopts the turn the server finished", async ({ page, request }) => {
  await openSheet(page);
  await firstTurn(page);
  await setScenario(request, "thread_drop", { keepThreads: true });
  await ask(page, "What would break this approach?");
  const sheet = page.getByRole("dialog");
  // The follow-up's stream died; the auto re-fetch finds no turn for it yet.
  await expect(sheet.getByTestId("interrupted")).toContainText(
    "That answer was interrupted. Ask it again.",
  );

  // The server finished the answer after the stream died, as the phone's re-fetch discovers.
  let reads = 0;
  await page.route("**/api/threads/**", async (route) => {
    reads += 1;
    await route.fulfill({
      json: {
        id: "any",
        title: "What do coding agents measure?",
        turns: [
          {
            n: 1,
            question: "What do coding agents measure?",
            answer: ANSWER,
            search_mode: "hybrid",
            created_at: new Date().toISOString(),
          },
          {
            n: 2,
            question: "What would break this approach?",
            answer: ANSWER,
            search_mode: "hybrid",
            created_at: new Date().toISOString(),
          },
        ],
      },
    });
  });
  await page.evaluate(() => document.dispatchEvent(new Event("visibilitychange")));

  await expect(sheet.getByTestId("interrupted")).toHaveCount(0);
  await expect(sheet.getByTestId("answer")).toContainText(
    "Coding agents are measured on real repository bugs",
  );
  await expect(sheet.getByTestId("turns-chip")).toHaveText("2 turns");
  expect(reads).toBeGreaterThan(0);
  // The adopted answer is stored: the sheet says nothing it cannot know about a filter.
  await expect(sheet.getByTestId("asked-question")).toHaveText("What would break this approach?");
  await expect(sheet.getByText("peer-reviewed only")).toHaveCount(0);
});

test("recent threads reopen a conversation", async ({ page }) => {
  await openSheet(page);
  const sheet = page.getByRole("dialog");

  const recent = sheet.getByRole("button", { name: "Recent", exact: true });
  await recent.click();
  const list = sheet.getByTestId("recent-threads");
  await expect(list).toBeVisible();
  const entry = list.getByRole("button", {
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

test("escape closes the recent list before it closes the sheet", async ({ page }) => {
  await openSheet(page);
  const sheet = page.getByRole("dialog");
  const recent = sheet.getByRole("button", { name: "Recent", exact: true });
  await recent.click();
  await expect(sheet.getByTestId("recent-threads")).toBeVisible();

  await page.keyboard.press("Escape");
  await expect(sheet.getByTestId("recent-threads")).toHaveCount(0);
  await expect(sheet).toBeVisible();
  // Focus goes back to the control that opened the list.
  await expect(recent).toBeFocused();
  await expect(recent).toHaveAttribute("aria-expanded", "false");

  await page.keyboard.press("Escape");
  await expect(sheet).toHaveCount(0);
});

test("a second click on Recent closes the list", async ({ page }) => {
  await openSheet(page);
  const sheet = page.getByRole("dialog");
  const recent = sheet.getByRole("button", { name: "Recent", exact: true });
  await recent.click();
  await expect(recent).toHaveAttribute("aria-expanded", "true");
  await recent.click();
  await expect(recent).toHaveAttribute("aria-expanded", "false");
  await expect(sheet.getByTestId("recent-threads")).toHaveCount(0);
});

test("the filter a chip asked with is shown and kept for a retry", async ({ page, request }) => {
  await openSheet(page);
  await firstTurn(page, "How do coding agents get evaluated?");
  const sheet = page.getByRole("dialog");
  await sheet.getByRole("button", { name: "Only peer-reviewed sources" }).click();
  await expect(sheet.getByTestId("turns-chip")).toHaveText("2 turns");
  // The sheet says which filter produced this answer, instead of claiming all tiers.
  await expect(sheet.getByText("peer-reviewed only · all topics")).toBeVisible();

  // An interrupted re-ask keeps the filter: the retry searches what the chip asked for.
  await setScenario(request, "thread_drop", { keepThreads: true });
  await sheet.getByRole("button", { name: "Only peer-reviewed sources" }).click();
  await expect(sheet.getByTestId("interrupted")).toBeVisible();
  // The interrupted block offers the retry, so the composer is not refilled with it as well.
  await expect(sheet.getByTestId("composer")).toHaveValue("");
  await setScenario(request, "normal", { keepThreads: true });
  await sheet.getByRole("button", { name: "Ask it again" }).click();
  await expect(sheet.getByTestId("answer").locator("sup")).toHaveCount(1);
  const sent = await lastAsk(request);
  expect(sent).toMatchObject({ tiers: ["peer_reviewed"] });
});

test("the sheet body is a live region and the Recent list is a disclosure", async ({ page }) => {
  await openSheet(page);
  const sheet = page.getByRole("dialog");
  await expect(sheet.locator("[aria-live='polite']")).not.toHaveCount(0);
  const recent = sheet.getByRole("button", { name: "Recent", exact: true });
  await recent.click();
  await expect(recent).toHaveAttribute("aria-expanded", "true");
  // No menu roles: the list is an ordinary disclosure of buttons.
  await expect(sheet.getByRole("menu")).toHaveCount(0);
  await expect(sheet.getByTestId("recent-threads").locator("button").first()).toBeVisible();
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
