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

const LEAD_TITLE = "Replay Buffers Beat Summaries for Long-Running Coding Agents";
const LEAD_ID = "33333333-3333-4333-8333-333333333301";

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
  await expect(sheet.getByTestId("composer")).toHaveAttribute("placeholder", "Ask a follow-up...");
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
  const read = page.waitForRequest("**/api/threads/**");
  await page.evaluate(() => document.dispatchEvent(new Event("visibilitychange")));
  // The return does read the thread, and finds no finished turn for this question.
  expect((await read).url()).toContain("/api/threads/");

  // Still streaming after the return, and the answer arrives instead of being killed.
  await expect(sheet.getByTestId("caret")).toBeVisible();
  await expect(sheet.getByTestId("interrupted")).toHaveCount(0);
  await expect(sheet.getByTestId("answer").locator("sup")).toHaveCount(1, { timeout: 15_000 });
  await expect(sheet.getByTestId("turns-chip")).toHaveText("2 turns");
});

test("returning to a reopened conversation does not duplicate its last turn", async ({ page }) => {
  await openSheet(page);
  const sheet = page.getByRole("dialog");
  await sheet.getByRole("button", { name: "Recent", exact: true }).click();
  await sheet
    .getByTestId("recent-threads")
    .getByRole("button", { name: /What do recent papers say about memory for long-running/ })
    .click();
  await expect(sheet.getByTestId("turns-chip")).toHaveText("3 turns");
  await expect(sheet.getByTestId("turn-1")).toBeVisible();
  await expect(sheet.getByTestId("turn-2")).toBeVisible();

  const reads: string[] = [];
  page.on("request", (request) => {
    if (/\/api\/threads\//.test(request.url())) reads.push(request.url());
  });
  // An iPhone locks and unlocks on every look: each return used to move the last turn into the
  // earlier ones as well. A finished conversation has nothing to recover, so nothing is read.
  await page.evaluate(() => document.dispatchEvent(new Event("visibilitychange")));
  await page.evaluate(() => document.dispatchEvent(new Event("visibilitychange")));
  await expect(sheet.getByTestId("asked-question")).toHaveText("What are the limitations?");
  await expect(sheet.getByTestId("turn-1")).toHaveCount(1);
  await expect(sheet.getByTestId("turn-2")).toHaveCount(1);
  await expect(sheet.locator('[data-testid^="turn-"]')).toHaveCount(2);
  // Give a stray read the time to appear before asserting there was none.
  await page.waitForTimeout(300);
  expect(reads).toEqual([]);
});

test("a refused follow-up keeps its error when the reader comes back", async ({
  page,
  request,
}) => {
  await openSheet(page);
  await firstTurn(page);
  // The daily cap is spent: a return must not turn that into "That answer was interrupted",
  // which would tell the reader to retry something that cannot succeed until it resets.
  await setScenario(request, "daily_cap", { keepThreads: true });
  await ask(page, "What would break this approach?");
  const sheet = page.getByRole("dialog");
  await expect(sheet.getByTestId("ask-error")).toContainText("The daily model limit is spent", {
    timeout: 15_000,
  });

  await page.evaluate(() => document.dispatchEvent(new Event("visibilitychange")));
  await expect(sheet.getByTestId("ask-error")).toContainText("The daily model limit is spent");
  await expect(sheet.getByTestId("ask-error")).toBeVisible();
  await expect(sheet.getByTestId("interrupted")).toHaveCount(0);
});

test("a refusal that lands while the return's read is in flight keeps its error", async ({
  page,
}) => {
  await openSheet(page);
  await firstTurn(page);
  const sheet = page.getByRole("dialog");

  // The follow-up's answer is held open, so the sheet is streaming when the reader returns.
  let releaseAsk!: () => void;
  const askGate = new Promise<void>((resolve) => {
    releaseAsk = resolve;
  });
  await page.route("**/api/ask", async (route) => {
    await askGate;
    await route.fulfill({
      status: 200,
      headers: { "content-type": "text/event-stream" },
      body: `event: error\ndata: ${JSON.stringify({ code: "daily_cap", message: "The daily model limit is spent.", retry_at: "2026-10-26T00:00:00Z" })}\n\n`,
    });
  });
  // The thread read is held too, and holds only the turn that was stored before this request.
  let releaseRead!: () => void;
  const readGate = new Promise<void>((resolve) => {
    releaseRead = resolve;
  });
  await page.route("**/api/threads/**", async (route) => {
    await readGate;
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
        ],
      },
    });
  });

  await sheet
    .getByRole("textbox", { name: "Your question" })
    .fill("What would break this approach?");
  await sheet.getByRole("button", { name: "Send the question" }).click();
  await expect(sheet.getByTestId("asked-question")).toHaveText("What would break this approach?");

  const readRequest = page.waitForRequest("**/api/threads/**");
  const readResponse = page.waitForResponse("**/api/threads/**");
  await page.evaluate(() => document.dispatchEvent(new Event("visibilitychange")));
  await readRequest;

  // The refusal lands first, then the read taken before it: the refusal has to stay, because a
  // retry cannot succeed until the cap resets and the interrupted line would demand one.
  releaseAsk();
  await expect(sheet.getByTestId("ask-error")).toContainText("The daily model limit is spent", {
    timeout: 15_000,
  });
  releaseRead();
  await readResponse;
  await expect(sheet.getByTestId("ask-error")).toContainText("The daily model limit is spent");
  await expect(sheet.getByTestId("interrupted")).toHaveCount(0);
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

test("a dropped follow-up whose thread read fails still offers the retry", async ({
  page,
  request,
}) => {
  await openSheet(page);
  await firstTurn(page);
  await setScenario(request, "thread_drop", { keepThreads: true });
  // The re-fetch that would say what the server finished fails too: without the block's own
  // retry the reader has to retype the question.
  await page.route("**/api/threads/**", (route) =>
    route.fulfill({ status: 500, json: { detail: "threads are unavailable" } }),
  );

  await ask(page, "What would break this approach?");
  const sheet = page.getByRole("dialog");
  const block = sheet.getByTestId("ask-error");
  await expect(block).toContainText("The connection dropped.");
  await expect(sheet.getByTestId("interrupted")).toHaveCount(0);
  await expect(sheet.getByTestId("composer")).toHaveValue("");

  await page.unroute("**/api/threads/**");
  await setScenario(request, "normal", { keepThreads: true });
  await block.getByRole("button", { name: "Ask it again" }).click();
  await expect(sheet.getByTestId("answer").locator("sup")).toHaveCount(1);
  const sent = await lastAsk(request);
  expect(sent).toMatchObject({ question: "What would break this approach?" });
  expect(typeof sent.thread_id).toBe("string");
});

test("ask about it after a conversation starts a fresh one about the item", async ({
  page,
  request,
}) => {
  await openSheet(page);
  await firstTurn(page);
  const sheet = page.getByRole("dialog");
  await sheet.getByRole("button", { name: "Close the sheet" }).click();
  await expect(sheet).toHaveCount(0);

  await page
    .getByTestId("picks-section")
    .getByTestId("lead-pick")
    .getByRole("button", { name: /^Ask about it/ })
    .click();
  await expect(sheet.getByTestId("asking-about")).toContainText(LEAD_TITLE);
  await ask(page, "What did they measure?");
  await expect(sheet.getByTestId("answer").locator("sup")).toHaveCount(1);

  // The question is about the item the banner names: a new thread that registers it as S1.
  const sent = await lastAsk(request);
  expect(sent).toEqual({ question: "What did they measure?", item_id: LEAD_ID });
  await expect(sheet.locator('[data-testid^="turn-"]')).toHaveCount(0);
  await expect(sheet.getByTestId("turns-chip")).toHaveText("1 turn");
});

test("opening a recent conversation lets go of the item", async ({ page, request }) => {
  await page
    .getByTestId("picks-section")
    .getByTestId("lead-pick")
    .getByRole("button", { name: /^Ask about it/ })
    .click();
  const sheet = page.getByRole("dialog");
  await expect(sheet.getByTestId("asking-about")).toContainText(LEAD_TITLE);

  await sheet.getByRole("button", { name: "Recent", exact: true }).click();
  await sheet
    .getByTestId("recent-threads")
    .getByRole("button", { name: /What do recent papers say about memory for long-running/ })
    .click();
  await expect(sheet.getByTestId("turns-chip")).toHaveText("3 turns");
  // The stored conversation is not about the card's item, so the sheet no longer says it is.
  await expect(sheet.getByTestId("asking-about")).toHaveCount(0);

  await ask(page, "Which of them is peer-reviewed?");
  await expect(sheet.getByTestId("answer").locator("sup")).toHaveCount(1);
  const sent = await lastAsk(request);
  expect(sent).not.toHaveProperty("item_id");
  expect(typeof sent.thread_id).toBe("string");
});

test("a follow-up waits for the thread event that follows the answer", async ({
  page,
  request,
}) => {
  await setScenario(request, "thread_slow");
  await openSheet(page);
  await ask(page, "What do coding agents measure?");
  const sheet = page.getByRole("dialog");
  await expect(sheet.getByTestId("answer").locator("sup")).toHaveCount(1);

  // The answer is on screen, but the conversation it belongs to is not named yet: no way of
  // sending a follow-up may start a second, history-less thread in that gap.
  const chip = sheet.getByRole("button", { name: "What are the limitations?" });
  await expect(chip).toBeDisabled();
  await expect(sheet.getByRole("button", { name: "Send the question" })).toBeDisabled();
  await sheet.getByTestId("composer").fill("And how was it measured?");
  await sheet.getByTestId("composer").press("ControlOrMeta+Enter");
  await expect(sheet.getByTestId("asked-question")).toHaveText("What do coding agents measure?");
  await expect(sheet.getByTestId("composer")).toHaveValue("And how was it measured?");
  await expect(sheet.getByTestId("turns-chip")).toHaveCount(0);

  await expect(sheet.getByTestId("turns-chip")).toHaveText("1 turn");
  await expect(chip).toBeEnabled();
  await chip.click();
  await expect(sheet.getByTestId("asked-question")).toHaveText("What are the limitations?");
  const sent = await lastAsk(request);
  expect(typeof sent.thread_id).toBe("string");
});

test("a recent pick whose read fails while an answer streams offers the retry", async ({
  page,
  request,
}) => {
  await setScenario(request, "slow");
  await openSheet(page);
  await ask(page, "What do coding agents measure?");
  const sheet = page.getByRole("dialog");
  await expect(sheet.getByTestId("caret")).toBeVisible();

  await page.route(
    (url) => /\/api\/threads\/[^/]+$/.test(url.pathname),
    (route) => route.fulfill({ status: 500, json: { detail: "threads are unavailable" } }),
  );
  await sheet.getByRole("button", { name: "Recent", exact: true }).click();
  await sheet
    .getByTestId("recent-threads")
    .getByRole("button", { name: /What do recent papers say about memory for long-running/ })
    .click();

  // The pick stopped the stream: the sheet says so and offers the retry instead of spinning.
  const block = sheet.getByTestId("ask-error");
  await expect(block).toContainText("The connection dropped.");
  await expect(sheet.getByTestId("caret")).toHaveCount(0);
  await expect(sheet.getByRole("button", { name: "What are the limitations?" })).toBeEnabled();
  // The first question's early thread already exists, so the switched scenario keeps it.
  await setScenario(request, "normal", { keepThreads: true });
  await block.getByRole("button", { name: "Ask it again" }).click();
  await expect(sheet.getByTestId("answer").locator("sup")).toHaveCount(1);
  expect(await lastAsk(request)).toMatchObject({ question: "What do coding agents measure?" });
});

test("the trace counts the sources the answer cites", async ({ page }) => {
  await openSheet(page);
  await firstTurn(page);
  const sheet = page.getByRole("dialog");
  // A follow-up that only re-reads an earlier paper announces no `source` event of its own; the
  // count must still agree with the footnotes the answer carries.
  await page.route("**/api/ask", (route) =>
    route.fulfill({
      status: 200,
      headers: { "content-type": "text/event-stream" },
      body: `event: step\ndata: ${JSON.stringify({ round: 1, ms: 12, new_sources: 0 })}\n\nevent: answer\ndata: ${JSON.stringify(ANSWER)}\n\n`,
    }),
  );
  await ask(page, "And how was it measured?");
  const trace = sheet.getByTestId("trace");
  await expect(trace).toBeVisible();
  await expect(trace).toContainText("1 source");
  await expect(trace).not.toContainText("0 sources");
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
  await expect(sheet.getByTestId("composer")).toHaveAttribute("placeholder", "Ask a follow-up...");
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

test("tab from Recent reaches the list before the close button", async ({ page }, testInfo) => {
  test.skip(
    !testInfo.project.name.includes("chromium"),
    "desktop keyboard focus order is checked in Chromium",
  );
  await openSheet(page);
  const sheet = page.getByRole("dialog");
  const recent = sheet.getByRole("button", { name: "Recent", exact: true });
  await recent.click();
  const list = sheet.getByTestId("recent-threads");
  await expect(list.locator("button").first()).toBeVisible();
  // The list follows its trigger in the document, so Tab lands in it instead of on Close.
  await page.keyboard.press("Tab");
  await expect(list.locator("button").first()).toBeFocused();
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
  for (const label of [
    "Only peer-reviewed sources",
    "What are the limitations?",
    "How was it measured?",
  ]) {
    await expect(sheet.getByRole("button", { name: label })).toBeVisible();
  }
  const composer = await sheet.getByTestId("composer").boundingBox();
  expect(composer).not.toBeNull();
  expect(composer!.y + composer!.height).toBeLessThanOrEqual(viewport!.height);
  expect(composer!.y + composer!.height).toBeGreaterThanOrEqual(viewport!.height - 200);
  // The current answer is the one on screen: the earlier turn is above it, not in its place.
  await expect(sheet.getByTestId("asked-question")).toHaveText("And how was it measured?");
});
