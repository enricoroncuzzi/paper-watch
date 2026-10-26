import { expect, test } from "@playwright/test";

import { lastAsk, setScenario, signIn } from "./helpers";

// The picks and the Rising shelf, against the fake API only. The mockup is the visual reference;
// these tests pin the behaviour the design text decides.

const LEAD_TITLE = "Replay Buffers Beat Summaries for Long-Running Coding Agents";
const LEAD_ID = "33333333-3333-4333-8333-333333333301";

test.beforeEach(async ({ page, request }) => {
  await setScenario(request, "normal");
  await signIn(page);
});

test("the lead pick and four compact picks render with their facts", async ({ page }) => {
  const picks = page.getByTestId("picks-section");
  const lead = picks.getByTestId("lead-pick");
  await expect(lead).toContainText(LEAD_TITLE);
  await expect(lead).toContainText("Pick 01");
  await expect(lead).toContainText("peer-reviewed");
  await expect(lead).toContainText("ICLR 2027");
  await expect(lead).toContainText("31 cites");
  await expect(lead).toContainText("arXiv 2610.04412");
  await expect(lead).toContainText("Why it is here");
  await expect(lead).toContainText("peer-reviewed via venue: ICLR 2027");
  await expect(lead).toContainText("2.4k ★");
  await expect(lead).toContainText("2026-10-20");
  await expect(lead).toContainText("Agentic engineering");
  await expect(lead).toContainText("0.87");
  // A value the item never had prints n/a rather than an empty cell.
  await expect(lead).toContainText("Hugging Face");
  await expect(picks.getByTestId("compact-pick")).toHaveCount(4);
  // Seven cards share the same actions, so each one names the item it acts on.
  await expect(lead.getByRole("button", { name: `Not for me: ${LEAD_TITLE}` })).toBeVisible();
  await expect(lead.getByRole("link", { name: `Read the original: ${LEAD_TITLE}` })).toBeVisible();
  await expect(lead.getByRole("button", { name: `Ask about it: ${LEAD_TITLE}` })).toBeVisible();
  // A reader never sees a database enum: every tier and basis is printed as words.
  const text = await picks.innerText();
  for (const raw of [
    "peer_reviewed",
    "vetted_preprint",
    "lab_blog",
    "hf_upvotes",
    "lab_post",
    "lab_author",
    "agentic-engineering",
    "ai-forensics",
    "ml-architectures",
  ]) {
    expect(text).not.toContain(raw);
  }
  await expect(picks).toContainText("vetted preprint");
  await expect(picks).toContainText("lab post");
});

test("a checked summary shows quotes and labels interpretations", async ({ page }) => {
  const thirty = page.getByTestId("lead-pick").getByTestId("thirty-seconds");
  await expect(thirty).toContainText("In 30 seconds");
  await expect(thirty).toContainText("What they did");
  await expect(thirty).toContainText("The result");
  await expect(thirty).toContainText(
    "we replace the running summary with a replay buffer of raw tool observations",
  );
  await expect(thirty).toContainText("41% fewer tokens per task");
  await expect(thirty).toContainText("interpretation");
  // Only the two interpretation fields carry the label.
  await expect(thirty.getByTestId("interpretation-label")).toHaveCount(2);
});

test("a pending summary shows the excerpt and is replaced after a refresh", async ({ page, request }) => {
  await setScenario(request, "summary_pending");
  await page.reload();

  const lead = page.getByTestId("lead-pick");
  const thirty = lead.getByTestId("thirty-seconds");
  // The first list arrives before the background summary does: the excerpt is what is true now.
  await expect(thirty).toContainText("summary on its way");
  await expect(thirty).toContainText("We replace the running summary");
  // The section refreshes itself and the fake serves the checked list from then on.
  await expect(thirty).toContainText("41% fewer tokens per task", { timeout: 15_000 });
  await expect(thirty).not.toContainText("summary on its way");
});

test("dismissing removes the card and posts the event", async ({ page }) => {
  const picks = page.getByTestId("picks-section");
  const posted = page.waitForRequest(
    (request) => /\/api\/suggestions\/.+\/events$/.test(request.url()) && request.method() === "POST",
  );
  await picks.getByTestId("lead-pick").getByRole("button", { name: /^Not for me/ }).click();
  expect((await posted).postDataJSON()).toEqual({ event: "dismissed" });

  // The card goes at once, and the next pick takes the lead slot.
  await expect(picks.getByTestId("lead-pick")).toContainText(
    "Detecting Re-encoded Watermarks in Generated Images",
  );
  await expect(picks.getByTestId("compact-pick")).toHaveCount(3);
  await expect(picks).not.toContainText(LEAD_TITLE);
});

test("read the original opens the stored url in a new tab", async ({ page }) => {
  // The page never reaches the network in a test: the external article is fulfilled locally.
  await page.route("https://arxiv.org/**", (route) =>
    route.fulfill({ status: 200, contentType: "text/html", body: "the original" }),
  );
  const link = page.getByTestId("lead-pick").getByRole("link", { name: /Read the original/ });
  expect(await link.getAttribute("href")).toBe("https://arxiv.org/abs/2610.04412");
  expect(await link.getAttribute("target")).toBe("_blank");

  // A middle click opens the paper too, so it counts as an opening as well.
  const middle = page.waitForRequest(
    (request) => /\/api\/suggestions\/.+\/events$/.test(request.url()) && request.method() === "POST",
  );
  await link.click({ button: "middle" });
  expect((await middle).postDataJSON()).toEqual({ event: "opened" });

  const beacon = page.waitForRequest(
    (request) => /\/api\/suggestions\/.+\/events$/.test(request.url()) && request.method() === "POST",
  );
  const [popup] = await Promise.all([page.waitForEvent("popup"), link.click()]);
  await expect(popup).toHaveURL("https://arxiv.org/abs/2610.04412");
  expect((await beacon).postDataJSON()).toEqual({ event: "opened" });
});

test("ask about it opens the sheet with the item", async ({ page }) => {
  const picks = page.getByTestId("picks-section");
  await picks.getByTestId("lead-pick").getByRole("button", { name: /^Ask about it/ }).click();

  const sheet = page.getByRole("dialog");
  await expect(sheet).toBeVisible();
  // The item is pre-registered as S1 by the API, so the sheet says which item the question is about.
  await expect(sheet).toContainText(LEAD_TITLE);

  const sent = page.waitForRequest(
    (request) => request.url().endsWith("/api/ask") && request.method() === "POST",
  );
  await sheet.getByTestId("composer").fill("What did they measure?");
  await sheet.getByRole("button", { name: "Send the question" }).click();
  expect((await sent).postDataJSON()).toEqual({
    question: "What did they measure?",
    item_id: LEAD_ID,
  });
});

test("a dropped ask about it is retried about the same item", async ({ page, request }) => {
  await setScenario(request, "drop");
  await page
    .getByTestId("picks-section")
    .getByTestId("lead-pick")
    .getByRole("button", { name: /^Ask about it/ })
    .click();
  const sheet = page.getByRole("dialog");
  await sheet.getByTestId("composer").fill("What did they measure?");
  await sheet.getByRole("button", { name: "Send the question" }).click();
  await expect(sheet.getByTestId("interrupted")).toBeVisible();

  // The early thread event named an empty thread: the retry continues it and still carries the
  // item, so the answer is about the paper the reader asked about.
  await setScenario(request, "normal", { keepThreads: true });
  await sheet.getByRole("button", { name: "Ask it again" }).click();
  await expect(sheet.getByTestId("answer").locator("sup")).toHaveCount(1);
  const sent = await lastAsk(request);
  expect(sent.item_id).toBe(LEAD_ID);
  expect(typeof sent.thread_id).toBe("string");
});

test("rising shows the not yet independently vetted note", async ({ page }) => {
  const rising = page.getByTestId("rising-section");
  // Rising is a shelf inside This week, not a separate page.
  await expect(page.getByTestId("picks-section").getByTestId("rising-section")).toBeVisible();
  await expect(rising).toContainText("Rising, not yet independently vetted.");
  await expect(rising.getByTestId("rising-card")).toHaveCount(3);
  await expect(rising).toContainText("+380 upvotes in 1 day");
  await expect(rising).toContainText("Sparse Attention With a Learned Sketch");
  await expect(rising.getByRole("button", { name: /^Ask about it/ }).first()).toBeVisible();
});

test("a javascript link is never rendered as a link", async ({ page, request }) => {
  await setScenario(request, "javascript");
  await page.reload();

  const hrefs = await page
    .locator("a[href]")
    .evaluateAll((elements) => elements.map((element) => element.getAttribute("href") ?? ""));
  expect(hrefs.length).toBeGreaterThan(0);
  for (const href of hrefs) expect(href.toLowerCase()).not.toContain("javascript:");
  // The unsafe card keeps its title and loses only its link.
  await expect(page.getByTestId("picks-section")).toContainText(LEAD_TITLE);
  await expect(
    page.getByTestId("lead-pick").getByRole("link", { name: /Read the original/ }),
  ).toHaveCount(0);
});

test("an empty list says nothing passed the gate", async ({ page, request }) => {
  await setScenario(request, "empty");
  await page.reload();
  await expect(page.getByTestId("picks-section")).toContainText(
    "Nothing passed the gate in the last three weeks.",
  );
});

test("an unreachable API shows the error box", async ({ page, request }) => {
  await setScenario(request, "error");
  await page.reload();
  await expect(page.getByTestId("picks-error")).toContainText(
    "The API is unreachable right now. The ingestion is unaffected.",
  );
});

test("the phone layout is one column with touch-sized actions", async ({ page }, testInfo) => {
  test.skip(!testInfo.project.name.includes("iphone"), "checked on the iPhone project");

  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - window.innerWidth,
  );
  expect(overflow).toBeLessThanOrEqual(0);

  const lead = page.getByTestId("lead-pick");
  const title = await lead.getByRole("heading", { level: 3 }).boundingBox();
  const aside = await lead.getByTestId("lead-facts").boundingBox();
  // Section 8.5 of the design report: the aside drops below the title as a full-width block.
  expect(title).not.toBeNull();
  expect(aside).not.toBeNull();
  if (title && aside) {
    expect(aside.y).toBeGreaterThan(title.y + title.height);
    expect(aside.width).toBeGreaterThan(title.width * 0.9);
  }

  const actions = lead.locator(
    "a:has-text('Read the original'), button:has-text('Ask about it')",
  );
  await expect(actions).toHaveCount(2);
  for (let index = 0; index < 2; index += 1) {
    const target = await actions.nth(index).boundingBox();
    expect(target).not.toBeNull();
    if (target) expect(target.height).toBeGreaterThanOrEqual(44);
  }
});
