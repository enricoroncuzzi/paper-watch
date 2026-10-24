import { expect, test } from "@playwright/test";

import { setScenario, signIn } from "./helpers";

// The corpus section and the item page, against the fake API only. The mockup's `.library` is the
// visual reference; these tests pin the behaviour the design text decides.

const LEAD_TITLE = "Replay Buffers Beat Summaries for Long-Running Coding Agents";
const LEAD_ID = "195e9246-327f-58e9-8cc3-44290a454cc6";
const NEWEST_TITLE = "Evaluating deep agents in production";

test.beforeEach(async ({ page, request }) => {
  await setScenario(request, "normal");
  await signIn(page);
});

test("the corpus lists admitted items newest first with their facts", async ({ page }) => {
  const corpus = page.getByTestId("corpus-section");
  await expect(corpus.getByRole("heading", { name: "The corpus" })).toBeVisible();
  await expect(corpus).toContainText("12 admitted");

  const rows = corpus.getByTestId("corpus-row");
  // Ten rows is the page's size: the fixture holds twelve, so one more page is offered.
  await expect(rows).toHaveCount(10);
  await expect(rows.first()).toContainText(NEWEST_TITLE);
  await expect(rows.filter({ hasText: LEAD_TITLE })).toHaveCount(1);
  await expect(rows.filter({ hasText: LEAD_TITLE })).toContainText("peer-reviewed");
  await expect(rows.filter({ hasText: LEAD_TITLE })).toContainText("31 cites");
  await expect(rows.filter({ hasText: LEAD_TITLE })).toContainText("ICLR 2027");
  await expect(rows.filter({ hasText: LEAD_TITLE })).toContainText("2026-10-20");
  await expect(rows.filter({ hasText: LEAD_TITLE })).toContainText("agentic-engineering");

  // The footer sentence is the reader's cue for why this table is not the whole corpus.
  await expect(corpus).toContainText("Only admitted items are searchable.");
  await expect(corpus).toContainText("10 of 12 · j k to move");
  await expect(corpus).toContainText("Load more");

  // A reader never sees a database enum: every tier and rule is printed as words.
  const text = await corpus.innerText();
  for (const raw of ["peer_reviewed", "vetted_preprint", "lab_blog", "hf_upvotes", "lab_author"]) {
    expect(text).not.toContain(raw);
  }
});

test("load more appends the next page", async ({ page }) => {
  const corpus = page.getByTestId("corpus-section");
  const rows = corpus.getByTestId("corpus-row");
  await expect(rows).toHaveCount(10);
  await corpus.getByRole("button", { name: "Load more items" }).click();
  await expect(rows).toHaveCount(12);
  await expect(corpus).toContainText("12 of 12 · j k to move");
  // Nothing follows the last page, so the control goes away rather than fetching nothing.
  await expect(corpus.getByRole("button", { name: "Load more items" })).toHaveCount(0);
  await expect(corpus).toContainText("Routing tokens with a learned gate");
});

test("the corpus filters by tier and searches", async ({ page }) => {
  const corpus = page.getByTestId("corpus-section");
  const rows = corpus.getByTestId("corpus-row");
  const search = corpus.getByRole("searchbox", { name: "Search the corpus" });

  // The tier tabs are the mockup's four, and each one narrows the table.
  await expect(corpus.getByRole("tab", { name: "All" })).toHaveAttribute("aria-selected", "true");
  await corpus.getByRole("tab", { name: "Peer reviewed" }).click();
  await expect(corpus.getByRole("tab", { name: "Peer reviewed" })).toHaveAttribute(
    "aria-selected",
    "true",
  );
  await expect(rows).toHaveCount(3);
  await expect(corpus).toContainText("3 admitted matching");
  await expect(rows.filter({ hasText: LEAD_TITLE })).toHaveCount(1);
  // A lab post is not in the peer-reviewed tier, whatever its topic.
  await expect(rows.filter({ hasText: NEWEST_TITLE })).toHaveCount(0);

  // A search runs over the stored text with no model, in every search mode; the table narrows.
  await corpus.getByRole("tab", { name: "All" }).click();
  await expect(rows).toHaveCount(10);
  await search.fill("watermark");
  await expect(rows).toHaveCount(2);
  await expect(rows.filter({ hasText: "Detecting Re-encoded Watermarks" })).toHaveCount(1);
  await expect(rows.filter({ hasText: "Watermark removal, measured" })).toHaveCount(1);

  // A query that matches nothing says so, and names the query it could not answer.
  await search.fill("rotary embeddings drift");
  await expect(corpus.getByTestId("corpus-empty")).toContainText("Nothing matches");
  await expect(rows).toHaveCount(0);
});

test("slash focuses the search and j k move the selection", async ({ page }) => {
  const corpus = page.getByTestId("corpus-section");
  const rows = corpus.getByTestId("corpus-row");
  const search = corpus.getByRole("searchbox", { name: "Search the corpus" });

  await page.keyboard.press("/");
  await expect(search).toBeFocused();

  // Typing a letter the section also listens for is just typing: the selection does not move.
  await search.fill("jj");
  await expect(rows.first()).toHaveAttribute("data-selected", "true");
  await search.fill("");

  // Outside a field the same keys move the selection, and Enter opens the selected item.
  await corpus.getByRole("heading", { name: "The corpus" }).click();
  await page.keyboard.press("j");
  await expect(rows.nth(1)).toHaveAttribute("data-selected", "true");
  await expect(rows.first()).not.toHaveAttribute("data-selected", "true");
  await page.keyboard.press("k");
  await expect(rows.first()).toHaveAttribute("data-selected", "true");

  // "k" at the top stays at the top instead of wrapping to the end.
  await page.keyboard.press("k");
  await expect(rows.first()).toHaveAttribute("data-selected", "true");

  await page.keyboard.press("j");
  await page.keyboard.press("j");
  await expect(rows.nth(2)).toHaveAttribute("data-selected", "true");
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(new RegExp(`/items/${LEAD_ID}$`));
});

test("an item page shows its history and signals", async ({ page }) => {
  await page.goto(`/items/${LEAD_ID}`);
  const item = page.getByTestId("item-page");
  await expect(item.getByRole("heading", { level: 1 })).toHaveText(LEAD_TITLE);
  await expect(item).toContainText("peer-reviewed");
  await expect(item).toContainText("peer-reviewed via venue: ICLR 2027");
  await expect(item).toContainText("ICLR 2027");
  await expect(item).toContainText("agentic-engineering");

  // The facts, with n/a where the corpus has no value rather than an empty cell.
  await expect(item).toContainText("Ada Lovelace, Grace Hopper");
  await expect(item).toContainText("2.4k \u2605");
  await expect(item.getByRole("link", { name: "Code" })).toHaveAttribute(
    "href",
    "https://github.com/example/replay-buffers",
  );

  // The tier timeline reads as sentences, never as enum values.
  const history = item.getByTestId("tier-history");
  await expect(history.getByRole("listitem")).toHaveCount(2);
  await expect(history).toContainText("first seen to vetted preprint");
  await expect(history).toContainText("vetted preprint to peer-reviewed");
  await expect(history).toContainText("upvotes rule");
  const historyText = await history.innerText();
  expect(historyText).not.toContain("vetted_preprint");
  expect(historyText).not.toContain("peer_reviewed");

  // The signal series is a small table by date, oldest first.
  const signals = item.getByTestId("signal-series");
  await expect(signals.getByRole("row")).toHaveCount(4);
  await expect(signals).toContainText("2026-10-20");
  await expect(signals).toContainText("1.8k");
  await expect(signals).toContainText("2.4k");

  // The cached summary travels with its quotes, like a pick's.
  await expect(item).toContainText("In 30 seconds");
  await expect(item).toContainText(
    "We replace the running summary with a replay buffer of raw tool observations",
  );

  // The stored text, at the reading measure.
  await expect(item).toContainText("Replaying exact tool results cut tokens per task by 41%");

  await expect(item.getByRole("link", { name: "Read the original" })).toHaveAttribute(
    "href",
    "https://arxiv.org/abs/2610.04412",
  );
  await expect(item.getByRole("button", { name: `Ask about it: ${LEAD_TITLE}` })).toBeVisible();
  await expect(
    item.getByRole("button", { name: `LinkedIn post preview: ${LEAD_TITLE}` }),
  ).toBeVisible();
  await expect(item).toContainText("Already in Drafts: 1");
});

test("the item page previews a LinkedIn draft", async ({ page }) => {
  await page.goto(`/items/${LEAD_ID}`);
  await page.getByRole("button", { name: `LinkedIn post preview: ${LEAD_TITLE}` }).click();
  const dialog = page.getByRole("dialog");
  await expect(dialog).toBeVisible();
  await expect(dialog).toContainText(LEAD_TITLE);
  await expect(dialog.getByRole("button", { name: "Generate draft" })).toBeVisible();
});

test("an item that is not admitted says nothing here", async ({ page }) => {
  // A well-formed id the API does not serve reads as "nothing here", not as a server error.
  await page.goto("/items/00000000-0000-4000-8000-000000000000");
  await expect(page.getByRole("heading", { name: "Nothing here" })).toBeVisible();
  // A mistyped id is the same empty state, never a server error.
  await page.goto("/items/not-a-uuid");
  await expect(page.getByRole("heading", { name: "Nothing here" })).toBeVisible();
});

test("the phone layout keeps the table inside the screen", async ({ page, request }, testInfo) => {
  test.skip(!testInfo.project.name.includes("iphone"), "checked on the iPhone project");
  await setScenario(request, "normal");
  await page.reload();

  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - window.innerWidth,
  );
  expect(overflow).toBeLessThanOrEqual(0);

  const corpus = page.getByTestId("corpus-section");
  // The two narrow columns are the ones a phone drops; the title, tier and actions stay.
  await expect(corpus.getByRole("columnheader", { name: "Signals" })).toBeHidden();
  await expect(corpus.getByRole("columnheader", { name: "Source" })).toBeHidden();
  const first = corpus.getByTestId("corpus-row").first();
  await expect(first).toContainText(NEWEST_TITLE);
  const tab = corpus.getByRole("tab", { name: "Vetted preprints" });
  const box = await tab.boundingBox();
  expect(box).not.toBeNull();
});
