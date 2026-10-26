import { expect, test, type Locator, type Page } from "@playwright/test";

import { setScenario, signIn } from "./helpers";

// The corpus section and the item page, against the fake API only. The mockup's `.library` is the
// visual reference; these tests pin the behaviour the design text decides.

const LEAD_TITLE = "Replay Buffers Beat Summaries for Long-Running Coding Agents";
const LEAD_ID = "195e9246-327f-58e9-8cc3-44290a454cc6";
const NEWEST_TITLE = "Evaluating deep agents in production";
// The item whose stored links are a javascript: repo, a javascript: project page and a data: url.
const HOSTILE_ID = "75ef66c0-f5a3-5235-8c96-9abbad4b765a";
// An item that only reached Rising, so the draft of it must be labelled as early work.
const EARLY_ID = "36ddae68-0992-5624-87b0-2837bd1485b7";

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
  // The topic is a label, not the config file's slug.
  await expect(rows.filter({ hasText: LEAD_TITLE })).toContainText("Agentic engineering");

  // The footer sentence is the reader's cue for why this table is not the whole corpus.
  await expect(corpus).toContainText("Only admitted items are searchable.");
  await expect(corpus.getByRole("status")).toHaveText("Showing 10 of 12 · j k to move");
  await expect(corpus).toContainText("Load more");

  // A reader never sees a database enum or a config slug: every tier, rule and topic is words.
  const text = await corpus.innerText();
  for (const raw of [
    "peer_reviewed",
    "vetted_preprint",
    "lab_blog",
    "hf_upvotes",
    "lab_author",
    "agentic-engineering",
    "ai-forensics",
    "ml-architectures",
  ]) {
    expect(text).not.toContain(raw);
  }
});

test("load more appends the next page and keeps the reader's place", async ({ page }) => {
  const corpus = page.getByTestId("corpus-section");
  const rows = corpus.getByTestId("corpus-row");
  await expect(rows).toHaveCount(10);

  // Select a row that is not the first one, so a reset would be visible. The section has to be
  // interactive before a key is sent, and the keyboard keeps the pointer out of it: with the
  // pointer over the table, appended rows can move under it.
  await interactive(page, corpus);
  await page.keyboard.press("j");
  await page.keyboard.press("j");
  await expect(rows.nth(2)).toHaveAttribute("data-selected", "true");

  await corpus.getByRole("button", { name: "Load more items" }).click();
  await expect(rows).toHaveCount(12);
  await expect(corpus.getByRole("status")).toHaveText("Showing 12 of 12 · j k to move");
  // Selection stays where the reader left it, and focus lands on the first new row (the button
  // that had focus unmounts with the last page).
  await expect(rows.nth(2)).toHaveAttribute("data-selected", "true");
  await expect(rows.nth(10).getByRole("link")).toBeFocused();
  await expect(corpus.getByRole("button", { name: "Load more items" })).toHaveCount(0);
  await expect(corpus).toContainText("Routing tokens with a learned gate");
});

test("the selection follows a pointer that moves over the table", async ({ page }, testInfo) => {
  // A touch device has no pointer that hovers: this is the desktop behaviour.
  test.skip(testInfo.project.name.includes("iphone"), "no hovering pointer on a touch device");
  const corpus = page.getByTestId("corpus-section");
  const rows = corpus.getByTestId("corpus-row");
  await interactive(page, corpus);

  // A real move selects the row under the pointer. This one test cannot tell a move from a
  // synthesised event (a real move fires both); the resting-pointer test below is the guard that
  // a scroll under a still pointer never takes the selection. A reader can only point at a row
  // that is on screen, so each row is scrolled into view first.
  async function point_at(index: number) {
    const row = rows.nth(index);
    await row.scrollIntoViewIfNeeded();
    const box = await row.boundingBox();
    expect(box).not.toBeNull();
    if (box) await page.mouse.move(box.x + 20, box.y + box.height / 2);
    await expect(row).toHaveAttribute("data-selected", "true");
  }
  await point_at(3);
  await point_at(5);
  await expect(rows.nth(3)).not.toHaveAttribute("data-selected", "true");
});

test("a pointer resting on the table does not take the selection back when j scrolls", async ({
  page,
}, testInfo) => {
  test.skip(testInfo.project.name.includes("iphone"), "no hovering pointer on a touch device");
  // A short window, so the table scrolls as the selection walks down it.
  await page.setViewportSize({ width: 1440, height: 520 });
  const corpus = page.getByTestId("corpus-section");
  const rows = corpus.getByTestId("corpus-row");
  await interactive(page, corpus);
  await corpus.scrollIntoViewIfNeeded();

  // Rest the pointer on row 1 and leave it there: two moves, so a real position is recorded.
  const box = await rows.nth(1).boundingBox();
  expect(box).not.toBeNull();
  if (box) {
    await page.mouse.move(box.x + 30, box.y + box.height / 2);
    await page.mouse.move(box.x + 34, box.y + box.height / 2);
  }
  await expect(rows.nth(1)).toHaveAttribute("data-selected", "true");

  for (let step = 2; step <= 9; step += 1) {
    await page.keyboard.press("j");
    await expect(rows.nth(step)).toHaveAttribute("data-selected", "true");
    // WebKit delivers a synthetic mousemove a moment after a scroll; the selection must hold.
    await page.waitForTimeout(200);
    await expect(rows.nth(step)).toHaveAttribute("data-selected", "true");
  }
});

test("the corpus filters by tier and searches", async ({ page }) => {
  const corpus = page.getByTestId("corpus-section");
  const rows = corpus.getByTestId("corpus-row");
  const search = corpus.getByRole("searchbox", { name: "Search the corpus" });

  // The tier filters are plain toggles: one table, not one panel per tier.
  const all = corpus.getByRole("button", { name: "All" });
  const peer = corpus.getByRole("button", { name: "Peer reviewed" });
  await expect(all).toHaveAttribute("aria-pressed", "true");
  await peer.click();
  await expect(peer).toHaveAttribute("aria-pressed", "true");
  await expect(all).toHaveAttribute("aria-pressed", "false");
  await expect(rows).toHaveCount(3);
  await expect(corpus).toContainText("3 admitted matching");
  await expect(rows.filter({ hasText: LEAD_TITLE })).toHaveCount(1);
  // A lab post is not in the peer-reviewed tier, whatever its topic.
  await expect(rows.filter({ hasText: NEWEST_TITLE })).toHaveCount(0);

  // A search runs over the stored text with no model, in every search mode; the table narrows.
  await all.click();
  await expect(rows).toHaveCount(10);
  await search.fill("watermark");
  await expect(rows).toHaveCount(2);
  await expect(rows.filter({ hasText: "Detecting Re-encoded Watermarks" })).toHaveCount(1);
  await expect(rows.filter({ hasText: "Watermark removal, measured" })).toHaveCount(1);

  // A query that matches nothing says so, and asks for a narrower one.
  await search.fill("rotary embeddings drift");
  await expect(corpus.getByTestId("corpus-empty")).toContainText("Nothing matches");
  await expect(corpus.getByTestId("corpus-empty")).toContainText(
    "No admitted item matches. Try fewer words.",
  );
  await expect(rows).toHaveCount(0);
});

/** One filter click, waiting for the answer it starts: that answer resets the selection. */
async function filter(page: Page, button: Locator) {
  await Promise.all([
    page.waitForResponse((response) => response.url().includes("/api/corpus?")),
    button.click(),
  ]);
  await expect(button).toHaveAttribute("aria-pressed", "true");
}

// The section's keys are a client listener: pressing one before hydration is lost, so a test that
// presses a key straight after sign-in first proves the section answers a click. The helper waits
// for both answers, because a filter response resets the selection and a key sent before it lands
// would be undone by it.
async function interactive(page: Page, corpus: Locator) {
  await filter(page, corpus.getByRole("button", { name: "Vetted preprints" }));
  await filter(page, corpus.getByRole("button", { name: "All" }));
  // The section's keys belong to the page, not to a control: end on the heading, whose click
  // leaves focus on the body in every engine.
  await corpus.getByRole("heading", { name: "The corpus" }).click();
}

/** Which row carries the selection, as an index, so a test never assumes a particular one. */
async function selectedRow(rows: import("@playwright/test").Locator): Promise<number> {
  return rows.evaluateAll((nodes) =>
    nodes.findIndex((node) => node.getAttribute("data-selected") === "true"),
  );
}

test("slash focuses the search and j k move the selection", async ({ page }) => {
  const corpus = page.getByTestId("corpus-section");
  const rows = corpus.getByTestId("corpus-row");
  const search = corpus.getByRole("searchbox", { name: "Search the corpus" });
  await interactive(page, corpus);

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

test("Enter stays with the control that has focus", async ({ page }) => {
  const corpus = page.getByTestId("corpus-section");
  const rows = corpus.getByTestId("corpus-row");
  await interactive(page, corpus);

  // A tier toggle: Enter presses the toggle, not the selected row.
  await corpus.getByRole("button", { name: "Vetted preprints" }).focus();
  await page.keyboard.press("Enter");
  await expect(corpus.getByRole("button", { name: "Vetted preprints" })).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  expect(page.url()).not.toContain("/items/");
  await corpus.getByRole("button", { name: "All" }).click();
  await expect(rows).toHaveCount(10);

  // Load more: Enter loads the next page and stays on the page it is on.
  await corpus.getByRole("button", { name: "Load more items" }).focus();
  await page.keyboard.press("Enter");
  await expect(rows).toHaveCount(12);
  expect(page.url()).not.toContain("/items/");

  // A letter key is not the section's while a control has focus. Before the masthead link below,
  // which navigates and lets the browser move focus to the fragment.
  await corpus.getByRole("button", { name: "Peer reviewed" }).focus();
  await expect(corpus.getByRole("button", { name: "Peer reviewed" })).toBeFocused();
  const selectedBefore = await selectedRow(rows);
  await page.keyboard.press("j");
  await page.keyboard.press("k");
  expect(await selectedRow(rows)).toBe(selectedBefore);
  expect(page.url()).not.toContain("/items/");

  // A masthead link: Enter follows the link to its section.
  await page.getByRole("navigation").getByRole("link", { name: "Runs" }).focus();
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(/#runs$/);
  await expect(page).not.toHaveURL(/\/items\//);

  // The Ask sheet: Enter in its box stays in the box.
  await page.getByRole("button", { name: /^Ask the corpus/ }).click();
  const sheet = page.getByRole("dialog");
  await expect(sheet).toBeVisible();
  const question = sheet.getByRole("textbox").first();
  await question.focus();
  const before = page.url();
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(before);
  await expect(sheet).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(sheet).toHaveCount(0);

  // The draft dialog: Enter generates, and never navigates away from the page.
  await page.goto("/");
  const lead = page.getByTestId("lead-pick");
  await lead.getByRole("button", { name: new RegExp("^LinkedIn post preview:") }).click();
  const dialog = page.getByRole("dialog");
  await dialog.getByRole("button", { name: "Generate draft" }).focus();
  await page.keyboard.press("Enter");
  await expect(dialog.getByTestId("pov-saved")).toBeVisible();
  await expect(page).toHaveURL(/\/$/);
});

test("every corpus row opens an item page of its own", async ({ page }) => {
  const corpus = page.getByTestId("corpus-section");
  // The first row is not one of the hand-written fixtures: the fake API still answers for it.
  const first = corpus.getByTestId("corpus-row").first();
  const title = (await first.getByRole("link").innerText()).trim();
  await first.getByRole("link").click();
  await expect(page).toHaveURL(/\/items\//);
  await expect(page.getByTestId("item-page").getByRole("heading", { level: 1 })).toHaveText(title);
});

test("an item page shows its history and signals", async ({ page }) => {
  await page.goto(`/items/${LEAD_ID}`);
  const item = page.getByTestId("item-page");
  await expect(item.getByRole("heading", { level: 1 })).toHaveText(LEAD_TITLE);
  await expect(item).toContainText("peer-reviewed");
  await expect(item).toContainText("peer-reviewed via venue: ICLR 2027");
  await expect(item).toContainText("ICLR 2027");
  // The topic is the label, never the slug.
  await expect(item).toContainText("Agentic engineering");
  expect(await item.innerText()).not.toContain("agentic-engineering");

  // The facts, with the item's own numbers: the column is never below the newest reading.
  await expect(item).toContainText("Ada Lovelace, Grace Hopper");
  await expect(item).toContainText("31");
  await expect(item).toContainText("38 \u2191");
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

test("a slow answer to an older query never overwrites the table", async ({ page }) => {
  const corpus = page.getByTestId("corpus-section");
  const rows = corpus.getByTestId("corpus-row");
  const search = corpus.getByRole("searchbox", { name: "Search the corpus" });

  // The answer to "zzz" (no matches) is held back so it lands after the answer to "watermark".
  // Both waits below are on the network, so the test does not depend on a timer under load.
  let held = 0;
  await page.route("**/api/corpus?*", async (route) => {
    if (new URL(route.request().url()).searchParams.get("q") === "zzz") {
      held += 1;
      await new Promise((resolve) => setTimeout(resolve, 1500));
    }
    await route.continue();
  });
  const asked = page.waitForRequest(
    (request) => new URL(request.url()).searchParams.get("q") === "zzz",
  );
  const answered = page.waitForResponse(
    (response) => new URL(response.url()).searchParams.get("q") === "zzz",
  );

  await search.fill("zzz");
  // The held request is in flight, so the next query really races it.
  await asked;
  await search.fill("watermark");
  await expect(rows).toHaveCount(2);
  await expect(corpus.getByRole("status")).toHaveText("Showing 2 of 2 · j k to move");

  // The older answer lands here; a table that took it would show "zzz" with no rows.
  await answered;
  expect(held).toBeGreaterThan(0);
  await expect(rows).toHaveCount(2);
  await expect(corpus.getByRole("status")).toHaveText("Showing 2 of 2 · j k to move");
});

test("a load more that lands after the reader moved on keeps their focus", async ({ page }) => {
  const corpus = page.getByTestId("corpus-section");
  const rows = corpus.getByTestId("corpus-row");
  const search = corpus.getByRole("searchbox", { name: "Search the corpus" });

  // Hold the next page back, so the reader has time to move to the search box.
  await page.route("**/api/corpus?*", async (route) => {
    if (new URL(route.request().url()).searchParams.get("cursor")) {
      await new Promise((resolve) => setTimeout(resolve, 1200));
    }
    await route.continue();
  });

  await corpus.getByRole("button", { name: "Load more items" }).click();
  await search.click();
  await expect(rows).toHaveCount(12);
  // The new row mounts while the caret is in the box: the reader keeps it.
  await expect(search).toBeFocused();
});

test("a stored link that is not http renders nothing at all", async ({ page }) => {
  await page.goto(`/items/${HOSTILE_ID}`);
  const item = page.getByTestId("item-page");
  await expect(item.getByRole("heading", { level: 1 })).toContainText("must never become links");
  // safeHref refuses a javascript: or data: url, and the page prints neither a link nor its text.
  await expect(item.getByRole("link", { name: "Code" })).toHaveCount(0);
  await expect(item.getByRole("link", { name: "Project page" })).toHaveCount(0);
  await expect(item.getByRole("link", { name: "Read the original" })).toHaveCount(0);
  const text = await item.innerText();
  expect(text).not.toContain("javascript:");
  expect(text).not.toContain("data:text/html");
});

test("the item page previews a LinkedIn draft", async ({ page }) => {
  await page.goto(`/items/${LEAD_ID}`);
  await page.getByRole("button", { name: `LinkedIn post preview: ${LEAD_TITLE}` }).click();
  const dialog = page.getByRole("dialog");
  await expect(dialog).toBeVisible();
  await expect(dialog).toContainText(LEAD_TITLE);
  await expect(dialog.getByRole("button", { name: "Generate draft" })).toBeVisible();
});

test("an early item warns before its draft is generated", async ({ page }) => {
  await page.goto(`/items/${EARLY_ID}`);
  await page.getByRole("button", { name: /^LinkedIn post preview:/ }).click();
  // The same flag the API sets on the draft's own notice, so the reader is warned before a call.
  await expect(page.getByRole("dialog")).toContainText("Early work: the draft says so.");
});

test("an item that is not admitted says nothing here", async ({ page }) => {
  // A well-formed id the API does not serve reads as "nothing here", not as a server error.
  await page.goto("/items/00000000-0000-4000-8000-000000000000");
  await expect(page.getByRole("heading", { name: "Nothing here" })).toBeVisible();
  // A mistyped id is the same empty state, never a server error.
  await page.goto("/items/not-a-uuid");
  await expect(page.getByRole("heading", { name: "Nothing here" })).toBeVisible();
});

test("no masthead section is current on an item page", async ({ page }) => {
  await page.goto(`/items/${LEAD_ID}`);
  await expect(page.getByRole("navigation").locator("[aria-current]")).toHaveCount(0);
  // On the home page the corpus anchor makes Corpus the current section.
  await page.goto("/#corpus");
  const current = page.getByRole("navigation").locator("[aria-current='true']");
  await expect(current).toHaveCount(1);
  await expect(current).toHaveText("Corpus");
});

test("the masthead marks the section the reader clicked", async ({ page }) => {
  const nav = page.getByRole("navigation");
  // An in-page click is a client navigation: no hashchange fires, so the mark has to follow it.
  await nav.getByRole("link", { name: "Corpus" }).click();
  await expect(page).toHaveURL(/#corpus$/);
  await expect(nav.locator("[aria-current='true']")).toHaveText("Corpus");

  await nav.getByRole("link", { name: "Runs" }).click();
  await expect(page).toHaveURL(/#runs$/);
  await expect(nav.locator("[aria-current='true']")).toHaveText("Runs");

  // The today strip's link is an in-page anchor too, and it means the same section.
  await page.getByTestId("today-strip").getByRole("link").click();
  await expect(page).toHaveURL(/#runs$/);
  await expect(nav.locator("[aria-current='true']")).toHaveText("Runs");
});

test("a click that opens another tab leaves the mark where it is", async ({
  page,
  context,
}, testInfo) => {
  test.skip(testInfo.project.name.includes("iphone"), "a touch context has no modifier click");
  const nav = page.getByRole("navigation");
  await nav.getByRole("link", { name: "Corpus" }).click();
  await expect(nav.locator("[aria-current='true']")).toHaveText("Corpus");
  await expect(page).toHaveURL(/#corpus$/);
  const url = page.url();

  // A cmd-click opens the link in a background tab: this page did not navigate, so its mark stays.
  const [opened] = await Promise.all([
    context.waitForEvent("page"),
    nav.getByRole("link", { name: "Rising" }).click({ modifiers: ["ControlOrMeta"] }),
  ]);
  await opened.close();
  expect(page.url()).toBe(url);
  await expect(nav.locator("[aria-current='true']")).toHaveText("Corpus");
});

test("an in-page link that names no section marks nothing", async ({ page, request }) => {
  await setScenario(request, "citations");
  await page.goto("/#corpus");
  const nav = page.getByRole("navigation");
  await expect(nav.locator("[aria-current='true']")).toHaveText("Corpus");

  await page.getByRole("button", { name: /Ask the corpus/ }).click();
  const sheet = page.getByRole("dialog");
  await sheet
    .getByRole("textbox", { name: "Your question" })
    .fill("What do agents measure and remember?");
  await sheet.getByRole("button", { name: "Send the question" }).click();
  await expect(sheet.getByTestId("answer").locator("sup")).toHaveCount(2);

  // A citation is "#fn-S1": a place inside the answer, not a section, so no link claims to be it.
  await sheet.getByTestId("answer").locator("sup a").first().click();
  await expect(page).toHaveURL(/#fn-S\d$/);
  await expect(nav.locator("[aria-current]")).toHaveCount(0);
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

  // The tier toggles are touch-sized like every other control on the phone.
  for (const label of ["All", "Vetted preprints"]) {
    const box = await corpus.getByRole("button", { name: label }).boundingBox();
    expect(box).not.toBeNull();
    if (box) expect(box.height).toBeGreaterThanOrEqual(44);
  }
});
