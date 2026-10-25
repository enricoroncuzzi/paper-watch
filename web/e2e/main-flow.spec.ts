import { expect, test, type Page } from "@playwright/test";

import { setScenario, signIn } from "./helpers";

// The main flow: one continuous session that walks the product the way the owner does, from the
// sign-in page to a saved draft, a conversation with a follow-up and the runs. The per-slice specs
// pin each section in depth; this one catches a break in the path between them. It runs on the
// desktop (Chromium) and the phone (WebKit, iPhone 15) projects only, which are the two the owner
// uses.

const LEAD_TITLE = "Replay Buffers Beat Summaries for Long-Running Coding Agents";
const POV = "Replaying tool results is the part I want to test on our own harness.";
// The summary sentence the lead pick carries, and the body the fake API saves for a new draft.
const LEAD_CONTEXT = "Swapped the summarised memory in a coding harness for a replay buffer";
const CREATED_BODY = "It cut tokens by 41% on 40 tasks, which is worth a look before adopting it.";
const QUESTION = "What do coding agents measure?";
const FOLLOW_UP = "And how was it measured?";

async function openSheet(page: Page): Promise<void> {
  await page.getByRole("button", { name: /Ask the corpus/ }).click();
  await expect(page.getByRole("dialog")).toBeVisible();
}

async function send(page: Page, question: string): Promise<void> {
  const sheet = page.getByRole("dialog");
  await sheet.getByTestId("composer").fill(question);
  await sheet.getByRole("button", { name: "Send the question" }).click();
  await expect(sheet.getByTestId("asked-question")).toHaveText(question);
}

test.beforeEach(async ({ request }) => {
  await setScenario(request, "normal");
});

test("the main flow", async ({ page }) => {
  // 1. Sign in.
  await signIn(page);

  // 2. The today strip says what the last run did.
  const strip = page.getByTestId("today-strip");
  await expect(strip).toContainText("healthy");
  await expect(page.getByTestId("run-day")).toHaveText("Today");
  await expect(strip).toContainText("9 stored");
  await expect(strip).toContainText("41 fetched");

  // 3. The lead pick, and the four answers of its 30-second summary.
  const lead = page.getByTestId("lead-pick");
  await expect(lead).toContainText(LEAD_TITLE);
  const thirty = lead.getByTestId("thirty-seconds");
  await expect(thirty).toContainText("In 30 seconds");
  await expect(thirty).toContainText("What they did");
  await expect(thirty).toContainText("The result");
  await expect(thirty).toContainText(
    "we replace the running summary with a replay buffer of raw tool observations",
  );
  await expect(thirty).toContainText("41% fewer tokens per task");

  // 4. Open the pick: the dialog carries the item and the same summary context.
  await lead.getByRole("button", { name: `LinkedIn post preview: ${LEAD_TITLE}` }).click();
  const dialog = page.getByTestId("pov-dialog");
  await expect(dialog).toBeVisible();
  await expect(dialog).toContainText(LEAD_TITLE);
  await expect(dialog).toContainText(LEAD_CONTEXT);

  // 5. Generate a draft with a point of view. There is no save step: generating saves it.
  await dialog.getByTestId("pov-input").fill(POV);
  await dialog.getByRole("button", { name: "Generate draft" }).click();
  await expect(dialog.getByTestId("pov-preview")).toBeVisible();
  await expect(dialog.getByTestId("pov-saved")).toContainText("Saved to Drafts");
  await expect(dialog.getByTestId("draft-check-ok")).toContainText("All checks passed");
  await expect(dialog).toContainText(CREATED_BODY);
  // The refresh behind the dialog adds the draft above the two fixtures.
  await expect(page.getByTestId("drafts-section").getByTestId("draft-card")).toHaveCount(3);
  await dialog.getByRole("button", { name: "Close", exact: true }).click();
  await expect(dialog).toBeHidden();

  // 6. Find it in Drafts.
  await page.getByRole("navigation", { name: "Sections" }).getByRole("link", { name: "Drafts" }).click();
  await expect(page).toHaveURL(/\/#drafts$/);
  const drafts = page.getByTestId("drafts-section");
  await expect(drafts).toContainText("3 saved");
  const newest = drafts.getByTestId("draft-card").first();
  await expect(newest).toContainText(LEAD_TITLE);
  await expect(newest).toContainText(CREATED_BODY);
  // A saved-now chip, not one of the fixtures' dates, proves this is the draft just generated.
  await expect(newest).toContainText(/saved today \d{2}:\d{2}/);

  // 7. Ask a question and a follow-up, in the one conversation the sheet keeps.
  await openSheet(page);
  const sheet = page.getByRole("dialog");
  await send(page, QUESTION);
  await expect(sheet.getByTestId("answer")).toContainText(
    "Coding agents are measured on real repository bugs",
  );
  await expect(sheet.getByTestId("sources")).toContainText("SWE-bench Verified");
  await expect(sheet.getByTestId("turns-chip")).toHaveText("1 turn");
  // The composer invites a follow-up, and the sheet keeps the thread it belongs to.
  await expect(sheet.getByTestId("composer")).toHaveAttribute("placeholder", "Ask a follow-up...");

  await send(page, FOLLOW_UP);
  await expect(sheet.getByTestId("turn-1")).toContainText(QUESTION);
  await expect(sheet.getByTestId("answer").locator("sup")).toHaveCount(1);
  await expect(sheet.getByTestId("turns-chip")).toHaveText("2 turns");
  await sheet.getByRole("button", { name: "Close the sheet" }).click();
  await expect(sheet).toBeHidden();

  // 8. Open Runs: the last run's counts and the rows that open each run.
  await page.getByRole("navigation", { name: "Sections" }).getByRole("link", { name: "Runs" }).click();
  await expect(page).toHaveURL(/\/#runs$/);
  const runs = page.getByTestId("runs-section");
  await expect(runs.getByRole("heading", { name: "Runs" })).toBeVisible();
  await expect(runs).toContainText("tokens");
  const row = runs.locator("a[href^='/runs/']").first();
  await expect(row).toBeVisible();
  await expect(row).toHaveAttribute("href", /^\/runs\/[0-9a-f-]{36}$/);
});
