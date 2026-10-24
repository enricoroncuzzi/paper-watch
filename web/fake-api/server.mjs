// The fake API the page is built and tested against. Dependency-free and local: the e2e suite and
// local development never reach the real server, hold a key or call a model.
//
// Every fixture is validated against the API's own pydantic models by
// tests/unit/test_web_fake_api_fixtures.py, so this server cannot drift from the contract.
import { randomUUID } from "node:crypto";
import { createServer } from "node:http";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { fakeRunFinish, fakeRunStart } from "./dates.mjs";

const ROOT = dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.FAKE_API_PORT ?? 8787);
const TOKEN = "e2e";

// The scenario a test picks with POST /__scenario. Tests run one worker, so one global is safe.
let scenario = "normal";
// How many suggestion lists this scenario has served: the summary arrives after the first one.
let suggestionRequests = 0;
// The drafts this fake serves. POST /drafts appends, regenerate prepends, and a scenario switch
// resets the list, so every test starts from the same two fixtures.
let drafts = [];
let draftedItemIds = new Set();

/** The clipboard text, the same rule as `src/lib/linkedin-text.ts` and `draft_text` in Python. */
function linkedinText(draft) {
  const tags = draft.hashtags.map((tag) => `#${tag}`).join(" ");
  const parts = [draft.hook, "", draft.body];
  if (tags) parts.push("", tags);
  if (draft.source_link) parts.push("", draft.source_link);
  return parts.join("\n").trim();
}

// The corpus the fake serves: the fixture's rows, in the API's own order (newest published
// first), filtered and sliced the way the real route does. Cursors stay opaque to the page.
function corpusRows() {
  return [...fixture("corpus").items].sort((a, b) =>
    a.published_at === b.published_at
      ? a.id < b.id
        ? 1
        : -1
      : a.published_at < b.published_at
        ? 1
        : -1,
  );
}

function corpusOffset(value) {
  if (!value) return 0;
  try {
    const parsed = JSON.parse(Buffer.from(value, "base64url").toString("utf8"));
    const offset = Number(parsed.o);
    return Number.isInteger(offset) && offset > 0 ? offset : 0;
  } catch {
    return 0;
  }
}

function corpusCursor(offset) {
  return Buffer.from(JSON.stringify({ k: "newest", o: offset }), "utf8").toString("base64url");
}

function resetDrafts() {
  drafts = fixture("drafts").drafts.map((draft) => ({ ...draft }));
  draftedItemIds = new Set();
}

resetDrafts();

// The stored conversations. POST /ask appends a finished turn and serves the `thread` event; the
// two read routes serve the Recent menu and the re-fetch after a dropped stream. A scenario switch
// reseeds them unless it asks to keep them, so one test can ask, then change the stream behaviour.
let threads = [];
let lastAsk = null;

function resetThreads() {
  const seeded = fixture("thread");
  const now = Date.now();
  threads = [
    {
      ...seeded,
      // The fixture's dates are fixed; the page labels them in Rome time, so they are relative.
      turns: seeded.turns.map((turn, index) => ({
        ...turn,
        created_at: new Date(now - (seeded.turns.length - index) * 120_000).toISOString(),
      })),
      created_at: new Date(now - 600_000).toISOString(),
      updated_at: new Date(now - 60_000).toISOString(),
    },
  ];
}

resetThreads();

/** Append one finished answer to its thread, creating the thread for a first question. */
function saveTurn(ask, answer) {
  const question = String(ask.question ?? "").trim();
  const now = new Date().toISOString();
  let thread = ask.thread_id ? threads.find((one) => one.id === ask.thread_id) : undefined;
  if (!thread) {
    thread = { id: randomUUID(), title: question.slice(0, 80), turns: [], created_at: now };
    threads.push(thread);
  }
  thread.turns.push({
    n: thread.turns.length + 1,
    question,
    answer,
    search_mode: answer.search_mode ?? "hybrid",
    created_at: now,
  });
  thread.updated_at = now;
  return thread;
}

const SCENARIOS = new Set([
  "normal",
  "empty",
  "error",
  "unhealthy",
  "followup",
  "text_only",
  "daily_cap",
  "citations",
  "flagged",
  "refused",
  "yesterday",
  "slow",
  "drop",
  "summary_pending",
  "javascript",
  "drafts_empty",
  "draft_flags",
  "draft_refused",
  "draft_cap",
  "draft_slow",
  "drafts_ellipsis",
  "draft_detail_failed",
  "thread_drop",
  "thread_slow",
]);

function fixture(name) {
  return JSON.parse(readFileSync(join(ROOT, "fixtures", `${name}.json`), "utf8"));
}

function stream(name) {
  return JSON.parse(readFileSync(join(ROOT, "streams", `${name}.json`), "utf8"));
}

// The gate's declared refusal bodies, kept in a file so the Python test can validate them against
// the API's own Refusal and RateLimited models.
const REFUSALS = JSON.parse(readFileSync(join(ROOT, "refusals.json"), "utf8"));

function refused(response, status) {
  const body = REFUSALS[String(status)];
  const headers = status === 429 ? { "retry-after": String(body.retry_after_s) } : {};
  json(response, status, body, headers);
}

function json(response, status, body, headers = {}) {
  const payload = JSON.stringify(body);
  response.writeHead(status, {
    "content-type": "application/json",
    "content-length": Buffer.byteLength(payload),
    "cache-control": "no-store",
    ...headers,
  });
  response.end(payload);
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// The real API always serves recent runs; the fake rewrites its fixtures' dates relative to now so
// the strip's Today/Yesterday label and every screenshot match what the captain sees.
function runAt(run, daysAgo) {
  const now = Date.now();
  // Never before midnight Rome for its day: the strip labels by the Rome calendar date, so a run
  // dated "three hours ago" would read Yesterday between 00:00 and 03:00.
  const started = new Date(fakeRunStart(daysAgo, now));
  const finished = new Date(fakeRunFinish(started.getTime(), now));
  const dated = {
    ...run,
    started_at: started.toISOString(),
    finished_at: finished.toISOString(),
  };
  if (run.followup) {
    dated.followup = {
      ...run.followup,
      next_at: new Date(Date.now() + 2 * 60 * 60 * 1000).toISOString(),
    };
  }
  return dated;
}

function todayFor(name) {
  if (name === "unhealthy") return dated(fixture("today-unhealthy"), 0);
  if (name === "followup") return dated(fixture("today-followup"), 0);
  if (name === "text_only") return dated(fixture("today-text-only"), 0);
  if (name === "yesterday") return dated(fixture("today"), 1);
  if (name === "empty") {
    return { run: null, search: { mode: "hybrid", reason: "ok", since: null } };
  }
  return dated(fixture("today"), 0);
}

function dated(today, daysAgo) {
  return { ...today, run: runAt(today.run, daysAgo) };
}

function runsFor(name) {
  // The rows are dated one day apart, newest first, so the list reads like a real week.
  const base = fixture("runs").runs.map((run, index) => runAt(run, index));
  // A scenario about today's run must not leave the Runs grid showing a different, healthy one.
  if (name === "unhealthy" || name === "followup" || name === "text_only" || name === "yesterday") {
    const first = todayFor(name).run;
    return { runs: [first, ...base.filter((run) => run.id !== first.id)] };
  }
  return { runs: base };
}

async function serveStream(name, response, request, ask) {
  const frames = stream(name);
  // A scenario where the client leaves mid-answer must not record the turn the real server would
  // have stored: the request is aborted, and nothing after it is written.
  let aborted = false;
  request?.on("close", () => {
    aborted = true;
  });
  response.on("error", () => {
    aborted = true;
  });
  response.writeHead(200, {
    "content-type": "text/event-stream",
    "cache-control": "no-cache, no-transform",
    "x-accel-buffering": "no",
  });
  for (const frame of frames) {
    await sleep(frame.delay_ms ?? 0);
    if (aborted || response.writableEnded || response.destroyed) return;
    if (frame.comment) {
      // A comment line is the server's heartbeat: no event, no data, only bytes.
      response.write(`${frame.comment}\n\n`);
      continue;
    }
    response.write(`event: ${frame.event}\ndata: ${JSON.stringify(frame.data)}\n\n`);
  }
  // `drop` ends here without an answer event, which is what a dead connection looks like.
  const last = frames.at(-1);
  if (last?.event === "answer" && ask) {
    // Only a finished answer is stored, and its `thread` event is the stream's last frame, exactly
    // as the real API orders them.
    // `thread_slow` holds the stored turn back, so a test can act between the answer and it.
    if (scenario === "thread_slow") await sleep(1500);
    if (aborted || response.writableEnded || response.destroyed) return;
    const thread = saveTurn(ask, last.data);
    response.write(
      `event: thread\ndata: ${JSON.stringify({ id: thread.id, turn: thread.turns.length })}\n\n`,
    );
  }
  response.end();
}

async function handle(request, response) {
  const url = new URL(request.url ?? "/", `http://${request.headers.host ?? "127.0.0.1"}`);
  const path = url.pathname;

  if (path === "/healthz") {
    json(response, 200, { status: "ok" });
    return;
  }

  if (path === "/__scenario" && request.method === "POST") {
    let body = "";
    for await (const chunk of request) body += chunk;
    const picked = JSON.parse(body || "{}").name;
    if (!SCENARIOS.has(picked)) {
      json(response, 400, { detail: `unknown scenario: ${picked}` });
      return;
    }
    scenario = picked;
    suggestionRequests = 0;
    resetDrafts();
    // A test that asks, then changes how the next stream behaves, keeps the turns it stored.
    if (!JSON.parse(body || "{}").keepThreads) resetThreads();
    lastAsk = null;
    json(response, 200, { scenario });
    return;
  }

  // The real gate checks the credential first, then refuses any request carrying an Origin.
  if (request.headers.authorization !== `Bearer ${TOKEN}`) {
    refused(response, 401);
    return;
  }

  if (request.headers.origin) {
    refused(response, 403);
    return;
  }

  if (path === "/status") {
    json(response, 200, fixture("status"));
    return;
  }

  if (path === "/today") {
    if (scenario === "error") {
      json(response, 500, { detail: "internal server error" });
      return;
    }
    if (scenario === "slow") await sleep(1500);
    json(response, 200, todayFor(scenario));
    return;
  }

  if (path === "/runs") {
    if (scenario === "error") {
      json(response, 500, { detail: "internal server error" });
      return;
    }
    if (scenario === "slow") await sleep(1500);
    if (scenario === "empty") {
      json(response, 200, { runs: [] });
      return;
    }
    json(response, 200, runsFor(scenario));
    return;
  }

  if (path.startsWith("/runs/")) {
    if (scenario === "error") {
      json(response, 500, { detail: "internal server error" });
      return;
    }
    const id = path.slice("/runs/".length);
    // FastAPI refuses a path id that is not a UUID with 422 before the route runs; the page turns
    // that into "nothing here" without asking.
    if (!/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(id)) {
      json(response, 422, {
        detail: [
          { type: "uuid_parsing", loc: ["path", "run_id"], msg: "Input should be a valid UUID" },
        ],
      });
      return;
    }
    const detail = fixture("run-detail");
    if (id !== detail.id) {
      json(response, 404, { detail: "run not found" });
      return;
    }
    // Dated like its row in the Runs list, so the heading's day matches the scenario.
    json(response, 200, runAt(detail, scenario === "yesterday" ? 1 : 0));
    return;
  }

  if (path === "/items" && request.method === "GET") {
    if (scenario === "error") {
      json(response, 500, { detail: "internal server error" });
      return;
    }
    if (scenario === "empty") {
      json(response, 200, fixture("corpus-empty"));
      return;
    }
    const q = (url.searchParams.get("q") ?? "").trim().toLowerCase();
    const tier = url.searchParams.get("tier");
    const source = url.searchParams.get("source");
    const profile = url.searchParams.get("profile");
    const limit = Math.min(Number(url.searchParams.get("limit") ?? 25), 50);
    const offset = corpusOffset(url.searchParams.get("cursor"));
    const rows = corpusRows().filter(
      (item) =>
        (!tier || item.tier === tier) &&
        (!source || item.source === source) &&
        (!profile || item.profile === profile) &&
        (!q || item.title.toLowerCase().includes(q)),
    );
    const total = rows.length;
    json(response, 200, {
      items: rows.slice(offset, offset + limit),
      next_cursor: offset + limit < total ? corpusCursor(offset + limit) : null,
      total,
    });
    return;
  }

  if (path.startsWith("/items/")) {
    if (scenario === "error") {
      json(response, 500, { detail: "internal server error" });
      return;
    }
    const id = path.slice("/items/".length);
    // FastAPI refuses a path id that is not a UUID with 422 before the route runs.
    if (!/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(id)) {
      json(response, 422, {
        detail: [{ type: "uuid_parsing", loc: ["path", "item_id"], msg: "Input should be a valid UUID" }],
      });
      return;
    }
    const detail = fixture("item-detail");
    if (id !== detail.id) {
      json(response, 404, { detail: "item not found" });
      return;
    }
    json(response, 200, detail);
    return;
  }

  if (path === "/suggestions" && request.method === "GET") {
    if (scenario === "error") {
      json(response, 500, { detail: "internal server error" });
      return;
    }
    if (scenario === "slow") await sleep(1500);
    if (scenario === "empty") {
      json(response, 200, fixture("suggestions-empty"));
      return;
    }
    if (scenario === "summary_pending") {
      // The first list is served before the background summary is written, the next one after.
      suggestionRequests += 1;
      json(
        response,
        200,
        fixture(suggestionRequests === 1 ? "suggestions-pending" : "suggestions"),
      );
      return;
    }
    // A drafted item leaves the list, like the real API's `drafted` event does.
    const body = fixture(scenario === "javascript" ? "suggestions-javascript" : "suggestions");
    const keep = (card) => !draftedItemIds.has(card.item_id);
    json(response, 200, {
      ...body,
      picks: body.picks.filter(keep),
      rising: body.rising.filter(keep),
    });
    return;
  }

  if (path === "/drafts" && request.method === "GET") {
    if (scenario === "drafts_empty") {
      json(response, 200, fixture("drafts-empty"));
      return;
    }
    if (scenario === "drafts_ellipsis") {
      json(response, 200, fixture("drafts-ellipsis"));
      return;
    }
    json(response, 200, { drafts });
    return;
  }

  if (path === "/drafts" && request.method === "POST") {
    let raw = "";
    for await (const chunk of request) raw += chunk;
    let body = {};
    try {
      body = JSON.parse(raw || "{}");
    } catch {
      json(response, 400, { detail: "bad body" });
      return;
    }
    if (scenario === "draft_refused") {
      json(response, 503, fixture("draft-refused"));
      return;
    }
    if (scenario === "draft_cap") {
      json(response, 503, fixture("draft-capped"));
      return;
    }
    if (scenario === "draft_slow") {
      // Long enough for a test to see the button's writing state, short enough for CI.
      await sleep(1200);
    }
    const template = fixture(scenario === "draft_flags" ? "draft-flagged" : "draft-created");
    const card = [...fixture("suggestions").picks, ...fixture("suggestions").rising].find(
      (entry) => entry.item_id === body.item_id,
    );
    const created = {
      ...template,
      id: randomUUID(),
      item_id: body.item_id,
      item_title: card?.title ?? template.item_title,
      item_url: card?.url ?? template.item_url,
      created_at: new Date().toISOString(),
      regenerated_from: null,
    };
    created.chars = linkedinText(created).length;
    drafts = [created, ...drafts];
    draftedItemIds = new Set([...draftedItemIds, body.item_id]);
    json(response, 201, created);
    return;
  }

  if (/^\/drafts\/[0-9a-f-]+\/regenerate$/.test(path) && request.method === "POST") {
    const id = path.split("/")[2];
    const old = drafts.find((draft) => draft.id === id);
    if (!old) {
      json(response, 404, { detail: "draft not found" });
      return;
    }
    const created = {
      ...old,
      id: randomUUID(),
      created_at: new Date().toISOString(),
      regenerated_from: id,
      hook: "Regenerated: a fresh cut of the same evidence.",
    };
    created.chars = linkedinText(created).length;
    drafts = [created, ...drafts];
    json(response, 201, created);
    return;
  }

  if (/^\/drafts\/[0-9a-f-]+$/.test(path) && request.method === "GET") {
    if (scenario === "draft_detail_failed") {
      json(response, 500, { detail: "internal server error" });
      return;
    }
    const id = path.split("/")[2];
    const found = drafts.find((draft) => draft.id === id);
    if (!found) {
      json(response, 404, { detail: "draft not found" });
      return;
    }
    json(response, 200, { ...found, pov: fixture("draft-detail").pov });
    return;
  }

  // Events are fire-and-forget from the page: a beacon for an opened link, a fetch for a dismissal.
  if (/^\/suggestions\/[0-9a-f-]+\/events$/.test(path) && request.method === "POST") {
    response.writeHead(204);
    response.end();
    return;
  }

  if (path === "/__last_ask" && request.method === "GET") {
    // The test seam for what the page sent: the chip's filter and the follow-up's thread id.
    json(response, 200, lastAsk ?? {});
    return;
  }

  if (path === "/threads" && request.method === "GET") {
    const list = [...threads]
      .sort((a, b) => (a.updated_at < b.updated_at ? 1 : -1))
      .map((thread) => ({
        id: thread.id,
        title: thread.title,
        turns: thread.turns.length,
        updated_at: thread.updated_at,
      }));
    json(response, 200, { threads: list });
    return;
  }

  const thread = /^\/threads\/([0-9a-f-]{36})$/.exec(path);
  if (thread && request.method === "GET") {
    const found = threads.find((one) => one.id === thread[1]);
    if (!found) {
      json(response, 404, { detail: "thread not found" });
      return;
    }
    json(response, 200, { id: found.id, title: found.title, turns: found.turns });
    return;
  }

  if (path === "/ask" && request.method === "POST") {
    let raw = "";
    try {
      for await (const chunk of request) raw += chunk;
    } catch {
      // A client that leaves while its body is being read is not a server failure.
      return;
    }
    try {
      lastAsk = JSON.parse(raw || "{}");
    } catch {
      lastAsk = {};
    }
    if (scenario === "refused") {
      refused(response, 429);
      return;
    }
    if (lastAsk.thread_id && !threads.some((one) => one.id === lastAsk.thread_id)) {
      // The real API refuses an unknown thread before its stream starts, so the page can see the
      // same 404 here instead of the fake quietly starting a second conversation.
      json(response, 404, { detail: "thread not found" });
      return;
    }
    if (scenario === "thread_drop" && lastAsk.thread_id) {
      // A follow-up whose connection dies: the server cancels it, so nothing is stored.
      await serveStream("drop", response, request, lastAsk);
      return;
    }
    const name =
      scenario === "normal" || scenario === "empty" || scenario === "thread_slow"
        ? "normal"
        : scenario;
    await serveStream(name, response, request, lastAsk);
    return;
  }

  json(response, 404, { detail: "not found" });
}

// One request can fail on its own without taking the fixture server down: a rejected handler would
// be an unhandled rejection, which exits the process and turns every later test into ECONNREFUSED.
const server = createServer((request, response) => {
  handle(request, response).catch((error) => {
    console.error("fake API request failed:", error);
    if (!response.headersSent) {
      json(response, 500, { detail: "fake API error" });
      return;
    }
    response.destroy();
  });
});

server.listen(PORT, "127.0.0.1", () => {
  console.log(`fake API on http://127.0.0.1:${PORT}`);
});
