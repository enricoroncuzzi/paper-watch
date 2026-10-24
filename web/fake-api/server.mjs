// The fake API the page is built and tested against. Dependency-free and local: the e2e suite and
// local development never reach the real server, hold a key or call a model.
//
// Every fixture is validated against the API's own pydantic models by
// tests/unit/test_web_fake_api_fixtures.py, so this server cannot drift from the contract.
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

async function serveStream(name, response) {
  const frames = stream(name);
  response.writeHead(200, {
    "content-type": "text/event-stream",
    "cache-control": "no-cache, no-transform",
    "x-accel-buffering": "no",
  });
  for (const frame of frames) {
    await sleep(frame.delay_ms ?? 0);
    if (frame.comment) {
      // A comment line is the server's heartbeat: no event, no data, only bytes.
      response.write(`${frame.comment}\n\n`);
      continue;
    }
    response.write(`event: ${frame.event}\ndata: ${JSON.stringify(frame.data)}\n\n`);
  }
  // `drop` ends here without an answer event, which is what a dead connection looks like.
  response.end();
}

const server = createServer(async (request, response) => {
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
    json(response, 200, detail);
    return;
  }

  if (path === "/ask" && request.method === "POST") {
    if (scenario === "refused") {
      refused(response, 429);
      return;
    }
    const name = scenario === "normal" || scenario === "empty" ? "normal" : scenario;
    await serveStream(name, response);
    return;
  }

  json(response, 404, { detail: "not found" });
});

server.listen(PORT, "127.0.0.1", () => {
  console.log(`fake API on http://127.0.0.1:${PORT}`);
});
