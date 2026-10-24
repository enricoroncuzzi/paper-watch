// The fake API the page is built and tested against. Dependency-free and local: the e2e suite and
// local development never reach the real server, hold a key or call a model.
//
// Every fixture is validated against the API's own pydantic models by
// tests/unit/test_web_fake_api_fixtures.py, so this server cannot drift from the contract.
import { createServer } from "node:http";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

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
  "slow",
  "drop",
]);

function fixture(name) {
  return JSON.parse(readFileSync(join(ROOT, "fixtures", `${name}.json`), "utf8"));
}

function stream(name) {
  return JSON.parse(readFileSync(join(ROOT, "streams", `${name}.json`), "utf8"));
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

function todayFor(name) {
  if (name === "unhealthy") return fixture("today-unhealthy");
  if (name === "followup") return fixture("today-followup");
  if (name === "text_only") return fixture("today-text-only");
  if (name === "empty") {
    return { run: null, search: { mode: "hybrid", reason: "ok", since: null } };
  }
  return fixture("today");
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

  if (request.headers.authorization !== `Bearer ${TOKEN}`) {
    json(response, 401, { detail: "unauthorized" });
    return;
  }

  if (path === "/status") {
    json(response, 200, fixture("status"));
    return;
  }

  if (path === "/today") {
    if (scenario === "error") {
      json(response, 503, { status: "unavailable" });
      return;
    }
    if (scenario === "slow") await sleep(1500);
    json(response, 200, todayFor(scenario));
    return;
  }

  if (path === "/runs") {
    if (scenario === "error") {
      json(response, 503, { status: "unavailable" });
      return;
    }
    if (scenario === "slow") await sleep(1500);
    if (scenario === "empty") {
      json(response, 200, { runs: [] });
      return;
    }
    json(response, 200, fixture("runs"));
    return;
  }

  if (path.startsWith("/runs/")) {
    if (scenario === "error") {
      json(response, 503, { status: "unavailable" });
      return;
    }
    const detail = fixture("run-detail");
    if (path.slice("/runs/".length) !== detail.id) {
      json(response, 404, { detail: "run not found" });
      return;
    }
    json(response, 200, detail);
    return;
  }

  if (path === "/ask" && request.method === "POST") {
    const name = scenario === "normal" || scenario === "empty" ? "normal" : scenario;
    await serveStream(name, response);
    return;
  }

  json(response, 404, { detail: "not found" });
});

server.listen(PORT, "127.0.0.1", () => {
  console.log(`fake API on http://127.0.0.1:${PORT}`);
});
