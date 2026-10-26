# Project agent memory

Paper Watch builds a growing, queryable corpus of AI research (agentic engineering first, AI
forensics in scope, new ML architectures as a fallback topic) and answers from it with hybrid
retrieval. `README.md` covers the product and quickstart, `docs/architecture.md` the unit map and
one ingestion run, and `docs/learn/` one lesson per technology. This file holds only the map and
the standing rules.

## Map

- `src/paperwatch/sources/` has one client per source: arXiv OAI-PMH (`arxiv.py`), Hugging Face
  daily papers (`hf_daily.py`), Semantic Scholar (`semantic_scholar.py`) and the official lab
  feeds in `config/feeds.yaml` (`feeds.py`). arXiv and the feeds return a normalised `SourceItem`
  from `domain.py`; Hugging Face returns an `HfPaper` (upvotes plus extras) and Semantic Scholar
  returns venue and citation signals for arXiv ids. The shared HTTP client and retries are in
  `sources/base.py`.
- `src/paperwatch/ingest/` is the run: window, dedupe, classify, admit, embed, upsert, re-check
  late evidence, then write counts, tokens and a health verdict (`health.py`). One Postgres
  advisory lock (`INGEST_LOCK_KEY` in `config.py`) allows one run at a time.
- `src/paperwatch/admission.py` applies the explicit tier rules (venue, HF upvotes, citations,
  lab author) from `config/admission.yaml`; a tier never goes down. Relevance comes from the
  embedding-similarity classifier in `llm/classifier.py`, with per-profile thresholds in
  `config/profiles.yaml`.
- `src/paperwatch/store/` holds the ORM tables (`models.py`), the repository (`repository.py`)
  and the engine and session factory (`db.py`). The schema is in `migrations/versions/` and is
  applied with Alembic (`alembic.ini`).
- `src/paperwatch/retrieval/hybrid.py` fuses a pgvector cosine search with Postgres full-text
  search using reciprocal rank fusion. Only `admitted` items are searchable.
- `src/paperwatch/llm/` is the provider boundary: the local fastembed embedder, the
  embedding-similarity classifier, the Azure implementations kept for the Azure proof, the token
  ledger with its per-run cap (`ledger.py`), and the OpenAI-compatible chat models with the
  daily-call `CallGuard` (`chat.py`).
- `src/paperwatch/suggestions/` is the weekly picks: `ranking.py` gates admitted items, scores six
  components, merges near-duplicates and diversifies with maximal marginal relevance, all pure and
  weighted from `config/suggestions.yaml`; `summary.py` writes the 30-second summary and checks its
  quotes, numbers and names against the stored text, falling back to the abstract's excerpt;
  `cards.py` builds the card a reader sees. The route, its 5-minute cache and the background
  summary queue are `api/routes/suggestions.py` and `api/summary_queue.py` (`docs/learn/18`).
- `src/paperwatch/drafts/` is the LinkedIn draft: `generate.py` makes the one OpenRouter-only call
  (`purpose="draft"`, the chat model or `PW_DRAFT_MODEL` when set, no fallback), `check.py` is the
  code check over its numbers, quotes, links, first-person claims, attribution, early notice and
  copying, and `prompts.py` holds the versioned prompt. The route is `api/routes/drafts.py`, the
  table is migration 0007, and the page's dialog and Drafts section are `web/src/components/drafts/`
  (`docs/adr/0015-linkedin-drafts.md`). The point of view is stored with the draft and never logged.
- `src/paperwatch/agent/` is the answering layer: the read-only corpus tools (`tools.py`) with
  the `SourceRegistry` that issues the aliases S1, S2, ... (`sources.py`; a follow-up restores its
  thread's aliases from `refs()`), the LangGraph graph that loops model and tools and replays a
  thread's last turns (`graph.py`), and `citations.py`, which checks every finished answer's
  citations against the sources those tools retrieved (`docs/learn/11`).
- `src/paperwatch/api/` is the HTTP face: `gate.py` puts the credential, host, origin and size
  checks in front of `app.py` (`/healthz`, `/readyz`, `/status`, SSE `POST /ask`) and
  `run_question`, the one place an agent failure becomes an error event and the session is
  released; `auth.py` verifies the Vercel OIDC or HMAC token, `limits.py` bounds requests,
  `pacing.py` adds heartbeats and the deadline, `search_mode.py` decides hybrid or keyword-only and
  `contract.py` rewrites `docs/api/openapi.json` (`uv run python -m paperwatch.api.contract`).
  The page's read routes (`/today`, `/runs`, `/runs/{id}`) are in `api/routes/runs.py`, a thread's
  list and turns (`/threads`, `/threads/{id}`) in `api/routes/threads.py` (a finished answer is
  stored as the thread's next turn on `AgentDeps.write_session_factory`, ADR 0016), and the corpus
  table and the item page read `api/routes/items.py`: a keyset page on `(published_at, id)`, or the
  full-text ranking paged by offset, and never a model.
- `web/` is the Next.js page and the only caller of the API: server components and route handlers
  check the GitHub session themselves (`web/src/auth.ts`), the API's address and token live only on
  the server (`web/src/lib/api.ts`, `api-token.ts`), the ask sheet keeps one conversation and reads
  its thread back after a dropped stream (`web/src/lib/use-ask.ts`, `app/api/threads/`), and
  `web/fake-api/` is the fixture server every page test runs against (`docs/learn/17`). The corpus
  table is `web/src/components/corpus/`, its filter, search and paging calls go through
  `web/src/app/api/corpus/route.ts`, and `/items/[id]` renders one item and reuses the drafts
  dialog.
- `evals/` holds the retrieval eval (`retrieval_eval.py`: recall@k, MRR, nDCG@k over the
  deterministic fake embedder), the relevance eval (`relevance_eval.py`: classifier
  calibration and held-out scores) and the QA eval (`qa_eval.py`: recorded answers replayed
  through the citation check), each with a committed baseline.
- `config/` holds the hand-edited YAML (topic profiles, admission thresholds, feeds, suggestion
  weights),
  `deploy/vps/` the systemd units (the ingestion, backup and tunnel as user units, the API as a
  system unit under its own `paperwatch-api` account), the ingestion's slice and drop-in and
  the two check scripts, `docs/adr/` records architecture decisions, `docs/learn/` explains each
  technology, and `docs/architecture.md` is the current unit map.
- `src/paperwatch/cli.py` exposes `paperwatch ingest`, `paperwatch search`, `paperwatch suggest`,
  `paperwatch draft`, `paperwatch ask`, `paperwatch serve`, `paperwatch api-token`,
  `paperwatch eval` and `paperwatch qa-eval`.

## Standing rules

- Secrets come only from environment variables or the gitignored `.env`; never commit `.env` or
  a real key. `.env.example` lists the `PW_*` names, every default is in `config.py`, and
  gitleaks runs in pre-commit and CI.
- Model and deployment names live only in `src/paperwatch/config.py`. The thresholds in
  `config/profiles.yaml` are calibrated for that exact local model, and the classifier refuses
  to apply them to another one (`llm/classifier.py`).
- The default test suite mocks every HTTP call and never loads a real model; the CI `test`
  job runs exactly that suite (`-m 'not model'` in `pyproject.toml`). `paperwatch ingest`,
  `search`, `suggest`, `draft`, `ask`, `serve` and `api-token` refuse to run without `--live`
  (`cli.py`), and real-model tests are opt-in with `-m model`.
- Tests must build settings from the test's own environment: pass `_env_file=None` and clear
  `PW_*` (the `build_settings` fixture in `tests/conftest.py`), so a developer's `.env`
  never changes a result.
- The retrieval baseline changes only through `uv run paperwatch eval --write-baseline`, and the
  gate fails if any gated metric, including exact-name text-only recall and MRR, drops by more
  than 0.01 (`evals/retrieval_eval.py`). After changing a profile description or the embedding
  model, recalibrate with `uv run python -m evals.relevance_eval --write`.
- An architecture decision gets an ADR in `docs/adr/` in the same pull request that makes it
  (`docs/adr/0001-record-architecture-decisions.md`); small reversible choices do not.
- Explain non-obvious constraints in a short why-comment above the line; the existing comments
  state the reason rather than restating the code.
- Commits are conventional (`feat:`, `fix:`, `docs:`, `chore:`). release-please derives the
  version and `CHANGELOG.md` from them, so never edit the changelog by hand.
- Use `uv` for Python commands (`uv sync`, `uv run ...`). The project targets Python 3.12
  (`.python-version`) and CI installs with `uv sync --locked`, so change dependencies through
  `uv`, not by editing `uv.lock`.

## Tests and CI

- `uv run pytest` runs the unit and integration tests. Docker must be running: the integration
  tests start a `pgvector/pgvector:0.8.7-pg16` container through testcontainers
  (`tests/conftest.py`).
- The page's gates run in `web/`: `npm test` (Vitest), `npm run e2e` (Playwright, Chromium and
  WebKit against `web/fake-api/`), `npm run lint`, `npm run typecheck` and `npm run build`. After an
  API change, `npm run gen:api` regenerates `web/src/lib/api-types.ts` in the same commit.
- `uv run pytest -m model --no-cov` runs the real-model tests; they load the local embedding
  model (about 180 MB, downloaded once).
- Lint and type checks are `uv run ruff check .`, `uv run ruff format --check .` and
  `uv run mypy`. `uv run pre-commit install` adds ruff, gitleaks, hygiene checks, actionlint
  and zizmor to every commit.
- `.github/workflows/ci.yml` runs `lint` (ruff), `typecheck` (mypy), `test` (pytest), `eval`
  (`paperwatch eval` and `paperwatch qa-eval` against a pgvector service container), `security`
  (pip-audit and gitleaks), `workflow-lint` (actionlint and zizmor) and `web` (the page's npm
  gates and Playwright) on every pull request and on push to `main`.
  `codeql.yml` adds code scanning; `model-evals.yml` runs the `model` tests on model-affecting
  paths, weekly and by hand; `release-please.yml` maintains the release PR.
- Local database: `docker compose up -d db`, then `uv run alembic upgrade head`. The
  `paperwatch eval` command creates and migrates its own throwaway database, so the role needs
  `CREATEDB` and must be able to create the `vector` extension (superuser, or installed in
  `template1`).

## Maintaining this file

Keep this file to the map and rules useful to almost every change. Do not repeat what
`README.md`, `docs/architecture.md`, `docs/learn/` or the code already say; point to them
instead. Rewrite or prune existing entries rather than appending new ones, and keep entries
short and true to the code.
