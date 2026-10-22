# Paper Watch

Paper Watch builds a growing, queryable corpus of AI research focused on agentic engineering
(agent harnesses, bots, agent tooling), with AI forensics in scope and new ML architectures as a
fallback topic. Items come only from reliable sources, in three tiers: peer-reviewed venues,
arXiv preprints with a quality signal, and official AI-lab engineering blogs.

The corpus answers questions on demand through retrieval-augmented generation, and produces one
weekly LinkedIn draft grounded in stored sources, which the author posts, edits, skips, or defers
by hand. Paper Watch never posts to LinkedIn itself.

It is also a deliberate learning project: every technology it uses has a load-bearing job in the
code and a short lesson in [`docs/learn/`](docs/learn/).

## Quickstart

Requires Docker and [uv](https://docs.astral.sh/uv/).

```bash
docker compose up -d db       # local Postgres + pgvector
uv sync                       # create .venv and install dependencies
uv run alembic upgrade head   # create the database schema
uv run pytest                 # run the test suite
```

Configuration comes from `PW_*` environment variables; see [`.env.example`](.env.example). One
matters for production: `PW_USER_AGENT` should identify the deployment with an address arXiv can
contact (for example `paper-watch/0.1 (+https://github.com/enricoroncuzzi/paper-watch; you@example.com)`),
because arXiv asks bulk harvesters to leave a contact and the committed default carries none.

## M1: what you can do now

With the database running and migrated (above):

```bash
uv run paperwatch ingest --live              # fetch, classify, admit and embed new items
uv run paperwatch search "agent memory" --live  # hybrid retrieval over the corpus
uv run paperwatch eval                       # retrieval metrics against the committed baseline
uv run paperwatch eval --live                # same set with the real local embedder; report only, never gates
uv run paperwatch qa-eval                    # replay the recorded answers through the checker; no model or network
```

The eval runs in a uniquely named throwaway database on the configured server: it creates it,
migrates it, seeds its own synthetic corpus there and drops it afterwards, so repeated local runs
never accumulate index state and a populated development database cannot report a false
regression. The database role needs the `CREATEDB` privilege and, because the first migration adds
the `vector` extension, superuser (or the extension installed in `template1`); the CI service user
is a superuser. A killed run can leave its database behind; the next run drops this role's leftover
eval databases older than a day.

The relevance calibration needs no database:

```bash
uv run python -m evals.relevance_eval        # print thresholds and scores (--write updates the config and baseline)
```

`relevance_eval` recalibrates the relevance thresholds after changing a profile description or the
embedding model (`--write` updates `config/profiles.yaml` and `evals/relevance_baseline.json`). The
labelled abstracts and their criteria are in `evals/data/relevance/`, whose "Reading the scores"
section explains what the numbers do and do not mean; the real-model gate runs in the `model-evals`
workflow.

The live commands need no API key: embeddings and the relevance classifier run locally with the
pinned `ibm-granite/granite-embedding-small-english-r2` model (384 dimensions, about 180 MB
downloaded once on first use). Set `PW_LOCAL_EMBEDDING_CACHE_DIR` to a persistent directory to
keep that download, for example one baked into a deployed image.

**CV line (true as of M1):** Built a tiered AI-research corpus with hybrid (pgvector + full-text)
retrieval and retrieval evals.

## M2: ask questions

Answers are streamed. The agent cites the stored items it retrieved, and code checks that each
citation points at an item retrieved for this answer; claims without a valid citation are
flagged, not silently dropped ([lesson 11](docs/learn/11-citations-and-grounding.md)).

```bash
# Put PW_OPENROUTER_API_KEY in .env first (a free key from openrouter.ai; never commit it).
uv run paperwatch ask "what is new on agent memory?" --live   # stream one answer in the terminal
uv run paperwatch serve --live                                # POST /ask, server-sent events
curl -N -X POST localhost:8000/ask \
  -H 'content-type: application/json' \
  -d '{"question": "which papers evaluate coding agents?"}'
```

The chat model is an OpenRouter free model by default, with Groq as an optional fallback
(`PW_GROQ_API_KEY` and `PW_GROQ_CHAT_MODEL`); embedding and relevance classification stay local, so
only the question and the retrieved abstracts and lab posts leave the machine. `ask` needs
`--live`, the same as the other commands that touch a real service. `serve` listens on
`127.0.0.1:8000` only: the API has no sign-in until M3, so do not put it on a public interface.

`docs/api/README.md` documents the event stream and the client rules; `docs/learn/13` explains how
the stream works. Every answer stays within `PW_CHAT_MAX_CALLS_PER_DAY` (800, counted from UTC
midnight) and `PW_CHAT_MAX_TOKENS_PER_ANSWER`, and the real usage of every call is stored in the
`chat_calls` table.

**CV line (true as of M2):** Built a LangGraph agent with tool use and citation-checked answers
over the corpus.

## Documentation

- [`docs/deploy/vps.md`](docs/deploy/vps.md): running the daily ingestion and its nightly backup
  on the shared VPS, and browsing the corpus from a desktop client over an SSH tunnel.
- [`docs/learn/`](docs/learn/): one lesson per technology, starting with
  [how to read this repo](docs/learn/00-how-to-read-this-repo.md).
- [`docs/adr/`](docs/adr/): architecture decision records.

## Data sources

Venue and citation data from Semantic Scholar (https://www.semanticscholar.org).
