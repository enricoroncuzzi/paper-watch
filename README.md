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

Configuration comes from `PW_*` environment variables; see [`.env.example`](.env.example).

## Documentation

- [`docs/learn/`](docs/learn/): one lesson per technology, starting with
  [how to read this repo](docs/learn/00-how-to-read-this-repo.md).
- [`docs/adr/`](docs/adr/): architecture decision records.
