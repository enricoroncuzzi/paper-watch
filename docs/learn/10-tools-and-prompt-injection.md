# Tools and prompt injection

Tool calling lets a model ask the program to run a named function with JSON arguments; the program runs it and returns the result as a message. The model never touches the database: it only sees what tools return, and the tools decide what it may see. That boundary is also the security boundary. Retrieved text is written by strangers, so a paper could contain "ignore your instructions"; this is prompt injection. The defences here are structural: every tool is read-only, so an injected instruction has nothing harmful to trigger; source text sits inside delimited blocks with markup escaped, so it cannot close its block and pose as instructions; and the model refers to sources by short aliases issued by the program, so it cannot cite something no tool returned.

## The three tools

`src/paperwatch/agent/tools.py` builds them with `make_tools(session, embedder, registry, base_filters=Filters(), k=8)`. The signature carries the security shape: the tools get a database session that only ever reads, the embedder, and the answer's `SourceRegistry`; later tasks add the graph, not more powers.

- `search_corpus(query, tiers, profiles, since)` runs the production `hybrid_search` with the tool's own `base_filters` combined with whatever the model asked for, and both must hold: filters narrow by intersection, and the later `since` bound wins. It registers every hit and returns one `<source>` block per hit with the 300-character snippet.
- `get_item(source_id)` returns one already-registered source in full, with authors, venue and citation count, its text clipped to `GET_ITEM_MAX_CHARS`.
- `compare_items(source_ids)` returns two to four registered sources in the requested order, each clipped to `COMPARE_MAX_CHARS`. Repeated aliases count once, so the range is over distinct sources.

Filters and limits are not only model advice: an empty list from the model means "no restriction", while the code-built base filter can still match nothing on purpose, and the admission filter is applied inside both retrieval branches. `admitted_details` reads only admitted rows, so the model cannot be shown a candidate or an item that has left the corpus, whichever id an alias names.

The descriptions are written for the model, which is what makes a tool callable at all; the schema lives in pydantic classes next to each tool, and `Field(description=...)` is the text the model sees for each argument.

## Aliases instead of ids

`src/paperwatch/agent/sources.py` holds `SourceRegistry`, created per answer and thrown away with it. `register(detail)` assigns `S1`, `S2`, ... in first-seen order and returns the same alias when the same item comes back, so a paper that two searches found is one source, not two. The answer may only cite aliases the registry holds; anything else is a lookup miss that Task 15 turns into an `unknown_source` flag. This is why the tools never expose raw UUIDs or external ids: an id is a thing the model could invent, an alias is a promise the program made.

## The block format

Every tool result is:

```
<source id="S3" tier="vetted_preprint" basis="hf_upvotes" published="2026-10-14" url="https://arxiv.org/abs/2510.01234">
Title of the paper
The text, escaped.
</source>
```

Title and text are escaped with `html.escape(value, quote=False)`, and attribute values (the URL in particular) with `quote=True`. So a stored text containing `</source>` renders as `&lt;/source&gt;`: it stays data inside one block instead of ending the block and starting a new one, and a URL with a quote cannot break out of its attribute. `tier` and `basis` travel in the header because they are the reliability judgement the reader needs, and a lab's own blog post is visibly `tier="lab_blog"` rather than looking like a peer-reviewed paper. An item is not bounded by the schema, so a block clips every unbounded field too: the title to 300 characters with an ellipsis, each author name to 100 characters and the author line to the first 20 names plus `and N more`, the printed URL attribute to 2,048 characters, and the text to the tool's own limit. A result stays small whatever the stored item looks like, and the unclipped URL stays in `SourceRef` for citation rendering.

An unknown or expired alias raises `ToolException`, and every tool is built with `handle_tool_error=True`, so the model gets a short error message it can act on (search again, use a known alias) instead of the graph crashing on a guessed id. The message clips the alias it echoes back, since that text is model-controlled, and lists the aliases this answer knows so a small model can recover. A blank query and a stray NUL character are the same kind of input: the query is stripped of NUL and refused as blank with a `ToolException`, because `hybrid_search` would otherwise raise a `ValueError` and Postgres a `DataError`, and neither comes back as a message.

## Measuring what the tools return

`search_corpus` logs one structlog line per call with its hits' tiers in rank order. That is the only record of what real agent queries put in front of the model, and it exists because the tier mix of the retrieval is not neutral: the text branch prefers long documents, and `docs/learn/07` records the measured tier-3 shares on the live corpus and how they compare with the base rate.

## What the tests pin

`tests/integration/test_tools.py` covers stable aliases across three searches (the third ranks the last-issued alias first, so a print-time renumber cannot pass), the base and model filters together (both tiers and profiles), the no-hit answer, a blank query, a NUL or a lone surrogate in the query and a NUL in a profile, an over-long query clipped instead of refused, an unseen alias with its known-alias list (and `known: none` for an empty registry), text clipping and the truncation marker, the many-author, long-title and 100,000-character name and URL clips plus their exact boundaries (a 300-character title and a 20-name list print unchanged), compare and snippet clips, the 2-to-4 range with five distinct aliases refused, a repeated alias refused as not distinct, and the requested order of a comparison, one `admitted_details` query per search, the injection case (a stored `</source>` in the body, title, authors, venue or URL stays escaped inside one block), lab-blog attribution, that only admitted rows come back (a real candidate row is seeded), and the tier log. The shared `seed_items` fixture and `AGENT_ROWS` in `tests/conftest.py` seed the three-row corpus, and Tasks 16 and 17 reuse them for the graph and trajectory tests.
