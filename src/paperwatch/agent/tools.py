"""The agent's read-only tools: search the corpus, open one item, compare a few.

The model never sees the database. It sees what these tools return, and they return only admitted
items, always as delimited `<source>` blocks with markup escaped. That is the structural defence
against prompt injection: stored text is written by strangers, so it can say "ignore previous
instructions" but it cannot close its block, open a new one, or name a source no tool returned.
Each call also logs its hits' tiers in rank order, so the tier mix of real agent queries can be
measured against the figures in `docs/learn/07`.

`ToolNode` runs one message's tool calls concurrently, but the tools share one `Session` and the
`SourceRegistry` is not thread-safe, so `make_tools` holds one lock around every tool body. Each
body also runs in a savepoint: one failed statement would otherwise abort the session's transaction
and fail every later tool call in the same answer.
"""

import contextlib
import html
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Literal

import structlog
from langchain_core.tools import BaseTool, StructuredTool, ToolException
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from paperwatch.agent.prompts import TEXT_ONLY_NOTE
from paperwatch.agent.sources import SourceRef, SourceRegistry
from paperwatch.domain import Tier
from paperwatch.llm.embedder import Embedder, EmbedderUnavailable
from paperwatch.retrieval.hybrid import Filters, hybrid_search
from paperwatch.store.repository import ItemDetail, Repository

log = structlog.get_logger()

# A full lab post runs to thousands of words, and the model's context is shared with the answer;
# these caps keep one tool result from crowding out the rest of the conversation.
GET_ITEM_MAX_CHARS = 6000
COMPARE_MAX_CHARS = 1500
COMPARE_MIN, COMPARE_MAX = 2, 4
# An item's title, author names and URL are not bounded by the schema (arXiv papers can list
# thousands of authors, feed fields are arbitrary strings), so a block clips all three: a result
# stays bounded whatever the stored item looks like. Escaping grows text at most 5x ("&" becomes
# "&amp;") and an attribute at most 6x (a quote becomes "&quot;"), so one get_item's clipped fields
# stay under 6x their stored length. The URL clip applies to the printed attribute only; SourceRef
# keeps the full value for citations.
TITLE_MAX_CHARS = 300
AUTHOR_MAX_CHARS = 100
AUTHORS_SHOWN = 20
URL_MAX_CHARS = 2048
# Around 30,000 terms make the any-term tsquery exceed Postgres's stack depth, and that
# OperationalError aborts the session's transaction, so a model query is cut to this many
# characters first (far more than any useful search needs).
QUERY_MAX_CHARS = 2000
NO_HITS = "No admitted item matched this search."
# A module-level default, not a call in the signature: the filters are immutable, and ruff's
# B008 keeps function calls out of argument defaults.
_NO_FILTERS = Filters()


class SearchArgs(BaseModel):
    query: str = Field(description="Search terms, or a natural-language question.")
    tiers: list[Tier] | None = Field(
        default=None,
        description="Optional reliability tiers: peer_reviewed, vetted_preprint, lab_blog.",
    )
    profiles: list[str] | None = Field(
        default=None,
        description="Optional topics: agentic-engineering, ai-forensics, ml-architectures.",
    )
    since: date | None = Field(
        default=None, description="Optional earliest publication date, YYYY-MM-DD."
    )


class GetItemArgs(BaseModel):
    source_id: str = Field(description="A source alias from a previous search result, such as S3.")


class CompareArgs(BaseModel):
    source_ids: list[str] = Field(
        description="2 to 4 source aliases from previous search results, in the order to compare."
    )


def _narrow(
    base: Filters,
    tiers: list[Tier] | None,
    profiles: list[str] | None,
    since: date | None,
) -> Filters:
    """Combine the tool's fixed filters with the model's: both apply, the narrower one wins."""

    def both(fixed: frozenset[str] | None, asked: list[str] | None) -> frozenset[str] | None:
        # An empty list from the model is no request, not a request for nothing (which is what an
        # empty frozenset in Filters means); only the code-built base filter can match nothing.
        if not asked:
            return fixed
        wanted = frozenset(asked)
        return wanted if fixed is None else fixed & wanted

    since_at = base.since
    if since is not None:
        # A date means midnight UTC of that day; a later base bound still wins.
        asked_at = datetime(since.year, since.month, since.day, tzinfo=UTC)
        since_at = asked_at if since_at is None else max(since_at, asked_at)
    return Filters(
        tiers=both(base.tiers, [t.value for t in tiers] if tiers else None),
        profiles=both(base.profiles, profiles),
        since=since_at,
    )


def _clip(text: str, limit: int) -> str:
    # An ellipsis tells the model the text continues; an item field has no other bound.
    return text[:limit] + "..." if len(text) > limit else text


def _clean(value: str) -> str:
    # A NUL reaches Postgres as a DataError and a lone surrogate fails UTF-8 encoding; neither is
    # text, so both are dropped rather than crashing the tool call.
    return value.replace("\x00", "").encode("utf-8", "ignore").decode()


def _block(ref: SourceRef, title: str, text: str, meta: str | None = None) -> str:
    # Stored text is untrusted data: escaping `&`, `<` and `>` keeps it from closing this block
    # or opening another, and the attribute values are escaped with quotes so a crafted URL cannot
    # break out of its attribute either. The model is told to treat everything inside as data.
    # The printed URL is clipped (no ellipsis: a truncated URL is already unusable, and "..."
    # would look like part of it); SourceRef keeps the full value for citation rendering.
    head = (
        f'<source id="{html.escape(ref.alias, quote=True)}"'
        f' tier="{html.escape(ref.tier, quote=True)}"'
        f' basis="{html.escape(ref.basis, quote=True)}"'
        f' published="{ref.published_at.date().isoformat()}"'
        f' url="{html.escape(ref.url[:URL_MAX_CHARS], quote=True)}">'
    )
    lines = [head, html.escape(_clip(title, TITLE_MAX_CHARS), quote=False)]
    if meta is not None:
        lines.append(html.escape(meta, quote=False))
    lines += [html.escape(text, quote=False), "</source>"]
    return "\n".join(lines)


def _meta(detail: ItemDetail) -> str:
    authors = [_clip(a, AUTHOR_MAX_CHARS) for a in detail.authors[:AUTHORS_SHOWN]]
    shown = ", ".join(authors) if authors else "unknown"
    if len(detail.authors) > AUTHORS_SHOWN:
        shown += f" and {len(detail.authors) - AUTHORS_SHOWN} more"
    parts = [f"Authors: {shown}"]
    if detail.venue:
        parts.append(f"Venue: {detail.venue}")
    parts.append(f"Citations: {detail.citation_count}")
    return " | ".join(parts)


@contextmanager
def _database_work(session: Session, tool: str) -> Iterator[None]:
    # A failed statement aborts the session's transaction, which would fail every later tool call
    # in this answer; a savepoint contains the damage to this call.
    try:
        with session.begin_nested():
            yield
    except SQLAlchemyError as exc:
        # The engine hides bound parameters, so this carries the SQL but not the model's query.
        log.error("tool_database_error", tool=tool, error=str(exc))
        raise ToolException(f"{tool} failed on a database error; try the same call again.") from exc


@dataclass
class SearchLog:
    """What the searches of one answer did, so the answer can label its own retrieval mode."""

    text_only: bool = False


def make_tools(
    session: Session | None,
    embedder: Embedder | None,
    registry: SourceRegistry,
    base_filters: Filters = _NO_FILTERS,
    k: int = 8,
    *,
    mode: Callable[[], Literal["hybrid", "text_only"]] | None = None,
    search_log: SearchLog | None = None,
    session_factory: Callable[[], Session] | None = None,
) -> list[BaseTool]:
    """Build the three read-only tools for one answer.

    `session_factory` is how the API gives each tool call its own short read-only session: the
    answer then holds no connection while the model thinks. Without it (the CLI, the eval, tests)
    the tools use the session the caller already made read-only.

    `mode` is read before every search: when it says `text_only`, or the embedding child fails,
    the search runs on the text branch alone and its result says so. `search_log` records that, so
    the answer can carry `search_mode`.
    """
    lock = threading.Lock()

    @contextmanager
    def session_scope(tool: str) -> Iterator[Session]:
        if session_factory is None:
            if session is None:
                raise RuntimeError("make_tools needs a session or a session_factory")
            with _database_work(session, tool):
                yield session
            return
        short = None
        try:
            short = session_factory()
            # The tools only read, so the database enforces it per session: a tool added later
            # cannot write even if its code tries. It must be the transaction's first statement.
            short.execute(text("SET TRANSACTION READ ONLY"))
        except SQLAlchemyError as exc:
            # A pool timeout or a connect failure must reach the model as a tool message, not as a
            # raw OperationalError whose repr ends up in the prompt. A session that was checked
            # out must go back to the pool even though its first statement failed.
            if short is not None:
                with contextlib.suppress(Exception):
                    short.rollback()
                with contextlib.suppress(Exception):
                    short.close()
            log.error("tool_database_error", tool=tool, error=str(exc))
            raise ToolException(
                f"{tool} failed on a database error; try the same call again."
            ) from exc
        try:
            with _database_work(short, tool):
                yield short
        finally:
            # A read-only session has nothing to commit; rollback clears a failed statement and
            # close returns the connection before the model's next round.
            short.rollback()
            short.close()

    child_failed = False

    def choose() -> tuple[Embedder | None, bool]:
        nonlocal child_failed
        if embedder is None or child_failed:
            return None, True
        if mode is not None and mode() == "text_only":
            return None, True
        return embedder, False

    def search(
        query: str,
        tiers: list[Tier] | None = None,
        profiles: list[str] | None = None,
        since: date | None = None,
    ) -> str:
        nonlocal child_failed
        # A small model can send a blank query, a stray NUL or a lone surrogate: the last two
        # would reach Postgres as a DataError or a UTF-8 failure, and the blank query trips
        # hybrid_search's own ValueError, so all three are handled here and come back as a message.
        # The clip keeps an absurd query from making the any-term tsquery too deep to plan.
        query = _clean(query)[:QUERY_MAX_CHARS]
        if not query.strip():
            raise ToolException("query must not be blank.")
        clean_profiles = [_clean(p) for p in profiles] if profiles else profiles
        with lock, session_scope("search_corpus") as active:
            chosen, text_only = choose()
            try:
                hits = hybrid_search(
                    active,
                    chosen,
                    query,
                    k=k,
                    filters=_narrow(base_filters, tiers, clean_profiles, since),
                )
            except EmbedderUnavailable:
                if chosen is None:
                    # Already keyword-only, so there was no embedder to fail; anything here is a
                    # tool error the model can retry, never a crash that ends the answer.
                    raise ToolException("search failed; try the same call again.") from None
                # A dead or timed-out child costs this search its semantic half, not the answer.
                # The rest of the answer stays keyword-only: another search would only wait for
                # the same cold start to fail again.
                child_failed = True
                text_only = True
                hits = hybrid_search(
                    active,
                    None,
                    query,
                    k=k,
                    filters=_narrow(base_filters, tiers, clean_profiles, since),
                )
            if text_only and search_log is not None:
                search_log.text_only = True
            # One line per call, tiers in rank order: real query mixes can be compared with the
            # measured tier shares in docs/learn/07. The query text itself is model-controlled and
            # can be huge, so only its length is logged.
            log.info("search_corpus", query_chars=len(query), tiers=[h.tier for h in hits])
            if not hits:
                return NO_HITS
            # One query for every hit, never one per hit, and only admitted rows come back.
            details = Repository(active).admitted_details([h.item_id for h in hits])
            blocks = [
                _block(registry.register(detail), detail.title, hit.snippet)
                for hit in hits
                if (detail := details.get(hit.item_id)) is not None
            ]
            body = "\n".join(blocks) if blocks else NO_HITS
            return f"{TEXT_ONLY_NOTE}\n{body}" if text_only else body

    def registered(active: Session, alias: str) -> tuple[SourceRef, ItemDetail]:
        ref = registry.get(alias)
        if ref is None:
            known = ", ".join(r.alias for r in registry.all()) or "none"
            # The alias is model-controlled text, so the echo is clipped; the known list helps a
            # small model recover from a wrong-case or invented id.
            raise ToolException(
                f"{alias[:40]!r} is not a source in this answer (known: {known}). Only aliases a "
                "search returned can be used."
            )
        detail = Repository(active).admitted_details([ref.item_id]).get(ref.item_id)
        if detail is None:
            raise ToolException(f"{alias} is no longer in the admitted corpus.")
        return ref, detail

    def get_item(source_id: str) -> str:
        with lock, session_scope("get_item") as active:
            ref, detail = registered(active, source_id)
            text = detail.text[:GET_ITEM_MAX_CHARS]
            if len(detail.text) > GET_ITEM_MAX_CHARS:
                text += "\n[text truncated]"
            return _block(ref, detail.title, text, _meta(detail))

    def compare(source_ids: list[str]) -> str:
        # A repeated alias is still one source, so dedupe before counting the 2-4 range.
        aliases = list(dict.fromkeys(source_ids))
        if not COMPARE_MIN <= len(aliases) <= COMPARE_MAX:
            raise ToolException(
                f"compare_items needs {COMPARE_MIN} to {COMPARE_MAX} distinct source ids, "
                f"got {len(aliases)}."
            )
        with lock, session_scope("compare_items") as active:
            blocks = []
            for alias in aliases:
                ref, detail = registered(active, alias)
                blocks.append(_block(ref, detail.title, detail.text[:COMPARE_MAX_CHARS]))
            return "\n".join(blocks)

    return [
        StructuredTool.from_function(
            func=search,
            name="search_corpus",
            description=(
                "Search the admitted paper corpus by hybrid retrieval. Returns ranked <source> "
                "blocks with a short snippet of each item; use get_item for the full text. Only "
                "sources this tool returns may be cited."
            ),
            args_schema=SearchArgs,
            handle_tool_error=True,
        ),
        StructuredTool.from_function(
            func=get_item,
            name="get_item",
            description=(
                "Return one source in full, by the alias a search returned, with its authors, "
                "venue and citation count. The text is clipped to keep the answer's budget."
            ),
            args_schema=GetItemArgs,
            handle_tool_error=True,
        ),
        StructuredTool.from_function(
            func=compare,
            name="compare_items",
            description=(
                "Return the text of 2 to 4 sources from earlier searches, in the order given, "
                "each clipped, so they can be compared side by side."
            ),
            args_schema=CompareArgs,
            handle_tool_error=True,
        ),
    ]
