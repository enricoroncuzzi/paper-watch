"""The agent's read-only tools: search the corpus, open one item, compare a few.

The model never sees the database. It sees what these tools return, and they return only admitted
items, always as delimited `<source>` blocks with markup escaped. That is the structural defence
against prompt injection: stored text is written by strangers, so it can say "ignore previous
instructions" but it cannot close its block, open a new one, or name a source no tool returned.
Each call also logs its hits' tiers in rank order, so the tier mix of real agent queries can be
measured against the figures in `docs/learn/07`.
"""

import html
from datetime import UTC, date, datetime

import structlog
from langchain_core.tools import BaseTool, StructuredTool, ToolException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from paperwatch.agent.sources import SourceRef, SourceRegistry
from paperwatch.domain import Tier
from paperwatch.llm.embedder import Embedder
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
# must stay small whatever the stored item looks like. The URL clip applies to the printed
# attribute only; SourceRef keeps the full value for citation rendering.
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


def make_tools(
    session: Session,
    embedder: Embedder,
    registry: SourceRegistry,
    base_filters: Filters = _NO_FILTERS,
    k: int = 8,
) -> list[BaseTool]:
    repo = Repository(session)

    def search(
        query: str,
        tiers: list[Tier] | None = None,
        profiles: list[str] | None = None,
        since: date | None = None,
    ) -> str:
        # A small model can send a blank query, a stray NUL or a lone surrogate: the last two
        # would reach Postgres as a DataError or a UTF-8 failure, and the blank query trips
        # hybrid_search's own ValueError, so all three are handled here and come back as a message.
        # The clip keeps an absurd query from making the any-term tsquery too deep to plan.
        query = _clean(query)[:QUERY_MAX_CHARS]
        if not query.strip():
            raise ToolException("query must not be blank.")
        clean_profiles = [_clean(p) for p in profiles] if profiles else profiles
        hits = hybrid_search(
            session,
            embedder,
            query,
            k=k,
            filters=_narrow(base_filters, tiers, clean_profiles, since),
        )
        # One line per call, tiers in rank order: real query mixes can be compared with the
        # measured tier shares in docs/learn/07. The query text itself is model-controlled and
        # can be huge, so only its length is logged.
        log.info("search_corpus", query_chars=len(query), tiers=[h.tier for h in hits])
        if not hits:
            return NO_HITS
        # One query for every hit, never one per hit, and only admitted rows come back.
        details = repo.admitted_details([h.item_id for h in hits])
        blocks = [
            _block(registry.register(detail), detail.title, hit.snippet)
            for hit in hits
            if (detail := details.get(hit.item_id)) is not None
        ]
        return "\n".join(blocks) if blocks else NO_HITS

    def registered(alias: str) -> tuple[SourceRef, ItemDetail]:
        ref = registry.get(alias)
        if ref is None:
            known = ", ".join(r.alias for r in registry.all()) or "none"
            # The alias is model-controlled text, so the echo is clipped; the known list helps a
            # small model recover from a wrong-case or invented id.
            raise ToolException(
                f"{alias[:40]!r} is not a source in this answer (known: {known}). Only aliases a "
                "search returned can be used."
            )
        detail = repo.admitted_details([ref.item_id]).get(ref.item_id)
        if detail is None:
            raise ToolException(f"{alias} is no longer in the admitted corpus.")
        return ref, detail

    def get_item(source_id: str) -> str:
        ref, detail = registered(source_id)
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
        blocks = []
        for alias in aliases:
            ref, detail = registered(alias)
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
