"""The card shapes the suggestions route serves, built from scored candidates (Task 36).

`cards.py` is where a stored item becomes a sentence a reader can act on: what the evidence for it
is, how fast it is moving, and what the summary checked. It is pure apart from reading the stored
summary row it is handed, so the labels are testable without a database.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from uuid import UUID

from paperwatch.domain import AdmissionBasis, Tier
from paperwatch.store.models import ItemSummary
from paperwatch.suggestions.ranking import LAB_POST_SOURCE, Candidate, Scored
from paperwatch.suggestions.summary import excerpt

SECONDS_PER_DAY = 86_400.0


@dataclass(frozen=True)
class AlsoData:
    """A near-duplicate folded into this card, so the reader can see the lab's own write-up."""

    title: str
    url: str
    source: str


@dataclass(frozen=True)
class SummaryFieldData:
    text: str
    quote: str | None
    # True for "why" and "catch": the page labels these as interpretation, not the authors' words.
    interpretation: bool


@dataclass(frozen=True)
class SummaryData:
    # "pending" is a card whose summary the API has queued but not written yet.
    status: Literal["pending", "checked", "excerpt"]
    what: SummaryFieldData | None
    result: SummaryFieldData | None
    why: SummaryFieldData | None
    catch: SummaryFieldData | None
    excerpt: str


@dataclass(frozen=True)
class SuggestionCard:
    item_id: UUID
    title: str
    url: str
    source: str
    tier: str
    basis: str
    basis_label: str
    venue: str | None
    citation_count: int
    hf_upvotes: int | None
    github_repo: str | None
    github_stars: int | None
    project_page: str | None
    org: str | None
    published_at: datetime
    profile: str | None
    relevance: float
    momentum_label: str | None
    also: tuple[AlsoData, ...]
    summary: SummaryData


def basis_label(
    tier: Tier,
    basis: AdmissionBasis,
    venue: str | None,
    hf_upvotes: int | None,
    citation_count: int,
) -> str:
    """Why this item is here, in the words a reader can check against the facts beside it."""
    if tier is Tier.PEER_REVIEWED:
        # A venue is what "peer reviewed" means; the name is only printed when it is known.
        return f"peer-reviewed via venue: {venue}" if venue else "peer-reviewed at a venue"
    if tier is Tier.LAB_BLOG:
        return "lab post: the lab's own claim"
    if basis is AdmissionBasis.HF_UPVOTES:
        return f"vetted preprint via {hf_upvotes or 0} Hugging Face upvotes"
    if basis is AdmissionBasis.CITATIONS:
        return f"vetted preprint via {citation_count} citations"
    return "preprint from a recognised lab's authors, not yet independently vetted"


def momentum_label(candidate: Candidate, now: datetime) -> str | None:
    """Upvotes gained since the momentum baseline, as one line; None when there is no gain."""
    if candidate.source == LAB_POST_SOURCE or candidate.upvote_baseline is None:
        return None
    current = candidate.upvote_series[-1][1] if candidate.upvote_series else candidate.hf_upvotes
    if current is None:
        return None
    baseline_at, baseline_value = candidate.upvote_baseline
    gained = current - baseline_value
    if gained <= 0:
        return None
    days = max(1, round((now - baseline_at).total_seconds() / SECONDS_PER_DAY))
    return f"+{gained} upvotes in {days} {'day' if days == 1 else 'days'}"


def card(
    scored: Scored,
    summary: ItemSummary | None,
    now: datetime,
    *,
    text: str,
) -> SuggestionCard:
    candidate = scored.candidate
    return SuggestionCard(
        item_id=candidate.item_id,
        title=candidate.title,
        url=candidate.url,
        source=candidate.source,
        tier=candidate.tier.value,
        basis=candidate.basis.value,
        basis_label=basis_label(
            candidate.tier,
            candidate.basis,
            candidate.venue,
            candidate.hf_upvotes,
            candidate.citation_count,
        ),
        venue=candidate.venue,
        citation_count=candidate.citation_count,
        hf_upvotes=candidate.hf_upvotes,
        github_repo=candidate.github_repo,
        github_stars=candidate.github_stars,
        project_page=candidate.project_page,
        org=candidate.org,
        published_at=candidate.published_at,
        profile=candidate.profile,
        relevance=candidate.relevance,
        momentum_label=momentum_label(candidate, now),
        also=tuple(
            AlsoData(title=other.title, url=other.url, source=other.source) for other in scored.also
        ),
        summary=summary_data(summary, text),
    )


def summary_data(row: ItemSummary | None, text: str) -> SummaryData:
    # The card always has something true to show: a pending or failed summary falls back to the
    # abstract's own first sentences, never to nothing and never to an unchecked claim.
    if row is None:
        return SummaryData("pending", None, None, None, None, excerpt(text))
    fields = row.fields or {}
    if row.status != "checked":
        stored = fields.get("excerpt")
        return SummaryData(
            "excerpt",
            None,
            None,
            None,
            None,
            stored if isinstance(stored, str) and stored else excerpt(text),
        )
    return SummaryData(
        "checked",
        _field(fields, "what"),
        _field(fields, "result"),
        _field(fields, "why"),
        _field(fields, "catch"),
        "",
    )


def _field(fields: dict[str, object], name: str) -> SummaryFieldData | None:
    value = fields.get(name)
    if not isinstance(value, dict):
        return None
    return SummaryFieldData(
        text=str(value.get("text") or ""),
        quote=value.get("quote") if isinstance(value.get("quote"), str) else None,
        interpretation=bool(value.get("interpretation")),
    )
