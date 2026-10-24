"""The drafts routes: generate one checked LinkedIn draft, list the saved ones, regenerate one.

Every model-spending request goes through the process's `RequestLimiter` (the same one in front of
an answer) and through the chat guard's daily cap, so a loop here cannot spend the free tier's
calls. A draft is saved before the response and its `drafted` event takes the item out of the
picks, so the list and the draft cannot disagree about what was already used.

The point of view is stored with its draft and returned only by the detail route. It never reaches
a log line, a refusal body or a provider's error text: the shared sentence names the failure, and
the provider's own message stays out of every response.
"""

from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from uuid import UUID

import structlog
from fastapi import APIRouter, HTTPException, Query, Request, Security
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from paperwatch.api.auth import Principal
from paperwatch.api.limits import Refused, RequestLimiter
from paperwatch.api.responses import BEARER, GATE_RESPONSES, RateLimited, Refusal
from paperwatch.api.routes.suggestions import SuggestionService
from paperwatch.config import Settings
from paperwatch.drafts.check import DraftCheck, check_draft
from paperwatch.drafts.generate import (
    DRAFT_PROMPT_VERSION,
    NO_COMPLIANT_MODEL,
    DraftSource,
    DraftText,
    DraftUnusable,
    draft_text,
    generate_draft,
)
from paperwatch.drafts.prompts import MAX_POV_CHARS
from paperwatch.llm.chat import (
    CallGuard,
    ChatCallStore,
    ChatModel,
    DailyCapReached,
    ModelUnavailable,
)
from paperwatch.llm.ledger import BudgetExceeded, Ledger
from paperwatch.store.models import ItemSummary
from paperwatch.store.repository import DraftItem, DraftRow, Repository
from paperwatch.suggestions.cards import basis_label
from paperwatch.suggestions.config import SuggestionsConfig
from paperwatch.suggestions.ranking import is_pick_grade

log = structlog.get_logger()

# The Drafts section asks for the newest 50; a larger page is a client bug, not a reader's need.
DRAFT_LIST_LIMIT = 50
LANGUAGE_MAX_CHARS = 8
# The daily cap's sentence names the failure the reader can act on. The reset time travels as a
# timestamp, so the page never has to parse prose.
CAP_MESSAGE = "The daily model cap is spent, so the draft was not generated."
BUDGET_MESSAGE = "The reply was too long to keep as a draft, so nothing was saved."
UNUSABLE_MESSAGE = "The model's reply could not be read as a draft, so nothing was saved."


class DraftError(BaseModel):
    """A 503 the page can show: which failure, a sentence, and the cap's reset time."""

    code: Literal["daily_cap", "model_unavailable", "unusable"]
    message: str
    retry_at: datetime | None = None


class DraftFlagData(BaseModel):
    """One numbered note under a draft; `text` is the claim it is about."""

    n: int
    kind: str
    text: str
    detail: str


class DraftData(BaseModel):
    """One saved draft as the list shows it; the detail adds the owner's own words."""

    id: UUID
    item_id: UUID
    item_title: str
    item_url: str
    created_at: datetime
    regenerated_from: UUID | None
    language: str
    model: str
    hook: str
    body: str
    hashtags: list[str]
    source_link: str
    flags: list[DraftFlagData]
    counts: dict[str, tuple[int, int]]
    chars: int


class DraftDetail(DraftData):
    """The detail adds the owner's own words, so "Open" shows what the draft was built from."""

    pov: str


class DraftsData(BaseModel):
    drafts: list[DraftData]


class DraftRequest(BaseModel):
    item_id: UUID
    # The API's bound; the page counts to the same number, and an empty point of view is allowed.
    pov: str = Field(default="", max_length=MAX_POV_CHARS)
    language: str = Field(default="en", min_length=2, max_length=LANGUAGE_MAX_CHARS)

    @field_validator("language")
    @classmethod
    def _language(cls, value: str) -> str:
        # The prompt receives this as the requested language, so a stray newline would be an
        # instruction line of its own; a code is letters and a dash.
        cleaned = value.strip().lower()
        if not cleaned.replace("-", "").isalpha():
            raise ValueError("language must be a language code such as en or pt-br")
        return cleaned


class DraftRefused(Exception):
    """A generation failure with the code and status the route turns into its response."""

    def __init__(
        self,
        code: Literal["daily_cap", "model_unavailable", "unusable"],
        message: str,
        *,
        status: int = 503,
        retry_at: datetime | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.retry_at = retry_at


def draft_source(
    item: DraftItem, summary: ItemSummary | None, config: SuggestionsConfig
) -> DraftSource:
    """The stored item plus the summary it will be drafted from.

    `early` reuses the ranking's own gate, so the dialog's notice, the draft's early sentence and
    the Rising shelf all mean the same thing by construction.
    """
    candidate = item.candidate
    return DraftSource(
        item_id=item.item_id,
        title=item.title,
        url=item.url,
        source=item.source,
        tier=candidate.tier.value,
        basis_label=basis_label(
            candidate.tier,
            candidate.basis,
            candidate.venue,
            candidate.hf_upvotes,
            candidate.citation_count,
        ),
        venue=candidate.venue,
        org=candidate.org,
        text=item.text,
        github_repo=item.github_repo,
        project_page=item.project_page,
        # Only a checked summary is context: an "excerpt" row is the abstract itself, which the
        # source block already carries, and a pending one has nothing to say yet.
        summary=(
            dict(summary.fields) if summary is not None and summary.status == "checked" else None
        ),
        early=not is_pick_grade(candidate, config),
    )


class DraftService:
    """Generation, storage, the list and the detail, with the limits in one place per process."""

    def __init__(
        self,
        *,
        session_factory: Callable[[], Session],
        models_factory: Callable[[], list[ChatModel]],
        settings: Settings,
        chat_calls: ChatCallStore,
        limiter: RequestLimiter,
        suggestions: SuggestionService,
        config: SuggestionsConfig,
        clock: Callable[[], datetime],
    ) -> None:
        self._session_factory = session_factory
        # A factory, not the chain: a build without a key fails when a draft is asked for, not
        # when the app is created (the contract build has no key and needs no model).
        self._models_factory = models_factory
        self._settings = settings
        self._chat_calls = chat_calls
        self.limiter = limiter
        # The same service the suggestions route serves from, so a saved draft's event drops the
        # cached list in this process.
        self.suggestions = suggestions
        self._config = config
        self._clock = clock

    def generate(self, body: DraftRequest) -> DraftData:
        with self._session_factory() as session:
            repo = Repository(session)
            item = repo.draft_item(body.item_id)
            if item is None:
                raise HTTPException(status_code=404, detail="item not found")
            summary = repo.summaries([body.item_id]).get(body.item_id)
        source = draft_source(item, summary, self._config)
        return self._run(source, pov=body.pov, language=body.language, regenerated_from=None)

    def regenerate(self, draft_id: UUID) -> DraftData:
        with self._session_factory() as session:
            repo = Repository(session)
            existing = repo.draft(draft_id)
            if existing is None:
                raise HTTPException(status_code=404, detail="draft not found")
            item = repo.draft_item(existing.item_id)
        if item is None:
            # The foreign key cascades, so this is a deleted item, not a missing draft.
            raise HTTPException(status_code=404, detail="item not found")
        return self._run(
            _source_from_inputs(item, existing.inputs),
            pov=existing.pov,
            language=existing.language,
            regenerated_from=draft_id,
        )

    def list(self, limit: int) -> DraftsData:
        with self._session_factory() as session:
            rows = Repository(session).drafts(limit)
        return DraftsData(drafts=[_data(row) for row in rows])

    def get(self, draft_id: UUID) -> DraftDetail:
        with self._session_factory() as session:
            row = Repository(session).draft(draft_id)
        if row is None:
            raise HTTPException(status_code=404, detail="draft not found")
        return DraftDetail(**_data(row).model_dump(), pov=row.pov)

    def _run(
        self,
        source: DraftSource,
        *,
        pov: str,
        language: str,
        regenerated_from: UUID | None,
    ) -> DraftData:
        if len(source.text) < self._config.min_text_chars:
            raise _unprocessable(
                f"stored text is under {self._config.min_text_chars} characters", "item_id"
            )
        models = self._models_factory()
        guard = CallGuard(
            self._chat_calls,
            Ledger(self._settings.chat_max_tokens_per_answer),
            self._settings.chat_max_calls_per_day,
            self._clock,
        )
        try:
            text, model = generate_draft(models, guard, source, pov=pov, language=language)
        except DailyCapReached as exc:
            raise DraftRefused(
                "daily_cap", CAP_MESSAGE, retry_at=_next_midnight(self._clock())
            ) from exc
        except ModelUnavailable as exc:
            raise DraftRefused("model_unavailable", NO_COMPLIANT_MODEL) from exc
        except DraftUnusable as exc:
            raise DraftRefused("unusable", UNUSABLE_MESSAGE) from exc
        except BudgetExceeded as exc:
            # A reply past the per-answer budget has no draft in it either: same code, same
            # promise that nothing was saved.
            raise DraftRefused("unusable", BUDGET_MESSAGE) from exc
        check = check_draft(text, source, pov=pov)
        with self._session_factory() as session:
            repo = Repository(session)
            row = repo.add_draft(
                item_id=source.item_id,
                inputs=_inputs(source),
                pov=pov,
                language=language,
                model=model,
                prompt_version=DRAFT_PROMPT_VERSION,
                hook=text.hook,
                body=text.body,
                hashtags=text.hashtags,
                source_link=text.source_link,
                check=_check_payload(check),
                regenerated_from=regenerated_from,
                at=self._clock(),
            )
            session.commit()
        # After the save: the model call was paid for, so a failure recording the event must not
        # turn a saved draft into an error the reader would retry into a duplicate.
        try:
            self.suggestions.record(source.item_id, "drafted")
        except Exception as exc:  # noqa: BLE001 - the draft is already saved
            log.warning(
                "draft_event_failed",
                item_id=str(source.item_id),
                error_type=type(exc).__name__,
            )
        log.info(
            "draft_saved",
            item_id=str(source.item_id),
            draft_id=str(row.id),
            model=row.model,
            flags=len(row.flags),
            chars=len(draft_text(text)),
        )
        return _data(row)


def _source_from_inputs(item: DraftItem, inputs: Mapping[str, object]) -> DraftSource:
    """The same source a regenerate must re-run: the stored inputs, with the item's stored text."""
    summary = inputs.get("summary")
    return DraftSource(
        item_id=item.item_id,
        title=str(inputs.get("title") or item.title),
        url=str(inputs.get("url") or item.url),
        source=item.source,
        tier=str(inputs.get("tier") or item.candidate.tier.value),
        basis_label=str(inputs.get("basis_label") or ""),
        venue=_text_or_none(inputs.get("venue")),
        org=_text_or_none(inputs.get("org")),
        text=item.text,
        github_repo=item.github_repo,
        project_page=item.project_page,
        summary=summary if isinstance(summary, Mapping) else None,
        early=bool(inputs.get("early")),
    )


def _inputs(source: DraftSource) -> dict[str, object]:
    return {
        "title": source.title,
        "url": source.url,
        "tier": source.tier,
        "basis_label": source.basis_label,
        "venue": source.venue,
        "org": source.org,
        "summary": dict(source.summary) if source.summary is not None else None,
        "early": source.early,
    }


def _check_payload(check: DraftCheck) -> dict[str, object]:
    return {
        "flags": [
            {"kind": flag.kind, "text": flag.text, "detail": flag.detail} for flag in check.flags
        ],
        "counts": {kind: [checked, passed] for kind, (checked, passed) in check.counts.items()},
    }


def _data(row: DraftRow) -> DraftData:
    text = DraftText(
        hook=row.hook,
        body=row.body,
        hashtags=row.hashtags,
        source_link=row.source_link,
    )
    return DraftData(
        id=row.id,
        item_id=row.item_id,
        item_title=row.item_title,
        item_url=row.item_url,
        created_at=row.created_at,
        regenerated_from=row.regenerated_from,
        language=row.language,
        model=row.model,
        hook=row.hook,
        body=row.body,
        hashtags=list(row.hashtags),
        source_link=row.source_link,
        flags=[
            DraftFlagData(
                n=number,
                kind=str(flag.get("kind") or ""),
                text=str(flag.get("text") or ""),
                detail=str(flag.get("detail") or ""),
            )
            for number, flag in enumerate(row.flags, start=1)
        ],
        counts=dict(row.counts),
        chars=len(draft_text(text)),
    )


def _text_or_none(value: object) -> str | None:
    return str(value) if isinstance(value, str) and value else None


def _next_midnight(now: datetime) -> datetime:
    """The UTC midnight when the free tier's daily call count starts again."""
    today = now.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    return today + timedelta(days=1)


def _unprocessable(message: str, field: str) -> HTTPException:
    """A route-level 422 in the shape the contract documents for a validation error."""
    return HTTPException(
        status_code=422,
        detail=[{"type": "value_error", "loc": ["body", field], "msg": message}],
    )


_DRAFT_RESPONSES: dict[int | str, dict[str, Any]] = {
    **GATE_RESPONSES,
    404: {"model": Refusal, "description": "No admitted item, or no draft, with that id."},
    422: {"description": "The stored text is too short to draft from, or a field is invalid."},
    429: {
        "model": RateLimited,
        "description": "The per-client limit refused the request; also sends Retry-After.",
    },
    503: {
        "model": DraftError,
        "description": (
            "No draft was generated: the daily cap (`retry_at`), no compliant provider, or a "
            "reply that could not be read as a draft."
        ),
    },
}


def _limited(service: DraftService, request: Request, work: Callable[[], Any]) -> Any:
    """Admit against the shared limiter, run the work, turn a refusal into its response."""
    principal: Principal = request.state.principal
    admitted = service.limiter.admit(principal.subject)
    if isinstance(admitted, Refused):
        return JSONResponse(
            status_code=429,
            content={"detail": admitted.reason, "retry_after_s": admitted.retry_after_s},
            headers={"Retry-After": str(admitted.retry_after_s)},
        )
    try:
        return work()
    except DraftRefused as exc:
        return JSONResponse(
            status_code=exc.status,
            content=DraftError(
                code=exc.code, message=exc.message, retry_at=exc.retry_at
            ).model_dump(mode="json"),
        )
    finally:
        admitted.release()


def make_drafts_router(service: DraftService) -> APIRouter:
    """Build the router from the service, so the app owns one draft service per process."""
    router = APIRouter()

    @router.post(
        "/drafts",
        status_code=201,
        response_model=DraftData,
        dependencies=[Security(BEARER)],
        responses=_DRAFT_RESPONSES,
    )
    def create_draft(body: DraftRequest, request: Request) -> Any:
        """Generate, check and save one draft; generating is saving."""
        return _limited(service, request, lambda: service.generate(body))

    @router.post(
        "/drafts/{draft_id}/regenerate",
        status_code=201,
        response_model=DraftData,
        dependencies=[Security(BEARER)],
        responses=_DRAFT_RESPONSES,
    )
    def regenerate_draft(draft_id: UUID, request: Request) -> Any:
        """Re-run the stored inputs and point of view; the old draft stays, linked."""
        return _limited(service, request, lambda: service.regenerate(draft_id))

    @router.get(
        "/drafts",
        response_model=DraftsData,
        dependencies=[Security(BEARER)],
        responses={**GATE_RESPONSES},
    )
    def list_drafts(limit: int = Query(DRAFT_LIST_LIMIT, ge=1, le=DRAFT_LIST_LIMIT)) -> DraftsData:
        return service.list(limit)

    @router.get(
        "/drafts/{draft_id}",
        response_model=DraftDetail,
        dependencies=[Security(BEARER)],
        responses={
            **GATE_RESPONSES,
            404: {"model": Refusal, "description": "No saved draft with that id."},
        },
    )
    def get_draft(draft_id: UUID) -> DraftDetail:
        return service.get(draft_id)

    return router


__all__ = [
    "DraftData",
    "DraftDetail",
    "DraftService",
    "DraftsData",
    "make_drafts_router",
]
