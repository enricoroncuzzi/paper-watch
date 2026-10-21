"""QA eval: record live answers once per model, then replay the scoring offline and repeatably.

A retrieval eval asks whether the right items came back; a QA eval asks whether the answer used
them honestly. It needs questions whose supporting evidence is known in advance, plus questions
the corpus cannot answer, because a system that always answers looks perfect until you ask it
something it does not know. Live model calls are slow, rate-limited and not reproducible, so the
answers are recorded once per model and the scoring is replayed from the recording in CI: CI then
guards the checker and the scoring exactly, and a new recording is the deliberate act of measuring
a new model. The same frozen set later compares two agent frameworks on equal terms.

`score` re-runs the production `check_answer` on every recorded answer with a registry rebuilt
from the sources that answer's tools returned, so a checker change is measured, not remembered.
The recording stores the aliases and external ids, not the checker's verdict: a recording is the
model's output plus what the tools gave it, and nothing the current code decided.
"""

import json
import math
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
from sqlalchemy.orm import Session

from evals.retrieval_eval import seed_corpus
from paperwatch.agent.citations import NO_SUPPORT, check_answer
from paperwatch.agent.graph import answer_question, build_agent
from paperwatch.agent.prompts import prompt_sha
from paperwatch.agent.sources import SourceRegistry
from paperwatch.agent.tools import make_tools
from paperwatch.config import CHAT_MAX_CALLS_PER_DAY, CHAT_MAX_TOKENS_PER_ANSWER
from paperwatch.llm.chat import CallGuard, ChatModel, ModelUnavailable
from paperwatch.llm.embedder import FakeEmbedder
from paperwatch.llm.ledger import Ledger
from paperwatch.store.repository import ItemDetail, Repository

log = structlog.get_logger()

# Bump this whenever `evals/data/qa_questions.jsonl` changes: M5 compares frameworks on the set,
# so a recording made against another version of it is not comparable and the tests refuse it.
QA_SET_VERSION = 1
# The four metrics the gate holds; `mean_tools` and `mean_tokens` are reported, not gated.
GATED = ("support_recall", "citation_precision", "clean_rate", "abstention")
_KINDS = frozenset({"answerable", "unanswerable"})
# The synthetic item id the checker's citations carry when a recording is rescored, derived from
# the external id so it needs no database and stays stable across runs.
_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "paperwatch-qa-eval")


@dataclass(frozen=True)
class RecordedSource:
    """One source the tools returned for an answer, as the model could cite it."""

    alias: str
    external_id: str
    tier: str


@dataclass(frozen=True)
class Recorded:
    """One recorded answer: the model's checked text and everything needed to rescore it.

    `provider_error` marks a question the provider never answered. The row is kept, not dropped,
    so the recording says what happened and the score can count a failure as a failure instead
    of as a model that chose not to answer. `prompt_sha` identifies the prompts it was made with;
    a different prompt makes the recording stale and the replay refuses it.
    """

    question_id: str
    model: str
    set_version: int
    answer_text: str
    sources: tuple[RecordedSource, ...]
    tools: tuple[str, ...]
    tokens: int
    provider_error: bool = False
    prompt_sha: str = ""


def _rows(path: Path) -> list[dict[str, Any]]:
    # JSONL: one object per line, blank lines ignored.
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_recorded(path: Path) -> list[Recorded]:
    return [
        Recorded(
            question_id=row["question_id"],
            model=row["model"],
            set_version=int(row["set_version"]),
            answer_text=row["answer_text"],
            sources=tuple(
                RecordedSource(
                    alias=source["alias"],
                    external_id=source["external_id"],
                    tier=source["tier"],
                )
                for source in row["sources"]
            ),
            tools=tuple(row["tools"]),
            tokens=int(row["tokens"]),
            provider_error=bool(row.get("provider_error", False)),
            prompt_sha=str(row.get("prompt_sha", "")),
        )
        for row in _rows(path)
    ]


def write_recorded(path: Path, rows: list[Recorded]) -> None:
    # asdict turns the frozen dataclasses into plain dicts; tuples become JSON arrays.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(asdict(row)) + "\n" for row in rows))


def _questions(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in _rows(path):
        question_id, kind = row["id"], row["kind"]
        if question_id in out:
            raise ValueError(f"duplicate question id: {question_id}")
        if kind not in _KINDS:
            raise ValueError(f"{question_id} has an unknown kind: {kind}")
        out[question_id] = row
    return out


def record(
    session: Session, models: list[ChatModel], questions: Path, corpus: Path
) -> list[Recorded]:
    """Answer every question with the given model chain, over the shared eval corpus.

    Retrieval is the offline `FakeEmbedder`, exactly as the CI retrieval eval, so every model
    meets the same search results and only the chat model differs. Each question gets a fresh
    registry, ledger and guard, because aliases, spending and the call cap belong to one answer.
    """
    if not models:
        raise ValueError("recording needs at least one chat model")
    embedder = FakeEmbedder()
    by_item = seed_corpus(session, embedder, corpus)
    # The requested id, which the provider may route to a variant; the recording names the model
    # it was asked for so two files cannot be confused.
    model_id = str(getattr(models[0].model, "model_name", None) or models[0].provider)
    rows: list[Recorded] = []
    for question in _questions(questions).values():
        registry = SourceRegistry()
        ledger = Ledger(CHAT_MAX_TOKENS_PER_ANSWER)
        guard = CallGuard(
            Repository(session), ledger, CHAT_MAX_CALLS_PER_DAY, lambda: datetime.now(UTC)
        )
        tools = make_tools(session, embedder, registry)
        graph = build_agent(models, tools, registry, guard)
        try:
            _, tool_names, text = answer_question(graph, question["question"])
        except ModelUnavailable as exc:
            # Every provider failed for this question. Keep going so the run can apply Step 5's
            # 2-of-20 rule and record what the other questions said; the row carries the marker
            # so the score never mistakes a provider failure for a model that declined to answer.
            log.warning(
                "qa_question_failed", question=question["id"], error_type=type(exc).__name__
            )
            rows.append(
                Recorded(
                    question_id=question["id"],
                    model=model_id,
                    set_version=QA_SET_VERSION,
                    answer_text="",
                    sources=(),
                    tools=(),
                    tokens=ledger.tokens_used,
                    provider_error=True,
                    prompt_sha=prompt_sha(),
                )
            )
            continue
        rows.append(
            Recorded(
                question_id=question["id"],
                model=model_id,
                set_version=QA_SET_VERSION,
                answer_text=text,
                sources=tuple(
                    RecordedSource(ref.alias, by_item[ref.item_id], ref.tier)
                    for ref in registry.all()
                ),
                tools=tuple(tool_names),
                tokens=ledger.tokens_used,
                prompt_sha=prompt_sha(),
            )
        )
    return rows


def _alias_index(alias: str) -> int:
    # A recording always holds S1..Sn in registry order; anything else is not one of ours.
    if not alias.startswith("S") or not alias[1:].isdigit():
        raise ValueError(f"recorded alias is not S<number>: {alias!r}")
    return int(alias[1:])


def _restore(row: Recorded) -> tuple[SourceRegistry, dict[uuid.UUID, str]]:
    """Rebuild the answer's registry and the item-id to external-id map from the recording."""
    registry = SourceRegistry()
    by_item: dict[uuid.UUID, str] = {}
    for position, source in enumerate(sorted(row.sources, key=lambda s: _alias_index(s.alias)), 1):
        if source.alias != f"S{position}":
            raise ValueError(f"{row.question_id}: recorded aliases are not S1..Sn: {row.sources}")
        item_id = uuid.uuid5(_ID_NAMESPACE, source.external_id)
        # Only identity and tier matter to the checker; attribution belongs to the live answer.
        registry.register(
            ItemDetail(
                item_id=item_id,
                title="",
                authors=(),
                text="",
                url="",
                tier=source.tier,
                basis="",
                published_at=datetime(1970, 1, 1, tzinfo=UTC),
                venue=None,
                citation_count=0,
                hf_upvotes=None,
                profile=None,
            )
        )
        by_item[item_id] = source.external_id
    return registry, by_item


def check_complete(recorded: list[Recorded], questions: Path, model: str) -> None:
    """Refuse a recording the replay cannot trust, before any metric is computed.

    `score` stays lenient because the plan's tests score a single row; the CLI replay uses this:
    a recording must cover the whole set, come from this set version and this prompt, and name the
    model its file is for. A stale recording is re-recorded, never silently scored.
    """
    expected = set(_questions(questions))
    covered = {row.question_id for row in recorded}
    if covered != expected:
        raise ValueError(
            f"the recording covers {len(covered)} of {len(expected)} questions; re-record it"
        )
    versions = {row.set_version for row in recorded}
    if versions != {QA_SET_VERSION}:
        raise ValueError(
            f"the recording is set version {sorted(versions)}, not {QA_SET_VERSION}; re-record it"
        )
    models = {row.model for row in recorded}
    if models != {model}:
        raise ValueError(f"the recording is for {sorted(models)}, not {model}; check the file name")
    current = prompt_sha()
    stale = [row.question_id for row in recorded if row.prompt_sha != current]
    if stale:
        raise ValueError(
            f"the recording was made with a different prompt ({len(stale)} of "
            f"{len(recorded)} rows); re-record it"
        )


def score(recorded: list[Recorded], questions: Path) -> dict[str, float]:
    """Score recorded answers against the question set, re-running the current checker."""
    if not recorded:
        raise ValueError("there is nothing to score: the recording is empty")
    if len({row.question_id for row in recorded}) != len(recorded):
        raise ValueError("the recording answers the same question more than once")
    known = _questions(questions)
    unknown = sorted({row.question_id for row in recorded} - set(known))
    if unknown:
        raise ValueError(f"recordings name questions the set does not hold: {unknown}")

    answerable = hits = clean = 0
    cited_total = cited_supporting = 0
    unanswerable = abstained = 0
    for row in recorded:
        question = known[row.question_id]
        supporting = set(question["supporting"])
        registry, by_item = _restore(row)
        checked = check_answer(row.answer_text, registry)
        cited = [by_item[citation.item_id] for citation in checked.citations]
        if question["kind"] == "answerable":
            answerable += 1
            hits += bool(supporting.intersection(cited))
            # A failed call, or an answer withheld as NO_SUPPORT, is not a clean answer, whatever
            # the flags say: the NO_SUPPORT text is flagged today only because it is long enough.
            clean += not checked.flags and checked.text != NO_SUPPORT and not row.provider_error
            cited_total += len(cited)
            cited_supporting += sum(external_id in supporting for external_id in cited)
        else:
            unanswerable += 1
            # A provider failure is not an abstention: the system did not choose to say so.
            abstained += checked.text == NO_SUPPORT and not row.provider_error

    # An empty side of the set scores 1.0: no question failed its own metric. The committed set
    # always has both kinds, and its shape is pinned by a test.
    return {
        "support_recall": hits / answerable if answerable else 1.0,
        "citation_precision": cited_supporting / cited_total if cited_total else 1.0,
        "clean_rate": clean / answerable if answerable else 1.0,
        "abstention": abstained / unanswerable if unanswerable else 1.0,
        "mean_tools": sum(len(row.tools) for row in recorded) / len(recorded),
        "mean_tokens": sum(row.tokens for row in recorded) / len(recorded),
        "provider_errors": float(sum(row.provider_error for row in recorded)),
    }


def gate(metrics: dict[str, float], baseline: dict[str, float]) -> list[str]:
    """The gated metrics below their baseline. Replay is deterministic, so any drop bites."""
    # The baseline must carry exactly the gated keys, so a metric cannot enter (or leave) the gate
    # without a visible diff; the metrics may also carry the reported-only keys.
    if set(baseline) != set(GATED):
        raise ValueError(
            f"baseline must hold exactly the gated keys {GATED}, got {tuple(sorted(baseline))}"
        )
    missing = [name for name in GATED if name not in metrics]
    if missing:
        raise ValueError(f"metrics are missing gated keys: {missing}")
    for name in GATED:
        # A non-finite value would pass every comparison by accident; refuse it instead.
        if not math.isfinite(metrics[name]) or not math.isfinite(baseline[name]):
            raise ValueError(f"{name} is not finite: {metrics[name]!r} / {baseline[name]!r}")
    return [
        f"{name} dropped from {baseline[name]:.3f} to {metrics[name]:.3f}"
        for name in GATED
        if metrics[name] < baseline[name]
    ]
