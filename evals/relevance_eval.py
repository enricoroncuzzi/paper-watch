"""Relevance classifier eval: calibrate each profile's threshold, then score it on held-out data.

Run from the repository root, with the configured local embedding model:

    uv run python -m evals.relevance_eval           # print thresholds and scores
    uv run python -m evals.relevance_eval --write   # also update profiles.yaml and the baseline

The labelled abstracts are split in two by a hash of their arXiv id. Thresholds are chosen on
the validation half only; the test half then measures them, as the classifier will meet new
papers. Routing goes through the production `EmbeddingClassifier`, so the eval scores the
classifier itself, not a copy of its logic.
"""

import argparse
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import yaml

from paperwatch.config import get_settings
from paperwatch.domain import Profile, SourceItem, load_profiles
from paperwatch.llm.classifier import Classifier, EmbeddingClassifier, FakeClassifier
from paperwatch.llm.embedder import Embedder, LocalEmbedder, embedding_text, make_local_model

DATA = Path("evals/data/relevance/abstracts.jsonl")
PROFILES = Path("config/profiles.yaml")
BASELINE = Path("evals/relevance_baseline.json")
# The label of an abstract that fits no profile: the classifier should give it no profile.
NONE = "none"
# The scores the gate holds: precision of what is admitted, and the macro F1 balance.
GATED = ("macro_f1", "admitted_precision")


@dataclass(frozen=True)
class Example:
    """One hand-labelled abstract: a profile name, or NONE."""

    arxiv_id: str
    title: str
    abstract: str
    label: str

    @property
    def split(self) -> str:
        # By a hash of the id, not by file order, so an example keeps its split as the set grows.
        digest = hashlib.sha256(self.arxiv_id.encode()).digest()
        return "validation" if digest[0] % 2 == 0 else "test"

    def item(self) -> SourceItem:
        return SourceItem(
            "arxiv",
            self.arxiv_id,
            self.title,
            (),
            self.abstract,
            f"https://arxiv.org/abs/{self.arxiv_id}",
            datetime(2026, 1, 1, tzinfo=UTC),
        )


@dataclass(frozen=True)
class ProfileScore:
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class Scores:
    macro_precision: float
    macro_recall: float
    macro_f1: float
    # Of the abstracts given any profile, the share given the right one: what admission trusts.
    admitted_precision: float
    # The one-of-four choice, NONE included.
    routing_accuracy: float
    profiles: dict[str, ProfileScore]


@dataclass(frozen=True)
class Calibration:
    model: str
    fingerprint: str
    thresholds: dict[str, float]
    validation: Scores
    test: Scores
    # The thresholds profiles.yaml held before this run, and the keyword classifier, on test.
    previous_test: Scores
    keyword_test: Scores
    n_validation: int
    n_test: int


def load_examples(path: Path, profile_names: list[str]) -> list[Example]:
    allowed = {*profile_names, NONE}
    examples: list[Example] = []
    seen: set[str] = set()
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        example = Example(row["arxiv_id"], row["title"], row["abstract"], row["label"])
        if example.label not in allowed:
            raise ValueError(f"{path}:{n}: unknown label {example.label!r}")
        if example.arxiv_id in seen:
            raise ValueError(f"{path}:{n}: duplicate arxiv_id {example.arxiv_id}")
        if not (example.title.strip() and example.abstract.strip()):
            raise ValueError(f"{path}:{n}: blank title or abstract")
        seen.add(example.arxiv_id)
        examples.append(example)
    return examples


class _Memo:
    """Embeds each distinct text once: calibration and routing reuse the same vectors."""

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._cache: dict[str, list[float]] = {}

    def embed(self, texts: list[str]) -> list[list[float]]:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            self._cache.update(zip(missing, self._embedder.embed(missing), strict=True))
        return [self._cache[t] for t in texts]


def _cosine(u: list[float], v: list[float]) -> float:
    dot = sum(a * b for a, b in zip(u, v, strict=True))
    norms = math.sqrt(sum(a * a for a in u)) * math.sqrt(sum(b * b for b in v))
    return dot / norms if norms else 0.0


def best_threshold(scores: list[float], positives: list[bool]) -> float:
    # Try each observed score as the cut (admit at or above it) and keep the best F1. Ascending
    # with >=, so on a tie the higher cut wins: fewer admitted, more precise.
    if not any(positives):
        raise ValueError("no positive example to calibrate on")
    ordered = sorted(set(scores))
    best_f1, best = -1.0, 0
    for i, cut in enumerate(ordered):
        admitted = [p for s, p in zip(scores, positives, strict=True) if s >= cut]
        # F1 = 2TP / (2TP + FP + FN) = 2TP / (admitted + actual positives).
        f1 = 2 * sum(admitted) / (len(admitted) + sum(positives))
        if f1 >= best_f1:
            best_f1, best = f1, i
    # Halfway down to the highest rejected score: the most room on both sides for new papers.
    below = ordered[best - 1] if best else ordered[best]
    return min(0.999, max(0.001, round((below + ordered[best]) / 2, 3)))


def calibrate(
    embedder: Embedder, profiles: list[Profile], examples: list[Example]
) -> dict[str, float]:
    # One-vs-rest per profile: an abstract is positive for the profile it is labelled with.
    profile_vectors = embedder.embed([p.description for p in profiles])
    item_vectors = embedder.embed([embedding_text(e.item()) for e in examples])
    return {
        p.name: best_threshold(
            [_cosine(v, pv) for v in item_vectors], [e.label == p.name for e in examples]
        )
        for p, pv in zip(profiles, profile_vectors, strict=True)
    }


def score(classifier: Classifier, examples: list[Example], names: list[str]) -> Scores:
    predicted = [classifier.classify(e.item()).profile or NONE for e in examples]
    gold = [e.label for e in examples]
    per: dict[str, ProfileScore] = {}
    for name in names:
        tp = sum(1 for p, g in zip(predicted, gold, strict=True) if p == g == name)
        n_predicted, support = predicted.count(name), gold.count(name)
        precision = tp / n_predicted if n_predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * tp / (n_predicted + support) if n_predicted + support else 0.0
        per[name] = ProfileScore(precision, recall, f1, support)
    admitted = [(p, g) for p, g in zip(predicted, gold, strict=True) if p != NONE]
    return Scores(
        macro_precision=sum(s.precision for s in per.values()) / len(per),
        macro_recall=sum(s.recall for s in per.values()) / len(per),
        macro_f1=sum(s.f1 for s in per.values()) / len(per),
        admitted_precision=sum(p == g for p, g in admitted) / len(admitted) if admitted else 0.0,
        routing_accuracy=sum(p == g for p, g in zip(predicted, gold, strict=True)) / len(gold),
        profiles=per,
    )


def fingerprint(model: str, profiles: list[Profile], data: Path) -> str:
    # Everything a threshold depends on: the model, each description, and the labelled set.
    # The baseline records it, so a stale threshold fails the offline tests.
    payload = {
        "model": model,
        "descriptions": {p.name: p.description for p in profiles},
        "data": hashlib.sha256(data.read_bytes()).hexdigest(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _with_thresholds(profiles: list[Profile], thresholds: dict[str, float]) -> list[Profile]:
    return [
        Profile(p.name, p.weight, p.categories, p.terms, p.description, thresholds[p.name])
        for p in profiles
    ]


def evaluate(embedder: Embedder, model: str, profiles: list[Profile], data: Path) -> Calibration:
    names = [p.name for p in profiles]
    examples = load_examples(data, names)
    validation = [e for e in examples if e.split == "validation"]
    test = [e for e in examples if e.split == "test"]
    memo = _Memo(embedder)
    thresholds = calibrate(memo, profiles, validation)
    calibrated = _with_thresholds(profiles, thresholds)
    return Calibration(
        model=model,
        fingerprint=fingerprint(model, profiles, data),
        thresholds=thresholds,
        validation=score(EmbeddingClassifier(memo, calibrated), validation, names),
        test=score(EmbeddingClassifier(memo, calibrated), test, names),
        previous_test=score(EmbeddingClassifier(memo, profiles), test, names),
        keyword_test=score(FakeClassifier(profiles), test, names),
        n_validation=len(validation),
        n_test=len(test),
    )


def gate(scores: Scores, baseline: dict[str, Any], tolerance: float = 0.03) -> list[str]:
    return [
        f"{name} dropped from {baseline[name]:.3f} to {getattr(scores, name):.3f}"
        for name in GATED
        if getattr(scores, name) < baseline[name] - tolerance
    ]


_THRESHOLD_LINE = re.compile(r"^(\s*relevance_threshold:\s*)\S+", re.MULTILINE)


def write_thresholds(path: Path, thresholds: dict[str, float]) -> None:
    # Edit only the threshold lines, so the file's comments and layout survive (a YAML dump
    # would drop them), then check the result parses to exactly the new values before saving.
    text = path.read_text()
    names = [p["name"] for p in yaml.safe_load(text)["profiles"]]
    if len(_THRESHOLD_LINE.findall(text)) != len(names):
        raise ValueError(f"{path}: expected one relevance_threshold line per profile")
    values = iter(thresholds[n] for n in names)
    new = _THRESHOLD_LINE.sub(lambda m: f"{m.group(1)}{next(values)}", text)
    written = {p["name"]: p.get("relevance_threshold") for p in yaml.safe_load(new)["profiles"]}
    if written != thresholds:
        raise ValueError(f"{path}: the relevance_threshold lines are not in profile order")
    path.write_text(new)


def _rounded(value: Any) -> Any:
    # Four decimals keep the committed baseline readable and its diffs small.
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {k: _rounded(v) for k, v in value.items()}
    return value


def _row(label: str, s: Scores) -> str:
    return (
        f"{label:<22} macro P {s.macro_precision:.3f}  R {s.macro_recall:.3f}  "
        f"F1 {s.macro_f1:.3f}  admitted precision {s.admitted_precision:.3f}  "
        f"routing accuracy {s.routing_accuracy:.3f}"
    )


def format_report(c: Calibration, profiles: list[Profile]) -> str:
    lines = [
        f"model {c.model}: {c.n_validation} validation and {c.n_test} test abstracts",
        "",
        f"{'profile':<22}{'threshold':>10}{'previous':>10}{'test P':>8}{'test R':>8}"
        f"{'test F1':>8}{'n':>4}",
    ]
    for p in profiles:
        s = c.test.profiles[p.name]
        lines.append(
            f"{p.name:<22}{c.thresholds[p.name]:>10.3f}{p.relevance_threshold:>10.3f}"
            f"{s.precision:>8.3f}{s.recall:>8.3f}{s.f1:>8.3f}{s.support:>4}"
        )
    lines += [
        "",
        _row("validation", c.validation),
        _row("test", c.test),
        _row("test, previous", c.previous_test),
        _row("test, keyword", c.keyword_test),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evals.relevance_eval",
        description="Calibrate the relevance thresholds and score them on held-out abstracts.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help=f"write the thresholds into {PROFILES} and the scores into {BASELINE}",
    )
    args = parser.parse_args(argv)
    settings = get_settings()
    # Always the local model: the thresholds are cosines of the model that will classify.
    embedder = LocalEmbedder(make_local_model(settings), settings.local_embedding_batch)
    profiles = load_profiles(PROFILES)
    calibration = evaluate(embedder, settings.local_embedding_model, profiles, DATA)
    print(format_report(calibration, profiles))
    if args.write:
        write_thresholds(PROFILES, calibration.thresholds)
        record = {"calibrated_on": date.today().isoformat(), **_rounded(asdict(calibration))}
        BASELINE.write_text(json.dumps(record, indent=2) + "\n")
        print(f"\nwrote {PROFILES} and {BASELINE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
