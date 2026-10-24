import { profileLabel } from "@/lib/format";
import type { SuggestionCard } from "@/lib/types";

const NO_VALUE = "n/a";

/** Stars are read as a magnitude, like the mockup's "2.4k ★", never as a five-digit number. */
export function compactCount(value: number): string {
  if (value >= 1000) return `${(value / 1000).toFixed(1).replace(/\.0$/, "")}k`;
  return String(value);
}

/** A published day, in the stored UTC date, because it is the item's date and not the reader's. */
function publishedDay(iso: string): string {
  return iso.slice(0, 10);
}

/**
 * The evidence beside a card. Every missing value prints "n/a": an empty cell reads as zero, and a
 * reader deciding whether to trust a pick needs to know the difference.
 */
export function Facts({
  card,
  compact = false,
  testId,
}: {
  card: SuggestionCard;
  compact?: boolean;
  testId?: string;
}) {
  const rows: [string, string][] = [
    // The API's own label, so a change to the admission rules never leaves the page describing them wrong.
    ["Tier basis", card.basis_label],
    ["Venue", card.venue ?? NO_VALUE],
  ];
  if (!compact) {
    rows.push(["Citations", card.citation_count > 0 ? String(card.citation_count) : NO_VALUE]);
  }
  rows.push([
    "Hugging Face",
    card.hf_upvotes === null ? NO_VALUE : `${card.hf_upvotes} \u2191`,
  ]);
  rows.push([
    "GitHub",
    card.github_stars === null ? NO_VALUE : `${compactCount(card.github_stars)} \u2605`,
  ]);
  rows.push(["Published", publishedDay(card.published_at)]);
  if (!compact) {
    rows.push(["Topic", card.profile ? profileLabel(card.profile) : NO_VALUE]);
    rows.push(["Relevance", card.relevance.toFixed(2)]);
  }
  return (
    <dl data-testid={testId} className="flex flex-col">
      {rows.map(([label, value]) => (
        <div
          key={label}
          className="flex justify-between gap-2.5 border-b border-dotted border-rule-2 py-1 text-[11px] sm:text-[12px]"
        >
          <dt className="whitespace-nowrap text-ink-3">{label}</dt>
          <dd className="text-right tabular-nums text-ink">{value}</dd>
        </div>
      ))}
    </dl>
  );
}
