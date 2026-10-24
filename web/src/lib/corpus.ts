import type { CorpusItem, TierChange } from "@/lib/types";

/** The tier tabs, and the `tier` each one sends; the empty value is "All". */
export const TIER_TABS = [
  { value: "", label: "All" },
  { value: "peer_reviewed", label: "Peer reviewed" },
  { value: "vetted_preprint", label: "Vetted preprints" },
  { value: "lab_blog", label: "Lab blogs" },
] as const;

/** The mockup's footer sentence: it is the reader's only cue for why the table is not the corpus. */
export const CORPUS_NOTE = "Only admitted items are searchable.";

/**
 * Stars and upvotes are read as a magnitude, like the mockup's "2.4k ★", never as a five-digit
 * number. Kept here rather than in a component so the table and the item page print it the same.
 */
export function compactCount(value: number): string {
  if (value >= 1000) return `${(value / 1000).toFixed(1).replace(/\.0$/, "")}k`;
  return String(value);
}

/** A published day in the stored UTC date: it is the item's date, not the reader's. */
export function publishedDay(iso: string): string {
  return iso.slice(0, 10);
}

function plural(count: number, word: string): string {
  return `${count} ${word}${count === 1 ? "" : "s"}`;
}

/**
 * The number the row's own rule is about, or "n/a": a reader checking "citations rule" beside a
 * Hugging Face count would be reading two different claims. An empty cell would read as zero.
 */
export function signalLine(item: CorpusItem): string {
  if (item.basis === "hf_upvotes" && item.hf_upvotes !== null && item.hf_upvotes > 0) {
    return `HF ${item.hf_upvotes} \u2191`;
  }
  if (item.basis === "citations" && item.citation_count > 0) {
    return plural(item.citation_count, "cite");
  }
  if (item.hf_upvotes !== null && item.hf_upvotes > 0) return `HF ${item.hf_upvotes} \u2191`;
  if (item.citation_count > 0) return plural(item.citation_count, "cite");
  if (item.github_stars !== null && item.github_stars > 0) {
    return `${compactCount(item.github_stars)} \u2605`;
  }
  return "n/a";
}

// A reader never sees a database enum: these are the words the timeline prints for a rule.
const BASIS_WORD: Record<string, string> = {
  venue: "venue rule",
  hf_upvotes: "upvotes rule",
  citations: "citations rule",
  lab_author: "lab authors",
  feed: "official feed",
  seed: "first sight",
};

export function basisWord(basis: string): string {
  return BASIS_WORD[basis] ?? "admitted";
}

/** One step of the tier timeline: "vetted preprint to peer-reviewed, venue rule". */
export function tierMove(change: TierChange): string {
  const from = change.from_tier ? tierWord(change.from_tier) : "first seen";
  return `${from} to ${tierWord(change.to_tier)}`;
}

// The chips print the same words; here they are needed inside a sentence.
const TIER_WORD: Record<string, string> = {
  peer_reviewed: "peer-reviewed",
  vetted_preprint: "vetted preprint",
  lab_blog: "lab post",
};

export function tierWord(tier: string): string {
  return TIER_WORD[tier] ?? tier.replaceAll("_", " ");
}
