import type { FollowupState, RunCounts, RunSummary } from "@/lib/types";

/** Every time on the page is the captain's time: the ingestion runs at 06:00 Rome, and so does he. */
export const ROME = "Europe/Rome";

export function formatRome(iso: string | null | undefined): string {
  if (!iso) return "";
  return new Intl.DateTimeFormat("en-GB", {
    timeZone: ROME,
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(new Date(iso));
}

/** The day a run started, for the Runs rows: short, local and unambiguous. */
export function formatDay(iso: string): string {
  return new Intl.DateTimeFormat("en-GB", {
    timeZone: ROME,
    month: "short",
    day: "numeric",
  }).format(new Date(iso));
}

/** Tokens are read as a magnitude, not a number: "38.4k" is what the mockup shows. */
export function compactTokens(tokens: number): string {
  if (tokens >= 1000) return `${(tokens / 1000).toFixed(1)}k`;
  return String(tokens);
}

/**
 * "Today", "Yesterday" or the date, in Rome. The strip shows the newest run whatever its date, so
 * after a missed ingestion it must not still say "Today": the label is the reader's only cue.
 */
export function runDayLabel(startedAt: string, now: Date = new Date()): string {
  const dayOf = (date: Date) =>
    new Intl.DateTimeFormat("en-CA", {
      timeZone: ROME,
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
    }).format(date);
  const runDay = dayOf(new Date(startedAt));
  const today = dayOf(now);
  if (runDay === today) return "Today";
  // The day before on the calendar, not 24 hours back: a Rome day lasts 23 or 25 hours at a DST change.
  const yesterday = new Date(`${today}T00:00:00Z`);
  yesterday.setUTCDate(yesterday.getUTCDate() - 1);
  if (runDay === yesterday.toISOString().slice(0, 10)) return "Yesterday";
  return formatDay(startedAt);
}

// The profiles the corpus classifies into. The YAML name is a slug; a reader sees words.
const PROFILE_LABELS: Record<string, string> = {
  "agentic-engineering": "Agentic engineering",
  "ai-forensics": "AI forensics",
  "ml-architectures": "ML architectures",
};

/**
 * The topic a card belongs to, as words. A profile added to config/profiles.yaml later still
 * reads as a name instead of a database slug.
 */
export function profileLabel(name: string): string {
  const known = PROFILE_LABELS[name];
  if (known) return known;
  const words = name.replaceAll("-", " ").trim();
  return words ? words.charAt(0).toUpperCase() + words.slice(1) : name;
}

/** The schedule the ingestion unit runs on, kept next to the page's copy of it. */
export const RUN_SCHEDULE_LABEL = "daily 06:00 Europe/Rome · one run at a time";

/** The feeds are whatever was fetched that arXiv was not; the counts carry no feeds key. */
export function feedCount(counts: RunCounts): number {
  return Math.max(0, (counts.fetched ?? 0) - (counts.fetched_arxiv ?? 0));
}

export type Verdict = "healthy" | "unhealthy" | "unknown";

export function verdict(run: Pick<RunSummary, "healthy">): Verdict {
  if (run.healthy === true) return "healthy";
  if (run.healthy === false) return "unhealthy";
  // A run closed without a verdict (a crash closed by the next run) is neither.
  return "unknown";
}

export type StripStatus = "ok" | "warn" | "bad";

/** The strip's dot: bad for an unhealthy run, warn for warnings or an unknown verdict, ok else. */
export function stripStatus(run: RunSummary): StripStatus {
  if (run.healthy === false) return "bad";
  if (run.healthy === null || run.warnings.length > 0) return "warn";
  return "ok";
}

/** "follow-up pass 3 of 40 at 10:30, 12 papers pending", or null when no pass is waiting. */
export function followupLine(followup: FollowupState | null): string | null {
  if (!followup) return null;
  const pending = followup.pending_s2 ?? 0;
  return `follow-up pass ${followup.pass} of ${followup.of} at ${formatRome(
    followup.next_at,
  )}, ${pending} papers pending`;
}
