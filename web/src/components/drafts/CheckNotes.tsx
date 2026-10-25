import type { Draft, DraftFlag } from "@/lib/types";

// The reader sees words, never the check's enum: the kind is a code fact, the label is the page's.
const KIND_LABEL: Record<string, string> = {
  number: "Numbers",
  quote: "Quotes",
  link: "Links",
  first_person: "First person",
  attribution: "Attribution",
  preprint_fact: "Preprint claim",
  early_notice: "Early work",
  long_copy: "Copied text",
};
const COUNT_ORDER = [
  "number",
  "quote",
  "link",
  "first_person",
  "attribution",
  "preprint_fact",
  "early_notice",
  "long_copy",
];
// The three factual kinds are always printed, even at 0/0: "Quotes 0/0" tells the reader the
// check ran and found nothing to verify, which is not the same as a check that never happened.
const ALWAYS_PRINTED = new Set(["number", "quote", "link"]);

const ROW = "grid grid-cols-[14px_minmax(0,1fr)] gap-x-[7px] text-[11.5px] leading-[1.45] text-ink-2";

function countsLine(counts: Draft["counts"]): string {
  return COUNT_ORDER.filter(
    (kind) => ALWAYS_PRINTED.has(kind) || (counts[kind]?.[0] ?? 0) > 0,
  )
    .map((kind) => `${KIND_LABEL[kind]} ${counts[kind]?.[0] ?? 0}/${counts[kind]?.[1] ?? 0}`)
    .join(" · ");
}

function CheckIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.4"
      aria-hidden="true"
      className="mt-0.5 h-3 w-3"
    >
      <path d="m5 13 4 4L19 7" />
    </svg>
  );
}

function CrossIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.2"
      aria-hidden="true"
      className="mt-0.5 h-3 w-3"
    >
      <path d="M6 6l12 12M18 6 6 18" />
    </svg>
  );
}

/**
 * The code's verdict on a draft. Clean counts are one "ch ok" line; a failure is one numbered
 * "ch bad" line per flag, in the order the API numbered them, naming the claim it is about.
 */
export function CheckNotes({
  flags,
  counts,
  testId = "draft-checks",
}: {
  flags: DraftFlag[];
  counts: Draft["counts"];
  testId?: string;
}) {
  if (flags.length === 0) {
    return (
      <div data-testid={testId} className="flex flex-col gap-[5px]">
        <div data-testid="draft-check-ok" className={ROW}>
          <span className="text-ok">
            <CheckIcon />
          </span>
          <span>
            <b className="font-medium text-ink">All checks passed.</b> {countsLine(counts)}.
          </span>
        </div>
      </div>
    );
  }
  return (
    <div data-testid={testId} className="flex flex-col gap-[5px]">
      {flags.map((flag) => (
        <div key={flag.n} data-testid="draft-check-note" className={ROW}>
          <span className="text-bad">
            <CrossIcon />
          </span>
          <span>
            <b className="font-medium text-ink">
              {flag.n}. {KIND_LABEL[flag.kind] ?? "Check"}.
            </b>
            <span className="block text-ink">{flag.text}</span>
            <span>{flag.detail}</span>
          </span>
        </div>
      ))}
    </div>
  );
}
