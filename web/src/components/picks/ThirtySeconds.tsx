import type { SummaryData, SummaryFieldData } from "@/lib/types";

const CAPTION = {
  checked: "quotes from the stored text",
  excerpt: "the abstract's own words",
  pending: "summary on its way",
} as const;

const ROWS: [string, keyof Pick<SummaryData, "what" | "why" | "result" | "catch">][] = [
  ["What they did", "what"],
  ["Why it matters", "why"],
  ["The result", "result"],
  ["The catch", "catch"],
];

function Field({ label, field }: { label: string; field: SummaryFieldData }) {
  return (
    <div className="grid grid-cols-1 gap-0.5 border-t border-rule py-2 first:border-t-0 first:pt-0 sm:grid-cols-[88px_minmax(0,1fr)] sm:gap-3.5">
      <span className="text-[9px] uppercase tracking-[0.13em] text-ink-3 sm:pt-[3px]">
        {label}
        {field.interpretation ? (
          // The reader must be able to tell the authors' words from the summariser's opinion.
          <span
            data-testid="interpretation-label"
            className="mt-0.5 block text-[8.5px] normal-case tracking-normal text-ink-3"
          >
            interpretation
          </span>
        ) : null}
      </span>
      <p className="text-[13.5px] leading-[1.55] text-ink">
        {field.text}
        {field.quote ? (
          <span className="mt-1.5 block border-l-2 border-rule-2 pl-2.5 font-serif text-[13px] italic leading-[1.45] text-ink-2">
            &ldquo;{field.quote}&rdquo;
          </span>
        ) : null}
      </p>
    </div>
  );
}

/**
 * The four answers a reader gets in 30 seconds. A checked summary shows only what the checker
 * accepted; a pending or failed one shows the abstract's own sentences, never a claim of its own.
 */
export function ThirtySeconds({
  summary,
  compact = false,
}: {
  summary: SummaryData;
  compact?: boolean;
}) {
  const checked = summary.status === "checked";
  return (
    <div
      data-testid="thirty-seconds"
      className={`mt-3.5 rounded-[2px] border border-rule bg-card px-3.5 py-3 ${
        compact ? "sm:px-3.5 sm:py-3" : ""
      }`}
    >
      <div className="mb-3 flex flex-wrap items-center gap-x-2.5 gap-y-1 text-[9.5px] uppercase tracking-[0.15em] text-ink-3">
        <b className="font-semibold tracking-[0.12em] text-ink">In 30 seconds</b>
        <span className="ml-auto font-mono text-[9.5px] normal-case tracking-normal">
          {CAPTION[summary.status]}
        </span>
      </div>
      {checked ? (
        ROWS.map(([label, key]) => {
          const field = summary[key];
          return field ? <Field key={label} label={label} field={field} /> : null;
        })
      ) : (
        <div className="grid grid-cols-1 gap-0.5 sm:grid-cols-[88px_minmax(0,1fr)] sm:gap-3.5">
          <span className="text-[9px] uppercase tracking-[0.13em] text-ink-3 sm:pt-[3px]">
            Abstract excerpt
          </span>
          <p className="text-[13.5px] leading-[1.55] text-ink">{summary.excerpt}</p>
        </div>
      )}
    </div>
  );
}
