import Link from "next/link";

import { EmptyState } from "@/components/SectionState";
import { compactTokens, feedCount, formatDay, RUN_SCHEDULE_LABEL, verdict } from "@/lib/format";
import type { RunSummary } from "@/lib/types";

const VERDICT_STYLE = { healthy: "text-ok", unhealthy: "text-bad", unknown: "text-warn" };

function Cell({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-card px-3 py-2.5">
      <span className="mb-1 block text-[9px] uppercase tracking-[0.13em] text-ink-3">{label}</span>
      <span
        className={`font-serif text-[24px] leading-none tracking-[-0.02em] ${
          value === "0" ? "text-ink-3" : ""
        }`}
      >
        {value}
      </span>
    </div>
  );
}

/** The last run's six counts, its warnings, then the last seven runs as rows. */
export function RunsSection({ runs }: { runs: RunSummary[] }) {
  const latest = runs[0];
  return (
    <section
      id="runs"
      data-testid="runs-section"
      aria-labelledby="runs-title"
      className="mt-[30px] border-t-[1.5px] border-ink pt-3 sm:mt-[52px]"
    >
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <h2 id="runs-title" className="font-serif text-[24px] font-normal tracking-[-0.01em]">
          Runs
        </h2>
        <span className="text-[11px] text-ink-3">{RUN_SCHEDULE_LABEL}</span>
      </div>
      {!latest ? (
        <EmptyState
          testId="runs-empty"
          title="No runs recorded"
          body="The first daily ingestion writes a record here: window, counts, tokens and a health verdict."
        />
      ) : (
        <>
          <div className="mt-3 grid grid-cols-3 gap-px border border-rule bg-rule sm:grid-cols-6">
            <Cell label="arxiv" value={String(latest.counts.fetched_arxiv)} />
            <Cell label="feeds" value={String(feedCount(latest.counts))} />
            <Cell label="stored" value={String(latest.counts.stored)} />
            <Cell label="irrelevant" value={String(latest.counts.irrelevant)} />
            <Cell label="failed" value={String(latest.counts.failed)} />
            <Cell label="tokens" value={compactTokens(latest.tokens)} />
          </div>
          {latest.warnings.map((warning, index) => (
            <div
              key={`${index}-${warning}`}
              className="mt-2.5 flex items-start gap-2 rounded-[2px] bg-warn-soft px-2.5 py-2 text-[11.5px] text-warn"
            >
              <svg
                width="12"
                height="12"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                aria-hidden="true"
                className="mt-[3px] flex-[0_0_12px]"
              >
                <circle cx="12" cy="12" r="9" />
                <path d="M12 8v5M12 17v.01" />
              </svg>
              <span>
                <b className="font-medium">Warning, not a problem:</b> {warning}
              </span>
            </div>
          ))}
          <div className="mt-3.5 text-[12px]">
            {runs.slice(0, 7).map((run) => (
              <Link
                key={run.id}
                href={`/runs/${run.id}`}
                className="grid grid-cols-[64px_1fr_auto] sm:grid-cols-[76px_1fr_auto] items-center gap-3 border-b border-rule px-0.5 py-2 transition-colors hover:bg-card-2/60"
              >
                <span className="font-mono text-[10.5px] text-ink-2">
                  {formatDay(run.started_at)}
                </span>
                <span className="flex flex-wrap gap-x-2.5 text-[11px] text-ink-3">
                  <span>
                    <b className="font-medium text-ink-2">{run.counts.fetched_arxiv}</b> arxiv
                  </span>
                  <span>
                    <b className="font-medium text-ink-2">{run.counts.new}</b> new
                  </span>
                  <span>
                    <b className="font-medium text-ink-2">{run.counts.stored}</b> stored
                  </span>
                  <span>
                    <b className="font-medium text-ink-2">{compactTokens(run.tokens)}</b> tokens
                  </span>
                  {run.counts.failed > 0 ? <span>{run.counts.failed} failed</span> : null}
                </span>
                <span
                  className={`flex items-center gap-1.5 text-[9.5px] uppercase tracking-[0.1em] ${VERDICT_STYLE[verdict(run)]}`}
                >
                  <span aria-hidden="true" className="h-[5px] w-[5px] rounded-full bg-current" />
                  {verdict(run)}
                </span>
              </Link>
            ))}
          </div>
        </>
      )}
    </section>
  );
}
