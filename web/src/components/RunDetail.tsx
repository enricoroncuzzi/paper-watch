import Link from "next/link";

import { compactTokens, feedCount, followupLine, formatDay, formatRome, verdict } from "@/lib/format";
import type { RunDetailData } from "@/lib/types";

const VERDICT_STYLE = { healthy: "text-ok", unhealthy: "text-bad", unknown: "text-warn" };

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-[9px] uppercase tracking-[0.13em] text-ink-3">{label}</dt>
      <dd className="font-serif text-[22px] leading-tight">{value}</dd>
    </div>
  );
}

/** One run: its window, counts, problems, warnings, follow-up state and dead letters. */
export function RunDetail({ run }: { run: RunDetailData }) {
  const state = verdict(run);
  const followup = followupLine(run.followup);
  return (
    <section data-testid="run-detail" className="mt-6">
      <Link href="/#runs" className="font-mono text-[10.5px] text-accent-text">
        ← Runs
      </Link>
      <div className="mt-2 flex flex-wrap items-baseline gap-x-3 gap-y-1 border-b border-rule pb-3">
        <h1 className="font-serif text-[28px] font-normal tracking-[-0.015em]">
          {formatRome(run.started_at)} Europe/Rome
        </h1>
        <span
          className={`flex items-center gap-1.5 text-[9.5px] uppercase tracking-[0.1em] ${VERDICT_STYLE[state]}`}
        >
          <span aria-hidden="true" className="h-[5px] w-[5px] rounded-full bg-current" />
          {state}
        </span>
        <span className="font-mono text-[10.5px] text-ink-3">{run.status}</span>
      </div>

      <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-4">
        <Stat label="arxiv" value={String(run.counts.fetched_arxiv)} />
        <Stat label="feeds" value={String(feedCount(run.counts))} />
        <Stat label="stored" value={String(run.counts.stored)} />
        <Stat label="irrelevant" value={String(run.counts.irrelevant)} />
        <Stat label="failed" value={String(run.counts.failed)} />
        <Stat label="admitted late" value={String(run.counts.admitted_late)} />
        <Stat label="follow-up passes" value={String(run.counts.followup_passes)} />
        <Stat label="tokens" value={compactTokens(run.tokens)} />
      </dl>

      <div className="mt-4 border-t border-rule pt-3 text-[12px] text-ink-2">
        <p>
          <span className="text-ink-3">window </span>
          {formatDay(run.window_start)} {formatRome(run.window_start)} to {formatDay(run.window_end)}{" "}
          {formatRome(run.window_end)} Europe/Rome
        </p>
        {followup ? <p className="mt-1 text-ink-3">{followup}</p> : null}
      </div>

      {run.problems.length > 0 ? (
        <div className="mt-3 border border-bad/40 bg-bad-soft px-4 py-3">
          <h2 className="mb-1 font-serif text-[18px] font-normal text-bad">Problems</h2>
          <ul className="list-disc pl-4 text-[12.5px] leading-6 text-ink-2">
            {run.problems.map((problem) => (
              <li key={problem}>{problem}</li>
            ))}
          </ul>
        </div>
      ) : null}

      {run.warnings.map((warning, index) => (
        <div
          key={`${index}-${warning}`}
          className="mt-2 rounded-[2px] bg-warn-soft px-2.5 py-2 text-[11.5px] text-warn"
        >
          <b className="font-medium">Warning, not a problem:</b> {warning}
        </div>
      ))}

      <h2 className="mt-6 font-serif text-[20px] font-normal">Dead letters</h2>
      {run.dead_letters.length === 0 ? (
        <p className="mt-1 text-[12px] text-ink-3">None: every source entry was handled.</p>
      ) : (
        <div className="mt-2 border border-rule bg-card">
          {run.dead_letters.map((letter) => (
            <div
              key={`${letter.source}-${letter.ref}`}
              className="grid gap-1 border-b border-rule px-3 py-2.5 last:border-b-0 sm:grid-cols-[110px_1fr_auto] sm:gap-3"
            >
              <span className="font-mono text-[10px] uppercase tracking-[0.08em] text-ink-3">
                {letter.source}
              </span>
              <span className="min-w-0">
                <span className="block break-all font-mono text-[10.5px] text-ink-2">
                  {letter.ref}
                </span>
                <span className="mt-0.5 block break-words text-[11.5px] text-ink-2">
                  {letter.error}
                </span>
              </span>
              <span className="text-[10px] text-ink-3">
                {letter.attempts} attempt{letter.attempts === 1 ? "" : "s"} ·{" "}
                {letter.resolved_at ? "resolved" : "open"}
              </span>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
