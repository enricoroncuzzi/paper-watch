import { followupLine, formatRome, runDayLabel, stripStatus, verdict } from "@/lib/format";
import type { TodayData } from "@/lib/types";

const DOT_COLOR = {
  ok: "bg-ok shadow-[0_0_0_3px_var(--ok-soft)]",
  warn: "bg-warn shadow-[0_0_0_3px_var(--warn-soft)]",
  bad: "bg-bad shadow-[0_0_0_3px_var(--bad-soft)]",
};
const VERDICT_COLOR = { ok: "text-ok", warn: "text-warn", bad: "text-bad" };

/** One line about the last run, the way the mockup's runline reads it. */
export function TodayStrip({ today }: { today: TodayData }) {
  const run = today.run;
  const textOnly = today.search.mode === "text_only" && today.search.reason === "ingestion";

  if (!run) {
    return (
      <div
        id="today"
        data-testid="today-strip"
        className="flex flex-wrap items-center gap-x-2.5 gap-y-1 border-b border-rule py-2.5 text-[12px] text-ink-2"
      >
        <span className="text-[9.5px] uppercase tracking-[0.14em] text-ink-3">Today</span>
        <span>No run recorded yet</span>
        <a href="#runs" className="ml-auto flex items-center gap-1 text-[11.5px] text-accent-text">
          Runs <span aria-hidden="true">›</span>
        </a>
      </div>
    );
  }

  const status = stripStatus(run);
  const state = verdict(run);
  const followup = followupLine(run.followup);
  const problems = run.problems.length;

  return (
    <div
      id="today"
      data-testid="today-strip"
      className="flex flex-wrap items-center gap-x-2.5 gap-y-1 border-b border-rule py-2.5 text-[12px] text-ink-2"
    >
      <span
        data-testid="run-day"
        className={`text-[9.5px] uppercase tracking-[0.14em] ${
          runDayLabel(run.started_at) === "Today" ? "text-ink-3" : "text-warn"
        }`}
      >
        {runDayLabel(run.started_at)}
      </span>
      <span aria-hidden="true" className={`h-1.5 w-1.5 rounded-full ${DOT_COLOR[status]}`} />
      <b className="font-medium text-ink">{formatRome(run.started_at)}</b>
      <span className="text-[10.5px] text-ink-3">Europe/Rome</span>
      <span className="text-rule-2">·</span>
      <span className={`font-medium ${VERDICT_COLOR[status]}`}>{state}</span>
      {run.healthy === false && problems > 0 ? (
        <span className="text-bad">· {run.problems[0]}</span>
      ) : null}
      <span className="text-rule-2 max-[760px]:hidden">·</span>
      <span className="max-[760px]:hidden">
        <b className="font-medium text-ink">{run.counts.stored}</b> stored
      </span>
      <span className="max-[760px]:hidden">
        <b className="font-medium text-ink">{run.counts.fetched}</b> fetched
      </span>
      {run.warnings.length > 0 ? (
        <span className="text-warn max-[760px]:hidden">
          · {run.warnings.length} warning{run.warnings.length === 1 ? "" : "s"}
        </span>
      ) : null}
      {followup ? <span className="text-ink-3 max-[760px]:hidden">· {followup}</span> : null}
      {textOnly ? (
        <span className="text-accent-text">· keyword search while the corpus updates</span>
      ) : null}
      <a href="#runs" className="ml-auto flex items-center gap-1 text-[11.5px] text-accent-text">
        Runs <span aria-hidden="true">›</span>
      </a>
    </div>
  );
}
