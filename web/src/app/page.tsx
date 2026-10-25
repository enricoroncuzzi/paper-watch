import { Suspense } from "react";

import { requireOwner } from "@/auth";
import { AskProvider } from "@/components/ask/AskProvider";
import { DraftsSection } from "@/components/drafts/DraftsSection";
import { Footer } from "@/components/Footer";
import { Masthead } from "@/components/Masthead";
import { PicksSection } from "@/components/picks/PicksSection";
import { RunsSection } from "@/components/RunsSection";
import { ErrorState, LoadingState } from "@/components/SectionState";
import { TodayStrip } from "@/components/TodayStrip";
import { apiGet } from "@/lib/api";
import type { DraftsData, RunSummary, SuggestionsData, TodayData } from "@/lib/types";

// A private page reads a live corpus: nothing here may be cached or statically rendered.
export const dynamic = "force-dynamic";

async function TodaySection() {
  const result = await apiGet<TodayData>("/today");
  if (!result.ok) {
    return (
      <ErrorState kind={result.kind} testId="today-error" detail="GET /today" retryAfterS={result.retryAfterS} />
    );
  }
  return <TodayStrip today={result.data} />;
}

async function SuggestionsSectionData({ refreshMs }: { refreshMs: number }) {
  const result = await apiGet<SuggestionsData>("/suggestions");
  if (!result.ok) {
    return (
      <ErrorState
        kind={result.kind}
        testId="picks-error"
        detail="GET /suggestions"
        retryAfterS={result.retryAfterS}
      />
    );
  }
  return <PicksSection suggestions={result.data} refreshMs={refreshMs} />;
}

async function DraftsSectionData() {
  const result = await apiGet<DraftsData>("/drafts?limit=50");
  if (!result.ok) {
    return (
      <ErrorState
        kind={result.kind}
        testId="drafts-error"
        detail="GET /drafts"
        retryAfterS={result.retryAfterS}
      />
    );
  }
  return <DraftsSection drafts={result.data.drafts} />;
}

async function RunsSectionData() {
  const result = await apiGet<{ runs: RunSummary[] }>("/runs?limit=7");
  if (!result.ok) {
    return (
      <ErrorState kind={result.kind} testId="runs-error" detail="GET /runs" retryAfterS={result.retryAfterS} />
    );
  }
  return <RunsSection runs={result.data.runs} />;
}

/** How often a pending summary is asked for again; the e2e suite shortens the wait. */
function summaryRefreshMs(): number {
  const configured = Number(process.env.PW_SUMMARY_REFRESH_MS ?? 20_000);
  return Number.isFinite(configured) && configured > 0 ? configured : 20_000;
}

export default async function Home() {
  // Every server component checks the session itself; the proxy only saves an anonymous render.
  const session = await requireOwner();
  return (
    <AskProvider>
      <main className="mx-auto w-full max-w-[1180px] px-5 pb-24 sm:px-7">
        <Masthead session={session} />
        <Suspense fallback={<LoadingState testId="today-loading" lines={1} />}>
          <TodaySection />
        </Suspense>
        <Suspense fallback={<LoadingState testId="picks-loading" lines={3} />}>
          <SuggestionsSectionData refreshMs={summaryRefreshMs()} />
        </Suspense>
        <Suspense fallback={<LoadingState testId="drafts-loading" lines={2} />}>
          <DraftsSectionData />
        </Suspense>
        <Suspense fallback={<LoadingState testId="runs-loading" lines={4} />}>
          <RunsSectionData />
        </Suspense>
        <Footer />
      </main>
    </AskProvider>
  );
}
