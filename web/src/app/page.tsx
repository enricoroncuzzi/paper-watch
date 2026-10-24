import { Suspense } from "react";

import { requireOwner } from "@/auth";
import { Footer } from "@/components/Footer";
import { Masthead } from "@/components/Masthead";
import { RunsSection } from "@/components/RunsSection";
import { ErrorState, LoadingState } from "@/components/SectionState";
import { TodayStrip } from "@/components/TodayStrip";
import { apiGet } from "@/lib/api";
import type { RunSummary, TodayData } from "@/lib/types";

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

async function RunsSectionData() {
  const result = await apiGet<{ runs: RunSummary[] }>("/runs?limit=7");
  if (!result.ok) {
    return (
      <ErrorState kind={result.kind} testId="runs-error" detail="GET /runs" retryAfterS={result.retryAfterS} />
    );
  }
  return <RunsSection runs={result.data.runs} />;
}

export default async function Home() {
  // Every server component checks the session itself; the proxy only saves an anonymous render.
  const session = await requireOwner();
  return (
    <main className="mx-auto w-full max-w-[1180px] px-5 pb-24 sm:px-7">
      <Masthead session={session} />
      <Suspense fallback={<LoadingState testId="today-loading" lines={1} />}>
        <TodaySection />
      </Suspense>
      <Suspense fallback={<LoadingState testId="runs-loading" lines={4} />}>
        <RunsSectionData />
      </Suspense>
      <Footer />
    </main>
  );
}
