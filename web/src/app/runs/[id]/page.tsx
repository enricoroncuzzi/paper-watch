import { notFound } from "next/navigation";

import { requireOwner } from "@/auth";
import { Footer } from "@/components/Footer";
import { AskPill } from "@/components/ask/AskPill";
import { Masthead } from "@/components/Masthead";
import { RunDetail } from "@/components/RunDetail";
import { ErrorState } from "@/components/SectionState";
import { apiGet } from "@/lib/api";
import type { RunDetailData } from "@/lib/types";

export const dynamic = "force-dynamic";

export default async function RunPage({ params }: { params: Promise<{ id: string }> }) {
  const session = await requireOwner();
  const { id } = await params;
  const result = await apiGet<RunDetailData>(`/runs/${encodeURIComponent(id)}`);
  if (!result.ok && result.kind === "not_found") notFound();

  return (
    <main className="mx-auto w-full max-w-[1180px] px-5 pb-24 sm:px-7">
      <Masthead session={session} />
      {result.ok ? (
        <RunDetail run={result.data} />
      ) : (
        <ErrorState kind={result.kind} testId="run-error" detail="GET /runs/{id}" />
      )}
      <Footer />
      <AskPill />
    </main>
  );
}
