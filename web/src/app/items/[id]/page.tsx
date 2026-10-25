import { notFound } from "next/navigation";

import { requireOwner } from "@/auth";
import { AskProvider } from "@/components/ask/AskProvider";
import { ItemPage } from "@/components/corpus/ItemPage";
import { Footer } from "@/components/Footer";
import { Masthead } from "@/components/Masthead";
import { ErrorState } from "@/components/SectionState";
import { apiGet } from "@/lib/api";
import type { CorpusItemDetail } from "@/lib/types";

export const dynamic = "force-dynamic";

// A route id that is not a UUID can only be a bad link: the API would answer 422, which reads as a
// server error, so the page says "nothing here" without asking.
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export default async function ItemPageRoute({ params }: { params: Promise<{ id: string }> }) {
  const session = await requireOwner();
  const { id } = await params;
  if (!UUID.test(id)) notFound();
  const result = await apiGet<CorpusItemDetail>(`/items/${encodeURIComponent(id)}`);
  if (!result.ok && result.kind === "not_found") notFound();

  return (
    // The provider wraps the page: "Ask about it" opens the sheet that is rendered beside it.
    <AskProvider>
      <main className="mx-auto w-full max-w-[1180px] px-5 pb-24 sm:px-7">
        <Masthead session={session} />
        {result.ok ? (
          <ItemPage item={result.data} />
        ) : (
          <ErrorState kind={result.kind} testId="item-error" detail="GET /items/{id}" />
        )}
        <Footer />
      </main>
    </AskProvider>
  );
}
