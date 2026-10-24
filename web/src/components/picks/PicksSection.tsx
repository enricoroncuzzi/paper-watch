"use client";

import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";

import { PovDialog } from "@/components/drafts/PovDialog";
import { CompactPick } from "@/components/picks/CompactPick";
import { LeadPick } from "@/components/picks/LeadPick";
import { RisingShelf } from "@/components/picks/RisingShelf";
import { EmptyState } from "@/components/SectionState";
import type { SuggestionCard, SuggestionsData } from "@/lib/types";

/** The API writes summaries in the background, so a pending card asks the server again this often. */
export const SUMMARY_REFRESH_MS = 20_000;
const MAX_REFRESHES = 6;

function useRefreshWhilePending(pending: boolean, intervalMs: number): void {
  const router = useRouter();
  const refreshes = useRef(0);
  useEffect(() => {
    if (!pending || refreshes.current >= MAX_REFRESHES) return;
    const timer = setInterval(() => {
      refreshes.current += 1;
      router.refresh();
      if (refreshes.current >= MAX_REFRESHES) clearInterval(timer);
    }, intervalMs);
    return () => clearInterval(timer);
  }, [pending, intervalMs, router]);
}

/**
 * The picks and, inside the same section, the Rising shelf. One component owns the dismissed set
 * and the summary poll because both are page state over one list: a dismissal must not wait for a
 * server round trip, and a pending summary should be asked for once, not once per card.
 */
export function PicksSection({
  suggestions,
  refreshMs = SUMMARY_REFRESH_MS,
}: {
  suggestions: SuggestionsData;
  refreshMs?: number;
}) {
  const router = useRouter();
  const [dismissed, setDismissed] = useState<ReadonlySet<string>>(new Set());
  // One dialog for the whole section: the card that opened it, and whether it is early work.
  const [preview, setPreview] = useState<{ card: SuggestionCard; early: boolean } | null>(null);
  const [previewOpen, setPreviewOpen] = useState(false);
  // The element that opened the dialog, so focus returns there when it closes. A click event
  // carries it because WebKit does not focus the clicked button.
  const [opener, setOpener] = useState<HTMLElement | null>(null);

  function openPreview(card: SuggestionCard, early: boolean, element: HTMLElement): void {
    setOpener(element);
    setPreview({ card, early });
    setPreviewOpen(true);
  }
  const picks = suggestions.picks.filter((card) => !dismissed.has(card.item_id));
  const rising = suggestions.rising.filter((card) => !dismissed.has(card.item_id));
  const pending = [...suggestions.picks, ...suggestions.rising].some(
    (card) => card.summary.status === "pending",
  );
  useRefreshWhilePending(pending, refreshMs);

  async function dismiss(itemId: string): Promise<void> {
    // The card leaves the page first; a failed request puts it back rather than lying about it.
    setDismissed((current) => new Set(current).add(itemId));
    try {
      const response = await fetch(`/api/suggestions/${itemId}/events`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ event: "dismissed" }),
      });
      if (!response.ok) throw new Error("dismissal refused");
    } catch {
      setDismissed((current) => {
        const next = new Set(current);
        next.delete(itemId);
        return next;
      });
    }
  }

  const [lead, ...rest] = picks;
  return (
    <section
      id="picks"
      data-testid="picks-section"
      aria-labelledby="picks-title"
      className="mt-[30px] border-t-[1.5px] border-ink pt-3 sm:mt-11"
    >
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <span className="font-mono text-[10px] uppercase tracking-[0.1em] text-accent-text">01</span>
        <h2 id="picks-title" className="font-serif text-[24px] font-normal tracking-[-0.01em] sm:text-[27px]">
          The picks
        </h2>
        <span className="text-[11.5px] text-ink-3">3 to 5 a week, evidence first</span>
      </div>
      <p className="mb-4 mt-1.5 max-w-[64ch] text-[12.5px] text-ink-2 sm:text-[13px]">
        Ranked by relevance and admitted evidence, not by likes. Each one answers four questions in
        30 seconds, and every answer carries the sentence it came from.
      </p>

      {lead ? (
        <>
          <LeadPick
            card={lead}
            onDismiss={dismiss}
            onPreview={(opener) => openPreview(lead, false, opener)}
          />
          <div className="mt-2">
            {rest.map((card, index) => (
              <CompactPick
                key={card.item_id}
                card={card}
                rank={index + 2}
                onDismiss={dismiss}
                onPreview={(opener) => openPreview(card, false, opener)}
              />
            ))}
          </div>
        </>
      ) : (
        <EmptyState
          testId="picks-empty"
          title="Nothing passed the gate in the last three weeks."
          body="Every suggestion is an admitted item with enough stored text to summarise. The next run re-checks the corpus."
        />
      )}

      <RisingShelf
        rising={rising}
        note={suggestions.rising_note}
        onDismiss={dismiss}
        onPreview={(card, opener) => openPreview(card, true, opener)}
      />

      <PovDialog
        key={preview?.card.item_id ?? "closed"}
        card={preview?.card ?? null}
        early={preview?.early ?? false}
        open={previewOpen}
        returnFocus={opener}
        onOpenChange={(next) => {
          if (next) {
            setPreviewOpen(true);
            return;
          }
          // Keep the card for the closing frame, so Radix can return focus to the opener before
          // the component unmounts.
          setPreviewOpen(false);
          window.setTimeout(() => setPreview(null), 0);
        }}
        onSaved={() => router.refresh()}
      />
    </section>
  );
}
