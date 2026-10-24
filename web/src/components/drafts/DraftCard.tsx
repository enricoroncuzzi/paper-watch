"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

import { CheckNotes } from "@/components/drafts/CheckNotes";
import { CopyButton } from "@/components/drafts/CopyButton";
import { draftParagraphs, previewParagraphs } from "@/lib/draft-body";
import { draftFailureMessage } from "@/lib/draft-errors";
import { formatRome, runDayLabel } from "@/lib/format";
import { linkedinText } from "@/lib/linkedin-text";
import type { Draft, DraftDetail } from "@/lib/types";

// The collapsed card's body limit. Long enough to judge the post, short enough that eight drafts
// still read as a list; "Open" is where the full text and the point of view live.
const PREVIEW_CHARS = 220;

const SMALL =
  "inline-flex h-11 items-center rounded-[2px] border border-rule-2 px-2.5 text-[11.5px] font-medium text-ink hover:border-ink hover:bg-card sm:h-7";
// The two chips differ only in font; the sans one is nudged 0.5 px down so their baselines meet.
const CHIP = "inline-flex h-[21px] items-center rounded-[2px] border border-rule px-2 text-[10.5px] leading-none text-ink-2";

function headline(draft: Draft): string {
  // "saved today 12:41" reads as one label; a date keeps its capital ("saved Oct 21 12:41").
  const day = runDayLabel(draft.created_at);
  const label = day === "Today" || day === "Yesterday" ? day.toLowerCase() : day;
  return `${label} ${formatRome(draft.created_at)}`;
}

/**
 * One saved draft: the post, its numbered check notes, and the three actions a reader has.
 * "Open" fetches the detail for the point of view, because the list deliberately does not carry
 * the owner's own words; "Regenerate" asks for a new draft above this one.
 */
export function DraftCard({ draft }: { draft: Draft }) {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [detail, setDetail] = useState<DraftDetail | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [regenerating, setRegenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const text = linkedinText(draft);
  const preview = previewParagraphs(draft.body, PREVIEW_CHARS);
  const paragraphs = open ? draftParagraphs(draft.body) : preview.paragraphs;

  async function toggleOpen() {
    if (open) {
      setOpen(false);
      return;
    }
    setOpen(true);
    if (detail || loadingDetail) return;
    setLoadingDetail(true);
    try {
      const response = await fetch(`/api/drafts/${draft.id}`, { cache: "no-store" });
      if (!response.ok) {
        const body = await response.json().catch(() => null);
        setError(draftFailureMessage(response.status, body));
        return;
      }
      setDetail((await response.json()) as DraftDetail);
    } catch {
      setError("The point of view could not be loaded. Try again.");
    } finally {
      setLoadingDetail(false);
    }
  }

  async function regenerate() {
    if (regenerating) return;
    setRegenerating(true);
    setError(null);
    try {
      const response = await fetch(`/api/drafts/${draft.id}/regenerate`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: "{}",
      });
      if (!response.ok) {
        const body = await response.json().catch(() => null);
        setError(draftFailureMessage(response.status, body));
        return;
      }
      // The saved draft is in the server's list; a refresh puts the newer one above this card.
      router.refresh();
    } catch {
      setError("The draft could not be regenerated. Try again.");
    } finally {
      setRegenerating(false);
    }
  }

  return (
    <article data-testid="draft-card" className="rounded-[2px] border border-rule bg-card">
      <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1.5 border-b border-rule px-3.5 py-2.5">
        <span className="text-[12px] text-ink-2">
          From <b className="font-medium text-ink">{draft.item_title}</b>
        </span>
        <span className={`${CHIP} font-mono`}>saved {headline(draft)}</span>
        <span className={`${CHIP} pt-[0.5px]`}>{draft.chars.toLocaleString("en-GB")} chars</span>
        {draft.regenerated_from ? <span className={CHIP}>Regenerated</span> : null}
        <div className="ml-auto flex flex-wrap items-center gap-1.5">
          <CopyButton text={text} title={draft.item_title} />
          <button
            type="button"
            onClick={() => void toggleOpen()}
            aria-expanded={open}
            aria-label={`${open ? "Close" : "Open"} draft: ${draft.item_title}`}
            className={SMALL}
          >
            {open ? "Close" : "Open"}
          </button>
          <button
            type="button"
            onClick={() => void regenerate()}
            disabled={regenerating}
            aria-label={`Regenerate: ${draft.item_title}`}
            className={`${SMALL} disabled:opacity-60`}
          >
            {regenerating ? "Regenerating..." : "Regenerate"}
          </button>
        </div>
      </div>

      <div className="px-4 py-3.5">
        <p className="mb-2.5 font-serif text-[17.5px] leading-[1.62] text-ink">{draft.hook}</p>
        {paragraphs.map((paragraph, index) => (
          <p
            key={`${index}-${paragraph.slice(0, 24)}`}
            data-testid="draft-paragraph"
            className="mb-2.5 whitespace-pre-line font-serif text-[17.5px] leading-[1.62] text-ink"
          >
            {paragraph}
            {!open && preview.truncated && index === paragraphs.length - 1 ? "..." : ""}
          </p>
        ))}
        {draft.hashtags.length > 0 ? (
          <p className="mb-2 text-[12px] text-ink-2">
            {draft.hashtags.map((tag) => `#${tag}`).join(" ")}
          </p>
        ) : null}
        {draft.source_link ? (
          <p className="break-all font-mono text-[10.5px] text-ink-3">{draft.source_link}</p>
        ) : null}
        {open ? (
          <div className="mt-3 border-t border-rule pt-3">
            <p className="mb-1 text-[9.5px] uppercase tracking-[0.13em] text-ink-3">
              Your point of view
            </p>
            {loadingDetail && !detail ? (
              <p className="text-[12px] text-ink-3">Loading...</p>
            ) : (
              <p className="whitespace-pre-line border-l-2 border-accent pl-3 font-serif text-[15px] leading-[1.55] text-ink-2">
                {detail?.pov || "No point of view was written for this draft."}
              </p>
            )}
          </div>
        ) : null}
      </div>

      <div className="border-t border-rule bg-card-2/55 px-3.5 py-2.5">
        <CheckNotes flags={draft.flags} counts={draft.counts} testId="draft-checks" />
      </div>

      {error ? (
        <p role="alert" className="border-t border-bad/30 px-3.5 py-2 text-[11.5px] text-bad">
          {error}
        </p>
      ) : null}
    </article>
  );
}
