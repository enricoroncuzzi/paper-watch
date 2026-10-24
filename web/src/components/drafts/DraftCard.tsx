"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

import { CheckNotes } from "@/components/drafts/CheckNotes";
import { CopyButton } from "@/components/drafts/CopyButton";
import { formatRome, runDayLabel } from "@/lib/format";
import { linkedinText } from "@/lib/linkedin-text";
import type { Draft, DraftDetail } from "@/lib/types";

// How much of the body the collapsed card shows. Long enough to judge the post, short enough that
// eight drafts still read as a list; "Open" is where the full text and the point of view live.
const PREVIEW_CHARS = 220;

const SMALL =
  "inline-flex h-11 items-center rounded-[2px] border border-rule-2 px-2.5 text-[11.5px] font-medium text-ink hover:border-ink hover:bg-card sm:h-7";
const PLAIN =
  "inline-flex h-11 items-center rounded-[2px] px-2 text-[11.5px] text-ink-2 hover:bg-card-2 hover:text-ink sm:h-7";

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
  const body =
    open || draft.body.length <= PREVIEW_CHARS
      ? draft.body
      : `${draft.body.slice(0, PREVIEW_CHARS).trimEnd()}...`;

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
      if (!response.ok) throw new Error("the detail was refused");
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
        const body = (await response.json().catch(() => null)) as { message?: string } | null;
        setError(body?.message ?? "The draft could not be regenerated. Try again.");
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
    <article
      data-testid="draft-card"
      className="rounded-[2px] border border-rule bg-card"
    >
      <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1.5 border-b border-rule px-3.5 py-2.5">
        <span className="text-[12px] text-ink-2">
          From <b className="font-medium text-ink">{draft.item_title}</b>
        </span>
        <span className="inline-flex h-[21px] items-center rounded-[2px] border border-rule px-2 font-mono text-[10.5px] text-ink-2">
          saved {headline(draft)}
        </span>
        <span className="inline-flex h-[21px] items-center rounded-[2px] border border-rule px-2 text-[10.5px] text-ink-2">
          {draft.chars.toLocaleString("en-GB")} chars
        </span>
        {draft.regenerated_from ? (
          <span className="inline-flex h-[21px] items-center rounded-[2px] border border-rule px-2 text-[10.5px] text-ink-3">
            Regenerated
          </span>
        ) : null}
        <div className="ml-auto flex flex-wrap items-center gap-1.5">
          <CopyButton text={text} />
          <button type="button" onClick={() => void toggleOpen()} className={SMALL}>
            {open ? "Close" : "Open"}
          </button>
          <button
            type="button"
            onClick={() => void regenerate()}
            disabled={regenerating}
            className={`${SMALL} disabled:opacity-60`}
          >
            {regenerating ? "Regenerating..." : "Regenerate"}
          </button>
        </div>
      </div>

      <div className="px-4 py-3.5">
        <p className="mb-2.5 font-serif text-[17.5px] leading-[1.62] text-ink">{draft.hook}</p>
        {body ? (
          <p className="mb-2.5 whitespace-pre-line font-serif text-[17.5px] leading-[1.62] text-ink">
            {body}
          </p>
        ) : null}
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
