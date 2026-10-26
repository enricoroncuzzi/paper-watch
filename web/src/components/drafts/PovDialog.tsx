"use client";

import * as Dialog from "@radix-ui/react-dialog";
import { useRef, useState } from "react";

import { CheckNotes } from "@/components/drafts/CheckNotes";
import { CopyButton } from "@/components/drafts/CopyButton";
import { Chip, TierChip } from "@/components/picks/Chips";
import { draftParagraphs } from "@/lib/draft-body";
import { draftFailureMessage } from "@/lib/draft-errors";
import { profileLabel } from "@/lib/format";
import { linkedinText } from "@/lib/linkedin-text";
import type { Draft, DraftTarget } from "@/lib/types";

// The API's own bound and the route's maxDuration: the button says it is writing for as long as
// the request can legitimately take, then gives up with a sentence instead of spinning forever.
const POV_MAX_CHARS = 2000;
const GENERATE_TIMEOUT_MS = 300_000;

const SMALL =
  "inline-flex h-11 items-center justify-center rounded-[2px] border border-rule-2 px-2.5 text-[11.5px] font-medium text-ink hover:border-ink hover:bg-card disabled:opacity-60 sm:h-7";
const PRIMARY =
  "inline-flex h-11 items-center justify-center rounded-[2px] border border-accent bg-accent px-3 text-[11.5px] font-medium text-accent-ink hover:border-accent-2 hover:bg-accent-2 disabled:opacity-60 sm:h-8";

function Context({ card }: { card: DraftTarget }) {
  const context = card.summary?.what?.text ?? card.summary?.excerpt ?? null;
  return (
    <>
      <div className="mb-3 flex flex-wrap items-center gap-1.5 text-[10.5px] text-ink-3">
        <TierChip tier={card.tier} />
        {(card.venue ?? card.org) ? <Chip>{card.venue ?? card.org}</Chip> : null}
        {card.profile ? <Chip>{profileLabel(card.profile)}</Chip> : null}
      </div>
      {context ? (
        <p className="mb-4 max-w-[46ch] font-serif text-[15.5px] leading-[1.5] text-ink-2">
          {context}
        </p>
      ) : null}
    </>
  );
}

/**
 * The point-of-view dialog: the item and its 30-second context, the owner's words, then the
 * checked draft. Generating saves, so the footer has no Save button; it has the copy a reader
 * came for. Closing is disabled while a request is in flight, so the same call cannot be started
 * twice into two saved drafts.
 */
export function PovDialog({
  card,
  early,
  open,
  onOpenChange,
  onSaved,
  returnFocus,
}: {
  // A pick's card, or a corpus item page: the dialog reads only the fields a draft needs (slice 7).
  card: DraftTarget | null;
  early: boolean;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSaved: () => void;
  // The button that opened the dialog. WebKit does not focus a clicked button, so the section
  // hands the element over rather than the dialog reading document.activeElement.
  returnFocus: HTMLElement | null;
}) {
  const [pov, setPov] = useState("");
  const [draft, setDraft] = useState<Draft | null>(null);
  const [failure, setFailure] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const input = useRef<HTMLTextAreaElement>(null);

  // A different item is a different draft: the section keys this component by the card's item id,
  // so opening another item mounts a fresh dialog instead of carrying the old text over.

  async function generate() {
    if (!card || pending) return;
    setPending(true);
    setFailure(null);
    setDraft(null);
    const timeout = new AbortController();
    const timer = window.setTimeout(() => timeout.abort(), GENERATE_TIMEOUT_MS);
    try {
      const response = await fetch("/api/drafts", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ item_id: card.item_id, pov, language: "en" }),
        signal: timeout.signal,
      });
      const body = await response.json().catch(() => null);
      if (response.status === 201) {
        setDraft(body as Draft);
        onSaved();
        return;
      }
      setFailure(draftFailureMessage(response.status, body));
    } catch {
      setFailure("The draft request did not complete. Try again, or check the API.");
    } finally {
      window.clearTimeout(timer);
      setPending(false);
    }
  }

  if (!card) return null;

  return (
    <Dialog.Root
      open={open}
      onOpenChange={(next) => {
        // A request in flight is a provider call already spent: closing now would hide it, and
        // reopening would offer a second Generate that saves a duplicate.
        if (!pending) onOpenChange(next);
      }}
    >
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-[94] bg-black/25 backdrop-blur-[5px]" />
        <Dialog.Content
          data-testid="pov-dialog"
          onOpenAutoFocus={(event) => {
            event.preventDefault();
            input.current?.focus();
          }}
          onCloseAutoFocus={(event) => {
            // The card that opened this may have left the picks; Radix's trigger no longer exists.
            // Focus returns to whatever opened the dialog, so a keyboard user does not restart at
            // the top of the page.
            event.preventDefault();
            // A saved draft always takes the heading: the refresh that removes the opener may
            // land after this runs, so a connected opener here is not a safe target.
            if (draft === null && returnFocus?.isConnected) {
              returnFocus.focus();
              return;
            }
            // A saved draft takes its item off the page, so the opener no longer exists: the
            // Drafts heading is the next useful place for a keyboard or screen-reader reader.
            document.getElementById("drafts-title")?.focus();
          }}
          className="fixed inset-0 z-[95] flex h-[100dvh] w-full flex-col overflow-hidden border border-rule bg-card shadow-[var(--shadow)] sm:inset-auto sm:left-1/2 sm:top-1/2 sm:max-h-[min(86dvh,760px)] sm:w-[min(660px,94vw)] sm:-translate-x-1/2 sm:-translate-y-1/2"
        >
          <header className="flex flex-none items-start gap-3 border-b border-rule px-5 py-4">
            <div>
              <p className="mb-1.5 text-[9.5px] uppercase tracking-[0.14em] text-accent-text">
                LinkedIn post preview
              </p>
              <Dialog.Title className="max-w-[34ch] font-serif text-[21px] font-normal leading-[1.25] tracking-[-0.015em]">
                {card.title}
              </Dialog.Title>
              <Dialog.Description className="sr-only">
                Write your own point of view. The draft quotes the item&apos;s stored text, and code
                checks every number, quote and link before saving it.
              </Dialog.Description>
            </div>
            <Dialog.Close
              aria-label="Close the dialog"
              disabled={pending}
              className="ml-auto grid h-11 w-11 flex-none place-items-center rounded-full text-ink-3 hover:bg-card-2 hover:text-ink disabled:opacity-60"
            >
              <svg
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.8"
                aria-hidden="true"
              >
                <path d="m6 6 12 12M18 6 6 18" />
              </svg>
            </Dialog.Close>
          </header>

          <div className="min-h-0 flex-1 overflow-auto px-5 py-4">
            <Context card={card} />

            <label htmlFor="pov-input" className="mb-1.5 block text-[11.5px] text-ink-2">
              Your point of view. This is your voice in the draft; it is never rewritten.
            </label>
            <textarea
              id="pov-input"
              ref={input}
              data-testid="pov-input"
              value={pov}
              maxLength={POV_MAX_CHARS}
              onChange={(event) => setPov(event.target.value)}
              placeholder="What do you actually think? What would you test? What does this change for your own harness?"
              className="min-h-[120px] w-full resize-y rounded-[2px] border border-rule-2 bg-paper px-3 py-2.5 text-[16px] leading-[1.55] text-ink focus:outline-2 focus:outline-offset-1 focus:outline-accent"
            />
            <div className="mt-1.5 flex justify-between gap-2.5 text-[10.5px] text-ink-3">
              <span>Two or three sentences are enough.</span>
              <span>{pov.length.toLocaleString("en-GB")} / 2,000 characters</span>
            </div>

            <div className="mt-3.5 flex flex-wrap items-center gap-2.5">
              <button
                type="button"
                onClick={() => void generate()}
                // After a save the same inputs would only make a duplicate; another draft is
                // "Regenerate" in Drafts, which links the new one to the saved one.
                disabled={pending || draft !== null}
                className={PRIMARY}
              >
                {pending
                  ? "Writing and checking the draft..."
                  : draft
                    ? "Draft saved"
                    : "Generate draft"}
              </button>
              <span className="max-w-[34ch] text-[10.5px] leading-[1.4] text-ink-3">
                The draft is grounded in the item&apos;s stored text. Numbers, quotes and links are
                checked by code afterwards.
              </span>
            </div>

            {failure ? (
              <p
                data-testid="pov-error"
                role="alert"
                className="mt-3.5 border border-bad/40 bg-bad-soft px-3 py-2.5 text-[12px] leading-[1.5] text-bad"
              >
                {failure}
              </p>
            ) : null}

            {draft ? (
              <div data-testid="pov-preview" className="mt-4 border-t border-rule pt-3.5">
                <div className="mb-2 flex flex-wrap items-center gap-2 text-[10px] uppercase tracking-[0.13em] text-ink-3">
                  <span>Draft preview</span>
                  <span
                    data-testid="pov-check-state"
                    className={`normal-case tracking-normal ${
                      draft.flags.length === 0 ? "text-ok" : "text-warn"
                    }`}
                  >
                    {draft.flags.length === 0
                      ? "checked by code"
                      : `${draft.flags.length} ${draft.flags.length === 1 ? "note" : "notes"}`}
                  </span>
                </div>
                <div className="max-w-[34rem] font-serif text-[16.5px] leading-[1.6] text-ink">
                  <p className="mb-2.5">{draft.hook}</p>
                  {draftParagraphs(draft.body).map((paragraph, index) => (
                    <p
                      key={`${index}-${paragraph.slice(0, 24)}`}
                      data-testid="pov-paragraph"
                      className="mb-2.5 whitespace-pre-line"
                    >
                      {paragraph}
                    </p>
                  ))}
                  {draft.hashtags.length > 0 ? (
                    <p className="mb-2 font-sans text-[12px] text-ink-2">
                      {draft.hashtags.map((tag) => `#${tag}`).join(" ")}
                    </p>
                  ) : null}
                  {draft.source_link ? (
                    <p className="break-all font-mono text-[10.5px] font-normal text-ink-3">
                      {draft.source_link}
                    </p>
                  ) : null}
                </div>
                <div className="mt-3">
                  <CheckNotes flags={draft.flags} counts={draft.counts} />
                </div>
              </div>
            ) : null}
          </div>

          <footer className="flex flex-none flex-wrap items-center gap-2 border-t border-rule bg-card-2 px-5 py-2.5">
            {draft ? (
              <span
                data-testid="pov-saved"
                className="inline-flex items-center gap-1.5 text-[10.5px] text-ok"
              >
                <svg
                  width="12"
                  height="12"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2.4"
                  aria-hidden="true"
                >
                  <path d="m5 13 4 4L19 7" />
                </svg>
                Saved to Drafts
              </span>
            ) : (
              <span className="text-[10.5px] text-ink-3">Nothing generated yet</span>
            )}
            {early ? (
              <span className="text-[10.5px] text-ink-3">Early work: the draft says so.</span>
            ) : null}
            <div className="ml-auto flex items-center gap-2">
              <button
                type="button"
                onClick={() => onOpenChange(false)}
                disabled={pending}
                className={SMALL}
              >
                Close
              </button>
              {draft ? <CopyButton text={linkedinText(draft)} title={card.title} /> : null}
            </div>
          </footer>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
