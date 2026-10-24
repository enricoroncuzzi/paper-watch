"use client";

import { useAskLauncher } from "@/components/ask/AskProvider";
import { safeHref } from "@/lib/safe-href";
import type { SuggestionCard } from "@/lib/types";

const BUTTON =
  "inline-flex h-11 items-center justify-center gap-1.5 rounded-[2px] border px-2.5 text-[11.5px] font-medium sm:h-7";
const OUTLINE = `${BUTTON} border-rule-2 text-ink hover:border-ink hover:bg-card`;
const PRIMARY = `${BUTTON} border-accent bg-accent text-accent-ink hover:border-accent-2 hover:bg-accent-2`;

/** "opened" is the one event the API counts from the reader's own click, so it is sent as a beacon. */
function recordOpened(itemId: string): void {
  if (typeof navigator === "undefined" || !navigator.sendBeacon) return;
  // A string body keeps the beacon visible to a test's request listener on every browser; the
  // route reads the raw text and parses it, so the content type does not matter.
  navigator.sendBeacon(`/api/suggestions/${itemId}/events`, JSON.stringify({ event: "opened" }));
}

/** The way to the paper. A stored URL that fails safeHref gets no button at all, only its title. */
export function ReadOriginal({ card }: { card: SuggestionCard }) {
  const href = safeHref(card.url);
  if (!href) return null;
  return (
    <a
      href={href}
      target="_blank"
      rel="noreferrer noopener"
      onClick={() => recordOpened(card.item_id)}
      // A middle click opens the paper too, and must count as an opening like the left one.
      onAuxClick={(event) => {
        if (event.button === 1) recordOpened(card.item_id);
      }}
      aria-label={`Read the original: ${card.title}`}
      className={OUTLINE}
    >
      Read the original
      <svg
        width="11"
        height="11"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        aria-hidden="true"
      >
        <path d="M7 17 17 7M9 7h8v8" />
      </svg>
    </a>
  );
}

/**
 * The way into the draft dialog. The card owns which item is previewed, so this button only says
 * which one it is for: seven of them sit on one page.
 */
export function LinkedInPreview({
  card,
  onPreview,
}: {
  card: SuggestionCard;
  onPreview: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onPreview}
      aria-label={`LinkedIn post preview: ${card.title}`}
      className={PRIMARY}
    >
      LinkedIn post preview
    </button>
  );
}

/** Opens the reading sheet with this item, so the API registers it as S1 before the question. */
export function AskAbout({ card }: { card: SuggestionCard }) {
  const { askAbout } = useAskLauncher();
  return (
    <button
      type="button"
      onClick={() => askAbout({ id: card.item_id, title: card.title })}
      aria-label={`Ask about it: ${card.title}`}
      className={PRIMARY}
    >
      Ask about it
    </button>
  );
}
