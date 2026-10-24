"use client";

import { useState } from "react";

const STYLE =
  "inline-flex h-11 items-center justify-center rounded-[2px] border border-rule-2 px-2.5 text-[11.5px] font-medium text-ink hover:border-ink hover:bg-card sm:h-7";

/**
 * The one way a draft leaves this page: the exact plain text on the clipboard, never markdown and
 * never HTML. A browser that refuses the clipboard leaves the text on screen to select by hand.
 * The confirmation is a polite live region, so a screen reader hears it without losing the button.
 */
export function CopyButton({
  text,
  title,
  testId,
}: {
  text: string;
  title: string;
  testId?: string;
}) {
  const [copied, setCopied] = useState(false);

  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      // No permission, or an insecure context: the button stays, the reader can still copy.
    }
  }

  return (
    <>
      <button
        type="button"
        onClick={() => void copy()}
        data-testid={testId}
        aria-label={`Copy for LinkedIn: ${title}`}
        className={STYLE}
      >
        {copied ? "Copied" : "Copy for LinkedIn"}
      </button>
      <span aria-live="polite" className="sr-only">
        {copied ? "Copied to the clipboard" : ""}
      </span>
    </>
  );
}
