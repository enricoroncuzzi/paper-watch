"use client";

import { useEffect, useState } from "react";

import { AskSheet } from "@/components/ask/AskSheet";

/**
 * The floating way in. Clicking it, or Cmd+Enter / Ctrl+Enter from anywhere on the page, opens the
 * reading sheet; the hint shows the shortcut on a keyboard and is hidden on touch.
 */
export function AskPill() {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
        event.preventDefault();
        setOpen(true);
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="fixed bottom-[max(16px,env(safe-area-inset-bottom))] left-1/2 z-[60] flex h-11 w-[min(520px,90vw)] -translate-x-1/2 items-center gap-2.5 rounded-full border border-rule-2 bg-card px-4 text-left text-[13px] text-ink-2 shadow-[var(--shadow)] transition-colors hover:border-accent hover:text-ink"
      >
        <svg
          width="15"
          height="15"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.8"
          aria-hidden="true"
        >
          <path d="M4 5.5h16v10H9l-5 4v-14Z" />
        </svg>
        <span>Ask the corpus…</span>
        <span className="ml-auto hidden font-mono text-[10px] text-ink-3 sm:inline">⌘↵</span>
        <span className="grid h-7 w-7 flex-none place-items-center rounded-full bg-accent text-accent-ink">
          <svg
            width="13"
            height="13"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            aria-hidden="true"
          >
            <path d="M12 19V5M6 11l6-6 6 6" />
          </svg>
        </span>
      </button>
      <AskSheet open={open} onOpenChange={setOpen} />
    </>
  );
}
