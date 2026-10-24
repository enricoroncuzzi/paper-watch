"use client";

/** The floating way in. The sheet itself lives in the provider, so a pick can open it too. */
export function AskPill({ onOpen }: { onOpen: () => void }) {
  return (
    <button
      type="button"
      onClick={onOpen}
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
      {/* The label takes the free space, so the send circle stays at the end when the hint is hidden. */}
      <span className="mr-auto">Ask the corpus…</span>
      <span className="hidden font-mono text-[10px] text-ink-3 sm:inline">⌘↵</span>
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
  );
}
