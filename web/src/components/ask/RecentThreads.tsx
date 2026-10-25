"use client";

import { useEffect, useRef, useState } from "react";

import { formatRome } from "@/lib/format";
import type { ThreadSummary } from "@/lib/use-ask";

/**
 * The Recent menu: the stored conversations, newest first, read when it opens.
 *
 * A disclosure, not a menu: the trigger keeps focus, Tab reaches the list, and every row is an
 * ordinary button. `role="menu"` would promise arrow-key behaviour this list does not need.
 */
export function RecentThreads({
  id,
  opener,
  onPick,
  onDismiss,
}: {
  id: string;
  // The trigger: a pointerdown inside it is the button's own toggle, not a dismiss.
  opener: React.RefObject<HTMLButtonElement | null>;
  onPick: (threadId: string) => void;
  onDismiss: () => void;
}) {
  const [threads, setThreads] = useState<ThreadSummary[] | null>(null);
  const [failed, setFailed] = useState(false);
  const menu = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let active = true;
    async function load() {
      try {
        const response = await fetch("/api/threads", { cache: "no-store" });
        if (!response.ok) throw new Error("threads");
        const body = (await response.json()) as { threads: ThreadSummary[] };
        if (active) setThreads(body.threads);
      } catch {
        if (active) setFailed(true);
      }
    }
    void load();
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    function onPointerDown(event: MouseEvent) {
      const target = event.target as Node;
      // A click on the trigger is handled by its own onClick toggle; dismissing here as well
      // would close the list and reopen it in the same click.
      if (menu.current?.contains(target) || opener.current?.contains(target)) return;
      onDismiss();
    }
    document.addEventListener("pointerdown", onPointerDown);
    return () => document.removeEventListener("pointerdown", onPointerDown);
  }, [onDismiss, opener]);

  return (
    <div
      id={id}
      ref={menu}
      data-testid="recent-threads"
      // Anchored to the header, so its top edge is the header's rule rather than the button's.
      className="absolute top-full right-2 z-10 mt-0.5 w-[min(22rem,88vw)] border border-rule bg-card shadow-[var(--shadow)]"
    >
      {failed ? (
        <p className="px-3 py-2.5 text-[11.5px] text-ink-3">
          The conversation list is unavailable right now.
        </p>
      ) : threads === null ? (
        <p className="px-3 py-2.5 text-[11.5px] text-ink-3">Reading recent conversations...</p>
      ) : threads.length === 0 ? (
        <p className="px-3 py-2.5 text-[11.5px] text-ink-3">No conversations yet.</p>
      ) : (
        <ul className="max-h-[60dvh] overflow-auto py-1">
          {threads.map((thread) => (
            <li key={thread.id}>
              <button
                type="button"
                onClick={() => onPick(thread.id)}
                aria-label={`Open ${thread.title}, ${thread.turns} turn${thread.turns === 1 ? "" : "s"}`}
                className="block w-full px-3 py-2 text-left hover:bg-card-2"
              >
                <span className="line-clamp-2 block font-serif text-[13.5px] leading-[1.35] text-ink">
                  {thread.title}
                </span>
                <span className="mt-0.5 block font-mono text-[9.5px] text-ink-3">
                  {thread.turns} turn{thread.turns === 1 ? "" : "s"} ·{" "}
                  {formatRome(thread.updated_at)} Europe/Rome
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
