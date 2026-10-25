"use client";

import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";

import { CorpusHead, CorpusRow } from "@/components/corpus/CorpusTable";
import { EmptyState } from "@/components/SectionState";
import { CORPUS_NOTE, corpusTotalLabel, TIER_TABS } from "@/lib/corpus";
import type { ApiErrorKind } from "@/lib/api";
import type { CorpusData, CorpusItem } from "@/lib/types";

// Ten rows is what a reader scans before deciding to scroll; the API's own default is 25, so the
// page is the side that asks for a smaller page.
const PAGE_SIZE = 10;
// Long enough that a typed word is one request, short enough that the table feels live.
const SEARCH_DEBOUNCE_MS = 300;

const TAB = "relative inline-flex min-h-11 items-center text-[12px] text-ink-3 hover:text-ink sm:min-h-0 sm:py-1.5";
const TAB_ON = "relative inline-flex min-h-11 items-center text-[12px] text-ink sm:min-h-0 sm:py-1.5";

type Page = { rows: CorpusItem[]; total: number; cursor: string | null };

function pageOf(data: CorpusData): Page {
  return { rows: data.items, total: data.total, cursor: data.next_cursor };
}

/**
 * The corpus section: the mockup's `.library` table over the API's `/items`. The first page is
 * rendered on the server, so the section is never empty while it waits; filtering, searching and
 * "Load more" go through the page's own `/api/corpus` proxy, which keeps the token on the server.
 */
export function CorpusSection({ initial }: { initial: CorpusData }) {
  const router = useRouter();
  const input = useRef<HTMLInputElement>(null);
  const table = useRef<HTMLTableElement>(null);
  const [page, setPage] = useState<Page>(pageOf(initial));
  const [text, setText] = useState("");
  const [filters, setFilters] = useState({ q: "", tier: "" });
  const [selected, setSelected] = useState(0);
  // The same index in a ref: two key presses in one frame must step twice, not once, because a
  // state updater re-reads the value React has not re-rendered yet.
  const selectedRef = useRef(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<ApiErrorKind | null>(null);
  // The row index to focus after an append, so the reader lands on the first new row.
  const [focusAfter, setFocusAfter] = useState<number | null>(null);
  const first = useRef(true);
  const select = useCallback((next: number) => {
    selectedRef.current = next;
    setSelected(next);
  }, []);

  // The last position a real pointer move reported. A scroll under a resting pointer makes WebKit
  // send a synthetic mousemove at the same coordinates; only a changed position selects a row.
  const pointer = useRef<{ x: number; y: number } | null>(null);
  const selectFromPointer = useCallback(
    (index: number, x: number, y: number) => {
      const last = pointer.current;
      if (last && last.x === x && last.y === y) return;
      pointer.current = { x, y };
      select(index);
    },
    [select],
  );

  // The page as the ref sees it: an append merges into it without waiting for a render.
  const pageRef = useRef(page);
  // The Load more button that was pressed, so focus stays put when the reader has moved on.
  const loader = useRef<HTMLElement | null>(null);
  // Every request carries a number: a response that is no longer the newest one is dropped, so a
  // slow answer to an older query can never overwrite the table.
  const request = useRef(0);

  const load = useCallback(
    async (next: { q: string; tier: string; cursor: string | null; append: boolean }) => {
      const id = (request.current += 1);
      setLoading(true);
      setError(null);
      const query = new URLSearchParams({ limit: String(PAGE_SIZE) });
      if (next.q) query.set("q", next.q);
      if (next.tier) query.set("tier", next.tier);
      if (next.cursor) query.set("cursor", next.cursor);
      try {
        const response = await fetch(`/api/corpus?${query}`, { cache: "no-store" });
        if (id !== request.current) return;
        if (!response.ok) {
          const body = (await response.json().catch(() => null)) as { kind?: ApiErrorKind } | null;
          setError(body?.kind ?? "unavailable");
          return;
        }
        const data = (await response.json()) as CorpusData;
        if (id !== request.current) return;
        const fresh = pageOf(data);
        const current = pageRef.current;
        if (!next.append) {
          pageRef.current = fresh;
          setPage(fresh);
          // A new query starts at the top; an appended page keeps the reader where they were.
          select(0);
          return;
        }
        // A row can be served twice when the ranking shifts between pages (an ingestion lands
        // mid-walk); the id keeps React's keys unique and the table free of duplicates.
        const seen = new Set(current.rows.map((row) => row.id));
        const added = fresh.rows.filter((row) => !seen.has(row.id));
        if (added.length > 0) setFocusAfter(current.rows.length);
        const merged: Page = {
          rows: [...current.rows, ...added],
          total: fresh.total,
          cursor: fresh.cursor,
        };
        pageRef.current = merged;
        setPage(merged);
      } catch {
        if (id === request.current) setError("unavailable");
      } finally {
        if (id === request.current) setLoading(false);
      }
    },
    [select],
  );

  // The server already rendered the first page: a reload on mount would only flash it.
  useEffect(() => {
    if (first.current) {
      first.current = false;
      return;
    }
    void load({ ...filters, cursor: null, append: false });
  }, [filters, load]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      const q = text.trim();
      setFilters((current) => (current.q === q ? current : { ...current, q }));
    }, SEARCH_DEBOUNCE_MS);
    return () => window.clearTimeout(timer);
  }, [text]);

  // A ref on the first row an append adds: React calls it as that row mounts, so the reader lands
  // there instead of at the top of the table (the Load more button unmounts under them). Only when
  // focus is still on the page or on the button, so a reader who moved to the search box mid-load
  // keeps their caret.
  const takeFocus = useCallback((node: HTMLAnchorElement | null) => {
    if (!node) return;
    const active = document.activeElement;
    const onPage = active === null || active === document.body || active === loader.current;
    if (!onPage) return;
    node.focus();
    node.scrollIntoView({ block: "nearest" });
  }, []);

  // One listener for the section's three keys, and only when the page itself has focus: a focused
  // link, button, tab or field, or an open dialog, keeps its own Enter and its own letters. It is
  // attached once and reads the rows and the selection from refs, because a listener that detaches
  // and reattaches on every render can miss a press that lands in the gap.
  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (document.querySelector("[role=dialog]")) return;
      const target = event.target as HTMLElement | null;
      const onPage = target === null || target === document.body || target === document.documentElement;
      const inTable = target?.closest('[data-corpus="table"]') != null;
      if (!onPage && !inTable) return;
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      if (event.key === "/") {
        event.preventDefault();
        input.current?.focus();
        return;
      }
      if (event.key !== "j" && event.key !== "k" && event.key !== "Enter") return;
      const rows = pageRef.current.rows;
      if (rows.length === 0) return;
      // Enter is ours only on the page itself: on a row link the browser activates the link, so a
      // keyboard reader who tabbed to another row opens that row, not the selection.
      if (event.key === "Enter" && !onPage) return;
      event.preventDefault();
      if (event.key === "Enter") {
        const row = rows[selectedRef.current];
        if (row) router.push(`/items/${row.id}`);
        return;
      }
      const next =
        event.key === "j"
          ? Math.min(selectedRef.current + 1, rows.length - 1)
          : Math.max(selectedRef.current - 1, 0);
      select(next);
      table.current
        ?.querySelector(`[data-row-index="${next}"] a`)
        ?.scrollIntoView({ block: "nearest" });
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [router, select]);

  const shown = page.rows.length;
  // A capped search says so: "10 of 200+" is honest about a list nobody can page through.
  const totalLabel = corpusTotalLabel(page.total, filters.q !== "");
  return (
    <section
      id="corpus"
      data-testid="corpus-section"
      aria-labelledby="corpus-title"
      className="mt-[30px] border-t-[1.5px] border-ink pt-3 sm:mt-[52px]"
    >
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-2">
        <h2 id="corpus-title" className="font-serif text-[24px] font-normal tracking-[-0.01em]">
          The corpus
        </h2>
        <span className="text-[11px] text-ink-3">
          {totalLabel} admitted
          {filters.tier || filters.q ? " matching" : ""}
        </span>
        <label className="ml-auto flex h-8 min-w-[240px] items-center gap-2 rounded-[2px] border border-rule-2 bg-card px-2.5 text-[12px] text-ink-3">
          <svg
            width="13"
            height="13"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            aria-hidden="true"
            className="flex-[0_0_13px]"
          >
            <circle cx="11" cy="11" r="7" />
            <path d="m20 20-3.5-3.5" />
          </svg>
          <span className="sr-only">Search the corpus</span>
          <input
            ref={input}
            type="search"
            value={text}
            onChange={(event) => setText(event.target.value)}
            placeholder="Search titles and full text"
            aria-label="Search the corpus"
            className="min-w-0 flex-1 bg-transparent text-[12px] text-ink outline-none placeholder:text-ink-3"
          />
          <kbd className="hidden font-mono text-[10px] text-ink-3 sm:block">/</kbd>
        </label>
      </div>

      {/* Plain toggles rather than a tab pattern: there is one table, not one panel per tier. */}
      <div role="group" aria-label="Tier filter" className="mt-1 flex gap-3.5 border-b border-rule">
        {TIER_TABS.map((tab) => {
          const on = filters.tier === tab.value;
          return (
            <button
              key={tab.label}
              type="button"
              aria-pressed={on}
              onClick={() =>
                setFilters((current) =>
                  current.tier === tab.value ? current : { ...current, tier: tab.value },
                )
              }
              className={on ? TAB_ON : TAB}
            >
              {tab.label}
              {on ? (
                <span
                  aria-hidden="true"
                  className="absolute inset-x-0 -bottom-px block h-[1.5px] bg-accent"
                />
              ) : null}
            </button>
          );
        })}
      </div>

      {error ? (
        <p
          data-testid="corpus-error"
          className="mt-3 border border-bad/40 bg-bad-soft px-3 py-2.5 text-[12px] text-ink-2"
        >
          <span className="font-medium text-bad">The corpus could not be read.</span> The ingestion
          is unaffected; try again in a moment.
        </p>
      ) : null}

      {shown === 0 && !error ? (
        <EmptyState
          testId="corpus-empty"
          title={filters.q ? `Nothing matches \u201c${filters.q}\u201d` : "No admitted items yet"}
          body={
            filters.q
              ? "No admitted item matches. Try fewer words."
              : filters.tier
                ? "No admitted item in this tier yet. Clear the tier filter to search all three."
                : "The ingestion admits items every morning; the table fills as it runs."
          }
        />
      ) : null}

      {shown > 0 ? (
        <table ref={table} data-corpus="table" aria-busy={loading} className="mt-1 w-full border-collapse">
          <caption className="sr-only">
            Admitted items, {filters.q ? "best match first" : "newest published first"}
          </caption>
          <CorpusHead />
          <tbody>
            {page.rows.map((item, index) => (
              <CorpusRow
                key={item.id}
                item={item}
                index={index}
                selected={index === selected}
                focusRef={index === focusAfter ? takeFocus : undefined}
                onSelect={(x, y) => selectFromPointer(index, x, y)}
              />
            ))}
          </tbody>
        </table>
      ) : null}

      <div className="flex flex-wrap items-center justify-between gap-2 py-2.5 text-[11px] text-ink-3">
        <span>{CORPUS_NOTE}</span>
        <span className="flex items-center gap-3">
          <span role="status" className="font-mono text-[10.5px]">
            Showing {shown} of {totalLabel} · j k to move
          </span>
          {page.cursor ? (
            <button
              type="button"
              disabled={loading}
              aria-label="Load more items"
              onClick={(event) => {
                loader.current = event.currentTarget;
                void load({ ...filters, cursor: page.cursor, append: true });
              }}
              className="inline-flex h-11 items-center justify-center rounded-[2px] border border-rule-2 px-2.5 text-[11.5px] font-medium text-ink transition-colors hover:border-ink hover:bg-card disabled:opacity-60 sm:h-7"
            >
              {loading ? "Loading…" : "Load more"}
            </button>
          ) : null}
        </span>
      </div>
    </section>
  );
}
