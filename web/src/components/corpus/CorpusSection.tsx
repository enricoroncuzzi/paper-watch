"use client";

import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";

import { CorpusHead, CorpusRow } from "@/components/corpus/CorpusTable";
import { EmptyState } from "@/components/SectionState";
import { CORPUS_NOTE, TIER_TABS } from "@/lib/corpus";
import type { ApiErrorKind } from "@/lib/api";
import type { CorpusData, CorpusItem } from "@/lib/types";

// Ten rows is what a reader scans before deciding to scroll; the API's own default is 25, so the
// page is the side that asks for a smaller page.
const PAGE_SIZE = 10;
// Long enough that a typed word is one request, short enough that the table feels live.
const SEARCH_DEBOUNCE_MS = 300;

const TAB = "relative py-1.5 text-[12px] text-ink-3 hover:text-ink";
const TAB_ON = "relative py-1.5 text-[12px] text-ink";

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
  const [page, setPage] = useState<Page>(pageOf(initial));
  const [text, setText] = useState("");
  const [filters, setFilters] = useState({ q: "", tier: "" });
  const [selected, setSelected] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<ApiErrorKind | null>(null);
  const first = useRef(true);

  const load = useCallback(
    async (next: { q: string; tier: string; cursor: string | null; append: boolean }) => {
      setLoading(true);
      setError(null);
      const query = new URLSearchParams({ limit: String(PAGE_SIZE) });
      if (next.q) query.set("q", next.q);
      if (next.tier) query.set("tier", next.tier);
      if (next.cursor) query.set("cursor", next.cursor);
      try {
        const response = await fetch(`/api/corpus?${query}`, { cache: "no-store" });
        if (!response.ok) {
          const body = (await response.json().catch(() => null)) as { kind?: ApiErrorKind } | null;
          setError(body?.kind ?? "unavailable");
          return;
        }
        const data = (await response.json()) as CorpusData;
        setPage((current) => {
          const fresh = pageOf(data);
          return next.append
            ? { ...fresh, rows: [...current.rows, ...fresh.rows] }
            : fresh;
        });
        setSelected(0);
      } catch {
        setError("unavailable");
      } finally {
        setLoading(false);
      }
    },
    [],
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

  // One listener for the section's three keys. It never fires while a field has focus, so typing
  // a title that contains "j" or "/" into the search box is just typing.
  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null;
      const typing =
        target !== null &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.tagName === "SELECT" ||
          target.isContentEditable);
      if (typing || event.metaKey || event.ctrlKey || event.altKey) return;
      if (event.key === "/") {
        event.preventDefault();
        input.current?.focus();
        return;
      }
      if (event.key !== "j" && event.key !== "k" && event.key !== "Enter") return;
      if (page.rows.length === 0) return;
      event.preventDefault();
      if (event.key === "Enter") {
        const row = page.rows[selected];
        if (row) router.push(`/items/${row.id}`);
        return;
      }
      setSelected((current) =>
        event.key === "j"
          ? Math.min(current + 1, page.rows.length - 1)
          : Math.max(current - 1, 0),
      );
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [page.rows, selected, router]);

  const shown = page.rows.length;
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
          {page.total} admitted
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

      <div
        role="tablist"
        aria-label="Tier filter"
        className="mt-3 flex gap-3.5 border-b border-rule"
      >
        {TIER_TABS.map((tab) => {
          const on = filters.tier === tab.value;
          return (
            <button
              key={tab.label}
              type="button"
              role="tab"
              aria-selected={on}
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
          title={
            filters.q ? `Nothing matches \u201c${filters.q}\u201d` : "No admitted items yet"
          }
          body={
            filters.tier
              ? "No admitted item in this tier matches. Clear the tier filter to search all three."
              : "The ingestion admits items every morning; the table fills as it runs."
          }
        />
      ) : null}

      {shown > 0 ? (
        <table aria-busy={loading} className="mt-1 w-full border-collapse">
          <caption className="sr-only">
            Admitted items, {filters.q ? "best match first" : "newest published first"}
          </caption>
          <CorpusHead />
          <tbody>
            {page.rows.map((item, index) => (
              <CorpusRow
                key={item.id}
                item={item}
                selected={index === selected}
                onSelect={() => setSelected(index)}
              />
            ))}
          </tbody>
        </table>
      ) : null}

      <div className="flex flex-wrap items-center justify-between gap-2 py-2.5 text-[11px] text-ink-3">
        <span>{CORPUS_NOTE}</span>
        <span className="flex items-center gap-3">
          <span className="font-mono text-[10.5px]">
            {shown} of {page.total} · j k to move
          </span>
          {page.cursor ? (
            <button
              type="button"
              disabled={loading}
              aria-label="Load more items"
              onClick={() => void load({ ...filters, cursor: page.cursor, append: true })}
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
