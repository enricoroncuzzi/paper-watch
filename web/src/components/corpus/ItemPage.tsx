"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";

import { useAskLauncher } from "@/components/ask/AskProvider";
import { PovDialog } from "@/components/drafts/PovDialog";
import { Chip, TierChip } from "@/components/picks/Chips";
import { ThirtySeconds } from "@/components/picks/ThirtySeconds";
import { basisWord, compactCount, publishedDay, tierMove } from "@/lib/corpus";
import { formatDay, profileLabel } from "@/lib/format";
import { safeHref } from "@/lib/safe-href";
import type { CorpusItemDetail, DraftTarget } from "@/lib/types";

const BUTTON =
  "inline-flex h-11 items-center justify-center gap-1.5 rounded-[2px] border px-2.5 text-[11.5px] font-medium sm:h-7";
const OUTLINE = `${BUTTON} border-rule-2 text-ink hover:border-ink hover:bg-card`;
const PRIMARY = `${BUTTON} border-accent bg-accent text-accent-ink hover:border-accent-2 hover:bg-accent-2`;

/** The item as the draft dialog needs it: the same fields a pick's card carries. */
function draftTarget(item: CorpusItemDetail): DraftTarget {
  return {
    item_id: item.id,
    title: item.title,
    tier: item.tier,
    venue: item.venue,
    org: item.org,
    profile: item.profile,
    summary: item.summary,
  };
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-3 border-b border-dotted border-rule-2 py-1 text-[11.5px]">
      <dt className="whitespace-nowrap text-ink-3">{label}</dt>
      <dd className="text-right tabular-nums text-ink">{value}</dd>
    </div>
  );
}

/** A stored URL that fails safeHref renders nothing at all: no link, and no text either. */
function LinkButton({ href, label }: { href: string | null; label: string }) {
  const safe = safeHref(href);
  if (!safe) return null;
  return (
    <a href={safe} target="_blank" rel="noreferrer noopener" className={OUTLINE}>
      {label}
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
 * One corpus item: its facts, the tier timeline, the signal series, the stored text, the cached
 * summary, and the three things a reader can do with it (read the original, ask about it, preview
 * a LinkedIn post). The page is server rendered; only the actions need the browser.
 */
export function ItemPage({ item }: { item: CorpusItemDetail }) {
  const router = useRouter();
  const { askAbout } = useAskLauncher();
  const [previewOpen, setPreviewOpen] = useState(false);
  // The button that opened the dialog, so WebKit puts focus back on it when the dialog closes.
  const [opener, setOpener] = useState<HTMLElement | null>(null);
  // The API lists drafts newest first, so the first one is the date the page prints.
  const newestDraft = item.drafts[0];

  return (
    <section data-testid="item-page" className="mt-6">
      <Link href="/#corpus" className="font-mono text-[10.5px] text-accent-text">
        ← Corpus
      </Link>

      <div className="mt-2 flex flex-wrap items-center gap-1.5 text-[10.5px] text-ink-3">
        <TierChip tier={item.tier} />
        {item.venue ? <Chip>{item.venue}</Chip> : null}
        {item.profile ? <Chip>{profileLabel(item.profile)}</Chip> : null}
        <span className="font-mono text-[10.5px]">{publishedDay(item.published_at)}</span>
      </div>

      <h1 className="mt-2 max-w-[46ch] font-serif text-[27px] font-normal leading-[1.25] tracking-[-0.015em] sm:text-[31px]">
        {item.title}
      </h1>
      <p className="mt-1.5 max-w-[52ch] text-[12px] text-ink-2">{item.basis_label}</p>

      <div className="mt-3 flex flex-wrap gap-2">
        <LinkButton href={item.url} label="Read the original" />
        <button
          type="button"
          onClick={() => askAbout({ id: item.id, title: item.title })}
          aria-label={`Ask about it: ${item.title}`}
          className={PRIMARY}
        >
          Ask about it
        </button>
        <button
          type="button"
          onClick={(event) => {
            setOpener(event.currentTarget);
            setPreviewOpen(true);
          }}
          aria-label={`LinkedIn post preview: ${item.title}`}
          className={PRIMARY}
        >
          LinkedIn post preview
        </button>
      </div>

      <div className="mt-5 grid grid-cols-1 gap-x-8 gap-y-6 sm:grid-cols-[minmax(0,1fr)_260px]">
        <div className="min-w-0">
          {item.summary ? (
            <ThirtySeconds summary={item.summary} />
          ) : (
            <p className="text-[12px] text-ink-3">
              No 30-second summary yet. It is written when the item first reaches the picks.
            </p>
          )}

          <h2 className="mt-5 text-[9px] uppercase tracking-[0.13em] text-ink-3">The stored text</h2>
          <p className="mt-1.5 max-w-[var(--measure)] whitespace-pre-line font-serif text-[15px] leading-[1.65] text-ink-2">
            {item.text}
          </p>
        </div>

        <aside className="flex flex-col gap-5">
          <div>
            <h2 className="mb-1 text-[9px] uppercase tracking-[0.13em] text-ink-3">Facts</h2>
            <dl>
              <Fact label="Authors" value={item.authors.join(", ") || "n/a"} />
              <Fact label="Citations" value={String(item.citation_count)} />
              <Fact
                label="Hugging Face"
                value={item.hf_upvotes === null ? "n/a" : `${item.hf_upvotes} \u2191`}
              />
              <Fact
                label="GitHub"
                value={item.github_stars === null ? "n/a" : `${compactCount(item.github_stars)} \u2605`}
              />
              <Fact label="Organisation" value={item.org ?? "n/a"} />
              <Fact label="Source" value={item.source === "arxiv" ? "arXiv" : "feed"} />
            </dl>
            <div className="mt-2 flex flex-wrap gap-2">
              <LinkButton href={item.github_repo} label="Code" />
              <LinkButton href={item.project_page} label="Project page" />
            </div>
          </div>

          <div>
            <h2 className="mb-1 text-[9px] uppercase tracking-[0.13em] text-ink-3">
              Tier history
            </h2>
            {item.tier_history.length === 0 ? (
              <p className="text-[11.5px] text-ink-3">No tier moves recorded.</p>
            ) : (
              <ol data-testid="tier-history" className="border-l border-rule pl-3">
                {item.tier_history.map((change) => (
                  <li key={`${change.changed_at}-${change.to_tier}`} className="relative py-1">
                    <span
                      aria-hidden="true"
                      className="absolute -left-[16.5px] top-[9px] h-[5px] w-[5px] rounded-full bg-rule-2"
                    />
                    <span className="block text-[11.5px] text-ink">{tierMove(change)}</span>
                    <span className="block text-[10.5px] text-ink-3">
                      {formatDay(change.changed_at)} · {basisWord(change.basis)}
                    </span>
                  </li>
                ))}
              </ol>
            )}
          </div>

          <div>
            <h2 className="mb-1 text-[9px] uppercase tracking-[0.13em] text-ink-3">Signals</h2>
            {item.signals.length === 0 ? (
              <p className="text-[11.5px] text-ink-3">No readings yet.</p>
            ) : (
              <table data-testid="signal-series" className="w-full border-collapse text-[11.5px]">
                <caption className="sr-only">Signal readings by date, oldest first</caption>
                <thead>
                  <tr className="text-[9px] uppercase tracking-[0.1em] text-ink-3">
                    <th scope="col" className="border-b border-rule py-1 text-left font-medium">
                      Date
                    </th>
                    <th scope="col" className="border-b border-rule py-1 text-right font-medium">
                      Upvotes
                    </th>
                    <th scope="col" className="border-b border-rule py-1 text-right font-medium">
                      Stars
                    </th>
                  </tr>
                </thead>
                <tbody className="tabular-nums text-ink-2">
                  {item.signals.map((point) => (
                    <tr key={point.observed_at}>
                      <td className="border-b border-dotted border-rule-2 py-1 font-mono text-[10.5px]">
                        {publishedDay(point.observed_at)}
                      </td>
                      <td className="border-b border-dotted border-rule-2 py-1 text-right">
                        {point.hf_upvotes ?? "n/a"}
                      </td>
                      <td className="border-b border-dotted border-rule-2 py-1 text-right">
                        {point.github_stars === null ? "n/a" : compactCount(point.github_stars)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>

          {newestDraft ? (
            <p className="text-[11.5px] text-ink-3">
              Already in Drafts: {item.drafts.length} (newest {formatDay(newestDraft.created_at)}).{" "}
              <Link href="/#drafts" className="text-accent-text underline decoration-rule-2">
                Open Drafts
              </Link>
            </p>
          ) : null}
        </aside>
      </div>

      <PovDialog
        card={previewOpen ? draftTarget(item) : null}
        // The same flag the draft generator uses, so the dialog warns before a call that will
        // produce the "early work" sentence.
        early={item.early}
        open={previewOpen}
        returnFocus={opener}
        onOpenChange={(next) => {
          if (next) {
            setPreviewOpen(true);
            return;
          }
          // Keep the target for the closing frame, so Radix can return focus to the opener first.
          setPreviewOpen(false);
        }}
        onSaved={() => router.refresh()}
      />
    </section>
  );
}
