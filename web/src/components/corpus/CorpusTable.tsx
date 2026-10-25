import Link from "next/link";

import { TierChip } from "@/components/picks/Chips";
import { basisWord, publishedDay, signalLine } from "@/lib/corpus";
import { profileLabel } from "@/lib/format";
import type { CorpusItem } from "@/lib/types";

const CELL = "border-b border-rule px-2 py-2.5 align-top";

function SourceCell({ item }: { item: CorpusItem }) {
  // The stored source is "arxiv" or "feed"; a reader sees the two words the mockup prints.
  return (
    <td className={`${CELL} hidden text-[12px] text-ink-2 sm:table-cell`}>
      {item.source === "arxiv" ? "arXiv" : "feed"}
    </td>
  );
}

function SignalsCell({ item }: { item: CorpusItem }) {
  const number = signalLine(item);
  return (
    <td
      className={`${CELL} hidden whitespace-nowrap text-right tabular-nums text-[12px] text-ink-2 sm:table-cell`}
    >
      {/* A row whose rule is the lab's own feed has no number to print; the words say why. */}
      {number !== "n/a" ? <span className="block">{number}</span> : null}
      <span className="mt-0.5 block text-[10.5px] text-ink-3">
        {item.venue ?? basisWord(item.basis)}
      </span>
    </td>
  );
}

/**
 * One row of the corpus table, the row the keyset cursor walks. The title is the control: it is a
 * link a reader can open in a new tab, and `selected` only moves a visual highlight over it.
 */
export function CorpusRow({
  item,
  index,
  selected,
  focusRef,
  onSelect,
}: {
  item: CorpusItem;
  index: number;
  selected: boolean;
  // Set on the first row an append adds, so that row takes focus as it mounts.
  focusRef?: (node: HTMLAnchorElement | null) => void;
  onSelect: () => void;
}) {
  return (
    <tr
      data-testid="corpus-row"
      data-row-index={index}
      data-selected={selected ? "true" : undefined}
      // mousemove, not mouseenter: WebKit fires a synthetic mouseenter when the rows move under a
      // still pointer (Load more appends and focus scrolls), which would select a row nobody
      // pointed at. A move is the only thing that means "the reader is pointing here".
      onMouseMove={onSelect}
      className={selected ? "bg-accent-soft/60" : "transition-colors hover:bg-card"}
    >
      <td className={`${CELL} title-cell`}>
        <Link
          ref={focusRef}
          href={`/items/${item.id}`}
          className="block max-w-[52ch] font-serif text-[15px] leading-[1.4] hover:underline hover:decoration-rule-2 hover:underline-offset-2"
        >
          {item.title}
        </Link>
        <span className="mt-0.5 block text-[10.5px] text-ink-3">
          {item.profile ? profileLabel(item.profile) : "no topic"} ·{" "}
          {publishedDay(item.published_at)}
        </span>
      </td>
      <td className={`${CELL} whitespace-nowrap`}>
        <TierChip tier={item.tier} />
      </td>
      <SourceCell item={item} />
      <SignalsCell item={item} />
    </tr>
  );
}

/** The table's head; the two narrow columns are the ones a phone leaves out. */
export function CorpusHead() {
  return (
    <thead>
      <tr className="text-[9.5px] uppercase tracking-[0.13em] text-ink-3">
        <th scope="col" className="border-b border-rule px-2 py-2 text-left font-medium">
          Item
        </th>
        <th scope="col" className="border-b border-rule px-2 py-2 text-left font-medium">
          Tier
        </th>
        <th
          scope="col"
          className="hidden border-b border-rule px-2 py-2 text-left font-medium sm:table-cell"
        >
          Source
        </th>
        <th
          scope="col"
          className="hidden border-b border-rule px-2 py-2 text-right font-medium sm:table-cell"
        >
          Signals
        </th>
      </tr>
    </thead>
  );
}
