import { AskAbout, LinkedInPreview, ReadOriginal } from "@/components/picks/CardActions";
import { AlsoLine, arxivLabel, Chip, TierChip } from "@/components/picks/Chips";
import { DismissButton } from "@/components/picks/DismissButton";
import { Facts } from "@/components/picks/Facts";
import { ThirtySeconds } from "@/components/picks/ThirtySeconds";
import { profileLabel } from "@/lib/format";
import type { SuggestionCard } from "@/lib/types";

/**
 * The lead pick: the week's top item, with its facts and its 30-second summary beside it. The
 * facts come before the summary on a phone, as the mockup's phone layout does; each child is
 * placed explicitly so the desktop two-column grid can keep that source order.
 */
export function LeadPick({
  card,
  onDismiss,
  onPreview,
}: {
  card: SuggestionCard;
  onDismiss: (itemId: string) => void;
  onPreview: (opener: HTMLElement) => void;
}) {
  const arxiv = arxivLabel(card.url);
  return (
    <article data-testid="lead-pick" className="pt-5 sm:grid sm:grid-cols-[minmax(0,1fr)_300px] sm:gap-x-[52px]">
      <div className="mb-2.5 flex flex-wrap items-center gap-x-2.5 gap-y-1 text-[10px] uppercase tracking-[0.15em] text-ink-3 sm:col-span-2">
        <b className="font-semibold text-accent-text">Pick 01</b>
        <span aria-hidden="true" className="hidden sm:inline">
          ·
        </span>
        <span className="hidden sm:inline">Top of the week</span>
        {card.profile ? (
          <>
            <span aria-hidden="true">·</span>
            <span>{profileLabel(card.profile)}</span>
          </>
        ) : null}
        <span className="ml-auto normal-case tracking-normal">
          <DismissButton title={card.title} onDismiss={() => onDismiss(card.item_id)} />
        </span>
      </div>

      <h3 className="max-w-[21ch] font-serif text-[30px] font-normal leading-[1.07] tracking-[-0.022em] text-balance sm:col-start-1 sm:row-start-2 sm:text-[40px] lg:text-[46px]">
        {card.title}
      </h3>
      <p className="mt-3.5 max-w-[36ch] font-serif text-[18px] leading-[1.5] text-ink-2 sm:col-start-1 sm:row-start-3 sm:text-[19px]">
        {card.summary.what?.text ?? card.summary.excerpt}
      </p>
      <div className="sm:col-start-1 sm:row-start-4">
        <div className="mt-4 flex flex-wrap items-center gap-2 text-[11.5px] text-ink-3">
          <TierChip tier={card.tier} />
          {card.venue ?? card.org ? <Chip>{card.venue ?? card.org}</Chip> : null}
          {card.citation_count > 0 ? <Chip>{card.citation_count} cites</Chip> : null}
          {card.hf_upvotes !== null ? <Chip>HF {card.hf_upvotes} ↑</Chip> : null}
          {arxiv ? <span className="font-mono text-[11px]">{arxiv}</span> : null}
        </div>
        <AlsoLine card={card} />
      </div>

      <aside
        data-testid="lead-facts"
        className="mt-5 border-t-[1.5px] border-ink pt-3 sm:col-start-2 sm:row-start-2 sm:row-span-4 sm:mt-0"
      >
        <h4 className="mb-2 text-[9.5px] uppercase tracking-[0.15em] text-ink-3">Why it is here</h4>
        <Facts card={card} />
        <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-1 sm:gap-1.5">
          <ReadOriginal card={card} />
          <AskAbout card={card} />
          <div className="col-span-2 sm:col-span-1 [&>button]:w-full">
            <LinkedInPreview card={card} onPreview={onPreview} />
          </div>
        </div>
        {card.momentum_label ? (
          <p className="mt-2.5 text-[10.5px] leading-[1.45] text-ink-3">{card.momentum_label}</p>
        ) : null}
      </aside>

      <div className="sm:col-start-1 sm:row-start-5">
        <ThirtySeconds summary={card.summary} />
      </div>
    </article>
  );
}
