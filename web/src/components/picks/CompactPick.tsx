import { AskAbout, LinkedInPreview, ReadOriginal } from "@/components/picks/CardActions";
import { AlsoLine, arxivLabel, Chip, TierChip } from "@/components/picks/Chips";
import { DismissButton } from "@/components/picks/DismissButton";
import { Facts } from "@/components/picks/Facts";
import { ThirtySeconds } from "@/components/picks/ThirtySeconds";
import type { SuggestionCard } from "@/lib/types";

/** A pick after the lead: the same four answers, with the facts in a narrow side column. */
export function CompactPick({
  card,
  rank,
  onDismiss,
  onPreview,
}: {
  card: SuggestionCard;
  rank: number;
  onDismiss: (itemId: string) => void;
  onPreview: () => void;
}) {
  const arxiv = arxivLabel(card.url);
  return (
    <article
      data-testid="compact-pick"
      className="border-t border-rule py-5 sm:grid sm:grid-cols-[34px_minmax(0,1fr)_216px] sm:gap-4"
    >
      <div className="hidden pt-0.5 font-serif text-[26px] leading-none text-ink-3 sm:block">
        {String(rank).padStart(2, "0")}
      </div>
      <div>
        <h3 className="max-w-[30ch] font-serif text-[20px] font-normal leading-[1.2] tracking-[-0.015em] sm:text-[23px]">
          {card.title}
        </h3>
        <div className="mt-2 flex flex-wrap items-center gap-1.5 text-[11px] text-ink-3">
          <TierChip tier={card.tier} />
          {card.venue ?? card.org ? <Chip>{card.venue ?? card.org}</Chip> : null}
          {card.hf_upvotes !== null ? <Chip>HF {card.hf_upvotes} ↑</Chip> : null}
          {card.citation_count > 0 ? <Chip>{card.citation_count} cites</Chip> : null}
          {card.profile ? <Chip>{card.profile}</Chip> : null}
          {arxiv ? <span className="font-mono text-[11px]">{arxiv}</span> : null}
        </div>
        <AlsoLine card={card} />
        <ThirtySeconds summary={card.summary} compact />
      </div>

      <div className="mt-3 flex flex-col gap-2 sm:mt-0">
        <Facts card={card} compact />
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-1 sm:gap-1.5">
          <ReadOriginal card={card} />
          <AskAbout card={card} />
          <div className="col-span-2 sm:col-span-1 [&>button]:w-full">
            <LinkedInPreview card={card} onPreview={onPreview} />
          </div>
        </div>
        <div className="flex justify-end">
          <DismissButton title={card.title} onDismiss={() => onDismiss(card.item_id)} />
        </div>
      </div>
    </article>
  );
}
