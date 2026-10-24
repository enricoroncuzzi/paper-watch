import { AskAbout, LinkedInPreview, ReadOriginal } from "@/components/picks/CardActions";
import { arxivLabel } from "@/components/picks/Chips";
import { DismissButton } from "@/components/picks/DismissButton";
import { compactCount } from "@/components/picks/Facts";
import { EmptyState } from "@/components/SectionState";
import type { SuggestionCard } from "@/lib/types";

/** The one signal beside the title: a gain if there is one, else the count that earned the card. */
function signalFor(card: SuggestionCard): string | null {
  if (card.momentum_label) return card.momentum_label;
  if (card.hf_upvotes !== null) return `${card.hf_upvotes} \u2191 Hugging Face`;
  if (card.github_stars !== null) return `${compactCount(card.github_stars)} \u2605 GitHub`;
  if (card.citation_count > 0) return `${card.citation_count} citations`;
  return null;
}

function RisingCard({
  card,
  onDismiss,
  onPreview,
}: {
  card: SuggestionCard;
  onDismiss: (itemId: string) => void;
  onPreview: (opener: HTMLElement) => void;
}) {
  const signal = signalFor(card);
  // The stored location a reader can recognise: an arXiv id, the repository, or the lab's name.
  // The stored value as it is: a repository URL is what the ingestion saved, not a bare slug.
  const where = arxivLabel(card.url) ?? card.github_repo ?? card.venue ?? card.org ?? "stored item";
  return (
    <div
      data-testid="rising-card"
      className="grid gap-x-4 gap-y-2 border-b border-rule px-3.5 py-3 last:border-b-0 sm:grid-cols-[minmax(0,1fr)_auto] sm:items-start"
    >
      <div>
        <div className="font-serif text-[16.5px] leading-[1.35]">{card.title}</div>
        <div className="mt-1 flex flex-wrap items-center gap-x-1.5 gap-y-1 text-[10.5px] text-ink-3">
          <span>{where}</span>
          {card.profile ? (
            <>
              <span aria-hidden="true">·</span>
              <span>{card.profile}</span>
            </>
          ) : null}
          <span aria-hidden="true">·</span>
          <span>{card.published_at.slice(0, 10)}</span>
        </div>
        {/* The basis gets its own line so the separator before it never dangles at a wrap. */}
        <div className="mt-0.5 text-[10.5px] text-ink-3">{card.basis_label}</div>
      </div>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 sm:flex-col sm:items-end">
        {signal ? (
          <b className="text-[12.5px] font-semibold text-ink sm:max-w-[220px] sm:text-right">
            {signal}
          </b>
        ) : null}
        {/* Every Rising item is not pick-grade, so every draft from here says it is early work. */}
        <p className="text-[10.5px] leading-[1.4] text-warn sm:max-w-[220px] sm:text-right">
          This item is early work; the draft will say so.
        </p>
        <div className="flex flex-wrap items-center justify-end gap-1.5">
          <ReadOriginal card={card} />
          <AskAbout card={card} />
          <DismissButton title={card.title} onDismiss={() => onDismiss(card.item_id)} />
          <LinkedInPreview card={card} onPreview={onPreview} />
        </div>
      </div>
    </div>
  );
}

/**
 * Rising is a shelf inside This week: the same cards without the evidence a pick needs, labelled
 * so a pick never quietly becomes a hunch. The note comes from the API, next to the cards it is about.
 */
export function RisingShelf({
  rising,
  note,
  onDismiss,
  onPreview,
}: {
  rising: SuggestionCard[];
  note: string;
  onDismiss: (itemId: string) => void;
  onPreview: (card: SuggestionCard, opener: HTMLElement) => void;
}) {
  return (
    <section
      id="rising"
      data-testid="rising-section"
      aria-labelledby="rising-title"
      className="mt-8 border-t-[1.5px] border-ink pt-3 sm:mt-12"
    >
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <span className="font-mono text-[10px] uppercase tracking-[0.1em] text-accent-text">02</span>
        <h2 id="rising-title" className="font-serif text-[23px] font-normal tracking-[-0.02em] sm:text-[27px]">
          Rising
        </h2>
        <span className="text-[11.5px] text-ink-3">
          {rising.length} {rising.length === 1 ? "item" : "items"}
        </span>
      </div>
      <p className="mb-4 mt-1.5 max-w-[64ch] text-[12.5px] text-ink-2 sm:text-[13px]">
        Trending this week without the evidence a pick needs: no venue, no citation base yet.
        Labelled on purpose, so a pick never quietly becomes a hunch.
      </p>
      {rising.length === 0 ? (
        <EmptyState
          testId="rising-empty"
          title="Nothing rising"
          body="No item moved enough this week to leave the rank it was admitted at."
        />
      ) : (
        <div className="overflow-hidden rounded-[2px] border border-rule bg-card">
          <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1 border-b border-rule bg-card-2 px-3.5 py-2.5 text-[11px] text-ink-2">
            <span className="inline-flex items-center gap-1.5 text-[9.5px] font-semibold uppercase tracking-[0.12em] text-warn">
              <svg
                width="11"
                height="11"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.2"
                aria-hidden="true"
              >
                <path d="M4 18 10 11l4 3 6-8" />
                <path d="M14 6h6v6" />
              </svg>
              {note}
            </span>
            <span className="ml-auto hidden text-[10.5px] text-ink-3 sm:inline">re-checked every run</span>
          </div>
          {rising.map((card) => (
            <RisingCard
              key={card.item_id}
              card={card}
              onDismiss={onDismiss}
              onPreview={(opener) => onPreview(card, opener)}
            />
          ))}
        </div>
      )}
    </section>
  );
}
