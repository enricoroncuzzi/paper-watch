import { DraftCard } from "@/components/drafts/DraftCard";
import { EmptyState } from "@/components/SectionState";
import type { Draft } from "@/lib/types";

/**
 * The saved drafts, newest first. The check under each one is code, not a model, so the section's
 * copy says what the notes are before the reader meets one.
 */
export function DraftsSection({ drafts }: { drafts: Draft[] }) {
  return (
    <section
      id="drafts"
      data-testid="drafts-section"
      aria-labelledby="drafts-title"
      className="mt-[30px] border-t-[1.5px] border-ink pt-3 sm:mt-[52px]"
    >
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <span className="font-mono text-[10px] uppercase tracking-[0.1em] text-accent-text">
          03
        </span>
        <h2
          id="drafts-title"
          className="font-serif text-[24px] font-normal tracking-[-0.01em] sm:text-[27px]"
        >
          Drafts
        </h2>
        <span className="text-[11.5px] text-ink-3">
          {drafts.length} saved · nothing posted
        </span>
      </div>
      <p className="mb-4 mt-1.5 max-w-[64ch] text-[12.5px] text-ink-2 sm:text-[13px]">
        Generated from a pick plus your own words. The check under each draft is code, not a model:
        it names the numbers, quotes and links it could verify, and the ones it could not.
      </p>
      {drafts.length === 0 ? (
        <EmptyState
          testId="drafts-empty"
          title="No drafts yet"
          body={
            'Open a pick, press "LinkedIn post preview", and write your point of view. The draft and its check land here.'
          }
        />
      ) : (
        <div className="flex flex-col gap-3.5">
          {drafts.map((draft) => (
            <DraftCard key={draft.id} draft={draft} />
          ))}
        </div>
      )}
    </section>
  );
}
