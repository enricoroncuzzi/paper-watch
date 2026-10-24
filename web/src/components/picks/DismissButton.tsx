"use client";

/**
 * The one control that takes a card off this reader's page. The card goes before the request
 * returns: a dismissal is the reader's own decision and must not wait on a tunnel.
 */
export function DismissButton({ title, onDismiss }: { title: string; onDismiss: () => void }) {
  return (
    <button
      type="button"
      onClick={onDismiss}
      // A screen reader meets seven of these: the item's title is what tells them apart.
      aria-label={`Not for me: ${title}`}
      title="Dismiss this card"
      // The text ends on the content edge by pulling the whole box out, not by removing one
      // side's padding: the hover and focus box stays symmetric.
      className="inline-flex h-11 items-center rounded-[2px] border border-transparent px-2 -mr-2 text-[11.5px] text-ink-3 transition-colors hover:border-rule-2 hover:bg-card hover:text-ink sm:h-7"
    >
      Not for me
    </button>
  );
}
