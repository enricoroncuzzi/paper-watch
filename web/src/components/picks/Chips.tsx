import { safeHref } from "@/lib/safe-href";
import type { SuggestionCard } from "@/lib/types";

// The mockup's chip: a hairline box, one colour per tier so the evidence reads at a glance.
const TIER_STYLE: Record<string, string> = {
  peer_reviewed: "border-ok/45 text-ok",
  lab_blog: "border-warn/45 text-warn",
};
// A reader sees words, not the database's enum: the code stays only in the colour lookup above.
const TIER_LABEL: Record<string, string> = {
  peer_reviewed: "peer-reviewed",
  vetted_preprint: "vetted preprint",
  lab_blog: "lab post",
  admitted: "admitted",
};
const BASE = "inline-flex h-[21px] items-center gap-1 whitespace-nowrap border px-2 text-[11px]";

export function TierChip({ tier }: { tier: string }) {
  return (
    <span className={`${BASE} ${TIER_STYLE[tier] ?? "border-rule-2 text-ink-2"}`}>
      {TIER_LABEL[tier] ?? tier.replaceAll("_", " ")}
    </span>
  );
}

export function Chip({ children }: { children: React.ReactNode }) {
  return <span className={`${BASE} border-rule text-ink-2`}>{children}</span>;
}

/**
 * The arXiv id a reader can quote elsewhere, taken from the stored URL's own path. Nothing is
 * invented: a URL that is not an arXiv abstract page has no id to show.
 */
export function arxivLabel(url: string): string | null {
  const href = safeHref(url);
  if (!href) return null;
  try {
    const parsed = new URL(href);
    if (parsed.hostname !== "arxiv.org" && !parsed.hostname.endsWith(".arxiv.org")) return null;
    const match = /^\/abs\/(.+)$/.exec(parsed.pathname);
    return match ? `arXiv ${match[1]}` : null;
  } catch {
    return null;
  }
}

/** A near-duplicate the ranking folded into this card, usually the lab's own write-up. */
export function AlsoLine({ card }: { card: SuggestionCard }) {
  if (card.also.length === 0) return null;
  return (
    <p className="mt-3 flex flex-wrap items-center gap-x-1.5 gap-y-1 text-[11.5px] text-ink-3">
      <span>Also:</span>
      {card.also.map((entry) => {
        const href = safeHref(entry.url);
        return href ? (
          <a
            key={entry.url}
            href={href}
            target="_blank"
            rel="noreferrer noopener"
            className="underline decoration-rule-2 underline-offset-2 hover:text-ink"
          >
            {entry.title}
          </a>
        ) : (
          <span key={`${entry.source}-${entry.title}`}>{entry.title}</span>
        );
      })}
    </p>
  );
}
