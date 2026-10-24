import { safeHref, safeHost } from "@/lib/safe-href";
import type { SourceRef } from "@/lib/use-ask";

/**
 * The answer's sources, one footnote each. Every link comes from a stored URL that passed
 * `safeHref`; the visible text is the parsed host, so a crafted path cannot disguise the target.
 */
export function Sources({
  sources,
  numbers,
}: {
  sources: SourceRef[];
  numbers: Map<string, number>;
}) {
  if (sources.length === 0) return null;
  return (
    <div data-testid="sources" className="mt-5 max-w-[38rem] border-t border-rule pt-3">
      <h4 className="mb-2 text-[9.5px] uppercase tracking-[0.13em] text-ink-3">Sources</h4>
      <ol className="flex flex-col gap-2">
        {sources.map((source, index) => {
          const href = safeHref(source.url);
          const host = safeHost(source.url);
          return (
            <li
              key={source.id}
              id={`fn-${source.id}`}
              className="grid grid-cols-[20px_1fr] gap-2.5 text-[11.5px] leading-[1.45] text-ink-2"
            >
              <span className="pt-0.5 text-right font-mono text-[10px] text-accent-text">
                {numbers.get(source.id) ?? index + 1}
              </span>
              <span>
                {href ? (
                  <a
                    href={href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="mb-0.5 block font-serif text-[14px] text-ink hover:underline"
                  >
                    {source.title}
                  </a>
                ) : (
                  <span className="mb-0.5 block font-serif text-[14px] text-ink">
                    {source.title}
                  </span>
                )}
                <span className="flex flex-wrap items-center gap-1.5 text-[10.5px] text-ink-3">
                  <span className="rounded-full border border-rule-2 px-1.5 py-px font-mono text-[9.5px]">
                    {source.tier}
                  </span>
                  <span>{source.basis.replaceAll("_", " ")}</span>
                  {host ? <span>· {host}</span> : null}
                </span>
              </span>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
