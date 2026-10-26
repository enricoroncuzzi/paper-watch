import type { ApiErrorKind } from "@/lib/api";

/**
 * The three states every section has. The copy is fixed per kind: a section never shows a raw
 * error, because a driver message can carry a host or a path the page should not reveal.
 */
const ERROR_COPY: Record<ApiErrorKind, { title: string; body: string }> = {
  unauthorized: {
    title: "This page was refused",
    body: "The API did not accept this page's credential. Sign in again.",
  },
  rate_limited: {
    title: "The API is busy",
    body: "The answer limit is spent for now. The page itself is fine; try again shortly.",
  },
  unavailable: {
    title: "The API is unreachable",
    body: "The API is unreachable right now. The ingestion is unaffected.",
  },
  not_found: {
    title: "Nothing here",
    body: "The API has no record with that id.",
  },
  error: {
    title: "The API answered with an error",
    body: "The request reached the API and it could not answer. The ingestion is unaffected.",
  },
};

export function LoadingState({ testId, lines = 3 }: { testId: string; lines?: number }) {
  return (
    <div data-testid={testId} aria-busy="true" className="mt-3 space-y-2">
      {Array.from({ length: lines }, (_, index) => (
        <div
          key={index}
          className="h-[34px] animate-pulse rounded-[2px] bg-card-2"
          style={{ animationDelay: `${index * 120}ms` }}
        />
      ))}
    </div>
  );
}

export function EmptyState({
  title,
  body,
  testId,
}: {
  title: string;
  body: string;
  testId?: string;
}) {
  return (
    <div
      data-testid={testId}
      className="mt-3 border border-dashed border-rule-2 px-5 py-8 text-center text-ink-3"
    >
      <h3 className="mb-1.5 font-serif text-[20px] font-normal text-ink">{title}</h3>
      <p className="mx-auto max-w-[40ch] text-[12.5px] leading-[1.55]">{body}</p>
    </div>
  );
}

export function ErrorState({
  kind,
  testId,
  detail,
  retryAfterS,
}: {
  kind: ApiErrorKind;
  testId: string;
  detail: string;
  retryAfterS?: number;
}) {
  const copy = ERROR_COPY[kind];
  const wait = retryAfterS === undefined ? "" : ` (retry in ${retryAfterS}s)`;
  return (
    <div data-testid={testId} className="mt-3 border border-bad/40 bg-bad-soft px-4 py-3.5">
      <h3 className="mb-1 font-serif text-[18px] font-normal text-bad">{copy.title}</h3>
      <p className="mb-2.5 max-w-[52ch] text-[12.5px] leading-[1.55] text-ink-2">{copy.body}</p>
      <span className="font-mono text-[10px] text-ink-3">
        {detail} -&gt; {kind}
        {wait}
      </span>
    </div>
  );
}
