import Link from "next/link";

/**
 * A link to something that is not here (an unknown run id, a mistyped UUID) gets the page's own
 * empty state instead of Next's default 404: same palette, same type, one way back.
 */
export default function NotFound() {
  return (
    <main className="mx-auto flex min-h-dvh w-full max-w-[35rem] flex-col justify-center px-6 py-16">
      <p className="font-mono text-[11px] uppercase tracking-[0.14em] text-ink-3">404</p>
      <h1 className="mt-3 font-serif text-4xl font-medium tracking-[-0.02em]">Nothing here</h1>
      <p className="mt-4 text-[15px] leading-7 text-ink-2">
        That page is not part of the reading room. The corpus and its runs are one link away.
      </p>
      <Link href="/" className="mt-6 font-mono text-[11px] text-accent-text">
        Back to the page
      </Link>
    </main>
  );
}
