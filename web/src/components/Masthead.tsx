import Link from "next/link";
import type { Session } from "next-auth";

import { signOut } from "@/auth";
import { CommandPalette } from "@/components/CommandPalette";
import { ThemeToggle } from "@/components/ThemeToggle";

async function signOutAction() {
  "use server";
  await signOut({ redirectTo: "/signin" });
}

function initials(session: Session): string {
  const name = session.user?.name ?? session.githubId ?? "?";
  const parts = name.trim().split(/\s+/);
  return (parts[0]?.[0] ?? "?") + (parts[1]?.[0] ?? parts[0]?.[1] ?? "");
}

/**
 * The masthead carries only the sections this release has: This week, Drafts and Corpus arrive
 * with their slices. The avatar opens a small menu with the sign-out.
 */
export function Masthead({ session }: { session: Session }) {
  const image = session.user?.image;
  const label = session.user?.name ?? session.githubId ?? "the owner";
  return (
    <header className="sticky top-0 z-40 flex flex-wrap items-center gap-x-4 gap-y-1 border-b border-rule bg-paper/90 py-2 backdrop-blur-md sm:flex-nowrap sm:py-3">
      <Link
        href="/"
        className="flex min-h-11 items-center font-serif text-[19px] font-medium tracking-[-0.02em]"
      >
        Paper Watch
      </Link>
      <nav
        aria-label="Sections"
        className="order-3 flex w-full gap-4 text-[12px] text-ink-2 sm:order-none sm:w-auto"
      >
        <Link
          href="/#picks"
          aria-current="true"
          className="relative flex min-h-11 min-w-11 items-center justify-center text-ink sm:min-h-0 sm:min-w-0 sm:py-0.5"
        >
          This week
          <span aria-hidden="true" className="absolute inset-x-0 bottom-1 h-[1.5px] bg-accent" />
        </Link>
        <Link
          href="/#rising"
          className="flex min-h-11 min-w-11 items-center justify-center hover:text-ink sm:min-h-0 sm:min-w-0"
        >
          Rising
        </Link>
        <Link
          href="/#runs"
          className="flex min-h-11 min-w-11 items-center justify-center hover:text-ink sm:min-h-0 sm:min-w-0"
        >
          Runs
        </Link>
      </nav>
      <div className="ml-auto flex items-center gap-1.5 text-[11.5px] text-ink-3">
        <span className="hidden items-center gap-1.5 rounded-full border border-rule px-2 py-0.5 text-[10.5px] text-ink-2 sm:flex">
          <svg
            width="12"
            height="12"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            aria-hidden="true"
          >
            <rect x="4.5" y="10.5" width="15" height="9.5" rx="2" />
            <path d="M8 10.5V8a4 4 0 0 1 8 0v2.5" />
          </svg>
          Private · GitHub
        </span>
        <CommandPalette />
        <ThemeToggle />
        <details className="relative">
          <summary
            aria-label={`Signed in as ${label}`}
            className="flex min-h-11 min-w-11 cursor-pointer list-none items-center justify-center sm:min-h-0 sm:min-w-0"
          >
            {image ? (
              // A plain img: the avatar host is not in next.config, and the browser sends no
              // referrer from this page anyway.
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={image}
                alt=""
                width={26}
                height={26}
                referrerPolicy="no-referrer"
                className="h-[26px] w-[26px] rounded-full border border-rule-2"
              />
            ) : (
              <span className="grid h-[26px] w-[26px] place-items-center rounded-full border border-rule-2 bg-card-2 font-mono text-[10px] uppercase text-ink-2">
                {initials(session)}
              </span>
            )}
          </summary>
          <div className="absolute right-0 top-full z-50 mt-1 w-[190px] border border-rule-2 bg-card p-3 shadow-[var(--shadow)]">
            <p className="mb-2 border-b border-rule pb-2 text-[10.5px] leading-4 text-ink-3">
              Signed in as {label}
            </p>
            <form action={signOutAction}>
              <button
                type="submit"
                className="flex min-h-11 w-full items-center text-[12.5px] text-ink-2 hover:text-ink"
              >
                Sign out
              </button>
            </form>
          </div>
        </details>
      </div>
    </header>
  );
}
