"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

import { currentSection, sectionHashFromClick, SECTIONS } from "@/lib/nav";

/**
 * The section links. Which one is current depends on the path and the hash, so it is decided after
 * mount: a section hash is a place on the page, not a route.
 */
export function MastheadNav() {
  const pathname = usePathname();
  const [hash, setHash] = useState("");

  useEffect(() => {
    const read = () => setHash(window.location.hash);
    read();
    window.addEventListener("hashchange", read);
    // An in-page link is a client navigation: the hash changes without a hashchange event, so the
    // mark is taken from the link the reader pressed. This also covers the strip's "Runs" link,
    // and `sectionHashFromClick` keeps a click that opens a tab or another site from moving it.
    function onClick(event: MouseEvent) {
      const anchor = (event.target as Element | null)?.closest?.("a[href]");
      if (!anchor) return;
      const target = sectionHashFromClick(
        {
          href: anchor.getAttribute("href"),
          target: anchor.getAttribute("target"),
          button: event.button,
          metaKey: event.metaKey,
          ctrlKey: event.ctrlKey,
          shiftKey: event.shiftKey,
          altKey: event.altKey,
        },
        window.location,
      );
      if (target) setHash(target);
    }
    document.addEventListener("click", onClick);
    return () => {
      window.removeEventListener("hashchange", read);
      document.removeEventListener("click", onClick);
    };
  }, [pathname]);

  const current = currentSection(pathname, hash);
  return (
    <nav
      aria-label="Sections"
      className="order-3 flex w-full gap-4 text-[12px] text-ink-2 sm:order-none sm:w-auto"
    >
      {SECTIONS.map((section) => (
        <Link
          key={section.id}
          href={section.href}
          aria-current={current === section.id ? "true" : undefined}
          className={`relative flex min-h-11 min-w-11 items-center justify-center sm:min-h-0 sm:min-w-0 sm:py-0.5 ${
            current === section.id ? "text-ink" : "hover:text-ink"
          }`}
        >
          {section.label}
          {current === section.id ? (
            <span aria-hidden="true" className="absolute inset-x-0 bottom-1 h-[1.5px] bg-accent" />
          ) : null}
        </Link>
      ))}
    </nav>
  );
}
