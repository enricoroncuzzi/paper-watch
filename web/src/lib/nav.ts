const SECTIONS = [
  { id: "picks", href: "/#picks", label: "This week" },
  { id: "rising", href: "/#rising", label: "Rising" },
  { id: "drafts", href: "/#drafts", label: "Drafts" },
  { id: "corpus", href: "/#corpus", label: "Corpus" },
  { id: "runs", href: "/#runs", label: "Runs" },
] as const;

export type SectionId = (typeof SECTIONS)[number]["id"];

/**
 * The section the reader is looking at, from the path and the hash. The home page shows all of
 * them, so the hash decides; a run detail belongs to Runs, and an item page belongs to no section,
 * so no link claims to be the current location there.
 */
export function currentSection(pathname: string, hash: string): SectionId | null {
  if (pathname === "/runs" || pathname.startsWith("/runs/")) return "runs";
  if (pathname !== "/") return null;
  const id = hash.replace(/^#/, "");
  const match = SECTIONS.find((section) => section.id === id);
  // An empty hash is the top of the page, which is This week; any other unknown hash is a place
  // that is not a section (a citation "fn-S1", a stored link's "results"), so nothing is marked.
  return match ? match.id : id === "" ? "picks" : null;
}

/** What a click on a link carries, as the guard below reads it. */
export type ClickTarget = {
  href: string | null;
  target: string | null;
  button: number;
  metaKey: boolean;
  ctrlKey: boolean;
  shiftKey: boolean;
  altKey: boolean;
};

/**
 * The section hash a click moves the mark to, or null when the click does not land on this page:
 * a modifier or middle click opens another tab, `target` opens a window, another origin is not
 * this page, and a hash that names no section is not a place. A prevented default is deliberately
 * not a guard: Link prevents the click it handles, which is exactly the click that does land here.
 */
export function sectionHashFromClick(click: ClickTarget, here: { href: string }): string | null {
  if (click.button !== 0) return null;
  if (click.metaKey || click.ctrlKey || click.shiftKey || click.altKey) return null;
  if (click.target !== null && click.target !== "" && click.target !== "_self") return null;
  if (!click.href) return null;
  let url: URL;
  try {
    url = new URL(click.href, here.href);
  } catch {
    return null;
  }
  if (url.origin !== new URL(here.href).origin) return null;
  const id = url.hash.replace(/^#/, "");
  return SECTIONS.some((section) => section.id === id) ? url.hash : null;
}

export { SECTIONS };
