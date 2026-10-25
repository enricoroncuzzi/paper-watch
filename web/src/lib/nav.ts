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
  return match ? match.id : "picks";
}

export { SECTIONS };
