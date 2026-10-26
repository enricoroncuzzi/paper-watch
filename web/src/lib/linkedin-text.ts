import type { Draft } from "@/lib/types";

/** The fields the clipboard text is built from; a saved draft always has all of them. */
export type LinkedInDraft = Pick<Draft, "hook" | "body" | "hashtags" | "source_link">;

/** One hash sign, no spaces: a stored tag never reads as "##ai agents" on the clipboard. */
function tag(value: string): string {
  return value.trim().replace(/^#+/, "").replace(/\s+/g, "");
}

/**
 * The "Copy for LinkedIn" text: hook, body, hashtags, source link, plain text only.
 *
 * The API's `chars` counts this exact string (its Python `draft_text` builds the same one), so a
 * change here must change `src/paperwatch/drafts/generate.py` too, or the count on a card stops
 * matching the clipboard.
 */
export function linkedinText(draft: LinkedInDraft): string {
  const tags = draft.hashtags
    .map(tag)
    .filter(Boolean)
    .map((value) => `#${value}`)
    .join(" ");
  const parts = [draft.hook, "", draft.body];
  if (tags) parts.push("", tags);
  if (draft.source_link) parts.push("", draft.source_link);
  return parts.join("\n").trim();
}
