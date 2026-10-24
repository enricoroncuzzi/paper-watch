/**
 * The model writes one body paragraph per blank line. A blank line is a paragraph break, not a
 * bigger gap: rendering the raw text with `pre-line` would turn it into an empty 28 px line.
 */
export function draftParagraphs(body: string): string[] {
  return body
    .split(/\n{2,}/)
    .map((paragraph) => paragraph.trim())
    .filter(Boolean);
}

/**
 * The collapsed card's body. It keeps whole paragraphs up to `limit`, or, when the first
 * paragraph is already too long, whole sentences from it, so the cut never lands mid-sentence.
 */
export function previewParagraphs(
  body: string,
  limit = 220,
): { paragraphs: string[]; truncated: boolean } {
  const all = draftParagraphs(body);
  const kept: string[] = [];
  let used = 0;
  for (const paragraph of all) {
    if (kept.length > 0 && used + paragraph.length > limit) break;
    if (kept.length === 0 && paragraph.length > limit) {
      let first = "";
      for (const sentence of paragraph.split(/(?<=[.!?])\s+/)) {
        if (first && first.length + 1 + sentence.length > limit) break;
        first = first ? `${first} ${sentence}` : sentence;
      }
      kept.push(first);
      return { paragraphs: kept, truncated: true };
    }
    kept.push(paragraph);
    used += paragraph.length + 2;
  }
  return { paragraphs: kept, truncated: kept.length < all.length };
}
