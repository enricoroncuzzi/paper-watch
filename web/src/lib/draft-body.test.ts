import { describe, expect, it } from "vitest";

import { draftParagraphs, previewParagraphs } from "@/lib/draft-body";

describe("draft body", () => {
  it("splits on blank lines and keeps single newlines inside a paragraph", () => {
    expect(draftParagraphs("First line\nstill first.\n\nSecond paragraph.")).toEqual([
      "First line\nstill first.",
      "Second paragraph.",
    ]);
  });

  it("keeps whole paragraphs up to the limit", () => {
    const body = "One short paragraph.\n\nAnother short paragraph.\n\nA third one.";
    expect(previewParagraphs(body, 40)).toEqual({
      paragraphs: ["One short paragraph."],
      truncated: true,
    });
  });

  it("cuts a long first paragraph at a sentence boundary, never mid-sentence", () => {
    const body =
      "This sentence is long enough to pass the limit on its own. A second sentence follows it.";
    const { paragraphs, truncated } = previewParagraphs(body, 44);
    expect(truncated).toBe(true);
    expect(paragraphs).toHaveLength(1);
    expect(paragraphs[0]?.endsWith(".")).toBe(true);
    expect(paragraphs[0]).toBe("This sentence is long enough to pass the limit on its own.");
  });
});
