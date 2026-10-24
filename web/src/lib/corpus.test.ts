import { describe, expect, it } from "vitest";

import { basisWord, compactCount, publishedDay, signalLine, tierMove, TIER_TABS } from "./corpus";
import type { CorpusItem } from "./types";

function row(overrides: Partial<CorpusItem> = {}): CorpusItem {
  return {
    id: "11111111-1111-4111-8111-111111111111",
    title: "Replay Buffers Beat Summaries for Long-Running Coding Agents",
    source: "arxiv",
    tier: "peer_reviewed",
    basis: "venue",
    basis_label: "peer-reviewed via venue: ICLR 2027",
    profile: "agentic-engineering",
    venue: "ICLR 2027",
    citation_count: 31,
    hf_upvotes: null,
    github_stars: 2400,
    published_at: "2026-10-20T00:00:00Z",
    ...overrides,
  };
}

describe("the corpus labels", () => {
  it("prints the strongest number a row has", () => {
    expect(signalLine(row())).toBe("31 cites");
    expect(signalLine(row({ citation_count: 1 }))).toBe("1 cite");
    expect(signalLine(row({ hf_upvotes: 63 }))).toBe("HF 63 \u2191");
    // The number follows the rule the row's basis names.
    expect(signalLine(row({ basis: "citations", citation_count: 9, hf_upvotes: 4 }))).toBe("9 cites");
    expect(signalLine(row({ basis: "hf_upvotes", citation_count: 9, hf_upvotes: 4 }))).toBe(
      "HF 4 \u2191",
    );
    expect(signalLine(row({ citation_count: 0 }))).toBe("2.4k \u2605");
    expect(signalLine(row({ citation_count: 0, github_stars: null }))).toBe("n/a");
  });

  it("prints a big number as a magnitude", () => {
    expect(compactCount(999)).toBe("999");
    expect(compactCount(1000)).toBe("1k");
    expect(compactCount(2450)).toBe("2.5k");
  });

  it("keeps the published date in the item's own UTC day", () => {
    expect(publishedDay("2026-10-20T23:30:00Z")).toBe("2026-10-20");
  });

  it("never shows a database enum for a rule", () => {
    expect(basisWord("venue")).toBe("venue rule");
    expect(basisWord("hf_upvotes")).toBe("upvotes rule");
    expect(basisWord("lab_author")).toBe("lab authors");
    expect(basisWord("something_new")).toBe("admitted");
  });

  it("writes a tier move as a sentence", () => {
    expect(
      tierMove({
        from_tier: null,
        to_tier: "vetted_preprint",
        basis: "seed",
        changed_at: "2026-10-20T00:00:00Z",
      }),
    ).toBe("first seen to vetted preprint");
    expect(
      tierMove({
        from_tier: "vetted_preprint",
        to_tier: "peer_reviewed",
        basis: "venue",
        changed_at: "2026-10-22T00:00:00Z",
      }),
    ).toBe("vetted preprint to peer-reviewed");
  });

  it("offers the four tiers as tabs, All first", () => {
    expect(TIER_TABS.map((tab) => tab.label)).toEqual([
      "All",
      "Peer reviewed",
      "Vetted preprints",
      "Lab blogs",
    ]);
  });
});
