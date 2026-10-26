import { describe, expect, it } from "vitest";

import { linkedinText } from "@/lib/linkedin-text";

const DRAFT = {
  hook: "Most agent harnesses forget by summarising.",
  body: "A replay buffer cut tokens by 41% on 40 tasks.\n\nThe code is open.",
  hashtags: ["agents", "memory"],
  source_link: "https://arxiv.org/abs/2610.04412",
};

describe("linkedin text", () => {
  it("joins hook body hashtags and link", () => {
    // The exact text the "Copy for LinkedIn" button puts on the clipboard, and the string the
    // API's `chars` counts: blank lines between blocks, nothing else.
    expect(linkedinText(DRAFT)).toBe(
      "Most agent harnesses forget by summarising.\n\n" +
        "A replay buffer cut tokens by 41% on 40 tasks.\n\nThe code is open.\n\n" +
        "#agents #memory\n\n" +
        "https://arxiv.org/abs/2610.04412",
    );
  });

  it("hashtags gain one hash sign and no spaces", () => {
    // A stored tag never shows as "##ai agents": the page owns the hash sign and the separator.
    expect(
      linkedinText({ ...DRAFT, hashtags: ["agents", "#memory", "ai safety", " #llm "] }),
    ).toContain("#agents #memory #aisafety #llm");
  });

  it("omits the empty blocks instead of leaving blank lines", () => {
    expect(linkedinText({ hook: "A hook.", body: "A body.", hashtags: [], source_link: "" })).toBe(
      "A hook.\n\nA body.",
    );
  });
});
