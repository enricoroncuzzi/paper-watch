import { describe, expect, it } from "vitest";

import { safeHref, safeHost } from "./safe-href";

describe("safeHref", () => {
  it("keeps http and https URLs", () => {
    expect(safeHref("https://arxiv.org/abs/2601.00001")).toBe("https://arxiv.org/abs/2601.00001");
    expect(safeHref("http://example.org/a?b=c")).toBe("http://example.org/a?b=c");
  });

  it("refuses non-http schemes", () => {
    expect(safeHref("javascript:alert(1)")).toBeUndefined();
    expect(safeHref("data:text/html,<script>alert(1)</script>")).toBeUndefined();
    expect(safeHref("vbscript:msgbox(1)")).toBeUndefined();
    expect(safeHref("file:///etc/passwd")).toBeUndefined();
  });

  it("refuses relative and malformed values", () => {
    expect(safeHref("/runs")).toBeUndefined();
    expect(safeHref("arxiv.org/abs/1")).toBeUndefined();
    expect(safeHref("https://")).toBeUndefined();
    expect(safeHref("")).toBeUndefined();
    expect(safeHref(null)).toBeUndefined();
    expect(safeHref(undefined)).toBeUndefined();
  });

  it("returns the parser's normalised href", () => {
    // A tab or newline inside a stored URL is dropped by the parser, not passed to the browser.
    expect(safeHref("https://example.org\t/abs")).toBe("https://example.org/abs");
  });

  it("refuses an invisible format character and an impossible port", () => {
    // A zero-width space and a bidi override are what a hostile stored URL uses to lie.
    expect(safeHref("https://example.org/\u200babs")).toBeUndefined();
    expect(safeHref("https://example.org/\u202eabc")).toBeUndefined();
    expect(safeHref("https://example.org:99999999/x")).toBeUndefined();
  });

  it("shows the host, not the path, as the link text", () => {
    expect(safeHost("https://github.com/acme/repo")).toBe("github.com");
    expect(safeHost("javascript:alert(1)")).toBeUndefined();
  });
});
