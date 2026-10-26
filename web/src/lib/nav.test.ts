import { describe, expect, it } from "vitest";

import { currentSection, sectionHashFromClick, type ClickTarget } from "./nav";

const HERE = { href: "https://paperwatch.example/" };

function click(overrides: Partial<ClickTarget> = {}): ClickTarget {
  return {
    href: "/#corpus",
    target: null,
    button: 0,
    metaKey: false,
    ctrlKey: false,
    shiftKey: false,
    altKey: false,
    ...overrides,
  };
}

describe("the masthead's current section", () => {
  it("follows the hash on the home page, This week by default", () => {
    expect(currentSection("/", "")).toBe("picks");
    expect(currentSection("/", "#picks")).toBe("picks");
    expect(currentSection("/", "#corpus")).toBe("corpus");
    expect(currentSection("/", "#drafts")).toBe("drafts");
  });

  it("marks no section for a hash that names none", () => {
    // A citation's "#fn-S1" or a stored link's "#results" is not a section: claiming This week
    // would be a mark the reader never asked for.
    expect(currentSection("/", "#changelog")).toBeNull();
    expect(currentSection("/", "#fn-S1")).toBeNull();
  });

  it("marks Runs on a run detail and nothing on an item page", () => {
    expect(currentSection("/runs", "")).toBe("runs");
    expect(currentSection("/runs/1234", "")).toBe("runs");
    expect(currentSection("/items/1234", "")).toBeNull();
    expect(currentSection("/items/1234", "#corpus")).toBeNull();
  });
});

describe("the hash a click moves the mark to", () => {
  it("takes a section hash from a plain left click on an in-page link", () => {
    expect(sectionHashFromClick(click(), HERE)).toBe("#corpus");
    expect(sectionHashFromClick(click({ href: "#runs" }), HERE)).toBe("#runs");
    expect(sectionHashFromClick(click({ href: "https://paperwatch.example/#drafts" }), HERE)).toBe(
      "#drafts",
    );
  });

  it("ignores a click that opens somewhere else", () => {
    // A cmd-click opens a background tab, a target=_blank link opens a new window, and a link on
    // another origin is not this page: none of them moves the mark here.
    expect(sectionHashFromClick(click({ metaKey: true }), HERE)).toBeNull();
    expect(sectionHashFromClick(click({ ctrlKey: true }), HERE)).toBeNull();
    expect(sectionHashFromClick(click({ shiftKey: true }), HERE)).toBeNull();
    expect(sectionHashFromClick(click({ altKey: true }), HERE)).toBeNull();
    expect(sectionHashFromClick(click({ button: 1 }), HERE)).toBeNull();
    expect(sectionHashFromClick(click({ target: "_blank" }), HERE)).toBeNull();
    expect(
      sectionHashFromClick(click({ href: "https://example.org/post#results" }), HERE),
    ).toBeNull();
    expect(sectionHashFromClick(click({ href: "mailto:someone@example.org#x" }), HERE)).toBeNull();
  });

  it("ignores a link with no hash and a hash that is no section", () => {
    expect(sectionHashFromClick(click({ href: "/" }), HERE)).toBeNull();
    expect(sectionHashFromClick(click({ href: null }), HERE)).toBeNull();
    expect(sectionHashFromClick(click({ href: "/#changelog" }), HERE)).toBeNull();
    expect(sectionHashFromClick(click({ href: "/#fn-S1" }), HERE)).toBeNull();
  });

  it("accepts an explicit _self target and a bare fragment", () => {
    expect(sectionHashFromClick(click({ target: "_self" }), HERE)).toBe("#corpus");
  });
});
