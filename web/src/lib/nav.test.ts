import { describe, expect, it } from "vitest";

import { currentSection } from "./nav";

describe("the masthead's current section", () => {
  it("follows the hash on the home page, This week by default", () => {
    expect(currentSection("/", "")).toBe("picks");
    expect(currentSection("/", "#picks")).toBe("picks");
    expect(currentSection("/", "#corpus")).toBe("corpus");
    expect(currentSection("/", "#drafts")).toBe("drafts");
    // A hash that names nothing is not a section, and the top of the page is the first one.
    expect(currentSection("/", "#changelog")).toBe("picks");
  });

  it("marks Runs on a run detail and nothing on an item page", () => {
    expect(currentSection("/runs", "")).toBe("runs");
    expect(currentSection("/runs/1234", "")).toBe("runs");
    expect(currentSection("/items/1234", "")).toBeNull();
    expect(currentSection("/items/1234", "#corpus")).toBeNull();
  });
});
