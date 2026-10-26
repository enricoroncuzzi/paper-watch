import { beforeEach, describe, expect, it, vi } from "vitest";

// The proxy only saves an anonymous page load from rendering; each page must check the session
// itself. These tests call the page components with a guard that refuses, so deleting the guard
// from a page fails here even though the proxy would still redirect a browser.
const requireOwner = vi.hoisted(() =>
  vi.fn(async () => {
    throw new Error("NEXT_REDIRECT:/signin");
  }),
);

vi.mock("@/auth", () => ({
  requireOwner,
  signOut: vi.fn(),
}));
vi.mock("next/navigation", () => ({
  notFound: () => {
    throw new Error("NEXT_NOT_FOUND");
  },
}));
vi.mock("@/lib/api", () => ({
  apiGet: vi.fn(async () => ({
    ok: true,
    data: {
      run: null,
      search: { mode: "hybrid", reason: "ok", since: null },
      runs: [],
    },
  })),
}));

import Home from "./page";
import ItemPageRoute from "./items/[id]/page";
import RunPage from "./runs/[id]/page";

const RUN_ID = "11111111-1111-4111-8111-111111111111";

beforeEach(() => {
  // A hoisted mock is not cleared between tests in this runner, so each count below covers one
  // render: a page that guarded twice would still fail here.
  requireOwner.mockClear();
});

describe("the per-page owner guard", () => {
  it("the home page refuses an anonymous render", async () => {
    await expect(Home()).rejects.toThrow("NEXT_REDIRECT:/signin");
    expect(requireOwner).toHaveBeenCalledTimes(1);
  });

  it("the run page refuses an anonymous render", async () => {
    await expect(RunPage({ params: Promise.resolve({ id: RUN_ID }) })).rejects.toThrow(
      "NEXT_REDIRECT:/signin",
    );
    // The count covers this render alone (see `beforeEach`): the run page must call the guard
    // exactly once, not lean on another test having called it before.
    expect(requireOwner).toHaveBeenCalledTimes(1);
  });

  it("the item page refuses an anonymous render", async () => {
    await expect(ItemPageRoute({ params: Promise.resolve({ id: RUN_ID }) })).rejects.toThrow(
      "NEXT_REDIRECT:/signin",
    );
    expect(requireOwner).toHaveBeenCalledTimes(1);
  });
});
