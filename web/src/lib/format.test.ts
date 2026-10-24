import { describe, expect, it } from "vitest";

import {
  compactTokens,
  feedCount,
  followupLine,
  formatDay,
  formatRome,
  runDayLabel,
  stripStatus,
  verdict,
} from "./format";
import type { RunSummary } from "./types";

function run(overrides: Partial<RunSummary> = {}): RunSummary {
  return {
    id: "11111111-1111-4111-8111-111111111111",
    started_at: "2026-10-24T04:00:00Z",
    finished_at: "2026-10-24T04:03:12Z",
    status: "succeeded",
    healthy: true,
    problems: [],
    warnings: [],
    counts: {
      fetched: 41,
      fetched_arxiv: 32,
      stored: 9,
      new: 4,
      irrelevant: 21,
      failed: 0,
      admitted_late: 1,
      followup_passes: 0,
      s2_refused: 0,
    },
    tokens: 812,
    followup: null,
    ...overrides,
  };
}

describe("formatting", () => {
  it("shows the run time in Europe/Rome, not UTC", () => {
    // October is CEST (UTC+2), so the 04:00 UTC run is the 06:00 the captain sees.
    expect(formatRome("2026-10-24T04:00:00Z")).toBe("06:00");
    expect(formatRome(null)).toBe("");
    expect(formatDay("2026-10-24T04:00:00Z")).toBe("24 Oct");
  });

  it("compacts tokens", () => {
    expect(compactTokens(812)).toBe("812");
    expect(compactTokens(38400)).toBe("38.4k");
  });

  it("counts feeds as fetched minus arxiv", () => {
    expect(feedCount(run().counts)).toBe(9);
    // A run that counted fewer arxiv items than fetched must not go negative.
    expect(feedCount({ ...run().counts, fetched: 2, fetched_arxiv: 5 })).toBe(0);
  });

  it("reads a missing verdict as unknown", () => {
    expect(verdict(run())).toBe("healthy");
    expect(verdict(run({ healthy: false }))).toBe("unhealthy");
    expect(verdict(run({ healthy: null }))).toBe("unknown");
  });

  it("colors the strip dot by the worst fact", () => {
    expect(stripStatus(run())).toBe("ok");
    expect(stripStatus(run({ warnings: ["a feed was quiet"] }))).toBe("warn");
    expect(stripStatus(run({ healthy: null }))).toBe("warn");
    expect(stripStatus(run({ healthy: false, problems: ["x"] }))).toBe("bad");
  });

  it("labels a run from an earlier day", () => {
    const now = new Date("2026-10-24T09:00:00Z");
    expect(runDayLabel("2026-10-24T04:00:00Z", now)).toBe("Today");
    expect(runDayLabel("2026-10-23T04:00:00Z", now)).toBe("Yesterday");
    expect(runDayLabel("2026-10-21T04:00:00Z", now)).toBe("21 Oct");
    // 23:30 UTC on the 23rd is already the 24th in Rome, so it is today's run.
    expect(runDayLabel("2026-10-23T23:30:00Z", now)).toBe("Today");
  });

  it("writes the waiting follow-up line", () => {
    expect(
      followupLine({
        pass: 3,
        of: 40,
        next_at: "2026-10-24T08:30:00Z",
        pending_s2: 12,
      }),
    ).toBe("follow-up pass 3 of 40 at 10:30, 12 papers pending");
    expect(followupLine(null)).toBeNull();
  });
});
