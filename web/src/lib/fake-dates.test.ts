import { describe, expect, it } from "vitest";

import { fakeRunFinish, fakeRunStart, romeDayStart } from "../../fake-api/dates.mjs";
import { formatRome, runDayLabel } from "./format";

// The fake API's run dates must never cross a Rome midnight: the strip labels a run by its Rome
// calendar day, so a "today" run dated three hours back would read Yesterday from 00:00 to 03:00.
const CLOCKS = [
  // Winter (CET, +01:00).
  "2026-01-15T00:00:30+01:00",
  "2026-01-15T00:30:00+01:00",
  "2026-01-15T02:59:00+01:00",
  "2026-01-15T03:01:00+01:00",
  "2026-01-15T12:00:00+01:00",
  // Summer (CEST, +02:00).
  "2026-07-15T00:00:30+02:00",
  "2026-07-15T00:30:00+02:00",
  "2026-07-15T02:59:00+02:00",
  "2026-07-15T03:01:00+02:00",
  "2026-07-15T12:00:00+02:00",
  // The two DST nights themselves, just after the change.
  "2026-03-29T03:01:00+02:00",
  "2026-10-25T02:01:00+01:00",
];

describe("the fake API's run dates", () => {
  it.each(CLOCKS)("labels today's and yesterday's run at %s", (iso) => {
    const now = new Date(iso);
    const today = fakeRunStart(0, now.getTime());
    const yesterday = fakeRunStart(1, now.getTime());

    expect(runDayLabel(new Date(today).toISOString(), now)).toBe("Today");
    expect(runDayLabel(new Date(yesterday).toISOString(), now)).toBe("Yesterday");
    // The day's start itself is midnight Rome, which is what makes the label stable at any hour.
    expect(formatRome(new Date(romeDayStart(0, now.getTime())).toISOString())).toBe("00:00");
    expect(formatRome(new Date(romeDayStart(1, now.getTime())).toISOString())).toBe("00:00");
  });

  it("finishes three minutes later, never in the future", () => {
    for (const iso of CLOCKS) {
      const now = new Date(iso).getTime();
      const today = fakeRunStart(0, now);
      const finished = fakeRunFinish(today, now);
      expect(finished).toBeGreaterThanOrEqual(today);
      expect(finished).toBeLessThanOrEqual(now);
    }
  });

  it("keeps the three-hour look-back later in the day", () => {
    const noon = new Date("2026-01-15T12:00:00+01:00").getTime();
    // At noon the look-back wins, so the run reads like a 09:00 ingestion, not midnight.
    expect(formatRome(new Date(fakeRunStart(0, noon)).toISOString())).toBe("09:00");
  });
});
