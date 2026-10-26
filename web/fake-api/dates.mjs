// The fake API's run dates, in one place so a unit test can evaluate the formula at any clock.
// The strip labels a run by its Rome calendar day, so a "today" run must never be dated before
// midnight Rome, whatever the hour the page is opened.

/** The offset of Europe/Rome from UTC at an instant, in milliseconds (positive east). */
function romeOffsetMs(ms) {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: "Europe/Rome",
    hourCycle: "h23",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).formatToParts(new Date(ms));
  const value = (type) => Number(parts.find((part) => part.type === type).value);
  const asUtc = Date.UTC(
    value("year"),
    value("month") - 1,
    value("day"),
    value("hour"),
    value("minute"),
    value("second"),
  );
  return asUtc - Math.floor(ms / 1000) * 1000;
}

/** The Rome calendar date that is `daysAgo` days before `nowMs`. */
function romeDate(daysAgo, nowMs) {
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone: "Europe/Rome",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).formatToParts(new Date(nowMs));
  const value = (type) => Number(parts.find((part) => part.type === type).value);
  // Count back on the calendar, not in 24-hour steps: a Rome day lasts 23 or 25 hours at a DST change.
  const date = new Date(Date.UTC(value("year"), value("month") - 1, value("day") - daysAgo));
  return { year: date.getUTCFullYear(), month: date.getUTCMonth() + 1, day: date.getUTCDate() };
}

/** The instant of 00:00 Rome on that date. Two passes settle the offset across a DST change. */
export function romeDayStart(daysAgo, nowMs) {
  const { year, month, day } = romeDate(daysAgo, nowMs);
  let start = Date.UTC(year, month - 1, day);
  for (let pass = 0; pass < 2; pass += 1) {
    start = Date.UTC(year, month - 1, day) - romeOffsetMs(start);
  }
  return start;
}

/**
 * When a fixture's run started: the later of the start of its Rome day and three hours before now,
 * so a "today" run is always on today's Rome date and a "yesterday" run always on yesterday's.
 */
export function fakeRunStart(daysAgo, nowMs) {
  return Math.max(
    romeDayStart(daysAgo, nowMs),
    nowMs - daysAgo * 24 * 60 * 60 * 1000 - 3 * 60 * 60 * 1000,
  );
}

/** A three-minute run, never in the future. */
export function fakeRunFinish(startedMs, nowMs) {
  return Math.min(startedMs + 3 * 60 * 1000, nowMs);
}
