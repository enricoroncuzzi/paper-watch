import "server-only";

import type { components } from "@/lib/api-types";
import { apiToken } from "@/lib/api-token";

/**
 * What a section renders instead of throwing. A failed read is a state on the page, not an error
 * page, so every status maps to one of these kinds and the UI owns the copy.
 */
export type ApiErrorKind = "unauthorized" | "rate_limited" | "unavailable" | "not_found" | "error";

export type ApiResult<T> =
  | { ok: true; data: T }
  | { ok: false; kind: ApiErrorKind; retryAfterS?: number };

/** Reads are a page load, not an answer: ten seconds is already a slow corpus query. */
export const READ_TIMEOUT_MS = 10_000;
/** A write can wait longer than a read, but still not forever. */
export const WRITE_TIMEOUT_MS = 20_000;

function kindFor(status: number): ApiErrorKind {
  if (status === 401 || status === 403) return "unauthorized";
  if (status === 404) return "not_found";
  if (status === 429) return "rate_limited";
  if (status >= 500) return "unavailable";
  return "error";
}

function retryAfter(response: Response): number | undefined {
  const header = response.headers.get("retry-after");
  if (!header) return undefined;
  const seconds = Number(header);
  return Number.isFinite(seconds) && seconds >= 0 ? seconds : undefined;
}

/** The gate's declared refusal bodies (`Refusal` for 400/401/403/413/415, `RateLimited` for 429). */
type RefusalBody = components["schemas"]["Refusal"];
type RateLimitedBody = components["schemas"]["RateLimited"];

async function declaredRefusal(
  response: Response,
): Promise<RefusalBody | RateLimitedBody | null> {
  try {
    return (await response.json()) as RefusalBody | RateLimitedBody;
  } catch {
    return null;
  }
}

async function call<T>(
  method: "GET" | "POST",
  path: string,
  options: { body?: unknown; timeoutMs: number },
): Promise<ApiResult<T>> {
  // The API's address lives only here, on the server; there is no NEXT_PUBLIC_ variable.
  const base = process.env.PW_API_URL;
  if (!base) return { ok: false, kind: "unavailable" };
  let token: string;
  try {
    token = await apiToken();
  } catch {
    // A missing or misconfigured credential is the same state to the page as a refused one.
    return { ok: false, kind: "unauthorized" };
  }
  // An explicit controller rather than `AbortSignal.timeout`: a test can advance its clock.
  const timeout = new AbortController();
  const timer = setTimeout(() => timeout.abort(), options.timeoutMs);
  try {
    const response = await fetch(new URL(path, base), {
      method,
      headers: {
        authorization: `Bearer ${token}`,
        ...(options.body === undefined ? {} : { "content-type": "application/json" }),
      },
      body: options.body === undefined ? undefined : JSON.stringify(options.body),
      cache: "no-store",
      signal: timeout.signal,
    });
    if (!response.ok) {
      // The wait is declared in the body as `retry_after_s` and repeated in `Retry-After`; the
      // header wins, and a body-only refusal still tells the reader when to try again.
      const body = await declaredRefusal(response);
      const declaredWait =
        body && "retry_after_s" in body ? body.retry_after_s : undefined;
      return {
        ok: false,
        kind: kindFor(response.status),
        retryAfterS: retryAfter(response) ?? declaredWait,
      };
    }
    return { ok: true, data: (await response.json()) as T };
  } catch {
    // A timeout, a refused connection and a body that is not JSON all read as "unreachable".
    return { ok: false, kind: "unavailable" };
  } finally {
    clearTimeout(timer);
  }
}

export function apiGet<T>(path: string): Promise<ApiResult<T>> {
  return call<T>("GET", path, { timeoutMs: READ_TIMEOUT_MS });
}

/** For the write routes the page adds later (drafts in slice 5); reads use `apiGet`. */
export function apiPost<T>(path: string, body: unknown): Promise<ApiResult<T>> {
  return call<T>("POST", path, { body, timeoutMs: WRITE_TIMEOUT_MS });
}
