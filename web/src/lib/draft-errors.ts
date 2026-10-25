import { formatRome } from "@/lib/format";

/**
 * The sentence a failed draft request shows, from the API's own body.
 *
 * The dialog and a regenerate share it so a daily-cap refusal always carries its reset time, and
 * a 429 always says how long to wait, wherever the reader met it.
 */
export function draftFailureMessage(status: number, body: unknown): string {
  const payload = body as
    | { message?: unknown; code?: unknown; retry_at?: unknown; retryAfterS?: unknown }
    | null;
  if (status === 503 && typeof payload?.message === "string") {
    if (payload.code === "daily_cap" && typeof payload.retry_at === "string") {
      return `${payload.message} It resets at ${formatRome(payload.retry_at)} Europe/Rome.`;
    }
    // model_unavailable and unusable are shown as the API wrote them, including the privacy
    // refusal sentence, so the page cannot soften a promise about the point of view.
    return payload.message;
  }
  if (status === 429) {
    return typeof payload?.retryAfterS === "number"
      ? `Too many drafts right now. Try again in ${payload.retryAfterS}s.`
      : "Too many drafts right now. Try again shortly.";
  }
  if (status === 401 || status === 403) {
    return "Your session is not accepted any more. Sign in again.";
  }
  if (status === 404) return "That item is no longer in the corpus.";
  if (status === 422) return "That item does not have enough stored text for a draft.";
  if (status === 413) return "That point of view is too long.";
  return "The draft could not be generated. Try again.";
}
