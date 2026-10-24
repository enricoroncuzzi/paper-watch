"use client";

import { useEffect, useMemo, useSyncExternalStore } from "react";

import { parseSse } from "@/lib/sse";

/** One source the API registered for this answer, with its stable alias (S1, S2, ...). */
export type SourceRef = {
  id: string;
  item_id: string;
  title: string;
  url: string;
  tier: string;
  basis: string;
};

export type AnswerPayload = {
  text: string;
  search_mode: "hybrid" | "text_only";
  supported: boolean;
  citations: SourceRef[];
  flags: { kind: string; sentence: string; detail: string }[];
};

export type NoticePayload = { code: "text_only"; message: string };
export type ToolPayload = { name: string; args: Record<string, unknown> };
export type StepPayload = { round: number; ms: number; new_sources: number };

/** Every failure the sheet knows how to phrase. `dropped` is the client's own watchdog. */
export type AskErrorCode =
  | "daily_cap"
  | "budget"
  | "model_unavailable"
  | "internal"
  | "deadline"
  | "rate_limited"
  | "busy"
  | "unauthorized"
  | "unavailable"
  | "dropped";

export type AskError = {
  code: AskErrorCode;
  message: string;
  retryAt: string | null;
  retryAfterS?: number;
};

export type AskStatus = "idle" | "streaming" | "done" | "error" | "dropped";

export type AskState = {
  status: AskStatus;
  question: string;
  tools: ToolPayload[];
  steps: StepPayload[];
  sources: SourceRef[];
  text: string;
  notice: NoticePayload | null;
  answer: AnswerPayload | null;
  error: AskError | null;
};

/**
 * Footnote positions by source alias. The API's `citations` hold only the sources the answer
 * cites, in first-cited order, so the number a reader sees is the position in that list, not the
 * alias number: an answer citing [S2] then [S1] has footnotes 1 and 2.
 */
export function footnoteNumbers(citations: SourceRef[]): Map<string, number> {
  return new Map(citations.map((citation, index) => [citation.id, index + 1]));
}

/** 45 s without a single received byte means the connection is gone, not thinking. */
export const WATCHDOG_MS = 45_000;

export const INITIAL_ASK_STATE: AskState = {
  status: "idle",
  question: "",
  tools: [],
  steps: [],
  sources: [],
  text: "",
  notice: null,
  answer: null,
  error: null,
};

export type AskAction =
  | { type: "start"; question: string }
  | { type: "event"; event: string; data: unknown }
  | { type: "dropped" }
  | { type: "abort" };

/** The reducer is exported so the stream logic is tested without rendering a component. */
export function askReducer(state: AskState, action: AskAction): AskState {
  switch (action.type) {
    case "start":
      return { ...INITIAL_ASK_STATE, status: "streaming", question: action.question };
    case "dropped":
      return state.status === "streaming" ? { ...state, status: "dropped" } : state;
    case "abort":
      return state.status === "streaming" ? { ...state, status: "idle" } : state;
    case "event":
      return applyEvent(state, action.event, action.data);
  }
}

function applyEvent(state: AskState, event: string, data: unknown): AskState {
  switch (event) {
    case "tool":
      return { ...state, tools: [...state.tools, data as ToolPayload] };
    case "step":
      return { ...state, steps: [...state.steps, data as StepPayload] };
    case "source": {
      const source = data as SourceRef;
      // The same alias is sent once, but a retry could repeat it; the first one wins.
      if (state.sources.some((existing) => existing.id === source.id)) return state;
      return { ...state, sources: [...state.sources, source] };
    }
    case "token":
      return { ...state, text: state.text + (data as { text: string }).text };
    case "notice":
      return { ...state, notice: data as NoticePayload };
    case "answer": {
      const answer = data as AnswerPayload;
      // The checked answer is authoritative: the streamed words can differ from it.
      return { ...state, status: "done", answer, text: answer.text };
    }
    case "error": {
      const error = data as { code: AskErrorCode; message: string; retry_at?: string | null };
      return {
        ...state,
        status: "error",
        error: {
          code: error.code,
          message: error.message,
          retryAt: error.retry_at ?? null,
        },
      };
    }
    default:
      return state;
  }
}

function errorFor(status: number, payload: { kind?: string; retryAfterS?: number }): AskError {
  const code: AskErrorCode =
    payload.kind === "rate_limited" || payload.kind === "busy" || payload.kind === "unauthorized"
      ? payload.kind
      : status === 401
        ? "unauthorized"
        : status === 503
          ? "unavailable"
          : "internal";
  return { code, message: "", retryAt: null, retryAfterS: payload.retryAfterS };
}

/**
 * The stream controller behind the sheet. Plain TypeScript on purpose: the watchdog, the abort and
 * the reducer are the parts worth testing, and none of them needs a React renderer.
 */
export class AskController {
  private state: AskState = INITIAL_ASK_STATE;
  private readonly listeners = new Set<() => void>();
  private inFlight: AbortController | null = null;
  private timer: ReturnType<typeof setTimeout> | null = null;

  constructor(
    private readonly options: {
      fetchImpl?: typeof fetch;
      watchdogMs?: number;
    } = {},
  ) {}

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };

  getState = (): AskState => this.state;

  private set(next: AskState): void {
    this.state = next;
    for (const listener of this.listeners) listener();
  }

  private dispatch(action: AskAction): void {
    this.set(askReducer(this.state, action));
  }

  private startWatchdog(controller: AbortController): void {
    this.stopWatchdog();
    this.timer = setTimeout(() => {
      if (this.state.status !== "streaming") return;
      this.dispatch({ type: "dropped" });
      controller.abort();
    }, this.options.watchdogMs ?? WATCHDOG_MS);
  }

  private stopWatchdog(): void {
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
  }

  ask = async (question: string): Promise<AskStatus> => {
    this.stopWatchdog();
    this.inFlight?.abort();
    const controller = new AbortController();
    this.inFlight = controller;
    this.dispatch({ type: "start", question });
    // The clock starts before the request: a tunnel that stalls before the response headers is a
    // dead connection too, and every received byte resets it below.
    this.startWatchdog(controller);

    const fetchImpl = this.options.fetchImpl ?? fetch;
    try {
      const response = await fetchImpl("/api/ask", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ question }),
        signal: controller.signal,
        cache: "no-store",
      });
      if (!response.ok || !response.body) {
        const payload = (await response.json().catch(() => ({}))) as {
          kind?: string;
          retryAfterS?: number;
        };
        this.set({ ...this.state, status: "error", error: errorFor(response.status, payload) });
        return this.state.status;
      }
      for await (const event of parseSse(response.body, () => this.startWatchdog(controller))) {
        this.dispatch({ type: "event", event: event.event, data: event.data });
      }
      // A stream that ends without an `answer` or an `error` lost its connection.
      this.dispatch({ type: "dropped" });
    } catch {
      // An abort is the user's own doing; anything else is a connection that died.
      if (!controller.signal.aborted) this.dispatch({ type: "dropped" });
    } finally {
      this.stopWatchdog();
      if (this.inFlight === controller) this.inFlight = null;
    }
    return this.state.status;
  };

  abort = (): void => {
    this.stopWatchdog();
    this.inFlight?.abort();
    this.inFlight = null;
    // A closed sheet starts clean: a partial, unchecked answer must not come back as if it were one.
    this.set(INITIAL_ASK_STATE);
  };
}

export function useAsk(): {
  state: AskState;
  ask: (question: string) => Promise<AskStatus>;
  abort: () => void;
} {
  const controller = useMemo(() => new AskController(), []);
  const state = useSyncExternalStore(
    controller.subscribe,
    controller.getState,
    controller.getState,
  );
  // Leaving the sheet mid-answer cancels it, like closing the tab does.
  useEffect(() => () => controller.abort(), [controller]);
  return { state, ask: controller.ask, abort: controller.abort };
}
