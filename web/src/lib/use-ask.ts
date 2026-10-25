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

/** One finished turn of the conversation, as the sheet re-renders it above the current one. */
export type ConversationTurn = {
  // A stable key for React and the footnote anchors: the server's number, or the moment an
  // answer that could not be saved was asked. `n` alone would collide in that case.
  id: string;
  n: number | null;
  question: string;
  answer: AnswerPayload;
  askedAt: string;
};

/** One turn as the API stores and returns it (`ThreadTurnData`). */
export type WireTurn = {
  n: number;
  question: string;
  answer: AnswerPayload;
  search_mode: string;
  created_at: string;
};

/** A thread as the Recent menu lists it (`ThreadSummaryData`). */
export type ThreadSummary = {
  id: string;
  title: string;
  turns: number;
  updated_at: string;
};

export type AskState = {
  status: AskStatus;
  question: string;
  askedAt: string | null;
  // The finished turns before the current one, oldest first.
  turns: ConversationTurn[];
  // The thread the conversation belongs to: set once the server saved a turn, or by opening one.
  threadId: string | null;
  // This turn's number, once the server saved it.
  turn: number | null;
  // The finished turns the thread holds, so the header chip can count them.
  turnCount: number;
  // The turn number this request will get. A repeated question (the peer-reviewed chip) must not
  // adopt an older answer with the same text, so adoption checks this as well as the text.
  expected: number | null;
  // True when the server has no finished turn for the question the sheet asked.
  interrupted: boolean;
  // The tiers line under the question, as the request asked for them; null when a stored turn is
  // shown and the filter it was asked with is not known.
  filter: string | null;
  // The filter itself, so "Ask it again" re-sends what was asked rather than everything.
  tiers: string[] | null;
  tools: ToolPayload[];
  steps: StepPayload[];
  sources: SourceRef[];
  text: string;
  notice: NoticePayload | null;
  answer: AnswerPayload | null;
  error: AskError | null;
};

export type AskOptions = {
  itemId?: string | null;
  threadId?: string | null;
  tiers?: string[] | null;
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
  askedAt: null,
  turns: [],
  threadId: null,
  turn: null,
  turnCount: 0,
  expected: null,
  interrupted: false,
  filter: null,
  tiers: null,
  tools: [],
  steps: [],
  sources: [],
  text: "",
  notice: null,
  answer: null,
  error: null,
};

export type AskAction =
  | {
      type: "start";
      question: string;
      at: string;
      threadId: string | null;
      filter: string | null;
      tiers: string[] | null;
    }
  | { type: "event"; event: string; data: unknown }
  | { type: "dropped" }
  | { type: "abort" }
  // The thread's finished turns, read back after a dropped stream or a return to the page.
  | { type: "recover"; turns: WireTurn[] }
  // The server finished this request's turn after the stream died: show its checked answer.
  | { type: "adopted"; turns: WireTurn[] }
  // A conversation the reader picked in the Recent menu.
  | { type: "open"; threadId: string; turns: WireTurn[] };

/** One stored turn as the sheet renders it. */
function conversationTurn(turn: WireTurn): ConversationTurn {
  return {
    id: `saved-${turn.n}`,
    n: turn.n,
    question: turn.question,
    answer: turn.answer,
    askedAt: turn.created_at,
  };
}

/** The tiers line the sheet prints: the two the sheet can send are the only two it can show. */
export function filterLabel(tiers: string[] | null): string | null {
  if (tiers === null) return "all tiers";
  return tiers.includes("peer_reviewed") ? "peer-reviewed only" : "all tiers";
}

/** The reducer is exported so the stream logic is tested without rendering a component. */
export function askReducer(state: AskState, action: AskAction): AskState {
  switch (action.type) {
    case "start": {
      // A finished current turn joins the earlier ones, so a follow-up reads as a conversation
      // instead of replacing what came before it. An answer the server could not save still moves
      // up: it happened, and the reader saw it.
      const finished =
        state.answer !== null && state.askedAt !== null
          ? [
              {
                id:
                  state.turn !== null
                    ? `saved-${state.turn}`
                    : `local-${state.askedAt}`,
                n: state.turn,
                question: state.question,
                answer: state.answer,
                askedAt: state.askedAt,
              },
            ]
          : [];
      return {
        ...INITIAL_ASK_STATE,
        status: "streaming",
        question: action.question,
        askedAt: action.at,
        turns: [...state.turns, ...finished],
        threadId: action.threadId,
        turnCount: state.turnCount,
        // The turn this request will be: a thread that holds N finished turns numbers this N+1.
        expected: state.turnCount + 1,
        filter: action.filter,
        tiers: action.tiers,
      };
    }
    case "dropped":
      return state.status === "streaming" ? { ...state, status: "dropped" } : state;
    case "abort":
      return state.status === "streaming" ? { ...state, status: "idle" } : state;
    case "event":
      return applyEvent(state, action.event, action.data);
    case "recover":
      return applyRecovery(state, action.turns);
    case "adopted":
      return adopt(state, action.turns);
    case "open":
      return openConversation(action.threadId, action.turns);
  }
}

/** Whether a stored turn is the one this request asked for: same words and same number. */
function isThisRequest(state: AskState, last: WireTurn | undefined): last is WireTurn {
  return (
    last !== undefined &&
    state.question !== "" &&
    last.question === state.question &&
    last.n === state.expected
  );
}

/**
 * The server finished this request's turn after the stream died: its checked answer replaces the
 * streamed text, and the turns before it are the thread's other finished ones.
 */
function adopt(state: AskState, wire: WireTurn[]): AskState {
  const last = wire.at(-1);
  if (last === undefined) return state;
  return {
    ...state,
    status: "done",
    interrupted: false,
    turns: wire.slice(0, -1).map(conversationTurn),
    turnCount: wire.length,
    turn: last.n,
    expected: last.n + 1,
    answer: last.answer,
    text: last.answer.text,
    sources: last.answer.citations,
    // `filter` and `tiers` stay: this is still the request the sheet sent, so it knows them.
  };
}

/**
 * A thread read back after a dropped stream: the server's finished turns are the truth about what
 * it checked. The matching turn is adopted by `adopt`; without one, the reader is told the answer
 * was interrupted. A stream that is still running is left alone (see `recover`).
 */
function applyRecovery(state: AskState, wire: WireTurn[]): AskState {
  if (isThisRequest(state, wire.at(-1))) return adopt(state, wire);
  return {
    ...state,
    // An answer already on screen stays: only the turns above it are refreshed.
    status: state.answer === null ? "dropped" : state.status,
    interrupted: state.answer === null && state.question !== "",
    turns: wire.map(conversationTurn),
    turnCount: wire.length,
  };
}

function openConversation(threadId: string, wire: WireTurn[]): AskState {
  const last = wire.at(-1);
  if (last === undefined) return { ...INITIAL_ASK_STATE, threadId };
  return {
    ...INITIAL_ASK_STATE,
    status: "done",
    threadId,
    turn: last.n,
    turnCount: wire.length,
    expected: last.n + 1,
    turns: wire.slice(0, -1).map(conversationTurn),
    question: last.question,
    askedAt: last.created_at,
    answer: last.answer,
    text: last.answer.text,
    sources: last.answer.citations,
  };
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
    case "thread": {
      const thread = data as { id: string; turn: number };
      // The finished turn is stored: its number is this conversation's, and the chip counts it.
      return { ...state, threadId: thread.id, turn: thread.turn, turnCount: thread.turn };
    }
    case "answer": {
      const answer = data as AnswerPayload;
      // The checked answer is authoritative: the streamed words can differ from it. It also ends
      // any interruption a late thread read left behind: this request did finish.
      return { ...state, status: "done", answer, text: answer.text, interrupted: false };
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
  // A tunnel that is down or stalls answers 502 or 504, and the proxy calls every upstream 5xx
  // "unavailable"; that copy is the right one for any 5xx, not only for a literal 503.
  const code: AskErrorCode =
    payload.kind === "rate_limited" ||
    payload.kind === "busy" ||
    payload.kind === "unauthorized" ||
    payload.kind === "unavailable"
      ? payload.kind
      : status === 401
        ? "unauthorized"
        : status >= 500
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
  // Bumped whenever a newer ask, a newer open or an abort takes over. Every continuation after an
  // await compares it before it touches state, so a late read or a superseded stream cannot land
  // on the request the reader is looking at now.
  private generation = 0;

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

  ask = async (question: string, options: AskOptions = {}): Promise<AskStatus> => {
    const generation = (this.generation += 1);
    this.stopWatchdog();
    this.inFlight?.abort();
    const controller = new AbortController();
    this.inFlight = controller;
    // A follow-up continues the conversation the sheet already holds unless the caller names
    // another thread; the chips do, and an ordinary send does not.
    const threadId = options.threadId ?? this.state.threadId;
    const tiers = options.tiers && options.tiers.length > 0 ? options.tiers : null;
    this.dispatch({
      type: "start",
      question,
      at: new Date().toISOString(),
      threadId,
      filter: filterLabel(tiers),
      tiers,
    });
    // The clock starts before the request: a tunnel that stalls before the response headers is a
    // dead connection too, and every received byte resets it below.
    this.startWatchdog(controller);

    const fetchImpl = this.options.fetchImpl ?? fetch;
    try {
      const response = await fetchImpl("/api/ask", {
        method: "POST",
        headers: { "content-type": "application/json" },
        // The API pre-registers `item_id` as S1 before the graph starts, and continues `thread_id`
        // with its stored turns and aliases. `tiers` narrows the search for a chip's re-ask.
        body: JSON.stringify({
          question,
          ...(options.itemId ? { item_id: options.itemId } : {}),
          ...(threadId ? { thread_id: threadId } : {}),
          ...(options.tiers && options.tiers.length > 0 ? { tiers: options.tiers } : {}),
        }),
        signal: controller.signal,
        cache: "no-store",
      });
      // A newer ask replaced this one while its headers were in flight: the response belongs to a
      // request nobody is waiting for, and writing its status would land on the newer answer.
      if (generation !== this.generation) return this.state.status;
      if (!response.ok || !response.body) {
        const payload = (await response.json().catch(() => ({}))) as {
          kind?: string;
          retryAfterS?: number;
        };
        // The body read can outlive the request, and its abort is swallowed by the `catch` above:
        // a superseded refusal must not turn the newer request into an error.
        if (generation !== this.generation) return this.state.status;
        this.set({ ...this.state, status: "error", error: errorFor(response.status, payload) });
        return this.state.status;
      }
      // Only the active request may re-arm the watchdog: a late byte on an aborted or superseded
      // stream would otherwise start a timer that nothing clears.
      for await (const event of parseSse(response.body, () => {
        if (this.inFlight === controller) this.startWatchdog(controller);
      })) {
        // A superseded stream can still deliver a buffered frame; it belongs to the request it
        // replaced, so the event is dropped with it.
        if (generation !== this.generation) break;
        this.dispatch({ type: "event", event: event.event, data: event.data });
      }
      // A stream that ends without an `answer` or an `error` lost its connection. A superseded
      // stream can still end normally (a queued frame, a proxy flush), and its `dropped` belongs
      // to the request it replaced, not to the newer one still streaming.
      if (generation === this.generation) this.dispatch({ type: "dropped" });
    } catch {
      // An abort is the user's own doing; anything else is a connection that died.
      if (!controller.signal.aborted) this.dispatch({ type: "dropped" });
    } finally {
      // Only the active request owns the watchdog: an older request's abort must not disarm it.
      if (this.inFlight === controller) {
        this.stopWatchdog();
        this.inFlight = null;
      }
    }
    return this.state.status;
  };

  /**
   * What the server finished for this request, read after a dropped stream or a return to the page.
   *
   * A hidden tab says nothing about whether its stream is alive: on a desktop a tab switch of one
   * second, and on iOS Safari a frozen timer, can both leave a healthy stream running. So a return
   * to the page does not abort anything. It reads the thread when there is one, adopts the turn
   * only when the server finished *this* request (its words and its number), and otherwise leaves
   * the stream alone; a dead connection still ends through the reader error or the watchdog.
   */
  recover = async (): Promise<void> => {
    const threadId = this.state.threadId;
    if (!threadId) return;
    const generation = this.generation;
    const fetchImpl = this.options.fetchImpl ?? fetch;
    try {
      const response = await fetchImpl(`/api/threads/${threadId}`, { cache: "no-store" });
      if (!response.ok) return;
      const detail = (await response.json()) as { turns: WireTurn[] };
      // A newer ask or an opened conversation took over while this read was in flight.
      if (generation !== this.generation) return;
      if (isThisRequest(this.state, detail.turns.at(-1))) {
        // The server finished the answer the client stopped hearing: adopting it is what makes
        // the abort below safe, because the answer is no longer only in the stream.
        this.generation += 1;
        this.stopWatchdog();
        this.inFlight?.abort();
        this.inFlight = null;
        this.dispatch({ type: "adopted", turns: detail.turns });
        return;
      }
      // A healthy stream is left to finish; only a stream that already ended is told it was
      // interrupted.
      if (this.state.status === "streaming") return;
      this.dispatch({ type: "recover", turns: detail.turns });
    } catch {
      // A failed re-fetch leaves the state as it was: the reader can ask the question again.
    }
  };

  /** Open a stored conversation from the Recent menu. */
  openThread = async (threadId: string): Promise<void> => {
    const generation = (this.generation += 1);
    this.stopWatchdog();
    this.inFlight?.abort();
    this.inFlight = null;
    const fetchImpl = this.options.fetchImpl ?? fetch;
    try {
      const response = await fetchImpl(`/api/threads/${threadId}`, { cache: "no-store" });
      if (!response.ok) return;
      const detail = (await response.json()) as { turns: WireTurn[] };
      if (generation !== this.generation) return;
      this.dispatch({ type: "open", threadId, turns: detail.turns });
    } catch {
      // The menu keeps the conversation it had.
    }
  };

  abort = (): void => {
    this.generation += 1;
    this.stopWatchdog();
    this.inFlight?.abort();
    this.inFlight = null;
    // A closed sheet starts clean: a partial, unchecked answer must not come back as if it were one.
    this.set(INITIAL_ASK_STATE);
  };
}

export function useAsk(): {
  state: AskState;
  ask: (question: string, options?: AskOptions) => Promise<AskStatus>;
  abort: () => void;
  recover: () => Promise<void>;
  openThread: (threadId: string) => Promise<void>;
} {
  const controller = useMemo(() => new AskController(), []);
  const state = useSyncExternalStore(
    controller.subscribe,
    controller.getState,
    controller.getState,
  );
  // Leaving the sheet mid-answer cancels it, like closing the tab does.
  useEffect(() => () => controller.abort(), [controller]);
  return {
    state,
    ask: controller.ask,
    abort: controller.abort,
    recover: controller.recover,
    openThread: controller.openThread,
  };
}
