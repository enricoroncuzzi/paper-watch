"use client";

import * as Dialog from "@radix-ui/react-dialog";
import { useCallback, useEffect, useRef, useState } from "react";

import { AnswerText } from "@/components/ask/AnswerText";
import { RecentThreads } from "@/components/ask/RecentThreads";
import { Sources } from "@/components/ask/Sources";
import { Trace } from "@/components/ask/Trace";
import { formatRome } from "@/lib/format";
import {
  footnoteNumbers,
  useAsk,
  type AskError,
  type AskOptions,
  type SourceRef,
} from "@/lib/use-ask";

type FollowUp = {
  label: string;
  question: string | null;
  tiers: string[] | null;
};

/** The three fixed follow-ups. Fixed on purpose: inventing them would cost a model call each. */
export const FOLLOW_UPS: FollowUp[] = [
  {
    label: "Only peer-reviewed sources",
    question: null,
    tiers: ["peer_reviewed"],
  },
  {
    label: "What are the limitations?",
    question: "What are the limitations?",
    tiers: null,
  },
  {
    label: "How was it measured?",
    question: "How was it measured?",
    tiers: null,
  },
];

function errorCopy(error: AskError): string {
  switch (error.code) {
    case "daily_cap":
      return `The daily model limit is spent. It resets at ${formatRome(error.retryAt)} Europe/Rome.`;
    case "budget":
      return "This answer ran out of tokens. Ask a shorter question.";
    case "model_unavailable":
      return "The model is unavailable right now. Try again shortly.";
    case "deadline":
      return "The answer took too long. Ask again.";
    case "rate_limited":
      return error.retryAfterS === undefined
        ? "Too many questions right now. Try again shortly."
        : `Too many questions right now. Try again in ${error.retryAfterS}s.`;
    case "busy":
      return "The server is busy. Try again shortly.";
    case "unauthorized":
      return "Your session is not accepted any more. Sign in again.";
    case "unavailable":
      return "The API is unreachable right now. The ingestion is unaffected.";
    case "dropped":
      return "The connection dropped.";
    case "internal":
      return "Something went wrong answering. Ask again.";
  }
}

function MatchingItems({ sources }: { sources: SourceRef[] }) {
  if (sources.length === 0) return null;
  return (
    <div className="mt-3 border-t border-bad/30 pt-2.5">
      <p className="mb-1.5 text-[9.5px] uppercase tracking-[0.13em] text-ink-3">
        Matching items
      </p>
      <ul className="flex flex-col gap-1 text-[12px] text-ink-2">
        {sources.map((source) => (
          <li key={source.id} className="font-serif text-[13.5px]">
            {source.title}
          </li>
        ))}
      </ul>
    </div>
  );
}

/** A finished turn, above the current one: its question, its checked answer and its sources. */
function PastTurn({
  id,
  n,
  question,
  askedAt,
  text,
  sources,
}: {
  id: string;
  // Null for an answer the server could not save: it still belongs above the next question.
  n: number | null;
  question: string;
  askedAt: string;
  text: string;
  sources: SourceRef[];
}) {
  const numbers = footnoteNumbers(sources);
  const prefix = `turn-${id}-`;
  return (
    <section
      data-testid={`turn-${n ?? "unsaved"}`}
      className="mb-8 border-b border-rule pb-7"
    >
      <h2 className="mb-2 max-w-[32ch] font-serif text-[19px] leading-[1.35] tracking-[-0.01em]">
        {question}
      </h2>
      <p className="mb-3.5 flex flex-wrap items-center gap-1.5 text-[11px] text-ink-3">
        <span>you</span>
        <span>·</span>
        <span>{formatRome(askedAt)} Europe/Rome</span>
      </p>
      <AnswerText
        text={text}
        streaming={false}
        checked
        numbers={numbers}
        anchorPrefix={prefix}
        testId="past-answer"
      />
      <Sources
        sources={sources}
        numbers={numbers}
        anchorPrefix={prefix}
        testId="past-sources"
      />
    </section>
  );
}

/**
 * The reading sheet: the conversation so far, the current question with what the agent did, the
 * checked answer with its citations, and the composer. On a phone it is the whole screen, with the
 * composer above the safe area.
 */
export function AskSheet({
  open,
  item,
  onOpenChange,
  onLeaveItem,
}: {
  open: boolean;
  // Set when a card's "Ask about it" opened the sheet; the API registers the item as S1.
  item: { id: string; title: string } | null;
  onOpenChange: (open: boolean) => void;
  // A conversation opened from Recent is not about the item, so the sheet stops naming it.
  onLeaveItem: () => void;
}) {
  const { state, ask, abort, recover, openThread } = useAsk();
  const [draft, setDraft] = useState("");
  const [recentOpen, setRecentOpen] = useState(false);
  const textarea = useRef<HTMLTextAreaElement>(null);
  const recentButton = useRef<HTMLButtonElement>(null);
  const body = useRef<HTMLDivElement>(null);
  const askedQuestion = useRef<HTMLHeadingElement>(null);

  const streaming = state.status === "streaming";
  // A follow-up waits for the stream to end: the `thread` event that stores its turn arrives after
  // the answer.
  const busy = streaming || state.streamOpen;
  const hasConversation = state.turns.length > 0 || state.question !== "";
  const numbers = footnoteNumbers(state.answer?.citations ?? []);
  // The last question the chips can act on: the current one, else the newest finished turn.
  const lastQuestion = state.question || state.turns.at(-1)?.question || "";
  // A dropped stream has no server payload: the client's own watchdog is the error, unless the
  // thread re-fetch already told the reader the answer was interrupted.
  const shownError: AskError | null = state.interrupted
    ? null
    : (state.error ??
      (state.status === "dropped"
        ? { code: "dropped", message: "", retryAt: null }
        : null));

  const closeRecent = useCallback(() => {
    setRecentOpen(false);
    recentButton.current?.focus();
  }, []);

  async function sendQuestion(question: string, options: AskOptions = {}) {
    const trimmed = question.trim();
    if (!trimmed || busy) return;
    const threadId = options.threadId ?? state.threadId;
    // The composer empties at send time, so it never repeats the question shown above the answer,
    // and whatever the reader types while the answer streams is theirs to keep.
    setDraft("");
    setRecentOpen(false);
    const status = await ask(trimmed, {
      // An item only registers on a thread with no finished turn; the API ignores it on a
      // follow-up. A dropped first question keeps it, so its retry is still about the item.
      itemId: state.turnCount > 0 ? null : (options.itemId ?? item?.id ?? null),
      threadId,
      tiers: options.tiers,
    });
    // A refusal puts the question back only when nothing else has been typed since. A dropped
    // answer does not: its own block offers "Ask it again", and refilling as well would both
    // duplicate the offer and discard the block's own question and filter.
    if (status === "error")
      setDraft((current) => (current === "" ? trimmed : current));
  }

  function onKeyDown(event: React.KeyboardEvent<HTMLTextAreaElement>) {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      void sendQuestion(draft);
    }
  }

  // The newest content stays in view: a new question at the top of the body, a finished answer at
  // its sources. Not during streaming, so a reader who scrolled up is not dragged back down.
  useEffect(() => {
    if (!state.question) return;
    if (state.status === "streaming") {
      askedQuestion.current?.scrollIntoView({ block: "start" });
    } else if (state.status !== "idle") {
      body.current?.scrollTo({ top: body.current.scrollHeight });
    }
  }, [state.question, state.status, state.interrupted, state.turns.length]);

  // A dropped stream asks the server what it finished before the sheet says anything is lost.
  useEffect(() => {
    if (state.status !== "dropped") return;
    void recover();
  }, [state.status, recover]);

  // A hidden tab says nothing about its stream: a desktop tab switch of a second leaves a healthy
  // answer running. `recover` reads the thread when there is one and aborts only after adopting a
  // turn the server finished, so a live stream is never killed by coming back to the page.
  useEffect(() => {
    function onVisibilityChange() {
      if (document.visibilityState === "visible") void recover();
    }
    document.addEventListener("visibilitychange", onVisibilityChange);
    return () =>
      document.removeEventListener("visibilitychange", onVisibilityChange);
  }, [state.status, recover]);

  return (
    <Dialog.Root
      open={open}
      onOpenChange={(next) => {
        // Closing the sheet mid-answer cancels it, like closing the tab does.
        if (!next) abort();
        onOpenChange(next);
      }}
    >
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-[90] bg-black/25 backdrop-blur-[5px]" />
        <Dialog.Content
          onOpenAutoFocus={(event) => {
            event.preventDefault();
            textarea.current?.focus();
          }}
          onEscapeKeyDown={(event) => {
            // Escape closes the menu first: a reader backing out of a list is not closing the sheet.
            if (recentOpen) {
              event.preventDefault();
              closeRecent();
            }
          }}
          className="fixed inset-0 z-[91] flex h-[100dvh] w-full flex-col overflow-hidden border border-rule bg-card shadow-[var(--shadow)] sm:inset-auto sm:left-1/2 sm:top-1/2 sm:h-[min(88dvh,880px)] sm:w-[min(880px,94vw)] sm:-translate-x-1/2 sm:-translate-y-1/2"
        >
          <header className="relative flex flex-none items-center gap-3 border-b border-rule px-5 py-3 sm:px-6">
            <Dialog.Title className="font-serif text-[17px] tracking-[-0.01em]">
              Ask the corpus
            </Dialog.Title>
            <Dialog.Description className="sr-only">
              Ask a question about the stored corpus, or a follow-up about the
              conversation so far. Answers cite their sources. Escape closes the
              sheet.
            </Dialog.Description>
            {state.turnCount > 0 ? (
              <span
                data-testid="turns-chip"
                className="inline-flex h-[21px] items-center rounded-[2px] border border-rule px-2 text-[11px] whitespace-nowrap text-ink-2"
              >
                {state.turnCount} turn{state.turnCount === 1 ? "" : "s"}
              </span>
            ) : null}
            <div className="ml-auto flex items-center gap-1.5">
              <button
                ref={recentButton}
                type="button"
                aria-expanded={recentOpen}
                aria-controls="recent-threads"
                onClick={() => setRecentOpen((current) => !current)}
                // The hit area stays the phone's 44 px; the visible control inside it is smaller.
                className="grid h-11 min-w-11 place-items-center px-2 text-ink-2 hover:text-ink sm:h-[26px] sm:min-w-0 sm:px-0"
              >
                <span
                  className={`inline-flex h-[26px] items-center rounded-[2px] border px-2.5 text-[11.5px] ${
                    recentOpen
                      ? "border-accent text-ink"
                      : "border-rule-2 hover:border-accent"
                  }`}
                >
                  Recent
                </span>
              </button>
              {/* Right after its trigger, so Tab reaches the list before the Close button; it is
                  positioned against the header rule, not against this row. */}
              {recentOpen ? (
                <RecentThreads
                  id="recent-threads"
                  opener={recentButton}
                  onPick={(threadId) => {
                    setRecentOpen(false);
                    void openThread(threadId).then((opened) => {
                      if (opened) onLeaveItem();
                    });
                  }}
                  onDismiss={closeRecent}
                />
              ) : null}
              <Dialog.Close
                aria-label="Close the sheet"
                className="grid h-11 w-11 place-items-center rounded-full text-ink-3 hover:bg-card-2 hover:text-ink"
              >
                <svg
                  width="14"
                  height="14"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  aria-hidden="true"
                >
                  <path d="m6 6 12 12M18 6 6 18" />
                </svg>
              </Dialog.Close>
            </div>
          </header>

          <div
            ref={body}
            // The body is the reading sheet's live region: a new answer, a check note and an
            // interrupted line reach a screen reader without moving focus.
            aria-live="polite"
            className="min-h-0 flex-1 overflow-auto px-5 py-6 sm:px-9"
          >
            {item ? (
              <p
                data-testid="asking-about"
                className="mb-3 max-w-[38rem] border-l-2 border-accent pl-2.5 text-[11.5px] leading-[1.5] text-ink-2"
              >
                About this item:{" "}
                <span className="font-serif text-[13px] text-ink">
                  {item.title}
                </span>
              </p>
            ) : null}

            {state.turns.map((turn) => (
              <PastTurn
                key={turn.id}
                id={turn.id}
                n={turn.n}
                question={turn.question}
                askedAt={turn.askedAt}
                text={turn.answer.text}
                sources={turn.answer.citations}
              />
            ))}

            {state.question ? (
              <>
                <h2
                  ref={askedQuestion}
                  data-testid="asked-question"
                  className="mb-3 max-w-[32ch] font-serif text-[24px] leading-[1.32] tracking-[-0.015em]"
                >
                  {state.question}
                </h2>
                <p className="mb-4 flex flex-wrap items-center gap-1.5 text-[11px] text-ink-3">
                  <span>you</span>
                  <span>·</span>
                  <span>{formatRome(state.askedAt)} Europe/Rome</span>
                  {state.filter ? (
                    <>
                      <span>·</span>
                      <span>{state.filter} · all topics</span>
                    </>
                  ) : null}
                </p>
              </>
            ) : state.turns.length === 0 ? (
              <p className="max-w-[34ch] font-serif text-[21px] leading-[1.45] text-ink-2">
                Ask about the stored corpus. Every answer cites the items it
                used.
              </p>
            ) : null}

            {state.notice ? (
              <p
                data-testid="notice"
                className="mb-4 max-w-[38rem] border border-rule bg-card-2 px-3 py-2 text-[11.5px] leading-[1.5] text-ink-2"
              >
                {state.notice.message}
              </p>
            ) : null}

            <Trace
              tools={state.tools}
              steps={state.steps}
              // The checked answer's own citations are what the footnote list prints, so the count
              // must agree with them: a follow-up that only re-reads an earlier source announces
              // no `source` event and would otherwise read "0 sources" above the answer.
              sourceCount={
                state.answer?.citations.length ?? state.sources.length
              }
            />

            {state.interrupted ? (
              <div
                data-testid="interrupted"
                className="mb-4 max-w-[38rem] border border-warn/40 bg-warn-soft px-4 py-3"
              >
                <p className="text-[13px] leading-[1.55] text-ink">
                  That answer was interrupted. Ask it again.
                </p>
                <button
                  type="button"
                  onClick={() =>
                    void sendQuestion(state.question, { tiers: state.tiers })
                  }
                  className="mt-2 inline-flex h-8 items-center rounded-[2px] border border-rule-2 bg-card px-3 text-[12px] text-ink hover:border-accent"
                >
                  Ask it again
                </button>
              </div>
            ) : null}

            {shownError ? (
              <div
                data-testid="ask-error"
                className="max-w-[38rem] border border-bad/40 bg-bad-soft px-4 py-3"
              >
                <p className="text-[13px] leading-[1.55] text-ink">
                  {errorCopy(shownError)}
                </p>
                {shownError.code === "daily_cap" ? (
                  <MatchingItems sources={state.sources} />
                ) : null}
                {shownError.code === "dropped" ? (
                  // A dropped answer has no server payload, and a follow-up's re-fetch can fail
                  // too: the block carries the retry itself, or the reader retypes the question.
                  <button
                    type="button"
                    onClick={() =>
                      void sendQuestion(state.question, { tiers: state.tiers })
                    }
                    className="mt-2 inline-flex h-8 items-center rounded-[2px] border border-rule-2 bg-card px-3 text-[12px] text-ink hover:border-accent"
                  >
                    Ask it again
                  </button>
                ) : null}
              </div>
            ) : state.interrupted ? null : (
              // An interrupted answer has no checked text: the partial stream and its unchecked
              // sources stay hidden behind the interrupted line, which offers to ask again.
              <>
                {state.answer &&
                (!state.answer.supported || state.answer.flags.length > 0) ? (
                  <div
                    data-testid="check-note"
                    className="mb-4 max-w-[38rem] border border-warn/40 bg-warn-soft px-3.5 py-2.5"
                  >
                    <p className="text-[12.5px] font-medium text-warn">
                      Some sentences are not backed by a retrieved source.
                    </p>
                    {state.answer.flags.map((flag, index) => (
                      <p
                        key={`${flag.kind}-${index}`}
                        className="mt-1 text-[11.5px] leading-[1.5] text-ink-2"
                      >
                        <span className="font-mono text-[10px] text-ink-3">
                          {flag.kind.replaceAll("_", " ")}
                        </span>{" "}
                        · {flag.sentence} {flag.detail}
                      </p>
                    ))}
                  </div>
                ) : null}
                <AnswerText
                  text={state.text}
                  streaming={streaming}
                  checked={state.answer !== null}
                  numbers={numbers}
                />
                <Sources
                  sources={state.answer?.citations ?? state.sources}
                  numbers={numbers}
                />
              </>
            )}
          </div>

          <footer className="flex-none border-t border-rule bg-card px-5 pb-[max(13px,env(safe-area-inset-bottom))] pt-3 sm:px-6">
            <div className="mb-2 flex flex-wrap gap-1.5">
              {FOLLOW_UPS.map((chip) => {
                const question = chip.question ?? lastQuestion;
                return (
                  <button
                    key={chip.label}
                    type="button"
                    disabled={busy || !hasConversation}
                    onClick={() =>
                      void sendQuestion(question, { tiers: chip.tiers })
                    }
                    className="rounded-full border border-rule bg-card-2 px-[10px] py-[3px] text-[11.5px] text-ink-2 hover:border-accent hover:text-ink disabled:opacity-40"
                  >
                    {chip.label}
                  </button>
                );
              })}
            </div>
            <div className="flex items-end gap-2 border border-rule-2 bg-paper px-3 py-2 focus-within:border-accent focus-within:shadow-[0_0_0_3px_var(--accent-soft)]">
              <textarea
                ref={textarea}
                data-testid="composer"
                rows={1}
                value={draft}
                maxLength={1000}
                onChange={(event) => setDraft(event.target.value)}
                onKeyDown={onKeyDown}
                placeholder={
                  hasConversation
                    ? "Ask a follow-up..."
                    : item
                      ? "Ask about this item"
                      : "Ask a new question"
                }
                aria-label="Your question"
                className="max-h-32 min-h-7 flex-1 resize-none bg-transparent font-serif text-[16px] leading-[1.4] text-ink outline-none focus-visible:outline-none placeholder:text-ink-3"
              />
              <button
                type="button"
                onClick={() => void sendQuestion(draft)}
                disabled={busy || draft.trim() === ""}
                aria-label="Send the question"
                className="grid h-11 w-11 flex-none place-items-center rounded-[2px] bg-accent text-accent-ink disabled:opacity-40 sm:h-[30px] sm:w-[30px]"
              >
                <svg
                  width="14"
                  height="14"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  aria-hidden="true"
                >
                  <path d="M12 19V5M6 11l6-6 6 6" />
                </svg>
              </button>
            </div>
            <p className="mt-1.5 flex gap-3 text-[10.5px] text-ink-3">
              <span className="hidden sm:inline">⌘↵ to send</span>
              <span className="hidden sm:inline">Esc to close</span>
              {draft.length > 800 ? (
                <span className="ml-auto">{draft.length}/1000</span>
              ) : null}
            </p>
          </footer>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
