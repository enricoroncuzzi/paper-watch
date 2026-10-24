"use client";

import * as Dialog from "@radix-ui/react-dialog";
import { useRef, useState } from "react";

import { AnswerText } from "@/components/ask/AnswerText";
import { Sources } from "@/components/ask/Sources";
import { Trace } from "@/components/ask/Trace";
import { formatRome } from "@/lib/format";
import { footnoteNumbers, useAsk, type AskError, type SourceRef } from "@/lib/use-ask";

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
      return "The connection dropped. Ask again.";
    case "internal":
      return "Something went wrong answering. Ask again.";
  }
}

function MatchingItems({ sources }: { sources: SourceRef[] }) {
  if (sources.length === 0) return null;
  return (
    <div className="mt-3 border-t border-bad/30 pt-2.5">
      <p className="mb-1.5 text-[9.5px] uppercase tracking-[0.13em] text-ink-3">Matching items</p>
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

/**
 * The reading sheet: the question, what the agent did, the checked answer with its citations, and
 * the composer. On a phone it is the whole screen, with the composer above the safe area.
 */
export function AskSheet({
  open,
  item,
  onOpenChange,
}: {
  open: boolean;
  // Set when a card's "Ask about it" opened the sheet; the API registers the item as S1.
  item: { id: string; title: string } | null;
  onOpenChange: (open: boolean) => void;
}) {
  const { state, ask, abort } = useAsk();
  const [draft, setDraft] = useState("");
  const [askedAt, setAskedAt] = useState<string | null>(null);
  const textarea = useRef<HTMLTextAreaElement>(null);

  const streaming = state.status === "streaming";
  const numbers = footnoteNumbers(state.answer?.citations ?? []);
  // A dropped stream has no server payload: the client's own watchdog is the error.
  const shownError: AskError | null =
    state.error ??
    (state.status === "dropped" ? { code: "dropped", message: "", retryAt: null } : null);

  async function send() {
    const question = draft.trim();
    if (!question || streaming) return;
    setAskedAt(new Date().toISOString());
    // The composer empties at send time, so it never repeats the question shown above the answer,
    // and whatever the reader types while the answer streams is theirs to keep. A failed ask puts
    // the question back, but only when nothing else has been typed since.
    setDraft("");
    const status = await ask(question, item?.id ?? null);
    if (status === "error" || status === "dropped") {
      setDraft((current) => (current === "" ? question : current));
    }
  }

  function onKeyDown(event: React.KeyboardEvent<HTMLTextAreaElement>) {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      void send();
    }
  }

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
          className="fixed inset-0 z-[91] flex h-[100dvh] w-full flex-col overflow-hidden border border-rule bg-card shadow-[var(--shadow)] sm:inset-auto sm:left-1/2 sm:top-1/2 sm:h-[min(88dvh,880px)] sm:w-[min(880px,94vw)] sm:-translate-x-1/2 sm:-translate-y-1/2"
        >
          <header className="flex flex-none items-center gap-3 border-b border-rule px-5 py-3 sm:px-6">
            <Dialog.Title className="font-serif text-[17px] tracking-[-0.01em]">
              Ask the corpus
            </Dialog.Title>
            <Dialog.Description className="sr-only">
              Ask a question about the stored corpus. Answers cite their sources. Escape closes the
              sheet.
            </Dialog.Description>
            <Dialog.Close
              aria-label="Close the sheet"
              className="ml-auto grid h-11 w-11 place-items-center rounded-full text-ink-3 hover:bg-card-2 hover:text-ink"
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
          </header>

          <div className="min-h-0 flex-1 overflow-auto px-5 py-6 sm:px-9">
            {item ? (
              <p
                data-testid="asking-about"
                className="mb-3 max-w-[38rem] border-l-2 border-accent pl-2.5 text-[11.5px] leading-[1.5] text-ink-2"
              >
                About this item: <span className="font-serif text-[13px] text-ink">{item.title}</span>
              </p>
            ) : null}

            {state.question ? (
              <>
                <h2
                  data-testid="asked-question"
                  className="mb-3 max-w-[32ch] font-serif text-[24px] leading-[1.32] tracking-[-0.015em]"
                >
                  {state.question}
                </h2>
                <p className="mb-4 flex flex-wrap items-center gap-1.5 text-[11px] text-ink-3">
                  <span>you</span>
                  <span>·</span>
                  <span>{formatRome(askedAt)} Europe/Rome</span>
                  <span>·</span>
                  <span>all tiers · all topics</span>
                </p>
              </>
            ) : (
              <p className="max-w-[34ch] font-serif text-[21px] leading-[1.45] text-ink-2">
                Ask about the stored corpus. Every answer cites the items it used.
              </p>
            )}

            {state.notice ? (
              <p
                data-testid="notice"
                className="mb-4 max-w-[38rem] border border-rule bg-card-2 px-3 py-2 text-[11.5px] leading-[1.5] text-ink-2"
              >
                {state.notice.message}
              </p>
            ) : null}

            <Trace tools={state.tools} steps={state.steps} sourceCount={state.sources.length} />

            {shownError ? (
              <div
                data-testid="ask-error"
                className="max-w-[38rem] border border-bad/40 bg-bad-soft px-4 py-3"
              >
                <p className="text-[13px] leading-[1.55] text-ink">{errorCopy(shownError)}</p>
                {shownError.code === "daily_cap" ? (
                  <MatchingItems sources={state.sources} />
                ) : null}
              </div>
            ) : (
              <>
                {state.answer && (!state.answer.supported || state.answer.flags.length > 0) ? (
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
                <Sources sources={state.answer?.citations ?? state.sources} numbers={numbers} />
              </>
            )}
          </div>

          <footer className="flex-none border-t border-rule bg-card px-5 pb-[max(13px,env(safe-area-inset-bottom))] pt-3 sm:px-6">
            <div className="flex items-end gap-2 border border-rule-2 bg-paper px-3 py-2 focus-within:border-accent focus-within:shadow-[0_0_0_3px_var(--accent-soft)]">
              <textarea
                ref={textarea}
                data-testid="composer"
                rows={1}
                value={draft}
                maxLength={1000}
                onChange={(event) => setDraft(event.target.value)}
                onKeyDown={onKeyDown}
                placeholder={item ? "Ask about this item" : "Ask a new question"}
                aria-label="Your question"
                className="max-h-32 min-h-7 flex-1 resize-none bg-transparent font-serif text-[16px] leading-[1.4] text-ink outline-none focus-visible:outline-none placeholder:text-ink-3"
              />
              <button
                type="button"
                onClick={() => void send()}
                disabled={streaming || draft.trim() === ""}
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
              {draft.length > 800 ? <span className="ml-auto">{draft.length}/1000</span> : null}
            </p>
          </footer>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
