import { Fragment } from "react";

const REFERENCE = /(\[S\d+\])/g;
const ONE_REFERENCE = /^\[(S\d+)\]$/;

/**
 * The answer, as text. Paragraphs and citation superscripts only: no markdown, no HTML. A `[S1]`
 * becomes a superscript only once the checked answer exists; before that the streamed words are
 * the model's draft and a link could point at a source the check later rejects.
 */
export function AnswerText({
  text,
  streaming,
  checked,
}: {
  text: string;
  streaming: boolean;
  checked: boolean;
}) {
  if (!text) return null;
  return (
    <div
      data-testid="answer"
      className="max-w-[34rem] font-serif text-[19px] leading-[1.66] tracking-[-0.003em] text-ink"
    >
      {text.split(/\n{2,}/).map((paragraph, index) => (
        <p key={index} className="mb-3.5">
          {checked ? renderReferences(paragraph) : paragraph}
          {streaming && index === text.split(/\n{2,}/).length - 1 ? (
            <span
              data-testid="caret"
              aria-hidden="true"
              className="ml-0.5 inline-block h-[19px] w-0.5 animate-pulse bg-accent align-[-4px]"
            />
          ) : null}
        </p>
      ))}
    </div>
  );
}

function renderReferences(paragraph: string): React.ReactNode[] {
  return paragraph.split(REFERENCE).map((piece, index) => {
    const match = ONE_REFERENCE.exec(piece);
    if (!match) return <Fragment key={index}>{piece}</Fragment>;
    const alias = match[1]!;
    return (
      <sup key={index} className="px-px text-[0.62em] font-semibold leading-none text-accent-text">
        <a href={`#fn-${alias}`} className="border-b border-transparent hover:border-current">
          {Number(alias.slice(1))}
        </a>
      </sup>
    );
  });
}
