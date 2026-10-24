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
  numbers,
}: {
  text: string;
  streaming: boolean;
  checked: boolean;
  numbers: Map<string, number>;
}) {
  if (!text) return null;
  const paragraphs = text.split(/\n{2,}/);
  return (
    <div
      data-testid="answer"
      className="max-w-[34rem] font-serif text-[19px] leading-[1.66] tracking-[-0.003em] text-ink"
    >
      {paragraphs.map((paragraph, index) => (
        <p key={index} className="mb-3.5">
          {checked ? renderReferences(paragraph, numbers) : paragraph}
          {streaming && index === paragraphs.length - 1 ? (
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

function renderReferences(paragraph: string, numbers: Map<string, number>): React.ReactNode[] {
  return paragraph.split(REFERENCE).map((piece, index) => {
    const match = ONE_REFERENCE.exec(piece);
    if (!match) return <Fragment key={index}>{piece}</Fragment>;
    const alias = match[1]!;
    const number = numbers.get(alias);
    // An alias the citation check dropped stays text: it has no footnote to link to.
    if (number === undefined) return <Fragment key={index}>{piece}</Fragment>;
    return (
      <sup key={index} className="px-px text-[0.62em] font-semibold leading-none text-accent-text">
        <a href={`#fn-${alias}`} className="border-b border-transparent hover:border-current">
          {number}
        </a>
      </sup>
    );
  });
}
