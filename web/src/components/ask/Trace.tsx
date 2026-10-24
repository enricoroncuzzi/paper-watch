import type { StepPayload, ToolPayload } from "@/lib/use-ask";

function shortArgs(args: Record<string, unknown>): string {
  const text = Object.entries(args)
    .map(([key, value]) => `${key}=${String(value)}`)
    .join(", ");
  return text.length > 52 ? `${text.slice(0, 51)}…` : text;
}

/**
 * What the agent did: one line per tool call with its arguments shortened, the rounds with their
 * timings, and how many sources the answer has. Nothing here is model text.
 */
export function Trace({
  tools,
  steps,
  sourceCount,
}: {
  tools: ToolPayload[];
  steps: StepPayload[];
  sourceCount: number;
}) {
  if (tools.length === 0 && steps.length === 0) return null;
  return (
    <div
      data-testid="trace"
      className="mb-5 flex flex-wrap items-center gap-x-2.5 gap-y-1.5 border-y border-rule py-2 text-[11px] text-ink-2"
    >
      {tools.map((tool, index) => (
        <span key={index} className="flex items-center gap-1.5 font-mono text-[10px]">
          <span aria-hidden="true" className="h-[5px] w-[5px] rounded-full bg-accent" />
          {tool.name}({shortArgs(tool.args)})
        </span>
      ))}
      {steps.map((step) => (
        <span key={step.round} className="font-mono text-[10px] text-ink-3">
          round {step.round} · {step.ms} ms · +{step.new_sources}
        </span>
      ))}
      <span className="ml-auto font-mono text-[9.5px] text-ink-3">
        {sourceCount} source{sourceCount === 1 ? "" : "s"}
      </span>
    </div>
  );
}
