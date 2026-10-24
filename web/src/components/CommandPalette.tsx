"use client";

import * as Dialog from "@radix-ui/react-dialog";
import { Command } from "cmdk";
import { useEffect, useState } from "react";

const SECTIONS = [
  { value: "today", label: "Today strip", hint: "The last run at a glance" },
  { value: "runs", label: "Runs", hint: "The last seven daily ingestions" },
];

/**
 * Cmd+K / Ctrl+K opens a palette of the sections this release has. The hint in the masthead is
 * its trigger too, so the shortcut is discoverable without a keyboard.
 */
export function CommandPalette() {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setOpen((current) => !current);
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  function go(section: string) {
    setOpen(false);
    // A hash, not a router push: the sections live on the same page and the link is shareable.
    window.history.pushState(null, "", `#${section}`);
    document.getElementById(section)?.scrollIntoView({ block: "start" });
  }

  return (
    <Dialog.Root open={open} onOpenChange={setOpen}>
      <Dialog.Trigger asChild>
        <button
          type="button"
          aria-label="Open the command palette"
          className="hidden min-h-11 items-center gap-1 rounded-md px-1.5 font-mono text-[10px] text-ink-3 transition-colors hover:text-ink sm:flex sm:min-h-0"
        >
          <kbd className="rounded border border-rule-2 bg-card-2 px-1 py-0.5">⌘</kbd>
          <kbd className="rounded border border-rule-2 bg-card-2 px-1 py-0.5">K</kbd>
        </button>
      </Dialog.Trigger>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-[90] bg-black/25 backdrop-blur-[4px]" />
        <Dialog.Content className="fixed left-1/2 top-[16vh] z-[91] w-[min(580px,92vw)] -translate-x-1/2 border border-rule-2 bg-card shadow-[var(--shadow)]">
          <Dialog.Title className="sr-only">Command palette</Dialog.Title>
          <Dialog.Description className="sr-only">
            Jump to a section of the page. Escape closes the palette.
          </Dialog.Description>
          <Command label="Command palette">
            <div className="flex items-center gap-2 border-b border-rule px-4 py-3">
              <svg
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.8"
                aria-hidden="true"
                className="text-ink-3"
              >
                <circle cx="11" cy="11" r="7" />
                <path d="m20 20-4-4" />
              </svg>
              <Command.Input
                autoFocus
                placeholder="Jump to a section"
                className="w-full bg-transparent font-serif text-[17px] text-ink outline-none placeholder:text-ink-3"
              />
            </div>
            <Command.List className="max-h-[46vh] overflow-auto p-1.5">
              <Command.Empty className="px-3 py-3 text-[12.5px] text-ink-3">
                Nothing matches.
              </Command.Empty>
              {SECTIONS.map((section) => (
                <Command.Item
                  key={section.value}
                  value={section.label}
                  onSelect={() => go(section.value)}
                  className="flex cursor-pointer items-center gap-2 rounded-[3px] px-3 py-2.5 text-[12.5px] text-ink-2 data-[selected=true]:bg-accent-soft data-[selected=true]:text-ink"
                >
                  <span className="min-w-0 flex-1">{section.label}</span>
                  <span className="text-[10.5px] text-ink-3">{section.hint}</span>
                </Command.Item>
              ))}
            </Command.List>
          </Command>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
