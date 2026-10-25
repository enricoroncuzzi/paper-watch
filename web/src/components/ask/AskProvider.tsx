"use client";

import { createContext, useContext, useEffect, useMemo, useState } from "react";

import { AskPill } from "@/components/ask/AskPill";
import { AskSheet } from "@/components/ask/AskSheet";

/** The item the sheet was opened from; the API registers it as S1 before the question runs. */
export type AskItem = { id: string; title: string };

const AskContext = createContext<{ askAbout: (item: AskItem) => void } | null>(null);

/** Cards use this to open the sheet with their item; a provider above them is required. */
export function useAskLauncher(): { askAbout: (item: AskItem) => void } {
  const launcher = useContext(AskContext);
  if (!launcher) throw new Error("useAskLauncher needs an AskProvider above it");
  return launcher;
}

/**
 * The one owner of the sheet and the floating way in. It sits above the page because a pick's
 * "Ask about it" has to open the same sheet the pill does, with that pick's item attached.
 */
export function AskProvider({ children }: { children?: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  const [item, setItem] = useState<AskItem | null>(null);

  const launcher = useMemo(
    () => ({
      askAbout(next: AskItem) {
        setItem(next);
        setOpen(true);
      },
    }),
    [],
  );

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
        event.preventDefault();
        setItem(null);
        setOpen(true);
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  return (
    <AskContext.Provider value={launcher}>
      {children}
      <AskPill
        onOpen={() => {
          setItem(null);
          setOpen(true);
        }}
      />
      <AskSheet
        open={open}
        item={item}
        onOpenChange={(next) => {
          if (!next) setItem(null);
          setOpen(next);
        }}
        onLeaveItem={() => setItem(null)}
      />
    </AskContext.Provider>
  );
}
