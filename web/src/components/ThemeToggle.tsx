"use client";

const NEXT_THEME = { system: "light", light: "dark", dark: "system" } as const;
type Theme = keyof typeof NEXT_THEME;

function currentTheme(): Theme {
  const set = document.documentElement.dataset.theme;
  return set === "light" || set === "dark" ? set : "system";
}

/**
 * Light, dark, system. The choice is a cookie so the server renders the right palette on the next
 * load; the attribute changes here so this page does not wait for a reload. The button reads the
 * current theme from the document at click time, which is also what a reload just set.
 */
export function ThemeToggle() {
  function toggle() {
    const next = NEXT_THEME[currentTheme()];
    // Secure on the real site; localhost is plain HTTP, where a Secure cookie is refused.
    const secure = window.location.protocol === "https:" ? "; secure" : "";
    document.cookie = `pw-theme=${next}; path=/; max-age=31536000; samesite=lax${secure}`;
    if (next === "system") {
      delete document.documentElement.dataset.theme;
    } else {
      document.documentElement.dataset.theme = next;
    }
  }

  return (
    <button
      type="button"
      onClick={toggle}
      aria-label="Toggle theme"
      className="grid min-h-11 min-w-11 place-items-center rounded-full text-ink-3 transition-colors hover:bg-card-2 hover:text-ink sm:min-h-0 sm:min-w-0 sm:h-[26px] sm:w-[26px]"
    >
      <svg
        width="15"
        height="15"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.6"
        aria-hidden="true"
      >
        <path d="M21 12.8A8.5 8.5 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z" />
      </svg>
    </button>
  );
}
