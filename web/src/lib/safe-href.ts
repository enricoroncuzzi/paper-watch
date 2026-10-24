const ALLOWED_PROTOCOLS = new Set(["http:", "https:"]);
// Unicode `Cf` characters are invisible: a zero-width space or a bidi override in a stored URL
// exists only to make a reader believe a link goes somewhere other than it does.
const FORMAT_CHARACTER = /\p{Cf}/u;

/**
 * The href for a stored URL, or undefined when it must not become a link. Only http(s) survives:
 * `javascript:`, `data:` and `vbscript:` URLs are text, not navigation.
 */
export function safeHref(url: string | null | undefined): string | undefined {
  if (!url || FORMAT_CHARACTER.test(url)) return undefined;
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    return undefined;
  }
  if (!ALLOWED_PROTOCOLS.has(parsed.protocol)) return undefined;
  // A port outside 1-65535 makes URL throw in current runtimes, but a stored string is checked
  // here rather than trusted to the parser version.
  if (parsed.port) {
    const port = Number(parsed.port);
    if (!Number.isInteger(port) || port < 1 || port > 65535) return undefined;
  }
  return url;
}

/** The text to show for a stored URL: its host, so a misleading path is not what the reader sees. */
export function safeHost(url: string | null | undefined): string | undefined {
  const href = safeHref(url);
  if (!href) return undefined;
  try {
    return new URL(href).hostname;
  } catch {
    return undefined;
  }
}
