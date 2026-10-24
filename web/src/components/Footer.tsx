/**
 * The page's footer: what the corpus owes its sources, and which release built this page. The
 * commit sha is Vercel's own variable, so it names the release without a build step of our own.
 */
export function Footer() {
  const sha = process.env.VERCEL_GIT_COMMIT_SHA?.slice(0, 7);
  return (
    <footer className="mt-11 flex flex-wrap gap-x-4 gap-y-1 border-t border-rule pt-3 text-[10.5px] text-ink-3">
      <span>Citations from Semantic Scholar. Upvotes from Hugging Face.</span>
      <span className="ml-auto font-mono">{sha ? `release ${sha}` : "release dev"}</span>
    </footer>
  );
}
