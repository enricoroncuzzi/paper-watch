import { requireOwner } from "@/auth";

// The page shell lands next; this placeholder proves the session gate end to end.
export default async function Home() {
  const session = await requireOwner();
  return (
    <main className="mx-auto w-full max-w-[35rem] px-6 py-16">
      <p className="font-mono text-[11px] uppercase tracking-[0.14em] text-ink-3">Reading Room</p>
      <h1 className="mt-3 font-serif text-4xl font-medium tracking-[-0.02em]">Paper Watch</h1>
      <p className="mt-4 text-[15px] leading-7 text-ink-2">
        Signed in as GitHub account {session.githubId}.
      </p>
    </main>
  );
}
