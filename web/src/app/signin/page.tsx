import { E2E_PROVIDER_ID, GITHUB_PROVIDER_ID, signIn } from "@/auth";
import { e2eSessionAllowed } from "@/lib/e2e";

async function signInWithGithub() {
  "use server";
  await signIn(GITHUB_PROVIDER_ID, { redirectTo: "/" });
}

async function signInAsTestOwner() {
  "use server";
  await signIn(E2E_PROVIDER_ID, { redirectTo: "/" });
}

export default function SignInPage() {
  // The test button is rendered only where the provider exists; the provider itself is the gate.
  const testOwner = e2eSessionAllowed(process.env);
  return (
    <main className="mx-auto flex min-h-dvh w-full max-w-[35rem] flex-col justify-center px-6 py-16">
      <p className="font-mono text-[11px] uppercase tracking-[0.14em] text-ink-3">Reading Room</p>
      <h1 className="mt-3 font-serif text-4xl font-medium tracking-[-0.02em]">Paper Watch</h1>
      <p className="mt-4 max-w-[30rem] text-[15px] leading-7 text-ink-2">
        A private reading room for the AI-research corpus. Sign in with the one GitHub account this
        page belongs to.
      </p>
      <form action={signInWithGithub} className="mt-8">
        <button
          type="submit"
          className="inline-flex min-h-11 items-center gap-2 rounded-md border border-rule-2 bg-card px-4 py-2.5 text-[13.5px] font-medium text-ink transition-colors hover:border-accent-text"
        >
          Sign in with GitHub
        </button>
      </form>
      {testOwner ? (
        <form action={signInAsTestOwner} className="mt-4">
          <button
            type="submit"
            className="inline-flex min-h-11 items-center rounded-md border border-dashed border-rule-2 px-4 py-2.5 font-mono text-[11px] text-ink-3"
          >
            Sign in as the test owner
          </button>
        </form>
      ) : null}
      <p className="mt-10 border-t border-rule pt-4 font-mono text-[11px] text-ink-3">
        Private. One account. No password.
      </p>
    </main>
  );
}
