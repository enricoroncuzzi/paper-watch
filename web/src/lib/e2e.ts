/**
 * The environment a guard reads. A plain map, not `NodeJS.ProcessEnv`, so a test can pass the
 * handful of variables it cares about without faking the rest of the process.
 */
export type Env = Readonly<Record<string, string | undefined>>;

/**
 * The test-only sign-in and API token exist only when a test asks for them and the app is not on
 * Vercel. Both checks are here, in one place, so neither path can be half-guarded.
 */
export function e2eSessionAllowed(env: Env): boolean {
  return env.PW_E2E_SESSION === "1" && !env.VERCEL && !env.VERCEL_ENV;
}
