// Where the two servers the e2e suite starts listen, shared by `playwright.config.ts` and the tests
// so an override moves both. A fixed port can belong to another checkout's leftover build, so a
// busy one must fail the run rather than be tested.
export const PORT = Number(process.env.PW_E2E_PORT ?? 3100);
export const FAKE_API_PORT = Number(process.env.PW_E2E_FAKE_API_PORT ?? 8787);
export const BASE_URL = `http://127.0.0.1:${PORT}`;
export const API_URL = `http://127.0.0.1:${FAKE_API_PORT}`;
