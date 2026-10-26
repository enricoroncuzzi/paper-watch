# The page

The Reading Room page: a Next.js App Router app that is the only caller of the API. Everything
under `src/` runs on the server unless it says otherwise; the browser receives rendered HTML and
the two route handlers it needs.

```sh
npm ci                 # install exactly the lockfile
npm run dev            # the page on :3000, against the fake API
npm run fake-api       # the fake API alone, on 127.0.0.1:8787 (FAKE_API_PORT moves it)
npm test               # Vitest: the pure logic (guards, safeHref, the API client, the SSE parser)
npm run e2e            # Playwright, Chromium and WebKit (including an iPhone 15), fake API only
npm run lint
npx prettier --check . # formatting at 100 columns (.prettierrc), as CI checks it
npm run typecheck      # next typegen, then tsc
npm run build
npm run gen:api        # regenerate src/lib/api-types.ts from ../docs/api/openapi.json
```

`npm run e2e` starts the fake API and a production build itself (see `playwright.config.ts`), so it
needs no other process: the page listens on 127.0.0.1:3100 and the fake API on 127.0.0.1:8787. A
busy port fails the run instead of testing whatever is on it, so set `PW_E2E_PORT` or
`PW_E2E_FAKE_API_PORT` to move one. It signs in through the test-only `e2e` provider, which exists
only when `PW_E2E_SESSION=1` and the app is not on Vercel.

## Developing against the fake API

Run the fake API in one terminal and the page in another:

```sh
npm run fake-api
PW_E2E_SESSION=1 PW_API_URL=http://127.0.0.1:8787 PW_API_AUTH=e2e \
  AUTH_SECRET=dev-secret AUTH_TRUST_HOST=true ALLOWED_GITHUB_ID=1 npm run dev
```

Then open <http://127.0.0.1:3000> and use "Sign in as the test owner". `POST /__scenario` on the fake
API picks which state the page shows (`normal`, `empty`, `error`, `unhealthy`, `followup`,
`text_only`, `slow`, `drop`).

Every fixture is validated against the API's own pydantic models by
`tests/unit/test_web_fake_api_fixtures.py`, so a fixture cannot drift from the contract.

## Environment variables

All are server-only; there is no `NEXT_PUBLIC_` variable.

| Name                                   | Purpose                                                   |
| -------------------------------------- | --------------------------------------------------------- |
| `PW_API_URL`                           | The API's base URL (the Funnel URL in production).        |
| `PW_API_AUTH`                          | `oidc` (default on Vercel production), `hmac`, or `e2e`.  |
| `PW_API_HMAC_SECRET`                   | The shared secret, only with `PW_API_AUTH=hmac`.          |
| `AUTH_GITHUB_ID`, `AUTH_GITHUB_SECRET` | The GitHub OAuth app.                                     |
| `AUTH_SECRET`                          | The Auth.js signing secret.                               |
| `ALLOWED_GITHUB_ID`                    | The one numeric GitHub id allowed in.                     |
| `PW_E2E_SESSION`                       | `1` enables the test provider and token, off Vercel only. |

## How a release deploys

Vercel builds only a release commit on the production branch: `web/scripts/vercel-ignore.sh` exits 1
(build) for a commit whose message starts with `chore(main): release`, and 0 (skip) for everything
else, including every preview. The page and the server therefore change together, and no
unprotected preview URL exists. The project's Root Directory is `web`, Node.js 24.
