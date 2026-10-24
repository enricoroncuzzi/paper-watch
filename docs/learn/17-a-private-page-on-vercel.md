# A private page on Vercel

The page is a Next.js app whose server is the only thing that talks to the API: components that
read data run on the server, and what a browser starts (a question) goes through a route handler.
Every route handler checks the GitHub session itself, because a check in middleware alone
has been bypassed before. Sign-in compares the numeric GitHub id, which never changes, not the
login name, which can. The API's address and credentials exist only on the server, so the browser
sees neither. Tests sign in through a credentials provider that exists only outside Vercel, and
talk to a fake API whose every fixture is validated against the real API's models, so a passing
end-to-end test means the page and the contract agree.

## The session, checked twice

`web/src/auth.config.ts` builds the Auth.js config: the GitHub provider, JWT sessions (there is one
owner and nothing to persist), and a `signIn` callback that returns true only when
`String(profile.id) === ALLOWED_GITHUB_ID`. The id, not the login: a GitHub login can be renamed and
then claimed by someone else, while the numeric id never changes.

`web/src/auth.ts` holds the one `NextAuth` instance and the guards. `ownerSession()` reads the
session and compares `githubId` with the allowed id again, so a session minted before the variable
changed stops working. `requireOwner()` redirects a page to `/signin`; `requireOwnerApi()` returns a
401 `{"kind": "unauthorized"}` for a route handler. They are two functions because one cannot both
throw a redirect and return a body, and the wrong one at a route handler would answer a fetch with a
307 the caller follows.

`web/src/proxy.ts` (Next 16's name for middleware) redirects an anonymous page load before the
component runs. That is a courtesy, not the check: every server component and every route handler
calls a guard itself, and the matcher excludes `/api` because those routes answer their own 401.

The test provider is a credentials provider with id `e2e`, added to the config only when
`e2eSessionAllowed` is true: `PW_E2E_SESSION=1` and neither `VERCEL` nor `VERCEL_ENV` set. The sign-in
page renders its button only under the same condition. Two unit tests pin the refusal, and the e2e
suite proves the normal path still works.

## The credential the page presents

`web/src/lib/api-token.ts` is the only place a token is minted, and it never leaves the server:

- on Vercel production (or `PW_API_AUTH=oidc`), `getVercelOidcToken()` from `@vercel/oidc`, which
  the API verifies against Vercel's public keys and pins to the production project. It is called
  with no audience option, so the token carries Vercel's default audience (the team URL): the API
  pins the issuer, the audience and the exact project plus production subject, and the subject is
  what confines the token to this project. `getVercelOidcToken({audience})` exists, but a custom
  audience would add a second lock and a `PW_VERCEL_OIDC_AUDIENCE` value to keep in step with the
  server, and the subject already does the confining;
- with `PW_API_AUTH=hmac`, a 60-second HS256 JWT signed with `jose`, with the same audience, subject,
  `iat`, `exp` and single-use `jti` that `HmacVerifier` in `src/paperwatch/api/auth.py` requires. This
  is also how the tunnel is rehearsed before the Vercel project exists;
- with `PW_API_AUTH=e2e`, the literal `e2e` the fake API accepts, and only when the e2e guard allows
  it. Asking for it on Vercel throws, so the token cannot exist there even if the variable is set.

## Errors are states, not exceptions

`web/src/lib/api.ts` maps every outcome to `{ok: true, data}` or `{ok: false, kind, retryAfterS?}`:
401 and 403 to `unauthorized`, 404 to `not_found`, 429 to `rate_limited` (with `Retry-After`), 5xx
to `unavailable`, and a timeout or refused connection also to `unavailable`. A section renders the
copy for its kind instead of the page failing. Reads time out at ten seconds and are never cached
(`cache: "no-store"`): a cached run list would show yesterday's ingestion as today's.

## Stored text stays text

React escapes what it renders, so a title with HTML in it is characters on the page. The one place a
stored string becomes a link is `safeHref`: only `http:` and `https:` survive, a URL with an
invisible Unicode `Cf` character (a zero-width space, a bidi override) is refused, and so is one
whose port does not parse. `safeHost` is what the reader sees as the link text, so a misleading path
is not the label. The rule exists because the corpus stores whatever a source published, including
a Hugging Face project page crafted to look like another site.

## The fake API and the contract

`web/fake-api/server.mjs` is a dependency-free `node:http` server on `127.0.0.1:8787`. It accepts
only `Authorization: Bearer e2e`, serves `fixtures/<name>.json` for the read routes, streams
`streams/<scenario>.json` for `POST /ask`, and takes `POST /__scenario` so a Playwright test can pick
`normal`, `empty`, `error`, `unhealthy`, `followup`, `text_only`, `slow` or `drop`. Tests run one
worker, so the global scenario is safe.

`tests/unit/test_web_fake_api_fixtures.py` validates every fixture against the API's pydantic
response models and every stream frame against its event payload model, by event name. That is the
link between the two halves of the repository: the fake cannot drift from the contract, and a
renamed field fails in Python before it becomes a blank cell on the page.

`npm run gen:api` regenerates `web/src/lib/api-types.ts` from `docs/api/openapi.json`, and CI fails
if the committed file differs. A route change therefore lands as one commit: contract, generated
types, and the page that reads them.

## Deploying the page

`web/vercel.json` sets an ignore command that runs `web/scripts/vercel-ignore.sh`: Vercel builds only
when `VERCEL_ENV` is `production` and the commit message starts with `chore(main): release`. The page
and the server therefore change together, and no preview deployment exists, so there is no
unprotected `*.vercel.app` URL to find. The project's Root Directory is `web`.

Environment variables, Production only, by name: `AUTH_GITHUB_ID`, `AUTH_GITHUB_SECRET`,
`AUTH_SECRET`, `ALLOWED_GITHUB_ID`, `PW_API_URL`. No `NEXT_PUBLIC_` variable exists: nothing in the
browser may know the API's address.
