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

## A conversation on the page

The sheet holds one thread (`web/src/lib/use-ask.ts`): the finished turns above the current one, the
thread id the server returned with the last answer, and the current question with its stream. A
follow-up sends that id, so the API replays the thread's last turns and keeps its source aliases, and
the reply's footnote numbers stay per answer. Three fixed chips sit under the composer; the
peer-reviewed one re-asks the last question with `tiers: ["peer_reviewed"]`, so a filter is not a
model call and the chips cost nothing to invent.

The thread ends when the sheet closes: the controller starts clean, so "Ask about it" on a card
always opens a new thread about that item, which the API registers as `S1`. A conversation picked
from Recent is read back with `GET /api/threads/{id}` and lets go of the item, so a follow-up there
never claims to be about a different paper. A follow-up also waits for the stream to end, not only
for the answer: the `thread` event that carries the id and the turn number comes after the server's
write, and a send in that gap would start a second, history-less thread.

A dropped stream does not end the conversation. The controller asks `GET /api/threads/{id}` and
shows what the server finished: the answer is adopted when the thread saved it, and when the thread
has no turn for the question the sheet says the answer was interrupted and offers to ask it again.
Adoption needs the turn's number as well as its words, because a chip re-asks the same question:
otherwise the unfiltered answer to the first ask would be shown as the answer to the filtered one.

Returning to the page runs that read too, because iOS Safari freezes a backgrounded tab and its
watchdog timer with it, but it never aborts the stream: a tab switch is not a dead connection. Only
the read that adopts a finished turn stops the stream, because the answer is then no longer only in
the stream. A live stream with no stored turn yet is left to finish, and a dead one still ends
through the reader error or the 45-second watchdog. Both the ask proxy and the thread reads are
route handlers, which check the session and hold the credential on the server.

## Stored text stays text

React escapes what it renders, so a title with HTML in it is characters on the page. The one place a
stored string becomes a link is `safeHref`: only `http:` and `https:` survive, a URL with an
invisible Unicode `Cf` character (a zero-width space, a bidi override) is refused, and so is one
whose port does not parse. `safeHost` is what the reader sees as the link text, so a misleading path
is not the label. The rule exists because the corpus stores whatever a source published, including
a Hugging Face project page crafted to look like another site.

## The fake API and the contract

`web/fake-api/server.mjs` is a dependency-free `node:http` server on `127.0.0.1:8787` (move it with
`FAKE_API_PORT`, or with `PW_E2E_FAKE_API_PORT` for the suite). It accepts
only `Authorization: Bearer e2e`, serves `fixtures/<name>.json` for the read routes, streams
`streams/<scenario>.json` for `POST /ask`, and takes `POST /__scenario` so a Playwright test can pick
`normal`, `empty`, `error`, `unhealthy`, `followup`, `text_only`, `slow`, `drop`, `thread_drop` or
`thread_slow` (the `thread` event held back, so a test can act between it and the answer).
It keeps the conversations a test created: a finished answer appends a turn and the stream ends with
the `thread` event, `GET /threads` and `GET /threads/{id}` read them back, and `keepThreads: true`
on a scenario switch lets one test ask, then change how the next stream behaves. `GET /__last_ask`
is the test seam for what the page sent (a follow-up's thread and the chip's filter). Tests run one
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
