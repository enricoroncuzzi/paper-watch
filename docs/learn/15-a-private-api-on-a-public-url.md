# A private API on a public URL

A tunnel makes a loopback service reachable from the internet without opening a port, which also
means anyone who learns the name can knock. So the first thing every request meets is the
credential, before any parsing or model work, and every failure looks the same so a caller learns
nothing. The credential here is a token Vercel signs for code running in one project's production
deployment: the API checks the signature against Vercel's published keys and pins the issuer, the
audience and the exact project and environment, so there is no shared secret to leak or rotate. The
browser never talks to the API at all; the page's server does, and it sends no `Origin`, so any
request that carries one came from a browser and is refused.

`src/paperwatch/api/gate.py` is an ASGI middleware, not a FastAPI dependency: it runs before a route
matches, before a body is parsed, and before `create_app` can hand anything to the agent. `gate.py`
holds the order and `auth.py` the two verifiers.

## The order, and why each step is where it is

| Step | Refusal | Reason it comes here |
|---|---|---|
| Authorization bearer token | 401, one body for every reason | The most expensive work (a model call) is behind it; a bad token never reads the body or the host list. |
| `Host` allow-list | 400 | A DNS-rebinding page can point at a loopback API; the tunnel's name is added by `PW_API_PUBLIC_HOST`. |
| Any `Origin` header | 403 | The page's server sends none; a browser always does, so the request came from a page that must not call this API. |
| `POST` requires `application/json` | 415 | A form-encoded body, or a plain text one, is never the shape the routes accept; refusing it before parsing keeps the validators simple. |
| `Content-Length` over 16 KB, or a chunked body that grows past it | 413 | A header can be absent or lie, so the receive channel counts bytes too. |

`/healthz` is the one route without a credential: a liveness probe has to answer when nothing else
does, and that route touches neither the database nor a model. Every other route, `/readyz`
included, needs the token. The gate's host check still applies to `/healthz`, so the health route
is not a way around the allow-list.

## Verifying a Vercel OIDC token

`OidcVerifier` accepts RS256 only. The algorithm is chosen in the code, never read from the token's
header, so `alg: none` and the classic HS256-signed-with-the-public-key trick both fail at the
first check. The token's key id selects a key from `JwksCache`, which fetches
`<issuer>/.well-known/jwks` once at startup and then on an unknown key id at most once per five
minutes: a key id is caller-chosen, so without the window a flood of made-up ids would be a free
amplification against Vercel's key endpoint. A failed fetch is remembered like a successful one and returns "no
key"; it never raises into the request.

After the signature, the claims are checked: `iss`, `aud`, and the exact `sub` (Vercel's subject
names the owner, project and environment, so a preview deployment's token cannot pass for
production), `exp` and `nbf` with 30 seconds of clock skew, and `iat` (which PyJWT does not check)
against the same skew. A claim mismatch is logged as `token_claim_mismatch` with the claim name and
both values: they are Vercel's own, not secrets, and this is how an operator reads the real claims
the first day a new project's token is refused.

## The HMAC fallback

`PW_API_AUTH=hmac` switches to `HmacVerifier`, for local operation and the tunnel rehearsal before a
Vercel project exists. `paperwatch api-token --live` mints one token with a 60 second life, and its
signature is compared with `hmac.compare_digest`. Like the OIDC token, one whose `iat` lies more
than 30 seconds in the future is refused. A token's `jti` is remembered until the token
expires, so it is single use: one copied out of a shell history or a log does not work twice. The
secret must be at least 32 characters, or `make_verifier` refuses to build the verifier and `serve`
exits 2 naming `PW_API_HMAC_SECRET`. Two credentials never run at once; the setting picks one.

## What each status code means

| Code | Meaning | What to do |
|---|---|---|
| 401 | The credential is missing, malformed, expired, from another project, or `alg` is not RS256. | Fetch a fresh token. Do not retry the same one. |
| 400 | The `Host` is not in the allow-list, or `Content-Length` is not a number. | Check the tunnel configuration and `PW_API_PUBLIC_HOST`. |
| 403 | The request carried an `Origin` header. | Call the API from server code, not from a browser. |
| 413 | The body is over 16 KB, by header or as it streams in. | Send less; the largest legitimate body is a 2,000 character points-of-view box. |
| 415 | A `POST` without `content-type: application/json`. | Set the header. |
| 429 | The per-client limit refused the request. | Wait for `Retry-After` seconds. |

The body of every refusal is `{"detail": "<short reason>"}`. Nothing in it names a header value, a
key or a host, and the log line carries a reason code only.

## What this does not do

The gate does not rate-limit (that is `api/limits.py`, which counts each credential subject the
gate stored and shares the answer slots across callers), does not decide which routes exist
(FastAPI's router still answers 405), and does not protect the database: the API's Postgres role is read-only apart from what
`docs/deploy/vps.md` grants it. `/docs`, `/redoc` and `/openapi.json` are served, not removed; the
gate puts the same credential in front of them, which is why a browser without the header gets
401.
