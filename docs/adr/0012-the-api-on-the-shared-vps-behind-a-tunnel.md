# 12. The API on the shared VPS sits behind a credential gate and a userspace tunnel

Date: 2026-10-24

## Status

Accepted. The API's Unix account and unit type are superseded by
[ADR 0017](0017-the-api-under-its-own-unix-account.md); the tunnel and everything else here still
hold.

## Context

The API must be reachable from the Vercel page without changing the shared VPS's networking,
firewall, SSH or Postgres configuration, without root and on free tiers only. The established
path (ADR 0009) runs Paper Watch as the `paperwatch` account with systemd user units in
`/home/paperwatch`, next to another production project, and the ingestion must never be stopped
for the API.

A tunnel makes a loopback service reachable from the internet without opening a port, which also
means anyone who learns the name can knock. Three shapes were on the table: publish through a
proxy that needs root (Caddy), use a hosted tunnel with a static domain (ngrok free), or join the
machine to a tailnet and publish one port through Funnel (Tailscale userspace). The deployment measurements
chose the userspace path, and it is the only one that needs neither root nor a TUN device.

For the credential, the page's server (not the browser) is the only caller. Vercel signs a token
for code running in one production deployment, and the API can verify it against Vercel's
published keys while pinning the issuer, the audience and the exact subject (owner, project and
environment), so there is no shared secret to leak or rotate and a preview deployment's token
cannot pass. Entra ID was an earlier sketch; it would add a tenant, a client secret and an
entra-specific verifier for no benefit over the platform the page already runs on. A short-lived
HMAC token stays as the fallback for local operation and for rehearsing the tunnel before the
Vercel project exists.

## Decision

- **Transport:** Tailscale in userspace as its own user unit (`deploy/vps/paperwatch-tunnel.service`,
  `MemoryMax=150M`, outside `paperwatch.slice`), one Funnel mapping of 443 to
  `127.0.0.1:8000` set once by the runbook. The API itself binds `127.0.0.1` only
  (`deploy/vps/paperwatch-api.service`). If any step needs root or a TUN device the rehearsal
  failed and the fallback (ngrok's free static domain) is a captain decision.
- **Credential:** `PW_API_AUTH=oidc` (default) verifies RS256 Vercel OIDC tokens against the
  issuer's `/.well-known/jwks`, with `iss`, `aud` and the exact `sub` pinned, 30 s of clock skew,
  and a key-set refetch at most once per five minutes per unknown key id. `PW_API_AUTH=hmac` is the
  fallback: a 60 s HS256 token compared with `hmac.compare_digest`, single use by `jti` so a token
  copied out of a log cannot be replayed. One credential at a time.
- **The gate:** `src/paperwatch/api/gate.py`, plain ASGI in front of every route, in this order:
  credential (401, one body for every reason, the body never read first), `Host` allow-list (400),
  any `Origin` (403), JSON content type on `POST` (415), 16 KB body cap by header and by counted
  bytes (413). `GET /healthz` is the one route without a credential: a liveness probe must answer
  when nothing else does, and that route touches neither the database nor a model. `/docs`,
  `/redoc` and `/openapi.json` are served only to an authenticated request, as the captain asked,
  not removed; the gate is in front of them like every other route. The committed contract
  (`docs/api/openapi.json`) stays the client's reference because Swagger UI cannot attach the
  header from a browser.
- **Limits:** `/ask` is limited per credential subject to 6 requests a minute and 100 a UTC day,
  and the process to 2 concurrent answers (`api/limits.py`), refused as 429 with `Retry-After`
  before any model call. An answer ends at 120 s with a `deadline` error, and `: ping` keeps a
  silent stream alive.
- **Units:** the API as a long-lived user unit with `MemoryMax=700M`, no swap, `OOMScoreAdjust=900`,
  joining `paperwatch.slice` (ADR 0013). `deploy/vps/api-check.sh` is the acceptance script; it
  takes a token-minting command so each authenticated request gets a fresh HMAC token.

## Consequences

- The tunnel's machine name appears in public Certificate Transparency logs, so it is neutral and
  holds nothing else; the tailnet account is free-tier.
- Nothing new listens on the VPS: the API stays on loopback, and there is no reverse proxy, so the
  body cap lives only in the gate. A Caddy fallback would need `request_body max_size 16KB` too.
- The Vercel project (and its three OIDC values) is created by the captain; until then the HMAC
  mode is how the tunnel is rehearsed. A leaked HMAC token is valid for at most 60 s and once.
- The deployment step reads live; builders stop at merged code, units, migrations and the runbook
  (section 15 of `docs/deploy/vps.md`).
