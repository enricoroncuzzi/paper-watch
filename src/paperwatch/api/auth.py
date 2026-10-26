"""The two ways a caller proves it may use the API, and the key set behind the main one.

The browser never calls the API: the page's server does, carrying a token Vercel signs for code in
one project's production deployment. Verifying it needs no shared secret, only Vercel's published
keys and the expected issuer, audience and subject, so a leaked token expires in minutes and there
is nothing else to rotate. The HMAC verifier is the flagged fallback for local operation and the
tunnel rehearsal before a Vercel project exists.

A verifier returns `None` for every refusal, never a reason: the gate answers every one of them
with the same 401, so a caller learns nothing about why.
"""

import base64
import contextlib
import hashlib
import hmac
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

import httpx
import jwt
import structlog
from jwt import PyJWK, PyJWKSet

from paperwatch.config import (
    HMAC_AUDIENCE,
    HMAC_TOKEN_TTL_S,
    JWKS_REFETCH_MIN_S,
    TOKEN_LEEWAY_S,
    Settings,
)

log = structlog.get_logger()

JWKS_FETCH_TIMEOUT_S = 5.0
# A fetch that failed is retried sooner than a successful one: a transient error on the first
# request after a restart must not lock the page out for the full key-rotation window.
JWKS_NEGATIVE_CACHE_S = 30.0
# A shared secret shorter than this is a guess away from being forged; the API refuses to start
# with one rather than pretending it is configured.
HMAC_SECRET_MIN_CHARS = 32


class ApiAuthNotConfigured(RuntimeError):
    """The credential cannot be verified because a setting is missing; the API refuses to start."""


@dataclass(frozen=True)
class Principal:
    """Who called: the credential's kind, and the subject the per-client limiter keys on."""

    kind: Literal["oidc", "hmac"]
    subject: str


class Verifier(Protocol):
    def verify(self, token: str) -> Principal | None: ...


class RefuseEveryToken:
    """The verifier of a process that serves no route, such as `paperwatch ask`: it accepts none."""

    def verify(self, token: str) -> Principal | None:
        return None


class JwksCache:
    """Vercel's signing keys, fetched on first use and on an unknown key id at most once a window.

    A key id is chosen by the caller, so a flood of made-up ids would otherwise be a free
    amplification against Vercel's endpoint. The window is claimed before the fetch and the fetch
    itself runs outside the lock, so one slow request never blocks the others; while it runs, a
    second caller is treated as "no key yet" and refused rather than queued. The API calls
    `prefetch` at startup so the first request after a restart does not race that fetch. A failed
    fetch is remembered for a shorter window than a successful one.
    """

    def __init__(
        self,
        url: str,
        fetch: Callable[[str], dict[str, Any]],
        clock: Callable[[], float],
        refetch_min_s: float = JWKS_REFETCH_MIN_S,
        negative_s: float = JWKS_NEGATIVE_CACHE_S,
    ) -> None:
        self._url = url
        self._fetch = fetch
        self._clock = clock
        self._refetch_min_s = refetch_min_s
        self._negative_s = negative_s
        self._lock = threading.Lock()
        self._keys: dict[str, PyJWK] = {}
        self._attempted_at: float | None = None
        self._failed = False

    def key(self, kid: str) -> PyJWK | None:
        with self._lock:
            if self._attempted_at is None:
                should_fetch = True
            else:
                key = self._keys.get(kid)
                if key is not None:
                    return key
                # An unknown key id may mean a rotation, so one refetch is allowed per window.
                window = self._negative_s if self._failed else self._refetch_min_s
                should_fetch = self._clock() - self._attempted_at >= window
                if not should_fetch:
                    return None
            # Claim the window before releasing the lock: a hundred concurrent callers must start
            # at most one fetch, and the fetch itself must not hold the lock.
            self._attempted_at = self._clock()
        key_set = self._fetch_keys()
        with self._lock:
            self._failed = key_set is None
            if key_set is not None:
                self._keys = key_set
            return self._keys.get(kid)

    def prefetch(self) -> None:
        """Fetch now, once, so the first request after a restart does not race the key set."""
        with self._lock:
            if self._attempted_at is not None:
                return
            self._attempted_at = self._clock()
        key_set = self._fetch_keys()
        with self._lock:
            self._failed = key_set is None
            if key_set is not None:
                self._keys = key_set

    def _fetch_keys(self) -> dict[str, PyJWK] | None:
        try:
            key_set = PyJWKSet.from_dict(self._fetch(self._url))
        except Exception as exc:  # noqa: BLE001 - any fetch failure means "no key", never a crash
            # Only the type: the URL and a provider body are not for the log.
            log.warning("jwks_fetch_failed", error_type=type(exc).__name__)
            return None
        return {key.key_id: key for key in key_set.keys if key.key_id}


class OidcVerifier:
    """Verify a Vercel OIDC token: RS256 only, pinned issuer, audience and exact subject."""

    def __init__(
        self,
        *,
        issuer: str,
        audience: str,
        subject: str,
        keys: JwksCache,
        clock: Callable[[], float],
    ) -> None:
        self._issuer = issuer
        self._audience = audience
        self._subject = subject
        self._keys = keys
        self._clock = clock

    def verify(self, token: str) -> Principal | None:
        try:
            header = jwt.get_unverified_header(token)
            kid = header.get("kid")
            # The algorithm is chosen here, never read from the token: `none`, or HS256 with the
            # public key's PEM as a shared secret, is the classic way past a naive verifier.
            if header.get("alg") != "RS256" or not isinstance(kid, str):
                return None
            key = self._keys.key(kid)
            if key is None:
                return None
            claims = jwt.decode(
                token,
                key=key,
                algorithms=["RS256"],
                audience=self._audience,
                issuer=self._issuer,
                # exp, nbf and iat are checked below against the injected clock, so a test can place
                # a token on either side of a boundary without sleeping or faking wall time.
                # verify_iat especially: PyJWT's own check has no leeway at all, so a server clock a
                # second or two behind Vercel's would refuse a fresh token before our 30 s check.
                options={
                    "require": ["exp", "iat", "aud", "iss", "sub"],
                    "verify_exp": False,
                    "verify_nbf": False,
                    "verify_iat": False,
                },
            )
        except jwt.InvalidAudienceError:
            self._claim_mismatch(token, "aud", self._audience)
            return None
        except jwt.InvalidIssuerError:
            self._claim_mismatch(token, "iss", self._issuer)
            return None
        except jwt.PyJWTError:
            return None
        if claims.get("sub") != self._subject:
            self._claim_mismatch(token, "sub", self._subject)
            return None
        now = self._clock()
        exp, nbf, iat = (claims.get(name) for name in ("exp", "nbf", "iat"))
        if not isinstance(exp, int | float) or exp + TOKEN_LEEWAY_S <= now:
            return None
        if isinstance(nbf, int | float) and nbf > now + TOKEN_LEEWAY_S:
            return None
        # PyJWT checks exp and nbf but not iat, so a token minted in the future is refused here.
        if not isinstance(iat, int | float) or iat > now + TOKEN_LEEWAY_S:
            return None
        return Principal("oidc", self._subject)

    def prefetch(self) -> None:
        """Warm the key set at API startup; a failed fetch is retried after the negative window."""
        self._keys.prefetch()

    def _claim_mismatch(self, token: str, claim: str, expected: str) -> None:
        # The signature already verified, so these values are Vercel's own and not secrets; this is
        # how an operator reads the real claims the first time a new project's token is refused.
        got: Any = None
        with contextlib.suppress(jwt.PyJWTError):
            got = jwt.decode(token, options={"verify_signature": False}).get(claim)
        # Both values are Vercel's, but a token's aud/sub is caller-influenced and unbounded, so
        # the log line is clipped and kept at info.
        log.info(
            "token_claim_mismatch",
            claim=claim,
            expected=str(expected)[:80],
            got=str(got)[:120],
        )


class HmacVerifier:
    """The fallback: a short-lived HS256 token, verified with `hmac.compare_digest`.

    A token is single use. Its `jti` is remembered until the token's own expiry, so a token copied
    out of a log or a shell history cannot be replayed against the API.
    """

    def __init__(self, secret: str, clock: Callable[[], float]) -> None:
        self._secret = secret.encode("utf-8")
        self._clock = clock
        self._lock = threading.Lock()
        self._seen: dict[str, float] = {}

    def verify(self, token: str) -> Principal | None:
        try:
            header = jwt.get_unverified_header(token)
            if header.get("alg") != "HS256":
                return None
            signing_input, _, signature = token.rpartition(".")
            if not signing_input or not signature:
                return None
            expected = hmac.new(
                self._secret, signing_input.encode("ascii"), hashlib.sha256
            ).digest()
            # The captain's rule for every shared secret: compare in constant time.
            if not hmac.compare_digest(expected, _b64decode(signature)):
                return None
            claims = jwt.decode(
                token,
                options={
                    "verify_signature": False,
                    "require": ["sub", "aud", "iat", "exp", "jti"],
                },
            )
        except (jwt.PyJWTError, ValueError, UnicodeEncodeError):
            return None
        now = self._clock()
        audience = claims.get("aud")
        iat, exp, jti, sub = (claims.get(name) for name in ("iat", "exp", "jti", "sub"))
        if audience != HMAC_AUDIENCE or not isinstance(jti, str) or not isinstance(sub, str):
            return None
        if not isinstance(iat, int | float) or not isinstance(exp, int | float):
            return None
        # A token minted here never lives longer than the TTL, whatever its issuer claims.
        if exp - iat > HMAC_TOKEN_TTL_S or exp <= now:
            return None
        # Refused before the jti is remembered, so a token only skewed into the future stays
        # usable once the clock catches up.
        if iat > now + TOKEN_LEEWAY_S:
            return None
        with self._lock:
            # Drop ids whose tokens have expired, so the set does not grow without bound.
            self._seen = {seen: until for seen, until in self._seen.items() if until > now}
            if jti in self._seen:
                return None
            self._seen[jti] = exp
        # One fixed subject, never the token's `sub`: that claim is chosen by whoever minted the
        # token, so honouring it would let one caller rotate the name with every request and dodge
        # the per-subject rate limit. The single-use jti is the only per-token identity there is.
        return Principal("hmac", "hmac")


def _b64decode(segment: str) -> bytes:
    # A JWT segment is base64url without padding; Python's decoder wants the padding.
    return base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4))


def mint_hmac_token(secret: str, subject: str, now: float) -> str:
    """Mint one short-lived token: the fallback, and what `paperwatch api-token` prints."""
    claims = {
        "sub": subject,
        "aud": HMAC_AUDIENCE,
        "iat": int(now),
        "exp": int(now) + HMAC_TOKEN_TTL_S,
        "jti": uuid.uuid4().hex,
    }
    return jwt.encode(claims, secret, algorithm="HS256")


def make_verifier(settings: Settings) -> Verifier:
    """Build the configured verifier, naming the missing variable instead of failing later."""
    if settings.api_auth == "hmac":
        secret = (
            settings.api_hmac_secret.get_secret_value().strip() if settings.api_hmac_secret else ""
        )
        if len(secret) < HMAC_SECRET_MIN_CHARS:
            raise ApiAuthNotConfigured(
                f"PW_API_HMAC_SECRET must be at least {HMAC_SECRET_MIN_CHARS} characters"
            )
        return HmacVerifier(secret, time.time)

    missing = [
        name
        for name, value in (
            ("PW_VERCEL_OIDC_ISSUER", settings.vercel_oidc_issuer),
            ("PW_VERCEL_OIDC_AUDIENCE", settings.vercel_oidc_audience),
            ("PW_VERCEL_OIDC_SUBJECT", settings.vercel_oidc_subject),
        )
        if not value
    ]
    if missing:
        raise ApiAuthNotConfigured(
            f"{missing[0]} is not set, so the API credential cannot be verified"
        )
    issuer = cast(str, settings.vercel_oidc_issuer).rstrip("/")

    def fetch(url: str) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            httpx.get(
                url, timeout=JWKS_FETCH_TIMEOUT_S, headers={"User-Agent": settings.user_agent}
            )
            .raise_for_status()
            .json(),
        )

    # One URL shape, so the configured issuer is the only thing that decides where keys come from.
    keys = JwksCache(f"{issuer}/.well-known/jwks", fetch, time.time)
    return OidcVerifier(
        issuer=issuer,
        audience=cast(str, settings.vercel_oidc_audience),
        subject=cast(str, settings.vercel_oidc_subject),
        keys=keys,
        clock=time.time,
    )
