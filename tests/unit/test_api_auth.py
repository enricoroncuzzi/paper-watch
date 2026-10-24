"""Verifying the API credential: the Vercel OIDC token and the HMAC fallback.

Every key here is generated in the test and every fetch is a stub that counts its calls, so the
suite never touches Vercel, the network or a real secret.
"""

import base64
import hashlib
import hmac
import json
import threading
import time
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from paperwatch.api.auth import (
    ApiAuthNotConfigured,
    HmacVerifier,
    JwksCache,
    OidcVerifier,
    Principal,
    make_verifier,
    mint_hmac_token,
)
from paperwatch.config import HMAC_AUDIENCE, HMAC_TOKEN_TTL_S, TOKEN_LEEWAY_S, Settings

ISSUER = "https://paperwatch.vercel.app"
AUDIENCE = "https://paperwatch.example"
SUBJECT = "owner:paperwatch:project:prj_123:environment:production"
KID = "key-1"
NOW = 1_000_000.0
# One throwaway secret for every HMAC test: long enough to pass make_verifier, never a real key.
TEST_SECRET = "a-test-secret-that-is-long-enough-to-pass"  # noqa: S105


def _b64json(value: dict[str, Any]) -> str:
    raw = json.dumps(value, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _key() -> tuple[Any, str, dict[str, Any]]:
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = private.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    jwk = jwt.algorithms.RSAAlgorithm.to_jwk(private.public_key(), as_dict=True)
    jwk.update({"kid": KID, "alg": "RS256", "use": "sig"})
    return private, pem, jwk


def _claims(**overrides: Any) -> dict[str, Any]:
    claims: dict[str, Any] = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "sub": SUBJECT,
        "iat": int(NOW),
        "nbf": int(NOW),
        "exp": int(NOW) + 300,
    }
    claims.update(overrides)
    return claims


class Clock:
    def __init__(self, now: float = NOW) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class Fetcher:
    """A fake JWKS endpoint that counts calls and serves whatever keys it is given."""

    def __init__(self, keys: list[dict[str, Any]]) -> None:
        self.keys = keys
        self.calls = 0

    def __call__(self, url: str) -> dict[str, Any]:
        self.calls += 1
        return {"keys": self.keys}


def _verifier(
    private: Any, jwk: dict[str, Any], clock: Clock | None = None, fetcher: Fetcher | None = None
) -> OidcVerifier:
    clock = clock or Clock()
    fetcher = fetcher or Fetcher([jwk])
    return OidcVerifier(
        issuer=ISSUER,
        audience=AUDIENCE,
        subject=SUBJECT,
        keys=JwksCache("https://keys.example/jwks", fetcher, clock),
        clock=clock,
    )


def _token(private: Any, kid: str = KID, **overrides: Any) -> str:
    return jwt.encode(_claims(**overrides), private, algorithm="RS256", headers={"kid": kid})


def test_a_production_token_is_accepted() -> None:
    private, _, jwk = _key()
    assert _verifier(private, jwk).verify(_token(private)) == Principal("oidc", SUBJECT)


@pytest.mark.parametrize(
    "case",
    [
        "expired",
        "not_yet_valid",
        "preview",
        "wrong_audience",
        "wrong_issuer",
        "alg_none",
        "hmac_with_public_key",
        "other_key",
        "future_iat",
        "not_a_jwt",
    ],
)
def test_oidc_refuses(case: str) -> None:
    private, pem, jwk = _key()
    verifier = _verifier(private, jwk)
    if case == "expired":
        token = _token(private, exp=int(NOW) - TOKEN_LEEWAY_S - 1)
    elif case == "not_yet_valid":
        token = _token(private, nbf=int(NOW) + TOKEN_LEEWAY_S + 1)
    elif case == "preview":
        token = _token(private, sub="owner:paperwatch:project:prj_123:environment:preview")
    elif case == "wrong_audience":
        token = _token(private, aud="https://somewhere-else.example")
    elif case == "wrong_issuer":
        token = _token(private, iss="https://evil.example")
    elif case == "alg_none":
        # The classic downgrade: an unsigned token must never be read, whatever its claims say.
        token = jwt.encode(_claims(), key="", algorithm="none")
    elif case == "hmac_with_public_key":
        # The other classic: sign HS256 with the public key's PEM, hoping the verifier picks the
        # algorithm from the token. Only RS256 is accepted, and the key is not a shared secret.
        header = _b64json({"alg": "HS256", "kid": KID, "typ": "JWT"})
        payload = _b64json(_claims())
        signing_input = f"{header}.{payload}"
        signature = hmac.new(pem.encode(), signing_input.encode(), hashlib.sha256).digest()
        token = f"{signing_input}.{base64.urlsafe_b64encode(signature).rstrip(b'=').decode()}"
    elif case == "other_key":
        other, _, _ = _key()
        token = _token(other)
    elif case == "future_iat":
        token = _token(private, iat=int(NOW) + TOKEN_LEEWAY_S + 1)
    else:
        token = "not-a-jwt"  # noqa: S105 - not a token at all
    assert verifier.verify(token) is None


def test_leeway_accepts_thirty_seconds_of_clock_skew() -> None:
    private, _, jwk = _key()
    verifier = _verifier(private, jwk)
    assert verifier.verify(_token(private, exp=int(NOW) - TOKEN_LEEWAY_S + 1)) is not None


def test_unknown_kids_refetch_the_key_set_at_most_once_per_window() -> None:
    private, _, jwk = _key()
    clock = Clock()
    fetcher = Fetcher([jwk])
    verifier = _verifier(private, jwk, clock=clock, fetcher=fetcher)
    # The first request fetches the set.
    assert verifier.verify(_token(private)) is not None
    assert fetcher.calls == 1
    # A flood of unknown kids inside the window must not cause a fetch per request: that would be
    # a free amplification against Vercel's key endpoint and a way to stall the API.
    for index in range(50):
        assert verifier.verify(_token(private, kid=f"x{index}")) is None
    assert fetcher.calls == 1
    clock.now += 300
    # A fresh unknown key id after the window is what a rotation looks like, and it may refetch.
    assert verifier.verify(_token(private, kid="rotated")) is None
    assert fetcher.calls == 2


def test_a_key_rotation_is_picked_up() -> None:
    private, _, jwk = _key()
    rotated_private, _, rotated_jwk = _key()
    rotated_jwk["kid"] = "key-2"
    clock = Clock()
    fetcher = Fetcher([jwk])
    verifier = _verifier(private, jwk, clock=clock, fetcher=fetcher)
    assert verifier.verify(_token(private)) is not None
    fetcher.keys = [rotated_jwk]
    clock.now += 300
    rotated = jwt.encode(_claims(), rotated_private, algorithm="RS256", headers={"kid": "key-2"})
    assert verifier.verify(rotated) is not None


def test_a_failing_key_fetch_refuses_without_raising() -> None:
    private, _, _ = _key()

    def broken(url: str) -> dict[str, Any]:
        raise RuntimeError("the key endpoint is down")

    verifier = OidcVerifier(
        issuer=ISSUER,
        audience=AUDIENCE,
        subject=SUBJECT,
        keys=JwksCache("https://keys.example/jwks", broken, Clock()),
        clock=Clock(),
    )
    assert verifier.verify(_token(private)) is None


def _hmac_verifier(secret: str, clock: Clock) -> HmacVerifier:
    return HmacVerifier(secret, clock)


def test_hmac_token_round_trip_and_replay() -> None:
    secret = TEST_SECRET
    clock = Clock()
    verifier = _hmac_verifier(secret, clock)
    token = mint_hmac_token(secret, "operator", clock.now)
    assert verifier.verify(token) == Principal("hmac", "operator")
    # A token is single use: replaying it must not work, so a leaked log line cannot be reused.
    assert verifier.verify(token) is None
    long_lived = jwt.encode(
        {
            "sub": "operator",
            "aud": HMAC_AUDIENCE,
            "iat": int(clock.now),
            "exp": int(clock.now) + HMAC_TOKEN_TTL_S + 1,
            "jti": "long",
        },
        secret,
        algorithm="HS256",
    )
    assert verifier.verify(long_lived) is None
    other_audience = jwt.encode(
        {
            "sub": "operator",
            "aud": "somewhere-else",
            "iat": int(clock.now),
            "exp": int(clock.now) + HMAC_TOKEN_TTL_S,
            "jti": "aud",
        },
        secret,
        algorithm="HS256",
    )
    assert verifier.verify(other_audience) is None
    forged = jwt.encode(
        {
            "sub": "operator",
            "aud": HMAC_AUDIENCE,
            "iat": int(clock.now),
            "exp": int(clock.now) + HMAC_TOKEN_TTL_S,
            "jti": "forged",
        },
        "another-secret-that-is-long-enough",
        algorithm="HS256",
    )
    assert verifier.verify(forged) is None


def test_mint_hmac_token_is_short_lived() -> None:
    token = mint_hmac_token(TEST_SECRET, "operator", NOW)
    claims = jwt.decode(token, options={"verify_signature": False})
    assert claims["aud"] == HMAC_AUDIENCE
    assert claims["exp"] - claims["iat"] == HMAC_TOKEN_TTL_S


@pytest.mark.parametrize(
    "missing",
    [
        "PW_VERCEL_OIDC_ISSUER",
        "PW_VERCEL_OIDC_AUDIENCE",
        "PW_VERCEL_OIDC_SUBJECT",
        "PW_API_HMAC_SECRET",
        "PW_API_HMAC_SECRET_SHORT",
    ],
)
def test_make_verifier_names_the_missing_variable(missing: str, build_settings: Any) -> None:
    values: dict[str, Any] = {
        "vercel_oidc_issuer": ISSUER,
        "vercel_oidc_audience": AUDIENCE,
        "vercel_oidc_subject": SUBJECT,
    }
    if missing == "PW_VERCEL_OIDC_ISSUER":
        values["vercel_oidc_issuer"] = None
    elif missing == "PW_VERCEL_OIDC_AUDIENCE":
        values["vercel_oidc_audience"] = None
    elif missing == "PW_VERCEL_OIDC_SUBJECT":
        values["vercel_oidc_subject"] = None
    elif missing == "PW_API_HMAC_SECRET":
        values.update(api_auth="hmac", api_hmac_secret="")
    elif missing == "PW_API_HMAC_SECRET_SHORT":
        values.update(api_auth="hmac", api_hmac_secret="x" * 31)
    settings = build_settings(**values)
    with pytest.raises(ApiAuthNotConfigured) as raised:
        make_verifier(settings)
    name = missing.removesuffix("_SHORT")
    assert name in str(raised.value)


def test_the_oidc_verifier_is_built_without_a_fetch(build_settings: Any) -> None:
    # Building the verifier must not reach Vercel: the key set is fetched by the first request.
    settings = build_settings(
        vercel_oidc_issuer=ISSUER,
        vercel_oidc_audience=AUDIENCE,
        vercel_oidc_subject=SUBJECT,
    )
    assert isinstance(make_verifier(settings), OidcVerifier)


def test_a_missing_hmac_secret_is_not_a_fallback_for_oidc(build_settings: Any) -> None:
    # One credential at a time: an HMAC secret set while api_auth is oidc must not weaken the
    # OIDC path, and an empty secret under hmac must refuse rather than verify anything.
    settings = build_settings(
        vercel_oidc_issuer=ISSUER,
        vercel_oidc_audience=AUDIENCE,
        vercel_oidc_subject=SUBJECT,
        api_hmac_secret=TEST_SECRET,
    )
    assert isinstance(make_verifier(settings), OidcVerifier)
    with pytest.raises(ApiAuthNotConfigured):
        make_verifier(Settings(_env_file=None, api_auth="hmac", api_hmac_secret=""))


def test_time_is_read_from_the_injected_clock() -> None:
    # The HMAC verifier's clock is the only notion of "now" it has, so a test can place a token
    # on either side of its expiry without sleeping.
    secret = TEST_SECRET
    clock = Clock()
    verifier = HmacVerifier(secret, clock)
    token = mint_hmac_token(secret, "operator", clock.now)
    clock.now += HMAC_TOKEN_TTL_S
    assert verifier.verify(token) is None


def test_a_token_is_not_replayed_after_its_expiry_window_prunes() -> None:
    # The replay store forgets a jti once its token has expired; forgetting earlier would refuse
    # a fresh token that a test or an operator minted with the same id by chance.
    secret = TEST_SECRET
    clock = Clock()
    verifier = HmacVerifier(secret, clock)
    first = mint_hmac_token(secret, "operator", clock.now)
    assert verifier.verify(first) is not None
    clock.now += HMAC_TOKEN_TTL_S + 1
    later = mint_hmac_token(secret, "operator", clock.now)
    assert later != first
    assert verifier.verify(later) is not None


def test_time_is_not_wall_clock_in_the_round_trip() -> None:
    # Guard against an accidental `time.time()` default hiding a broken injected clock.
    secret = TEST_SECRET
    clock = Clock(now=time.time() + 10_000)
    verifier = HmacVerifier(secret, clock)
    assert verifier.verify(mint_hmac_token(secret, "operator", clock.now)) is not None


def test_a_slow_key_fetch_does_not_block_another_request() -> None:
    # N1: the fetch runs outside the cache lock. A second caller while the fetch is in flight is
    # refused quickly ("no key yet"), not queued behind a network round trip.
    _, _, jwk = _key()
    clock = Clock()
    started = threading.Event()
    release = threading.Event()

    def slow(url: str) -> dict[str, Any]:
        started.set()
        release.wait(timeout=5)
        return {"keys": [jwk]}

    cache = JwksCache("https://keys.example/jwks", slow, clock)
    worker = threading.Thread(target=cache.key, args=(KID,))
    worker.start()
    assert started.wait(1)
    began = time.monotonic()
    assert cache.key("another-kid") is None
    assert time.monotonic() - began < 0.5
    release.set()
    worker.join(timeout=5)


def test_a_failed_key_fetch_is_retried_sooner_than_a_rotation() -> None:
    # N2: a transient failure on the first request must not lock the page out for five minutes.
    _, _, jwk = _key()
    clock = Clock()
    calls = {"n": 0}

    def flaky(url: str) -> dict[str, Any]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("the key endpoint blinked")
        return {"keys": [jwk]}

    cache = JwksCache("https://keys.example/jwks", flaky, clock)
    assert cache.key(KID) is None
    clock.now += 31
    assert cache.key(KID) is not None
    assert calls["n"] == 2
