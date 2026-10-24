"""The gate's declared refusals, shared by every gated route.

The decision itself is `api/gate.py`; these models only describe its bodies, so the committed
contract and the generated page client know what a refusal looks like. They live here rather than
in `app.py` because the route modules `app.py` builds need them, and importing the app from a route
would be a cycle.
"""

from typing import Any, Literal

from fastapi.security import HTTPBearer
from pydantic import BaseModel

# The bearer scheme is documentation here: the gate verifies the token before FastAPI runs, and
# `auto_error=False` keeps this dependency from rejecting anything on its own.
BEARER = HTTPBearer(
    auto_error=False,
    scheme_name="VercelOidcOrHmac",
    bearerFormat="JWT",
    description="A Vercel OIDC token, or an HMAC token from `paperwatch api-token --live`.",
)


class Refusal(BaseModel):
    """The body of every gate refusal except 429, as `docs/learn/15` lists them."""

    detail: str


class RateLimited(BaseModel):
    """A 429: the per-client limit refused the request before the stream started."""

    detail: Literal["rate_limited", "busy"]
    retry_after_s: int


# Declared on every gated route so the generated client knows them. The real decision is
# `api/gate.py`; these models only describe its bodies.
GATE_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {
        "model": Refusal,
        "description": "The Host is not allowed, or the Content-Length is bad.",
    },
    401: {
        "model": Refusal,
        "description": "The credential is missing or invalid; `WWW-Authenticate: Bearer`.",
    },
    403: {"model": Refusal, "description": "The request carries an Origin header."},
}

ASK_GATE_RESPONSES: dict[int | str, dict[str, Any]] = {
    **GATE_RESPONSES,
    413: {"model": Refusal, "description": "The body is over 16 KB, by header or by stream."},
    415: {"model": Refusal, "description": "A POST without application/json."},
    429: {
        "model": RateLimited,
        "description": "The per-client limit refused the request; also sends Retry-After.",
    },
}
