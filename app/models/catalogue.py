"""Is every model of the chain still listed by OpenRouter?

Free models are withdrawn often, and most of the time without notice. This module asks
OpenRouter's public catalogue about each model id and reports what it says. It needs no API
key and spends none of the free-request allowance.

It only reports. Nothing here changes which models the service calls. With a chain named in
``OPENROUTER_MODELS``, a model the catalogue no longer lists is still tried whenever its turn
comes, so a withdrawn first model shows up on every request as a failed attempt. With the
setting left blank, :mod:`app.models.chooser` acts on what this module reports, and never on
its word alone: a model leaves once a real request has failed on it *and* the lookup here says
it is not listed. Two things follow from that choice. A wrong answer from the catalogue can
never switch off a model that works, and "listed" is not a promise that a model works: only a
real request through the service shows that.

The same check runs in three places:

* at startup, where :func:`app.models.chooser.at_startup` logs the result;
* on demand, with ``python -m app.models.catalogue``;
* on a schedule, as one step of the command ``.github/workflows/model-check.yml`` runs
  (:mod:`app.models.published`).

The module also lists the free models OpenRouter has on offer (:func:`free_models`), which is
where the service's own test of the models starts from.

As a command it exits 0 only when every model is listed and none has an end date within
``EXPIRY_WARNING_DAYS`` days, and 1 otherwise.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal
from urllib.parse import urlsplit

from app.models.cache import FREE_MODEL_ID, is_free_model

__all__ = [
    "FREE_MODEL_ID",
    "Fetch",
    "Listing",
    "check_chain",
    "describe",
    "free_models",
    "is_free_model",
    "is_openrouter",
    "look_up",
    "lookup_url",
    "main",
]

#: Time limit for each network step of a lookup (connecting, then each read), not for the
#: lookup as a whole. The catalogue normally answers in well under a second.
LOOKUP_TIMEOUT_SECONDS = 3.0
#: A real answer is a kilobyte or two. Reading stops here, so a server that misbehaves cannot
#: make the check hold megabytes in memory.
MAX_ANSWER_BYTES = 1_000_000
#: An announced end date this close is worth a warning.
EXPIRY_WARNING_DAYS = 14
#: The lookup route exists only on OpenRouter, so no other host is asked: not as the base URL,
#: and not by following a redirect.
OPENROUTER_HOST = "openrouter.ai"
#: Says who is asking. OpenRouter refuses the default signature of Python's ``urllib`` on some
#: of its paths, and a named client is easier to make out.
USER_AGENT = "multi-agent-ops-pipeline model-check"

Status = Literal["listed", "not_listed", "unknown"]

#: ``fetch(url)`` returns ``(http_status, body_text)``. Tests pass a fake.
Fetch = Callable[[str], tuple[int, str]]


@dataclass(frozen=True)
class Listing:
    """What the catalogue said about one model id."""

    model: str
    status: Status
    #: End date announced by OpenRouter (``YYYY-MM-DD``), when there is one.
    expires: str | None = None
    #: Why the status is what it is, in words, when that is not obvious.
    note: str = ""

    def days_left(self, today: date | None = None) -> int | None:
        if self.expires is None:
            return None
        return (date.fromisoformat(self.expires) - (today or date.today())).days

    def needs_attention(self, today: date | None = None) -> bool:
        """True for a model that is not listed, or whose announced end is near."""
        left = self.days_left(today)
        return self.status == "not_listed" or (left is not None and left <= EXPIRY_WARNING_DAYS)


def lookup_url(base_url: str, model: str) -> str:
    """The catalogue address for one model.

    The id goes in as it is, slash included: a percent-encoded slash or a doubled one makes
    OpenRouter answer "Not Found" for a model that is perfectly alive.
    """
    return base_url.rstrip("/") + "/model/" + model


def is_openrouter(base_url: str) -> bool:
    """True when ``base_url`` is OpenRouter's own encrypted address.

    The lookup route, the list of free models and the count of the day exist only there, and
    the one request that carries the key must go nowhere else.
    """
    try:
        address = urlsplit(base_url)
        return address.scheme == "https" and address.hostname == OPENROUTER_HOST
    except ValueError:  # not an address at all
        return False


class _SameHostRedirects(urllib.request.HTTPRedirectHandler):
    """Follow a redirect only while it stays on OpenRouter's own host."""

    def redirect_request(
        self, req: urllib.request.Request, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> urllib.request.Request | None:
        target = urlsplit(newurl)
        if target.scheme != "https" or target.hostname != OPENROUTER_HOST:
            # None makes urllib report the redirect itself as an HTTP error, which look_up
            # reads as "could not be checked".
            return None
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_SameHostRedirects)


def _http_get(url: str) -> tuple[int, str]:
    request = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": USER_AGENT}
    )
    try:
        with _OPENER.open(request, timeout=LOOKUP_TIMEOUT_SECONDS) as response:
            return response.status, response.read(MAX_ANSWER_BYTES).decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:  # 3xx, 4xx and 5xx still carry a body worth reading
        return exc.code, exc.read(MAX_ANSWER_BYTES).decode("utf-8", "replace")


def _iso_date(value: object) -> str | None:
    """The date part of ``value`` when it has one; anything else is ignored."""
    try:
        return date.fromisoformat(str(value)[:10]).isoformat()
    except ValueError:
        return None


def look_up(model: str, base_url: str, fetch: Fetch | None = None) -> Listing:
    """Ask the catalogue about one id. Never raises.

    ``fetch`` defaults to the module's own request, looked up when the call is made (not when
    this function was defined), so a test can replace :func:`_http_get` once for everything.
    """
    if model.startswith("@") or "/" not in model:
        # Presets and aliases are not catalogue entries; asking would prove nothing.
        return Listing(model, "unknown", note="not a catalogue id, so it cannot be looked up")
    try:
        status, body = (fetch or _http_get)(lookup_url(base_url, model))
    except Exception as exc:  # noqa: BLE001 - a lookup must never break its caller
        return Listing(model, "unknown", note=f"catalogue not reachable ({type(exc).__name__})")
    try:
        payload = json.loads(body)
    except (ValueError, RecursionError):  # not JSON, or nested absurdly deep
        payload = None
    payload = payload if isinstance(payload, dict) else {}

    error = payload.get("error")
    message = str(error.get("message", "")) if isinstance(error, dict) else ""
    # Only OpenRouter's own "Model not found: <id>" counts. A bare "Not Found" is a wrong
    # address, not a statement about the model, and anything else (a rate limit, an outage, a
    # login wall) says nothing about the model either.
    if status == 404 and message.startswith("Model not found"):
        return Listing(model, "not_listed", note="withdrawn, or the id is mistyped")

    data = payload.get("data")
    if status == 200 and isinstance(data, dict) and isinstance(data.get("id"), str):
        # A routing variant such as ":nitro" is answered with the base entry, so an end date
        # is reported only when the entry is exactly the configured id.
        expires = _iso_date(data.get("expiration_date")) if data["id"] == model else None
        return Listing(model, "listed", expires=expires)
    return Listing(model, "unknown", note=f"unexpected answer (HTTP {status})")


def check_chain(models: Sequence[str], base_url: str, fetch: Fetch | None = None) -> list[Listing]:
    """Look up every model at once and return the answers in chain order."""
    if not is_openrouter(base_url):
        return [
            Listing(model, "unknown", note="not checked: the base URL is not OpenRouter's")
            for model in models
        ]
    if not models:
        return []
    with ThreadPoolExecutor(max_workers=len(models)) as pool:
        return list(pool.map(lambda model: look_up(model, base_url, fetch), models))


#: The address of the candidates: free models that say they support structured output.
FREE_MODELS_QUERY = "/models?supported_parameters=structured_outputs&max_price=0"


def free_models(base_url: str, fetch: Fetch | None = None) -> list[str]:
    """The ids of the free models OpenRouter lists as offering structured output.

    Unlike :func:`look_up` this raises when the list cannot be read, because "no answer" and
    "no free model" must never be mistaken for each other. Being on the list is a claim, not
    proof: the service tests a model before it uses it.

    Nothing is left to the address. Each entry is checked again here: the id is a free id
    (:func:`is_free_model`), both prices read as zero, and the entry itself says it supports
    structured output.
    """
    if not is_openrouter(base_url):
        raise ValueError("the base URL is not OpenRouter's, so there is no list to ask for")
    status, body = (fetch or _http_get)(base_url.rstrip("/") + FREE_MODELS_QUERY)
    if status != 200:
        raise ValueError(f"unexpected answer (HTTP {status})")
    payload = json.loads(body)
    entries = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise ValueError("unexpected answer (no list of models)")

    def free(entry: object) -> bool:
        if not isinstance(entry, dict) or not is_free_model(entry.get("id")):
            return False
        pricing = entry.get("pricing")
        parameters = entry.get("supported_parameters")
        try:
            costs_nothing = isinstance(pricing, dict) and all(
                float(pricing[side]) == 0 for side in ("prompt", "completion")
            )
        except (KeyError, TypeError, ValueError):
            return False
        return costs_nothing and isinstance(parameters, list) and "structured_outputs" in parameters

    # Each id once, in the order listed: a model listed twice would be tested twice.
    return list(dict.fromkeys(entry["id"] for entry in entries if free(entry)))


def describe(listing: Listing, today: date | None = None) -> str:
    """One line a person can act on."""
    if listing.status == "not_listed":
        return (
            f"{listing.model} is not listed by OpenRouter ({listing.note}). "
            "Replace it in OPENROUTER_MODELS, or in DEFAULT_MODEL_CHAIN in app/config.py."
        )
    if listing.status == "unknown":
        return f"{listing.model} could not be checked ({listing.note})."
    left = listing.days_left(today)
    if left is None:
        return f"{listing.model} is listed."
    return (
        f"{listing.model} is listed, and OpenRouter has announced its end: "
        f"{listing.expires} ({left} days from today)."
    )


def main() -> int:
    """Check the chain in use and say what was found. Returns the exit code."""
    from app.config import DEFAULT_MODEL_CHAIN, get_settings, parse_model_chain
    from app.models import cache

    settings = get_settings()
    chain = settings.model_chain
    # With OPENROUTER_MODELS left blank, a service on this machine may have saved a chain of
    # its own choosing. That is then the chain in use, and nobody named it.
    saved = not settings.models_pinned and bool(cache.chain())
    if saved:
        print(
            f"Checking the chain a service on this machine saved in {cache.PATH.name}, "
            "not the built-in default."
        )
    elif chain != parse_model_chain(DEFAULT_MODEL_CHAIN):
        print("Checking OPENROUTER_MODELS from your environment, not the built-in default.")
    listings = check_chain(chain, settings.openrouter_base_url)
    for listing in listings:
        if saved and listing.status == "not_listed":
            print(
                f"{listing.model} is not listed by OpenRouter ({listing.note}). The service "
                "replaces it by itself after the next request that fails. To go back to the "
                f"built-in default, delete {cache.PATH.name} and restart."
            )
        else:
            print(describe(listing))
    # A watchdog must not pass on an answer it could not read, so "unknown" fails too.
    healthy = all(item.status == "listed" and not item.needs_attention() for item in listings)
    return 0 if healthy else 1


if __name__ == "__main__":  # pragma: no cover - command-line entry
    sys.exit(main())
