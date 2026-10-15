"""The signed webhook: signatures, the canonical body, the payload, and the retry policy.

The dispatcher is driven over ``httpx.MockTransport`` with the backoff sleep replaced by a
recorder, so every retry path runs in milliseconds and the sleeps it would have taken are
asserted rather than waited for.
"""

from __future__ import annotations

import hmac
import json
from typing import Any

import httpx
import pytest

from app.config import Settings
from app.database import InMemoryWebhookStore
from app.services import webhook_dispatcher as module
from app.services.webhook_dispatcher import (
    DELIVERY_HEADER,
    MAX_SKEW_SECONDS,
    RETRY_BACKOFF_SECONDS,
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    WebhookDispatcher,
    build_payload,
    encode_payload,
    sign,
    verify,
)

from .conftest import good_draft, make_settings

SECRET = "a-long-random-string"
NOW = 1_760_000_000.0
BODY = b'{"event":"listing.approved"}'


# ------------------------------------------------------------------------ signatures


def test_sign_is_hmac_sha256_over_timestamp_dot_delivery_id_dot_body() -> None:
    signature = sign(SECRET, "1760000000", BODY, "d1")
    assert signature.startswith("sha256=") and len(signature) == 7 + 64
    assert signature != sign(SECRET, "1760000001", BODY, "d1")  # the timestamp is covered
    assert signature != sign(SECRET, "1760000000", BODY + b" ", "d1")  # every byte of the body
    assert signature != sign(SECRET, "1760000000", BODY, "d2")  # and the delivery id
    assert signature != sign("other", "1760000000", BODY, "d1")
    assert signature == sign(SECRET, "1760000000", BODY, "d1")


def test_verify_accepts_a_fresh_valid_signature_and_nothing_else() -> None:
    stamp = str(int(NOW))
    good = sign(SECRET, stamp, BODY, "d1")
    assert verify(SECRET, stamp, BODY, good, delivery_id="d1", now=NOW) is True
    assert verify(SECRET, stamp, BODY, good[:-1] + "0", delivery_id="d1", now=NOW) is False
    assert verify(SECRET, stamp, BODY, "", delivery_id="d1", now=NOW) is False
    assert verify(SECRET, stamp, BODY + b"x", good, delivery_id="d1", now=NOW) is False
    assert verify("other", stamp, BODY, good, delivery_id="d1", now=NOW) is False
    # A replay under a fresh delivery id is a forgery: the id is part of the signed material.
    assert verify(SECRET, stamp, BODY, good, delivery_id="d2", now=NOW) is False
    assert verify(SECRET, stamp, BODY, good, now=NOW) is False
    assert (
        verify(SECRET, "yesterday", BODY, sign(SECRET, "yesterday", BODY, "d1"), now=NOW) is False
    )


def test_verify_refuses_a_header_that_is_not_plain_ascii_instead_of_raising() -> None:
    stamp = str(int(NOW))
    good = sign(SECRET, stamp, BODY, "d1")
    # Starlette decodes header bytes as latin-1, so any byte above 0x7f reaches verify as text.
    assert verify(SECRET, stamp, BODY, "sha256=\xe9", delivery_id="d1", now=NOW) is False
    assert verify(SECRET, "\xa0" + stamp, BODY, good, delivery_id="d1", now=NOW) is False
    assert verify(SECRET, "１" + stamp[1:], BODY, good, delivery_id="d1", now=NOW) is False
    assert verify(SECRET, stamp, BODY, good, delivery_id="d\xe9", now=NOW) is False
    assert verify(SECRET, "1_760_000_000", BODY, good, delivery_id="d1", now=NOW) is False
    assert verify(SECRET, "", BODY, good, delivery_id="d1", now=NOW) is False


def test_verify_refuses_a_timestamp_outside_the_skew_window() -> None:
    stamp = str(int(NOW))
    signature = sign(SECRET, stamp, BODY)
    assert verify(SECRET, stamp, BODY, signature, now=NOW + MAX_SKEW_SECONDS) is True
    assert verify(SECRET, stamp, BODY, signature, now=NOW + MAX_SKEW_SECONDS + 1) is False
    assert verify(SECRET, stamp, BODY, signature, now=NOW - MAX_SKEW_SECONDS - 1) is False
    assert verify(SECRET, stamp, BODY, signature) is False  # the wall clock is far from NOW


def test_verify_compares_in_constant_time(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[bytes, bytes]] = []
    real = hmac.compare_digest

    def spy(a: bytes, b: bytes) -> bool:
        seen.append((a, b))
        return real(a, b)

    monkeypatch.setattr(module.hmac, "compare_digest", spy)
    stamp = str(int(NOW))
    verify(SECRET, stamp, BODY, "sha256=nope", now=NOW)
    assert len(seen) == 1 and seen[0][1] == b"sha256=nope"


# --------------------------------------------------------------------- the payload


def test_encode_payload_is_canonical() -> None:
    first = encode_payload({"b": 1, "a": {"y": [1, 2], "x": "é"}})
    second = encode_payload({"a": {"x": "é", "y": [1, 2]}, "b": 1})
    assert first == second == '{"a":{"x":"é","y":[1,2]},"b":1}'.encode()
    assert json.loads(first) == {"a": {"x": "é", "y": [1, 2]}, "b": 1}


def test_build_payload_shape() -> None:
    state = {
        "thread_id": "t-1",
        "sku_code": "NR-MAG-200",
        "target_channels": ["web_store", "social"],
        "campaign_angle": "Autumn restock",
        "draft_copy": good_draft().model_dump(),
        "compliance_report": {"is_compliant": True, "risk_level": "low", "violations": []},
        "revision_count": 1,
        "requires_human_escalation": False,
        "audit_history": [{"revision": 0}, {"revision": 1}],
        "human_review_status": "approved",
        "human_reviewer_id": "qa",
        "human_reviewer_notes": None,
        "inventory_data": {"product_name": "Mag", "npn_number": "80098765", "stock_quantity": 12},
    }
    payload = build_payload(state)
    assert payload == {
        "event": "listing.approved",
        "thread_id": "t-1",
        "sku_code": "NR-MAG-200",
        "target_channels": ["web_store", "social"],
        "campaign_angle": "Autumn restock",
        "copy": good_draft().model_dump(),
        "compliance": {
            "is_compliant": True,
            "risk_level": "low",
            "revisions": 1,
            "escalated": False,
            "audits": 2,
        },
        "review": {"status": "approved", "reviewer_id": "qa", "notes": None},
        "inventory": {"product_name": "Mag", "npn_number": "80098765", "stock_quantity": 12},
    }
    empty = build_payload({})
    assert empty["compliance"] == {
        "is_compliant": None,
        "risk_level": None,
        "revisions": 0,
        "escalated": False,
        "audits": 0,
    }
    assert empty["inventory"] == {"product_name": None, "npn_number": None, "stock_quantity": None}


# ------------------------------------------------------------------- the dispatcher


class Target:
    """A fake endpoint: answers in order, then repeats the last answer. Raising is allowed."""

    def __init__(self, *answers: int | Exception) -> None:
        self.answers = list(answers)
        self.requests: list[httpx.Request] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, Exception):
            raise answer
        return httpx.Response(answer, json={"ok": answer < 300}, request=request)


class Harness:
    def __init__(self, settings: Settings, target: Target) -> None:
        self.target = target
        self.sink = InMemoryWebhookStore()
        self.sleeps: list[float] = []

        async def sleep(seconds: float) -> None:
            self.sleeps.append(seconds)

        self.dispatcher = WebhookDispatcher(
            settings,
            client=httpx.AsyncClient(transport=httpx.MockTransport(target.handle)),
            audit=self.sink,
            sleep=sleep,
            clock=lambda: NOW,
        )


def _url_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "webhook_target": "url",
        "webhook_url": "https://hooks.example/in",
        "webhook_secret": SECRET,
        "webhook_max_attempts": 3,
    }
    values.update(overrides)
    return make_settings(**values)


PAYLOAD = {"event": "listing.approved", "thread_id": "thread-1", "copy": {"web_title": "T"}}


async def test_a_200_is_delivered_on_the_first_try_with_a_valid_signature() -> None:
    harness = Harness(_url_settings(), Target(200))
    result = await harness.dispatcher.dispatch(PAYLOAD)

    assert result.delivered is True and result.mode == "url"
    assert result.status_code == 200 and result.attempts == 1 and result.error is None
    assert result.url_host == "hooks.example"
    assert len(result.delivery_id) == 32 and result.duration_ms >= 0
    (request,) = harness.target.requests
    assert request.url == "https://hooks.example/in"
    assert request.content == encode_payload(PAYLOAD)
    assert request.headers["content-type"] == "application/json"
    assert request.headers[TIMESTAMP_HEADER] == str(int(NOW))
    assert request.headers[DELIVERY_HEADER] == result.delivery_id
    assert request.headers[SIGNATURE_HEADER] == result.signature
    assert verify(
        SECRET,
        request.headers[TIMESTAMP_HEADER],
        request.content,
        result.signature,
        delivery_id=result.delivery_id,
        now=NOW,
    )
    # The thread's state keeps a fingerprint of the signature, never the whole authenticator.
    assert result.as_state()["signature"] == result.signature[:19] + "..."
    assert result.as_state()["signature"].startswith("sha256=")
    assert request.headers["user-agent"].endswith(" webhook")
    assert harness.sleeps == []
    (entry,) = harness.sink.audit
    assert entry["ok"] is True and entry["attempt"] == 1 and entry["thread_id"] == "thread-1"
    assert entry["url_host"] == "hooks.example" and entry["target_mode"] == "url"


async def test_a_500_is_retried_after_the_first_backoff_and_then_succeeds() -> None:
    harness = Harness(_url_settings(), Target(500, 200))
    result = await harness.dispatcher.dispatch(PAYLOAD)
    assert result.delivered is True and result.attempts == 2 and result.status_code == 200
    assert harness.sleeps == [RETRY_BACKOFF_SECONDS[0]]
    assert len(harness.target.requests) == 2
    assert [row["ok"] for row in harness.sink.audit] == [False, True]
    assert (
        harness.sink.audit[0]["error"] == "HTTP 500" and harness.sink.audit[0]["status_code"] == 500
    )
    # Every attempt of one delivery carries the same signature and delivery id.
    first, second = harness.target.requests
    assert first.headers[SIGNATURE_HEADER] == second.headers[SIGNATURE_HEADER]
    assert first.headers[DELIVERY_HEADER] == second.headers[DELIVERY_HEADER]


async def test_a_429_is_retried_like_a_server_error() -> None:
    harness = Harness(_url_settings(), Target(429, 204))
    result = await harness.dispatcher.dispatch(PAYLOAD)
    assert result.delivered is True and result.attempts == 2 and result.status_code == 204


async def test_a_400_is_final() -> None:
    harness = Harness(_url_settings(), Target(400))
    result = await harness.dispatcher.dispatch(PAYLOAD)
    assert result.delivered is False and result.attempts == 1
    assert result.status_code == 400 and result.error == "HTTP 400"
    assert harness.sleeps == [] and len(harness.target.requests) == 1
    assert [row["ok"] for row in harness.sink.audit] == [False]


async def test_timeouts_are_retried_until_the_attempts_are_spent() -> None:
    settings = _url_settings()
    harness = Harness(settings, Target(httpx.ReadTimeout("slow")))
    result = await harness.dispatcher.dispatch(PAYLOAD)
    assert result.delivered is False
    assert result.attempts == settings.webhook_max_attempts == 3
    assert result.status_code is None and result.error == "timeout"
    assert harness.sleeps == [RETRY_BACKOFF_SECONDS[0], RETRY_BACKOFF_SECONDS[1]]
    assert len(harness.target.requests) == 3
    assert len(harness.sink.audit) == 3
    assert all(row["error"] == "timeout" and row["ok"] is False for row in harness.sink.audit)
    assert [row["attempt"] for row in harness.sink.audit] == [1, 2, 3]


async def test_connection_errors_are_retried_and_named_without_detail() -> None:
    harness = Harness(_url_settings(), Target(httpx.ConnectError("refused"), 200))
    result = await harness.dispatcher.dispatch(PAYLOAD)
    assert result.delivered is True and result.attempts == 2
    assert harness.sink.audit[0]["error"] == "connection error (ConnectError)"


async def test_the_backoff_never_runs_past_its_table() -> None:
    harness = Harness(_url_settings(webhook_max_attempts=8), Target(503))
    result = await harness.dispatcher.dispatch(PAYLOAD)
    assert result.attempts == 8 and result.delivered is False
    assert harness.sleeps == list(RETRY_BACKOFF_SECONDS) + [RETRY_BACKOFF_SECONDS[-1]] * 2


async def test_a_failing_audit_sink_never_breaks_the_delivery() -> None:
    class BrokenSink:
        async def record_webhook_attempt(self, entry: dict[str, Any]) -> None:
            raise OSError("database away")

    target = Target(200)
    dispatcher = WebhookDispatcher(
        _url_settings(),
        client=httpx.AsyncClient(transport=httpx.MockTransport(target.handle)),
        audit=BrokenSink(),
        clock=lambda: NOW,
    )
    result = await dispatcher.dispatch(PAYLOAD)
    assert result.delivered is True


async def test_dry_run_signs_and_records_but_sends_nothing() -> None:
    harness = Harness(make_settings(webhook_target="dry_run", webhook_secret=SECRET), Target(200))
    result = await harness.dispatcher.dispatch(PAYLOAD)
    assert result.mode == "dry_run" and result.delivered is False and result.error is None
    assert result.attempts == 0 and result.status_code is None and result.url_host is None
    assert result.signature == sign(
        SECRET, str(int(NOW)), encode_payload(PAYLOAD), result.delivery_id
    )
    assert harness.target.requests == [] and harness.sleeps == []
    (entry,) = harness.sink.audit
    assert entry["ok"] is True and entry["attempt"] == 0 and entry["target_mode"] == "dry_run"
    assert result.as_state()["mode"] == "dry_run"
    assert harness.dispatcher.describe() == {
        "target": "dry_run",
        "configured": True,
        "url_host": None,
    }


async def test_dry_run_without_a_secret_is_unsigned() -> None:
    harness = Harness(make_settings(), Target(200))
    result = await harness.dispatcher.dispatch(PAYLOAD)
    assert result.signature == "unsigned" and result.mode == "dry_run"
    assert result.as_state()["signature"] == "unsigned"


async def test_self_target_composes_the_receiver_url() -> None:
    settings = make_settings(
        webhook_target="self", webhook_secret=SECRET, public_base_url="https://demo.example/"
    )
    dispatcher = WebhookDispatcher(settings)
    assert dispatcher.mode == "self"
    assert dispatcher.target_url == "https://demo.example/api/v1/webhooks/receiver"
    assert dispatcher.describe() == {
        "target": "self",
        "configured": True,
        "url_host": "demo.example",
    }

    rendered = make_settings(
        webhook_target="self", webhook_secret=SECRET, render_external_url="https://x.onrender.com"
    )
    assert (
        WebhookDispatcher(rendered).target_url == "https://x.onrender.com/api/v1/webhooks/receiver"
    )

    preferred = make_settings(
        webhook_target="self",
        webhook_secret=SECRET,
        public_base_url="https://demo.example",
        render_external_url="https://x.onrender.com",
    )
    assert (
        WebhookDispatcher(preferred).target_url == "https://demo.example/api/v1/webhooks/receiver"
    )


async def test_self_target_without_an_address_is_recorded_as_undeliverable() -> None:
    harness = Harness(make_settings(webhook_target="self", webhook_secret=SECRET), Target(200))
    assert harness.dispatcher.describe()["configured"] is False
    result = await harness.dispatcher.dispatch(PAYLOAD)
    assert result.mode == "self" and result.delivered is False
    assert result.error == "no target address configured"
    assert harness.target.requests == []
    assert harness.sink.audit[0]["ok"] is False


async def test_close_releases_only_a_client_the_dispatcher_owns() -> None:
    target = Target(200)
    borrowed = httpx.AsyncClient(transport=httpx.MockTransport(target.handle))
    dispatcher = WebhookDispatcher(_url_settings(), client=borrowed, clock=lambda: NOW)
    await dispatcher.close()
    assert (await dispatcher.dispatch(PAYLOAD)).delivered is True  # the borrowed client lives on
    await borrowed.aclose()
    owned = WebhookDispatcher(make_settings())
    await owned.close()  # never opened a client: nothing to close, no error
