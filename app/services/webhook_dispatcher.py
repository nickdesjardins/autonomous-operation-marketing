"""The one external mutation: an HMAC-signed webhook, delivered with retries and audited.

Nothing leaves the system before a human approved it (the graph pauses before this node), and
when it leaves, the receiver can prove where it came from: the body is signed with HMAC-SHA256
over ``timestamp.delivery_id.body``, and the signature travels in ``X-Signature-256`` with the
timestamp in ``X-Timestamp`` and the delivery id in ``X-Delivery-Id``. The id is signed so that
a delivery cannot be replayed under a fresh id; a repeat under the same id is kept once by the
receiver. Transient failures (429, 5xx, timeouts, connection errors) are retried with backoff;
a 4xx is final. Every attempt is written to the ``webhook_audit_log`` table.

Three targets, chosen by ``WEBHOOK_TARGET``: an external URL, this service's own receiver (the
public demo, which never calls out), or a dry run that records the delivery and sends nothing
(signed when ``WEBHOOK_SECRET`` is set, marked ``unsigned`` otherwise).
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from typing import Any, Protocol
from urllib.parse import urlsplit

import httpx

from app.config import Settings
from app.schemas import utc_now_iso

logger = logging.getLogger(__name__)

SIGNATURE_HEADER = "X-Signature-256"
TIMESTAMP_HEADER = "X-Timestamp"
DELIVERY_HEADER = "X-Delivery-Id"
#: A signed timestamp older than this is refused by the receiver (replay protection).
MAX_SKEW_SECONDS = 300
RETRY_BACKOFF_SECONDS = (0.5, 1.0, 2.0, 4.0, 8.0)
#: How much of a signature the thread's state keeps: enough to match a receiver's record,
#: not the whole authenticator, since the state is read by whoever may read the thread.
SIGNATURE_FINGERPRINT_CHARS = 19  # "sha256=" and twelve hex digits


def signature_fingerprint(signature: str) -> str:
    """The start of ``signature``, as the state and the page show it; ``unsigned`` as it is."""
    if len(signature) <= SIGNATURE_FINGERPRINT_CHARS:
        return signature
    return signature[:SIGNATURE_FINGERPRINT_CHARS] + "..."


class WebhookAuditSink(Protocol):
    async def record_webhook_attempt(self, entry: dict[str, Any]) -> None: ...


@dataclass(slots=True)
class DeliveryResult:
    delivered: bool
    mode: str  # url | self | dry_run
    delivery_id: str
    url_host: str | None
    status_code: int | None
    attempts: int
    sent_at: str
    duration_ms: float
    signature: str
    error: str | None = None

    def as_state(self) -> dict[str, Any]:
        """What the thread keeps: the result with a fingerprint of the signature, not all of it."""
        return {**asdict(self), "signature": signature_fingerprint(self.signature)}


def sign(secret: str, timestamp: str, body: bytes, delivery_id: str = "") -> str:
    """``sha256=<hex>`` over ``timestamp + "." + delivery_id + "." + body``."""
    material = timestamp.encode("ascii") + b"." + delivery_id.encode("ascii") + b"." + body
    digest = hmac.new(secret.encode("utf-8"), material, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def verify(
    secret: str,
    timestamp: str,
    body: bytes,
    signature: str,
    *,
    delivery_id: str = "",
    now: float | None = None,
) -> bool:
    """Constant-time check of a delivery's signature, and of its age.

    The headers come from whoever sent the request: a value that is not plain ASCII digits
    (the timestamp) or plain ASCII (the signature, the delivery id) is refused here rather
    than left to raise while it is encoded.
    """
    timestamp, signature, delivery_id = timestamp or "", signature or "", delivery_id or ""
    if not (timestamp.isascii() and timestamp.isdigit()):
        return False
    if not (signature.isascii() and delivery_id.isascii()):
        return False
    sent = int(timestamp)
    current = time.time() if now is None else now
    if abs(current - sent) > MAX_SKEW_SECONDS:
        return False
    expected = sign(secret, timestamp, body, delivery_id)
    return hmac.compare_digest(expected.encode("ascii"), signature.encode("ascii"))


def build_payload(state: dict[str, Any]) -> dict[str, Any]:
    """The document a downstream system receives: the approved copy and its audit trail."""
    report = state.get("compliance_report") or {}
    return {
        "event": "listing.approved",
        "thread_id": state.get("thread_id"),
        "sku_code": state.get("sku_code"),
        "target_channels": list(state.get("target_channels") or []),
        "campaign_angle": state.get("campaign_angle"),
        "copy": state.get("draft_copy"),
        "compliance": {
            "is_compliant": report.get("is_compliant"),
            "risk_level": report.get("risk_level"),
            "revisions": state.get("revision_count", 0),
            "escalated": bool(state.get("requires_human_escalation")),
            "audits": len(state.get("audit_history") or []),
        },
        "review": {
            "status": state.get("human_review_status"),
            "reviewer_id": state.get("human_reviewer_id"),
            "notes": state.get("human_reviewer_notes"),
        },
        "inventory": {
            "product_name": (state.get("inventory_data") or {}).get("product_name"),
            "npn_number": (state.get("inventory_data") or {}).get("npn_number"),
            "stock_quantity": (state.get("inventory_data") or {}).get("stock_quantity"),
        },
    }


def encode_payload(payload: dict[str, Any]) -> bytes:
    """A canonical encoding, so the signature covers exactly the bytes that are sent."""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )


class WebhookDispatcher:
    """Signs, sends (or not), retries, and records."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: httpx.AsyncClient | None = None,
        audit: WebhookAuditSink | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.settings = settings
        self._client = client
        self._own_client = client is None
        self._audit = audit
        self._sleep = sleep
        self._clock = clock

    @property
    def mode(self) -> str:
        return self.settings.webhook_target

    @property
    def target_url(self) -> str | None:
        return self.settings.effective_webhook_url

    def describe(self) -> dict[str, Any]:
        url = self.target_url
        return {
            "target": self.mode,
            "configured": self.settings.webhook_configured,
            "url_host": urlsplit(url).hostname if url else None,
        }

    async def close(self) -> None:
        if self._own_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.settings.webhook_timeout_s)
        return self._client

    async def dispatch(self, payload: dict[str, Any]) -> DeliveryResult:
        """Deliver ``payload``. Never raises: the result says what happened."""
        body = encode_payload(payload)
        timestamp = str(int(self._clock()))
        secret = self.settings.secret("webhook_secret") or ""
        delivery_id = uuid.uuid4().hex
        signature = sign(secret, timestamp, body, delivery_id) if secret else "unsigned"
        thread_id = str(payload.get("thread_id") or "")
        started = time.perf_counter()
        url = self.target_url
        if self.mode == "dry_run" or url is None:
            result = DeliveryResult(
                delivered=False,
                mode="dry_run" if self.mode == "dry_run" else self.mode,
                delivery_id=delivery_id,
                url_host=None,
                status_code=None,
                attempts=0,
                sent_at=utc_now_iso(),
                duration_ms=0.0,
                signature=signature,
                error=None if self.mode == "dry_run" else "no target address configured",
            )
            await self._record(thread_id, result, attempt=0, ok=self.mode == "dry_run")
            return result

        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"{self.settings.service_name} webhook",
            SIGNATURE_HEADER: signature,
            TIMESTAMP_HEADER: timestamp,
            DELIVERY_HEADER: delivery_id,
        }
        host = urlsplit(url).hostname
        attempts = self.settings.webhook_max_attempts
        last_error: str | None = None
        status_code: int | None = None
        for attempt in range(1, attempts + 1):
            attempt_started = time.perf_counter()
            try:
                response = await self._http().post(url, content=body, headers=headers)
                status_code = response.status_code
                ok = 200 <= status_code < 300
                retryable = status_code == 429 or status_code >= 500
                last_error = None if ok else f"HTTP {status_code}"
            except httpx.TimeoutException:
                ok, retryable, last_error = False, True, "timeout"
            except httpx.HTTPError as exc:
                ok, retryable, last_error = False, True, f"connection error ({type(exc).__name__})"
            duration = (time.perf_counter() - attempt_started) * 1000.0
            await self._record_attempt(
                delivery_id, thread_id, attempt, host, status_code, ok, last_error, duration
            )
            if ok:
                return DeliveryResult(
                    delivered=True,
                    mode=self.mode,
                    delivery_id=delivery_id,
                    url_host=host,
                    status_code=status_code,
                    attempts=attempt,
                    sent_at=utc_now_iso(),
                    duration_ms=round((time.perf_counter() - started) * 1000.0, 1),
                    signature=signature,
                )
            logger.warning(
                "Webhook delivery %s attempt %d of %d failed: %s",
                delivery_id,
                attempt,
                attempts,
                last_error,
            )
            if not retryable or attempt == attempts:
                break
            await self._sleep(
                RETRY_BACKOFF_SECONDS[min(attempt - 1, len(RETRY_BACKOFF_SECONDS) - 1)]
            )
        return DeliveryResult(
            delivered=False,
            mode=self.mode,
            delivery_id=delivery_id,
            url_host=host,
            status_code=status_code,
            attempts=min(attempt, attempts),
            sent_at=utc_now_iso(),
            duration_ms=round((time.perf_counter() - started) * 1000.0, 1),
            signature=signature,
            error=last_error,
        )

    async def _record(
        self, thread_id: str, result: DeliveryResult, *, attempt: int, ok: bool
    ) -> None:
        await self._record_attempt(
            result.delivery_id,
            thread_id,
            attempt,
            result.url_host,
            result.status_code,
            ok,
            result.error,
            result.duration_ms,
        )

    async def _record_attempt(
        self,
        delivery_id: str,
        thread_id: str,
        attempt: int,
        host: str | None,
        status_code: int | None,
        ok: bool,
        error: str | None,
        duration_ms: float,
    ) -> None:
        if self._audit is None:
            return
        try:
            await self._audit.record_webhook_attempt(
                {
                    "delivery_id": delivery_id,
                    "thread_id": thread_id,
                    "attempt": attempt,
                    "target_mode": self.mode,
                    "url_host": host,
                    "status_code": status_code,
                    "ok": ok,
                    "error": error,
                    "duration_ms": round(duration_ms, 1),
                }
            )
        except Exception:  # noqa: BLE001 - the audit log must never block the delivery result
            logger.exception("Could not write the webhook audit log")
