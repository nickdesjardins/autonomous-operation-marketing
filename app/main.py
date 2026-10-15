"""FastAPI application: start a run, inspect it, review it, receive the webhook, and the playground.

All collaborators (database or in-memory stand-ins, the structured chat client, the webhook
dispatcher, the compiled graph, the demo budget) are assembled into a ``Components`` bundle at
startup. Tests build the app with fakes instead, so the whole request pipeline runs without
PostgreSQL or any network call, over the very same compiled graph topology.

``uvicorn app.main:app`` works as usual: the module-level ``app`` is created lazily on first
access (PEP 562), so merely importing this module never reads the environment.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import secrets
import time
import uuid
from collections.abc import AsyncIterator, Coroutine
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app import __version__
from app.budget import BudgetExceeded, RunBudget
from app.compliance.rules import rule_summaries
from app.config import Settings, get_settings
from app.database import Database, InMemoryInventory, InMemoryWebhookStore
from app.errors import ConfigurationError, InventoryError, ProviderError
from app.graph.builder import NODE_WEBHOOK, build_graph, topology
from app.graph.nodes import PipelineNodes
from app.graph.prompts import AUDIT_SYSTEM_PROMPT, SYNTHESIS_SYSTEM_PROMPT
from app.graph.state import STATUS_REJECTED, initial_state
from app.graph.tools import InventoryTools
from app.schemas import (
    ALL_CHANNELS,
    AuditEntry,
    ChannelCopyDraft,
    DatabaseStatus,
    HealthResponse,
    HumanReviewRequest,
    HumanReviewResponse,
    LlmStatus,
    PipelineStartRequest,
    PipelineStatusResponse,
    PlaygroundConfig,
    RuleSummary,
    SkuSummary,
    TraceEvent,
    WebhookDelivery,
    WebhookStatus,
    utc_now_iso,
)
from app.seed import SAMPLE_ANGLES
from app.services.llm_client import ModelFailure, StructuredChat, build_chat
from app.services.webhook_dispatcher import (
    DELIVERY_HEADER,
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    WebhookDispatcher,
    verify,
)

logger = logging.getLogger(__name__)

PLAYGROUND_DIR = Path(__file__).resolve().parent / "playground" / "static"
_BODYLESS_METHODS = {"GET", "HEAD", "OPTIONS"}
#: Calls one run can make at most: a draft and an audit per attempt.
CALLS_PER_ATTEMPT = 2
#: The header that carries a run's thread id when the run itself failed and the response body
#: is an error, so the thread can be found again (``GET /api/v1/workflows/{thread_id}``).
THREAD_HEADER = "X-Thread-Id"


# ------------------------------------------------------------------------- components


@dataclass
class Components:
    settings: Settings
    db: Database | None
    inventory: Any
    webhook_store: Any
    chat: StructuredChat | None
    dispatcher: WebhookDispatcher
    checkpointer: Any
    graph: Any
    budget: RunBudget | None = None
    inflight: int = 0
    started_at: str = field(default_factory=utc_now_iso)
    #: One lock per thread under review, with a count of the requests holding or awaiting it,
    #: so that two reviews of one thread never both pass the "waiting for review" check.
    locks: dict[str, list[Any]] = field(default_factory=dict)

    @asynccontextmanager
    async def serialized(self, thread_id: str) -> AsyncIterator[None]:
        """Hold ``thread_id``'s lock; the entry is dropped once nobody holds or awaits it.

        One worker is the documented deployment (the budget and the chooser live in process
        memory too), so an in-process lock is enough.
        """
        entry = self.locks.get(thread_id)
        if entry is None:
            entry = self.locks[thread_id] = [asyncio.Lock(), 0]
        entry[1] += 1
        try:
            async with entry[0]:
                yield
        finally:
            entry[1] -= 1
            if entry[1] == 0:
                self.locks.pop(thread_id, None)

    async def close(self) -> None:
        await self.dispatcher.close()
        if self.db is not None:
            await self.db.close()


async def _open_database(db: Database, settings: Settings) -> None:
    """Open the pool and prepare the schema, retrying while a serverless database wakes up."""
    attempts = max(1, settings.db_startup_retries + 1)
    for attempt in range(1, attempts + 1):
        try:
            await db.open()
            if settings.auto_create_schema:
                await db.init_schema()
                await db.setup_checkpointer()
            else:
                await db.verify_checkpointer()
            return
        except ConfigurationError:
            raise
        except Exception as exc:  # noqa: BLE001 - every driver error is "not ready yet"
            if attempt == attempts:
                raise
            delay = min(2.0 * attempt, 10.0)
            logger.warning(
                "Database not ready (%s), attempt %d of %d; retrying in %.0f s",
                type(exc).__name__,
                attempt,
                attempts,
                delay,
            )
            await asyncio.sleep(delay)


def _failure_listener(settings: Settings):  # type: ignore[no-untyped-def]
    """The model chooser learns about failed requests only for the automatic OpenRouter chain."""
    if settings.llm_provider != "openrouter":
        return None
    from app.models import chooser

    def listener(failures: list[ModelFailure], answered: bool) -> None:
        chooser.note(failures, answered)

    return listener


async def build_components(settings: Settings) -> Components:
    """Wire the production collaborators. Raises ``ConfigurationError`` on fatal misconfiguration."""
    from langgraph.checkpoint.memory import InMemorySaver

    db: Database | None = None
    if settings.uses_database:
        db = Database(settings)
        await _open_database(db, settings)
        checkpointer = db.checkpointer()
        inventory: Any = db
        webhook_store: Any = db
    else:
        checkpointer = InMemorySaver()
        inventory = InMemoryInventory()
        webhook_store = InMemoryWebhookStore()
        logger.warning("CHECKPOINTER=memory: every thread is forgotten when the service stops")

    chat: StructuredChat | None = None
    if settings.llm_configured:
        chat = build_chat(settings, failure_listener=_failure_listener(settings))
    else:
        logger.warning(
            "No API key for LLM_PROVIDER=%s; runs will answer 503 until it is configured",
            settings.llm_provider,
        )
    dispatcher = WebhookDispatcher(settings, audit=webhook_store)
    nodes = PipelineNodes(
        settings, tools=InventoryTools(inventory), chat=chat, dispatcher=dispatcher
    )
    graph = build_graph(nodes, checkpointer)
    budget = RunBudget(settings) if settings.demo_mode else None
    if settings.demo_mode:
        logger.info(
            "Demo mode: %d runs/visitor/min, %d/visitor/day, %d/min, %d/day, %d in flight, "
            "max_revisions capped at %d, webhook target %s",
            settings.run_rate_limit_per_minute,
            settings.run_daily_limit_per_client,
            settings.run_global_limit_per_minute,
            settings.run_daily_limit,
            settings.run_max_inflight,
            settings.demo_max_revisions,
            settings.webhook_target,
        )
    return Components(
        settings=settings,
        db=db,
        inventory=inventory,
        webhook_store=webhook_store,
        chat=chat,
        dispatcher=dispatcher,
        checkpointer=checkpointer,
        graph=graph,
        budget=budget,
    )


def get_components(request: Request) -> Components:
    components = getattr(request.app.state, "components", None)
    if components is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Application is still starting")
    return components


# ---------------------------------------------------------------------- who is asking


def _normalise_ip(value: str) -> str | None:
    """A rate-limit key for an address: IPv4 as is, IPv6 by its /64 network; None if invalid."""
    value = value.strip()
    if not value:
        return None
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        host = value
        if value.startswith("["):
            host = value[1:].split("]", 1)[0]
        elif value.count(":") == 1:
            host = value.split(":", 1)[0]
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            return None
    if isinstance(address, ipaddress.IPv6Address):
        if address.ipv4_mapped is not None:
            return str(address.ipv4_mapped)
        return f"{ipaddress.ip_network(f'{address}/64', strict=False).network_address}/64"
    return str(address)


def client_id_for(request: Request, settings: Settings) -> str:
    """The visitor's address, as far as it can be trusted, for the per-visitor limits.

    The left-most ``X-Forwarded-For`` entry is whatever the client wrote, so it is never used.
    Behind a proxy the address comes from ``CLIENT_IP_HEADER`` (a header the edge overwrites),
    or else from the entry ``TRUSTED_PROXY_HOPS`` places from the right.
    """
    if settings.trust_proxy_headers:
        if settings.client_ip_header:
            candidate = _normalise_ip(request.headers.get(settings.client_ip_header, ""))
            if candidate:
                return candidate
        hops = [
            hop.strip()
            for value in request.headers.getlist("x-forwarded-for")
            for hop in value.split(",")
            if hop.strip()
        ]
        if len(hops) >= settings.trusted_proxy_hops:
            candidate = _normalise_ip(hops[-settings.trusted_proxy_hops])
            if candidate:
                return candidate
    host = request.client.host if request.client else ""
    return _normalise_ip(host) or host or "unknown"


# ------------------------------------------------------------------------------- auth


def _key_matches(provided: str, expected: str) -> bool:
    return bool(provided) and secrets.compare_digest(
        provided.encode("utf-8"), expected.encode("utf-8")
    )


async def require_workflow_access(
    request: Request, components: Components = Depends(get_components)
) -> None:
    """Outside demo mode an ``API_KEY``, when set, protects every workflow route.

    In demo mode the routes are public on purpose: the budget protects runs, and the webhook
    can only reach this service's own receiver.
    """
    settings = components.settings
    if settings.demo_mode or settings.api_key is None:
        return
    if not _key_matches(request.headers.get("x-api-key", ""), settings.api_key.get_secret_value()):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Invalid or missing X-API-Key header",
            headers={"WWW-Authenticate": "ApiKey"},
        )


# ----------------------------------------------------------------------- middleware


class _BodyTooLarge(HTTPException):
    def __init__(self, detail: str) -> None:
        super().__init__(413, detail)


class RequestGuardMiddleware:
    """Pure-ASGI body limit, applied before FastAPI reads a request body."""

    def __init__(self, app: ASGIApp, *, settings: Settings) -> None:
        self.app = app
        self.settings = settings

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method", "GET") in _BODYLESS_METHODS:
            await self.app(scope, receive, send)
            return
        limit = self.settings.max_json_body_bytes
        detail = f"Request body exceeds MAX_JSON_BODY_BYTES ({limit} bytes)"
        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers", [])
        }
        content_length = headers.get("content-length", "")
        if content_length.isdigit() and int(content_length) > limit:
            await _send_json(send, 413, {"detail": detail})
            return

        received = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise _BodyTooLarge(detail)
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _BodyTooLarge:
            if not response_started:
                await _send_json(send, 413, {"detail": detail})


async def _send_json(send: Send, status_code: int, payload: dict[str, object]) -> None:
    body = json.dumps(payload).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status_code,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


# ------------------------------------------------------------------------- helpers


def _config_for(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def _valid_thread_id(thread_id: str) -> str:
    try:
        return str(uuid.UUID(thread_id))
    except ValueError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown thread") from exc


def _stored_draft(draft: Any) -> ChannelCopyDraft | dict[str, Any] | None:
    """The persisted draft as the strict model, or as it is stored when it no longer fits.

    Checkpoints outlive releases: a draft written before a field was added or renamed must
    stay readable, so a thread can be inspected and rejected. The webhook node keeps its own
    strict check before anything leaves.
    """
    if not draft:
        return None
    try:
        return ChannelCopyDraft.model_validate(draft)
    except ValidationError:
        logger.warning("A stored draft no longer matches ChannelCopyDraft; returned as stored")
        return dict(draft)


def _status_response(thread_id: str, snapshot: Any) -> PipelineStatusResponse:
    values: dict[str, Any] = dict(snapshot.values or {})
    next_nodes = list(snapshot.next or ())
    draft = values.get("draft_copy")
    configurable = (getattr(snapshot, "config", None) or {}).get("configurable", {})
    return PipelineStatusResponse(
        thread_id=thread_id,
        status=values.get("status", "unknown"),
        sku_code=values.get("sku_code"),
        target_channels=list(values.get("target_channels") or []),
        campaign_angle=values.get("campaign_angle"),
        next_nodes=next_nodes,
        awaiting_human_review=NODE_WEBHOOK in next_nodes,
        revision_count=int(values.get("revision_count", 0)),
        max_revisions=int(values.get("max_revisions", 0)),
        requires_human_escalation=bool(values.get("requires_human_escalation")),
        human_review_status=values.get("human_review_status") or "pending",
        human_reviewer_id=values.get("human_reviewer_id"),
        human_reviewer_notes=values.get("human_reviewer_notes"),
        inventory_data=values.get("inventory_data"),
        draft_copy=_stored_draft(draft),
        compliance_report=values.get("compliance_report"),
        audit_history=[AuditEntry.model_validate(e) for e in values.get("audit_history") or []],
        revision_instructions=list(values.get("revision_instructions") or []),
        webhook_delivery_result=values.get("webhook_delivery_result"),
        trace=[TraceEvent.model_validate(e) for e in values.get("trace") or []],
        error=values.get("error"),
        created_at=values.get("created_at"),
        updated_at=values.get("updated_at"),
        checkpoint_id=configurable.get("checkpoint_id"),
    )


def _provider_http_error(exc: ProviderError, thread_id: str | None = None) -> HTTPException:
    """Map a provider failure to 503 (retryable) or 502 (permanent) without echoing its body.

    The incident id is the link between the client's error and the log line; the log line
    also names the thread, which stays at its last checkpoint, so an operator can find it.
    """
    code = status.HTTP_503_SERVICE_UNAVAILABLE if exc.retryable else status.HTTP_502_BAD_GATEWAY
    incident = uuid.uuid4().hex[:12]
    logger.error(
        "Provider failure incident=%s thread=%s provider=%s retryable=%s: %s",
        incident,
        thread_id,
        exc.provider,
        exc.retryable,
        exc.message,
    )
    kind = "temporary failure, retry later" if exc.retryable else "permanent failure"
    headers = {"Retry-After": "30"} if exc.retryable else {}
    if thread_id:
        headers[THREAD_HEADER] = thread_id
    return HTTPException(
        code,
        f"Upstream provider error ({exc.provider}): {kind}; incident {incident}",
        headers=headers or None,
    )


def _database_http_error(exc: BaseException, thread_id: str | None = None) -> HTTPException:
    incident = uuid.uuid4().hex[:12]
    logger.error(
        "Database failure incident=%s thread=%s: %s", incident, thread_id, type(exc).__name__
    )
    headers = {"Retry-After": "10"}
    if thread_id:
        headers[THREAD_HEADER] = thread_id
    return HTTPException(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        f"The database is unavailable; retry later. Incident {incident}",
        headers=headers,
    )


def _is_database_error(exc: BaseException) -> bool:
    """Whether ``exc`` is the PostgreSQL driver's, its pool's, or a dropped connection.

    The checkpointer is handed the raw pool and does not wrap what the driver raises, so the
    driver's exceptions are recognised by their package rather than imported: the memory
    stack has no driver at all.
    """
    if isinstance(exc, ConnectionError):
        return True
    return any(
        cls.__module__.split(".", 1)[0] in ("psycopg", "psycopg_pool") for cls in type(exc).__mro__
    )


async def _run_graph(
    components: Components, payload: Any, config: dict[str, Any], thread_id: str | None = None
) -> Any:
    try:
        return await components.graph.ainvoke(payload, config)
    except ProviderError as exc:
        raise _provider_http_error(exc, thread_id) from exc
    except InventoryError as exc:
        raise _database_http_error(exc, thread_id) from exc
    except Exception as exc:
        # The checkpointer writes between nodes: its driver's errors are the database's.
        if _is_database_error(exc):
            raise _database_http_error(exc, thread_id) from exc
        raise


async def _snapshot(components: Components, config: dict[str, Any]) -> Any:
    try:
        return await components.graph.aget_state(config)
    except Exception as exc:  # noqa: BLE001 - the checkpointer is the database
        raise _database_http_error(exc) from exc


def _llm_status(components: Components) -> LlmStatus:
    settings = components.settings
    if settings.llm_provider == "openrouter":
        from app.models import chooser

        mode = chooser.mode(settings)
    else:
        mode = "fixed"
    described = components.chat.describe() if components.chat else {}
    return LlmStatus(
        provider=settings.llm_provider,
        configured=components.chat is not None,
        models=list(described.get("models") or []),
        mode=mode,
    )


def _webhook_status(components: Components) -> WebhookStatus:
    return WebhookStatus(**components.dispatcher.describe())


# -------------------------------------------------------------------------- routes

router = APIRouter()


async def _bounded_probe(probe: Coroutine[object, object, Any], timeout_s: float) -> Any:
    try:
        return await asyncio.wait_for(probe, timeout=timeout_s)
    except TimeoutError:
        logger.warning("Health probe timed out after %.1fs", timeout_s)
        return None
    except Exception as exc:  # noqa: BLE001 - a probe must never take the endpoint down
        logger.warning("Health probe failed: %s", type(exc).__name__)
        return None


@router.get("/livez", tags=["ops"])
async def livez() -> dict[str, str]:
    """Liveness for the host's health check: never touches the database."""
    return {"status": "alive", "version": __version__}


@router.head("/livez", include_in_schema=False)
async def livez_head() -> dict[str, str]:
    """The same answer for a HEAD probe.

    A separate route rather than one with two methods: FastAPI gives a route one operation id,
    and listing it twice in the OpenAPI schema is reported as a duplicate. Kept out of the
    schema, HEAD costs nothing there.
    """
    return await livez()


@router.get("/health", response_model=HealthResponse, tags=["ops"])
async def health(components: Components = Depends(get_components)) -> JSONResponse:
    """Deep check for people: database, checkpointer, model provider, webhook target."""
    settings = components.settings
    if components.db is not None:
        connected = bool(await _bounded_probe(components.db.ping(), settings.health_timeout_s))
        skus = (
            await _bounded_probe(components.db.count_skus(), settings.health_timeout_s)
            if connected
            else None
        )
        database = DatabaseStatus(
            kind="postgres",
            connected=connected,
            checkpointer_ready=components.db.checkpointer_ready,
            inventory_skus=skus,
        )
    else:
        database = DatabaseStatus(
            kind="memory",
            connected=True,
            checkpointer_ready=True,
            inventory_skus=await components.inventory.count_skus(),
        )
    llm = _llm_status(components)
    webhook = _webhook_status(components)
    healthy = database.connected and llm.configured and webhook.configured
    body = HealthResponse(
        status="ok" if healthy else "degraded",
        version=__version__,
        service=settings.service_name,
        environment=settings.environment,
        demo_mode=settings.demo_mode,
        database=database,
        llm=llm,
        webhook=webhook,
    )
    code = status.HTTP_200_OK if database.connected else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(status_code=code, content=body.model_dump(mode="json"))


@router.get("/api/v1/skus", response_model=list[SkuSummary], tags=["inventory"])
async def list_skus(components: Components = Depends(get_components)) -> list[SkuSummary]:
    try:
        return await components.inventory.list_skus()
    except InventoryError as exc:
        raise _database_http_error(exc) from exc


@router.get("/api/v1/rules", response_model=list[RuleSummary], tags=["compliance"])
async def list_rules() -> list[RuleSummary]:
    return rule_summaries()


@router.post(
    "/api/v1/workflows/start",
    response_model=PipelineStatusResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["workflows"],
    dependencies=[Depends(require_workflow_access)],
)
async def start_workflow(
    payload: PipelineStartRequest,
    request: Request,
    components: Components = Depends(get_components),
) -> PipelineStatusResponse:
    settings = components.settings
    max_revisions = (
        payload.max_revisions if payload.max_revisions is not None else settings.max_revisions
    )
    if settings.demo_mode:
        max_revisions = min(max_revisions, settings.demo_max_revisions)
        if components.inflight >= settings.run_max_inflight:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "The demo is running other pipelines; please try again in a few seconds.",
                headers={"Retry-After": "5"},
            )
    if components.budget is not None:
        try:
            await components.budget.admit(
                client_id_for(request, settings), calls=CALLS_PER_ATTEMPT * (max_revisions + 1)
            )
        except BudgetExceeded as exc:
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                exc.message,
                headers={"Retry-After": str(exc.retry_after_s)},
            ) from exc
    if components.chat is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"LLM provider '{settings.llm_provider}' is not configured (missing API key)",
        )

    thread_id = str(uuid.uuid4())
    config = _config_for(thread_id)
    state = initial_state(
        thread_id=thread_id,
        sku_code=payload.sku_code,
        target_channels=list(payload.target_channels),
        campaign_angle=payload.campaign_angle,
        max_revisions=max_revisions,
        now=utc_now_iso(),
    )
    components.inflight += 1
    started = time.perf_counter()
    try:
        await _run_graph(components, state, config, thread_id)
    except HTTPException as exc:
        # The thread stays at its last checkpoint. Its id is in the log line and in the
        # response header; the failure is written into the state so a read of the thread
        # says what happened, and nothing here may hide the error that is being reported.
        logger.error("Run %s failed: %s", thread_id, exc.detail)
        try:
            await components.graph.aupdate_state(
                config, {"error": f"the run failed: {exc.detail}", "updated_at": utc_now_iso()}
            )
        except Exception:  # noqa: BLE001 - the checkpointer may be the very thing that failed
            logger.warning("Run %s: the failure could not be written into the thread", thread_id)
        raise
    finally:
        components.inflight -= 1
    snapshot = await _snapshot(components, config)
    response = _status_response(thread_id, snapshot)
    logger.info(
        "Run %s: sku=%s status=%s revisions=%d escalated=%s awaiting=%s in %.0f ms",
        thread_id,
        payload.sku_code,
        response.status,
        response.revision_count,
        response.requires_human_escalation,
        response.awaiting_human_review,
        (time.perf_counter() - started) * 1000.0,
    )
    return response


@router.get(
    "/api/v1/workflows/{thread_id}",
    response_model=PipelineStatusResponse,
    tags=["workflows"],
    dependencies=[Depends(require_workflow_access)],
)
async def get_workflow(
    thread_id: str, components: Components = Depends(get_components)
) -> PipelineStatusResponse:
    thread_id = _valid_thread_id(thread_id)
    snapshot = await _snapshot(components, _config_for(thread_id))
    if not snapshot.values:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown thread")
    return _status_response(thread_id, snapshot)


@router.post(
    "/api/v1/workflows/{thread_id}/review",
    response_model=HumanReviewResponse,
    tags=["workflows"],
    dependencies=[Depends(require_workflow_access)],
)
async def review_workflow(
    thread_id: str,
    payload: HumanReviewRequest,
    components: Components = Depends(get_components),
) -> HumanReviewResponse:
    """Human sign-off on a paused thread: approve, edit the copy, or reject.

    The thread's lock is held from the snapshot read through the resume, so two overlapping
    reviews of one thread cannot both pass the check below and run the webhook node twice:
    the second one reads the finished thread and gets the 409.
    """
    thread_id = _valid_thread_id(thread_id)
    async with components.serialized(thread_id):
        return await _review(thread_id, payload, components)


async def _review(
    thread_id: str, payload: HumanReviewRequest, components: Components
) -> HumanReviewResponse:
    config = _config_for(thread_id)
    snapshot = await _snapshot(components, config)
    if not snapshot.values:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown thread")
    if tuple(snapshot.next or ()) != (NODE_WEBHOOK,):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"This thread is not waiting for review (status {snapshot.values.get('status')!r})",
        )
    now = utc_now_iso()
    decision = {"approve": "approved", "edit": "edited", "reject": "rejected"}[payload.action]
    values: dict[str, Any] = {
        "human_review_status": decision,
        "human_reviewer_id": payload.reviewer_id,
        "human_reviewer_notes": payload.notes,
        "status": decision,
        "updated_at": now,
        "trace": [
            {
                "node": "human_review",
                "at": now,
                "ms": None,
                "detail": {"action": payload.action, "reviewer_id": payload.reviewer_id},
            }
        ],
    }
    if payload.action == "edit" and payload.override_copy is not None:
        values["draft_copy"] = payload.override_copy.model_dump()
    try:
        if payload.action == "reject":
            # Applied as if the webhook node had run: the thread ends here and the node never
            # executes. Nothing leaves the system.
            values["status"] = STATUS_REJECTED
            await components.graph.aupdate_state(config, values, as_node=NODE_WEBHOOK)
        else:
            await components.graph.aupdate_state(config, values)
            await _run_graph(components, None, config, thread_id)
    except HTTPException:
        raise
    except Exception as exc:
        # The state update goes to the checkpointer, which is the database. Anything else is
        # a fault of this service and is reported as one, with its own incident id.
        if _is_database_error(exc):
            raise _database_http_error(exc, thread_id) from exc
        raise
    snapshot = await _snapshot(components, config)
    response = _status_response(thread_id, snapshot)
    logger.info(
        "Review %s: action=%s reviewer=%s -> status=%s",
        thread_id,
        payload.action,
        payload.reviewer_id,
        response.status,
    )
    return HumanReviewResponse(action=payload.action, **response.model_dump())


@router.post("/api/v1/webhooks/receiver", tags=["webhooks"])
async def webhook_receiver(
    request: Request, components: Components = Depends(get_components)
) -> dict[str, Any]:
    """A downstream system, built in: verifies the signature and keeps what it received."""
    settings = components.settings
    secret = settings.secret("webhook_secret")
    if not secret:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "WEBHOOK_SECRET is not configured")
    body = await request.body()
    signature = request.headers.get(SIGNATURE_HEADER, "")
    timestamp = request.headers.get(TIMESTAMP_HEADER, "")
    # The delivery id is part of the signed material, so a delivery cannot be replayed under
    # a fresh id: the same id is a repeat, which the inbox keeps once.
    delivery_id = request.headers.get(DELIVERY_HEADER, "")
    valid = verify(secret, timestamp, body, signature, delivery_id=delivery_id)
    if not valid:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing webhook signature")
    try:
        payload = json.loads(body)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "The body is not JSON") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "The body must be a JSON object")
    delivery = WebhookDelivery(
        delivery_id=delivery_id[:64],
        thread_id=str(payload.get("thread_id") or "")[:64] or None,
        received_at=utc_now_iso(),
        signature_valid=True,
        payload=payload,
    )
    try:
        await components.webhook_store.store_inbox(delivery)
    except Exception as exc:  # noqa: BLE001
        raise _database_http_error(exc) from exc
    return {"received": True, "delivery_id": delivery.delivery_id}


@router.get("/api/v1/webhooks/inbox", response_model=list[WebhookDelivery], tags=["webhooks"])
async def webhook_inbox(
    limit: int = 20, components: Components = Depends(get_components)
) -> list[WebhookDelivery]:
    try:
        return await components.webhook_store.list_inbox(limit=max(1, min(limit, 100)))
    except Exception as exc:  # noqa: BLE001
        raise _database_http_error(exc) from exc


# -------------------------------------------------------------------------- playground

playground_router = APIRouter()

REVALIDATE = {"Cache-Control": "no-cache"}


class RevalidatingStaticFiles(StaticFiles):
    def file_response(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        response = super().file_response(*args, **kwargs)
        response.headers.update(REVALIDATE)
        return response


@playground_router.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
async def playground_index() -> FileResponse:
    return FileResponse(PLAYGROUND_DIR / "index.html", headers=REVALIDATE)


@playground_router.get("/playground/config", response_model=PlaygroundConfig, tags=["ops"])
async def playground_config(components: Components = Depends(get_components)) -> PlaygroundConfig:
    settings = components.settings
    try:
        skus = await components.inventory.list_skus()
    except InventoryError as exc:
        raise _database_http_error(exc) from exc
    limits: dict[str, int | None] = dict(components.budget.snapshot()) if components.budget else {}
    return PlaygroundConfig(
        demo_mode=settings.demo_mode,
        service=settings.service_name,
        skus=skus,
        channels=list(ALL_CHANNELS),
        sample_angles=list(SAMPLE_ANGLES),
        max_revisions=settings.demo_max_revisions if settings.demo_mode else settings.max_revisions,
        limits=limits,
        llm=_llm_status(components),
        webhook=_webhook_status(components),
        rules=rule_summaries(),
        synthesis_prompt=SYNTHESIS_SYSTEM_PROMPT,
        audit_prompt=AUDIT_SYSTEM_PROMPT,
        topology=topology(),
    )


@playground_router.get("/playground/models", tags=["ops"])
async def playground_models(components: Components = Depends(get_components)) -> dict[str, Any]:
    """The chain in use, where it came from, and the last test of the free models."""
    from app.models import chooser

    return chooser.picture(components.settings)


@playground_router.post("/playground/models/test", tags=["ops"])
async def playground_models_test(
    request: Request, components: Components = Depends(get_components)
) -> JSONResponse:
    """Start a test of the free models (visitors may press it; the budget says when).

    A press is admitted like a run, for the requests a test sends at most, so the per-visitor
    and the global limits and the allowance reserve bound what the button can spend; the
    chooser then adds its own pause and its daily cap on page-started tests. The ``Accept``
    check keeps a plain form post or a link from starting one; it is not a protection.
    """
    from app.models import cases, chooser, rules

    settings = components.settings
    if settings.is_production and "application/json" not in request.headers.get("accept", ""):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "This route serves the page only")
    if components.budget is not None:
        try:
            await components.budget.admit(
                client_id_for(request, settings), calls=rules.CANDIDATE_LIMIT * len(cases.CASES)
            )
        except BudgetExceeded as exc:
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                exc.message,
                headers={"Retry-After": str(exc.retry_after_s)},
            ) from exc
    code, again_at = await chooser.start("page", {"code": "asked", "models": []})
    body = {
        "started": code == "started",
        "reason": None if code == "started" else code,
        "again_at": again_at,
    }
    return JSONResponse(status_code=202 if code == "started" else 409, content=body)


# ------------------------------------------------------------------------- app factory


def _configure_logging(settings: Settings) -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
        )
    logging.getLogger("app").setLevel(settings.log_level)


def create_app(settings: Settings | None = None, components: Components | None = None) -> FastAPI:
    settings = settings or get_settings()
    _configure_logging(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned = False
        if getattr(app.state, "components", None) is None:
            app.state.components = await build_components(settings)
            owned = True
            logger.info("Startup complete: %s v%s", settings.service_name, __version__)
        if settings.llm_provider == "openrouter" and owned:
            from app.models import chooser

            chooser.log_mode(settings)
            await chooser.at_startup(settings)
        try:
            yield
        finally:
            if settings.llm_provider == "openrouter" and owned:
                from app.models import chooser

                await chooser.stop()
            if owned:
                await app.state.components.close()

    app = FastAPI(
        title="Autonomous Multi-Agent Operations and Marketing Pipeline",
        version=__version__,
        description=(
            "A LangGraph pipeline that drafts channel copy for a Natural Health Product, audits "
            "it against Health Canada NHP rules, loops on violations, pauses for a human, and "
            "fires a signed webhook. Every step is checkpointed in PostgreSQL."
        ),
        lifespan=lifespan,
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url="/redoc" if settings.docs_enabled else None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
    )
    app.state.settings = settings
    app.state.components = components
    app.include_router(router)
    if settings.playground_enabled:
        app.include_router(playground_router)
        app.mount(
            "/playground/assets",
            RevalidatingStaticFiles(directory=PLAYGROUND_DIR / "assets"),
            name="playground-assets",
        )
    app.add_middleware(RequestGuardMiddleware, settings=settings)
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "X-API-Key"],
            max_age=600,
        )

    @app.exception_handler(ConfigurationError)
    async def _configuration_error(_: Request, exc: ConfigurationError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(Exception)
    async def _unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        # Every error response carries an incident id that matches a log line, this service's
        # own faults included; Starlette's bare "Internal Server Error" would carry none.
        incident = uuid.uuid4().hex[:12]
        logger.exception("Unhandled error incident=%s: %s", incident, type(exc).__name__)
        return JSONResponse(
            status_code=500, content={"detail": f"Internal error; incident {incident}"}
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI's default echoes each rejected value back ("input"); never reflect payloads.
        errors = [
            {"type": error.get("type"), "loc": list(error.get("loc", ())), "msg": error.get("msg")}
            for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": errors})

    return app


_lazy_app: FastAPI | None = None


def __getattr__(name: str) -> object:
    """Expose ``app`` for ``uvicorn app.main:app`` without building it at import time."""
    global _lazy_app
    if name == "app":
        if _lazy_app is None:
            _lazy_app = create_app()
        return _lazy_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
