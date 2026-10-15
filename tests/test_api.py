"""The FastAPI routes over the real compiled graph, with every collaborator a fake.

Requests go through ``httpx.ASGITransport``, so the middleware, the dependency that guards the
workflow routes, the error mapping and the response models are all exercised as deployed.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
import uuid
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from langgraph.checkpoint.memory import InMemorySaver

from app import __version__
from app.budget import RunBudget
from app.config import Settings
from app.database import Database
from app.errors import ProviderError
from app.graph.builder import NODE_WEBHOOK, build_graph
from app.graph.nodes import PipelineNodes
from app.graph.tools import InventoryTools
from app.main import THREAD_HEADER, Components, _open_database
from app.models import chooser
from app.services.webhook_dispatcher import (
    DELIVERY_HEADER,
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    WebhookDispatcher,
    encode_payload,
    sign,
)

from .conftest import (
    SKU,
    START_BODY,
    FakeChat,
    RecordingDispatcher,
    Stand,
    build_components,
    build_stand,
    client_for,
    compliant_audit,
    good_draft,
    make_settings,
    run_config,
)

REVIEW = {"action": "approve", "reviewer_id": "qa-lead", "notes": "Ship it."}


async def _start(http: AsyncClient, **changes: Any) -> dict[str, Any]:
    response = await http.post("/api/v1/workflows/start", json={**START_BODY, **changes})
    assert response.status_code == 201, response.text
    return response.json()


def _lenient_client(app: Any) -> AsyncClient:
    """A client that returns the app's 500 instead of re-raising the exception behind it."""
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    return AsyncClient(transport=transport, base_url="http://testserver")


def _with_saver(stand: Stand, saver: Any) -> None:
    """Recompile the stand's graph over another checkpointer."""
    components = stand.components
    nodes = PipelineNodes(
        stand.settings,
        tools=InventoryTools(components.inventory),
        chat=components.chat,
        dispatcher=components.dispatcher,
    )
    components.checkpointer = saver
    components.graph = build_graph(nodes, saver)


class FailingSaver(InMemorySaver):
    """An in-memory checkpointer whose writes fail once the run is under way."""

    def __init__(self, exc: BaseException, after: int = 2) -> None:
        super().__init__()
        self.exc = exc
        self.after = after
        self.puts = 0

    async def aput(self, *args: Any, **kwargs: Any) -> Any:
        self.puts += 1
        if self.puts > self.after:
            raise self.exc
        return await super().aput(*args, **kwargs)


# --------------------------------------------------------------------------- probes


async def test_livez_answers_without_touching_anything(client: AsyncClient) -> None:
    response = await client.get("/livez")
    assert response.status_code == 200
    assert response.json() == {"status": "alive", "version": __version__}
    assert (await client.head("/livez")).status_code == 200


async def test_health_reports_the_memory_stack(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["environment"] == "test" and body["demo_mode"] is False
    assert body["database"] == {
        "kind": "memory",
        "connected": True,
        "checkpointer_ready": True,
        "inventory_skus": 4,
    }
    assert body["llm"]["configured"] is True
    assert body["llm"]["provider"] == "openai_compatible" and body["llm"]["mode"] == "fixed"
    assert body["llm"]["models"] == ["fake/model:free"]
    assert body["webhook"] == {"target": "dry_run", "configured": True, "url_host": None}


async def test_health_is_degraded_without_a_model(settings: Settings) -> None:
    stand = build_stand(settings, chat=None)  # no API key: no chat client
    stand.components.chat = None
    async with client_for(stand.app) as http:
        body = (await http.get("/health")).json()
    assert body["status"] == "degraded" and body["llm"]["configured"] is False


async def test_the_inventory_and_the_rules_are_listed(client: AsyncClient) -> None:
    skus = (await client.get("/api/v1/skus")).json()
    assert [row["sku_code"] for row in skus] == [
        "NR-MAG-200",
        "NR-ORE-C80",
        "NR-PRO-50B",
        "NR-VITD-1000",
    ]
    assert {row["stock_quantity"] == 0 for row in skus} == {True, False}
    assert set(skus[0]) == {
        "sku_code",
        "product_name",
        "npn_number",
        "dosage_form",
        "stock_quantity",
    }
    rules = (await client.get("/api/v1/rules")).json()
    assert len(rules) == 6
    assert rules[0]["id"] == "NPN_DISPLAY" and rules[0]["severity"] == "high"


# ------------------------------------------------------------------------ the run


async def test_start_runs_to_the_gate(client: AsyncClient, stand: Stand) -> None:
    body = await _start(client)
    assert uuid.UUID(body["thread_id"])
    assert body["status"] == "awaiting_human_review"
    assert body["awaiting_human_review"] is True
    assert body["next_nodes"] == [NODE_WEBHOOK]
    assert body["sku_code"] == SKU and body["max_revisions"] == 2
    assert body["human_review_status"] == "pending"
    assert body["draft_copy"]["npn_displayed"] == "NPN 80098765"
    assert body["compliance_report"]["is_compliant"] is True
    assert len(body["audit_history"]) == 1 and body["revision_count"] == 0
    assert body["inventory_data"]["product_name"] == "Magnesium Bisglycinate 200 mg"
    assert [event["node"] for event in body["trace"]][-1] == "human_approval_gate"
    assert body["checkpoint_id"]
    assert body["webhook_delivery_result"] is None
    assert stand.components.inflight == 0


async def test_a_thread_can_be_read_back(client: AsyncClient) -> None:
    started = await _start(client)
    response = await client.get(f"/api/v1/workflows/{started['thread_id']}")
    assert response.status_code == 200
    body = response.json()
    assert body["thread_id"] == started["thread_id"]
    assert body["status"] == "awaiting_human_review" and body["next_nodes"] == [NODE_WEBHOOK]
    assert body["draft_copy"] == started["draft_copy"]


async def test_unknown_or_malformed_threads_are_404(client: AsyncClient) -> None:
    assert (await client.get(f"/api/v1/workflows/{uuid.uuid4()}")).status_code == 404
    assert (await client.get("/api/v1/workflows/not-a-thread")).status_code == 404
    assert (
        await client.post(f"/api/v1/workflows/{uuid.uuid4()}/review", json=REVIEW)
    ).status_code == 404
    assert (await client.post("/api/v1/workflows/nope/review", json=REVIEW)).status_code == 404


async def test_sku_code_is_upper_cased_and_trimmed(client: AsyncClient) -> None:
    body = await _start(client, sku_code="  nr-mag-200 ")
    assert body["sku_code"] == SKU


async def test_an_unknown_sku_ends_the_run_with_201_and_a_terminal_status(
    client: AsyncClient, stand: Stand
) -> None:
    body = await _start(client, sku_code="NR-NOPE-1")
    assert body["status"] == "sku_not_found" and body["next_nodes"] == []
    assert body["awaiting_human_review"] is False
    assert stand.chat.calls == []
    out = await _start(client, sku_code="NR-VITD-1000")
    assert out["status"] == "out_of_stock"


# ----------------------------------------------------------------------- the review


async def test_approve_dispatches_the_dry_run_webhook(client: AsyncClient, stand: Stand) -> None:
    started = await _start(client)
    response = await client.post(f"/api/v1/workflows/{started['thread_id']}/review", json=REVIEW)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["action"] == "approve"
    assert body["status"] == "dispatched"
    assert body["human_review_status"] == "approved"
    assert body["human_reviewer_id"] == "qa-lead" and body["human_reviewer_notes"] == "Ship it."
    assert body["awaiting_human_review"] is False and body["next_nodes"] == []
    assert body["webhook_delivery_result"]["mode"] == "dry_run"
    assert body["webhook_delivery_result"]["signature"] == "unsigned"
    nodes = [event["node"] for event in body["trace"]]
    assert nodes[-2:] == ["human_review", NODE_WEBHOOK]
    assert len(stand.components.dispatcher.payloads) == 1  # type: ignore[attr-defined]
    assert stand.components.webhook_store.audit[0]["target_mode"] == "dry_run"
    assert len(stand.chat.calls) == 2


async def test_edit_dispatches_the_override_copy(client: AsyncClient, stand: Stand) -> None:
    started = await _start(client)
    override = good_draft(web_title="Magnesium Bisglycinate 200 mg, edited by a human").model_dump()
    response = await client.post(
        f"/api/v1/workflows/{started['thread_id']}/review",
        json={"action": "edit", "reviewer_id": "editor", "override_copy": override},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "dispatched" and body["human_review_status"] == "edited"
    assert body["draft_copy"]["web_title"] == override["web_title"]
    (payload,) = stand.components.dispatcher.payloads  # type: ignore[attr-defined]
    assert payload["copy"]["web_title"] == override["web_title"]
    assert payload["review"]["status"] == "edited"


async def test_edit_needs_a_copy_and_the_copy_must_validate(client: AsyncClient) -> None:
    started = await _start(client)
    url = f"/api/v1/workflows/{started['thread_id']}/review"
    assert (await client.post(url, json={"action": "edit", "reviewer_id": "e"})).status_code == 422
    broken = {**good_draft().model_dump(), "marketplace_bullets": ["only one bullet"]}
    response = await client.post(
        url, json={"action": "edit", "reviewer_id": "e", "override_copy": broken}
    )
    assert response.status_code == 422
    with_copy = {**REVIEW, "override_copy": good_draft().model_dump()}
    assert (await client.post(url, json=with_copy)).status_code == 422  # approve takes no copy
    still = await client.get(f"/api/v1/workflows/{started['thread_id']}")
    assert still.json()["status"] == "awaiting_human_review"  # nothing moved


async def test_reject_ends_the_thread_and_a_second_review_is_409(
    client: AsyncClient, stand: Stand
) -> None:
    started = await _start(client)
    url = f"/api/v1/workflows/{started['thread_id']}/review"
    response = await client.post(url, json={"action": "reject", "reviewer_id": "qa-lead"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "rejected" and body["human_review_status"] == "rejected"
    assert body["next_nodes"] == [] and body["awaiting_human_review"] is False
    assert body["webhook_delivery_result"] is None
    assert stand.components.dispatcher.payloads == []  # type: ignore[attr-defined]

    again = await client.post(url, json=REVIEW)
    assert again.status_code == 409
    assert "rejected" in again.json()["detail"]
    assert (await client.get(f"/api/v1/workflows/{started['thread_id']}")).json()[
        "status"
    ] == "rejected"


async def test_a_finished_thread_cannot_be_reviewed_again(client: AsyncClient) -> None:
    started = await _start(client)
    url = f"/api/v1/workflows/{started['thread_id']}/review"
    assert (await client.post(url, json=REVIEW)).status_code == 200
    assert (await client.post(url, json=REVIEW)).status_code == 409


async def test_two_overlapping_approves_dispatch_the_webhook_once(settings: Settings) -> None:
    class SlowDispatcher(RecordingDispatcher):
        async def dispatch(self, payload: dict[str, Any]) -> Any:
            await asyncio.sleep(0.05)  # the HTTP round trip of a real delivery
            return await super().dispatch(payload)

    dispatcher = SlowDispatcher(settings)
    stand = build_stand(settings, dispatcher=dispatcher)
    async with client_for(stand.app) as http:
        started = await _start(http)
        url = f"/api/v1/workflows/{started['thread_id']}/review"
        first, second = await asyncio.gather(
            http.post(url, json=REVIEW), http.post(url, json=REVIEW)
        )
    assert sorted([first.status_code, second.status_code]) == [200, 409]
    assert len(dispatcher.payloads) == 1
    done = next(r for r in (first, second) if r.status_code == 200).json()
    assert [e["node"] for e in done["trace"]].count(NODE_WEBHOOK) == 1
    assert [e["node"] for e in done["trace"]].count("human_review") == 1
    assert stand.components.locks == {}  # nothing held once both requests are over


async def test_a_stored_draft_that_no_longer_validates_is_still_readable(
    client: AsyncClient, stand: Stand
) -> None:
    started = await _start(client)
    config = run_config(started["thread_id"])
    legacy = {**good_draft().model_dump(), "legacy_field": "written by an earlier release"}
    await stand.components.graph.aupdate_state(config, {"draft_copy": legacy})
    read = await client.get(f"/api/v1/workflows/{started['thread_id']}")
    assert read.status_code == 200, read.text
    assert read.json()["draft_copy"]["legacy_field"] == "written by an earlier release"
    rejected = await client.post(
        f"/api/v1/workflows/{started['thread_id']}/review",
        json={"action": "reject", "reviewer_id": "qa-lead"},
    )
    assert rejected.status_code == 200 and rejected.json()["status"] == "rejected"


# ---------------------------------------------------------------------- validation


async def test_validation_errors_never_echo_the_payload(client: AsyncClient) -> None:
    marker = "MARKER-" + uuid.uuid4().hex
    response = await client.post(
        "/api/v1/workflows/start",
        json={"sku_code": "bad sku!", "campaign_angle": marker, "extra": marker},
    )
    assert response.status_code == 422
    assert marker not in response.text
    errors = response.json()["detail"]
    assert errors and all(set(error) == {"type", "loc", "msg"} for error in errors)


@pytest.mark.parametrize(
    "changes",
    [
        {"sku_code": ""},
        {"campaign_angle": "ab"},
        {"campaign_angle": "x" * 301},
        {"campaign_angle": "line\x00break"},
        {"target_channels": []},
        {"target_channels": ["web_store", "web_store"]},
        {"target_channels": ["email"]},
        {"max_revisions": 6},
        {"max_revisions": -1},
    ],
)
async def test_malformed_start_requests_are_422(client: AsyncClient, changes: dict) -> None:
    response = await client.post("/api/v1/workflows/start", json={**START_BODY, **changes})
    assert response.status_code == 422, response.text


async def test_request_bodies_are_limited_by_content_length(settings: Settings) -> None:
    settings.max_json_body_bytes = 1024
    stand = build_stand(settings)
    async with client_for(stand.app) as http:
        big = await http.post(
            "/api/v1/workflows/start",
            content=json.dumps({**START_BODY, "campaign_angle": "a" * 4000}).encode(),
            headers={"Content-Type": "application/json"},
        )
        assert big.status_code == 413
        assert "MAX_JSON_BODY_BYTES" in big.json()["detail"]
        assert (await http.post("/api/v1/workflows/start", json=START_BODY)).status_code == 201
        assert (await http.get("/api/v1/skus")).status_code == 200  # GET is never limited
    assert stand.chat.calls and len(stand.chat.calls) == 2  # the big one never reached the graph


# ---------------------------------------------------------------------- demo mode


def _demo_settings(**overrides: Any) -> Settings:
    return make_settings(
        demo_mode=True,
        demo_max_revisions=1,
        run_rate_limit_per_minute=1,
        run_global_limit_per_minute=10,
        **overrides,
    )


async def test_demo_mode_budgets_each_visitor_and_caps_the_revisions() -> None:
    settings = _demo_settings()
    stand = build_stand(settings, budget=RunBudget(settings, allowance_reader=None))
    async with client_for(stand.app) as http:
        first = await _start(http, max_revisions=5)
        assert first["max_revisions"] == 1  # DEMO_MAX_REVISIONS wins over the request
        second = await http.post("/api/v1/workflows/start", json=START_BODY)
        assert second.status_code == 429
        assert (
            second.headers["retry-after"].isdigit()
            and 1 <= int(second.headers["retry-after"]) <= 60
        )
        assert "wait a minute" in second.json()["detail"]
        # Reading and reviewing cost nothing: the budget only guards starts.
        review = await http.post(f"/api/v1/workflows/{first['thread_id']}/review", json=REVIEW)
        assert review.status_code == 200
    assert stand.components.budget.snapshot()["used_today"] == 1  # type: ignore[union-attr]


async def test_demo_mode_sheds_load_beyond_the_inflight_limit() -> None:
    settings = _demo_settings(run_max_inflight=2)
    stand = build_stand(settings, budget=RunBudget(settings, allowance_reader=None))
    stand.components.inflight = 2
    async with client_for(stand.app) as http:
        busy = await http.post("/api/v1/workflows/start", json=START_BODY)
        assert busy.status_code == 503 and busy.headers["retry-after"] == "5"
        assert stand.components.budget.snapshot()["used_today"] == 0  # type: ignore[union-attr]
        stand.components.inflight = 0
        assert (await http.post("/api/v1/workflows/start", json=START_BODY)).status_code == 201
    assert stand.components.inflight == 0


async def test_demo_mode_opens_the_workflow_routes_without_the_api_key() -> None:
    settings = _demo_settings(api_key="operator")
    stand = build_stand(settings, budget=RunBudget(settings, allowance_reader=None))
    async with client_for(stand.app) as http:
        assert (await http.post("/api/v1/workflows/start", json=START_BODY)).status_code == 201


async def test_api_key_protects_the_workflow_routes_outside_demo_mode() -> None:
    stand = build_stand(make_settings(api_key="operator"))
    async with client_for(stand.app) as http:
        closed = await http.post("/api/v1/workflows/start", json=START_BODY)
        assert closed.status_code == 401
        assert closed.headers["www-authenticate"] == "ApiKey"
        wrong = await http.post(
            "/api/v1/workflows/start", json=START_BODY, headers={"X-API-Key": "nope"}
        )
        assert wrong.status_code == 401
        opened = await http.post(
            "/api/v1/workflows/start", json=START_BODY, headers={"X-API-Key": "operator"}
        )
        assert opened.status_code == 201
        thread = opened.json()["thread_id"]
        assert (await http.get(f"/api/v1/workflows/{thread}")).status_code == 401
        assert (await http.get("/api/v1/skus")).status_code == 200  # read-only, stays open
        assert (await http.get("/health")).status_code == 200
    assert len(stand.chat.calls) == 2  # the refused requests never reached a model


# ---------------------------------------------------------------- the webhook receiver


def _signed_headers(secret: str, body: bytes, *, timestamp: int | None = None) -> dict[str, str]:
    stamp = str(int(time.time()) if timestamp is None else timestamp)
    delivery_id = uuid.uuid4().hex
    return {
        "Content-Type": "application/json",
        SIGNATURE_HEADER: sign(secret, stamp, body, delivery_id),
        TIMESTAMP_HEADER: stamp,
        DELIVERY_HEADER: delivery_id,
    }


async def test_the_receiver_keeps_a_valid_delivery_and_the_inbox_lists_it() -> None:
    stand = build_stand(make_settings(webhook_secret="s3cret"))
    payload = {"event": "listing.approved", "thread_id": str(uuid.uuid4()), "copy": {"a": 1}}
    body = encode_payload(payload)
    headers = _signed_headers("s3cret", body)
    async with client_for(stand.app) as http:
        received = await http.post("/api/v1/webhooks/receiver", content=body, headers=headers)
        assert received.status_code == 200, received.text
        assert received.json() == {"received": True, "delivery_id": headers[DELIVERY_HEADER]}
        inbox = (await http.get("/api/v1/webhooks/inbox")).json()
    (entry,) = inbox
    assert entry["delivery_id"] == headers[DELIVERY_HEADER]
    assert entry["thread_id"] == payload["thread_id"]
    assert entry["signature_valid"] is True
    assert entry["payload"] == payload


async def test_the_receiver_refuses_a_bad_signature_a_stale_stamp_and_a_non_object() -> None:
    stand = build_stand(make_settings(webhook_secret="s3cret"))
    body = encode_payload({"thread_id": "t"})
    async with client_for(stand.app) as http:
        forged = _signed_headers("wrong-secret", body)
        assert (
            await http.post("/api/v1/webhooks/receiver", content=body, headers=forged)
        ).status_code == 401
        unsigned = await http.post("/api/v1/webhooks/receiver", content=body)
        assert unsigned.status_code == 401
        stale = _signed_headers("s3cret", body, timestamp=int(time.time()) - 3600)
        assert (
            await http.post("/api/v1/webhooks/receiver", content=body, headers=stale)
        ).status_code == 401
        array = b"[1, 2]"
        signed_array = _signed_headers("s3cret", array)
        assert (
            await http.post("/api/v1/webhooks/receiver", content=array, headers=signed_array)
        ).status_code == 400
        assert (await http.get("/api/v1/webhooks/inbox")).json() == []


async def test_the_receiver_is_503_without_a_secret(client: AsyncClient) -> None:
    response = await client.post("/api/v1/webhooks/receiver", content=b"{}")
    assert response.status_code == 503
    assert "WEBHOOK_SECRET" in response.json()["detail"]


async def test_the_receiver_answers_401_not_500_to_a_header_that_is_not_ascii() -> None:
    stand = build_stand(make_settings(webhook_secret="s3cret"))
    body = encode_payload({"thread_id": "t"})
    good = _signed_headers("s3cret", body)
    raw = {key.encode("latin-1"): value.encode("latin-1") for key, value in good.items()}
    async with client_for(stand.app) as http:
        bad_signature = {**raw, SIGNATURE_HEADER.encode(): b"sha256=\xe9"}
        response = await http.post("/api/v1/webhooks/receiver", content=body, headers=bad_signature)
        assert response.status_code == 401, response.text
        padded_stamp = {**raw, TIMESTAMP_HEADER.encode(): b"\xa0" + good[TIMESTAMP_HEADER].encode()}
        response = await http.post("/api/v1/webhooks/receiver", content=body, headers=padded_stamp)
        assert response.status_code == 401, response.text
        odd_id = {**raw, DELIVERY_HEADER.encode(): b"d\xe9"}
        response = await http.post("/api/v1/webhooks/receiver", content=body, headers=odd_id)
        assert response.status_code == 401, response.text
        assert (await http.get("/api/v1/webhooks/inbox")).json() == []


async def test_a_delivery_cannot_be_replayed_under_a_fresh_delivery_id() -> None:
    stand = build_stand(make_settings(webhook_secret="s3cret"))
    body = encode_payload({"event": "listing.approved", "thread_id": str(uuid.uuid4())})
    headers = _signed_headers("s3cret", body)
    async with client_for(stand.app) as http:
        first = await http.post("/api/v1/webhooks/receiver", content=body, headers=headers)
        assert first.status_code == 200, first.text
        # The same delivery again (a sender's retry) is kept once.
        again = await http.post("/api/v1/webhooks/receiver", content=body, headers=headers)
        assert again.status_code == 200
        # The same body, timestamp and signature under a new id is a forgery: the id is signed.
        replay = {**headers, DELIVERY_HEADER: uuid.uuid4().hex}
        refused = await http.post("/api/v1/webhooks/receiver", content=body, headers=replay)
        assert refused.status_code == 401
        without_id = {key: value for key, value in headers.items() if key != DELIVERY_HEADER}
        refused = await http.post("/api/v1/webhooks/receiver", content=body, headers=without_id)
        assert refused.status_code == 401
        inbox = (await http.get("/api/v1/webhooks/inbox")).json()
    assert [entry["delivery_id"] for entry in inbox] == [headers[DELIVERY_HEADER]]


async def test_the_self_target_delivers_to_the_receiver_end_to_end() -> None:
    settings = make_settings(
        webhook_target="self", webhook_secret="s3cret", public_base_url="http://testserver"
    )
    holder: dict[str, Any] = {}

    async def forward(scope: Any, receive: Any, send: Any) -> None:
        await holder["app"](scope, receive, send)

    from app.database import InMemoryWebhookStore

    store = InMemoryWebhookStore()
    dispatcher = WebhookDispatcher(settings, client=client_for(forward), audit=store)
    chat = FakeChat([good_draft()], [compliant_audit()], repeat_last=True)
    components = build_components(settings, chat, dispatcher=dispatcher, webhook_store=store)
    from app.main import create_app

    holder["app"] = create_app(settings=settings, components=components)

    async with client_for(holder["app"]) as http:
        health = (await http.get("/health")).json()
        assert health["webhook"] == {"target": "self", "configured": True, "url_host": "testserver"}
        started = await _start(http)
        review = await http.post(f"/api/v1/workflows/{started['thread_id']}/review", json=REVIEW)
        assert review.status_code == 200, review.text
        body = review.json()
        assert body["status"] == "dispatched"
        result = body["webhook_delivery_result"]
        assert result["mode"] == "self" and result["delivered"] is True
        assert result["status_code"] == 200 and result["attempts"] == 1
        assert result["url_host"] == "testserver"
        # A fingerprint of the signature, not the authenticator the receiver verified.
        assert result["signature"].startswith("sha256=") and result["signature"].endswith("...")
        assert len(result["signature"]) < 7 + 64
        inbox = (await http.get("/api/v1/webhooks/inbox")).json()
    (entry,) = inbox
    assert entry["signature_valid"] is True
    assert entry["thread_id"] == started["thread_id"]
    assert entry["delivery_id"] == result["delivery_id"]
    assert entry["payload"]["event"] == "listing.approved"
    assert entry["payload"]["copy"]["web_title"] == good_draft().web_title
    assert entry["payload"]["review"]["reviewer_id"] == "qa-lead"
    assert [row["ok"] for row in store.audit] == [True]
    await dispatcher.close()


# ------------------------------------------------------------------ provider failures


@pytest.mark.parametrize(
    "retryable, code, kind",
    [(True, 503, "temporary failure"), (False, 502, "permanent failure")],
)
async def test_provider_failures_map_to_503_or_502_without_upstream_text(
    settings: Settings, retryable: bool, code: int, kind: str
) -> None:
    secret_text = "UPSTREAM-BODY-" + uuid.uuid4().hex
    chat = FakeChat([ProviderError("fake/model:free", secret_text, retryable=retryable)])
    stand = build_stand(settings, chat=chat)
    async with client_for(stand.app) as http:
        response = await http.post("/api/v1/workflows/start", json=START_BODY)
    assert response.status_code == code
    detail = response.json()["detail"]
    assert "incident" in detail and kind in detail and "fake/model:free" in detail
    assert secret_text not in response.text
    assert (response.headers.get("retry-after") == "30") is retryable


async def test_a_failed_start_names_its_thread_and_the_thread_says_what_happened(
    settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    chat = FakeChat([ProviderError("fake/model:free", "rate limited (429)", retryable=True)])
    stand = build_stand(settings, chat=chat)
    async with client_for(stand.app) as http:
        with caplog.at_level("ERROR", logger="app.main"):
            response = await http.post("/api/v1/workflows/start", json=START_BODY)
        assert response.status_code == 503
        thread_id = response.headers[THREAD_HEADER]
        assert uuid.UUID(thread_id)
        incident = response.json()["detail"].rsplit("incident ", 1)[1]
        # The log links the incident to the thread, and the thread records the failure.
        assert any(
            incident in record.message and thread_id in record.message for record in caplog.records
        )
        read = (await http.get(f"/api/v1/workflows/{thread_id}")).json()
    assert read["status"] == "inventory_loaded" and read["next_nodes"] == ["copy_synthesis_agent"]
    assert read["error"] and incident in read["error"] and "the run failed" in read["error"]


async def test_a_checkpointer_failure_during_start_is_503_with_an_incident(
    settings: Settings,
) -> None:
    stand = build_stand(settings)
    _with_saver(stand, FailingSaver(ConnectionError("the connection was dropped")))
    async with client_for(stand.app) as http:
        response = await http.post("/api/v1/workflows/start", json=START_BODY)
    assert response.status_code == 503, response.text
    assert "database is unavailable" in response.json()["detail"]
    assert "Incident" in response.json()["detail"] and response.headers["retry-after"] == "10"
    assert stand.components.inflight == 0


async def test_a_fault_of_the_service_is_a_json_500_with_an_incident(settings: Settings) -> None:
    stand = build_stand(settings)
    _with_saver(stand, FailingSaver(RuntimeError("a bug, not the database")))
    async with _lenient_client(stand.app) as http:
        response = await http.post("/api/v1/workflows/start", json=START_BODY)
    assert response.status_code == 500
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["detail"].startswith("Internal error; incident ")
    assert "a bug" not in response.text
    assert stand.components.inflight == 0


async def test_a_draft_that_overflows_its_cap_after_typography_fixes_is_503(
    settings: Settings,
) -> None:
    at_the_cap = good_draft(web_title="M" * 157 + chr(0x2014) + "x")  # 159 characters
    stand = build_stand(settings, chat=FakeChat([at_the_cap], [compliant_audit()]))
    async with client_for(stand.app) as http:
        response = await http.post("/api/v1/workflows/start", json=START_BODY)
    assert response.status_code == 503, response.text
    assert "incident" in response.json()["detail"] and "fake/model:free" in response.text


async def test_a_missing_model_is_503_before_any_work(settings: Settings) -> None:
    stand = build_stand(settings, chat=None)
    stand.components.chat = None
    async with client_for(stand.app) as http:
        response = await http.post("/api/v1/workflows/start", json=START_BODY)
    assert response.status_code == 503
    assert "not configured" in response.json()["detail"]


async def test_a_failure_during_the_review_resume_is_reported_the_same_way(
    settings: Settings,
) -> None:
    class ExplodingDispatcher(WebhookDispatcher):
        async def dispatch(self, payload: dict[str, Any]) -> Any:
            raise ProviderError("hook", "boom", retryable=True)

    stand = build_stand(settings, dispatcher=ExplodingDispatcher(settings))
    async with client_for(stand.app) as http:
        started = await _start(http)
        response = await http.post(f"/api/v1/workflows/{started['thread_id']}/review", json=REVIEW)
        assert response.status_code == 503 and "incident" in response.json()["detail"]
        # The thread still waits at the gate: the review can be retried.
        again = (await http.get(f"/api/v1/workflows/{started['thread_id']}")).json()
        assert again["next_nodes"] == [NODE_WEBHOOK]


# ------------------------------------------------------------------- the playground


async def test_playground_config_has_the_shape_the_page_reads(
    client: AsyncClient, settings: Settings
) -> None:
    response = await client.get("/playground/config")
    assert response.status_code == 200
    config = response.json()
    assert set(config) == {
        "demo_mode",
        "service",
        "skus",
        "channels",
        "sample_angles",
        "max_revisions",
        "limits",
        "llm",
        "webhook",
        "rules",
        "synthesis_prompt",
        "audit_prompt",
        "topology",
    }
    assert config["demo_mode"] is False and config["service"] == settings.service_name
    assert len(config["skus"]) == 4 and len(config["rules"]) == 6
    assert config["channels"] == ["web_store", "marketplace", "social"]
    assert len(config["sample_angles"]) >= 3
    assert config["max_revisions"] == settings.max_revisions
    assert config["limits"] == {}  # no budget outside demo mode
    assert config["llm"]["mode"] == "fixed" and config["webhook"]["target"] == "dry_run"
    assert config["synthesis_prompt"].startswith("You are the Copy Synthesis Agent")
    assert config["audit_prompt"].startswith("You are the Compliance Validator Agent")
    assert [node["id"] for node in config["topology"]["nodes"]] == [
        "supervisor",
        "inventory_sql_agent",
        "copy_synthesis_agent",
        "compliance_validator",
        "human_approval_gate",
        "webhook_dispatch",
    ]
    assert config["topology"]["interrupt_before"] == [NODE_WEBHOOK]


async def test_playground_config_shows_the_demo_limits() -> None:
    settings = _demo_settings()
    stand = build_stand(settings, budget=RunBudget(settings, allowance_reader=None))
    async with client_for(stand.app) as http:
        config = (await http.get("/playground/config")).json()
    assert config["demo_mode"] is True and config["max_revisions"] == 1
    assert config["limits"]["per_client_per_minute"] == 1
    assert config["limits"]["used_today"] == 0


async def test_the_model_test_button_is_admitted_through_the_demo_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pressed: list[str] = []

    async def fake_start(by: str, reason: dict[str, Any]) -> tuple[str, str | None]:
        pressed.append(by)
        return "started", None

    monkeypatch.setattr(chooser, "start", fake_start)
    settings = _demo_settings()  # one run a minute per visitor
    stand = build_stand(settings, budget=RunBudget(settings, allowance_reader=None))
    async with client_for(stand.app) as http:
        first = await http.post("/playground/models/test", headers={"Accept": "application/json"})
        assert first.status_code == 202 and first.json()["started"] is True
        second = await http.post("/playground/models/test", headers={"Accept": "application/json"})
        assert second.status_code == 429 and second.headers["retry-after"].isdigit()
    assert pressed == ["page"]
    assert stand.components.budget.snapshot()["used_today"] == 1  # type: ignore[union-attr]


async def test_the_model_test_button_wants_the_pages_accept_header_in_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pressed: list[str] = []

    async def fake_start(by: str, reason: dict[str, Any]) -> tuple[str, str | None]:
        pressed.append(by)
        return "pause", "2026-10-15T12:00:00Z"

    monkeypatch.setattr(chooser, "start", fake_start)
    stand = build_stand(make_settings(environment="production"))
    async with client_for(stand.app) as http:
        assert (await http.post("/playground/models/test")).status_code == 403
        refused = await http.post("/playground/models/test", headers={"Accept": "application/json"})
        assert refused.status_code == 409 and refused.json()["reason"] == "pause"
    assert pressed == ["page"]


async def test_playground_can_be_switched_off() -> None:
    stand = build_stand(make_settings(playground_enabled=False))
    async with client_for(stand.app) as http:
        assert (await http.get("/playground/config")).status_code == 404
        assert (await http.get("/livez")).status_code == 200


async def test_docs_are_hidden_in_production() -> None:
    open_docs = build_stand(make_settings(environment="development"))
    async with client_for(open_docs.app) as http:
        assert (await http.get("/openapi.json")).status_code == 200
    closed = build_stand(make_settings(environment="production"))
    async with client_for(closed.app) as http:
        assert (await http.get("/openapi.json")).status_code == 404
        assert (await http.get("/docs")).status_code == 404


async def test_cors_headers_only_for_configured_origins(settings: Settings) -> None:
    plain = build_stand(settings)
    async with client_for(plain.app) as http:
        response = await http.get("/health", headers={"Origin": "https://nick.github.io"})
        assert "access-control-allow-origin" not in response.headers

    configured = build_stand(
        make_settings(cors_allow_origins="https://nick.github.io, https://example.org")
    )
    async with client_for(configured.app) as http:
        allowed = await http.get("/health", headers={"Origin": "https://nick.github.io"})
        assert allowed.headers["access-control-allow-origin"] == "https://nick.github.io"
        preflight = await http.options(
            "/api/v1/workflows/start",
            headers={
                "Origin": "https://example.org",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type, x-api-key",
            },
        )
        assert preflight.status_code == 200
        assert preflight.headers["access-control-allow-origin"] == "https://example.org"
        denied = await http.get("/health", headers={"Origin": "https://evil.example"})
        assert "access-control-allow-origin" not in denied.headers


async def test_the_components_bundle_closes_its_dispatcher(components: Components) -> None:
    await components.close()  # no database, an idle dispatcher: nothing to release, no error


# ------------------------------------------------------------------- the database start


class FakeDatabase:
    """Records which start-up steps ``_open_database`` takes."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.checkpointer_ready = False

    async def open(self) -> None:
        self.calls.append("open")

    async def init_schema(self) -> None:
        self.calls.append("init_schema")

    async def setup_checkpointer(self) -> None:
        self.calls.append("setup_checkpointer")
        self.checkpointer_ready = True

    async def verify_checkpointer(self) -> None:
        self.calls.append("verify_checkpointer")
        self.checkpointer_ready = True


async def test_the_database_start_creates_or_only_verifies_the_schema() -> None:
    created = FakeDatabase()
    await _open_database(created, make_settings(checkpointer="postgres"))  # type: ignore[arg-type]
    assert created.calls == ["open", "init_schema", "setup_checkpointer"]
    verified = FakeDatabase()
    await _open_database(
        verified,  # type: ignore[arg-type]
        make_settings(checkpointer="postgres", auto_create_schema=False),
    )
    assert verified.calls == ["open", "verify_checkpointer"] and verified.checkpointer_ready
    # The real class offers the same steps with the same nature: a probe is awaited, the
    # saver accessor is not.
    assert inspect.iscoroutinefunction(Database.verify_checkpointer)
    assert inspect.iscoroutinefunction(Database.setup_checkpointer)
    assert not inspect.iscoroutinefunction(Database.checkpointer)
