"""Live PostgreSQL integration tests (opt-in): the checkpointer really persists a paused thread.

Skipped unless ``TEST_DATABASE_URL`` is set, for example::

    TEST_DATABASE_URL=postgresql://ops:ops@localhost:5432/ops pytest tests/test_integration_db.py

or, without Docker, through the embedded server: ``python scripts/run_live_db_tests.py``.

The tests work inside their own schema (``ops_test``), created and dropped here, so they never
touch the tables a service uses. They open the real ``Database`` (pool, DDL, seed, checkpointer
setup), read the inventory, round-trip the webhook audit log and the receiver's inbox, and run
the compiled graph with a small fake model over the real ``AsyncPostgresSaver``: a run pauses
before the webhook node, a second graph over a second ``Database`` (as another process would
build after a restart) finds the paused thread in PostgreSQL, approves it and resumes it to a
dispatch. The fake model is defined here on purpose, so the file stands alone.
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from collections.abc import AsyncIterator, Sequence
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pytest

from app.config import Settings
from app.database import Database
from app.graph.builder import NODE_ORDER, NODE_WEBHOOK, build_graph
from app.graph.nodes import PipelineNodes
from app.graph.state import (
    STATUS_AWAITING_REVIEW,
    STATUS_DISPATCHED,
    STATUS_REJECTED,
    initial_state,
)
from app.graph.tools import InventoryTools
from app.schemas import ChannelCopyDraft, ComplianceAuditResult, WebhookDelivery, utc_now_iso
from app.services.llm_client import Attempt, ChatOutcome
from app.services.webhook_dispatcher import WebhookDispatcher

DATABASE_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="TEST_DATABASE_URL not set")

if DATABASE_URL and sys.platform == "win32":
    # psycopg's async connections cannot run on Windows' default proactor loop.
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

SCHEMA = "ops_test"
SKU = "NR-MAG-200"
NPN = "80098765"


def scoped_url(url: str) -> str:
    """The same server, with the test schema first on the search path (a libpq option)."""
    parts = urlsplit(url)
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k != "options"]
    query.append(("options", f"-csearch_path={SCHEMA}"))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def make_settings() -> Settings:
    return Settings(
        _env_file=None,
        _env_prefix="HERMETIC_TEST_",  # ignore ambient variables entirely
        environment="test",
        checkpointer="postgres",
        database_url=scoped_url(DATABASE_URL or ""),
        webhook_target="dry_run",
        webhook_secret="integration-test-secret",
        max_revisions=2,
        db_pool_min_size=1,
        db_pool_max_size=3,
        log_level="WARNING",
    )


# ------------------------------------------------------------------------- fake model


class FakeChat:
    """A model that writes one compliant draft for NR-MAG-200 and audits anything as compliant."""

    name = "fake"
    model = "fake/structured-model"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "models": [self.model]}

    async def complete(
        self,
        *,
        messages: Sequence[dict[str, str]],
        schema_model: type[Any],
        temperature: float | None,
        max_tokens: int,
        schema_name: str = "output",
    ) -> ChatOutcome:
        self.calls.append(schema_name)
        if schema_model is ChannelCopyDraft:
            parsed: Any = ChannelCopyDraft(
                web_title="Magnesium Bisglycinate 200 mg: calm evenings, steady routines",
                web_description_html=(
                    "<p>Magnesium Bisglycinate 200 mg helps to maintain proper muscle function, "
                    "including the heart muscle, and helps in energy metabolism and in tissue "
                    f"formation. One vegetable capsule, 1 to 2 times daily. NPN {NPN}.</p>"
                ),
                marketplace_bullets=[
                    "Helps to maintain proper muscle function, including the heart muscle.",
                    "Helps in the development and maintenance of bones and teeth.",
                    "Helps to prevent magnesium deficiency.",
                    "One vegetable capsule, 1 to 2 times daily, 200 mg of magnesium each.",
                    "Vegan, gluten free, non-GMO; chelated form chosen for tolerability.",
                ],
                social_post_copy=(
                    "Calm evenings start with steady habits. Magnesium Bisglycinate 200 mg "
                    f"helps in energy metabolism and in tissue formation. NPN {NPN}. #calm #routine"
                ),
                image_generation_prompt=(
                    "A matte white bottle on a wooden bedside table at dusk, soft lamp light, "
                    "a folded linen cloth beside it. Calm, warm, unhurried."
                ),
                disclaimers_included=[
                    "Consult a health care practitioner prior to use if you are pregnant or "
                    "breastfeeding.",
                    "Some people may experience diarrhoea.",
                ],
                npn_displayed=NPN,
            )
        else:
            parsed = ComplianceAuditResult(
                is_compliant=True,
                risk_level="low",
                violations=[],
                revision_summary="The draft keeps to the authorized claims and shows the NPN.",
            )
        return ChatOutcome(
            self.model,
            parsed.model_dump_json(),
            "stop",
            parsed=parsed,
            attempts=[Attempt(self.model, "answered", 0.01)],
        )


# -------------------------------------------------------------------------- fixtures


async def _run_sql(url: str, *statements: str) -> None:
    import psycopg

    async with await psycopg.AsyncConnection.connect(url, autocommit=True) as conn:
        for statement in statements:
            await conn.execute(statement)  # type: ignore[arg-type]


@pytest.fixture
async def schema() -> AsyncIterator[str]:
    assert DATABASE_URL
    await _run_sql(
        DATABASE_URL, f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE", f"CREATE SCHEMA {SCHEMA}"
    )
    try:
        yield SCHEMA
    finally:
        await _run_sql(DATABASE_URL, f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")


@pytest.fixture
def settings(schema: str) -> Settings:
    return make_settings()


@pytest.fixture
async def db(settings: Settings) -> AsyncIterator[Database]:
    database = Database(settings)
    await database.open()
    await database.init_schema()
    await database.setup_checkpointer()
    try:
        yield database
    finally:
        await database.close()


def _graph(settings: Settings, database: Database, chat: FakeChat):  # type: ignore[no-untyped-def]
    dispatcher = WebhookDispatcher(settings, audit=database)
    nodes = PipelineNodes(
        settings, tools=InventoryTools(database), chat=chat, dispatcher=dispatcher
    )
    return build_graph(nodes, database.checkpointer())


def _config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def _start(thread_id: str, *, max_revisions: int = 2) -> dict[str, Any]:
    return dict(
        initial_state(
            thread_id=thread_id,
            sku_code=SKU,
            target_channels=["web_store", "marketplace", "social"],
            campaign_angle="Autumn restock: calm evenings and better sleep routines.",
            max_revisions=max_revisions,
            now=utc_now_iso(),
        )
    )


# ----------------------------------------------------------------------------- tests


async def test_schema_seed_and_checkpointer_are_ready(db: Database, settings: Settings) -> None:
    assert await db.ping() is True
    assert db.checkpointer_ready is True
    assert await db.count_skus() == 4
    codes = [summary.sku_code for summary in await db.list_skus()]
    assert codes == sorted(codes) and SKU in codes and "NR-VITD-1000" in codes
    record = await db.get_sku(SKU)
    assert record is not None and record.npn_number == NPN and record.stock_quantity > 0
    assert [i.amount for i in record.active_ingredients] == [200]
    assert await db.get_sku("NR-NOPE-1") is None
    # The DDL, the seed and the checkpointer migrations are idempotent.
    await db.init_schema()
    await db.setup_checkpointer()
    assert await db.count_skus() == 4
    # Every table, the saver's included, was created in the test schema (the search path).
    import psycopg

    async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
        cursor = await conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = %s", (SCHEMA,)
        )
        rows = await cursor.fetchall()
    tables = {row[0] for row in rows}
    assert {"inventory_skus", "webhook_audit_log", "webhook_inbox"} <= tables
    assert {
        "checkpoints",
        "checkpoint_blobs",
        "checkpoint_writes",
        "checkpoint_migrations",
    } <= tables


async def test_webhook_audit_log_and_inbox_round_trip(db: Database, settings: Settings) -> None:
    delivery_id = uuid.uuid4().hex
    thread_id = str(uuid.uuid4())
    await db.record_webhook_attempt(
        {
            "delivery_id": delivery_id,
            "thread_id": thread_id,
            "attempt": 1,
            "target_mode": "self",
            "url_host": "record",
            "status_code": 200,
            "ok": True,
            "error": None,
            "duration_ms": 12.5,
        }
    )
    delivery = WebhookDelivery(
        delivery_id=delivery_id,
        thread_id=thread_id,
        received_at=utc_now_iso(),
        signature_valid=True,
        payload={"event": "listing.approved", "thread_id": thread_id, "copy": {"web_title": "x"}},
    )
    await db.store_inbox(delivery)
    await db.store_inbox(delivery)  # the same delivery id twice is stored once
    inbox = await db.list_inbox(limit=10)
    assert [d.delivery_id for d in inbox] == [delivery_id]
    assert inbox[0].thread_id == thread_id and inbox[0].signature_valid is True
    assert inbox[0].payload["copy"] == {"web_title": "x"}
    assert inbox[0].received_at.endswith("Z")
    import psycopg

    async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
        cursor = await conn.execute(
            "SELECT attempt, ok, status_code FROM webhook_audit_log WHERE delivery_id = %s",
            (delivery_id,),
        )
        rows = await cursor.fetchall()
    assert rows == [(1, True, 200)]


async def test_graph_pauses_in_postgres_and_another_process_resumes_it(
    db: Database, settings: Settings
) -> None:
    chat = FakeChat()
    graph = _graph(settings, db, chat)
    thread_id = str(uuid.uuid4())
    config = _config(thread_id)

    await graph.ainvoke(_start(thread_id), config)
    paused = await graph.aget_state(config)
    assert tuple(paused.next) == (NODE_WEBHOOK,)
    assert paused.values["status"] == STATUS_AWAITING_REVIEW
    assert paused.values["human_review_status"] == "pending"
    assert chat.calls == ["channel_copy_draft", "compliance_audit_result"]
    assert [e["revision"] for e in paused.values["audit_history"]] == [0]
    assert paused.values["audit_history"][0]["is_compliant"] is True
    assert paused.values["webhook_delivery_result"] is None
    assert [e["node"] for e in paused.values["trace"]] == list(NODE_ORDER[:-1])

    # Another process: a new pool, a new saver, a new compiled graph, a model that is never asked.
    other_db = Database(settings)
    await other_db.open()
    try:
        other_chat = FakeChat()
        other_graph = _graph(settings, other_db, other_chat)
        found = await other_graph.aget_state(config)
        assert tuple(found.next) == (NODE_WEBHOOK,)
        assert found.values["draft_copy"] == paused.values["draft_copy"]
        assert found.values["thread_id"] == thread_id
        assert (
            found.config["configurable"]["checkpoint_id"]
            == paused.config["configurable"]["checkpoint_id"]
        )

        now = utc_now_iso()
        await other_graph.aupdate_state(
            config,
            {
                "human_review_status": "approved",
                "human_reviewer_id": "qa-lead",
                "human_reviewer_notes": "Ship it.",
                "status": "approved",
                "updated_at": now,
                "trace": [
                    {
                        "node": "human_review",
                        "at": now,
                        "ms": None,
                        "detail": {"action": "approve", "reviewer_id": "qa-lead"},
                    }
                ],
            },
        )
        await other_graph.ainvoke(None, config)
        final = await other_graph.aget_state(config)
    finally:
        await other_db.close()

    assert tuple(final.next) == ()
    assert final.values["status"] == STATUS_DISPATCHED
    assert other_chat.calls == []  # the resume ran only the webhook node
    result = final.values["webhook_delivery_result"]
    assert result["mode"] == "dry_run" and result["attempts"] == 0
    assert result["delivered"] is False and result["error"] is None
    assert result["signature"].startswith("sha256=")  # a dry run still signs what it would send
    nodes = [e["node"] for e in final.values["trace"]]
    assert nodes == [*NODE_ORDER[:-1], "human_review", NODE_WEBHOOK]
    assert final.values["human_reviewer_id"] == "qa-lead"

    # The first graph reads the same checkpoint: the state lives in PostgreSQL, not in a process.
    again = await graph.aget_state(config)
    assert again.values["status"] == STATUS_DISPATCHED
    assert tuple(again.next) == ()


async def test_rejection_ends_the_thread_without_running_the_webhook_node(
    db: Database, settings: Settings
) -> None:
    chat = FakeChat()
    graph = _graph(settings, db, chat)
    thread_id = str(uuid.uuid4())
    config = _config(thread_id)
    await graph.ainvoke(_start(thread_id), config)
    assert tuple((await graph.aget_state(config)).next) == (NODE_WEBHOOK,)

    now = utc_now_iso()
    await graph.aupdate_state(
        config,
        {
            "human_review_status": "rejected",
            "human_reviewer_id": "qa-lead",
            "status": STATUS_REJECTED,
            "updated_at": now,
        },
        as_node=NODE_WEBHOOK,
    )
    final = await graph.aget_state(config)
    assert tuple(final.next) == ()
    assert final.values["status"] == STATUS_REJECTED
    assert final.values["webhook_delivery_result"] is None
    assert NODE_WEBHOOK not in [e["node"] for e in final.values["trace"]]
    assert chat.calls == ["channel_copy_draft", "compliance_audit_result"]
