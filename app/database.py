"""PostgreSQL through one psycopg pool: the inventory, the webhook tables and the checkpointer.

One driver, one pool. ``langgraph-checkpoint-postgres`` needs psycopg connections opened with
``autocommit=True`` and ``row_factory=dict_row`` (and ``prepare_threshold=0`` behind a
transaction-mode pooler such as Neon's); the inventory queries are happy with the same, so
``AsyncPostgresSaver`` is simply handed the pool the inventory reads use. The inventory
queries are parameterized and read-only; the only writes are the webhook audit log, the
receiver's inbox and the idempotent seed at start-up.

Two in-memory stand-ins live here too: ``InMemoryInventory`` and ``InMemoryWebhookStore``.
They serve ``CHECKPOINTER=memory`` (no database at all) and the hermetic test suite.

Windows note: psycopg's async connections cannot run on asyncio's default ProactorEventLoop.
uvicorn already picks the selector loop on Windows; plain ``asyncio.run`` scripts and pytest do
not, so :func:`ensure_compatible_event_loop` is called at import on Windows (a no-op elsewhere).
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any

from app.config import Settings
from app.errors import InventoryError
from app.schemas import SkuRecord, SkuSummary, WebhookDelivery, utc_now_iso
from app.seed import schema_statements, seed_records, seed_statements

logger = logging.getLogger(__name__)


def ensure_compatible_event_loop() -> None:
    """On Windows, make new event loops selector-based, as psycopg's async mode requires.

    Harmless when a loop is already running (uvicorn): the policy only shapes loops created
    afterwards. Scripts that call ``asyncio.run`` must import this module (or call this) first.
    """
    if sys.platform == "win32":
        policy = asyncio.get_event_loop_policy()
        if not isinstance(policy, asyncio.WindowsSelectorEventLoopPolicy):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


ensure_compatible_event_loop()

_SELECT_SKU = (
    "SELECT sku_code, product_name, npn_number, dosage_form, stock_quantity, serving_size, "
    "active_ingredients, authorized_claims, risk_information, contraindications, brand_notes "
    "FROM inventory_skus WHERE sku_code = %(sku_code)s"
)
_LIST_SKUS = (
    "SELECT sku_code, product_name, npn_number, dosage_form, stock_quantity "
    "FROM inventory_skus ORDER BY sku_code"
)
_COUNT_SKUS = "SELECT count(*) AS n FROM inventory_skus"
_INSERT_AUDIT = (
    "INSERT INTO webhook_audit_log (delivery_id, thread_id, attempt, target_mode, url_host, "
    "status_code, ok, error, duration_ms) VALUES (%(delivery_id)s, %(thread_id)s, %(attempt)s, "
    "%(target_mode)s, %(url_host)s, %(status_code)s, %(ok)s, %(error)s, %(duration_ms)s)"
)
_INSERT_INBOX = (
    "INSERT INTO webhook_inbox (delivery_id, thread_id, signature_valid, payload) "
    "VALUES (%(delivery_id)s, %(thread_id)s, %(signature_valid)s, %(payload)s::jsonb) "
    "ON CONFLICT (delivery_id) DO NOTHING"
)
_LIST_INBOX = (
    "SELECT delivery_id, thread_id, received_at, signature_valid, payload FROM webhook_inbox "
    "ORDER BY received_at DESC LIMIT %(limit)s"
)


def _record_from_row(row: dict[str, Any]) -> SkuRecord:
    return SkuRecord.model_validate(dict(row))


class Database:
    """Owns the pool and exposes the handful of operations the pipeline needs."""

    def __init__(self, settings: Settings) -> None:
        from psycopg.rows import dict_row
        from psycopg_pool import AsyncConnectionPool

        self.settings = settings
        self._pool = AsyncConnectionPool(
            settings.effective_database_url,
            min_size=settings.db_pool_min_size,
            max_size=settings.db_pool_max_size,
            open=False,
            timeout=settings.db_connect_timeout_s,
            # Serverless Postgres (Neon) suspends idle computes and drops their connections:
            # check a connection before handing it out, and do not keep idle ones for long.
            check=AsyncConnectionPool.check_connection,
            max_idle=240.0,
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        )
        self._checkpointer: Any = None
        self.checkpointer_ready = False

    # ------------------------------------------------------------------ lifecycle
    async def open(self) -> None:
        await self._pool.open(wait=True, timeout=self.settings.db_connect_timeout_s)

    async def close(self) -> None:
        await self._pool.close()

    async def init_schema(self) -> None:
        """Create the inventory tables and upsert the sample SKUs (idempotent)."""
        async with self._pool.connection() as conn:
            for statement in schema_statements():
                await conn.execute(statement)
            for statement in seed_statements():
                await conn.execute(statement)
        logger.info("Inventory schema ready (%d sample SKUs)", len(seed_records()))

    def checkpointer(self) -> Any:
        """The LangGraph saver on this pool (built once)."""
        if self._checkpointer is None:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            self._checkpointer = AsyncPostgresSaver(self._pool)
        return self._checkpointer

    async def setup_checkpointer(self) -> None:
        """Create the checkpoint tables and run the library's migrations (idempotent)."""
        await self.checkpointer().setup()
        self.checkpointer_ready = True

    async def verify_checkpointer(self) -> None:
        """Confirm the checkpoint tables exist without creating them (``AUTO_CREATE_SCHEMA=false``).

        A schema managed by migrations is probed, not written: one read of the main table.
        A missing table raises here, at start-up, instead of on the first run.
        """
        self.checkpointer()
        async with self._pool.connection() as conn:
            await conn.execute("SELECT 1 FROM checkpoints LIMIT 1")
        self.checkpointer_ready = True

    # --------------------------------------------------------------------- health
    async def ping(self) -> bool:
        try:
            async with self._pool.connection() as conn:
                await conn.execute("SELECT 1")
            return True
        except Exception as exc:  # noqa: BLE001 - a probe reports, it never raises
            logger.warning("Database ping failed: %s", type(exc).__name__)
            return False

    async def count_skus(self) -> int:
        async with self._pool.connection() as conn:
            cursor = await conn.execute(_COUNT_SKUS)
            row = await cursor.fetchone()
        return int(row["n"]) if row else 0

    # ---------------------------------------------------------------------- reads
    async def get_sku(self, sku_code: str) -> SkuRecord | None:
        try:
            async with self._pool.connection() as conn:
                cursor = await conn.execute(_SELECT_SKU, {"sku_code": sku_code})
                row = await cursor.fetchone()
        except Exception as exc:  # noqa: BLE001 - every driver error is one error to the graph
            raise InventoryError(f"inventory read failed ({type(exc).__name__})") from exc
        return _record_from_row(row) if row else None

    async def list_skus(self) -> list[SkuSummary]:
        try:
            async with self._pool.connection() as conn:
                cursor = await conn.execute(_LIST_SKUS)
                rows = await cursor.fetchall()
        except Exception as exc:  # noqa: BLE001
            raise InventoryError(f"inventory read failed ({type(exc).__name__})") from exc
        return [SkuSummary.model_validate(dict(row)) for row in rows]

    # --------------------------------------------------------------------- writes
    async def record_webhook_attempt(self, entry: dict[str, Any]) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(_INSERT_AUDIT, entry)

    async def store_inbox(self, delivery: WebhookDelivery) -> None:
        import json

        async with self._pool.connection() as conn:
            await conn.execute(
                _INSERT_INBOX,
                {
                    "delivery_id": delivery.delivery_id,
                    "thread_id": delivery.thread_id,
                    "signature_valid": delivery.signature_valid,
                    "payload": json.dumps(delivery.payload, ensure_ascii=False),
                },
            )

    async def list_inbox(self, limit: int = 20) -> list[WebhookDelivery]:
        async with self._pool.connection() as conn:
            cursor = await conn.execute(_LIST_INBOX, {"limit": int(limit)})
            rows = await cursor.fetchall()
        return [
            WebhookDelivery(
                delivery_id=row["delivery_id"],
                thread_id=row["thread_id"],
                received_at=row["received_at"].strftime("%Y-%m-%dT%H:%M:%SZ"),
                signature_valid=bool(row["signature_valid"]),
                payload=dict(row["payload"]),
            )
            for row in rows
        ]


class InMemoryInventory:
    """The sample SKUs in a dictionary: laptop mode and tests."""

    def __init__(self, records: list[SkuRecord] | None = None) -> None:
        self._records = {record.sku_code: record for record in (records or seed_records())}

    async def get_sku(self, sku_code: str) -> SkuRecord | None:
        record = self._records.get(sku_code)
        return record.model_copy(deep=True) if record else None

    async def list_skus(self) -> list[SkuSummary]:
        return [
            SkuSummary(
                sku_code=r.sku_code,
                product_name=r.product_name,
                npn_number=r.npn_number,
                dosage_form=r.dosage_form,
                stock_quantity=r.stock_quantity,
            )
            for r in sorted(self._records.values(), key=lambda r: r.sku_code)
        ]

    async def count_skus(self) -> int:
        return len(self._records)

    async def ping(self) -> bool:
        return True


class InMemoryWebhookStore:
    """The audit log and the receiver's inbox, in memory."""

    def __init__(self) -> None:
        self.audit: list[dict[str, Any]] = []
        self.inbox: list[WebhookDelivery] = []

    async def record_webhook_attempt(self, entry: dict[str, Any]) -> None:
        self.audit.append({**entry, "created_at": utc_now_iso()})

    async def store_inbox(self, delivery: WebhookDelivery) -> None:
        if all(item.delivery_id != delivery.delivery_id for item in self.inbox):
            self.inbox.append(delivery)

    async def list_inbox(self, limit: int = 20) -> list[WebhookDelivery]:
        return list(reversed(self.inbox))[: int(limit)]
