"""Deterministic, read-only inventory tools.

The Inventory SQL Agent never asks a language model to write SQL. It calls these two tools,
whose arguments are validated by Pydantic and whose queries are parameterized and read-only.
``TOOL_SPECS`` describes them in the JSON-schema form a tool-calling model would receive, so
a future planner node could pick between them; today the supervisor routes to ``lookup_sku``
directly, because a SKU lookup is not a decision a model should make.
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.schemas import SkuRecord, SkuSummary


class InventoryRepository(Protocol):
    """What the tools need from the database (or from the in-memory stand-in in tests)."""

    async def get_sku(self, sku_code: str) -> SkuRecord | None: ...

    async def list_skus(self) -> list[SkuSummary]: ...


class LookupSkuArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sku_code: str = Field(
        min_length=3, max_length=32, pattern=r"^[A-Z0-9][A-Z0-9-]*$", description="The SKU code."
    )


class ListSkusArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "lookup_sku",
        "description": (
            "Read one SKU from the inventory database: product name, NPN, dosage form, stock, "
            "serving, active ingredients with amounts, authorized claims, risk information."
        ),
        "parameters": LookupSkuArgs.model_json_schema(),
        "read_only": True,
    },
    {
        "name": "list_skus",
        "description": "List the SKU codes, names, NPNs and stock counts in the inventory.",
        "parameters": ListSkusArgs.model_json_schema(),
        "read_only": True,
    },
]


class InventoryTools:
    """The tools, bound to one repository."""

    def __init__(self, repository: InventoryRepository) -> None:
        self._repository = repository

    async def lookup_sku(self, sku_code: str) -> SkuRecord | None:
        args = LookupSkuArgs(sku_code=sku_code.strip().upper())
        return await self._repository.get_sku(args.sku_code)

    async def list_skus(self) -> list[SkuSummary]:
        ListSkusArgs()
        return await self._repository.list_skus()
