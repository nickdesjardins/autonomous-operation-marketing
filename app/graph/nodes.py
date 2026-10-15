"""The six nodes of the pipeline and the three routing functions between them.

Every node is a method of :class:`PipelineNodes`, bound to the collaborators it needs (the
inventory tools, the structured chat client, the webhook dispatcher, a clock). A node returns a
partial state update and never mutates the state it was given. Each one also appends a trace
event, so the playground can walk a visitor through exactly what happened and which model
answered.

The routers are pure functions of the state after the node ran. They decide the cycle: a
non-compliant draft goes back to synthesis while ``revision_count`` is below
``max_revisions``, and to the human gate (flagged as an escalation) once the budget is spent.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from langgraph.graph import END
from pydantic import ValidationError

from app.compliance.rules import (
    deterministic_findings,
    format_revision_instructions,
    merge_audit,
    overrule_model_findings,
)
from app.config import Settings
from app.errors import InventoryError, ProviderError
from app.graph.prompts import audit_messages, new_nonce, synthesis_messages
from app.graph.state import (
    STATUS_AWAITING_REVIEW,
    STATUS_COMPLIANT,
    STATUS_DISPATCH_FAILED,
    STATUS_DISPATCH_REFUSED,
    STATUS_DISPATCHED,
    STATUS_DRAFTED,
    STATUS_ESCALATED,
    STATUS_INVALID_REQUEST,
    STATUS_INVENTORY_LOADED,
    STATUS_OUT_OF_STOCK,
    STATUS_REVISION_REQUESTED,
    STATUS_SKU_NOT_FOUND,
    STATUS_STARTED,
    PipelineState,
)
from app.graph.tools import InventoryTools
from app.schemas import (
    ALL_CHANNELS,
    AuditEntry,
    ChannelCopyDraft,
    ComplianceAuditResult,
    utc_now_iso,
)
from app.services.llm_client import StructuredChat
from app.services.webhook_dispatcher import WebhookDispatcher, build_payload

logger = logging.getLogger(__name__)

# Node names, repeated here so that the routers can name their targets without importing the
# builder (which imports this module). The builder asserts the two sets agree.
NODE_INVENTORY = "inventory_sql_agent"
NODE_SYNTHESIS = "copy_synthesis_agent"
NODE_GATE = "human_approval_gate"
NODE_WEBHOOK = "webhook_dispatch"

#: The human reviewer's decision, as the webhook node needs it before it will send anything.
DISPATCHABLE_REVIEW_STATUSES = ("approved", "edited")

#: House typography, enforced in code rather than left to the model: dashes a model likes to
#: write become plain hyphens, and the non-breaking variants become their ordinary characters.
#: The characters are spelled by code point so that none of them appears in this file.
_TYPOGRAPHY = {
    chr(0x2014): " - ",  # em dash
    chr(0x2013): "-",  # en dash
    chr(0x2011): "-",  # non-breaking hyphen
    chr(0x2010): "-",  # unicode hyphen
    chr(0x00A0): " ",  # no-break space
}
_DOUBLE_SPACES = re.compile(r" {2,}")


def tidy_text(text: str) -> tuple[str, int]:
    """Replace the characters in ``_TYPOGRAPHY``; returns the text and how many were replaced."""
    count = 0
    for character, replacement in _TYPOGRAPHY.items():
        hits = text.count(character)
        if hits:
            count += hits
            text = text.replace(character, replacement)
    return (_DOUBLE_SPACES.sub(" ", text) if count else text), count


def tidy_draft(draft: ChannelCopyDraft) -> tuple[ChannelCopyDraft, int]:
    """Apply :func:`tidy_text` to every text field of a draft (the NPN field included)."""
    data = draft.model_dump()
    total = 0
    for key, value in data.items():
        if isinstance(value, str):
            data[key], hits = tidy_text(value)
            total += hits
        elif isinstance(value, list):
            tidied = []
            for item in value:
                item, hits = tidy_text(item)
                total += hits
                tidied.append(item)
            data[key] = tidied
    return ChannelCopyDraft.model_validate(data), total


class PipelineNodes:
    """The node implementations, bound to their collaborators."""

    def __init__(
        self,
        settings: Settings,
        *,
        tools: InventoryTools,
        chat: StructuredChat | None,
        dispatcher: WebhookDispatcher,
        clock: Callable[[], str] = utc_now_iso,
    ) -> None:
        self.settings = settings
        self.tools = tools
        self.chat = chat
        self.dispatcher = dispatcher
        self.clock = clock

    # ------------------------------------------------------------------ helpers
    def _event(self, node: str, started: float, **detail: Any) -> dict[str, Any]:
        return {
            "node": node,
            "at": self.clock(),
            "ms": round((time.perf_counter() - started) * 1000.0, 1),
            "detail": detail,
        }

    def _chat(self) -> StructuredChat:
        if self.chat is None:
            raise ProviderError(
                self.settings.llm_provider, "no language model is configured (missing API key)"
            )
        return self.chat

    # ------------------------------------------------------------------- nodes
    async def supervisor(self, state: PipelineState) -> dict[str, Any]:
        """Validate the request, reset the revision counter, decide where to start."""
        started = time.perf_counter()
        problems: list[str] = []
        sku_code = (state.get("sku_code") or "").strip()
        if not sku_code:
            problems.append("sku_code is missing")
        channels = list(state.get("target_channels") or [])
        if not channels:
            problems.append("no target channel was requested")
        unknown = [channel for channel in channels if channel not in ALL_CHANNELS]
        if unknown:
            problems.append(f"unknown channels: {', '.join(unknown)}")
        if not (state.get("campaign_angle") or "").strip():
            problems.append("campaign_angle is empty")
        max_revisions = int(state.get("max_revisions", self.settings.max_revisions))
        if max_revisions < 0:
            problems.append("max_revisions must be 0 or more")
        if problems:
            return {
                "status": STATUS_INVALID_REQUEST,
                "error": "; ".join(problems),
                "updated_at": self.clock(),
                "trace": [self._event("supervisor", started, outcome="invalid", problems=problems)],
            }
        has_inventory = bool(state.get("inventory_data"))
        route = NODE_SYNTHESIS if has_inventory else NODE_INVENTORY
        return {
            "sku_code": sku_code.upper(),
            "revision_count": int(state.get("revision_count", 0)) if has_inventory else 0,
            "max_revisions": max_revisions,
            "status": STATUS_STARTED,
            "human_review_status": state.get("human_review_status") or "pending",
            "error": None,
            "updated_at": self.clock(),
            "trace": [
                self._event(
                    "supervisor",
                    started,
                    outcome="routed",
                    route=route,
                    channels=channels,
                    max_revisions=max_revisions,
                )
            ],
        }

    def route_after_supervisor(self, state: PipelineState) -> str:
        if state.get("status") == STATUS_INVALID_REQUEST:
            return END
        return NODE_SYNTHESIS if state.get("inventory_data") else NODE_INVENTORY

    async def inventory_sql_agent(self, state: PipelineState) -> dict[str, Any]:
        """One deterministic, parameterized lookup. No model involved."""
        started = time.perf_counter()
        sku_code = state["sku_code"]
        try:
            record = await self.tools.lookup_sku(sku_code)
        except ValidationError as exc:
            return {
                "status": STATUS_SKU_NOT_FOUND,
                "error": f"SKU {sku_code!r} is not a valid SKU code",
                "updated_at": self.clock(),
                "trace": [
                    self._event(
                        "inventory_sql_agent",
                        started,
                        tool="lookup_sku",
                        outcome="invalid_args",
                        problems=[error.get("msg", "") for error in exc.errors()],
                    )
                ],
            }
        except InventoryError:
            raise
        if record is None:
            return {
                "status": STATUS_SKU_NOT_FOUND,
                "error": f"SKU {sku_code} is not in the inventory",
                "updated_at": self.clock(),
                "trace": [
                    self._event(
                        "inventory_sql_agent", started, tool="lookup_sku", outcome="not_found"
                    )
                ],
            }
        if record.stock_quantity <= 0:
            return {
                "inventory_data": record.as_state(),
                "status": STATUS_OUT_OF_STOCK,
                "error": (
                    f"SKU {sku_code} ({record.product_name}) has no stock: nothing to list until "
                    "the next production run"
                ),
                "updated_at": self.clock(),
                "trace": [
                    self._event(
                        "inventory_sql_agent",
                        started,
                        tool="lookup_sku",
                        outcome="out_of_stock",
                        product_name=record.product_name,
                        stock_quantity=record.stock_quantity,
                    )
                ],
            }
        return {
            "inventory_data": record.as_state(),
            "status": STATUS_INVENTORY_LOADED,
            "error": None,
            "updated_at": self.clock(),
            "trace": [
                self._event(
                    "inventory_sql_agent",
                    started,
                    tool="lookup_sku",
                    outcome="found",
                    product_name=record.product_name,
                    npn_number=record.npn_number,
                    stock_quantity=record.stock_quantity,
                    ingredients=len(record.active_ingredients),
                    authorized_claims=len(record.authorized_claims),
                )
            ],
        }

    def route_after_inventory(self, state: PipelineState) -> str:
        if state.get("status") in (STATUS_SKU_NOT_FOUND, STATUS_OUT_OF_STOCK):
            return END
        return NODE_SYNTHESIS

    async def copy_synthesis_agent(self, state: PipelineState) -> dict[str, Any]:
        """Draft (or redraft) the copy, grounded in the SKU data and the critic's feedback."""
        started = time.perf_counter()
        revision = int(state.get("revision_count", 0))
        report = state.get("compliance_report") or {}
        previous = state.get("draft_copy") if revision > 0 else None
        messages = synthesis_messages(
            inventory=state["inventory_data"] or {},
            target_channels=list(state.get("target_channels") or ALL_CHANNELS),
            campaign_angle=state.get("campaign_angle") or "",
            revision_count=revision,
            previous_draft=previous,
            violations=list(report.get("violations") or []) if revision > 0 else None,
            revision_instructions=list(state.get("revision_instructions") or []),
            nonce=new_nonce(),
        )
        outcome = await self._chat().complete(
            messages=messages,
            schema_model=ChannelCopyDraft,
            temperature=self.settings.synthesis_temperature,
            max_tokens=self.settings.synthesis_max_tokens,
            schema_name="channel_copy_draft",
        )
        if outcome.refused:
            raise ProviderError(outcome.model, "the model declined to write this copy")
        try:
            draft, replaced = tidy_draft(outcome.parsed)
        except ValidationError as exc:
            # An em dash becomes three characters: a field at its cap can overflow it. The
            # model's answer is unusable then, like any other, and the API says so (503).
            raise ProviderError(
                outcome.model,
                "the draft failed validation after typography fixes",
                retryable=True,
            ) from exc
        return {
            "draft_copy": draft.model_dump(),
            "status": STATUS_DRAFTED,
            "error": None,
            "updated_at": self.clock(),
            "trace": [
                self._event(
                    "copy_synthesis_agent",
                    started,
                    revision=revision,
                    model=outcome.model,
                    fallback_triggered=outcome.fallback_triggered,
                    json_mode=outcome.json_mode,
                    attempts=[asdict(attempt) for attempt in outcome.attempts],
                    typography_fixes=replaced,
                )
            ],
        }

    async def compliance_validator(self, state: PipelineState) -> dict[str, Any]:
        """Audit the draft: the model's verdict, then the deterministic checks on top."""
        started = time.perf_counter()
        inventory = state["inventory_data"] or {}
        draft = ChannelCopyDraft.model_validate(state["draft_copy"])
        outcome = await self._chat().complete(
            messages=audit_messages(
                draft=draft.model_dump(), inventory=inventory, nonce=new_nonce()
            ),
            schema_model=ComplianceAuditResult,
            temperature=self.settings.audit_temperature,
            max_tokens=self.settings.audit_max_tokens,
            schema_name="compliance_audit_result",
        )
        if outcome.refused:
            raise ProviderError(outcome.model, "the auditor declined to audit this copy")
        llm_audit: ComplianceAuditResult = outcome.parsed
        findings = deterministic_findings(draft, inventory)
        # The code decides the mechanical rules and discards wording-only findings on
        # restated authorized claims; the deterministic findings are then merged in.
        upheld, discarded = overrule_model_findings(
            llm_audit, findings, list(inventory.get("authorized_claims") or [])
        )
        audit = merge_audit(upheld, findings)
        added = len(audit.violations) - len(upheld.violations)
        revision = int(state.get("revision_count", 0))
        max_revisions = int(state.get("max_revisions", self.settings.max_revisions))
        entry = AuditEntry(
            revision=revision,
            is_compliant=audit.is_compliant,
            risk_level=audit.risk_level,
            violations=audit.violations,
            revision_summary=audit.revision_summary,
            deterministic_findings=added,
            discarded_findings=len(discarded),
            model=outcome.model,
            checked_at=self.clock(),
        )
        update: dict[str, Any] = {
            "compliance_report": audit.model_dump(),
            "audit_history": [entry.model_dump()],
            "error": None,
            "updated_at": self.clock(),
        }
        if audit.is_compliant:
            update["status"] = STATUS_COMPLIANT
            update["requires_human_escalation"] = False
            decision = "compliant"
        else:
            new_count = revision + 1
            escalate = new_count >= max_revisions
            update["revision_count"] = new_count
            update["revision_instructions"] = [format_revision_instructions(audit, new_count)]
            update["requires_human_escalation"] = escalate
            update["status"] = STATUS_ESCALATED if escalate else STATUS_REVISION_REQUESTED
            decision = "escalate" if escalate else "revise"
        update["trace"] = [
            self._event(
                "compliance_validator",
                started,
                revision=revision,
                model=outcome.model,
                decision=decision,
                risk_level=audit.risk_level,
                violations=len(audit.violations),
                deterministic_findings=added,
                discarded_findings=discarded,
                llm_said_compliant=llm_audit.is_compliant,
                fallback_triggered=outcome.fallback_triggered,
                attempts=[asdict(attempt) for attempt in outcome.attempts],
            )
        ]
        return update

    def route_after_compliance(self, state: PipelineState) -> str:
        report = state.get("compliance_report") or {}
        if report.get("is_compliant"):
            return NODE_GATE
        revision_count = int(state.get("revision_count", 0))
        max_revisions = int(state.get("max_revisions", self.settings.max_revisions))
        return NODE_SYNTHESIS if revision_count < max_revisions else NODE_GATE

    async def human_approval_gate(self, state: PipelineState) -> dict[str, Any]:
        """Mark the thread as waiting. The graph pauses right after this node."""
        started = time.perf_counter()
        escalated = bool(state.get("requires_human_escalation"))
        reason = (
            "the self-correction budget is spent and violations remain"
            if escalated
            else "the draft is compliant and waits for sign-off"
        )
        return {
            "status": STATUS_AWAITING_REVIEW,
            "human_review_status": "pending",
            "updated_at": self.clock(),
            "trace": [
                self._event("human_approval_gate", started, escalated=escalated, reason=reason)
            ],
        }

    async def webhook_dispatch(self, state: PipelineState) -> dict[str, Any]:
        """The only mutation: fire the signed webhook, and only after a human said yes."""
        started = time.perf_counter()
        review = state.get("human_review_status")
        if review not in DISPATCHABLE_REVIEW_STATUSES:
            return {
                "status": STATUS_DISPATCH_REFUSED,
                "error": f"webhook refused: the human review status is {review!r}",
                "updated_at": self.clock(),
                "trace": [
                    self._event("webhook_dispatch", started, outcome="refused", review=review)
                ],
            }
        # The copy that leaves is validated once more, whoever wrote it (the model or the human).
        ChannelCopyDraft.model_validate(state.get("draft_copy"))
        payload = build_payload(dict(state))
        result = await self.dispatcher.dispatch(payload)
        delivered = result.delivered or result.mode == "dry_run"
        return {
            "webhook_delivery_result": result.as_state(),
            "status": STATUS_DISPATCHED if delivered else STATUS_DISPATCH_FAILED,
            "error": result.error,
            "updated_at": self.clock(),
            "trace": [
                self._event(
                    "webhook_dispatch",
                    started,
                    outcome="delivered" if delivered else "failed",
                    mode=result.mode,
                    attempts=result.attempts,
                    status_code=result.status_code,
                    delivery_id=result.delivery_id,
                    review=review,
                )
            ],
        }
