"""The typed graph state and its reducers.

Every node returns a partial update. Keys without a reducer are overwritten (``draft_copy`` is
always the latest draft); keys annotated with ``operator.add`` accumulate across the
self-correction cycles, so the audit trail and the critic's feedback are never lost.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

HumanReviewStatus = Literal["pending", "approved", "edited", "rejected"]

# Status values a thread moves through. The API and the playground read these; a test holds
# the README's state table to this list.
STATUS_STARTED = "started"
STATUS_INVALID_REQUEST = "invalid_request"
STATUS_INVENTORY_LOADED = "inventory_loaded"
STATUS_SKU_NOT_FOUND = "sku_not_found"
STATUS_OUT_OF_STOCK = "out_of_stock"
STATUS_DRAFTED = "drafted"
STATUS_COMPLIANT = "compliant"
STATUS_REVISION_REQUESTED = "revision_requested"
STATUS_ESCALATED = "escalated_to_human"
STATUS_AWAITING_REVIEW = "awaiting_human_review"
STATUS_REJECTED = "rejected"
STATUS_DISPATCHED = "dispatched"
STATUS_DISPATCH_FAILED = "dispatch_failed"
STATUS_DISPATCH_REFUSED = "dispatch_refused"

TERMINAL_STATUSES = frozenset(
    {
        STATUS_INVALID_REQUEST,
        STATUS_SKU_NOT_FOUND,
        STATUS_OUT_OF_STOCK,
        STATUS_REJECTED,
        STATUS_DISPATCHED,
        STATUS_DISPATCH_FAILED,
        STATUS_DISPATCH_REFUSED,
    }
)


class PipelineState(TypedDict, total=False):
    thread_id: str
    sku_code: str
    target_channels: list[str]
    campaign_angle: str
    inventory_data: dict[str, Any] | None
    draft_copy: dict[str, Any] | None
    compliance_report: dict[str, Any] | None
    audit_history: Annotated[list[dict[str, Any]], operator.add]
    revision_instructions: Annotated[list[str], operator.add]
    revision_count: int
    max_revisions: int
    requires_human_escalation: bool
    human_review_status: HumanReviewStatus
    human_reviewer_id: str | None
    human_reviewer_notes: str | None
    webhook_delivery_result: dict[str, Any] | None
    status: str
    error: str | None
    trace: Annotated[list[dict[str, Any]], operator.add]
    created_at: str
    updated_at: str


def initial_state(
    *,
    thread_id: str,
    sku_code: str,
    target_channels: list[str],
    campaign_angle: str,
    max_revisions: int,
    now: str,
) -> PipelineState:
    """The state a new thread starts from. Everything the supervisor validates is here."""
    return PipelineState(
        thread_id=thread_id,
        sku_code=sku_code,
        target_channels=list(target_channels),
        campaign_angle=campaign_angle,
        inventory_data=None,
        draft_copy=None,
        compliance_report=None,
        audit_history=[],
        revision_instructions=[],
        revision_count=0,
        max_revisions=max_revisions,
        requires_human_escalation=False,
        human_review_status="pending",
        human_reviewer_id=None,
        human_reviewer_notes=None,
        webhook_delivery_result=None,
        status=STATUS_STARTED,
        error=None,
        trace=[],
        created_at=now,
        updated_at=now,
    )
