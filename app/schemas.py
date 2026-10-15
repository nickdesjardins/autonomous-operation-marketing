"""Pydantic v2 contracts: what the models must return, and what the API accepts and returns.

The language models never produce free text here. Every call is constrained to one of the two
output schemas below (``ChannelCopyDraft``, ``ComplianceAuditResult``), and the service validates
the JSON before it reaches the graph state. The API models keep that discipline at the boundary:
a request is validated before any model is called, and a response is built from typed data.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Channel = Literal["web_store", "marketplace", "social"]
ALL_CHANNELS: tuple[Channel, ...] = ("web_store", "marketplace", "social")
RiskLevel = Literal["low", "medium", "high"]
ReviewAction = Literal["approve", "edit", "reject"]
HumanReviewStatus = Literal["pending", "approved", "edited", "rejected"]

_SKU_CODE = re.compile(r"^[A-Z0-9][A-Z0-9-]{2,31}$")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class StrictModel(BaseModel):
    """A model that refuses unknown fields: a drifting provider answer fails loudly."""

    model_config = ConfigDict(extra="forbid")


# ------------------------------------------------------------------ model outputs


class ChannelCopyDraft(StrictModel):
    """One draft of the marketing copy for every channel, plus the image prompt.

    The field names are part of the compliance report: a violation names the field it was
    found in, and the human reviewer can override the whole draft with the same shape.
    """

    web_title: str = Field(min_length=5, max_length=160, description="E-commerce product title.")
    web_description_html: str = Field(
        min_length=120,
        max_length=6000,
        description=(
            "Product page body in simple HTML (p, ul, li, strong): two or three short paragraphs, "
            "the serving and the authorized claims. Must show the NPN."
        ),
    )
    marketplace_bullets: list[str] = Field(
        min_length=5, max_length=5, description="Exactly five marketplace bullet points."
    )
    social_post_copy: str = Field(
        min_length=10, max_length=1200, description="One social campaign post. Must show the NPN."
    )
    image_generation_prompt: str = Field(
        min_length=10,
        max_length=1500,
        description="A prompt for an image model: product, setting, mood. No text overlays.",
    )
    disclaimers_included: list[str] = Field(
        min_length=1,
        max_length=12,
        description="Every mandatory risk statement, copied verbatim from the licence.",
    )
    npn_displayed: str = Field(
        min_length=1, max_length=40, description="The NPN exactly as it appears in the copy."
    )

    @field_validator("marketplace_bullets", "disclaimers_included")
    @classmethod
    def _non_empty_lines(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value]
        if any(len(item) < 3 for item in cleaned):
            raise ValueError("every entry needs at least three characters")
        if any(len(item) > 400 for item in cleaned):
            raise ValueError("an entry is longer than 400 characters")
        return cleaned

    def text_fields(self) -> dict[str, str]:
        """Every text the deterministic checks and the auditor read, keyed by field name."""
        return {
            "web_title": self.web_title,
            "web_description_html": self.web_description_html,
            "marketplace_bullets": "\n".join(self.marketplace_bullets),
            "social_post_copy": self.social_post_copy,
            "image_generation_prompt": self.image_generation_prompt,
            "disclaimers_included": "\n".join(self.disclaimers_included),
            "npn_displayed": self.npn_displayed,
        }


class ViolationItem(StrictModel):
    """One regulatory problem in one field, with the fix the next draft must make."""

    field_name: str = Field(min_length=1, max_length=60)
    violating_text: str = Field(min_length=1, max_length=600)
    regulation_rule: str = Field(min_length=1, max_length=200)
    required_fix: str = Field(min_length=1, max_length=600)


class ComplianceAuditResult(StrictModel):
    """The auditor's verdict on one draft."""

    is_compliant: bool
    risk_level: RiskLevel
    violations: list[ViolationItem] = Field(default_factory=list, max_length=40)
    revision_summary: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def _verdict_matches_findings(self) -> ComplianceAuditResult:
        # The verdict follows from the findings, never the other way round. "Compliant" with
        # violations listed would send a draft out with known problems; "not compliant" with
        # no violation (a flag a model sets wrong, measured on 15 October 2026 with the text
        # "The draft is compliant with all rules") would loop with nothing to fix.
        self.is_compliant = not self.violations
        return self


# -------------------------------------------------------------------- inventory


class Ingredient(StrictModel):
    name: str = Field(min_length=1, max_length=120)
    amount: float = Field(gt=0)
    unit: str = Field(min_length=1, max_length=20)
    per: str = Field(default="serving", max_length=40)


class RiskStatement(StrictModel):
    text: str = Field(min_length=3, max_length=600)
    mandatory: bool = True


class SkuRecord(StrictModel):
    """One row of ``inventory_skus`` as the graph sees it."""

    sku_code: str
    product_name: str
    npn_number: str
    dosage_form: str
    stock_quantity: int
    serving_size: str
    active_ingredients: list[Ingredient]
    authorized_claims: list[str]
    risk_information: list[RiskStatement]
    contraindications: str | None = None
    brand_notes: str | None = None

    def as_state(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class SkuSummary(StrictModel):
    sku_code: str
    product_name: str
    npn_number: str
    dosage_form: str
    stock_quantity: int


# ------------------------------------------------------------------ API requests


class PipelineStartRequest(StrictModel):
    sku_code: str = Field(description="Inventory SKU code, for example NR-MAG-200.")
    target_channels: list[Channel] = Field(
        default_factory=lambda: list(ALL_CHANNELS), min_length=1, max_length=3
    )
    campaign_angle: str = Field(
        min_length=3,
        max_length=300,
        description="The brief for this campaign, in one or two sentences.",
    )
    max_revisions: int | None = Field(
        default=None, ge=0, le=5, description="Self-correction loops allowed (default 3)."
    )

    @field_validator("sku_code", mode="before")
    @classmethod
    def _upper_sku(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("sku_code")
    @classmethod
    def _sku_shape(cls, value: str) -> str:
        if not _SKU_CODE.match(value):
            raise ValueError("sku_code must be 3 to 32 upper-case letters, digits or hyphens")
        return value

    @field_validator("target_channels")
    @classmethod
    def _unique_channels(cls, value: list[Channel]) -> list[Channel]:
        if len(set(value)) != len(value):
            raise ValueError("target_channels must not repeat a channel")
        return value

    @field_validator("campaign_angle")
    @classmethod
    def _plain_text(cls, value: str) -> str:
        text = " ".join(value.split())
        if _CONTROL_CHARS.search(text):
            raise ValueError("campaign_angle must not contain control characters")
        if len(text) < 3:
            raise ValueError("campaign_angle is too short")
        return text


class HumanReviewRequest(StrictModel):
    action: ReviewAction
    reviewer_id: str = Field(min_length=1, max_length=80)
    notes: str | None = Field(default=None, max_length=2000)
    override_copy: ChannelCopyDraft | None = None

    @field_validator("reviewer_id", "notes")
    @classmethod
    def _no_control_chars(cls, value: str | None) -> str | None:
        if value is not None and _CONTROL_CHARS.search(value):
            raise ValueError("control characters are not allowed")
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _edit_needs_copy(self) -> HumanReviewRequest:
        if self.action == "edit" and self.override_copy is None:
            raise ValueError("action 'edit' requires override_copy")
        if self.action != "edit" and self.override_copy is not None:
            raise ValueError("override_copy is only accepted with action 'edit'")
        return self


# ----------------------------------------------------------------- API responses


class AuditEntry(BaseModel):
    """One compliance evaluation, as stored in ``audit_history``."""

    revision: int
    is_compliant: bool
    risk_level: RiskLevel
    violations: list[ViolationItem]
    revision_summary: str
    deterministic_findings: int = Field(
        default=0, description="How many of the violations came from the Python checks."
    )
    discarded_findings: int = Field(
        default=0,
        description="Model findings the deterministic checks or the rules overruled (not kept).",
    )
    model: str | None = None
    checked_at: str


class TraceEvent(BaseModel):
    """One node run, for the playground's walkthrough."""

    node: str
    at: str
    ms: float | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class PipelineStatusResponse(BaseModel):
    thread_id: str
    status: str
    sku_code: str | None = None
    target_channels: list[str] = Field(default_factory=list)
    campaign_angle: str | None = None
    next_nodes: list[str] = Field(default_factory=list)
    awaiting_human_review: bool = False
    revision_count: int = 0
    max_revisions: int = 0
    requires_human_escalation: bool = False
    human_review_status: HumanReviewStatus = "pending"
    human_reviewer_id: str | None = None
    human_reviewer_notes: str | None = None
    inventory_data: dict[str, Any] | None = None
    #: The strict model, or the draft as stored when a checkpoint written by an earlier
    #: release no longer fits it: a thread stays readable across schema changes.
    draft_copy: ChannelCopyDraft | dict[str, Any] | None = None
    compliance_report: dict[str, Any] | None = None
    audit_history: list[AuditEntry] = Field(default_factory=list)
    revision_instructions: list[str] = Field(default_factory=list)
    webhook_delivery_result: dict[str, Any] | None = None
    trace: list[TraceEvent] = Field(default_factory=list)
    error: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    checkpoint_id: str | None = None


class HumanReviewResponse(PipelineStatusResponse):
    action: ReviewAction


CheckpointerName = Literal["postgres", "memory"]


class DatabaseStatus(BaseModel):
    kind: CheckpointerName = "postgres"
    connected: bool
    checkpointer_ready: bool
    inventory_skus: int | None = None


class LlmStatus(BaseModel):
    provider: str
    configured: bool
    models: list[str] = Field(default_factory=list)
    mode: str = Field(description="pinned, automatic, no_key, not_openrouter, or fixed.")


class WebhookStatus(BaseModel):
    target: str
    configured: bool
    url_host: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    service: str
    environment: str
    demo_mode: bool
    database: DatabaseStatus
    llm: LlmStatus
    webhook: WebhookStatus


class WebhookDelivery(BaseModel):
    """What the built-in receiver stores for each delivery it accepts."""

    delivery_id: str
    thread_id: str | None
    received_at: str
    signature_valid: bool
    payload: dict[str, Any]


class RuleSummary(BaseModel):
    id: str
    title: str
    summary: str
    severity: RiskLevel


class PlaygroundConfig(BaseModel):
    demo_mode: bool
    service: str
    skus: list[SkuSummary]
    channels: list[str]
    sample_angles: list[str]
    max_revisions: int
    limits: dict[str, int | None]
    llm: LlmStatus
    webhook: WebhookStatus
    rules: list[RuleSummary]
    synthesis_prompt: str
    audit_prompt: str
    topology: dict[str, Any]


def utc_now_iso() -> str:
    """The moment, written the way every record in this project writes it (UTC, seconds)."""
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
