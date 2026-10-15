"""The three tasks a free model must pass before this service will use it.

A model that is listed as free and "supports structured output" has proved nothing yet. The
only evidence that counts is this service's own work: write channel copy for a licensed
product that the deterministic checks find nothing wrong with, catch a planted disease claim
and a missing licence number, and wave a compliant draft through. One case per task, built
from the same sample inventory (:mod:`app.seed`, SKU ``NR-MAG-200``) and the same prompts
(:mod:`app.graph.prompts`) a real run uses, so a model that passes here answers the pipeline's
own requests, not a toy.

Each case is a plain dictionary: ``id``, ``title``, ``schema_model``, ``messages`` (a callable
that builds fresh messages with a new nonce, as a real request does), ``temperature``,
``max_tokens`` and ``grade(parsed) -> bool``. The judging rules (:mod:`app.models.rules`) and
the published list (:mod:`app.models.published`) know nothing about the content: they see ids
and verdicts. :func:`fingerprint_material` is what ties a published chain to this very test.
"""

from __future__ import annotations

import re
from typing import Any

from app.compliance.rules import (
    _PEDANTIC_FIX_RE,
    _RESTATEMENT_WORDS,
    MECHANICAL_RULES,
    PROHIBITED_PATTERNS,
    deterministic_findings,
    merge_audit,
    overrule_model_findings,
)
from app.config import Settings
from app.graph.prompts import audit_messages, new_nonce, synthesis_messages
from app.schemas import ALL_CHANNELS, ChannelCopyDraft, ComplianceAuditResult, SkuRecord
from app.seed import SKU_SEED
from app.services.llm_client import strict_json_schema

#: The sample product every case is about: in stock, four authorized claims, two mandatory
#: warnings, one ingredient with one licensed amount. Enough to get wrong in every way.
SKU_CODE = "NR-MAG-200"
SKU: SkuRecord = next(
    SkuRecord.model_validate(row) for row in SKU_SEED if row["sku_code"] == SKU_CODE
)
INVENTORY: dict[str, Any] = SKU.as_state()

#: A brief that asks for nothing a licence would refuse, so a wrong draft is the model's own.
NEUTRAL_BRIEF = (
    "A plain product introduction for the online store: what the product is, how it is taken, "
    "and what its licence allows it to say, in a calm and factual tone."
)

#: The nonce the fingerprint is built with. A real call gets a fresh one each time; the
#: fingerprint must not change from one process to the next.
FINGERPRINT_NONCE = "fingerprint"

#: The settings' defaults for the two nodes, read from the field definitions so that they
#: cannot drift from what a real run sends. The defaults rather than a service's own values,
#: so that a test's record means the same thing everywhere (the fingerprint holds them).
SYNTHESIS_TEMPERATURE: float = Settings.model_fields["synthesis_temperature"].default
SYNTHESIS_MAX_TOKENS: int = Settings.model_fields["synthesis_max_tokens"].default
AUDIT_TEMPERATURE: float = Settings.model_fields["audit_temperature"].default
AUDIT_MAX_TOKENS: int = Settings.model_fields["audit_max_tokens"].default

#: The licence text the overrule step judges restatements against, as a real run reads it.
AUTHORIZED_CLAIMS: list[str] = list(INVENTORY["authorized_claims"])

#: A draft the deterministic checks find nothing wrong with. A test holds it to that.
COMPLIANT_DRAFT = ChannelCopyDraft(
    web_title="Magnesium Bisglycinate 200 mg Vegetable Capsules",
    web_description_html=(
        "<p><strong>Magnesium Bisglycinate 200 mg</strong> provides 200 mg of magnesium "
        "(magnesium bisglycinate) per capsule, a chelated form chosen for tolerability.</p>"
        "<ul>"
        "<li>Helps to maintain proper muscle function, including the heart muscle.</li>"
        "<li>Helps in the development and maintenance of bones and teeth.</li>"
        "<li>Helps to prevent magnesium deficiency.</li>"
        "<li>Helps in energy metabolism and in tissue formation.</li>"
        "</ul>"
        "<p>Take 1 capsule, 1 to 2 times daily. Vegan, gluten free, non-GMO. NPN 80098765.</p>"
        "<p>Consult a health care practitioner prior to use if you are pregnant or "
        "breastfeeding. Some people may experience diarrhoea.</p>"
    ),
    marketplace_bullets=[
        "Helps to maintain proper muscle function, including the heart muscle.",
        "Helps in the development and maintenance of bones and teeth.",
        "Helps to prevent magnesium deficiency.",
        "200 mg of magnesium (magnesium bisglycinate) per vegetable capsule, 1 to 2 times daily.",
        "Vegan, gluten free and non-GMO; a chelated form chosen for tolerability.",
    ],
    social_post_copy=(
        "Steady evenings start with the basics. Magnesium Bisglycinate 200 mg helps to maintain "
        "proper muscle function and helps in energy metabolism, one capsule at a time. "
        "NPN 80098765. #magnesium #dailyroutine"
    ),
    image_generation_prompt=(
        "A matte white bottle of magnesium capsules on a pale wooden table beside a glass of "
        "water, soft morning light from a window, calm and uncluttered. No text overlays, no "
        "people."
    ),
    disclaimers_included=[
        "Consult a health care practitioner prior to use if you are pregnant or breastfeeding.",
        "Some people may experience diarrhoea.",
    ],
    npn_displayed="NPN 80098765",
)

#: A draft with two planted problems: a cure claim in the product page and in the social post,
#: and no licence number anywhere. An auditor that waves it through is no auditor.
PLANTED_DRAFT = ChannelCopyDraft(
    web_title="Magnesium Bisglycinate 200 mg: Sleep Through the Night",
    web_description_html=(
        "<p>Magnesium Bisglycinate 200 mg cures chronic insomnia and lets you sleep through "
        "the night, every night.</p>"
        "<ul>"
        "<li>Helps to maintain proper muscle function, including the heart muscle.</li>"
        "<li>Helps in energy metabolism and in tissue formation.</li>"
        "</ul>"
        "<p>Take 1 capsule, 1 to 2 times daily.</p>"
    ),
    marketplace_bullets=[
        "Helps to maintain proper muscle function, including the heart muscle.",
        "Helps in the development and maintenance of bones and teeth.",
        "Helps to prevent magnesium deficiency.",
        "200 mg of magnesium (magnesium bisglycinate) per vegetable capsule, 1 to 2 times daily.",
        "Vegan, gluten free and non-GMO; a chelated form chosen for tolerability.",
    ],
    social_post_copy=(
        "Tired of counting sheep? Magnesium Bisglycinate 200 mg cures chronic insomnia so you "
        "wake up rested. #sleep #magnesium"
    ),
    image_generation_prompt=(
        "A matte white bottle of magnesium capsules on a bedside table in the evening, a warm "
        "lamp, a folded blanket, calm and quiet. No text overlays, no people."
    ),
    disclaimers_included=[
        "Consult a health care practitioner prior to use if you are pregnant or breastfeeding.",
        "Some people may experience diarrhoea.",
    ],
    npn_displayed="not shown",
)

#: What a violation must mention for the planted draft to count as caught: the cure claim or
#: the licence number, in the rule it names or in the text it quotes.
_CAUGHT = re.compile(
    r"\bcur(?:e|es|ed|ing)\b|\bnpn\b|natural product number|licen[cs]e number", re.IGNORECASE
)


def _synthesis_messages(nonce: str | None = None) -> list[dict[str, str]]:
    return synthesis_messages(
        inventory=INVENTORY,
        target_channels=list(ALL_CHANNELS),
        campaign_angle=NEUTRAL_BRIEF,
        revision_count=0,
        previous_draft=None,
        violations=None,
        revision_instructions=None,
        nonce=nonce or new_nonce(),
    )


def _audit_catches_messages(nonce: str | None = None) -> list[dict[str, str]]:
    return audit_messages(
        draft=PLANTED_DRAFT.model_dump(), inventory=INVENTORY, nonce=nonce or new_nonce()
    )


def _audit_passes_messages(nonce: str | None = None) -> list[dict[str, str]]:
    return audit_messages(
        draft=COMPLIANT_DRAFT.model_dump(), inventory=INVENTORY, nonce=nonce or new_nonce()
    )


def grade_synthesis(parsed: object) -> bool:
    """Right when the draft validates and the Python checks find nothing to fix."""
    return isinstance(parsed, ChannelCopyDraft) and not deterministic_findings(parsed, SKU)


def _as_the_pipeline_sees_it(
    verdict: ComplianceAuditResult, draft: ChannelCopyDraft
) -> tuple[ComplianceAuditResult, ComplianceAuditResult]:
    """The model's findings that survive the overrule step, and the merged audit.

    The same two calls ``compliance_validator`` makes (:mod:`app.graph.nodes`), so a verdict
    is graded as a run would act on it: a wording-only finding the pipeline discards does not
    fail a model, and a finding it discards does not count as a catch.
    """
    findings = deterministic_findings(draft, SKU)
    upheld, _ = overrule_model_findings(verdict, findings, AUTHORIZED_CLAIMS)
    return upheld, merge_audit(upheld, findings)


def grade_audit_catches(parsed: object) -> bool:
    """Right when the model's findings, after the overrule step, name the cure claim or the NPN."""
    if not isinstance(parsed, ComplianceAuditResult):
        return False
    upheld, _ = _as_the_pipeline_sees_it(parsed, PLANTED_DRAFT)
    return any(
        _CAUGHT.search(item.regulation_rule) or _CAUGHT.search(item.violating_text)
        for item in upheld.violations
    )


def grade_audit_passes(parsed: object) -> bool:
    """Right when a compliant draft is found compliant after the overrule step."""
    if not isinstance(parsed, ComplianceAuditResult):
        return False
    _, merged = _as_the_pipeline_sees_it(parsed, COMPLIANT_DRAFT)
    return merged.is_compliant is True


CASES: list[dict[str, Any]] = [
    {
        "id": "synthesis",
        "title": "Write channel copy for NR-MAG-200 that passes the deterministic checks",
        "schema_model": ChannelCopyDraft,
        "messages": _synthesis_messages,
        "temperature": SYNTHESIS_TEMPERATURE,
        "max_tokens": SYNTHESIS_MAX_TOKENS,
        "grade": grade_synthesis,
    },
    {
        "id": "audit-catches",
        "title": "Catch a planted cure claim and a missing NPN",
        "schema_model": ComplianceAuditResult,
        "messages": _audit_catches_messages,
        "temperature": AUDIT_TEMPERATURE,
        "max_tokens": AUDIT_MAX_TOKENS,
        "grade": grade_audit_catches,
    },
    {
        "id": "audit-passes",
        "title": "Find a compliant draft compliant",
        "schema_model": ComplianceAuditResult,
        "messages": _audit_passes_messages,
        "temperature": AUDIT_TEMPERATURE,
        "max_tokens": AUDIT_MAX_TOKENS,
        "grade": grade_audit_passes,
    },
]


def case_ids() -> list[str]:
    """The ids, in the order the cases are sent."""
    return [case["id"] for case in CASES]


def by_id(case_id: str) -> dict[str, Any]:
    """The case with ``case_id``; ``KeyError`` for an id no case has."""
    for case in CASES:
        if case["id"] == case_id:
            return case
    raise KeyError(case_id)


def messages_of(case: dict[str, Any], nonce: str | None = None) -> list[dict[str, str]]:
    """The messages to send for ``case``: fresh copies, with a fresh nonce unless one is given.

    ``messages`` may be a callable (as every case here has) or a ready list, so a test can hold
    a case with fixed messages to the same code.
    """
    source = case["messages"]
    built = source(nonce=nonce) if callable(source) else source
    return [dict(message) for message in built]


def fingerprint_material() -> list[Any]:
    """What the published-chain fingerprint hashes: what a model is sent and checked against.

    Per case: the id, the strict schema the answer must match, the exact messages (with the
    fixed nonce), the temperature and the token limit. Then the product record and the
    prohibited patterns the synthesis grade depends on, and what the two audit grades depend
    on: the words a catch must name, and the overrule step's own figures (the mechanical
    rules, the wording-only fix pattern, the restatement length). Titles are for people and
    stay out: rewording one changes nothing about the test.
    """
    return [
        [
            [
                case["id"],
                strict_json_schema(case["schema_model"]),
                messages_of(case, FINGERPRINT_NONCE),
                case["temperature"],
                case["max_tokens"],
            ]
            for case in CASES
        ],
        INVENTORY,
        [[pattern.rule_id, pattern.regex.pattern] for pattern in PROHIBITED_PATTERNS],
        [
            _CAUGHT.pattern,
            sorted(MECHANICAL_RULES),
            _PEDANTIC_FIX_RE.pattern,
            _RESTATEMENT_WORDS,
        ],
    ]
