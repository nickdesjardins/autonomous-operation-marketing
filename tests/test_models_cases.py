"""Tests for the three cases a free model is tested on (``app.models.cases``).

The cases are built from the sample inventory and the real prompts, and graded by the
project's own compliance checks. Nothing here calls a model.
"""

from __future__ import annotations

import json

from app.compliance.rules import deterministic_findings
from app.config import Settings
from app.graph.prompts import AUDIT_SYSTEM_PROMPT, SYNTHESIS_SYSTEM_PROMPT
from app.models import cases
from app.schemas import ChannelCopyDraft, ComplianceAuditResult, ViolationItem


def audit(compliant: bool, *violations: ViolationItem) -> ComplianceAuditResult:
    return ComplianceAuditResult(
        is_compliant=compliant,
        risk_level="low" if compliant else "high",
        violations=list(violations),
        revision_summary="Summary for the copywriter.",
    )


def violation(
    rule: str, text: str, field: str = "web_description_html", fix: str = "Fix it."
) -> ViolationItem:
    return ViolationItem(
        field_name=field, violating_text=text, regulation_rule=rule, required_fix=fix
    )


# ---------------------------------------------------------------------------
# What the cases are made of
# ---------------------------------------------------------------------------


def test_three_cases_in_a_fixed_order_with_the_fields_a_test_needs() -> None:
    assert cases.case_ids() == ["synthesis", "audit-catches", "audit-passes"]
    for case in cases.CASES:
        assert set(case) >= {
            "id",
            "title",
            "schema_model",
            "messages",
            "temperature",
            "max_tokens",
            "grade",
        }
        assert callable(case["messages"]) and callable(case["grade"])
        assert case["max_tokens"] >= 256 and 0.0 <= case["temperature"] <= 1.0
        assert cases.by_id(case["id"]) is case
    assert cases.by_id("synthesis")["schema_model"] is ChannelCopyDraft
    assert cases.by_id("audit-passes")["schema_model"] is ComplianceAuditResult


def test_the_cases_send_the_settings_own_defaults_for_the_two_nodes() -> None:
    fields = Settings.model_fields
    synthesis = cases.by_id("synthesis")
    assert synthesis["max_tokens"] == fields["synthesis_max_tokens"].default
    assert synthesis["temperature"] == fields["synthesis_temperature"].default
    for case_id in ("audit-catches", "audit-passes"):
        case = cases.by_id(case_id)
        assert case["max_tokens"] == fields["audit_max_tokens"].default
        assert case["temperature"] == fields["audit_temperature"].default


def test_the_sample_sku_is_the_seeds_own_and_in_stock() -> None:
    assert cases.SKU.sku_code == "NR-MAG-200" and cases.SKU.stock_quantity > 0
    assert cases.INVENTORY["npn_number"] == cases.SKU.npn_number


def test_the_compliant_draft_has_no_deterministic_finding_and_the_planted_one_has_several() -> None:
    assert deterministic_findings(cases.COMPLIANT_DRAFT, cases.SKU) == []
    planted = deterministic_findings(cases.PLANTED_DRAFT, cases.SKU)
    rules = {finding.regulation_rule.split(" (")[0] for finding in planted}
    assert rules == {
        "Show the Natural Product Number",
        "No treatment, cure or prevention claims for diseases",
    }
    fields = {finding.field_name for finding in planted}
    assert fields >= {"web_description_html", "social_post_copy", "npn_displayed"}


def test_the_planted_draft_plants_what_it_says_it_plants() -> None:
    assert "cures chronic insomnia" in cases.PLANTED_DRAFT.web_description_html
    assert "cures chronic insomnia" in cases.PLANTED_DRAFT.social_post_copy
    for text in cases.PLANTED_DRAFT.text_fields().values():
        assert cases.SKU.npn_number not in text


def test_the_messages_are_the_real_prompts_over_the_sample_sku() -> None:
    synthesis = cases.messages_of(cases.by_id("synthesis"))
    assert synthesis[0] == {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT}
    assert cases.NEUTRAL_BRIEF in synthesis[1]["content"]
    assert '"sku_code": "NR-MAG-200"' in synthesis[1]["content"]
    catches = cases.messages_of(cases.by_id("audit-catches"))
    assert catches[0] == {"role": "system", "content": AUDIT_SYSTEM_PROMPT}
    assert "cures chronic insomnia" in catches[1]["content"]
    passes = cases.messages_of(cases.by_id("audit-passes"))
    assert "NPN 80098765" in passes[1]["content"] and "cures" not in passes[1]["content"]


def test_every_call_gets_a_fresh_nonce_and_fresh_copies() -> None:
    case = cases.by_id("synthesis")
    first, second = cases.messages_of(case), cases.messages_of(case)
    assert first != second  # the nonce in the brief's tags differs
    assert first[0] == second[0]  # the system prompt is the same, byte for byte
    first[0]["content"] = "changed"
    assert cases.messages_of(case)[0]["content"] == SYNTHESIS_SYSTEM_PROMPT
    fixed = cases.messages_of(case, nonce="abcd1234")
    assert fixed == cases.messages_of(case, nonce="abcd1234")
    assert "campaign_brief_abcd1234" in fixed[1]["content"]


def test_a_case_may_hold_ready_messages_instead_of_a_callable() -> None:
    ready = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    case = {**cases.by_id("synthesis"), "messages": ready}
    copies = cases.messages_of(case)
    assert copies == ready and copies[0] is not ready[0]


# ---------------------------------------------------------------------------
# The grades
# ---------------------------------------------------------------------------


def test_synthesis_passes_only_a_draft_the_checks_find_nothing_wrong_with() -> None:
    grade = cases.by_id("synthesis")["grade"]
    assert grade(cases.COMPLIANT_DRAFT) is True
    assert grade(cases.PLANTED_DRAFT) is False
    assert grade(audit(True)) is False  # the wrong schema altogether
    assert grade(None) is False


def test_the_audit_must_catch_the_cure_claim_or_the_missing_npn() -> None:
    grade = cases.by_id("audit-catches")["grade"]
    assert grade(audit(False, violation("NO_DISEASE_CLAIMS", "cures chronic insomnia"))) is True
    assert (
        grade(audit(False, violation("Show the Natural Product Number", "no licence number")))
        is True
    )
    assert grade(audit(False, violation("NPN_DISPLAY", "The NPN does not appear"))) is True
    assert grade(audit(False, violation("Licence number missing", "The copy shows no NPN"))) is True
    # Not compliant, but for the wrong reason: the planted problems were not seen.
    assert grade(audit(False, violation("DOSAGE_ACCURACY", "200 mg"))) is False
    assert grade(audit(True)) is False
    assert grade(cases.COMPLIANT_DRAFT) is False
    # The pipeline discards an NPN finding in a field where the code found the number shown
    # (the bullets never have to show it), so that finding is no catch.
    bullets_only = violation("NPN_DISPLAY", "The bullets show no NPN", field="marketplace_bullets")
    assert grade(audit(False, bullets_only)) is False


def test_a_compliant_draft_must_be_found_compliant() -> None:
    grade = cases.by_id("audit-passes")["grade"]
    assert grade(audit(True)) is True
    assert grade(audit(False, violation("NO_DISEASE_CLAIMS", "anything"))) is False
    # The schema itself turns "compliant with violations" into "not compliant".
    assert grade(audit(True, violation("NO_DISEASE_CLAIMS", "anything"))) is False
    assert grade(None) is False
    # A wording-only finding on a restated claim is discarded by the pipeline, so a verdict
    # that carries nothing else is graded as the run would act on it: compliant.
    wording = violation(
        "AUTHORIZED_CLAIMS_ONLY",
        "helps to maintain proper muscle function",
        field="social_post_copy",
        fix="Restore the full authorized claim wording.",
    )
    assert grade(audit(False, wording)) is True
    # A new benefit on the same opening words is kept, whatever the fix says.
    benefit = violation(
        "AUTHORIZED_CLAIMS_ONLY",
        "helps to maintain proper muscle function and relieves arthritis pain",
        field="social_post_copy",
        fix="Restore the full authorized claim wording.",
    )
    assert grade(audit(False, benefit)) is False


# ---------------------------------------------------------------------------
# The fingerprint material
# ---------------------------------------------------------------------------


def test_the_fingerprint_material_is_the_same_from_one_call_to_the_next() -> None:
    first = json.dumps(cases.fingerprint_material(), sort_keys=True)
    second = json.dumps(cases.fingerprint_material(), sort_keys=True)
    assert first == second
    assert f"campaign_brief_{cases.FINGERPRINT_NONCE}" in first
    # Titles are for people and change nothing about the test.
    assert cases.by_id("synthesis")["title"] not in first


def test_the_fingerprint_material_holds_what_a_model_is_sent_and_checked_against() -> None:
    material = cases.fingerprint_material()
    per_case, inventory, patterns, grading = material
    assert [item[0] for item in per_case] == cases.case_ids()
    for case_id, schema, messages, temperature, max_tokens in per_case:
        case = cases.by_id(case_id)
        assert schema["additionalProperties"] is False and "required" in schema
        assert messages == cases.messages_of(case, cases.FINGERPRINT_NONCE)
        assert (temperature, max_tokens) == (case["temperature"], case["max_tokens"])
    assert inventory == cases.INVENTORY
    assert all(len(item) == 2 for item in patterns) and len(patterns) >= 5
    # What the audit grades depend on: the catch pattern and the overrule step's figures.
    caught, mechanical, pedantic, restatement = grading
    assert "cur" in caught and "npn" in caught
    assert mechanical == sorted(mechanical) and "NPN_DISPLAY" in mechanical
    assert "wording" in pedantic and isinstance(restatement, int) and restatement >= 2
