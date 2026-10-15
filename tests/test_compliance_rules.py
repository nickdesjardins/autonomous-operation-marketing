"""The deterministic compliance checks: what they catch, and what they must leave alone.

A false positive sends a compliant draft back for another model call, and a false negative
lets a disease claim reach the human gate with a clean report. Both are tested here against
the seed's own licence text: every authorized claim, risk statement and contraindication of
every SKU must pass every prohibited pattern.
"""

from __future__ import annotations

import pytest

from app.compliance.rules import (
    PROHIBITED_PATTERNS,
    RULES,
    deterministic_findings,
    dosage_findings,
    format_revision_instructions,
    merge_audit,
    normalize_text,
    npn_findings,
    overrule_model_findings,
    prohibited_findings,
    risk_statement_findings,
    rule_summaries,
)
from app.schemas import ComplianceAuditResult, SkuRecord, ViolationItem
from app.seed import seed_records

from .conftest import NPN, SKU, bad_draft, good_draft

#: One sentence per pattern label. A pattern added without a sentence here fails the suite.
TRIGGERS: dict[str, str] = {
    "claims to cure": "This capsule cures headaches in a week.",
    "claims to treat a disease": "Used to treat the symptoms of arthritis.",
    "claims to prevent, heal or reverse a disease": "Taken daily it helps prevent the flu.",
    "antibiotic or antiviral claim": "A natural antibiotic for the whole family.",
    "claims to kill pathogens": "Kills harmful bacteria on contact.",
    "presents the product as a substitute for medication": (
        "A gentle replacement for your prescriptions."
    ),
    "guaranteed results": "Results guaranteed or your money back.",
    "miracle claim": "A miracle in a bottle.",
    "safety absolute": "Completely safe, with no side effects.",
    "unsubstantiated proof claim": "Clinically proven to work.",
    "approval or endorsement claim": "A Health Canada approved formula.",
}


def _sku(code: str = SKU) -> SkuRecord:
    return next(record for record in seed_records() if record.sku_code == code)


def _licence_texts(record: SkuRecord) -> list[str]:
    texts = list(record.authorized_claims)
    texts += [statement.text for statement in record.risk_information]
    if record.contraindications:
        texts.append(record.contraindications)
    return texts


# ------------------------------------------------------------------ the sample drafts


def test_the_good_draft_passes_every_deterministic_check() -> None:
    assert deterministic_findings(good_draft(), _sku()) == []
    assert deterministic_findings(good_draft(), _sku().model_dump()) == []  # dict form too


def test_the_bad_draft_is_caught_for_the_cure_claim_and_the_missing_npn() -> None:
    findings = deterministic_findings(bad_draft(), _sku())
    rules = {finding.regulation_rule for finding in findings}
    fields = {finding.field_name for finding in findings}
    assert any("claims to cure" in rule for rule in rules)
    assert "Show the Natural Product Number" in rules
    assert {"npn_displayed", "web_description_html", "social_post_copy"} <= fields
    assert all("Include every mandatory risk statement" not in rule for rule in rules)
    assert all("State ingredient amounts" not in rule for rule in rules)


# -------------------------------------------------------------- prohibited patterns


def test_every_pattern_has_a_trigger_sentence() -> None:
    assert {pattern.label for pattern in PROHIBITED_PATTERNS} == set(TRIGGERS)


@pytest.mark.parametrize("pattern", PROHIBITED_PATTERNS, ids=lambda p: p.label)
def test_each_pattern_fires_on_its_sentence(pattern) -> None:
    assert pattern.regex.search(TRIGGERS[pattern.label]), pattern.label
    assert pattern.rule_id in {rule.id for rule in RULES}


@pytest.mark.parametrize("record", seed_records(), ids=lambda r: r.sku_code)
def test_no_pattern_fires_on_the_licence_text_of_any_sku(record: SkuRecord) -> None:
    for text in _licence_texts(record):
        for pattern in PROHIBITED_PATTERNS:
            assert not pattern.regex.search(text), f"{pattern.label!r} matched {text!r}"


def test_prohibited_findings_name_the_field_and_quote_a_snippet() -> None:
    draft = good_draft(
        social_post_copy="Magnesium cures insomnia and is guaranteed to work. NPN 80098765",
    )
    findings = prohibited_findings(draft)
    assert [(f.field_name, f.regulation_rule) for f in findings] == [
        (
            "social_post_copy",
            "No treatment, cure or prevention claims for diseases (claims to cure)",
        ),
        (
            "social_post_copy",
            "No guarantees, no safety absolutes, no approval claims (guaranteed results)",
        ),
    ]
    assert "cures" in findings[0].violating_text and "'cures'" in findings[0].required_fix
    # One finding per pattern per field: a second "cures" in the same field adds nothing.
    twice = good_draft(social_post_copy="It cures colds and cures flu. NPN 80098765")
    assert len(prohibited_findings(twice)) == 1


def test_disclaimers_and_the_npn_field_are_not_scanned() -> None:
    draft = good_draft(
        disclaimers_included=["Not a cure for anything.", *good_draft().disclaimers_included]
    )
    assert prohibited_findings(draft) == []


# -------------------------------------------------------------------- the NPN


def test_npn_checks_cover_the_reported_number_and_both_text_fields() -> None:
    assert npn_findings(good_draft(), NPN) == []
    missing = npn_findings(bad_draft(), NPN)
    assert [finding.field_name for finding in missing] == [
        "npn_displayed",
        "web_description_html",
        "social_post_copy",
    ]
    assert missing[0].violating_text == "not shown"
    assert all(f"NPN {NPN}" in finding.required_fix for finding in missing)

    only_reported = good_draft(
        web_description_html=(
            "<p>A fine product for every day. Helps to maintain proper muscle function, including "
            "the heart muscle. One vegetable capsule, one to two times daily, with food.</p>"
        )
    )
    assert [f.field_name for f in npn_findings(only_reported, NPN)] == ["web_description_html"]
    assert npn_findings(good_draft(npn_displayed=NPN), NPN) == []  # the bare number counts
    assert len(npn_findings(good_draft(), "80000000")) == 3  # the wrong licence number


# ------------------------------------------------------------------------- dosage


def test_a_wrong_amount_next_to_the_ingredient_is_a_finding_and_the_licensed_one_is_not() -> None:
    sku = _sku()
    wrong = good_draft(web_title="Magnesium Bisglycinate 400 mg for the whole family")
    (finding,) = dosage_findings(wrong, sku)
    assert finding.field_name == "web_title"
    assert finding.violating_text == "magnesium: 400 mg"
    assert "200 mg per capsule" in finding.required_fix
    assert dosage_findings(good_draft(web_title="Magnesium Bisglycinate 200 mg"), sku) == []
    # A unit the licence does not use is wrong even with the right number.
    assert len(dosage_findings(good_draft(web_title="Magnesium 200 mcg"), sku)) == 1
    # An amount far from any ingredient name is not about the ingredient.
    far = good_draft(web_title="A 500 mg claim about something else entirely, nothing more")
    assert dosage_findings(far, sku) == []


def test_a_multi_strain_total_is_allowed_next_to_a_strain() -> None:
    sku = _sku("NR-PRO-50B")
    total = good_draft(web_title="Lactobacillus acidophilus and friends, 50 billion CFU")
    assert dosage_findings(total, sku) == []
    own = good_draft(web_title="Lactobacillus acidophilus 20 billion CFU per capsule")
    assert dosage_findings(own, sku) == []
    wrong = good_draft(web_title="Lactobacillus acidophilus 30 billion CFU per capsule")
    (finding,) = dosage_findings(wrong, sku)
    assert finding.violating_text == "lactobacillus acidophilus: 30 billion cfu"


def test_a_standardization_percentage_is_not_a_dose() -> None:
    sku = _sku("NR-ORE-C80")
    fine = good_draft(web_title="Oregano oil standardized to 80% carvacrol, 100 mg per 4 drops")
    assert dosage_findings(fine, sku) == []
    wrong = good_draft(web_title="Oregano oil, 150 mg per serving")
    assert len(dosage_findings(wrong, sku)) == 1


def test_the_context_window_never_cuts_a_number_in_half() -> None:
    # Regression: with amounts read inside the 60-character window around the ingredient, the
    # window around "#magnesium" started inside "200 mg" and read "0 mg", a wrong dose.
    social = "Take 200 mg with dinner for calm evenings, every night. NPN 80098765 #magnesium"
    folded = normalize_text(social)
    amount, occurrence = folded.index("200 mg"), folded.rindex("magnesium")
    assert amount < occurrence - 60 < amount + 3, "the window edge must fall inside the number"
    assert dosage_findings(good_draft(social_post_copy=social), _sku()) == []


# ---------------------------------------------------------------- risk statements


def test_mandatory_statements_are_matched_after_normalization() -> None:
    sku = _sku()
    styled = good_draft(
        disclaimers_included=[
            "<em>“Consult a health care practitioner prior to use if you are pregnant or "
            "breastfeeding.”</em>",
            "Some  people may experience\ndiarrhoea.",
        ]
    )
    assert risk_statement_findings(styled, sku) == []


def test_a_paraphrase_or_an_omission_is_a_finding_but_optional_statements_are_not() -> None:
    sku = _sku()
    paraphrased = good_draft(
        disclaimers_included=[
            "Consult a health care practitioner prior to use if you are pregnant or breastfeeding.",
            "Some people might experience diarrhoea.",
        ]
    )
    (finding,) = risk_statement_findings(paraphrased, sku)
    assert finding.field_name == "disclaimers_included"
    assert finding.violating_text == "Missing: Some people may experience diarrhoea."
    assert finding.required_fix.endswith("Some people may experience diarrhoea.")
    nothing = good_draft(disclaimers_included=["Store in a cool, dry place."])
    assert len(risk_statement_findings(nothing, sku)) == 2
    # "Keep out of reach of children." is not mandatory for this SKU: its absence is fine.
    assert risk_statement_findings(good_draft(), sku) == []


def test_normalize_text_folds_tags_quotes_dashes_and_case() -> None:
    assert normalize_text("<p>Don’t  “Worry” \u2013 it’s FINE.</p>") == "don t worry it s fine"
    assert normalize_text("Ｍagnesium 200 mg") == "magnesium 200 mg"  # NFKC: full-width letters


# ------------------------------------------------------------------- merge_audit


def _npn_finding() -> ViolationItem:
    return npn_findings(bad_draft(), NPN)[0]


def test_findings_always_win_over_a_lenient_verdict() -> None:
    lenient = ComplianceAuditResult(
        is_compliant=True, risk_level="low", violations=[], revision_summary="All good."
    )
    merged = merge_audit(lenient, [_npn_finding()])
    assert merged.is_compliant is False
    assert merged.violations == [_npn_finding()]
    assert merged.risk_level == "high"  # the NPN rule is a high-severity rule
    assert merged.revision_summary == (
        "All good. Deterministic checks added 1 finding(s): npn_displayed: Show the Natural "
        "Product Number."
    )


def test_merge_dedupes_what_the_auditor_already_found_and_keeps_its_risk() -> None:
    finding = _npn_finding()
    already = ComplianceAuditResult(
        is_compliant=False,
        risk_level="medium",
        violations=[finding.model_copy(update={"violating_text": finding.violating_text.upper()})],
        revision_summary="The NPN is missing.",
    )
    merged = merge_audit(already, [finding])
    assert len(merged.violations) == 1
    assert merged.risk_level == "medium"  # nothing was added, so nothing escalates
    assert merged.revision_summary == "The NPN is missing."


def test_merge_without_findings_keeps_the_verdict_untouched() -> None:
    verdict = ComplianceAuditResult(
        is_compliant=True, risk_level="low", violations=[], revision_summary="Clean."
    )
    assert merge_audit(verdict, []) == verdict


def test_merge_counts_every_added_finding_and_escalates_to_the_worst() -> None:
    dosage = ViolationItem(
        field_name="web_title",
        violating_text="magnesium: 400 mg",
        regulation_rule="State ingredient amounts exactly as licensed",
        required_fix="State 200 mg.",
    )
    verdict = ComplianceAuditResult(
        is_compliant=True, risk_level="low", violations=[], revision_summary="Fine"
    )
    only_dosage = merge_audit(verdict, [dosage])
    assert only_dosage.risk_level == "medium"
    both = merge_audit(verdict, [dosage, _npn_finding()])
    assert both.risk_level == "high"
    assert "added 2 finding(s)" in both.revision_summary
    assert "web_title: State ingredient amounts exactly as licensed" in both.revision_summary


# ----------------------------------------------------- overrule_model_findings


def _model_finding(
    field_name: str, rule: str, text: str, fix: str = "Fix it."
) -> ComplianceAuditResult:
    return ComplianceAuditResult(
        is_compliant=False,
        risk_level="medium",
        violations=[
            ViolationItem(
                field_name=field_name, violating_text=text, regulation_rule=rule, required_fix=fix
            )
        ],
        revision_summary="One finding.",
    )


def _claims() -> list[str]:
    return list(_sku().authorized_claims)


@pytest.mark.parametrize(
    "fix",
    [
        "Delete 'relieves arthritis pain' and keep the authorized claim wording only.",
        "Remove the arthritis claim and use the complete authorized claim.",
        "Keep to the exact authorized claim.",
        "Restore the full authorized claim.",
        "Omit 'relieves arthritis pain'.",
        "Remove the arthritis claim: the licence does not authorize it.",
    ],
)
def test_a_quote_with_a_word_the_licence_does_not_contain_survives_whatever_the_fix_says(
    fix: str,
) -> None:
    quotes = [
        "helps to maintain proper muscle function and relieves arthritis pain",
        "Helps in the development and maintenance of bones and teeth and supports joint comfort",
        "Helps in energy metabolism and melts belly fat in two weeks",
        "Dramatically boosts energy metabolism and tissue formation within days",
    ]
    for quote in quotes:
        audit = _model_finding("web_description_html", "AUTHORIZED_CLAIMS_ONLY", quote, fix)
        upheld, discarded = overrule_model_findings(audit, [], _claims())
        assert discarded == [] and upheld.violations == audit.violations, quote
        assert merge_audit(upheld, []).is_compliant is False, quote


def test_a_wording_only_finding_on_a_restated_claim_is_discarded() -> None:
    for quote in (
        "helps to maintain proper muscle function",  # the claim without its tail
        "Helps in energy metabolism.",  # a sentence of its own, with a period
        "maintenance of bones and teeth and development",  # the claim's words, reordered
    ):
        audit = _model_finding(
            "marketplace_bullets",
            "AUTHORIZED_CLAIMS_ONLY",
            quote,
            "Restore the full claim wording.",
        )
        upheld, discarded = overrule_model_findings(audit, [], _claims())
        assert upheld.violations == [] and upheld.is_compliant is True, quote
        assert len(discarded) == 1 and "restates an authorized claim" in discarded[0]["reason"]
    # The same quote with a fix about substance is the model's call and stays.
    substance = _model_finding(
        "marketplace_bullets",
        "AUTHORIZED_CLAIMS_ONLY",
        "helps to maintain proper muscle function",
        "This claim is not authorized for this product.",
    )
    assert overrule_model_findings(substance, [], _claims())[1] == []


def test_a_mechanical_finding_stands_only_where_the_code_found_a_problem() -> None:
    npn = _npn_finding()  # the code found the NPN missing in npn_displayed
    elsewhere = _model_finding("social_post_copy", "NPN_DISPLAY", "no licence number")
    upheld, discarded = overrule_model_findings(elsewhere, [npn], _claims())
    assert upheld.violations == [] and discarded[0]["field_name"] == "social_post_copy"
    assert "deterministic check" in discarded[0]["reason"]
    same_field = _model_finding("npn_displayed", "NPN_DISPLAY", "not shown")
    assert overrule_model_findings(same_field, [npn], _claims())[1] == []
    # A disease finding is never touched, whatever the fix says.
    disease = _model_finding(
        "web_title", "NO_DISEASE_CLAIMS", "cures arthritis", "Use the exact wording."
    )
    assert overrule_model_findings(disease, [], _claims())[1] == []
    # Nothing discarded: the very same audit object comes back.
    assert overrule_model_findings(disease, [], _claims())[0] is disease


# -------------------------------------------------------- revision instructions


def test_format_revision_instructions_numbers_every_violation() -> None:
    audit = merge_audit(
        ComplianceAuditResult(
            is_compliant=True, risk_level="low", violations=[], revision_summary="Fine"
        ),
        deterministic_findings(bad_draft(), _sku()),
    )
    text = format_revision_instructions(audit, revision=2)
    lines = text.split("\n")
    assert lines[0] == "Revision 2: the previous draft failed the compliance audit (risk high)."
    assert lines[1] == audit.revision_summary
    numbered = lines[2:]
    assert len(numbered) == len(audit.violations)
    for number, (line, violation) in enumerate(zip(numbered, audit.violations, strict=True), 1):
        assert line.startswith(f"{number}. [{violation.field_name}] {violation.regulation_rule}: ")
        assert f'"{violation.violating_text}" -> {violation.required_fix}' in line


def test_rule_summaries_match_the_catalogue() -> None:
    summaries = rule_summaries()
    assert [summary.id for summary in summaries] == [rule.id for rule in RULES]
    assert len(summaries) == 6
    assert {summary.severity for summary in summaries} == {"high", "medium"}
