"""The regulatory rules the copy is audited against, and the checks Python runs itself.

The Compliance Validator asks a language model for a verdict, but a model can miss a planted
claim or wave an omission through. So the service also checks, deterministically, what can be
checked deterministically: the licence number is present, no prohibited disease or guarantee
language appears, the ingredient amounts stated next to an ingredient match the licence, and
every mandatory risk statement is included word for word. A finding from these checks is a
violation whatever the model said, and it goes back to the synthesis agent as a required fix.

The rules are modelled on Health Canada's framework for Natural Health Products (NHPs): a
licensed product carries an eight-digit Natural Product Number (NPN) that must appear on the
label, may only make the health claims its licence authorizes (structure-function and
traditional-use claims), must carry its risk information (cautions, warnings,
contraindications), and may not be advertised to the general public as a treatment, cure or
preventive for the serious diseases listed in Schedule A of the Food and Drugs Act. This
module encodes those principles for a demo; it is not legal advice.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from app.schemas import (
    ChannelCopyDraft,
    ComplianceAuditResult,
    RuleSummary,
    SkuRecord,
    ViolationItem,
)


@dataclass(frozen=True)
class Rule:
    id: str
    title: str
    summary: str
    severity: str  # low | medium | high


RULES: tuple[Rule, ...] = (
    Rule(
        "NPN_DISPLAY",
        "Show the Natural Product Number",
        "A licensed product is identified by its NPN. The product page and the social post must "
        "show it exactly as licensed, and the draft must report the number it displayed.",
        "high",
    ),
    Rule(
        "AUTHORIZED_CLAIMS_ONLY",
        "Make only the claims the licence authorizes",
        "Every health claim in the copy must be one of the product's authorized claims, or a "
        "plain restatement of one. New benefits, even plausible ones, are not allowed.",
        "high",
    ),
    Rule(
        "NO_DISEASE_CLAIMS",
        "No treatment, cure or prevention claims for diseases",
        "Advertising to the general public may not present the product as a treatment, cure, "
        "preventive or substitute for medical care for a disease (Schedule A of the Food and "
        "Drugs Act lists the serious ones). Words such as cure, treat, heal, reverse, "
        "antibiotic and kills infections are out.",
        "high",
    ),
    Rule(
        "DOSAGE_ACCURACY",
        "State ingredient amounts exactly as licensed",
        "An amount written next to an ingredient must be the licensed amount per serving (or "
        "the total of all strains for a multi-strain product). No rounding up, no new units.",
        "medium",
    ),
    Rule(
        "RISK_INFORMATION",
        "Include every mandatory risk statement verbatim",
        "The cautions and warnings the licence requires must be included word for word in the "
        "draft's disclaimers. Paraphrases do not count.",
        "high",
    ),
    Rule(
        "NO_GUARANTEES",
        "No guarantees, no safety absolutes, no approval claims",
        "Results cannot be guaranteed, the product cannot be called safe for everyone or free "
        "of side effects, 'clinically proven' needs evidence the licence does not provide, and "
        "licensing is not an endorsement or approval by Health Canada.",
        "medium",
    ),
)

RULE_INDEX: dict[str, Rule] = {rule.id: rule for rule in RULES}

_SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2}

#: Diseases and conditions a public advertisement may not claim to treat, cure or prevent.
_DISEASES = (
    r"cancer|tumou?rs?|diabetes|depression|anxiety|insomnia|arthritis|hypertension|"
    r"high blood pressure|covid(?:-19)?|coronavirus|alzheimer'?s?|dementia|heart disease|"
    r"obesity|asthma|infections?|flu|influenza|ibs|crohn'?s?|ulcers?|diseases?|disorders?|"
    r"illness(?:es)?|epilepsy|hepatitis|thrombosis|glaucoma|gangrene|leukemia|nephritis|"
    r"sexual dysfunction|impotence"
)

#: Up to four words may sit between the verb and the disease ("treat the symptoms of cancer").
_GAP = r"(?:\W+\w+){0,4}?\W+"


@dataclass(frozen=True)
class ProhibitedPattern:
    rule_id: str
    label: str
    regex: re.Pattern[str]


def _pattern(rule_id: str, label: str, expression: str) -> ProhibitedPattern:
    return ProhibitedPattern(rule_id, label, re.compile(expression, re.IGNORECASE))


PROHIBITED_PATTERNS: tuple[ProhibitedPattern, ...] = (
    _pattern("NO_DISEASE_CLAIMS", "claims to cure", r"\bcur(?:e|es|ed|ing)\b"),
    _pattern(
        "NO_DISEASE_CLAIMS",
        "claims to treat a disease",
        rf"\b(?:treat|treats|treated|treating|treatment)\b{_GAP}(?:{_DISEASES})\b",
    ),
    _pattern(
        "NO_DISEASE_CLAIMS",
        "claims to prevent, heal or reverse a disease",
        rf"\b(?:prevent|prevents|preventing|prevention|reverse|reverses|reversing|"
        rf"eliminat(?:e|es|ing)|heal|heals|healing|fight|fights|fighting)\b{_GAP}(?:{_DISEASES})\b",
    ),
    _pattern(
        "NO_DISEASE_CLAIMS",
        "antibiotic or antiviral claim",
        r"\b(?:natural\s+)?antibiotics?\b|\bantivirals?\b|\bantifungals?\b",
    ),
    _pattern(
        "NO_DISEASE_CLAIMS",
        "claims to kill pathogens",
        rf"\bkills?\b{_GAP}(?:bacteria|viruses|virus|germs|pathogens|infections?|parasites)\b",
    ),
    _pattern(
        "NO_DISEASE_CLAIMS",
        "presents the product as a substitute for medication",
        rf"\breplac(?:e|es|ing|ement for)\b{_GAP}(?:prescriptions?|medications?|medicines?|"
        r"your doctor|antibiotics?)\b",
    ),
    _pattern("NO_GUARANTEES", "guaranteed results", r"\bguarantee[ds]?\b"),
    _pattern("NO_GUARANTEES", "miracle claim", r"\bmiracle\b|\bmiraculous\b"),
    _pattern(
        "NO_GUARANTEES",
        "safety absolute",
        r"\b100\s?%\s?(?:safe|effective)\b|\bno side[- ]effects?\b|\bside[- ]effect[- ]free\b|"
        r"\bsafe for everyone\b|\bcompletely safe\b",
    ),
    _pattern(
        "NO_GUARANTEES",
        "unsubstantiated proof claim",
        r"\b(?:clinically|scientifically|medically)\s+proven\b",
    ),
    _pattern(
        "NO_GUARANTEES",
        "approval or endorsement claim",
        r"\b(?:fda|health canada|government)[- ]+(?:approved|endorsed|recommended)\b",
    ),
)

#: Fields the pattern scan covers. The disclaimers are licence text and are not scanned;
#: the displayed NPN is a number.
SCANNED_FIELDS = (
    "web_title",
    "web_description_html",
    "marketplace_bullets",
    "social_post_copy",
    "image_generation_prompt",
)

#: Fields that must show the NPN itself (the title and bullets are too short to require it).
NPN_FIELDS = ("web_description_html", "social_post_copy")

_TYPOGRAPHIC = (
    (chr(0x2019), "'"),  # right single quotation mark
    (chr(0x2018), "'"),  # left single quotation mark
    (chr(0x201C), '"'),  # left double quotation mark
    (chr(0x201D), '"'),  # right double quotation mark
    (chr(0x2013), "-"),  # en dash
    (chr(0x2014), "-"),  # em dash
)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s%]")
_AMOUNT_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(billion\s*cfu|billion|mg|mcg|µg|ug|iu|g|ml|%)(?![a-z])",
    re.IGNORECASE,
)
_CONTEXT_CHARS = 60


def normalize_text(text: str) -> str:
    """Fold a text for comparison: NFKC, lower case, HTML tags out, punctuation to spaces."""
    folded = unicodedata.normalize("NFKC", text)
    folded = _TAG_RE.sub(" ", folded)
    # Typographic quotes and dashes, spelled by code point so none of them is in this file.
    for character, plain in _TYPOGRAPHIC:
        folded = folded.replace(character, plain)
    folded = _PUNCT_RE.sub(" ", folded.lower())
    return _WS_RE.sub(" ", folded).strip()


def strip_html(text: str) -> str:
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", text)).strip()


def _snippet(text: str, start: int, end: int, margin: int = 40) -> str:
    left = max(0, start - margin)
    right = min(len(text), end + margin)
    piece = strip_html(text[left:right])
    return ("..." if left > 0 else "") + piece + ("..." if right < len(text) else "")


def rule_summaries() -> list[RuleSummary]:
    return [
        RuleSummary(id=rule.id, title=rule.title, summary=rule.summary, severity=rule.severity)  # type: ignore[arg-type]
        for rule in RULES
    ]


def worst_severity(*levels: str) -> str:
    return max(levels, key=lambda level: _SEVERITY_RANK.get(level, 0)) if levels else "low"


# ------------------------------------------------------------------ the checks


def npn_findings(draft: ChannelCopyDraft, npn: str) -> list[ViolationItem]:
    findings: list[ViolationItem] = []
    rule = RULE_INDEX["NPN_DISPLAY"]
    if npn not in draft.npn_displayed:
        findings.append(
            ViolationItem(
                field_name="npn_displayed",
                violating_text=draft.npn_displayed or "(empty)",
                regulation_rule=rule.title,
                required_fix=f"Report the licensed NPN exactly: NPN {npn}.",
            )
        )
    for field in NPN_FIELDS:
        if npn not in getattr(draft, field):
            findings.append(
                ViolationItem(
                    field_name=field,
                    violating_text=f"The NPN {npn} does not appear in {field}.",
                    regulation_rule=rule.title,
                    required_fix=f"Add the licence number, written as NPN {npn}, to {field}.",
                )
            )
    return findings


def prohibited_findings(draft: ChannelCopyDraft) -> list[ViolationItem]:
    findings: list[ViolationItem] = []
    texts = draft.text_fields()
    for field in SCANNED_FIELDS:
        text = texts[field]
        for pattern in PROHIBITED_PATTERNS:
            for match in pattern.regex.finditer(text):
                rule = RULE_INDEX[pattern.rule_id]
                findings.append(
                    ViolationItem(
                        field_name=field,
                        violating_text=_snippet(text, match.start(), match.end()),
                        regulation_rule=f"{rule.title} ({pattern.label})",
                        required_fix=(
                            f"Remove the wording '{match.group(0)}' and keep to the authorized "
                            "claims and the licensed risk information."
                        ),
                    )
                )
                break  # one finding per pattern per field is enough for the next draft
    return findings


def _ingredient_key(name: str) -> str:
    """The part of an ingredient name a copywriter would use: before any parenthesis or comma."""
    head = re.split(r"[(,]", name, maxsplit=1)[0]
    return normalize_text(head)


def _unit_key(unit: str) -> str:
    folded = unit.lower().replace(" ", "")
    if folded in {"billion", "billioncfu"}:
        return "billion cfu"
    if folded in {"µg", "ug"}:
        return "mcg"
    return folded


def dosage_findings(draft: ChannelCopyDraft, sku: SkuRecord) -> list[ViolationItem]:
    """Amounts written next to an ingredient name must be the licensed ones.

    The copy is scanned in its normalized form (tags out, lower case). For every occurrence of
    an ingredient's name, the amounts within 60 characters before and after are compared with
    the licensed amount for that ingredient, or with the total across ingredients that share a
    unit (a multi-strain probiotic is sold by its total CFU count).
    """
    findings: list[ViolationItem] = []
    rule = RULE_INDEX["DOSAGE_ACCURACY"]
    totals: dict[str, float] = {}
    for ingredient in sku.active_ingredients:
        unit = _unit_key(ingredient.unit)
        totals[unit] = totals.get(unit, 0.0) + float(ingredient.amount)
    texts = draft.text_fields()
    for field in SCANNED_FIELDS:
        text = normalize_text(texts[field])
        # Amounts are found on the whole text first, so a window can never cut a number in
        # half ("200 mg" read as "00 mg"); the window then only decides which amounts count.
        amounts = [
            (match.start(), match.end(), match.group(0), match.group(1), match.group(2))
            for match in _AMOUNT_RE.finditer(text)
        ]
        for ingredient in sku.active_ingredients:
            key = _ingredient_key(ingredient.name)
            if not key:
                continue
            licensed_unit = _unit_key(ingredient.unit)
            allowed = {float(ingredient.amount), totals[licensed_unit]}
            wrong: tuple[str, str] | None = None
            for occurrence in re.finditer(rf"\b{re.escape(key)}\b", text):
                low = occurrence.start() - _CONTEXT_CHARS
                high = occurrence.end() + _CONTEXT_CHARS
                for start, end, written, number, unit_text in amounts:
                    if end < low or start > high:
                        continue
                    unit = _unit_key(unit_text)
                    if unit == "%":
                        continue  # percentages describe standardization, not a dose
                    value = float(number.replace(",", "."))
                    if unit != licensed_unit or value not in allowed:
                        wrong = (key, written)
                        break
                if wrong:
                    break
            if wrong:
                findings.append(
                    ViolationItem(
                        field_name=field,
                        violating_text=f"{wrong[0]}: {wrong[1]}",
                        regulation_rule=rule.title,
                        required_fix=(
                            f"State {ingredient.name} as {ingredient.amount:g} "
                            f"{ingredient.unit} per {ingredient.per}, as licensed."
                        ),
                    )
                )
    return findings


def risk_statement_findings(draft: ChannelCopyDraft, sku: SkuRecord) -> list[ViolationItem]:
    findings: list[ViolationItem] = []
    rule = RULE_INDEX["RISK_INFORMATION"]
    included = " | ".join(normalize_text(item) for item in draft.disclaimers_included)
    for statement in sku.risk_information:
        if statement.mandatory and normalize_text(statement.text) not in included:
            findings.append(
                ViolationItem(
                    field_name="disclaimers_included",
                    violating_text=f"Missing: {statement.text}",
                    regulation_rule=rule.title,
                    required_fix=f"Add this statement word for word: {statement.text}",
                )
            )
    return findings


def deterministic_findings(
    draft: ChannelCopyDraft, sku: SkuRecord | dict[str, Any]
) -> list[ViolationItem]:
    """Every violation the Python checks can prove, in rule order."""
    record = sku if isinstance(sku, SkuRecord) else SkuRecord.model_validate(sku)
    findings = npn_findings(draft, record.npn_number)
    findings += prohibited_findings(draft)
    findings += dosage_findings(draft, record)
    findings += risk_statement_findings(draft, record)
    return findings


def _finding_severity(violation: ViolationItem) -> str:
    for rule in RULES:
        if violation.regulation_rule.startswith(rule.title):
            return rule.severity
    return "medium"


#: Rules the deterministic checks decide. A model's finding under one of them stands only when
#: the code found a problem in the same field: the licence number is present or it is not, an
#: amount matches or it does not, a mandatory statement is there word for word or it is not.
MECHANICAL_RULES = frozenset({"NPN_DISPLAY", "DOSAGE_ACCURACY", "RISK_INFORMATION"})

#: A fix that is about form, not substance. Measured on 15 October 2026: a model that answers
#: without reasoning files restated authorized claims as violations for a missing period, for
#: being shortened, or for being combined, in one audit out of three on a compliant draft.
#: Everyday words a fix about a real violation also uses ("omit the cure claim", "use the
#: exact claim", "the complete sentence") are not in it: the quoted text, not the fix, is what
#: decides whether a finding is about wording (see ``_restates_authorized_claim``).
_PEDANTIC_FIX_RE = re.compile(
    r"\b(?:periods?|punctuat\w*|verbatim|word for word|split|wording|capitali[sz]\w*|"
    r"match the authorized|full|restore|shorten\w*|truncat\w*|partial|rest of the claim)\b",
    re.IGNORECASE,
)


def rule_id_of(violation: ViolationItem) -> str | None:
    """Which rule a violation names: by id anywhere in the text, or by the rule's title."""
    text = violation.regulation_rule
    upper = text.upper()
    for rule in RULES:
        if rule.id in upper:
            return rule.id
    for rule in RULES:
        if text.startswith(rule.title):
            return rule.id
    return None


#: A quoted text restates a claim when it is made of that claim's words and nothing else, and
#: shares at least this many of them. Four words ("helps in energy metabolism") is the shortest
#: honest restatement of a claim. A quote that adds one word the licence does not contain
#: ("relieves arthritis pain", "supports joint comfort", "melts belly fat") is never a
#: restatement, whatever the fix says, so a new or strengthened benefit built on the same
#: opening words stays a violation.
_RESTATEMENT_WORDS = 4


def _restates_authorized_claim(text: str, authorized_claims: list[str]) -> bool:
    """True when the quoted text sits inside an authorized claim, or is made of its words only.

    A model quotes a claim without its tail, or with its words reordered, and calls that a
    violation; neither changes the claim. Every word of the quote must occur in one claim and
    the quote may not be longer than that claim: a quote with a word of its own is a claim of
    its own.
    """
    quoted = normalize_text(text)
    if not quoted:
        return False
    quoted_words = quoted.split()
    for claim in authorized_claims:
        folded = normalize_text(claim)
        if not folded:
            continue
        if quoted in folded:
            return True
        words = folded.split()
        if (
            len(quoted_words) >= min(_RESTATEMENT_WORDS, len(words))
            and len(quoted_words) <= len(words)
            and set(quoted_words) <= set(words)
        ):
            return True
    return False


def overrule_model_findings(
    llm_audit: ComplianceAuditResult,
    findings: list[ViolationItem],
    authorized_claims: list[str] | None = None,
) -> tuple[ComplianceAuditResult, list[dict[str, str]]]:
    """Drop the model's findings that the deterministic checks or the rules overrule.

    Two kinds go: a finding under a mechanical rule in a field where the code found nothing
    (the code is the authority on those), and a wording-only finding under
    AUTHORIZED_CLAIMS_ONLY whose quoted text is made of an authorized claim's own words
    (punctuation and rewording are not violations). Returns the audit without them and the
    list of what was discarded, with the reason, for the trace. A disease, guarantee or
    new-benefit finding is never touched here: a quote with one word the licence does not
    contain is kept whatever the fix says.
    """
    code_found = {(f.field_name, rule_id_of(f)) for f in findings}
    kept: list[ViolationItem] = []
    discarded: list[dict[str, str]] = []
    for violation in llm_audit.violations:
        rule_id = rule_id_of(violation)
        if rule_id in MECHANICAL_RULES and (violation.field_name, rule_id) not in code_found:
            discarded.append(
                {
                    "field_name": violation.field_name,
                    "regulation_rule": violation.regulation_rule,
                    "violating_text": violation.violating_text,
                    "reason": "decided by the deterministic check, which found nothing in this field",
                }
            )
            continue
        if (
            rule_id == "AUTHORIZED_CLAIMS_ONLY"
            and authorized_claims
            and _PEDANTIC_FIX_RE.search(violation.required_fix)
            and _restates_authorized_claim(violation.violating_text, authorized_claims)
        ):
            discarded.append(
                {
                    "field_name": violation.field_name,
                    "regulation_rule": violation.regulation_rule,
                    "violating_text": violation.violating_text,
                    "reason": "the quoted text restates an authorized claim; wording and "
                    "punctuation are not violations",
                }
            )
            continue
        kept.append(violation)
    if not discarded:
        return llm_audit, []
    summary = llm_audit.revision_summary
    if not kept:
        summary = "The model's findings were all overruled by the deterministic checks."
    return (
        ComplianceAuditResult(
            is_compliant=not kept,
            risk_level=llm_audit.risk_level if kept else "low",
            violations=kept,
            revision_summary=summary,
        ),
        discarded,
    )


def merge_audit(
    llm_audit: ComplianceAuditResult, findings: list[ViolationItem]
) -> ComplianceAuditResult:
    """Combine the model's verdict with the deterministic findings. Findings always win."""
    seen = {
        (v.field_name, v.regulation_rule, normalize_text(v.violating_text))
        for v in llm_audit.violations
    }
    added: list[ViolationItem] = []
    for finding in findings:
        key = (finding.field_name, finding.regulation_rule, normalize_text(finding.violating_text))
        if key not in seen:
            seen.add(key)
            added.append(finding)
    violations = list(llm_audit.violations) + added
    risk = llm_audit.risk_level
    for finding in added:
        risk = worst_severity(risk, _finding_severity(finding))  # type: ignore[assignment]
    summary = llm_audit.revision_summary
    if added:
        summary = (
            f"{summary.rstrip('.')}. Deterministic checks added {len(added)} finding(s): "
            + "; ".join(f"{v.field_name}: {v.regulation_rule}" for v in added)
            + "."
        )
    return ComplianceAuditResult(
        is_compliant=llm_audit.is_compliant and not violations,
        risk_level=risk,  # type: ignore[arg-type]
        violations=violations,
        revision_summary=summary[:2000],
    )


def format_revision_instructions(audit: ComplianceAuditResult, revision: int) -> str:
    """The feedback the synthesis agent receives for its next attempt."""
    lines = [
        f"Revision {revision}: the previous draft failed the compliance audit (risk {audit.risk_level})."
    ]
    lines.append(audit.revision_summary)
    for number, violation in enumerate(audit.violations, start=1):
        lines.append(
            f"{number}. [{violation.field_name}] {violation.regulation_rule}: "
            f'"{violation.violating_text}" -> {violation.required_fix}'
        )
    return "\n".join(lines)
