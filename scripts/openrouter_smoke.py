"""Live check of the zero-cost model path: OpenRouter free models through the service's own client.

Spends 2 free requests from the account's daily allowance (50 a day without credits).

    python scripts/openrouter_smoke.py            # reads OPENROUTER_API_KEY from .env / .env.local

What it prints:
1. which models of the chain in use OpenRouter's public catalogue still lists (no key needed);
2. the account's free-request allowance for today (GET /key, the one request that sends the key);
3. one synthesis and one audit for the sample SKU NR-MAG-200, with the model that answered, the
   deterministic findings on the draft, and the auditor's verdict on the planted draft.

Settings are read exactly as the service reads them (app/config.py: the environment, then .env
and .env.local). The key is never printed.
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# Settings read .env and .env.local relative to the working directory, as the service does.
os.chdir(ROOT)

from app.budget import read_openrouter_allowance  # noqa: E402
from app.compliance.rules import deterministic_findings  # noqa: E402
from app.config import Settings  # noqa: E402
from app.models import cases, catalogue  # noqa: E402
from app.schemas import ChannelCopyDraft, ComplianceAuditResult  # noqa: E402
from app.services.llm_client import build_chat  # noqa: E402


async def main() -> int:
    settings = Settings(llm_provider="openrouter")
    if settings.openrouter_api_key is None:
        print("OPENROUTER_API_KEY is not set (put it in .env or .env.local). Nothing was sent.")
        return 2

    chain = settings.model_chain
    where = "pinned by OPENROUTER_MODELS" if settings.models_pinned else "automatic"
    print(f"chain ({where}): {', '.join(chain)}")
    for listing in catalogue.check_chain(chain, settings.openrouter_base_url):
        print(f"catalogue: {catalogue.describe(listing)}")

    allowance = await read_openrouter_allowance(settings)
    if allowance is None:
        print("allowance: could not be read (see the log line above)")
    else:
        print(
            f"allowance: {allowance.used} of {allowance.limit} free requests used today, "
            f"{allowance.remaining} left"
        )

    chat = build_chat(settings)
    synthesis, audit = cases.by_id("synthesis"), cases.by_id("audit-catches")

    print(f"synthesis: asking {chain[0]} (fallbacks: {chain[1:] or 'none'}) ...")
    outcome = await chat.complete(
        messages=cases.messages_of(synthesis),
        schema_model=ChannelCopyDraft,
        temperature=synthesis["temperature"],
        max_tokens=synthesis["max_tokens"],
        schema_name="channel_copy_draft",
    )
    draft = outcome.parsed
    print(f"synthesis: answered by {outcome.model} (json_mode={outcome.json_mode})")
    if isinstance(draft, ChannelCopyDraft):
        print(f"synthesis: title: {draft.web_title}")
        print(f"synthesis: npn_displayed: {draft.npn_displayed}")
        findings = deterministic_findings(draft, cases.SKU)
        print(f"synthesis: deterministic findings: {len(findings)}")
        for finding in findings:
            print(f"  - [{finding.field_name}] {finding.regulation_rule}: {finding.violating_text}")
        synthesis_ok = not findings
    else:
        print("synthesis: the model declined (content filter)")
        synthesis_ok = False

    print("audit: asking the chain about the planted draft ...")
    outcome = await chat.complete(
        messages=cases.messages_of(audit),
        schema_model=ComplianceAuditResult,
        temperature=audit["temperature"],
        max_tokens=audit["max_tokens"],
        schema_name="compliance_audit_result",
    )
    verdict = outcome.parsed
    print(f"audit: answered by {outcome.model} (json_mode={outcome.json_mode})")
    if isinstance(verdict, ComplianceAuditResult):
        print(
            f"audit: is_compliant={verdict.is_compliant} risk={verdict.risk_level} "
            f"violations={len(verdict.violations)}"
        )
        for item in verdict.violations:
            print(f"  - [{item.field_name}] {item.regulation_rule}: {item.violating_text}")
        audit_ok = audit["grade"](verdict)
        print(f"audit: caught the planted problems: {audit_ok}")
    else:
        print("audit: the model declined (content filter)")
        audit_ok = False
    return 0 if synthesis_ok and audit_ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
