"""The graph end to end over an in-memory saver: the happy path, the cycle, the escalation.

Every test compiles the production topology (``app.graph.builder.build_graph``) with the
fake chat scripted for exactly the calls the test expects. Running out of script fails the
test, so "exactly three drafts were requested" is proven, not assumed.
"""

from __future__ import annotations

import uuid

import pytest

from app.compliance.rules import deterministic_findings
from app.errors import ProviderError
from app.graph.builder import (
    NODE_COMPLIANCE,
    NODE_GATE,
    NODE_INVENTORY,
    NODE_SUPERVISOR,
    NODE_SYNTHESIS,
    NODE_WEBHOOK,
)
from app.graph.state import (
    STATUS_AWAITING_REVIEW,
    STATUS_DISPATCHED,
    STATUS_INVALID_REQUEST,
    STATUS_OUT_OF_STOCK,
    STATUS_SKU_NOT_FOUND,
)
from app.main import Components
from app.seed import seed_records

from .conftest import (
    NPN,
    SKU,
    FakeChat,
    bad_draft,
    compliant_audit,
    failing_audit,
    good_draft,
    run_config,
    start_input,
)


def _thread() -> tuple[str, dict]:
    thread_id = str(uuid.uuid4())
    return thread_id, run_config(thread_id)


def _trace_nodes(state: dict) -> list[str]:
    return [event["node"] for event in state["trace"]]


# ----------------------------------------------------------------------------- happy path


async def test_happy_path_single_pass(pipeline: Components, chat: FakeChat) -> None:
    chat.script(drafts=[good_draft()], audits=[compliant_audit()])
    thread_id, config = _thread()

    paused = await pipeline.graph.ainvoke(start_input(thread_id), config)
    snapshot = await pipeline.graph.aget_state(config)

    assert snapshot.next == (NODE_WEBHOOK,)
    assert paused["status"] == STATUS_AWAITING_REVIEW
    assert paused["human_review_status"] == "pending"
    assert paused["revision_count"] == 0
    assert paused["requires_human_escalation"] is False
    assert paused["compliance_report"]["is_compliant"] is True
    assert len(paused["audit_history"]) == 1
    assert paused["webhook_delivery_result"] is None
    assert len(chat.draft_calls) == 1 and len(chat.audit_calls) == 1

    await pipeline.graph.aupdate_state(
        config, {"human_review_status": "approved", "human_reviewer_id": "qa-lead"}
    )
    final = await pipeline.graph.ainvoke(None, config)

    assert final["status"] == STATUS_DISPATCHED
    assert final["webhook_delivery_result"]["mode"] == "dry_run"
    assert final["webhook_delivery_result"]["attempts"] == 0
    assert (await pipeline.graph.aget_state(config)).next == ()
    assert pipeline.dispatcher.payloads[0]["review"]["reviewer_id"] == "qa-lead"  # type: ignore[attr-defined]
    assert len(chat.calls) == 2  # approval never calls a model again


async def test_trace_carries_every_node_in_order(pipeline: Components, chat: FakeChat) -> None:
    chat.script(drafts=[good_draft()], audits=[compliant_audit()])
    thread_id, config = _thread()

    paused = await pipeline.graph.ainvoke(start_input(thread_id), config)
    assert _trace_nodes(paused) == [
        NODE_SUPERVISOR,
        NODE_INVENTORY,
        NODE_SYNTHESIS,
        NODE_COMPLIANCE,
        NODE_GATE,
    ]
    assert paused["trace"][1]["detail"]["outcome"] == "found"
    assert paused["trace"][2]["detail"]["model"] == chat.model
    assert paused["trace"][3]["detail"]["decision"] == "compliant"
    assert all(event["ms"] >= 0 for event in paused["trace"])

    await pipeline.graph.aupdate_state(config, {"human_review_status": "approved"})
    final = await pipeline.graph.ainvoke(None, config)
    assert _trace_nodes(final)[-1] == NODE_WEBHOOK
    assert final["trace"][-1]["detail"]["outcome"] == "delivered"


async def test_synthesis_and_audit_use_their_own_temperatures_and_budgets(
    pipeline: Components, chat: FakeChat
) -> None:
    chat.script(drafts=[good_draft()], audits=[compliant_audit()])
    thread_id, config = _thread()
    await pipeline.graph.ainvoke(start_input(thread_id), config)

    (draft_call,) = chat.draft_calls
    (audit_call,) = chat.audit_calls
    settings = pipeline.settings
    assert draft_call.temperature == 0.2 == settings.synthesis_temperature
    assert audit_call.temperature == 0.0 == settings.audit_temperature
    assert draft_call.max_tokens == settings.synthesis_max_tokens
    assert audit_call.max_tokens == settings.audit_max_tokens
    assert draft_call.messages[0]["role"] == "system"
    assert "<product_data>" in draft_call.messages[1]["content"]
    assert "<product_data>" in audit_call.messages[1]["content"]
    # The audit reads the draft that was produced, not the product sheet alone.
    assert good_draft().web_title in audit_call.messages[1]["content"]


# ------------------------------------------------------------------------- the cycle


async def test_cyclic_self_correction_loop(pipeline: Components, chat: FakeChat) -> None:
    # The fake auditor waves the first draft through; the deterministic checks must not.
    chat.script(drafts=[bad_draft(), good_draft()], audits=[compliant_audit(), compliant_audit()])
    thread_id, config = _thread()

    paused = await pipeline.graph.ainvoke(start_input(thread_id, max_revisions=3), config)

    assert (await pipeline.graph.aget_state(config)).next == (NODE_WEBHOOK,)
    assert paused["status"] == STATUS_AWAITING_REVIEW
    assert paused["revision_count"] == 1
    assert paused["requires_human_escalation"] is False
    assert len(chat.draft_calls) == 2 and len(chat.audit_calls) == 2

    first, second = paused["audit_history"]
    assert first["revision"] == 0 and first["is_compliant"] is False
    assert first["risk_level"] == "high"
    rules = {violation["regulation_rule"] for violation in first["violations"]}
    assert any(rule.startswith("Show the Natural Product Number") for rule in rules)
    assert any("claims to cure" in rule for rule in rules)
    assert second["revision"] == 1 and second["is_compliant"] is True
    assert paused["compliance_report"]["is_compliant"] is True
    assert paused["draft_copy"]["npn_displayed"] == "NPN 80098765"

    (instruction,) = paused["revision_instructions"]
    assert instruction.startswith("Revision 1:")
    assert "cures chronic insomnia" in instruction
    assert f"NPN {NPN}" in instruction

    nodes = _trace_nodes(paused)
    assert nodes.count(NODE_SYNTHESIS) == 2 and nodes.count(NODE_COMPLIANCE) == 2
    assert paused["trace"][3]["detail"]["decision"] == "revise"
    assert paused["trace"][3]["detail"]["llm_said_compliant"] is True


async def test_revision_prompt_carries_the_previous_draft_and_the_violations(
    pipeline: Components, chat: FakeChat
) -> None:
    chat.script(drafts=[bad_draft(), good_draft()], audits=[failing_audit(), compliant_audit()])
    thread_id, config = _thread()
    await pipeline.graph.ainvoke(start_input(thread_id, max_revisions=2), config)

    first, second = chat.draft_calls
    assert "<previous_draft>" not in first.messages[1]["content"]
    revision = second.messages[1]["content"]
    assert "This is revision 1." in revision
    assert "<previous_draft>" in revision and "cures chronic insomnia" in revision
    assert "<violations_to_fix>" in revision
    assert failing_audit().violations[0].required_fix in revision  # the auditor's own finding
    assert f"Report the licensed NPN exactly: NPN {NPN}." in revision  # a deterministic one
    assert "<revision_notes>" in revision and "Revision 1:" in revision
    assert second.messages[0]["content"] == first.messages[0]["content"]  # static system prompt


async def test_audit_entry_counts_the_deterministic_findings(
    pipeline: Components, chat: FakeChat
) -> None:
    chat.script(drafts=[bad_draft()], audits=[failing_audit()])
    thread_id, config = _thread()
    paused = await pipeline.graph.ainvoke(start_input(thread_id, max_revisions=0), config)

    sku = next(record for record in seed_records() if record.sku_code == SKU)
    expected = len(deterministic_findings(bad_draft(), sku))
    (entry,) = paused["audit_history"]
    assert expected >= 2
    assert entry["deterministic_findings"] == expected
    assert len(entry["violations"]) == expected + len(failing_audit().violations)
    assert entry["model"] == chat.model
    assert paused["trace"][3]["detail"]["deterministic_findings"] == expected


# ------------------------------------------------------------------------ escalation


async def test_max_revisions_escalation(pipeline: Components, chat: FakeChat) -> None:
    chat.script(drafts=[bad_draft()] * 3, audits=[compliant_audit()] * 3)
    thread_id, config = _thread()

    paused = await pipeline.graph.ainvoke(start_input(thread_id, max_revisions=3), config)

    assert len(chat.draft_calls) == 3 and len(chat.audit_calls) == 3
    assert paused["revision_count"] == 3
    assert paused["requires_human_escalation"] is True
    assert paused["status"] == STATUS_AWAITING_REVIEW
    assert (await pipeline.graph.aget_state(config)).next == (NODE_WEBHOOK,)
    assert len(paused["audit_history"]) == 3
    assert all(entry["is_compliant"] is False for entry in paused["audit_history"])
    assert [entry["revision"] for entry in paused["audit_history"]] == [0, 1, 2]
    assert len(paused["revision_instructions"]) == 3
    assert paused["trace"][-2]["detail"]["decision"] == "escalate"
    assert paused["trace"][-1]["detail"]["escalated"] is True
    assert pipeline.dispatcher.payloads == []  # type: ignore[attr-defined]


async def test_zero_revisions_escalates_after_the_first_failing_audit(
    pipeline: Components, chat: FakeChat
) -> None:
    chat.script(drafts=[bad_draft()], audits=[compliant_audit()])
    thread_id, config = _thread()

    paused = await pipeline.graph.ainvoke(start_input(thread_id, max_revisions=0), config)

    assert len(chat.draft_calls) == 1 and len(chat.audit_calls) == 1
    assert paused["revision_count"] == 1
    assert paused["requires_human_escalation"] is True
    assert paused["status"] == STATUS_AWAITING_REVIEW
    assert (await pipeline.graph.aget_state(config)).next == (NODE_WEBHOOK,)


# --------------------------------------------------------------- stopping before a model


async def test_unknown_sku_ends_without_a_model_call(pipeline: Components, chat: FakeChat) -> None:
    thread_id, config = _thread()
    final = await pipeline.graph.ainvoke(start_input(thread_id, sku_code="NR-NOPE-1"), config)

    assert final["status"] == STATUS_SKU_NOT_FOUND
    assert "NR-NOPE-1" in final["error"]
    assert final["inventory_data"] is None
    assert (await pipeline.graph.aget_state(config)).next == ()
    assert chat.calls == []
    assert _trace_nodes(final) == [NODE_SUPERVISOR, NODE_INVENTORY]
    assert final["trace"][1]["detail"]["outcome"] == "not_found"


async def test_sku_code_that_is_not_a_code_is_refused_by_the_tool(
    pipeline: Components, chat: FakeChat
) -> None:
    thread_id, config = _thread()
    final = await pipeline.graph.ainvoke(start_input(thread_id, sku_code="drop table"), config)

    assert final["status"] == STATUS_SKU_NOT_FOUND
    assert final["trace"][1]["detail"]["outcome"] == "invalid_args"
    assert chat.calls == []


async def test_out_of_stock_sku_ends_before_any_copy_is_written(
    pipeline: Components, chat: FakeChat
) -> None:
    thread_id, config = _thread()
    final = await pipeline.graph.ainvoke(start_input(thread_id, sku_code="NR-VITD-1000"), config)

    assert final["status"] == STATUS_OUT_OF_STOCK
    assert final["inventory_data"]["stock_quantity"] == 0
    assert "Vitamin D3 1000 IU" in final["error"]
    assert (await pipeline.graph.aget_state(config)).next == ()
    assert chat.calls == []
    assert final["trace"][1]["detail"]["outcome"] == "out_of_stock"


async def test_invalid_request_ends_at_the_supervisor(pipeline: Components, chat: FakeChat) -> None:
    thread_id, config = _thread()
    final = await pipeline.graph.ainvoke(start_input(thread_id, target_channels=[]), config)

    assert final["status"] == STATUS_INVALID_REQUEST
    assert "no target channel" in final["error"]
    assert (await pipeline.graph.aget_state(config)).next == ()
    assert chat.calls == []
    assert _trace_nodes(final) == [NODE_SUPERVISOR]
    assert final["trace"][0]["detail"]["outcome"] == "invalid"


async def test_sku_code_is_upper_cased_by_the_supervisor(
    pipeline: Components, chat: FakeChat
) -> None:
    chat.script(drafts=[good_draft()], audits=[compliant_audit()])
    thread_id, config = _thread()
    paused = await pipeline.graph.ainvoke(start_input(thread_id, sku_code=" nr-mag-200 "), config)
    assert paused["sku_code"] == SKU
    assert paused["inventory_data"]["npn_number"] == NPN


# -------------------------------------------------------------------- provider failure


async def test_provider_error_propagates_and_the_thread_waits_at_the_failed_node(
    pipeline: Components, chat: FakeChat
) -> None:
    chat.script(drafts=[ProviderError("fake/model:free", "rate limited (429)", retryable=True)])
    thread_id, config = _thread()

    with pytest.raises(ProviderError) as info:
        await pipeline.graph.ainvoke(start_input(thread_id), config)
    assert info.value.retryable is True

    snapshot = await pipeline.graph.aget_state(config)
    assert snapshot.next == (NODE_SYNTHESIS,)
    assert snapshot.values["inventory_data"]["sku_code"] == SKU  # the lookup was checkpointed
    assert snapshot.values["draft_copy"] is None


async def test_audit_failure_leaves_the_draft_checkpointed(
    pipeline: Components, chat: FakeChat
) -> None:
    chat.script(drafts=[good_draft()], audits=[ProviderError("fake", "all models failed")])
    thread_id, config = _thread()

    with pytest.raises(ProviderError):
        await pipeline.graph.ainvoke(start_input(thread_id), config)

    snapshot = await pipeline.graph.aget_state(config)
    assert snapshot.next == (NODE_COMPLIANCE,)
    assert snapshot.values["draft_copy"]["web_title"] == good_draft().web_title


async def test_a_draft_at_its_cap_with_an_em_dash_is_a_provider_error_not_a_crash(
    pipeline: Components, chat: FakeChat
) -> None:
    # The schema accepts 159 characters; the em dash becomes " - " and the title overflows.
    chat.script(drafts=[good_draft(web_title="M" * 157 + chr(0x2014) + "x")])
    thread_id, config = _thread()
    with pytest.raises(ProviderError, match="typography") as info:
        await pipeline.graph.ainvoke(start_input(thread_id), config)
    assert info.value.retryable is True
    assert (await pipeline.graph.aget_state(config)).next == (NODE_SYNTHESIS,)
    # A dash with room to spare is simply tidied.
    chat.script(drafts=[good_draft(web_title="Magnesium" + chr(0x2014) + "steady evenings")])
    chat.script(audits=[compliant_audit()])
    thread_id, config = _thread()
    paused = await pipeline.graph.ainvoke(start_input(thread_id), config)
    assert paused["draft_copy"]["web_title"] == "Magnesium - steady evenings"


async def test_a_refusal_is_a_provider_error(pipeline: Components, chat: FakeChat) -> None:
    from .conftest import refused_outcome

    chat.script(drafts=[refused_outcome()])
    thread_id, config = _thread()
    with pytest.raises(ProviderError, match="declined"):
        await pipeline.graph.ainvoke(start_input(thread_id), config)


async def test_without_a_model_the_synthesis_node_raises_a_provider_error(settings) -> None:
    from .conftest import build_components

    components = build_components(settings, None)
    thread_id, config = _thread()
    with pytest.raises(ProviderError, match="no language model"):
        await components.graph.ainvoke(start_input(thread_id), config)
    assert (await components.graph.aget_state(config)).next == (NODE_SYNTHESIS,)
