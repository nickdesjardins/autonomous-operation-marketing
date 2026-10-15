"""The human-in-the-loop gate: the pause, the three decisions, and what the webhook node refuses.

The mechanics are LangGraph's ``interrupt_before`` plus ``aupdate_state``: approve and edit
update the paused state and resume, reject is written as if the webhook node had run so the
node never does. These tests pin that contract, which the API's review route relies on.
"""

from __future__ import annotations

import uuid

from app.graph.builder import NODE_GATE, NODE_WEBHOOK
from app.graph.state import (
    STATUS_AWAITING_REVIEW,
    STATUS_DISPATCH_REFUSED,
    STATUS_DISPATCHED,
    STATUS_REJECTED,
)
from app.main import Components
from app.services.webhook_dispatcher import build_payload

from .conftest import FakeChat, compliant_audit, good_draft, run_config, start_input

EDITED_TITLE = "Magnesium Bisglycinate 200 mg: the evening capsule (edited by the reviewer)"


async def _pause(pipeline: Components, chat: FakeChat) -> tuple[str, dict]:
    """Run a thread to the gate and hand back its id and config."""
    chat.script(drafts=[good_draft()], audits=[compliant_audit()])
    thread_id = str(uuid.uuid4())
    config = run_config(thread_id)
    await pipeline.graph.ainvoke(start_input(thread_id), config)
    return thread_id, config


def _payloads(pipeline: Components) -> list[dict]:
    return pipeline.dispatcher.payloads  # type: ignore[attr-defined]


async def test_the_graph_pauses_before_webhook_dispatch(
    pipeline: Components, chat: FakeChat
) -> None:
    _, config = await _pause(pipeline, chat)
    snapshot = await pipeline.graph.aget_state(config)

    assert snapshot.next == (NODE_WEBHOOK,)
    assert snapshot.values["status"] == STATUS_AWAITING_REVIEW
    assert snapshot.values["human_review_status"] == "pending"
    assert snapshot.values["webhook_delivery_result"] is None
    assert snapshot.values["trace"][-1]["node"] == NODE_GATE
    assert snapshot.values["trace"][-1]["detail"]["escalated"] is False
    assert _payloads(pipeline) == []


async def test_approve_resumes_and_runs_the_webhook_node_exactly_once(
    pipeline: Components, chat: FakeChat
) -> None:
    thread_id, config = await _pause(pipeline, chat)

    await pipeline.graph.aupdate_state(
        config,
        {
            "human_review_status": "approved",
            "human_reviewer_id": "qa-lead",
            "human_reviewer_notes": "Looks good.",
        },
    )
    assert (await pipeline.graph.aget_state(config)).next == (NODE_WEBHOOK,)

    final = await pipeline.graph.ainvoke(None, config)
    assert final["status"] == STATUS_DISPATCHED
    assert final["human_review_status"] == "approved"
    assert final["webhook_delivery_result"]["mode"] == "dry_run"
    assert [event["node"] for event in final["trace"]].count(NODE_WEBHOOK) == 1
    assert (await pipeline.graph.aget_state(config)).next == ()

    (payload,) = _payloads(pipeline)
    assert payload["thread_id"] == thread_id
    assert payload["review"] == {
        "status": "approved",
        "reviewer_id": "qa-lead",
        "notes": "Looks good.",
    }

    # Resuming a finished thread runs nothing: the webhook is not sent twice.
    again = await pipeline.graph.ainvoke(None, config)
    assert again["status"] == STATUS_DISPATCHED
    assert len(_payloads(pipeline)) == 1
    assert len(chat.calls) == 2


async def test_edit_dispatches_the_reviewers_copy(pipeline: Components, chat: FakeChat) -> None:
    _, config = await _pause(pipeline, chat)
    override = good_draft(web_title=EDITED_TITLE).model_dump()

    await pipeline.graph.aupdate_state(
        config,
        {"draft_copy": override, "human_review_status": "edited", "human_reviewer_id": "editor"},
    )
    final = await pipeline.graph.ainvoke(None, config)

    assert final["status"] == STATUS_DISPATCHED
    assert final["draft_copy"]["web_title"] == EDITED_TITLE
    assert build_payload(final)["copy"]["web_title"] == EDITED_TITLE
    (payload,) = _payloads(pipeline)
    assert payload["copy"]["web_title"] == EDITED_TITLE
    assert payload["review"]["status"] == "edited"
    # The audit trail of the model's draft is kept alongside the human's copy.
    assert payload["compliance"]["audits"] == 1 and payload["compliance"]["is_compliant"] is True


async def test_reject_ends_the_thread_without_running_the_node(
    pipeline: Components, chat: FakeChat
) -> None:
    _, config = await _pause(pipeline, chat)

    await pipeline.graph.aupdate_state(
        config,
        {
            "human_review_status": "rejected",
            "status": STATUS_REJECTED,
            "human_reviewer_id": "qa-lead",
        },
        as_node=NODE_WEBHOOK,
    )
    snapshot = await pipeline.graph.aget_state(config)

    assert snapshot.next == ()
    assert snapshot.values["status"] == STATUS_REJECTED
    assert snapshot.values["human_review_status"] == "rejected"
    assert snapshot.values["webhook_delivery_result"] is None
    assert _payloads(pipeline) == []

    # Nothing is left to run, so a resume is a no-op and still sends nothing.
    resumed = await pipeline.graph.ainvoke(None, config)
    assert resumed["status"] == STATUS_REJECTED
    assert _payloads(pipeline) == []


async def test_human_edit_and_reject(pipeline: Components, chat: FakeChat) -> None:
    """The two decisions besides approval, on two paused threads of the same graph."""
    _, edit_config = await _pause(pipeline, chat)
    _, reject_config = await _pause(pipeline, chat)

    # Edit: the reviewer's copy replaces the model's and is what the webhook carries.
    await pipeline.graph.aupdate_state(
        edit_config,
        {
            "draft_copy": good_draft(web_title=EDITED_TITLE).model_dump(),
            "human_review_status": "edited",
            "human_reviewer_id": "editor",
        },
    )
    edited = await pipeline.graph.ainvoke(None, edit_config)
    assert edited["status"] == STATUS_DISPATCHED
    (payload,) = _payloads(pipeline)
    assert payload["copy"]["web_title"] == EDITED_TITLE
    assert payload["review"]["status"] == "edited"

    # Reject: written as if the webhook node had run, so the thread ends and nothing leaves.
    await pipeline.graph.aupdate_state(
        reject_config,
        {
            "human_review_status": "rejected",
            "status": STATUS_REJECTED,
            "human_reviewer_id": "qa-lead",
        },
        as_node=NODE_WEBHOOK,
    )
    snapshot = await pipeline.graph.aget_state(reject_config)
    assert snapshot.next == () and snapshot.values["status"] == STATUS_REJECTED
    assert snapshot.values["webhook_delivery_result"] is None
    assert len(_payloads(pipeline)) == 1  # still only the edited thread's delivery
    assert len(chat.calls) == 4  # two drafts and two audits: a review never calls a model


async def test_resuming_without_a_decision_is_refused_by_the_node(
    pipeline: Components, chat: FakeChat
) -> None:
    _, config = await _pause(pipeline, chat)

    final = await pipeline.graph.ainvoke(None, config)  # nobody approved anything

    assert final["status"] == STATUS_DISPATCH_REFUSED
    assert "'pending'" in final["error"]
    assert final["webhook_delivery_result"] is None
    assert final["trace"][-1]["detail"] == {"outcome": "refused", "review": "pending"}
    assert _payloads(pipeline) == []
    assert (await pipeline.graph.aget_state(config)).next == ()


async def test_state_history_shows_the_interrupt_checkpoint(
    pipeline: Components, chat: FakeChat
) -> None:
    _, config = await _pause(pipeline, chat)
    await pipeline.graph.aupdate_state(config, {"human_review_status": "approved"})
    await pipeline.graph.ainvoke(None, config)

    history = [snapshot async for snapshot in pipeline.graph.aget_state_history(config)]
    assert history[0].next == ()  # newest first: the finished thread
    waiting = [
        snapshot
        for snapshot in history
        if snapshot.next == (NODE_WEBHOOK,)
        and snapshot.values.get("status") == STATUS_AWAITING_REVIEW
    ]
    assert waiting, "no checkpoint shows the thread waiting before webhook_dispatch"
    assert waiting[-1].values["human_review_status"] == "pending"
    assert (
        waiting[0].values["human_review_status"] == "approved"
    )  # the update is its own checkpoint
    assert all(snapshot.config["configurable"]["checkpoint_id"] for snapshot in history)
    assert history[-1].next == ("__start__",)  # the oldest checkpoint is the input itself
