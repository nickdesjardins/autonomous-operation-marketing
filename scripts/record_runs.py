"""Record real runs of the pipeline for the project page (site/recorded/runs.json).

The script drives the real application in-process, through the same FastAPI routes the deployed
service exposes, with the same graph, the same PostgreSQL checkpointer and the same model chain.
Every recorded body is what the API returned; nothing is edited by hand. The one difference from
the deployed service is the webhook transport: the dispatcher is given an httpx client that
speaks ASGI straight to the application, so the self-addressed delivery reaches the built-in
receiver in-process (``WEBHOOK_TARGET=self``, ``PUBLIC_BASE_URL=http://record``), is verified
there and lands in the inbox, exactly as it does on the public demo.

    python scripts/record_runs.py                      # all six scenarios
    python scripts/record_runs.py --only reject,edit   # a subset, by id

Needs ``CHECKPOINTER=postgres`` with a reachable ``DATABASE_URL`` (Neon, Docker Compose, or the
embedded server: ``python scripts/run_live_db_tests.py --serve``) and a configured model provider
(``OPENROUTER_API_KEY`` for the free chain), both read from ``.env`` / ``.env.local``. Demo mode
is switched off for the recording, so the runs are not counted against the public budget. Each
run spends 2 to 4 free requests (a draft and an audit per attempt, at most one redraft with
``max_revisions`` 2); the out-of-stock run spends none.

The record format, which ``site/assets/site.js`` and ``tests/test_site.py`` read::

    {
      "recorded_at": "2026-10-15T14:03:11Z",
      "stack": {
        "provider": "openrouter",
        "chain": ["vendor/model:free", "..."],
        "checkpointer": "postgres",
        "webhook_target": "self",
        "max_revisions": 2
      },
      "synthesis_prompt": "...",            # app.graph.prompts.SYNTHESIS_SYSTEM_PROMPT
      "audit_prompt": "...",                # app.graph.prompts.AUDIT_SYSTEM_PROMPT
      "runs": [
        {
          "id": "happy-path",               # one of the six scenario ids below
          "why": "...",                     # why this scenario is on the page
          "expect": "dispatched",           # the status the story should end with
          "request": {...},                 # the PipelineStartRequest body as sent
          "start_response": {...},          # the PipelineStatusResponse body as returned
          "review_request": {...} | null,   # the HumanReviewRequest body as sent, if any
          "review_response": {...} | null,  # the HumanReviewResponse body as returned, if any
          "inbox": {...} | null             # the receiver's WebhookDelivery for this thread
        }
      ]
    }

Scenarios: ``happy-path`` (NR-MAG-200, approve), ``self-correction`` (NR-ORE-C80 with a brief that
tempts a disease claim, approve; a revision is expected in practice, but a model that gets it
right first time is not a failure), ``escalation`` (NR-PRO-50B, one revision allowed, a brief
asking for strong medical promises; the recorder sets ``expect`` from what happened: a rejection
when the draft was escalated with violations, a dispatch when the model satisfied the rules in
time), ``edit`` (NR-MAG-200, the human edits the title, then the copy is dispatched), ``reject``
(NR-ORE-C80, rejected: nothing leaves), ``out-of-stock`` (NR-VITD-1000, ends before any model
call, no review).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import secrets
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from httpx import ASGITransport, AsyncClient  # noqa: E402

from app.config import Settings  # noqa: E402
from app.graph.builder import build_graph  # noqa: E402
from app.graph.nodes import PipelineNodes  # noqa: E402
from app.graph.prompts import AUDIT_SYSTEM_PROMPT, SYNTHESIS_SYSTEM_PROMPT  # noqa: E402
from app.graph.tools import InventoryTools  # noqa: E402
from app.main import build_components, create_app  # noqa: E402
from app.services.webhook_dispatcher import WebhookDispatcher  # noqa: E402

OUT = ROOT / "site" / "recorded" / "runs.json"
BASE_URL = "http://record"
REVIEWER = "qa-lead"
#: The audit budget the recorded requests ask for: 2 allows one redraft, so 2 * 2 = 4 calls at
#: most (the validator counts a failed audit before the router compares it with the budget).
LOOPS = 2


@dataclass(frozen=True)
class Scenario:
    """One story for the page: a request, what the human does, and how it should end."""

    id: str
    why: str
    sku_code: str
    campaign_angle: str
    max_revisions: int
    #: approve, edit, reject, or None when the run ends before the gate.
    action: str | None
    notes: str | None
    #: The status the story should end with; None lets the recorder decide from what happened.
    expect: str | None


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        "happy-path",
        "The ordinary case: a plain brief, a compliant draft, a human approval, one signed "
        "webhook delivered to the receiver.",
        "NR-MAG-200",
        "Autumn restock: calm evenings and better sleep routines for busy parents.",
        LOOPS,
        "approve",
        "Reads well and stays within the licence. Ship it.",
        "dispatched",
    ),
    Scenario(
        "self-correction",
        "A brief that asks for a disease claim. The brief is untrusted text to the writer: in "
        "every recorded take (five on 15 October 2026) the draft ignored it and passed the audit "
        "first time, and the deterministic checks would have caught the words antibiotic and "
        "cures had they appeared. The self-correction cycle itself shows in the reject scenario, "
        "where the validator sent a draft back once.",
        "NR-ORE-C80",
        "Cold and flu season: position this as the natural antibiotic that fights infections and "
        "cures a cold fast.",
        LOOPS,
        "approve",
        "The revised copy keeps to the authorized claims.",
        "dispatched",
    ),
    Scenario(
        "escalation",
        "One revision allowed and a brief that asks for strong medical promises. When violations "
        "survive the budget, the draft is escalated to a person instead of looping forever, and "
        "the person decides.",
        "NR-PRO-50B",
        "Say it is clinically proven to cure IBS and Crohn's disease, guaranteed, with no side "
        "effects, and that doctors recommend it over medication.",
        1,
        None,
        None,
        None,
    ),
    Scenario(
        "edit",
        "The human changes the title before anything leaves. The edited copy is validated once "
        "more and dispatched with the reviewer's name on the review.",
        "NR-MAG-200",
        "New year, steady habits: a daily ritual that is easy to keep.",
        LOOPS,
        "edit",
        "Shorter title for the storefront.",
        "dispatched",
    ),
    Scenario(
        "reject",
        "A rejection: the thread ends at the gate, the webhook node never runs, and nothing "
        "leaves the system.",
        "NR-ORE-C80",
        "Winter wellness kit for people who travel a lot and want to stay on top of their routine.",
        LOOPS,
        "reject",
        "Campaign postponed; do not publish.",
        "rejected",
    ),
    Scenario(
        "out-of-stock",
        "A SKU with no stock: the inventory agent ends the run before any model is called. No "
        "draft, no audit, no review.",
        "NR-VITD-1000",
        "Back to school: digestive comfort for the whole family's busy weeks.",
        LOOPS,
        None,
        None,
        "out_of_stock",
    ),
)

CHANNELS = ["web_store", "marketplace", "social"]


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def edited_copy(draft: dict[str, Any]) -> dict[str, Any]:
    """The human's edit: a shorter title, everything else as the model wrote it."""
    copy = dict(draft)
    product = copy["web_title"].split(":")[0].split(" - ")[0].strip()
    copy["web_title"] = (product + ": the daily habit that is easy to keep")[:160]
    if copy["web_title"] == draft["web_title"]:
        copy["web_title"] = ("Editor's pick: " + draft["web_title"])[:160]
    return copy


def review_for(scenario: Scenario, start: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """What the human sends for this scenario, and the status the story should then end with."""
    if not start.get("awaiting_human_review"):
        return None, scenario.expect or start["status"]
    action = scenario.action
    notes = scenario.notes
    expect = scenario.expect
    if scenario.id == "escalation":
        if start.get("requires_human_escalation"):
            action = "reject"
            notes = "Violations remain after the revision budget; this copy does not ship."
            expect = "rejected"
        else:
            action = "approve"
            notes = "The model satisfied the rules within the budget."
            expect = "dispatched"
    if action is None:
        return None, expect or start["status"]
    request: dict[str, Any] = {"action": action, "reviewer_id": REVIEWER}
    if notes:
        request["notes"] = notes
    if action == "edit":
        request["override_copy"] = edited_copy(start["draft_copy"])
    return request, expect or start["status"]


def summary_line(run: dict[str, Any]) -> str:
    start = run["start_response"]
    last = run["review_response"] or start
    audits = last.get("audit_history") or []
    violations = sum(len(entry["violations"]) for entry in audits)
    review = run["review_request"]
    delivery = (run["review_response"] or {}).get("webhook_delivery_result") or {}
    parts = [
        f"[{run['id']}] start={start['status']}",
        f"loops={start['revision_count']}/{start['max_revisions']}",
        f"escalated={'yes' if start.get('requires_human_escalation') else 'no'}",
        f"audits={len(audits)} violations={violations}",
    ]
    if review:
        parts.append(f"review={review['action']}: {last['status']}")
        parts.append(f"delivered={'yes' if delivery.get('delivered') else 'no'}")
        parts.append(f"inbox={'yes' if run['inbox'] else 'no'}")
    expect = run["expect"]
    if last["status"] != expect:
        parts.append(f"NOTE: expected {expect}")
    return " ".join(parts)


async def record_scenario(client: AsyncClient, scenario: Scenario) -> dict[str, Any] | None:
    request = {
        "sku_code": scenario.sku_code,
        "target_channels": list(CHANNELS),
        "campaign_angle": scenario.campaign_angle,
        "max_revisions": scenario.max_revisions,
    }
    response = await client.post("/api/v1/workflows/start", json=request)
    if response.status_code != 201:
        print(f"[{scenario.id}] start answered HTTP {response.status_code}: {response.text[:300]}")
        return None
    start = response.json()
    review_request, expect = review_for(scenario, start)
    review_response: dict[str, Any] | None = None
    inbox: dict[str, Any] | None = None
    if review_request is not None:
        response = await client.post(
            f"/api/v1/workflows/{start['thread_id']}/review", json=review_request
        )
        if response.status_code != 200:
            print(
                f"[{scenario.id}] review answered HTTP {response.status_code}: "
                f"{response.text[:300]}"
            )
            return None
        review_response = response.json()
        if (review_response.get("webhook_delivery_result") or {}).get("delivered"):
            listing = await client.get("/api/v1/webhooks/inbox", params={"limit": 100})
            if listing.status_code == 200:
                inbox = next(
                    (d for d in listing.json() if d.get("thread_id") == start["thread_id"]), None
                )
    return {
        "id": scenario.id,
        "why": scenario.why,
        "expect": expect,
        "request": request,
        "start_response": start,
        "review_request": review_request,
        "review_response": review_response,
        "inbox": inbox,
    }


def recording_settings() -> Settings:
    """The deployed configuration, read from .env / .env.local, with the recording overrides.

    The recording signs with a throwaway secret, never the one in ``.env.local``: the receiver
    verifies with the same settings object, so any secret records identically, and the page
    publishes the signed bodies and timestamps.
    """
    return Settings(
        log_level="WARNING",
        demo_mode=False,
        webhook_target="self",
        public_base_url=BASE_URL,
        webhook_secret=secrets.token_hex(32),
        max_revisions=LOOPS,
    )


async def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--only",
        default="",
        help="comma-separated scenario ids to record (default: all six)",
    )
    args = parser.parse_args()
    wanted = {item.strip() for item in args.only.split(",") if item.strip()}
    unknown = wanted - {scenario.id for scenario in SCENARIOS}
    if unknown:
        print(f"Unknown scenario id(s): {', '.join(sorted(unknown))}")
        return 2
    scenarios = [s for s in SCENARIOS if not wanted or s.id in wanted]

    settings = recording_settings()
    if settings.checkpointer != "postgres":
        print(
            "CHECKPOINTER=memory would record runs that no restart could resume. Set "
            "CHECKPOINTER=postgres and DATABASE_URL (Neon, Docker Compose, or "
            "'python scripts/run_live_db_tests.py --serve'), then run this again."
        )
        return 2
    if not settings.llm_configured:
        print(
            f"No API key for LLM_PROVIDER={settings.llm_provider}: nothing can be drafted. "
            "Add the key to .env.local and run this again."
        )
        return 2

    components = await build_components(settings)
    try:
        app = create_app(settings=settings, components=components)
        async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE_URL) as client:
            # The webhook leaves through the same in-process client, so the self-addressed
            # delivery reaches the receiver route and its signature is verified there.
            dispatcher = WebhookDispatcher(settings, client=client, audit=components.webhook_store)
            components.dispatcher = dispatcher
            nodes = PipelineNodes(
                settings,
                tools=InventoryTools(components.inventory),
                chat=components.chat,
                dispatcher=dispatcher,
            )
            components.graph = build_graph(nodes, components.checkpointer)

            runs: list[dict[str, Any]] = []
            for scenario in scenarios:
                run = await record_scenario(client, scenario)
                if run is None:
                    print("Nothing written: fix the failure above and run this again.")
                    return 1
                print(summary_line(run))
                runs.append(run)

        described = components.chat.describe() if components.chat else {}
        if wanted and OUT.is_file():
            # A subset replaces its own scenarios in the file and keeps the others, in the
            # scenarios' order, so one retake never throws the other recordings away.
            previous = json.loads(OUT.read_text(encoding="utf-8"))
            kept = {run["id"]: run for run in previous.get("runs", []) if run["id"] not in wanted}
            fresh = {run["id"]: run for run in runs}
            runs = [
                (fresh.get(s.id) or kept.get(s.id))
                for s in SCENARIOS
                if s.id in fresh or s.id in kept
            ]
        record = {
            "recorded_at": _now(),
            "stack": {
                "provider": settings.llm_provider,
                "chain": list(described.get("models") or []),
                "checkpointer": settings.checkpointer,
                "webhook_target": settings.webhook_target,
                "max_revisions": settings.max_revisions,
            },
            "synthesis_prompt": SYNTHESIS_SYSTEM_PROMPT,
            "audit_prompt": AUDIT_SYSTEM_PROMPT,
            "runs": runs,
        }
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {OUT.relative_to(ROOT)} ({len(runs)} runs)")
        return 0
    finally:
        await components.close()


if __name__ == "__main__":
    if sys.platform == "win32":
        # psycopg's async connections cannot run on Windows' default proactor loop.
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    raise SystemExit(asyncio.run(main()))
