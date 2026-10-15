"""Tests for the service's upkeep of its own model chain (``app.models.chooser``).

Nothing here calls a model or reaches the network. The chain client a test would call is
replaced by one that answers from a script, OpenRouter's list, its pages and the published
file by one fake request function, OpenRouter's count of the day by a plain value, and the
cache file lives in a temporary folder. Everything the file needs is defined in it.

Layout:

* automatic or pinned        - what a blank setting means, and startup;
* noticing a failure         - what starts a repair, and what does not;
* the test and the chain     - a repair, the button, one chain everywhere;
* guards and the picture     - one at a time, the pause, the allowance, the page's object.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from app import budget
from app.budget import Allowance
from app.config import DEFAULT_MODEL_CHAIN, Settings, parse_model_chain
from app.errors import ProviderAuthError, ProviderError
from app.models import cache, cases, catalogue, chooser, published, rules, testing
from app.schemas import ComplianceAuditResult, ViolationItem
from app.services.llm_client import Attempt, ChatOutcome, ModelFailure

FIRST, SECOND = parse_model_chain(DEFAULT_MODEL_CHAIN)[:2]
NEW, OTHER, BROKEN = "vendor/new:free", "vendor/other:free", "vendor/broken:free"
URL = "https://example.test/tested-models/chain.json"
RUN = "https://github.com/someone/somewhere/actions/runs/7"
BASE = "https://openrouter.ai/api/v1"


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "environment": "test",
        "checkpointer": "memory",
        "openrouter_api_key": "sk-or-test",
        "model_published_url": URL,
        "log_level": "WARNING",
    }
    values.update(overrides)
    return Settings(_env_file=None, _env_prefix="HERMETIC_TEST_", **values)


def right_answer(case_id: str) -> Any:
    if case_id == "synthesis":
        return cases.COMPLIANT_DRAFT
    if case_id == "audit-catches":
        return ComplianceAuditResult(
            is_compliant=False,
            risk_level="high",
            violations=[
                ViolationItem(
                    field_name="web_description_html",
                    violating_text="cures chronic insomnia",
                    regulation_rule="NO_DISEASE_CLAIMS",
                    required_fix="Remove the cure claim.",
                )
            ],
            revision_summary="A cure claim and no NPN.",
        )
    return ComplianceAuditResult(
        is_compliant=True, risk_level="low", violations=[], revision_summary="Compliant."
    )


def wrong_answer(case_id: str) -> Any:
    if case_id == "synthesis":
        return cases.PLANTED_DRAFT
    if case_id == "audit-catches":
        return right_answer("audit-passes")
    return ComplianceAuditResult(
        is_compliant=False,
        risk_level="medium",
        violations=[
            ViolationItem(
                field_name="web_title", violating_text="x", regulation_rule="y", required_fix="z"
            )
        ],
        revision_summary="Wrongly flagged.",
    )


class FakeChain:
    """Answers each case from ``script[model][case]``: right, wrong, lost, unusable, refused, auth."""

    name = "openrouter"

    def __init__(self, model: str, world: World, listener: Any) -> None:
        self.model, self.world, self.listener = model, world, listener

    async def complete(
        self,
        *,
        messages: Any,
        schema_model: Any,
        temperature: Any,
        max_tokens: int,
        schema_name: str = "output",
    ) -> ChatOutcome:
        self.world.test_calls.append((self.model, schema_name))
        what = self.world.script.get(self.model, {}).get(schema_name, "right")
        if callable(what):
            what = await what()
        if what == "auth":
            raise ProviderAuthError("openrouter-auth", "the API key was rejected (401)")
        if what in ("lost", "unusable"):
            kind, retryable, message = {
                "lost": ("ProviderError", True, "rate limited (429)"),
                "unusable": ("SchemaValidationError", False, "invalid output (ValidationError)"),
            }[what]
            self.listener([ModelFailure(self.model, kind, retryable, message)], False)
            raise ProviderError(self.model, f"all models failed ({self.model}: {message})")
        if what == "refused":
            return ChatOutcome(self.model, "", "content_filter", refused=True)
        parsed = wrong_answer(schema_name) if what == "wrong" else right_answer(schema_name)
        return ChatOutcome(
            self.model,
            json.dumps(parsed.model_dump()),
            "stop",
            parsed=parsed,
            attempts=[Attempt(self.model, "answered", 0.1)],
        )


class World:
    """Everything outside the process, as knobs: the catalogue, the published file, the day's count."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.pages: dict[str, str] = {FIRST: "listed", SECOND: "listed"}  # listed | gone | down
        self.listed: list[str] | Exception = [FIRST, SECOND]
        self.published: Any = (404, "")  # (status, text) or an exception to raise
        self.allowance: Allowance | None = Allowance(10, 1000, 990)
        self.script: dict[str, dict[str, Any]] = {}
        self.test_calls: list[tuple[str, str]] = []
        self.asked: list[str] = []

    def http_get(self, url: str) -> tuple[int, str]:
        self.asked.append(url)
        if url == URL:
            if isinstance(self.published, Exception):
                raise self.published
            return self.published
        if url.startswith(BASE + "/model/"):
            model = url.removeprefix(BASE + "/model/")
            state = self.pages.get(model, "listed")
            if state == "gone":
                return 404, json.dumps({"error": {"message": f"Model not found: {model}"}})
            if state == "down":
                raise TimeoutError("no answer")
            return 200, json.dumps({"data": {"id": model}})
        if url.startswith(BASE + "/models?"):
            if isinstance(self.listed, Exception):
                raise self.listed
            entries = [
                {
                    "id": model,
                    "pricing": {"prompt": "0", "completion": "0"},
                    "supported_parameters": ["structured_outputs"],
                }
                for model in self.listed
            ]
            return 200, json.dumps({"data": entries})
        raise AssertionError(f"unexpected address {url}")

    def factory(self, settings: Settings, model: str, listener: Any) -> FakeChain:
        return FakeChain(model, self, listener)

    @property
    def chain(self) -> list[str]:
        return self.settings.model_chain


def stamp(moment: datetime) -> str:
    return moment.strftime(published.STAMP)


def recent() -> datetime:
    """A moment a few minutes ago: newer than the default's measuring, and not in the future."""
    return datetime.now(UTC) - timedelta(minutes=5)


def published_file(
    chain: list[str],
    tested: datetime,
    settings: Settings,
    fingerprint: str | None = None,
    test: dict | None = None,
) -> tuple[int, str]:
    document = {
        "version": published.VERSION,
        "fingerprint": fingerprint or published.fingerprint(settings),
        "tested": stamp(tested),
        "chain": chain,
        "job": {"checked": stamp(tested), "found": chain, "outcome": "published"},
        "by": {"run": RUN},
    }
    if test is not None:
        document["test"] = test
    return 200, json.dumps(document)


def failure(
    model: str,
    kind: str = "ProviderError",
    retryable: bool | None = True,
    message: str = "rate limited (429)",
) -> ModelFailure:
    return ModelFailure(model, kind, retryable, message)


GONE_404 = ("ProviderError", False, "model not found (404); withdrawn?")
FORBIDDEN = ("ProviderError", False, "request forbidden (403)")


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> World:
    settings = make_settings()
    made = World(settings)
    monkeypatch.setattr(cache, "PATH", tmp_path / "model_cache.json")
    cache.forget()
    chooser.reset()
    chooser.configure(settings)
    monkeypatch.setattr(catalogue, "_http_get", made.http_get)
    monkeypatch.setattr(chooser, "chain_factory", made.factory)
    monkeypatch.setattr(rules, "SECONDS_BETWEEN_STARTS", 0.0)

    async def read_allowance(settings: Settings) -> Allowance | None:
        return made.allowance

    monkeypatch.setattr(budget, "read_openrouter_allowance", read_allowance)
    yield made
    chooser.reset()
    cache.forget()


async def settled() -> None:
    await chooser.wait()


# ---------------------------------------------------------------------------
# Automatic or pinned
# ---------------------------------------------------------------------------


def test_a_blank_setting_means_automatic_and_anything_else_says_why_not() -> None:
    assert chooser.mode(make_settings()) == "automatic"
    assert chooser.mode(make_settings(openrouter_models=f"{FIRST},{SECOND}")) == "pinned"
    assert chooser.mode(make_settings(openrouter_api_key=None)) == "no_key"
    assert (
        chooser.mode(make_settings(openrouter_base_url="https://api.example.com/v1"))
        == "not_openrouter"
    )
    assert set(chooser.REFUSALS) >= {"pinned", "no_key", "not_openrouter"}


async def test_startup_reads_a_newer_published_chain_and_uses_it(world: World) -> None:
    tested = recent()
    world.published = published_file([SECOND, NEW], tested, world.settings)
    world.pages[NEW] = "listed"

    await chooser.at_startup(world.settings)

    assert world.chain == [SECOND, NEW]
    assert cache.origin()["kind"] == "published" and cache.origin()["tested"] == stamp(tested)
    picture = chooser.picture(world.settings)
    assert picture["published"] == {"state": "used", "tested": stamp(tested)}
    assert picture["origin"]["kind"] == "published" and picture["origin"]["age_days"] >= 0
    # The file survives in the cache, so a restart on a machine that keeps its disk finds it.
    assert json.loads(cache.PATH.read_text(encoding="utf-8"))["chain"] == [SECOND, NEW]
    # The catalogue was asked about the chain in use, and nobody is gone.
    assert any(url.endswith(f"/model/{NEW}") for url in world.asked) and picture["gone"] == []


@pytest.mark.parametrize(
    ("answer", "state"),
    [
        ((404, ""), "missing"),
        ((503, "down"), "unreachable"),
        (ConnectionError("no route"), "unreachable"),
        ((200, "not json"), "invalid"),
        ("foreign", "foreign"),
        ("older", "older"),
    ],
)
async def test_no_unusable_published_list_changes_the_chain(
    world: World, answer: Any, state: str
) -> None:
    if answer == "foreign":
        answer = published_file([NEW], recent(), world.settings, fingerprint="0" * 16)
    elif answer == "older":
        answer = published_file(
            [NEW], testing.default_measured_at() - timedelta(days=1), world.settings
        )
    world.published = answer

    await chooser.at_startup(world.settings)

    assert world.chain == [FIRST, SECOND] and cache.origin() is None
    assert chooser.picture(world.settings)["published"]["state"] == state
    assert state in chooser.PUBLISHED_STATES


async def test_startup_never_raises_and_never_waits_long(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(text: str, expected: str, now: datetime) -> Any:
        raise RuntimeError("a bug in parse")

    world.published = (200, "{}")
    monkeypatch.setattr(published, "parse", explode)
    await chooser.at_startup(world.settings)  # the bug is logged, not raised
    assert world.chain == [FIRST, SECOND]

    def slow(url: str) -> tuple[int, str]:
        time.sleep(0.6)
        return 404, ""

    monkeypatch.setattr(catalogue, "_http_get", slow)
    monkeypatch.setattr(chooser, "READ_SECONDS", 0.2)
    started = time.monotonic()
    await chooser.at_startup(world.settings)
    assert time.monotonic() - started < 0.55


async def test_startup_names_a_model_of_the_chain_that_is_gone_without_dropping_it(
    world: World, caplog: pytest.LogCaptureFixture
) -> None:
    world.pages[FIRST] = "gone"
    with caplog.at_level(logging.WARNING, logger="app.models.chooser"):
        await chooser.at_startup(world.settings)
    assert world.chain == [FIRST, SECOND]  # the catalogue's word alone removes nothing
    assert chooser.picture(world.settings)["gone"] == [FIRST]
    assert any(
        FIRST in record.message and "not listed" in record.message for record in caplog.records
    )


async def test_a_pinned_or_keyless_service_reads_nothing_at_startup(world: World) -> None:
    for settings in (
        make_settings(openrouter_models=f"{FIRST},{SECOND}"),
        make_settings(openrouter_api_key=None),
    ):
        world.asked.clear()
        await chooser.at_startup(settings)
        assert world.asked == []
        chooser.log_mode(settings)  # says so, and raises nothing
        assert chooser.picture(settings)["button"]["offered"] is False


async def test_with_the_check_off_nothing_is_asked_at_startup(world: World) -> None:
    settings = make_settings(model_catalogue_check=False)
    await chooser.at_startup(settings)
    assert world.asked == [] and chooser.picture(settings)["published"]["state"] == "off"


# ---------------------------------------------------------------------------
# Noticing a failure
# ---------------------------------------------------------------------------


async def test_a_withdrawn_model_is_dropped_at_once_and_a_repair_test_fills_its_place(
    world: World,
) -> None:
    world.pages[FIRST] = "gone"
    world.listed = [SECOND, NEW, OTHER]
    world.script = {OTHER: {"audit-catches": "wrong"}}
    chains: list[list[str]] = []

    async def watch() -> str:
        chains.append(world.chain)  # what a request sees while the test goes on
        return "right"

    world.script[NEW] = {"synthesis": watch}

    chooser.note([failure(FIRST, *GONE_404)], answered=True)
    await settled()

    # The gone model left before the first test call, and the test added the one that passed.
    assert chains == [[SECOND]]
    assert world.chain == [SECOND, NEW]
    assert {model for model, _ in world.test_calls} == {
        NEW,
        OTHER,
    }  # the chain's own are not re-tested
    origin = cache.origin()
    assert origin["kind"] == "own" and origin["why"] == {"code": "test", "models": [NEW]}
    picture = chooser.picture(world.settings)
    assert picture["look"]["result"] == "started" and picture["look"]["reason"] == {
        "code": "gone",
        "models": [FIRST],
    }
    assert picture["last"]["by"] == "failure" and picture["last"]["outcome"] == "changed"
    assert picture["last"]["chain_before"] == [SECOND] and picture["last"]["kept"] == [SECOND]
    assert picture["test"] is None and picture["busy"] is False
    assert json.loads(cache.PATH.read_text(encoding="utf-8"))["chain"] == [SECOND, NEW]


async def test_a_rate_limited_model_that_is_still_listed_changes_nothing(world: World) -> None:
    chooser.note([failure(FIRST)], answered=True)
    await settled()
    assert world.chain == [FIRST, SECOND] and world.test_calls == []
    assert chooser.picture(world.settings)["look"]["result"] == "passing"


async def test_a_page_that_could_not_be_read_concludes_nothing(world: World) -> None:
    world.pages[FIRST] = "down"
    chooser.note([failure(FIRST, *GONE_404)], answered=False)
    await settled()
    assert world.chain == [FIRST, SECOND] and world.test_calls == []
    assert chooser.picture(world.settings)["gone"] == []


async def test_a_refused_key_starts_nothing(world: World) -> None:
    chooser.note(
        [failure(FIRST, "ProviderAuthError", False, "the API key was rejected (401)")],
        answered=False,
    )
    await settled()
    assert chooser.picture(world.settings)["look"] is None and world.asked == []


async def test_a_pinned_service_looks_into_nothing(world: World) -> None:
    chooser.configure(make_settings(openrouter_models=f"{FIRST},{SECOND}"))
    chooser.note([failure(FIRST, *GONE_404)], answered=False)
    await settled()
    assert (
        world.asked == []
        and chooser.picture(make_settings(openrouter_models=FIRST))["origin"]["kind"] == "pinned"
    )


async def test_a_burst_of_failures_leads_to_one_look_a_minute(world: World) -> None:
    world.pages[FIRST] = "gone"
    chooser.note([failure(FIRST, *GONE_404)], answered=True)
    chooser.note([failure(FIRST, *GONE_404)], answered=True)
    chooser.note([failure(SECOND)], answered=True)
    await settled()
    assert sum(1 for url in world.asked if "/model/" in url) == 1
    chooser.note([failure(SECOND)], answered=True)  # within the minute: dropped
    await settled()
    assert sum(1 for url in world.asked if "/model/" in url) == 1


async def test_a_failed_request_is_repaired_from_the_published_list_at_no_cost(
    world: World,
) -> None:
    tested = recent()
    world.published = published_file([SECOND, NEW], tested, world.settings)
    chooser.note([failure(FIRST, *GONE_404)], answered=False)
    await settled()
    assert world.chain == [SECOND, NEW] and world.test_calls == []
    assert chooser.picture(world.settings)["look"]["result"] == "published"
    assert chooser.picture(world.settings)["published"]["state"] == "used"


async def test_the_chain_that_has_just_failed_is_not_taken_again_from_the_published_list(
    world: World,
) -> None:
    tested = recent()
    world.published = published_file([FIRST, SECOND], tested, world.settings)
    chooser.note([failure(FIRST)], answered=True)
    await settled()
    assert world.chain == [FIRST, SECOND] and cache.origin() is None
    assert chooser.picture(world.settings)["published"]["state"] == "same"


async def test_a_closed_chain_with_a_free_place_gets_a_spare_beside_it(world: World) -> None:
    world.listed = [FIRST, SECOND, NEW]
    chooser.note([failure(FIRST, *FORBIDDEN), failure(SECOND, *FORBIDDEN)], answered=False)
    await settled()
    assert world.chain == [FIRST, SECOND, NEW]
    assert chooser.picture(world.settings)["look"]["reason"]["code"] == "closed"


async def test_a_full_chain_that_cannot_be_called_gets_no_test(world: World) -> None:
    cache.use(
        [FIRST, SECOND, OTHER],
        {
            "kind": "own",
            "tested": stamp(testing.default_measured_at()),
            "changed": "x",
            "why": {"code": "test", "models": [OTHER]},
        },
    )
    world.listed = [FIRST, SECOND, OTHER, NEW]
    chooser.note([failure(m, *FORBIDDEN) for m in (FIRST, SECOND, OTHER)], answered=False)
    await settled()
    assert world.chain == [FIRST, SECOND, OTHER] and world.test_calls == []
    assert chooser.picture(world.settings)["look"]["result"] == "no_place"


async def test_a_repair_that_was_refused_is_owed_and_tried_again_after_the_next_failure(
    world: World,
) -> None:
    world.pages[FIRST] = "gone"
    world.listed = [SECOND, NEW]
    world.allowance = Allowance(995, 1000, 5)  # too little left: the test is refused
    chooser.note([failure(FIRST, *GONE_404)], answered=True)
    await settled()
    assert world.chain == [SECOND] and world.test_calls == []
    picture = chooser.picture(world.settings)
    assert picture["look"]["result"] == "allowance" and picture["look"]["again_at"].endswith(
        "T00:00:00Z"
    )
    assert cache.origin()["why"] == {"code": "gone", "models": [FIRST]}

    # The next day, a passing failure on the model that is left: the owed repair starts.
    world.allowance = Allowance(10, 1000, 990)
    chooser._looked_at = -1e9
    chooser.note([failure(SECOND)], answered=True)
    await settled()
    assert world.chain == [SECOND, NEW]
    assert chooser.picture(world.settings)["last"]["reason"] == {"code": "gone", "models": [FIRST]}


# ---------------------------------------------------------------------------
# The test and the chain
# ---------------------------------------------------------------------------


async def test_the_button_tests_the_chain_and_the_newcomers_and_only_fills_the_free_place(
    world: World,
) -> None:
    world.listed = [FIRST, SECOND, NEW, OTHER, BROKEN]
    world.script = {FIRST: {"synthesis": "wrong"}, BROKEN: {"synthesis": "unusable"}}

    assert await chooser.start("page", {"code": "asked", "models": []}) == ("started", None)
    assert chooser.picture(world.settings)["doing"] == "testing"
    assert chooser.picture(world.settings)["button"]["reason"] == "running"
    await settled()

    # Four candidates at most: the chain first, then the newcomers in OpenRouter's order.
    assert [m for m, _ in world.test_calls if _ == "synthesis"] == [FIRST, SECOND, NEW, OTHER]
    assert world.chain == [FIRST, SECOND, NEW]  # a model of the chain keeps its place, wrong or not
    last = chooser.picture(world.settings)["last"]
    assert last["by"] == "page" and last["left_out"] == [BROKEN] and last["outcome"] == "changed"
    assert {row["model"]: row["placing"] for row in last["models"]} == {
        FIRST: "kept",
        SECOND: "kept",
        NEW: "added",
        OTHER: "no_place",
    }


async def test_the_models_left_out_go_first_next_time(world: World) -> None:
    world.listed = [FIRST, SECOND, NEW, OTHER, BROKEN]
    assert (await chooser.start("page", {"code": "asked", "models": []}))[0] == "started"
    await settled()
    chooser._test_ended_at = -1e9  # the pause is over
    world.test_calls.clear()
    assert (await chooser.start("page", {"code": "asked", "models": []}))[0] == "started"
    await settled()
    tested = chooser.picture(world.settings)["last"]["tested"]
    assert tested[:4] == [FIRST, SECOND, NEW, BROKEN]  # BROKEN waited; OTHER's turn is next


async def test_a_model_that_failed_is_not_tested_again_for_a_repair_until_it_passes(
    world: World,
) -> None:
    world.pages[FIRST] = "gone"
    world.listed = [SECOND, NEW, OTHER]
    world.script = {NEW: {"synthesis": "wrong"}, OTHER: {"synthesis": "lost"}}
    chooser.note([failure(FIRST, *GONE_404)], answered=True)
    await settled()
    assert world.chain == [SECOND]
    world.script = {}
    world.test_calls.clear()
    chooser._looked_at, chooser._test_ended_at = -1e9, -1e9
    chooser.note([failure(SECOND)], answered=True)  # the owed repair
    await settled()
    assert {m for m, _ in world.test_calls} == {OTHER}  # NEW failed here; OTHER was only weather
    assert world.chain == [SECOND, OTHER]


async def test_the_buttons_test_takes_out_a_model_that_is_gone_and_gives_no_answer(
    world: World,
) -> None:
    world.pages[FIRST] = "gone"
    world.listed = [SECOND, NEW]
    world.script = {FIRST: {"synthesis": "lost"}}
    assert (await chooser.start("page", {"code": "asked", "models": []}))[0] == "started"
    await settled()
    assert world.chain == [SECOND, NEW]
    rows = {
        row["model"]: row["placing"] for row in chooser.picture(world.settings)["last"]["models"]
    }
    assert rows[FIRST] == "dropped"


async def test_a_refused_key_stops_a_test_at_once(world: World) -> None:
    world.listed = [FIRST, SECOND, NEW]
    world.script = {SECOND: {"synthesis": "auth"}}
    assert (await chooser.start("page", {"code": "asked", "models": []}))[0] == "started"
    await settled()
    assert world.chain == [FIRST, SECOND] and NEW not in {m for m, _ in world.test_calls}
    assert chooser.picture(world.settings)["last"]["outcome"] == "key_refused"


async def test_a_test_that_breaks_changes_nothing_and_ends(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(settings: Settings, model: str, listener: Any) -> Any:
        raise RuntimeError("no client")

    monkeypatch.setattr(chooser, "chain_factory", broken)
    world.listed = [FIRST, SECOND, NEW]
    assert (await chooser.start("page", {"code": "asked", "models": []}))[0] == "started"
    await settled()
    picture = chooser.picture(world.settings)
    assert picture["last"]["outcome"] == "interrupted" and world.chain == [FIRST, SECOND]
    assert picture["test"] is None and picture["button"]["reason"] == "pause"


async def test_a_test_cut_off_by_shutdown_is_not_a_result(world: World) -> None:
    gate = asyncio.Event()

    async def hold() -> str:
        await gate.wait()
        return "right"

    world.listed = [FIRST, SECOND, NEW]
    world.script = {FIRST: {"synthesis": hold}}
    assert (await chooser.start("page", {"code": "asked", "models": []}))[0] == "started"
    await asyncio.sleep(0.01)
    assert chooser.picture(world.settings)["test"]["in_flight"][0]["model"] == FIRST
    await chooser.stop()
    picture = chooser.picture(world.settings)
    assert picture["test"] is None and world.chain == [FIRST, SECOND]
    assert picture["last"] is None  # a half-run test is not remembered as the last one


# ---------------------------------------------------------------------------
# Guards and the picture
# ---------------------------------------------------------------------------


async def test_the_button_is_refused_for_a_reason_and_answered_from_memory_until_it_changes(
    world: World,
) -> None:
    world.listed = TimeoutError("no list")
    code, again_at = await chooser.start("page", {"code": "asked", "models": []})
    assert code == "list_unreadable" and again_at > stamp(datetime.now(UTC))
    world.listed = [FIRST, SECOND, NEW]
    # What the last press was told is still true: asking again costs no request.
    world.asked.clear()
    assert (await chooser.start("page", {"code": "asked", "models": []}))[0] == "list_unreadable"
    assert world.asked == []
    # The repair is not held to the button's memory, and it tests only models outside the
    # chain: with none listed there is nothing to test (the button would test the chain's own).
    chooser._refusal = None
    world.listed = [FIRST, SECOND]
    assert (await chooser.start("failure", {"code": "gone", "models": [FIRST]}))[
        0
    ] == "nothing_to_test"


async def test_the_allowance_refuses_a_test_until_midnight(world: World) -> None:
    world.listed = [FIRST, SECOND, NEW]
    world.allowance = Allowance(990, 1000, 10)
    code, again_at = await chooser.start("page", {"code": "asked", "models": []})
    assert code == "allowance" and again_at.endswith("T00:00:00Z")
    chooser._refusal = None
    world.allowance = None
    assert (await chooser.start("page", {"code": "asked", "models": []}))[0] == "allowance_unknown"
    assert (await chooser.start("failure", {"code": "gone", "models": [FIRST]}))[0] == "started"
    await settled()


async def test_one_test_at_a_time_and_a_quarter_of_an_hour_between_tests(world: World) -> None:
    gate = asyncio.Event()

    async def hold() -> str:
        await gate.wait()
        return "right"

    world.listed = [FIRST, SECOND, NEW]
    world.script = {NEW: {"synthesis": hold}}
    assert (await chooser.start("page", {"code": "asked", "models": []}))[0] == "started"
    await asyncio.sleep(0.01)
    assert await chooser.start("page", {"code": "asked", "models": []}) == ("running", None)
    assert await chooser.start("failure", {"code": "gone", "models": []}) == ("running", None)
    gate.set()
    await settled()
    code, again_at = await chooser.start("page", {"code": "asked", "models": []})
    assert code == "pause"
    assert again_at <= stamp(datetime.now(UTC) + timedelta(seconds=chooser.PAUSE_SECONDS + 1))
    assert (await chooser.start("failure", {"code": "gone", "models": []}))[0] == "pause"


async def test_the_button_may_start_only_so_many_tests_a_day(world: World) -> None:
    world.listed = [FIRST, SECOND, NEW, OTHER]  # NEW takes the free place; OTHER stays outside
    for _ in range(chooser.PAGE_TESTS_A_DAY):
        chooser._test_ended_at = -1e9  # the pause is over
        assert (await chooser.start("page", {"code": "asked", "models": []}))[0] == "started"
        await settled()
    chooser._test_ended_at = -1e9
    code, again_at = await chooser.start("page", {"code": "asked", "models": []})
    assert code == "page_limit" and again_at.endswith("T00:00:00Z")
    assert chooser.picture(world.settings)["button"]["reason"] == "page_limit"
    # The repair after a failed request is the owner's own upkeep: not held to the button's share.
    world.pages[FIRST] = "gone"
    assert (await chooser.start("failure", {"code": "gone", "models": [FIRST]}))[0] == "started"
    await settled()
    # Yesterday's presses do not count.
    chooser._page_starts[:] = [moment - timedelta(days=1) for moment in chooser._page_starts]
    chooser._test_ended_at = -1e9
    assert (await chooser.start("page", {"code": "asked", "models": []}))[0] == "started"
    await settled()


async def test_the_picture_has_the_shape_the_page_draws_from(world: World) -> None:
    picture = chooser.picture(world.settings)
    assert set(picture) == {
        "now",
        "awake_since",
        "mode",
        "chain",
        "origin",
        "gone",
        "published",
        "busy",
        "doing",
        "look",
        "test",
        "last",
        "button",
        "rules",
    }
    assert set(picture["origin"]) == {"kind", "tested", "age_days", "changed", "why"}
    assert picture["origin"]["kind"] == "default" and picture["origin"]["tested"] == stamp(
        testing.default_measured_at()
    )
    assert set(picture["published"]) == {"state", "tested"}
    assert set(picture["button"]) == {"offered", "ready", "reason", "again_at"}
    assert picture["button"] == {"offered": True, "ready": True, "reason": None, "again_at": None}
    assert picture["rules"] == {
        "cases": 3,
        "places": 3,
        "pause_minutes": 15,
        "page_tests_a_day": 4,
        "timeout_seconds": 60.0,
        "requests_at_most": 12,
    }
    assert picture["mode"] == "automatic" and picture["chain"] == [FIRST, SECOND]
    assert picture["busy"] is False and picture["doing"] is None and picture["look"] is None
    assert picture["test"] is None and picture["last"] is None and picture["gone"] == []
    json.dumps(picture)  # the route serialises it as it is
    pinned = chooser.picture(make_settings(openrouter_models=f"{SECOND},{FIRST}"))
    assert pinned["origin"] == {
        "kind": "pinned",
        "tested": None,
        "age_days": None,
        "changed": None,
        "why": None,
    }
    assert pinned["button"] == {
        "offered": False,
        "ready": False,
        "reason": "pinned",
        "again_at": None,
    }
    assert pinned["chain"] == [SECOND, FIRST]


async def test_the_newer_of_the_two_real_tests_is_the_one_shown(world: World) -> None:
    tested = recent()
    test = {
        "started": stamp(tested - timedelta(minutes=2)),
        "tested": [FIRST, SECOND],
        "timeout_seconds": 60,
        "calls": [
            {
                "model": FIRST,
                "case": case,
                "outcome": "answered",
                "correct": True,
                "seconds": 2.0,
                "error_type": None,
            }
            for case in cases.case_ids()
        ],
        "chain_before": [FIRST, SECOND],
        "found": [FIRST, SECOND],
    }
    world.published = published_file([FIRST, SECOND], tested, world.settings, test=test)
    await chooser.at_startup(world.settings)
    shown = chooser.picture(world.settings)["last"]
    assert shown["by"] == "schedule" and shown["finished"] == stamp(tested)
    assert {row["model"]: row["verdict"] for row in shown["models"]} == {
        FIRST: "passed",
        SECOND: "not_judged",
    }

    chooser._test_ended_at = -1e9
    world.listed = [FIRST, SECOND, NEW]
    assert (await chooser.start("page", {"code": "asked", "models": []}))[0] == "started"
    await settled()
    assert chooser.picture(world.settings)["last"]["by"] == "page"


def test_the_page_has_a_code_for_everything_the_service_can_say() -> None:
    for name in (
        "REFUSALS",
        "ORIGINS",
        "ORIGIN_REASONS",
        "LOOK_RESULTS",
        "PUBLISHED_STATES",
        "DOING",
    ):
        codes = getattr(chooser, name)
        assert len(set(codes)) == len(codes) and all(code.isidentifier() for code in codes)
    assert set(chooser._WHY_NO_TEST) <= set(chooser.REFUSALS)
