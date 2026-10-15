"""Tests for one test of the free models (``app.models.testing``).

The chain client is replaced by a scripted fake handed in through the factory, so nothing here
reaches the network. Everything the file needs is defined in it.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

import pytest

from app import budget
from app.budget import Allowance
from app.config import DEFAULT_MODEL_CHAIN, Settings, parse_model_chain
from app.errors import ProviderAuthError, ProviderError
from app.models import cases, rules, testing
from app.schemas import ComplianceAuditResult, ViolationItem
from app.services.llm_client import Attempt, ChatOutcome, ModelFailure

FIRST, SECOND = parse_model_chain(DEFAULT_MODEL_CHAIN)[:2]
NEW, OTHER, LOST, BROKEN = (
    "vendor/new:free",
    "vendor/other:free",
    "vendor/lost:free",
    "vendor/broken:free",
)


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "environment": "test",
        "checkpointer": "memory",
        "openrouter_api_key": "sk-or-test",
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
                field_name="web_title",
                violating_text="Magnesium",
                regulation_rule="DOSAGE_ACCURACY",
                required_fix="None needed.",
            )
        ],
        revision_summary="Wrongly flagged.",
    )


class FakeChain:
    """Answers each case from ``script[model][case]`` and tells the listener what the real chain would.

    Words: ``right``, ``wrong``, ``lost`` (rate limited), ``gone`` (404), ``unusable`` (invalid
    output), ``refused`` (content filter), ``auth`` (the key was rejected). A callable in the
    script is awaited first, so a test can hold a call open.
    """

    name = "openrouter"

    def __init__(
        self, model: str, script: dict[str, dict[str, Any]], listener: Any, log: list
    ) -> None:
        self.model = model
        self.script = script
        self.listener = listener
        self.log = log

    async def complete(
        self,
        *,
        messages: Any,
        schema_model: Any,
        temperature: float | None,
        max_tokens: int,
        schema_name: str = "output",
    ) -> ChatOutcome:
        self.log.append((self.model, schema_name))
        what = self.script.get(self.model, {}).get(schema_name, "right")
        if callable(what):
            what = await what()
        if what == "auth":
            raise ProviderAuthError("openrouter-auth", "the API key was rejected (401)")
        if what in ("lost", "gone", "unusable"):
            kind, retryable, message = {
                "lost": ("ProviderError", True, "rate limited (429)"),
                "gone": ("ProviderError", False, "model not found (404); withdrawn?"),
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


def factory_for(script: dict[str, dict[str, Any]], log: list | None = None) -> testing.ChainFactory:
    log = [] if log is None else log

    def factory(settings: Settings, model: str, listener: Any) -> FakeChain:
        return FakeChain(model, script, listener, log)

    return factory


async def no_sleep(seconds: float) -> None:
    return None


def record_for(
    tested: list[str],
    chain: list[str] | None = None,
    gone: list[str] = (),
    settings: Settings | None = None,
) -> dict[str, Any]:
    chain = [FIRST, SECOND] if chain is None else chain
    settings = settings or make_settings()
    kept = [model for model in chain if model not in gone]
    return testing.new_record(
        "page",
        {"code": "asked", "models": []},
        chain,
        kept,
        list(gone),
        tested,
        tested,
        [],
        settings,
    )


@pytest.fixture(autouse=True)
def quick(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rules, "SECONDS_BETWEEN_STARTS", 0.0)


# ---------------------------------------------------------------------------
# A full test run
# ---------------------------------------------------------------------------


async def test_a_full_run_judges_every_candidate_and_repairs_the_chain() -> None:
    """One model passes, one answers wrongly, one is rate limited; the gone model leaves."""
    log: list = []
    script = {OTHER: {"audit-catches": "wrong"}, LOST: {"synthesis": "lost"}}
    record = record_for([NEW, OTHER, LOST], gone=[FIRST])

    await testing.run_test(record, make_settings(), chain_factory=factory_for(script, log))

    verdicts = {
        row["model"]: (row["verdict"], row["why"], row["placing"]) for row in record["models"]
    }
    assert verdicts == {
        NEW: ("passed", "all_right", "added"),
        OTHER: ("failed", "wrong_answers", None),
        LOST: ("not_judged", "no_answers", None),
    }
    # A model's test ends as soon as its verdict is settled: no call is spent for nothing.
    assert log == [
        (NEW, "synthesis"),
        (NEW, "audit-catches"),
        (NEW, "audit-passes"),
        (OTHER, "synthesis"),
        (OTHER, "audit-catches"),
        (LOST, "synthesis"),
    ]
    assert record["requests"] == 6 and len(record["calls"]) == 6
    assert record["chain_before"] == [FIRST, SECOND] and record["chain"] == [SECOND, NEW]
    assert record["outcome"] == "changed" and record["state"] == "done"
    assert record["in_flight"] == [] and record["finished"] is not None
    assert {tuple(call) for call in record["calls"]} == {testing.CALL_FIELDS}
    lost_call = next(call for call in record["calls"] if call["model"] == LOST)
    assert lost_call == {
        "model": LOST,
        "case": "synthesis",
        "outcome": "no_answer",
        "correct": None,
        "seconds": lost_call["seconds"],
        "error_type": "ProviderError",
    }
    wrong_call = next(
        call
        for call in record["calls"]
        if call["model"] == OTHER and call["case"] == "audit-catches"
    )
    assert wrong_call["outcome"] == "answered" and wrong_call["correct"] is False


def test_the_record_has_the_shape_the_page_and_the_published_file_expect() -> None:
    record = record_for([NEW], settings=make_settings(openrouter_timeout_s=45))
    assert list(record) == [
        "id", "by", "reason", "state", "started", "finished", "found", "tested", "left_out", "gone",
        "cases", "timeout_seconds", "calls", "in_flight", "models", "chain_before", "kept", "chain",
        "outcome", "requests",
    ]  # fmt: skip
    assert record["state"] == "running" and record["outcome"] is None
    assert record["timeout_seconds"] == 45 and record["kept"] == [FIRST, SECOND]
    assert [case["id"] for case in record["cases"]] == cases.case_ids()
    assert record["models"][0]["verdict"] is None
    assert record["by"] in testing.STARTERS and record["reason"]["code"] in testing.TEST_REASONS


async def test_outcomes_unchanged_and_nothing_passed() -> None:
    settled = record_for([FIRST, NEW], chain=[FIRST, SECOND, OTHER])
    await testing.run_test(settled, make_settings(), chain_factory=factory_for({}))
    assert settled["outcome"] == "unchanged" and settled["chain"] == [FIRST, SECOND, OTHER]
    nothing = record_for([NEW])
    await testing.run_test(
        nothing, make_settings(), chain_factory=factory_for({NEW: {"synthesis": "lost"}})
    )
    assert nothing["outcome"] == "nothing_passed" and nothing["chain"] == [FIRST, SECOND]
    # A gone model that was not tested gave no answer: it leaves, and that is a change.
    dropped = record_for([NEW], gone=[FIRST])
    await testing.run_test(
        dropped, make_settings(), chain_factory=factory_for({NEW: {"synthesis": "lost"}})
    )
    assert dropped["outcome"] == "changed" and dropped["chain"] == [SECOND]
    assert set(testing.OUTCOMES) >= {settled["outcome"], nothing["outcome"], "key_refused"}


async def test_a_refused_key_stops_the_test_at_once_and_changes_nothing() -> None:
    log: list = []
    script = {OTHER: {"synthesis": "auth"}}
    record = record_for([NEW, OTHER, LOST])

    await testing.run_test(record, make_settings(), chain_factory=factory_for(script, log))

    assert record["outcome"] == "key_refused" and record["chain"] == [FIRST, SECOND]
    assert log[-1] == (OTHER, "synthesis") and LOST not in {model for model, _ in log}
    assert record["calls"][-1]["error_type"] == "ProviderAuthError"
    assert {row["model"]: row["verdict"] for row in record["models"]}[NEW] == "passed"
    # It passed, and the chain it would have joined was not touched: no place for it.
    assert {row["model"]: row["placing"] for row in record["models"]}[NEW] == "no_place"


async def test_refused_and_unusable_answers_fail_a_model_with_the_right_error_type() -> None:
    script = {NEW: {"synthesis": "refused"}, OTHER: {"synthesis": "unusable"}}
    record = record_for([NEW, OTHER])
    await testing.run_test(record, make_settings(), chain_factory=factory_for(script))
    calls = {call["model"]: call for call in record["calls"]}
    assert calls[NEW]["outcome"] == "refused" and calls[NEW]["error_type"] == "ContentRefused"
    assert (
        calls[OTHER]["outcome"] == "no_answer"
        and calls[OTHER]["error_type"] == "SchemaValidationError"
    )
    for row in record["models"]:
        assert (row["verdict"], row["why"]) == ("failed", "unusable_answers")
    assert set(call["error_type"] for call in record["calls"]) <= set(testing.ERROR_TYPES)


async def test_a_grade_that_breaks_on_an_odd_answer_is_a_wrong_answer_not_a_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(parsed: object) -> bool:
        raise RuntimeError("odd answer")

    broken = {**cases.by_id("synthesis"), "grade": explode}
    monkeypatch.setattr(
        cases,
        "by_id",
        lambda case_id: (
            broken if case_id == "synthesis" else cases.CASES[cases.case_ids().index(case_id)]
        ),
    )
    record = record_for([NEW])
    await testing.run_test(record, make_settings(), chain_factory=factory_for({}))
    assert record["calls"][0]["correct"] is False and record["models"][0]["verdict"] == "failed"


async def test_only_free_ids_are_ever_sent_a_case() -> None:
    record = record_for([NEW, "vendor/paid"])
    with pytest.raises(ValueError, match=":free"):
        await testing.run_test(record, make_settings(), chain_factory=factory_for({}))


async def test_request_starts_are_paced_across_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rules, "SECONDS_BETWEEN_STARTS", 6.0)
    slept: list[float] = []

    async def sleep(seconds: float) -> None:
        slept.append(seconds)

    record = record_for([NEW, OTHER])
    await testing.run_test(record, make_settings(), chain_factory=factory_for({}), sleep=sleep)
    assert record["requests"] == 6
    # The first request goes out at once; every later one waits for the gap.
    assert len(slept) == 5 and all(5.5 < wait <= 6.0 for wait in slept)


async def test_a_cancelled_test_leaves_the_record_running_and_raises() -> None:
    gate = asyncio.Event()

    async def hold() -> str:
        await gate.wait()
        return "right"

    record = record_for([NEW])
    task = asyncio.create_task(
        testing.run_test(
            record, make_settings(), chain_factory=factory_for({NEW: {"synthesis": hold}})
        )
    )
    await asyncio.sleep(0)
    assert record["in_flight"] and record["in_flight"][0]["model"] == NEW
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert record["state"] == "running" and record["in_flight"] == []


async def test_a_factory_that_breaks_breaks_the_test_and_the_caller_hears_it() -> None:
    def factory(settings: Settings, model: str, listener: Any) -> Any:
        raise RuntimeError("no client")

    record = record_for([NEW])
    with pytest.raises(RuntimeError):
        await testing.run_test(record, make_settings(), chain_factory=factory)
    assert record["state"] == "running"


# ---------------------------------------------------------------------------
# The real factory, and the allowance
# ---------------------------------------------------------------------------


def test_the_default_factory_builds_a_one_model_free_only_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import openai

    built: list[dict[str, Any]] = []

    class FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            built.append(kwargs)

    monkeypatch.setattr(openai, "AsyncOpenAI", FakeOpenAI)
    settings = make_settings(
        openrouter_strict_json_schema=False,
        openrouter_timeout_s=42,
        openrouter_site_url="https://example.test",
    )
    chain = testing.default_chain_factory(settings, NEW, lambda failures, answered: None)

    assert chain.models == [NEW] and chain.free_only is True and chain.strict_schema is False
    assert chain.limiter.interval == 0.0  # the test paces; the client does not
    # The service's reasoning control travels with every test request, as with a real run.
    assert chain.extra_body == {"reasoning": {"enabled": False}}
    ((kwargs,),) = [built]
    assert kwargs["base_url"] == settings.openrouter_base_url and kwargs["timeout"] == 42
    assert kwargs["max_retries"] == 0
    assert kwargs["default_headers"] == {
        "HTTP-Referer": "https://example.test",
        "X-Title": "multi-agent-ops-pipeline",
    }
    assert kwargs["api_key"] == "sk-or-test"
    with_effort = testing.default_chain_factory(
        make_settings(openrouter_reasoning="low"), NEW, lambda failures, answered: None
    )
    assert with_effort.extra_body == {"reasoning": {"effort": "low"}}


def test_planned_requests_and_the_default_chain() -> None:
    assert testing.planned_requests([NEW, OTHER]) == 2 * len(cases.CASES) == 6
    assert testing.default_chain() == parse_model_chain(DEFAULT_MODEL_CHAIN)
    assert (
        testing.stamp(testing.default_measured_at()) == f"{testing.DEFAULT_MEASURED_ON}T00:00:00Z"
    )
    assert testing.default_measured_at().tzinfo is not None


def allowance_reader(answer: Allowance | None) -> Callable[..., Any]:
    async def read(settings: Settings) -> Allowance | None:
        return answer

    return read


async def test_the_allowance_decides_whether_a_test_may_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = make_settings(openrouter_allowance_reserve=100)
    # 1000 a day: the reserve is 100, so 12 requests need 112 left.
    monkeypatch.setattr(
        budget, "read_openrouter_allowance", allowance_reader(Allowance(888, 1000, 112))
    )
    assert await testing.why_no_test(settings, "page", 12) is None
    monkeypatch.setattr(
        budget, "read_openrouter_allowance", allowance_reader(Allowance(889, 1000, 111))
    )
    assert await testing.why_no_test(settings, "page", 12) == "allowance"
    assert await testing.why_no_test(settings, "failure", 12) == "allowance"
    assert await testing.why_no_test(settings, "schedule", 12) == "allowance"
    # A 50-a-day account keeps a fifth of its day, not the whole reserve.
    monkeypatch.setattr(
        budget, "read_openrouter_allowance", allowance_reader(Allowance(28, 50, 22))
    )
    assert await testing.why_no_test(settings, "failure", 12) is None
    monkeypatch.setattr(
        budget, "read_openrouter_allowance", allowance_reader(Allowance(29, 50, 21))
    )
    assert await testing.why_no_test(settings, "failure", 12) == "allowance"


async def test_only_the_button_is_refused_when_openrouter_cannot_say_what_is_left(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(budget, "read_openrouter_allowance", allowance_reader(None))
    settings = make_settings()
    assert await testing.why_no_test(settings, "page", 12) == "allowance_unknown"
    assert await testing.why_no_test(settings, "failure", 12) is None
    assert await testing.why_no_test(settings, "schedule", 12) is None


async def test_with_the_allowance_check_off_nothing_is_asked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def never(settings: Settings) -> Allowance | None:
        raise AssertionError("the allowance must not be read")

    monkeypatch.setattr(budget, "read_openrouter_allowance", never)
    assert (
        await testing.why_no_test(make_settings(openrouter_allowance_check=False), "page", 12)
        is None
    )
