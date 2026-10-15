"""One test of the free models: send the three cases, record every call, decide.

A test sends each candidate the three cases of :mod:`app.models.cases` through the service's
own chain client (:class:`app.services.llm_client.OpenAICompatibleChain`, one model at a time
with no fallback, so a result here is what the service would get from that model on its own),
and judges what comes back by the rules in :mod:`app.models.rules`. Everything it learns goes
into one plain dictionary, the *record*. The page draws its board from that record and the
scheduled job publishes it. One format, two sources.

Two callers run tests: the running service (:mod:`app.models.chooser`), which puts the result
to use, and the scheduled job (:mod:`app.models.published`), which publishes it. Nothing here
changes the chain in use.

Candidates are taken one after the other, and request starts are kept
:data:`app.models.rules.SECONDS_BETWEEN_STARTS` apart, so a test never crowds out a visitor's
run. Project 1 of this portfolio let a model answer real requests as a *stand-in* while its
test went on, because seven documents took minutes. Three cases take under a minute per
candidate, so stand-ins are left out here on purpose: a model answers for the service only
once a test has judged it.

The chain client is built by a factory that tests replace, so the suite never touches the
network. A :class:`app.errors.ProviderAuthError` stops a test at once (outcome
``key_refused``): no model can fix a refused key, and every call spends allowance.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from typing import Any

from app import budget
from app.config import DEFAULT_MODEL_CHAIN, Settings, parse_model_chain
from app.errors import ProviderAuthError, ProviderError
from app.models import cases, rules
from app.models.cache import is_free_model
from app.services.llm_client import (
    FailureListener,
    OpenAICompatibleChain,
    StructuredChat,
    reasoning_body,
)

logger = logging.getLogger(__name__)

STAMP = "%Y-%m-%dT%H:%M:%SZ"  #: how a moment is written for the page and in files (UTC)

#: The day the built-in default chain was measured by hand (``app/config.py``). A published
#: list tested after that day began is newer than the default: a list made before the default
#: was pushed carries another fingerprint (the default chain is part of it) and is never
#: adopted, so the start of the day is the safe moment, and a job that runs on the day of the
#: push publishes a list the service takes up.
DEFAULT_MEASURED_ON = "2026-10-15"

#: Who started a test, why, and how it ended. The page has words for every code.
STARTERS = ("failure", "page", "schedule")
TEST_REASONS = ("gone", "closed", "asked", "schedule")
OUTCOMES = ("changed", "unchanged", "nothing_passed", "key_refused", "interrupted")

#: The fields of one call that the page and the published file may hold. The chain's sanitised
#: message is not among them: it can name a setting, and visitors read this.
CALL_FIELDS = ("model", "case", "outcome", "correct", "seconds", "error_type")
CALL_OUTCOMES = ("answered", "refused", "no_answer")
#: The error types a call may carry: the service's own exception names, plus the two the
#: chain reports without an exception (invalid output, a content filter).
ERROR_TYPES = (
    "ProviderError",
    "ProviderAuthError",
    "SchemaValidationError",
    "ContentRefused",
    "NotFree",
)

#: Builds the chain client for one candidate: ``factory(settings, model, listener)``.
ChainFactory = Callable[[Settings, str, FailureListener], StructuredChat]
Sleep = Callable[[float], Awaitable[None]]


def now() -> datetime:
    return datetime.now(UTC)


def stamp(moment: datetime) -> str:
    return moment.strftime(STAMP)


def default_chain() -> list[str]:
    return parse_model_chain(DEFAULT_MODEL_CHAIN)


def default_measured_at() -> datetime:
    """The first second of the day the default chain was measured."""
    return datetime.strptime(DEFAULT_MEASURED_ON, "%Y-%m-%d").replace(tzinfo=UTC)


def default_chain_factory(
    settings: Settings, model: str, listener: FailureListener
) -> StructuredChat:
    """The real client: OpenRouter, one free model, no pacing of its own (the test paces).

    The request is the service's own in everything else, the reasoning control included
    (``OPENROUTER_REASONING``): a model measured with reasoning left on is not the model the
    service calls with it off.
    """
    headers: dict[str, str] = {}
    if settings.openrouter_site_url:
        headers["HTTP-Referer"] = settings.openrouter_site_url
    if settings.openrouter_app_name:
        headers["X-Title"] = settings.openrouter_app_name
    return OpenAICompatibleChain(
        provider="openrouter",
        api_key=settings.secret("openrouter_api_key"),
        base_url=settings.openrouter_base_url,
        models=[model],
        requests_per_minute=0,
        strict_schema=settings.openrouter_strict_json_schema,
        free_only=True,
        timeout_s=settings.openrouter_timeout_s,
        headers=headers,
        failure_listener=listener,
        extra_body=reasoning_body(settings.openrouter_reasoning),
    )


def planned_requests(tested: Sequence[str]) -> int:
    """How many free requests a test of ``tested`` sends at most."""
    return len(tested) * len(cases.CASES)


async def why_no_test(settings: Settings, by: str, planned: int) -> str | None:
    """Why a test of ``planned`` requests may not start now, or None when it may.

    OpenRouter's own count of today's free requests is asked (``GET /key``, the one request
    that carries the key, sent to OpenRouter's address only). The test is refused when fewer
    than ``planned`` plus the reserve would be left: the reserve is
    ``OPENROUTER_ALLOWANCE_RESERVE``, never more than a fifth of the day, the same rule
    :class:`app.budget.RunBudget` applies to visitors' runs. When the count cannot be read,
    only the button (``by == "page"``) is refused: a stranger may press it, while the repair
    and the scheduled job are the owner's own upkeep, each bounded by the size of one test.
    With ``OPENROUTER_ALLOWANCE_CHECK`` off nothing is asked and nothing is refused.
    """
    if not settings.openrouter_allowance_check:
        return None
    allowance = await budget.read_openrouter_allowance(settings)
    if allowance is None:
        return "allowance_unknown" if by == "page" else None
    reserve = min(settings.openrouter_allowance_reserve, allowance.limit // 5)
    if allowance.remaining < planned + reserve:
        return "allowance"
    return None


def new_record(
    by: str,
    reason: dict[str, Any],
    chain: Sequence[str],
    kept: Sequence[str],
    gone: Sequence[str],
    listed: Sequence[str],
    tested: Sequence[str],
    left_out: Sequence[str],
    settings: Settings,
) -> dict[str, Any]:
    """The record of a test that is about to run.

    ``chain`` is the chain in use, ``kept`` its models that OpenRouter still offers, ``gone``
    the ones it does not, ``listed`` what OpenRouter lists, ``tested`` the models this test
    will call and ``left_out`` those it has no room for.
    """
    started = now()
    ids = cases.case_ids()
    return {
        "id": started.strftime("%Y%m%dT%H%M%SZ"),
        "by": by,
        "reason": reason,
        "state": "running",
        "started": stamp(started),
        "finished": None,
        "found": list(listed),
        "tested": list(tested),
        "left_out": list(left_out),
        "gone": list(gone),
        "cases": [{"id": case["id"], "title": case["title"]} for case in cases.CASES],
        "timeout_seconds": settings.openrouter_timeout_s,
        "calls": [],
        "in_flight": [],
        "models": rules.judge(tested, [], ids, settings.openrouter_timeout_s),
        "chain_before": list(chain),
        "kept": list(kept),
        "chain": list(chain),
        "outcome": None,
        "requests": 0,
    }


class _Pace:
    """Keeps request starts a fixed number of seconds apart, across all candidates."""

    def __init__(self, gap: float, sleep: Sleep) -> None:
        self.gap = gap
        self.sleep = sleep
        self._last: float | None = None

    async def wait(self) -> None:
        if self._last is not None and self.gap > 0:
            delay = self._last + self.gap - time.monotonic()
            if delay > 0:
                await self.sleep(delay)
        self._last = time.monotonic()


async def _one_call(
    chain: StructuredChat, case: dict[str, Any], failures: list[Any]
) -> tuple[dict[str, Any], bool]:
    """Send one case to one model. Returns the call row and whether the key was refused."""
    failures.clear()
    started = time.perf_counter()
    row: dict[str, Any] = {"model": "", "case": case["id"]}
    key_refused = False
    try:
        outcome = await chain.complete(
            messages=cases.messages_of(case),
            schema_model=case["schema_model"],
            temperature=case["temperature"],
            max_tokens=case["max_tokens"],
            schema_name=case["id"],
        )
    except ProviderAuthError:
        key_refused = True
        row.update(outcome="no_answer", correct=None, error_type="ProviderAuthError")
    except ProviderError:
        # The chain reports what went wrong on the model through the listener (a rate limit,
        # invalid output); the exception it raises sums every model up, and is not kept.
        kind = failures[-1].error_type if failures else "ProviderError"
        row.update(outcome="no_answer", correct=None, error_type=kind)
    else:
        if outcome.refused:
            row.update(outcome="refused", correct=None, error_type="ContentRefused")
        else:
            try:
                correct = bool(case["grade"](outcome.parsed))
            except Exception:  # noqa: BLE001 - a grade that breaks on an odd answer is "wrong"
                correct = False
            row.update(outcome="answered", correct=correct, error_type=None)
    row["seconds"] = round(time.perf_counter() - started, 2)
    return row, key_refused


async def run_test(
    record: dict[str, Any],
    settings: Settings,
    *,
    chain_factory: ChainFactory | None = None,
    sleep: Sleep = asyncio.sleep,
) -> None:
    """Send the cases, judge, and decide the chain. Fills in ``record``.

    Candidates are taken in the record's order, one at a time. A candidate's test ends as soon
    as its verdict is settled (one wrong or unusable answer fails it, one unanswered call
    leaves it unjudged, three right answers pass it), so no call is spent on a model that can
    no longer pass. The record's ``chain`` says what the chain should be when the test is over.
    Applying it is the caller's business.
    """
    if not all(is_free_model(model) for model in record["tested"]):
        # The last door before a model is called, as cache.use() is the last door before one
        # is chosen: whatever a list said, only free ids are sent a case.
        raise ValueError("only ids ending in ':free' are ever tested")
    factory = chain_factory or default_chain_factory
    ids = [case["id"] for case in record["cases"]]
    to_send = [cases.by_id(case_id) for case_id in ids]
    pace = _Pace(rules.SECONDS_BETWEEN_STARTS, sleep)
    key_refused = False
    failures: list[Any] = []

    def listener(failed: Sequence[Any], answered: bool) -> None:
        failures.extend(failed)

    try:
        for model in record["tested"]:
            chain = factory(settings, model, listener)
            for case in to_send:
                await pace.wait()
                record["requests"] += 1
                record["in_flight"] = [{"model": model, "case": case["id"], "since": stamp(now())}]
                try:
                    call, key_refused = await _one_call(chain, case, failures)
                finally:
                    record["in_flight"] = []
                call["model"] = model
                record["calls"].append({key: call[key] for key in CALL_FIELDS})
                record["models"] = rules.judge(
                    record["tested"], record["calls"], ids, record["timeout_seconds"]
                )
                # The error's type, never the chain's sentence: that one can name a setting.
                logger.info(
                    "test %s: %s on %s: %s, %s s%s",
                    record["id"],
                    model,
                    case["id"],
                    {True: "right", False: "wrong", None: "no answer"}[call["correct"]],
                    call["seconds"],
                    f" ({call['error_type']})" if call["error_type"] else "",
                )
                if key_refused:
                    break
                row = next(row for row in record["models"] if row["model"] == model)
                if row["verdict"] is not None:
                    break  # settled: passed every case, or can no longer pass this test
            if key_refused:
                break
    finally:
        record["in_flight"] = []
    rows = rules.judge(
        record["tested"], record["calls"], ids, record["timeout_seconds"], finished=True
    )
    before = record["chain_before"]
    if key_refused:
        chain_after, outcome = list(before), "key_refused"
    else:
        # Which of the chain's own models leave is for the rules to say (a model OpenRouter no
        # longer offers stays when it still answered here).
        chain_after = rules.next_chain(
            before, rows, record["gone"], record["timeout_seconds"], measured=default_chain()
        )
        passed = any(row["verdict"] in ("passed", "slow") for row in rows)
        outcome = (
            "changed"
            if chain_after != list(before)
            else "unchanged"
            if passed
            else "nothing_passed"
        )
    rules.place(rows, before, chain_after)
    record.update(
        models=rows, chain=chain_after, outcome=outcome, state="done", finished=stamp(now())
    )
