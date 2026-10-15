"""The demo-mode run budget, driven by a clock the test moves by hand.

Five rules in order: the visitor's minute, the visitor's day, everyone's minute, everyone's
day, then OpenRouter's own count of free requests read through a fake reader. Nothing here
touches a socket: the reader is a coroutine the test controls.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.budget import ALLOWANCE_TTL_S, Allowance, BudgetExceeded, RunBudget

from .conftest import make_settings

NOON = 1_760_011_200.0  # 2025-10-09T12:00:00Z: twelve hours from midnight either way
SECONDS_TO_MIDNIGHT_AT_NOON = 12 * 60 * 60


class Clock:
    """A monotonic clock and a wall clock that move together."""

    def __init__(self) -> None:
        self.mono = 1_000.0
        self.wall = NOON

    def monotonic(self) -> float:
        return self.mono

    def time(self) -> float:
        return self.wall

    def advance(self, seconds: float) -> None:
        self.mono += seconds
        self.wall += seconds


def _budget(clock: Clock, reader: Any = None, **overrides: Any) -> RunBudget:
    values: dict[str, Any] = {
        "run_rate_limit_per_minute": 10,
        "run_daily_limit_per_client": 10,
        "run_global_limit_per_minute": 10,
        "run_daily_limit": 10,
        "openrouter_allowance_reserve": 5,
    }
    values.update(overrides)
    return RunBudget(
        make_settings(**values), allowance_reader=reader, clock=clock.monotonic, wall=clock.time
    )


async def _refusal(budget: RunBudget, client: str, **kwargs: Any) -> BudgetExceeded:
    with pytest.raises(BudgetExceeded) as info:
        await budget.admit(client, **kwargs)
    return info.value


# ---------------------------------------------------------------------- per client


async def test_each_address_gets_its_runs_per_minute() -> None:
    clock = Clock()
    budget = _budget(clock, run_rate_limit_per_minute=2)
    await budget.admit("a")
    clock.advance(15)
    await budget.admit("a")

    refused = await _refusal(budget, "a")
    assert "this address" in refused.message and "minute" in refused.message
    assert refused.retry_after_s == 45  # the oldest run leaves the minute at second 60
    await budget.admit("b")  # another visitor is unaffected

    clock.advance(44.2)
    assert (await _refusal(budget, "a")).retry_after_s == 1  # a fraction is rounded up, never 0
    clock.advance(1)
    await budget.admit("a")  # the first run has left the window


async def test_each_address_gets_its_runs_per_day() -> None:
    clock = Clock()
    budget = _budget(clock, run_daily_limit_per_client=2)
    await budget.admit("a")
    clock.advance(61)
    await budget.admit("a")
    clock.advance(61)

    refused = await _refusal(budget, "a")
    assert "for today" in refused.message
    assert refused.retry_after_s == SECONDS_TO_MIDNIGHT_AT_NOON - 122
    await budget.admit("b")

    clock.advance(refused.retry_after_s)  # midnight UTC: a new day for everyone
    await budget.admit("a")
    assert budget.snapshot()["used_today"] == 1


# ------------------------------------------------------------------------ for everyone


async def test_everyone_shares_a_minute() -> None:
    clock = Clock()
    budget = _budget(clock, run_global_limit_per_minute=2)
    await budget.admit("a")
    clock.advance(20)
    await budget.admit("b")

    refused = await _refusal(budget, "c")
    assert "busy" in refused.message and refused.retry_after_s == 40
    clock.advance(40)
    await budget.admit("c")


async def test_everyone_shares_a_day() -> None:
    clock = Clock()
    budget = _budget(clock, run_daily_limit=2)
    await budget.admit("a")
    clock.advance(61)
    await budget.admit("b")
    clock.advance(61)

    refused = await _refusal(budget, "c")
    assert "daily run limit" in refused.message
    assert 1 <= refused.retry_after_s <= 86_400
    assert budget.snapshot()["used_today"] == 2
    clock.advance(86_400)
    await budget.admit("c")
    assert budget.snapshot()["used_today"] == 1


async def test_retry_after_is_never_below_one_second() -> None:
    clock = Clock()
    budget = _budget(clock, run_rate_limit_per_minute=1)
    await budget.admit("a")
    clock.advance(59.9)  # a tenth of a second left: told 1, never 0
    assert (await _refusal(budget, "a")).retry_after_s == 1
    assert BudgetExceeded("x", 0).retry_after_s == 1
    assert BudgetExceeded("x", -5).retry_after_s == 1
    clock.advance(0.1)
    await budget.admit("a")


# -------------------------------------------------------------- the OpenRouter allowance


async def test_the_allowance_reserve_closes_the_demo_and_reads_are_cached() -> None:
    clock = Clock()
    remaining = {"value": 7}
    reads = {"count": 0}

    async def reader() -> Allowance | None:
        reads["count"] += 1
        return Allowance(used=1000 - remaining["value"], limit=1000, remaining=remaining["value"])

    budget = _budget(clock, reader, openrouter_allowance_reserve=5)
    await budget.admit("a")  # 7 - 0 - 1 = 6 > 5
    assert budget.snapshot()["free_requests_remaining_today"] == 6
    refused = await _refusal(budget, "b")  # 7 - 1 - 1 = 5: at the reserve
    assert "allowance" in refused.message and refused.retry_after_s == SECONDS_TO_MIDNIGHT_AT_NOON
    assert reads["count"] == 1  # one read for the minute

    clock.advance(ALLOWANCE_TTL_S)
    remaining["value"] = 100
    await budget.admit("b")
    assert reads["count"] == 2
    assert budget.snapshot()["free_requests_remaining_today"] == 99


async def test_the_calls_parameter_spends_the_allowance() -> None:
    clock = Clock()

    async def reader() -> Allowance | None:
        return Allowance(used=990, limit=1000, remaining=10)

    budget = _budget(clock, reader, openrouter_allowance_reserve=2)
    await budget.admit("a", calls=4)  # 10 - 0 - 4 = 6 > 2
    assert budget.snapshot()["free_requests_remaining_today"] == 6
    refused = await _refusal(budget, "b", calls=4)  # 10 - 4 - 4 = 2: not above the reserve
    assert "allowance" in refused.message
    await budget.admit("b", calls=1)  # 6 - 1 = 5 > 2
    assert budget.snapshot()["free_requests_remaining_today"] == 5


async def test_the_reserve_never_exceeds_a_fifth_of_the_day() -> None:
    clock = Clock()

    async def reader() -> Allowance | None:
        # An account without credits gets 50 free requests a day: a reserve of 100 would close
        # the demo all day; a fifth of the day (10) is kept instead.
        return Allowance(used=30, limit=50, remaining=20)

    budget = _budget(clock, reader, openrouter_allowance_reserve=100)
    for _ in range(9):
        await budget.admit("a")
    refused = await _refusal(budget, "a")  # 20 - 9 - 1 = 10: the reserve
    assert "allowance" in refused.message


async def test_an_unreadable_allowance_admits_and_is_not_asked_again_for_a_minute() -> None:
    clock = Clock()
    reads = {"count": 0}

    async def failing_reader() -> Allowance | None:
        reads["count"] += 1
        return None

    budget = _budget(clock, failing_reader)
    await budget.admit("a")
    await budget.admit("a")
    assert reads["count"] == 1
    assert budget.snapshot()["free_requests_remaining_today"] is None
    clock.advance(ALLOWANCE_TTL_S)
    await budget.admit("a")
    assert reads["count"] == 2


def test_no_reader_outside_openrouter_and_when_the_check_is_off() -> None:
    clock = Clock()
    assert _budget(clock)._reader is None  # the fake provider never calls OpenRouter
    openrouter = make_settings(llm_provider="openrouter", openrouter_api_key="k")
    assert RunBudget(openrouter, clock=clock.monotonic, wall=clock.time)._reader is not None
    off = make_settings(
        llm_provider="openrouter", openrouter_api_key="k", openrouter_allowance_check=False
    )
    assert RunBudget(off, clock=clock.monotonic, wall=clock.time)._reader is None


# -------------------------------------------------------------------------- snapshot


async def test_snapshot_shows_the_limits_and_the_day() -> None:
    clock = Clock()
    budget = _budget(
        clock,
        run_rate_limit_per_minute=2,
        run_daily_limit_per_client=12,
        run_global_limit_per_minute=6,
        run_daily_limit=120,
    )
    assert budget.snapshot() == {
        "per_client_per_minute": 2,
        "per_client_per_day": 12,
        "per_minute": 6,
        "per_day": 120,
        "used_today": 0,
        "free_requests_remaining_today": None,
    }
    await budget.admit("a")
    assert budget.snapshot()["used_today"] == 1


async def test_the_rules_are_checked_in_order() -> None:
    clock = Clock()
    budget = _budget(clock, run_rate_limit_per_minute=1, run_global_limit_per_minute=1)
    await budget.admit("a")
    assert "this address" in (await _refusal(budget, "a")).message  # the visitor's rule first
    assert "busy" in (await _refusal(budget, "b")).message  # then everyone's
