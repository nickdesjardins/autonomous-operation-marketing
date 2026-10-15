"""Tests for what a test of the free models concludes (``app.models.rules``).

Pure functions over made-up calls. No network, no clock, no file.

Layout:

* the verdict     - passed, slow, failed, not judged;
* the chain       - repair, never rank;
* the numbers     - what a test may spend, and the cache's own copy of the chain length.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.config import DEFAULT_MODEL_CHAIN, parse_model_chain
from app.models import cache, cases, rules
from app.models.rules import judge, next_chain, place

CASES = cases.case_ids()
LIMIT = 60.0
FIRST, SECOND = parse_model_chain(DEFAULT_MODEL_CHAIN)[:2]
NEW, OTHER, SLOW = "vendor/new:free", "vendor/other:free", "vendor/slow:free"


def calls(model: str, plan: str, seconds: float = 3.0) -> list[dict[str, Any]]:
    """One call per letter, in the order the cases are sent.

    ``r`` right, ``w`` wrong, ``l`` lost to a rate limit, ``u`` unusable output, ``f`` refused
    by the model's content filter.
    """
    made = []
    for case, letter in zip(CASES, plan, strict=False):
        answered = letter in "rw"
        made.append(
            {
                "model": model,
                "case": case,
                "outcome": "answered" if answered else "refused" if letter == "f" else "no_answer",
                "correct": {"r": True, "w": False}.get(letter),
                "seconds": seconds if answered else 0.2,
                "error_type": {
                    "l": "ProviderError",
                    "u": "SchemaValidationError",
                    "f": "ContentRefused",
                }.get(letter),
            }
        )
    return made


def verdict(
    plan: str, seconds: float = 3.0, finished: bool = False
) -> tuple[str | None, str | None]:
    row = judge([NEW], calls(NEW, plan, seconds), CASES, LIMIT, finished=finished)[0]
    return row["verdict"], row["why"]


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("plan", "expected"),
    [
        ("rrr", ("passed", "all_right")),
        ("w", ("failed", "wrong_answers")),  # one wrong answer ends its test: it cannot pass
        ("rw", ("failed", "wrong_answers")),
        ("u", ("failed", "unusable_answers")),  # rejects the format: one call, no more
        ("f", ("failed", "unusable_answers")),  # a content filter on a known-answerable case
        ("rru", ("failed", "unusable_answers")),
        ("l", ("not_judged", "no_answers")),  # weather: nothing is held against it
        ("rl", ("not_judged", "unanswered")),
        ("rrl", ("not_judged", "unanswered")),
        ("rr", (None, None)),  # still running
        ("", (None, None)),
    ],
)
def test_the_verdict(plan: str, expected: tuple[str | None, str | None]) -> None:
    assert verdict(plan) == expected


def test_a_slow_model_is_right_but_slow() -> None:
    assert verdict("rrr", seconds=LIMIT / 2) == ("passed", "all_right")
    assert verdict("rrr", seconds=LIMIT / 2 + 0.1) == ("slow", "all_right")


def test_a_test_that_was_cut_short_judges_nobody_by_what_is_missing() -> None:
    assert verdict("rr", finished=True) == ("not_judged", "unanswered")
    assert verdict("", finished=True) == ("not_judged", "no_answers")


def test_a_model_that_was_not_called_has_an_empty_row() -> None:
    row = judge([NEW], [], CASES, LIMIT)[0]
    assert row["calls"] == 0 and row["median_seconds"] is None and row["verdict"] is None
    assert row["placing"] is None


def test_the_summary_times_only_the_calls_that_answered() -> None:
    made = calls(NEW, "rl", seconds=10.0)
    row = rules.summarise(made)[0]
    assert row["median_seconds"] == 10.0 and row["no_answer"] == 1
    assert row["errors"] == {"ProviderError": 1}


def test_the_numbers_bound_what_a_test_spends() -> None:
    assert rules.MAX_MISSES == 0 and rules.GIVE_UP_AFTER == 1
    assert rules.PLACES == 3 == cache.MAX_CHAIN  # the cache's own copy of the chain length
    assert rules.CANDIDATE_LIMIT * len(CASES) == 12  # the most one test can send
    assert 60 / rules.SECONDS_BETWEEN_STARTS <= 10  # half of the 20 a minute OpenRouter allows
    assert set(rules.UNUSABLE) <= {"SchemaValidationError", "ContentRefused"}
    assert rules.KEY_PROBLEMS == ("ProviderAuthError",)
    for name in ("VERDICTS", "REASONS", "PLACINGS"):
        assert len(set(getattr(rules, name))) == len(getattr(rules, name))


# ---------------------------------------------------------------------------
# The chain
# ---------------------------------------------------------------------------


def chain_after(
    current: list[str],
    plans: dict[str, tuple[str, float]],
    gone: tuple[str, ...] = (),
    measured: tuple[str, ...] = (),
) -> list[str]:
    made = [
        call for model, (plan, seconds) in plans.items() for call in calls(model, plan, seconds)
    ]
    rows = judge(list(plans), made, CASES, LIMIT, finished=True)
    chain = next_chain(current, rows, list(gone), LIMIT, measured)
    place(rows, current, chain)
    return chain


GOOD, LOST, BAD, SLOWLY = ("rrr", 3.0), ("l", 3.0), ("rw", 3.0), ("rrr", 45.0)


@pytest.mark.parametrize(
    ("current", "plans", "gone", "expected"),
    [
        # Everything works: nothing moves, and the free place gets the newcomer.
        ([FIRST, SECOND], {FIRST: GOOD, SECOND: GOOD, NEW: GOOD}, (), [FIRST, SECOND, NEW]),
        # Weather on the first model: it keeps first place.
        ([FIRST, SECOND], {FIRST: LOST, SECOND: GOOD}, (), [FIRST, SECOND]),
        # A model of the chain that answers wrongly keeps its place too: ranking is a person's job.
        ([FIRST, SECOND], {FIRST: BAD, SECOND: GOOD}, (), [FIRST, SECOND]),
        # Gone and silent: it leaves, the rest moves up, a newcomer goes behind.
        ([FIRST, SECOND], {FIRST: LOST, SECOND: GOOD, NEW: GOOD}, (FIRST,), [SECOND, NEW]),
        # OpenRouter says "gone" but the model answers: the catalogue alone removes nothing.
        ([FIRST, SECOND], {FIRST: GOOD, SECOND: GOOD}, (FIRST,), [FIRST, SECOND]),
        # Silent but still listed: it stays.
        ([FIRST, SECOND], {FIRST: LOST, SECOND: LOST, NEW: GOOD}, (), [FIRST, SECOND, NEW]),
        # A repair tests only the models outside the chain; the gone one was not asked again.
        ([FIRST, SECOND], {NEW: GOOD}, (FIRST,), [SECOND, NEW]),
        # A newcomer that failed, or could not be judged, gets nothing.
        ([FIRST, SECOND], {FIRST: GOOD, SECOND: GOOD, NEW: BAD, OTHER: LOST}, (), [FIRST, SECOND]),
        # Quick models first, in the README's order; one slow model may take the last place.
        (
            [FIRST],
            {FIRST: GOOD, NEW: GOOD, OTHER: ("rrr", 2.0), SLOW: SLOWLY},
            (),
            [FIRST, OTHER, NEW],
        ),
        ([FIRST, SECOND], {FIRST: GOOD, SECOND: GOOD, SLOW: SLOWLY}, (), [FIRST, SECOND, SLOW]),
        ([FIRST], {FIRST: GOOD, SLOW: SLOWLY, OTHER: SLOWLY}, (), [FIRST, SLOW]),
        # A full chain takes nobody, however good.
        (
            [FIRST, SECOND, SLOW],
            {FIRST: GOOD, SECOND: GOOD, SLOW: SLOWLY, NEW: GOOD},
            (),
            [FIRST, SECOND, SLOW],
        ),
        # The owner's case: every model of the chain is gone.
        ([FIRST, SECOND], {SLOW: SLOWLY, OTHER: ("u", 3.0)}, (FIRST, SECOND), [SLOW]),
        # Nothing answers at all: nothing changes, and a chain is never emptied.
        ([FIRST, SECOND], {NEW: LOST}, (FIRST, SECOND), [FIRST, SECOND]),
        ([FIRST, SECOND], {FIRST: LOST, SECOND: LOST, NEW: LOST}, (), [FIRST, SECOND]),
    ],
)
def test_the_chain_is_repaired_and_never_reshuffled(
    current: list[str],
    plans: dict[str, tuple[str, float]],
    gone: tuple[str, ...],
    expected: list[str],
) -> None:
    chain = chain_after(current, plans, gone)
    assert chain == expected
    assert len(chain) == len(set(chain)) <= rules.PLACES
    # Whoever was in the chain and still is has kept the order it had.
    kept = [model for model in current if model in chain]
    assert [model for model in chain if model in kept] == kept


@pytest.mark.parametrize(
    ("current", "plans", "expected"),
    [
        # The first model was withdrawn for a few days and is back: it leads again.
        ([SECOND, SLOW], {FIRST: GOOD}, [FIRST, SECOND, SLOW]),
        # The second one comes back: behind the first, in front of everybody else.
        ([FIRST, SLOW], {SECOND: GOOD}, [FIRST, SECOND, SLOW]),
        ([SLOW, OTHER], {SECOND: GOOD}, [SECOND, SLOW, OTHER]),
        # Both come back in one test, the second with the better figures: the order is the measured one.
        ([SLOW], {SECOND: ("rrr", 1.0), FIRST: ("rrr", 9.0)}, [FIRST, SECOND, SLOW]),
        # One free place and two that passed: the measured model takes it, whatever one test's figures say.
        ([SECOND, SLOW], {NEW: ("rrr", 1.0), FIRST: ("rrr", 9.0)}, [FIRST, SECOND, SLOW]),
        ([SECOND], {NEW: ("rrr", 1.0), FIRST: ("rrr", 9.0)}, [FIRST, SECOND, NEW]),
        # Right but slow is good for the last place only, also for a model that was measured.
        ([SECOND, OTHER], {FIRST: SLOWLY}, [SECOND, OTHER, FIRST]),
        # Models that are in the chain are never moved, measured or not.
        ([SECOND, FIRST], {SECOND: GOOD, FIRST: GOOD, NEW: GOOD}, [SECOND, FIRST, NEW]),
        # A full chain takes nobody, and a model that did not pass gets no place.
        ([SECOND, SLOW, OTHER], {FIRST: GOOD}, [SECOND, SLOW, OTHER]),
        ([SECOND, SLOW], {FIRST: LOST}, [SECOND, SLOW]),
    ],
)
def test_a_measured_model_that_passes_goes_back_to_its_measured_place(
    current: list[str], plans: dict[str, tuple[str, float]], expected: list[str]
) -> None:
    assert chain_after(current, plans, measured=(FIRST, SECOND)) == expected
    # Told of no measured chain, a model that passes goes to the back like any other.
    if plans.get(FIRST) is GOOD and current == [SECOND, SLOW]:
        assert chain_after(current, plans) == [SECOND, SLOW, FIRST]


def test_models_outside_the_chain_are_taken_in_turn() -> None:
    """A test has room for a few. However many OpenRouter lists, each gets its turn."""
    outside = [f"vendor/number-{number:02}:free" for number in range(1, 13)]
    tested: list[str] = []
    left_out: list[str] = []
    had_a_turn: list[str] = []
    for _ in range(3):
        order = rules.in_turn(outside, tested, left_out)
        tested, left_out = order[: rules.CANDIDATE_LIMIT], order[rules.CANDIDATE_LIMIT :]
        had_a_turn += tested

    assert had_a_turn == outside
    # Those that waited come first, in the order they waited; then a model nobody has seen
    # yet; then the ones the last test had room for.
    assert rules.in_turn(["a", "b", "c", "new"], ["a"], ["c", "b"]) == ["c", "b", "new", "a"]


def test_each_row_says_what_became_of_its_model() -> None:
    plans = {FIRST: LOST, SECOND: GOOD, NEW: GOOD, OTHER: GOOD, SLOW: SLOWLY}
    made = [
        call for model, (plan, seconds) in plans.items() for call in calls(model, plan, seconds)
    ]
    rows = judge(list(plans), made, CASES, LIMIT, finished=True)
    chain = next_chain([FIRST, SECOND], rows, [FIRST], LIMIT)
    place(rows, [FIRST, SECOND], chain)

    assert chain == [SECOND, NEW, OTHER]
    assert {row["model"]: row["placing"] for row in rows} == {
        FIRST: "dropped",
        SECOND: "kept",
        NEW: "added",
        OTHER: "added",
        SLOW: "no_place",
    }


def test_a_chain_is_seen_working_only_when_one_of_its_models_passed() -> None:
    rows = judge([FIRST, NEW], calls(FIRST, "l") + calls(NEW, "rrr"), CASES, LIMIT, finished=True)
    assert rules.seen_working(rows, [FIRST, NEW]) is True
    assert rules.seen_working(rows, [FIRST]) is False
