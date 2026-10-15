"""What a test of the free models concludes, as pure functions.

Calls go in; a verdict per model and a chain come out. No file, no clock, no network, so every
rule here can be read and tested on its own. Three callers share them: the service's own test
(:mod:`app.models.chooser`), the scheduled job (:mod:`app.models.published`) and the page that
shows the last test.

The rules, in three sentences:

1. **Repair, never rank.** A model that is in the chain keeps its place and its order,
   whatever one test says about it. Three cases can rule a model out; they cannot rank two
   models, and free models are rate limited in bursts.
2. **A model leaves only when two sources agree**: OpenRouter's own page for it says "Model
   not found", and it gave no answer here.
3. **A free place goes to a model that passed**, behind the models already there. One model
   that is right but slow may take the last place, because a slow right answer is better than
   none.

One exception to "behind", and it ranks nothing: a model of the built-in default chain that
passes goes back to where the person who measured that chain put it. A first model that was
withdrawn for a few days leads again when it returns.

Changing the order of two models that both work stays a person's job: measure over several
passes and change ``DEFAULT_MODEL_CHAIN`` in ``app/config.py`` (README, "Model selection").
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from typing import Any

#: Models in a chain, at most: two places like the tested default, and one spare. The spare is
#: where a slow or newly found model can do no harm, because the last model is called only
#: when every model before it has failed.
PLACES = 3
#: Cases a model may miss (a wrong answer, or no answer) and still pass. None: the pipeline's
#: two tasks are its whole job, and a model that fails one of them fails a run.
MAX_MISSES = 0
#: A model's test ends once this many of its calls were not fully right: it can no longer
#: pass, and every call counts against the daily allowance.
GIVE_UP_AFTER = MAX_MISSES + 1
#: Models tested at once, at most: 4 x 3 cases = 12 requests.
CANDIDATE_LIMIT = 4
#: Seconds between request starts during a test: 10 a minute, half of the 20 a minute
#: OpenRouter allows an account on free models, so a visitor's run still finds room while a
#: test is going on. Twelve requests take just over a minute.
SECONDS_BETWEEN_STARTS = 6.0

#: No usable answer to a case that is known to be answerable: the model's own doing.
UNUSABLE = ("SchemaValidationError", "ContentRefused")
#: The key was refused. No model can fix that, so a test stops.
KEY_PROBLEMS = ("ProviderAuthError",)

#: Every verdict :func:`judge` can give, and every reason. The page has words for each.
VERDICTS = ("passed", "slow", "failed", "not_judged")
REASONS = ("all_right", "wrong_answers", "unusable_answers", "no_answers", "unanswered")
#: What became of a model once the chain was decided.
PLACINGS = ("kept", "added", "dropped", "no_place")


def summarise(calls: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per model, in the order the models first appear in ``calls``.

    A call that ``answered`` counts as answered and as correct when its ``correct`` is true.
    Every other call (``no_answer``, ``refused``) counts as no answer, under its ``error_type``.
    Only calls that answered are timed: a refused request returns in a fraction of a second
    and says nothing about the model's speed.
    """
    rows: dict[str, dict[str, Any]] = {}
    for call in calls:
        row = rows.setdefault(
            call["model"],
            {
                "model": call["model"],
                "calls": 0,
                "no_answer": 0,
                "errors": {},
                "answered": 0,
                "correct": 0,
                "seconds": [],
            },
        )
        row["calls"] += 1
        if call["outcome"] == "answered":
            row["answered"] += 1
            row["correct"] += 1 if call.get("correct") else 0
            row["seconds"].append(float(call["seconds"]))
        else:
            row["no_answer"] += 1
            kind = call.get("error_type") or (
                "ContentRefused" if call["outcome"] == "refused" else "unknown"
            )
            row["errors"][kind] = row["errors"].get(kind, 0) + 1
    for row in rows.values():
        seconds = row.pop("seconds")
        row["median_seconds"] = round(statistics.median(seconds), 1) if seconds else None
        row["slowest_seconds"] = round(max(seconds), 1) if seconds else None
    return list(rows.values())


def rank(rows: Sequence[dict[str, Any]], timeout_seconds: float) -> list[dict[str, Any]]:
    """The README's ordering rule, applied to rows of :func:`summarise`.

    Step 1 leaves out a model that never answered, or whose usual (median) answer takes more
    than half of the time limit. Steps 2 to 4 order the rest: fewest wrong answers, then
    fewest calls with no answer, and speed only breaks ties.
    """
    kept = [
        row for row in rows if row["answered"] > 0 and row["median_seconds"] <= timeout_seconds / 2
    ]
    return sorted(
        kept,
        key=lambda row: (row["answered"] - row["correct"], row["no_answer"], row["median_seconds"]),
    )


def _empty_row(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "calls": 0,
        "no_answer": 0,
        "errors": {},
        "answered": 0,
        "correct": 0,
        "median_seconds": None,
        "slowest_seconds": None,
    }


def judge(
    tested: Sequence[str],
    calls: Sequence[dict[str, Any]],
    cases: Sequence[str],
    timeout_seconds: float,
    finished: bool = False,
) -> list[dict[str, Any]]:
    """One row per tested model: the summary, a verdict and its reason.

    ``cases`` are the ids of the cases, in the order they are sent. ``verdict`` stays ``None``
    while a model's calls are still coming in.

    * ``passed``: every case was sent and answered right, and the usual (median) answer took
      at most half of the time limit.
    * ``slow``: the same, but the usual answer took longer. Good for the last place only.
    * ``failed``: an answer was wrong or unusable. This says something about the model.
    * ``not_judged``: a call went unanswered (rate limit, timeout, upstream error). This says
      something about the moment, so nothing is held against the model.
    """
    measured = {row["model"]: row for row in summarise(calls)}
    rows = []
    for model in tested:
        row = measured.get(model) or _empty_row(model)
        wrong = row["answered"] - row["correct"]
        unusable = sum(row["errors"].get(kind, 0) for kind in UNUSABLE)
        misses = row["calls"] - row["correct"]
        done = finished or row["calls"] >= len(cases) or misses >= GIVE_UP_AFTER
        verdict: str | None
        why: str | None
        if not done:
            verdict, why = None, None
        elif wrong + unusable >= GIVE_UP_AFTER:
            verdict, why = "failed", "wrong_answers" if wrong else "unusable_answers"
        elif row["calls"] >= len(cases) and misses <= MAX_MISSES:
            quick = bool(rank([row], timeout_seconds))
            verdict, why = ("passed" if quick else "slow"), "all_right"
        else:
            verdict, why = "not_judged", "unanswered" if row["answered"] else "no_answers"
        rows.append({**row, "verdict": verdict, "why": why, "placing": None})
    return rows


def next_chain(
    current: Sequence[str],
    rows: Sequence[dict[str, Any]],
    gone: Sequence[str],
    timeout_seconds: float,
    measured: Sequence[str] = (),
) -> list[str]:
    """The chain to use after a test. It can only be repaired, never reshuffled.

    ``current`` is the chain in use, ``rows`` what :func:`judge` said, ``gone`` the models
    OpenRouter's own page answered "Model not found" for, and ``measured`` the built-in
    default chain, in the order a person gave it by measuring.

    A model of the chain leaves only when it is gone *and* gave no answer in this test (a
    model that was not tested gave none). Every other model of the chain keeps its place and
    its order, whatever its verdict. Free places are then filled behind the models that are
    already there: first the models that passed, in the README's order, then at most one model
    that is right but slow. A test in which nothing can be kept or added changes nothing: a
    chain is never emptied.

    A measured model that passed is taken before other newcomers and goes back to its
    measured place: behind the measured models that stand before it, in front of everybody
    else. Without that, a first model that was withdrawn for a few days would come back as
    the last, and nothing short of a new default would put it first again.
    """
    by_model = {row["model"]: row for row in rows}
    kept = [
        model
        for model in current
        if not (model in gone and not by_model.get(model, {}).get("answered"))
    ]
    new = [row for row in rows if row["model"] not in current]
    quick = rank([row for row in new if row["verdict"] == "passed"], timeout_seconds)
    quick.sort(key=lambda row: row["model"] not in measured)  # a stable sort: the rest keep rank
    chain = (kept + [row["model"] for row in quick])[:PLACES]
    if len(chain) < PLACES:
        slow = sorted(
            (row for row in new if row["verdict"] == "slow"),
            key=lambda row: (
                row["answered"] - row["correct"],
                row["no_answer"],
                row["median_seconds"],
            ),
        )
        chain += [row["model"] for row in slow[:1]]
    added = [row["model"] for row in quick]
    for place, model in enumerate(measured):
        if model in added and model in chain:
            chain.remove(model)
            before = [chain.index(earlier) for earlier in measured[:place] if earlier in chain]
            chain.insert(max(before, default=-1) + 1, model)
    return chain or list(current)


def in_turn(
    outside: Sequence[str], tested_last: Sequence[str], left_out_last: Sequence[str]
) -> list[str]:
    """The models outside the chain, in the order the next test should take them.

    A test has room for :data:`CANDIDATE_LIMIT` models. Those the last test had no room for
    go first, in the order they have been waiting; then models it never saw; then the ones it
    tested. Taken in this order, every model gets its turn, however many OpenRouter lists.
    """

    def turn(model: str) -> tuple[int, int]:
        if model in left_out_last:
            return 0, list(left_out_last).index(model)
        return (2, 0) if model in tested_last else (1, 0)

    return sorted(outside, key=turn)  # a stable sort: OpenRouter's own order breaks ties


def place(rows: Sequence[dict[str, Any]], before: Sequence[str], after: Sequence[str]) -> None:
    """Write into each row what became of its model: kept, added, dropped or no place."""
    for row in rows:
        model = row["model"]
        if model in before:
            row["placing"] = "kept" if model in after else "dropped"
        elif model in after:
            row["placing"] = "added"
        elif row["verdict"] in ("passed", "slow"):
            row["placing"] = "no_place"


def seen_working(rows: Sequence[dict[str, Any]], chain: Sequence[str]) -> bool:
    """True when at least one model of ``chain`` passed in this very test.

    The scheduled job publishes a chain as "tested today" only then. A test in which every
    call was turned away proves nothing about the chain.
    """
    return any(row["model"] in chain and row["verdict"] in ("passed", "slow") for row in rows)
