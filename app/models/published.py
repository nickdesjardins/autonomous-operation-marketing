"""The tested chain this project publishes, and how a service reads it.

A service that nobody uses tests nothing, and on a free host it even forgets what it knew:
the host stops it after a quarter of an hour without visitors and starts it again with an
empty disk. So after a month of silence its models may all be gone, and the first visitor
would be the one to find out.

To spare that visitor, a scheduled job on GitHub (``.github/workflows/model-check.yml``) runs
the service's own test of the free models about once a week and writes what it found to one
small file, ``chain.json``, on the ``tested-models`` branch of the repository. That file is
the memory that lasts. A service reads it when it starts and after a failed request: one
request, no key, no free request spent.

The file is a claim like any other. Nothing in it is used before it is checked:

* only ids that match ``vendor/model:free`` are accepted, whatever the file says, so no file
  can make this service call a model that costs money;
* it must come from the very same test: same prompts, schemas, cases, rules and built-in
  default (:func:`fingerprint`). A file made by other code says nothing about this code;
* whether it is newer than what the service already has is for the caller to decide
  (:func:`app.models.chooser.read_published`).

This module holds the format, the checks, the request, the rule for when a new test is due,
and the command the scheduled job runs::

    python -m app.models.published --previous FILE --out FILE [--force]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from app import config
from app.models import cases, catalogue, rules, testing
from app.models.cache import is_free_model
from app.models.catalogue import Fetch

#: Raised when the meaning of the file changes. A file with another number is not used.
#: 2: the two audit cases are graded after the overrule step, as a run acts on a verdict.
VERSION = 2
#: A real file is a few kilobytes. Nothing larger is looked at.
MAX_BYTES = 64_000
#: A full test is due once the last one is this old: six and a half days, so that the daily
#: run of day seven is the one that tests.
RETEST_AFTER_HOURS = 156
#: After a test that showed nothing (every call turned away), the next daily run tries again.
RETRY_AFTER_HOURS = 20
#: Two weekly tests missed. From here on the log and the page say how old the list is.
STALE_AFTER_DAYS = 14
#: Three weekly tests missed. The job then fails on purpose, because a failed scheduled run is
#: the one thing GitHub tells the owner about.
ALARM_AFTER_DAYS = 21
#: More calls than any test sends (4 models x 3 cases = 12).
MAX_CALLS = 100

STAMP = testing.STAMP  #: how a moment is written in the file (UTC)

#: The one address a file may name for the log: the run of the job that wrote it. Anything
#: else a file says about itself could be any sentence, and the server log is where this
#: project tells its admin what to do.
_RUN_ADDRESS = re.compile(
    r"https://github\.com/[A-Za-z0-9_.-]{1,100}/[A-Za-z0-9_.-]{1,100}/actions/runs/[0-9]{1,20}"
)

#: The test's chain client, replaced by the suite so that the job never reaches the network.
chain_factory: testing.ChainFactory | None = None


class Unusable(Exception):
    """Why a published file is not used.

    The message is one clause for the server log. ``kind`` is for the page, which words it for
    visitors: ``missing`` (nothing is published), ``unreachable`` (it could not be fetched),
    ``invalid`` (it failed a check) or ``foreign`` (it was made by other code). ``tested`` is
    the file's own date when it had a readable one.
    """

    def __init__(self, message: str, kind: str = "invalid", tested: str | None = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.tested = tested


@dataclass(frozen=True)
class Published:
    """A published file that passed every check in :func:`parse`."""

    chain: tuple[str, ...]
    #: When a test last saw this chain working.
    tested: datetime
    #: That test, cleaned for the page: every call, every verdict. None when unusable.
    record: dict[str, Any] | None = None
    #: What the job remembers between runs: when it last tested, and what was listed then.
    job: Mapping[str, Any] = field(default_factory=dict)
    #: The address of the job's run that wrote the file, under "run". For the log only.
    by: Mapping[str, str] = field(default_factory=dict)


def moment(written: object) -> datetime | None:
    """The moment a file names, or None for anything that is not such a moment."""
    try:
        return datetime.strptime(str(written), STAMP).replace(tzinfo=UTC)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# What makes two tests the same test
# ---------------------------------------------------------------------------


def fingerprint(settings: Any) -> str:
    """Changes whenever the test itself changes.

    A published chain is only worth trusting to a service that would have run the very same
    test: the same prompts and schemas, the same cases and checks, the same rules, the same
    response format, the same reasoning control and the same built-in default (a new default
    means somebody measured again by hand, which is newer evidence than any file made before
    it). A fork that rewrites a prompt gets another fingerprint and so ignores this project's
    file by itself.

    What goes in is what a model is sent and what its answer is checked against, not how
    either is worded for people. The code that judges the calls is not in it either. Whoever
    changes that code raises :data:`VERSION`.
    """
    material = json.dumps(
        [
            cases.fingerprint_material(),
            [rules.PLACES, rules.MAX_MISSES],
            config.parse_model_chain(config.DEFAULT_MODEL_CHAIN),
            bool(settings.openrouter_strict_json_schema),
            str(settings.openrouter_reasoning),
        ],
        sort_keys=True,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def fetch(url: str, get: Fetch | None = None) -> str:
    """The published file as text. Raises :class:`Unusable` when there is none to read.

    The request is the catalogue's own: a named client, no key, a time limit on each network
    step and at most 1 MB read. Tests pass ``get``; nothing else does.
    """
    address = urlsplit(url)
    if address.scheme != "https" or not address.hostname:
        raise Unusable("its address is not an https address")
    try:
        status, body = (get or catalogue._http_get)(url)
    except Exception as exc:  # noqa: BLE001 - whatever went wrong, there is no file
        raise Unusable(f"it could not be fetched ({type(exc).__name__})", "unreachable") from exc
    if status == 404:
        raise Unusable("nothing is published at that address (HTTP 404)", "missing")
    if status != 200:
        raise Unusable(f"unexpected answer (HTTP {status})", "unreachable")
    return body


def _ids(value: object, most: int | None = None) -> list[str]:
    """The free model ids in ``value``, each once, and at most ``most`` of them looked at."""
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(item for item in value[:most] if is_free_model(item)))


def _clean_call(call: object, models: Sequence[str], ids: Sequence[str]) -> dict[str, Any] | None:
    """One call record with nothing in it but what the page may show, or None."""
    if not isinstance(call, dict):
        return None
    model, case, seconds = call.get("model"), call.get("case"), call.get("seconds")
    if (
        not isinstance(model, str)
        or model not in models
        or not isinstance(case, str)
        or case not in ids
        or call.get("outcome") not in testing.CALL_OUTCOMES
        or not (call.get("correct") is None or isinstance(call.get("correct"), bool))
        or isinstance(seconds, bool)
        or not isinstance(seconds, int | float)
        or not 0 <= seconds <= 3600
    ):
        return None
    error = call.get("error_type")
    return {
        "model": model,
        "case": case,
        "outcome": call["outcome"],
        "correct": call.get("correct"),
        "seconds": round(float(seconds), 2),
        # The sanitised message is never published or read: it can name a setting.
        "error_type": error if isinstance(error, str) and error in testing.ERROR_TYPES else None,
    }


def _clean_record(test: object, chain: Sequence[str], tested: datetime) -> dict[str, Any] | None:
    """The scheduled test as the page may show it, with the verdicts worked out again here.

    Only known fields survive, and the verdicts are never taken from the file: they are
    computed from its calls by this service's own rules.
    """
    if not isinstance(test, dict):
        return None
    ids = cases.case_ids()
    models = _ids(test.get("tested"), rules.CANDIDATE_LIMIT)
    limit = test.get("timeout_seconds")
    started = moment(test.get("started"))
    if (
        not models
        or isinstance(limit, bool)
        or not isinstance(limit, int | float)
        # Python reads 1e999 and NaN as JSON numbers, and neither is a time limit.
        or not 0 < limit <= 3600
        or started is None
    ):
        return None
    raw = test.get("calls") if isinstance(test.get("calls"), list) else []
    calls = [call for call in (_clean_call(item, models, ids) for item in raw[:MAX_CALLS]) if call]
    before = _ids(test.get("chain_before"), rules.PLACES)
    rows = rules.judge(models, calls, ids, float(limit), finished=True)
    rules.place(rows, before, chain)
    return {
        "id": started.strftime("%Y%m%dT%H%M%SZ"),
        "by": "schedule",
        "reason": {"code": "schedule", "models": []},
        "state": "done",
        "started": started.strftime(STAMP),
        "finished": tested.strftime(STAMP),
        "found": _ids(test.get("found"), MAX_CALLS),
        "tested": models,
        "left_out": _ids(test.get("left_out"), MAX_CALLS),
        "gone": _ids(test.get("gone"), rules.PLACES),
        "cases": [{"id": case["id"], "title": case["title"]} for case in cases.CASES],
        "timeout_seconds": float(limit),
        "calls": calls,
        "in_flight": [],
        "models": rows,
        "chain_before": before,
        "kept": before,
        "chain": list(chain),
        "outcome": "changed" if before and before != list(chain) else "unchanged",
        "requests": len(calls),
    }


def parse(text: str, expected: str, now: datetime) -> Published:
    """Check a published file from top to bottom. Raises :class:`Unusable` with the reason.

    ``expected`` is :func:`fingerprint` for the service that is asking. One bad id discards
    the whole file. Never raises anything but :class:`Unusable`.
    """
    try:
        if len(text.encode("utf-8", "replace")) > MAX_BYTES:
            raise Unusable(f"it is larger than {MAX_BYTES} bytes")
        try:
            data = json.loads(text)
        except (ValueError, RecursionError) as exc:
            raise Unusable("it is not valid JSON") from exc
        if not isinstance(data, dict) or data.get("version") != VERSION:
            raise Unusable(f"it is not a version {VERSION} file")
        tested = moment(data.get("tested"))
        if tested is None:
            raise Unusable("it does not say when it was tested")
        written = tested.strftime(STAMP)
        soon = now + timedelta(minutes=10)  # two clocks are never quite alike
        if tested > soon:
            # A date far ahead would make the file "the newest" for ever.
            raise Unusable("its date is in the future")
        if data.get("fingerprint") != expected:
            raise Unusable(
                f"it was made by another version of the test (it is dated {written})",
                "foreign",
                written,
            )
        chain = data.get("chain")
        if (
            not isinstance(chain, list)
            or not 1 <= len(chain) <= rules.PLACES
            or not all(is_free_model(model) for model in chain)
            or len(set(chain)) != len(chain)
        ):
            raise Unusable(
                f"its chain is not a list of 1 to {rules.PLACES} different ids ending in ':free'"
            )
        job = data.get("job") if isinstance(data.get("job"), dict) else {}
        by = data.get("by") if isinstance(data.get("by"), dict) else {}
        checked = moment(job.get("checked"))
        run = by.get("run")
        outcome = job.get("outcome")
        return Published(
            chain=tuple(chain),
            tested=tested,
            record=_clean_record(data.get("test"), chain, tested),
            job={
                # A moment ahead of now would put the next test off for as long.
                "checked": job["checked"] if checked is not None and checked <= soon else written,
                # Every id, however many: the next run compares today's list with this one.
                "found": _ids(job.get("found")),
                "outcome": outcome if outcome in ("published", "nothing") else "published",
            },
            by={"run": run} if isinstance(run, str) and _RUN_ADDRESS.fullmatch(run) else {},
        )
    except Unusable:
        raise
    except Exception as exc:  # noqa: BLE001 - an odd file is an unusable file, never a crash
        raise Unusable(f"it could not be read ({type(exc).__name__})") from exc


# ---------------------------------------------------------------------------
# Writing (the scheduled job only)
# ---------------------------------------------------------------------------


def build(
    record: Mapping[str, Any], expected: str, now: datetime, by: Mapping[str, str]
) -> dict[str, Any]:
    """The file to publish for a finished test. Refuses a chain that is not free."""
    chain = list(record["chain"])
    if not chain or not all(is_free_model(model) for model in chain):
        raise ValueError("only a chain of ids ending in ':free' is ever published")
    written = now.strftime(STAMP)
    keep = (
        "started",
        "found",
        "tested",
        "left_out",
        "gone",
        "cases",
        "timeout_seconds",
        "calls",
        "models",
        "chain_before",
        "outcome",
        "requests",
    )
    return {
        "version": VERSION,
        "fingerprint": expected,
        "tested": written,
        "chain": chain,
        # For readers: the service's page, and the static front door, which has no rules of
        # its own and so reads the verdicts as they are written here.
        "test": {key: record[key] for key in keep},
        "job": {"checked": written, "found": sorted(record["found"]), "outcome": "published"},
        "by": dict(by),
    }


def due(
    previous: Published | None,
    problem: str,
    listed: Sequence[str] | None,
    now: datetime,
    forced: bool = False,
) -> tuple[bool, str]:
    """Whether the scheduled job should run a full test now, and why or why not.

    ``previous`` is the file published last time, or None with ``problem`` saying why it
    cannot be used. ``listed`` holds the free models OpenRouter lists today, or is None when
    the list could not be read. Asking costs nothing: the list needs no key.

    A test spends free requests, so one is due only for a reason: nothing usable is
    published; the last test is about a week old; the last test showed nothing and a day has
    passed; or OpenRouter's list changed in a way a test could act on. That is a model of the
    published chain leaving the list (it may have to go), or a model the last test did not
    see while the chain has a free place for it. A model that comes and goes outside a full
    chain changes nothing a test could change, and would otherwise cost a test at every
    change. What counts is the last *test*, not the last publication: a test that could
    publish nothing is not repeated every day.
    """
    if forced:
        return True, "a test was asked for by hand"
    if previous is None:
        return True, f"there is no usable published list ({problem})"
    checked = moment(previous.job.get("checked")) or previous.tested
    hours = (now - checked).total_seconds() / 3600
    if hours >= RETEST_AFTER_HOURS:
        return True, f"the last test is {hours / 24:.1f} days old"
    if previous.job.get("outcome") != "published" and hours >= RETRY_AFTER_HOURS:
        return True, "the last test could confirm nothing"
    if listed is None:
        return False, "OpenRouter's list could not be read to see whether anything changed"
    seen = set(previous.job.get("found", ()))
    left = [model for model in previous.chain if model in seen and model not in listed]
    room = len(previous.chain) < rules.PLACES
    changed = left + (sorted(set(listed) - seen) if room else [])
    if changed:
        return (
            True,
            f"OpenRouter's list of free models changed since the last test: {', '.join(changed)}",
        )
    return False, (
        f"the last test is {hours / 24:.1f} days old and OpenRouter's list has not changed "
        "in a way a test could act on"
    )


# ---------------------------------------------------------------------------
# The command the scheduled job runs
# ---------------------------------------------------------------------------


def _say(line: str) -> None:
    print(line)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:  # GitHub shows this file on the run's page
        with open(summary, "a", encoding="utf-8") as out:
            out.write(line + "\n\n")


def _warn(line: str) -> None:
    _say(line)
    print(f"::warning::{line}")


def _fail(why: str) -> int:
    """Say why this run fails on purpose, and return the status that turns it red.

    Failing the run is how the owner gets told: GitHub emails about failed scheduled runs.
    """
    _say(f"Failing this run on purpose: {why}")
    return 1


def _words(code: object) -> str:
    return str(code).replace("_", " ")


def _run_address() -> dict[str, str]:
    server, repository, run = (
        os.environ.get(name, "")
        for name in ("GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID")
    )
    return {
        "repository": repository,
        "commit": os.environ.get("GITHUB_SHA", ""),
        "run": f"{server}/{repository}/actions/runs/{run}" if server and repository and run else "",
    }


async def _job(
    previous_path: Path, out_path: Path, forced: bool, settings: config.Settings | None = None
) -> int:
    settings = settings or config.get_settings()
    now = datetime.now(UTC)
    expected = fingerprint(settings)
    base_url = settings.openrouter_base_url
    default = testing.default_chain()

    repository = os.environ.get("GITHUB_REPOSITORY", "")
    if repository and f"/{repository}/" not in config.DEFAULT_PUBLISHED_URL:
        _warn(
            f"This job publishes to {repository}, but services read "
            f"{config.DEFAULT_PUBLISHED_URL}. Change DEFAULT_PUBLISHED_URL in app/config.py if "
            "this repository is the one to read."
        )

    previous, previous_text, problem = None, "", "nothing has been published yet"
    if previous_path.is_file():
        try:
            previous_text = previous_path.read_text(encoding="utf-8")
            previous = parse(previous_text, expected, now)
        except (OSError, ValueError, Unusable) as exc:  # ValueError: bytes that are not UTF-8
            problem = str(exc) or type(exc).__name__
    age = (now - previous.tested).days if previous else None
    stale = previous is None or (age is not None and age >= ALARM_AFTER_DAYS)
    no_list = "no tested chain younger than three weeks is published, so services are on their own."

    try:
        listed: list[str] | None = catalogue.free_models(base_url)
    except Exception as exc:  # noqa: BLE001 - without the list nothing can be decided or tested
        listed = None
        _say(f"OpenRouter's list of free models could not be read ({type(exc).__name__}).")

    # What OpenRouter says about the built-in default: a notice, since the published list is
    # what services use. It is the alarm only when this job cannot test.
    listings = catalogue.check_chain(default, base_url)
    default_gone = [item.model for item in listings if item.status == "not_listed"]
    for item in listings:
        if item.status == "not_listed":
            _say(
                f"The built-in default: {item.model} is not listed by OpenRouter ({item.note}). "
                "To change the default, measure again and change DEFAULT_MODEL_CHAIN in "
                "app/config.py."
            )
        elif item.status != "listed" or item.expires:
            _say("The built-in default: " + catalogue.describe(item))

    is_due, why = due(previous, problem, listed, now, forced)
    if not is_due:
        _say(f"No test today: {why}.")
        return 0
    if not settings.openrouter_configured:
        _warn(
            f"A test is due ({why}), but this job has no key, so nothing was tested and "
            "nothing was published. To let it test: add a repository secret named "
            "OPENROUTER_API_KEY (Settings, Secrets and variables, Actions)."
        )
        if default_gone or (previous is not None and stale):
            return _fail(
                "without a published list a service starts from the built-in default, and "
                "that default or the list is out of date."
            )
        return 0
    if listed is None:
        return _fail(no_list) if stale else 0

    chain = list(previous.chain) if previous else default
    gone = [
        item.model for item in catalogue.check_chain(chain, base_url) if item.status == "not_listed"
    ]
    last = (previous.record if previous else None) or {}
    others = rules.in_turn(
        [model for model in listed if model not in chain],
        last.get("tested", ()),
        last.get("left_out", ()),
    )
    tested = (chain + others)[: rules.CANDIDATE_LIMIT]
    left_out = (chain + others)[rules.CANDIDATE_LIMIT :]
    refused = await testing.why_no_test(settings, "schedule", testing.planned_requests(tested))
    if refused:
        _warn(
            f"A test is due ({why}), but too few free requests are left today for it and for "
            "visitors, so nothing was tested."
        )
        return _fail(no_list) if stale else 0

    _say(f"Testing the free models: {why}. Models: {', '.join(tested)}.")
    record = testing.new_record(
        "schedule",
        {"code": "schedule", "models": []},
        chain,
        [model for model in chain if model not in gone],
        gone,
        listed,
        tested,
        left_out,
        settings,
    )
    await testing.run_test(record, settings, chain_factory=chain_factory)

    for row in record["models"]:
        _say(
            f"- {row['model']}: {_words(row['verdict'])} ({_words(row['why'])}), "
            f"{row['correct']} of {row['calls']} right"
            + (f", {_words(row['placing'])}" if row["placing"] else "")
        )
    if record["outcome"] == "key_refused":
        return _fail(
            "OpenRouter refused the API key (401). Replace the repository secret "
            "OPENROUTER_API_KEY."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rules.seen_working(record["models"], record["chain"]):
        document = build(record, expected, datetime.now(UTC), _run_address())
        out_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8", newline="\n")
        if previous is None:
            was = "Nothing was published before."
        elif list(previous.chain) == document["chain"]:
            was = "Unchanged."
        else:
            was = f"Before: {', '.join(previous.chain)}."
        _say(
            f"Published chain: {', '.join(document['chain'])} "
            f"({record['requests']} free requests). {was}"
        )
        wrong = [
            row["model"]
            for row in record["models"]
            if row["model"] in document["chain"] and row["verdict"] == "failed"
        ]
        if wrong:
            # Published all the same: services need the date, and a test repairs a chain, it
            # never reorders one. The owner is the one who can.
            return _fail(
                f"{', '.join(wrong)} failed this test and keeps its place in the chain, because "
                "a test never reorders a chain. Measure again and change DEFAULT_MODEL_CHAIN "
                "in app/config.py."
            )
        return 0

    _warn(
        "The test could not confirm a working chain (no model of it passed), so the published "
        "chain was not changed." + (f" It is {age} days old." if age is not None else "")
    )
    if previous is not None:
        # Remember that a test ran and what was listed, so that the same test is not repeated
        # every day; the chain and its date stay as they were.
        document = json.loads(previous_text)
        document["job"] = {
            "checked": datetime.now(UTC).strftime(STAMP),
            "found": sorted(listed),
            "outcome": "nothing",
        }
        out_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8", newline="\n")
    silent = [row["model"] for row in record["models"] if not row["answered"]]
    if all(model in gone and model in silent for model in chain):
        # Two sources agree about every model of the chain, so this is not the weather.
        return _fail(
            f"OpenRouter no longer offers any model of the chain ({', '.join(chain)}), and no "
            "other free model passed the test, so services have no model that works. The job "
            "tests again tomorrow. Until a model passes, a deployment only answers with "
            "working models named in its OPENROUTER_MODELS."
        )
    return _fail(no_list) if stale else 0


def main(argv: Sequence[str] | None = None, settings: config.Settings | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.models.published",
        description=(
            "Decide whether a full test of the free models is due, run it if so, and write "
            "the file to publish. Without a key it only says what is due."
        ),
    )
    parser.add_argument(
        "--previous",
        type=Path,
        required=True,
        help="the file published last time (it may not exist)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="where to write the new file; written only after a test",
    )
    parser.add_argument("--force", action="store_true", help="test even when no test is due")
    args = parser.parse_args(argv)
    return asyncio.run(_job(args.previous, args.out, args.force, settings))


if __name__ == "__main__":  # pragma: no cover - command-line entry
    sys.exit(main())
