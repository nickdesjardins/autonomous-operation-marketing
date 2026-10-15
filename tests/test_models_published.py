"""Tests for the published list (``app.models.published``) and the scheduled job that writes it.

Nothing here reaches GitHub, OpenRouter or a model: the request for the published file is a
function passed in, the catalogue is a plain value, and the job's test runs against a scripted
chain client. Everything the file needs is defined in it.

Layout:

* the file     - what the job writes is what a service reads; the checks;
* when         - a test is due only for a reason;
* the job      - what is written, and the exit codes;
* the workflow - the files a push starts it for, and the two-job trust split.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from app import budget
from app.budget import Allowance
from app.config import DEFAULT_MODEL_CHAIN, Settings, parse_model_chain
from app.errors import ProviderAuthError, ProviderError
from app.models import cache, cases, catalogue, published, rules, testing
from app.models.cache import FREE_MODEL_ID
from app.models.catalogue import Listing
from app.models.published import Published, Unusable, due, fingerprint, parse
from app.schemas import ComplianceAuditResult, ViolationItem
from app.services.llm_client import Attempt, ChatOutcome, ModelFailure

FIRST, SECOND = parse_model_chain(DEFAULT_MODEL_CHAIN)[:2]
NEW, OTHER, BROKEN = "vendor/new:free", "vendor/other:free", "vendor/broken:free"
RUN = "https://github.com/someone/somewhere/actions/runs/7"
NOW = datetime(2026, 10, 20, 6, 30, tzinfo=UTC)
ROOT = Path(__file__).resolve().parents[1]


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "environment": "test",
        "checkpointer": "memory",
        "openrouter_api_key": "sk-or-test",
        "log_level": "WARNING",
    }
    values.update(overrides)
    return Settings(_env_file=None, _env_prefix="HERMETIC_TEST_", **values)


def stamp(moment: datetime) -> str:
    return moment.strftime(published.STAMP)


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
                    required_fix="Remove it.",
                )
            ],
            revision_summary="A cure claim and no NPN.",
        )
    return ComplianceAuditResult(
        is_compliant=True, risk_level="low", violations=[], revision_summary="Compliant."
    )


class FakeChain:
    name = "openrouter"

    def __init__(self, model: str, script: dict[str, str], listener: Any, log: list) -> None:
        self.model, self.script, self.listener, self.log = model, script, listener, log

    async def complete(
        self,
        *,
        messages: Any,
        schema_model: Any,
        temperature: Any,
        max_tokens: int,
        schema_name: str = "output",
    ) -> ChatOutcome:
        self.log.append((self.model, schema_name))
        what = self.script.get(self.model, "right")
        if what == "auth":
            raise ProviderAuthError("openrouter-auth", "the API key was rejected (401)")
        if what == "lost":
            self.listener(
                [ModelFailure(self.model, "ProviderError", True, "rate limited (429)")], False
            )
            raise ProviderError(self.model, "all models failed")
        if what == "wrong":
            parsed = (
                ComplianceAuditResult(
                    is_compliant=True,
                    risk_level="low",
                    violations=[],
                    revision_summary="Nothing wrong.",
                )
                if schema_name == "audit-catches"
                else right_answer(schema_name)
            )
        else:
            parsed = right_answer(schema_name)
        return ChatOutcome(
            self.model,
            json.dumps(parsed.model_dump()),
            "stop",
            parsed=parsed,
            attempts=[Attempt(self.model, "answered", 0.1)],
        )


class Job:
    """The job's world: the catalogue as values, the day's count, the models' answers."""

    def __init__(self) -> None:
        self.listed: list[str] | Exception = [FIRST, SECOND, NEW]
        self.gone: set[str] = set()
        self.allowance: Allowance | None = Allowance(10, 1000, 990)
        self.script: dict[str, str] = {}
        self.log: list[tuple[str, str]] = []

    def free_models(self, base_url: str, fetch: Any = None) -> list[str]:
        if isinstance(self.listed, Exception):
            raise self.listed
        return list(self.listed)

    def check_chain(self, models: list[str], base_url: str, fetch: Any = None) -> list[Listing]:
        return [
            Listing(m, "not_listed", note="withdrawn") if m in self.gone else Listing(m, "listed")
            for m in models
        ]

    def factory(self, settings: Settings, model: str, listener: Any) -> FakeChain:
        return FakeChain(model, self.script, listener, self.log)


@pytest.fixture
def job(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Job:
    made = Job()
    monkeypatch.setattr(cache, "PATH", tmp_path / "model_cache.json")
    cache.forget()
    monkeypatch.setattr(catalogue, "free_models", made.free_models)
    monkeypatch.setattr(catalogue, "check_chain", made.check_chain)
    monkeypatch.setattr(published, "chain_factory", made.factory)
    monkeypatch.setattr(rules, "SECONDS_BETWEEN_STARTS", 0.0)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)

    async def read_allowance(settings: Settings) -> Allowance | None:
        return made.allowance

    monkeypatch.setattr(budget, "read_openrouter_allowance", read_allowance)
    return made


async def no_sleep(seconds: float) -> None:
    """The pacing between request starts, without the waiting: a test of the file is not timed."""


async def finished_record(
    tested: list[str], chain: list[str], gone: list[str] = (), script: dict[str, str] | None = None
) -> dict[str, Any]:
    settings = make_settings()
    record = testing.new_record(
        "schedule",
        {"code": "schedule", "models": []},
        chain,
        [m for m in chain if m not in gone],
        list(gone),
        tested,
        tested,
        [],
        settings,
    )
    log: list = []
    await testing.run_test(
        record,
        settings,
        chain_factory=lambda s, m, listener: FakeChain(m, script or {}, listener, log),
        sleep=no_sleep,
    )
    return record


def valid(settings: Settings | None = None, **changes: Any) -> dict[str, Any]:
    document = {
        "version": published.VERSION,
        "fingerprint": fingerprint(settings or make_settings()),
        "tested": stamp(NOW - timedelta(days=2)),
        "chain": [FIRST, SECOND],
        "job": {
            "checked": stamp(NOW - timedelta(days=2)),
            "found": [FIRST, SECOND],
            "outcome": "published",
        },
        "by": {"run": RUN},
    }
    document.update(changes)
    return document


# ---------------------------------------------------------------------------
# The file
# ---------------------------------------------------------------------------


async def test_what_the_job_writes_is_what_the_service_reads() -> None:
    settings = make_settings()
    record = await finished_record([FIRST, SECOND, NEW], [FIRST, SECOND], script={SECOND: "wrong"})
    document = published.build(
        record, fingerprint(settings), NOW, {"run": RUN, "repository": "x", "commit": "y"}
    )
    text = json.dumps(document)
    assert len(text.encode()) < published.MAX_BYTES

    found = parse(text, fingerprint(settings), NOW)

    assert found.chain == (FIRST, SECOND, NEW) and found.tested == NOW
    assert found.by == {"run": RUN} and found.job == {
        "checked": stamp(NOW),
        "found": sorted([FIRST, SECOND, NEW]),
        "outcome": "published",
    }
    assert found.record is not None and found.record["by"] == "schedule"
    assert (
        found.record["calls"] == record["calls"] and found.record["requests"] == record["requests"]
    )
    # The verdicts are worked out again by this service's own rules, not read from the file.
    assert {row["model"]: (row["verdict"], row["placing"]) for row in found.record["models"]} == {
        FIRST: ("passed", "kept"),
        SECOND: ("failed", "kept"),
        NEW: ("passed", "added"),
    }
    assert found.record["outcome"] == "changed" and found.record["chain"] == [FIRST, SECOND, NEW]
    assert set(found.record) == set(record)


@pytest.mark.parametrize(
    ("change", "kind", "words"),
    [
        (lambda d: d.update(chain=[FIRST, "vendor/paid"]), "invalid", "':free'"),
        (lambda d: d.update(chain=[FIRST, FIRST]), "invalid", "different"),
        (lambda d: d.update(chain=[FIRST, SECOND, NEW, OTHER]), "invalid", "1 to 3"),
        (lambda d: d.update(chain=[]), "invalid", "1 to 3"),
        (lambda d: d.update(chain="not a list"), "invalid", "list"),
        (
            lambda d: d.update(version=published.VERSION + 1),
            "invalid",
            f"version {published.VERSION}",
        ),
        (lambda d: d.update(tested="yesterday"), "invalid", "when it was tested"),
        (lambda d: d.pop("tested"), "invalid", "when it was tested"),
        (lambda d: d.update(tested=stamp(NOW + timedelta(hours=1))), "invalid", "future"),
        (lambda d: d.update(fingerprint="0" * 16), "foreign", "another version of the test"),
    ],
    ids=[
        "paid-id",
        "twice",
        "too-long",
        "empty",
        "not-a-list",
        "version",
        "bad-date",
        "no-date",
        "future",
        "foreign",
    ],
)
def test_a_file_that_fails_a_check_is_not_used(change: Any, kind: str, words: str) -> None:
    document = valid()
    change(document)
    with pytest.raises(Unusable) as caught:
        parse(json.dumps(document), fingerprint(make_settings()), NOW)
    assert caught.value.kind == kind and words in str(caught.value)


def test_a_foreign_file_still_says_how_old_it_is() -> None:
    with pytest.raises(Unusable) as caught:
        parse(json.dumps(valid(fingerprint="0" * 16)), fingerprint(make_settings()), NOW)
    assert caught.value.kind == "foreign" and caught.value.tested == stamp(NOW - timedelta(days=2))


@pytest.mark.parametrize(
    "text",
    [
        "",
        "not json",
        "[1, 2]",
        "null",
        "x" * (published.MAX_BYTES + 1),
        "[" * 20_000 + "]" * 20_000,
    ],
    ids=["empty", "not-json", "a-list", "null", "too-large", "nested-absurdly"],
)
def test_text_that_is_no_file_is_unusable_and_never_a_crash(text: str) -> None:
    with pytest.raises(Unusable) as caught:
        parse(text, fingerprint(make_settings()), NOW)
    assert caught.value.kind == "invalid"


def test_the_fingerprint_follows_the_test_and_nothing_else(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = make_settings()
    assert (
        fingerprint(settings) == fingerprint(make_settings()) and len(fingerprint(settings)) == 16
    )
    assert fingerprint(make_settings(openrouter_strict_json_schema=False)) != fingerprint(settings)
    # The reasoning control changes what a model answers, so a list tested under another
    # choice is another test.
    assert fingerprint(make_settings(openrouter_reasoning="low")) != fingerprint(settings)
    assert fingerprint(make_settings(openrouter_timeout_s=30, log_level="DEBUG")) == fingerprint(
        settings
    )
    before = fingerprint(settings)
    monkeypatch.setattr(cases, "fingerprint_material", lambda: ["another test"])
    assert fingerprint(settings) != before
    monkeypatch.setattr(rules, "MAX_MISSES", 1)
    monkeypatch.setattr(cases, "fingerprint_material", lambda: cases.CASES and [])
    assert fingerprint(settings) != before


@pytest.mark.parametrize("number", ["1e999", "-1e999", "NaN", "Infinity", "0", "86400", "true"])
def test_a_time_limit_that_is_no_time_limit_drops_the_record_not_the_file(number: str) -> None:
    test = f'{{"started": "{stamp(NOW)}", "tested": ["{FIRST}"], "timeout_seconds": {number}, "calls": []}}'
    text = json.dumps(valid())[:-1] + f', "test": {test}}}'
    found = parse(text, fingerprint(make_settings()), NOW)
    assert found.record is None and found.chain == (FIRST, SECOND)


def test_what_a_file_says_about_its_own_test_is_cleaned_and_not_believed() -> None:
    calls = [
        {
            "model": FIRST,
            "case": "synthesis",
            "outcome": "answered",
            "correct": True,
            "seconds": 1.234,
            "error_type": None,
            "message": "secret setting",
        },
        {
            "model": FIRST,
            "case": "audit-catches",
            "outcome": "no_answer",
            "correct": None,
            "seconds": 0.1,
            "error_type": "SomethingElse",
        },
        {
            "model": "vendor/paid",
            "case": "synthesis",
            "outcome": "answered",
            "correct": True,
            "seconds": 1.0,
        },
        {
            "model": FIRST,
            "case": "no-such-case",
            "outcome": "answered",
            "correct": True,
            "seconds": 1.0,
        },
        {
            "model": FIRST,
            "case": "audit-passes",
            "outcome": "answered",
            "correct": "yes",
            "seconds": 1.0,
        },
        {
            "model": FIRST,
            "case": "audit-passes",
            "outcome": "answered",
            "correct": True,
            "seconds": -1,
        },
        "not a call",
    ]
    test = {
        "started": stamp(NOW - timedelta(hours=1)),
        "tested": [FIRST, "vendor/paid", NEW],
        "left_out": [OTHER, "@preset/x"],
        "gone": [SECOND],
        "found": [FIRST, NEW, OTHER],
        "timeout_seconds": 60,
        "calls": calls,
        "chain_before": [FIRST, SECOND],
        "models": [{"model": FIRST, "verdict": "passed"}],
        "outcome": "changed",
        "extra": "dropped",
    }
    found = parse(
        json.dumps(valid(chain=[FIRST, NEW], test=test)), fingerprint(make_settings()), NOW
    )
    record = found.record
    assert record is not None and record["tested"] == [FIRST, NEW] and record["left_out"] == [OTHER]
    assert record["calls"] == [
        {
            "model": FIRST,
            "case": "synthesis",
            "outcome": "answered",
            "correct": True,
            "seconds": 1.23,
            "error_type": None,
        },
        {
            "model": FIRST,
            "case": "audit-catches",
            "outcome": "no_answer",
            "correct": None,
            "seconds": 0.1,
            "error_type": None,
        },
    ]
    assert {row["model"]: row["verdict"] for row in record["models"]} == {
        FIRST: "not_judged",
        NEW: "not_judged",
    }
    assert (
        record["gone"] == [SECOND]
        and record["chain_before"] == [FIRST, SECOND]
        and record["chain"] == [FIRST, NEW]
    )
    assert "extra" not in record and record["outcome"] == "changed" and record["requests"] == 2


@pytest.mark.parametrize(
    "run",
    [
        "https://evil.example/actions/runs/7",
        "https://github.com/a/b/actions/runs/x",
        7,
        None,
        "https://github.com/a/b/actions/runs/7/../../../",
    ],
)
def test_only_the_address_of_the_jobs_run_reaches_the_log(run: Any) -> None:
    found = parse(json.dumps(valid(by={"run": run})), fingerprint(make_settings()), NOW)
    assert found.by == {}
    assert parse(json.dumps(valid()), fingerprint(make_settings()), NOW).by == {"run": RUN}


def test_what_the_job_remembers_cannot_put_the_next_test_off() -> None:
    ahead = {"checked": "2099-01-01T00:00:00Z", "found": [FIRST], "outcome": "odd"}
    found = parse(json.dumps(valid(job=ahead)), fingerprint(make_settings()), NOW)
    assert found.job == {
        "checked": stamp(NOW - timedelta(days=2)),
        "found": [FIRST],
        "outcome": "published",
    }
    assert (
        parse(json.dumps(valid(job="junk", by="junk")), fingerprint(make_settings()), NOW).by == {}
    )


def test_the_request_names_what_went_wrong() -> None:
    with pytest.raises(Unusable) as caught:
        published.fetch("http://example.test/chain.json", lambda url: (200, "{}"))
    assert caught.value.kind == "invalid" and "https" in str(caught.value)

    def broken(url: str) -> tuple[int, str]:
        raise ConnectionError("no route")

    with pytest.raises(Unusable) as caught:
        published.fetch("https://example.test/chain.json", broken)
    assert caught.value.kind == "unreachable" and "ConnectionError" in str(caught.value)
    with pytest.raises(Unusable) as caught:
        published.fetch("https://example.test/chain.json", lambda url: (404, ""))
    assert caught.value.kind == "missing"
    with pytest.raises(Unusable) as caught:
        published.fetch("https://example.test/chain.json", lambda url: (500, ""))
    assert caught.value.kind == "unreachable"
    assert published.fetch("https://example.test/chain.json", lambda url: (200, "{}")) == "{}"


def test_build_never_writes_an_id_that_is_not_free() -> None:
    record = {"chain": [FIRST, "vendor/paid"], "found": []}
    with pytest.raises(ValueError, match=":free"):
        published.build(record, "f" * 16, NOW, {})
    with pytest.raises(ValueError):
        published.build({"chain": [], "found": []}, "f" * 16, NOW, {})


# ---------------------------------------------------------------------------
# When a test is due
# ---------------------------------------------------------------------------


def seen(
    job: dict[str, Any] | None = None,
    tested: datetime = NOW - timedelta(days=3),
    chain: tuple[str, ...] = (FIRST, SECOND),
) -> Published:
    return Published(
        chain=chain,
        tested=tested,
        job=job
        or {"checked": stamp(tested), "found": [FIRST, SECOND, OTHER], "outcome": "published"},
    )


def test_a_test_is_due_only_for_a_reason() -> None:
    listed = [FIRST, SECOND, OTHER]
    assert due(None, "nothing has been published yet", listed, NOW) == (
        True,
        "there is no usable published list (nothing has been published yet)",
    )
    assert due(seen(), "", listed, NOW, forced=True) == (True, "a test was asked for by hand")
    assert due(seen(), "", listed, NOW)[0] is False  # three days old, same list
    assert due(seen(), "", None, NOW) == (
        False,
        "OpenRouter's list could not be read to see whether anything changed",
    )
    assert due(seen(), "", listed + [NEW], NOW) == (
        True,
        f"OpenRouter's list of free models changed since the last test: {NEW}",
    )
    assert due(seen(), "", [SECOND, OTHER], NOW)[1].endswith(
        f"changed since the last test: {FIRST}"
    )
    week_ago = NOW.replace(hour=6, minute=22) - timedelta(days=7)
    assert due(seen(tested=week_ago), "", listed, NOW.replace(hour=6, minute=17)) == (
        True,
        "the last test is 7.0 days old",
    )
    assert (
        due(seen(tested=week_ago + timedelta(days=1)), "", listed, NOW.replace(hour=6, minute=17))[
            0
        ]
        is False
    )


def test_a_test_that_showed_nothing_is_repeated_the_next_day_and_not_every_day() -> None:
    nothing = {"checked": stamp(NOW - timedelta(hours=24)), "found": [OTHER], "outcome": "nothing"}
    assert due(seen(nothing), "", [OTHER], NOW) == (True, "the last test could confirm nothing")
    assert published.RETRY_AFTER_HOURS == 20
    assert (
        due(seen({**nothing, "checked": stamp(NOW - timedelta(hours=20))}), "", [OTHER], NOW)[0]
        is True
    )
    assert (
        due(seen({**nothing, "checked": stamp(NOW - timedelta(hours=19))}), "", [OTHER], NOW)[0]
        is False
    )
    done = {"checked": stamp(NOW - timedelta(hours=24)), "found": [OTHER], "outcome": "published"}
    assert due(seen(done), "", [OTHER], NOW)[0] is False


def test_a_change_of_the_list_that_no_test_could_act_on_costs_no_test() -> None:
    full = seen(
        chain=(FIRST, SECOND, OTHER),
        job={
            "checked": stamp(NOW - timedelta(days=3)),
            "found": [FIRST, SECOND, OTHER, BROKEN],
            "outcome": "published",
        },
    )
    assert due(full, "", [FIRST, SECOND, OTHER, BROKEN, NEW], NOW)[0] is False  # no place for it
    assert due(full, "", [FIRST, SECOND, OTHER], NOW)[0] is False  # a model outside the chain went
    assert due(full, "", [FIRST, OTHER, BROKEN], NOW)[1].endswith(
        f"changed since the last test: {SECOND}"
    )
    assert (
        due(seen(), "", [FIRST, SECOND, OTHER, NEW], NOW)[0] is True
    )  # a free place, and a newcomer
    assert due(seen(), "", [FIRST, SECOND], NOW)[0] is False  # a model outside went: nothing to do


# ---------------------------------------------------------------------------
# The job
# ---------------------------------------------------------------------------


def run_job(
    tmp_path: Path,
    settings: Settings,
    previous: dict[str, Any] | str | None = None,
    force: bool = False,
) -> tuple[int, dict[str, Any] | None]:
    before, out = tmp_path / "previous" / "chain.json", tmp_path / "out" / "chain.json"
    if previous is not None:
        before.parent.mkdir(parents=True, exist_ok=True)
        before.write_text(
            previous if isinstance(previous, str) else json.dumps(previous), encoding="utf-8"
        )
    argv = ["--previous", str(before), "--out", str(out)] + (["--force"] if force else [])
    code = published.main(argv, settings=settings)
    return code, json.loads(out.read_text(encoding="utf-8")) if out.is_file() else None


def test_the_job_tests_and_writes_a_file_a_service_accepts(
    job: Job, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.com")
    monkeypatch.setenv("GITHUB_REPOSITORY", "someone/somewhere")
    monkeypatch.setenv("GITHUB_RUN_ID", "7")
    settings = make_settings()

    code, written = run_job(tmp_path, settings)

    assert code == 0 and written is not None
    assert written["chain"] == [FIRST, SECOND, NEW] and written["fingerprint"] == fingerprint(
        settings
    )
    assert written["by"]["run"] == RUN and written["job"]["outcome"] == "published"
    found = parse(json.dumps(written), fingerprint(settings), datetime.now(UTC))
    assert found.chain == (FIRST, SECOND, NEW) and found.record is not None
    assert len(job.log) == 9  # three models, three cases each
    printed = capsys.readouterr().out
    assert "Testing the free models: there is no usable published list" in printed
    assert (
        f"Published chain: {FIRST}, {SECOND}, {NEW} (9 free requests). Nothing was published before."
        in printed
    )
    assert "::warning::This job publishes to someone/somewhere" in printed  # a fork is told
    assert "Published chain" in summary.read_text(encoding="utf-8")


def test_the_job_spends_nothing_when_no_test_is_due(
    job: Job, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = make_settings()
    job.listed = [FIRST, SECOND]
    code, written = run_job(
        tmp_path,
        settings,
        valid(
            settings,
            tested=stamp(datetime.now(UTC) - timedelta(days=1)),
            job={
                "checked": stamp(datetime.now(UTC) - timedelta(days=1)),
                "found": [FIRST, SECOND],
                "outcome": "published",
            },
        ),
    )
    assert code == 0 and written is None and job.log == []
    assert "No test today" in capsys.readouterr().out


def test_the_job_starts_from_the_published_chain_and_drops_a_model_only_when_gone_and_silent(
    job: Job, tmp_path: Path
) -> None:
    settings = make_settings()
    job.listed = [SECOND, NEW]
    job.gone = {FIRST}
    job.script = {FIRST: "lost"}
    previous = valid(
        settings,
        chain=[FIRST, SECOND],
        tested=stamp(datetime.now(UTC) - timedelta(days=8)),
        job={
            "checked": stamp(datetime.now(UTC) - timedelta(days=8)),
            "found": [FIRST, SECOND],
            "outcome": "published",
        },
    )
    code, written = run_job(tmp_path, settings, previous)
    assert code == 0 and written["chain"] == [SECOND, NEW]
    # Gone but answering: it stays, and the chain is published with it in place.
    job.script = {}
    job.log.clear()
    code, written = run_job(tmp_path, settings, previous)
    assert code == 0 and written["chain"] == [FIRST, SECOND, NEW]


@pytest.mark.parametrize(("default_gone", "exit_code"), [(False, 0), (True, 1)])
def test_the_job_without_a_key_says_what_is_due_and_tests_nothing(
    job: Job, tmp_path: Path, default_gone: bool, exit_code: int, capsys: pytest.CaptureFixture[str]
) -> None:
    if default_gone:
        job.gone = {FIRST}
    code, written = run_job(tmp_path, make_settings(openrouter_api_key=None))
    assert code == exit_code and written is None and job.log == []
    printed = capsys.readouterr().out
    assert "::warning::A test is due" in printed and "OPENROUTER_API_KEY" in printed
    assert ("Failing this run on purpose" in printed) is default_gone


def test_a_refused_key_fails_the_run_at_once(
    job: Job, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    job.script = {FIRST: "auth"}
    code, written = run_job(tmp_path, make_settings())
    assert code == 1 and written is None and len(job.log) == 1
    assert "Failing this run on purpose: OpenRouter refused the API key" in capsys.readouterr().out


def test_a_test_that_confirms_nothing_keeps_the_chain_and_remembers_the_attempt(
    job: Job, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = make_settings()
    job.script = {FIRST: "lost", SECOND: "lost", NEW: "lost"}
    tested = datetime.now(UTC) - timedelta(days=8)
    previous = valid(
        settings,
        tested=stamp(tested),
        job={"checked": stamp(tested), "found": [FIRST, SECOND], "outcome": "published"},
    )
    code, written = run_job(tmp_path, settings, previous)
    assert code == 0 and written["chain"] == [FIRST, SECOND] and written["tested"] == stamp(tested)
    assert written["job"]["outcome"] == "nothing" and written["job"]["found"] == sorted(
        [FIRST, SECOND, NEW]
    )
    assert "::warning::The test could not confirm a working chain" in capsys.readouterr().out
    # Three weeks without a confirmed chain: the run turns red.
    old = valid(
        settings,
        tested=stamp(tested - timedelta(days=14)),
        job={
            "checked": stamp(tested - timedelta(days=14)),
            "found": [FIRST, SECOND],
            "outcome": "published",
        },
    )
    assert run_job(tmp_path, settings, old)[0] == 1


def test_the_job_fails_when_no_model_of_the_chain_is_left(
    job: Job, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    job.gone = {FIRST, SECOND}
    job.listed = [NEW]
    job.script = {FIRST: "lost", SECOND: "lost", NEW: "wrong"}
    code, written = run_job(tmp_path, make_settings())
    assert code == 1 and written is None
    assert "OpenRouter no longer offers any model of the chain" in capsys.readouterr().out


def test_the_job_publishes_and_fails_when_a_model_of_the_chain_failed_the_test(
    job: Job, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    job.script = {FIRST: "wrong"}
    code, written = run_job(tmp_path, make_settings())
    assert code == 1 and written["chain"] == [FIRST, SECOND, NEW]
    assert (
        f"Failing this run on purpose: {FIRST} failed this test and keeps its place"
        in capsys.readouterr().out
    )


def test_the_job_leaves_the_reserve_for_visitors_and_tests_although_the_count_is_unknown(
    job: Job, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = make_settings()
    # A list three days old that does not know NEW: a test is due, and the chain is not stale.
    previous = valid(
        settings,
        tested=stamp(datetime.now(UTC) - timedelta(days=3)),
        job={
            "checked": stamp(datetime.now(UTC) - timedelta(days=3)),
            "found": [FIRST, SECOND],
            "outcome": "published",
        },
    )
    job.allowance = Allowance(900, 1000, 100)  # 9 requests would leave 91: under the reserve
    code, written = run_job(tmp_path, settings, previous)
    assert code == 0 and written is None and job.log == []
    assert "too few free requests are left today" in capsys.readouterr().out
    # With nothing published at all, the same refusal turns the run red: services are on their own.
    assert run_job(tmp_path / "nothing-published", settings)[0] == 1
    job.allowance = None
    code, written = run_job(tmp_path, settings, previous)
    assert code == 0 and written is not None


def test_a_previous_file_that_cannot_be_read_is_nothing_published(
    job: Job, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, written = run_job(tmp_path, make_settings(), previous="not json")
    assert code == 0 and written is not None
    assert "there is no usable published list (it is not valid JSON)" in capsys.readouterr().out


def test_the_job_takes_four_models_and_the_others_go_first_next_time(
    job: Job, tmp_path: Path
) -> None:
    settings = make_settings()
    job.listed = [FIRST, SECOND, NEW, OTHER, BROKEN]
    code, written = run_job(tmp_path, settings, force=True)
    assert code == 0 and written["test"]["tested"] == [FIRST, SECOND, NEW, OTHER]
    assert written["test"]["left_out"] == [BROKEN]
    job.log.clear()
    code, written = run_job(tmp_path, settings, written, force=True)
    assert written["test"]["tested"] == [FIRST, SECOND, NEW, BROKEN] and written["test"][
        "left_out"
    ] == [OTHER]


def test_an_unreadable_list_is_no_test(
    job: Job, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    job.listed = TimeoutError("no list")
    code, written = run_job(tmp_path, make_settings())
    assert (
        code == 1 and written is None and job.log == []
    )  # nothing published, nothing testable: red
    assert "could not be read (TimeoutError)" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The workflow
# ---------------------------------------------------------------------------


def test_the_workflow_starts_on_the_files_the_test_is_made_of() -> None:
    workflows = ROOT / ".github" / "workflows"
    workflow = (workflows / "model-check.yml").read_text(encoding="utf-8")
    block = r"^  push:\n(?:    .*\n)*?    paths:\n((?:      - \S+\n)+)"
    listed = re.search(block, workflow, re.MULTILINE)
    assert listed is not None, "the workflow no longer names the files a push starts it for"
    paths = re.findall(r"^      - (\S+)$", listed.group(1), re.MULTILINE)
    for path in paths:
        assert (ROOT / path).is_file(), f"{path} is named in the workflow and is not in the project"
    assert set(paths) >= {
        "app/config.py",
        "app/schemas.py",
        "app/graph/prompts.py",
        "app/compliance/rules.py",
        "app/seed.py",
        "app/models/cases.py",
        "app/models/catalogue.py",
        "app/models/rules.py",
        "app/models/testing.py",
        "app/models/published.py",
        "app/services/llm_client.py",
        "requirements.txt",
        ".github/workflows/model-check.yml",
    }
    assert (
        "python -m app.models.published --previous previous/chain.json --out out/chain.json $force"
        in workflow
    )
    suite = (workflows / "tests.yml").read_text(encoding="utf-8")
    actions = set(re.findall(r"uses: (\S+)", workflow))
    assert actions and actions == set(re.findall(r"uses: (\S+)", suite))
    assert (
        "python-version-file: .python-version" in workflow and (ROOT / ".python-version").is_file()
    )
    assert re.search(r'^\s+- cron: "\d{1,2} \d{1,2} \* \* \*"$', workflow, re.MULTILINE)
    assert "workflow_dispatch:" in workflow and "force:" in workflow


def test_the_workflow_checks_the_file_again_and_keeps_key_and_write_access_apart() -> None:
    text = (ROOT / ".github" / "workflows" / "model-check.yml").read_text(encoding="utf-8")
    test_job, publish_job = text.split("\n  publish:\n")
    publish_job, _ = publish_job.split("\n  verdict:\n")
    # The same pattern, size, length and version as the service's own checks.
    assert f'free = re.compile(r"{FREE_MODEL_ID.pattern}")' in publish_job
    assert (
        "and all(isinstance(model, str) and free.fullmatch(model) for model in chain)"
        in publish_job
    )
    assert "and len(set(chain)) == len(chain)" in publish_job
    assert f'os.path.getsize("../new-chain.json") <= {published.MAX_BYTES:_}\n' in publish_job
    assert f"1 <= len(chain) <= {rules.PLACES}" in publish_job
    assert f'new.get("version") == {published.VERSION}' in publish_job
    written = re.search(r'stamp = re\.compile\(r"([^"]+)"\)', publish_job)
    assert written is not None and re.fullmatch(written.group(1), stamp(NOW))
    assert 'and stamp.fullmatch(new["tested"])' in publish_job
    assert (
        "    if: needs.test.outputs.file != '' && github.ref == 'refs/heads/main'\n" in publish_job
    )
    assert "\npermissions:\n  contents: read\n" in test_job
    assert publish_job.count("contents: write") == 1 and "contents: write" not in test_job
    assert text.count("secrets.OPENROUTER_API_KEY") == 1 and "secrets." not in publish_job
    assert "pip install" not in publish_job and "actions/checkout" not in publish_job
    assert (
        "git show FETCH_HEAD:chain.json > previous/chain.json || rm -f previous/chain.json"
        in test_job
    )
    assert "git ls-remote --exit-code --heads origin tested-models" in test_job
    assert "tested-models" in publish_job and "chain.json" in publish_job
