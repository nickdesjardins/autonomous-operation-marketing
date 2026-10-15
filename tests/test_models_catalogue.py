"""Tests for the OpenRouter catalogue lookups (``app.models.catalogue``).

Nothing here reaches the network: every lookup is given a fake ``fetch``. The fixtures and
helpers are local to this file, so it passes with or without ``tests/conftest.py``.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from app.config import DEFAULT_MODEL_CHAIN, Settings, parse_model_chain
from app.models import cache, catalogue
from app.models.catalogue import Listing, check_chain, free_models, look_up

BASE = "https://openrouter.ai/api/v1"
MODEL_CHAIN = parse_model_chain(DEFAULT_MODEL_CHAIN)


def answer(status: int, payload: Any) -> catalogue.Fetch:
    return lambda url: (status, json.dumps(payload))


def entry(model: str, prompt: str = "0", completion: str = "0", structured: bool = True) -> dict:
    parameters = ["temperature", "structured_outputs"] if structured else ["temperature"]
    return {
        "id": model,
        "pricing": {"prompt": prompt, "completion": completion},
        "supported_parameters": parameters,
    }


# ---------------------------------------------------------------------------
# One model
# ---------------------------------------------------------------------------


def test_the_lookup_address_keeps_the_slash_as_it_is() -> None:
    assert catalogue.lookup_url(BASE + "/", "a/b:free") == BASE + "/model/a/b:free"


def test_a_listed_model() -> None:
    fetch = answer(200, {"data": {"id": "a/b:free", "expiration_date": "2027-01-31"}})
    assert look_up("a/b:free", BASE, fetch) == Listing("a/b:free", "listed", expires="2027-01-31")


def test_only_openrouters_own_model_not_found_counts_as_withdrawn() -> None:
    withdrawn = answer(404, {"error": {"message": "Model not found: a/b:free"}})
    assert look_up("a/b:free", BASE, withdrawn).status == "not_listed"
    # A bare "Not Found" is a wrong address, not a statement about the model.
    wrong_address = answer(404, {"error": {"message": "Not Found"}})
    assert look_up("a/b:free", BASE, wrong_address).status == "unknown"


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (200, {"data": {"nothing": "here"}}),
        (429, {"error": {"message": "rate limited"}}),
        (503, "<html>down</html>"),
        (200, "not json at all"),
    ],
)
def test_any_other_answer_is_unknown(status: int, body: Any) -> None:
    fetch = (lambda url: (status, body)) if isinstance(body, str) else answer(status, body)
    listing = look_up("a/b:free", BASE, fetch)
    assert listing.status == "unknown" and f"HTTP {status}" in listing.note


def test_an_unreachable_catalogue_is_unknown_and_never_raises() -> None:
    def fetch(url: str) -> tuple[int, str]:
        raise TimeoutError("no route")

    listing = look_up("a/b:free", BASE, fetch)
    assert listing.status == "unknown" and "TimeoutError" in listing.note


@pytest.mark.parametrize("model", ["@preset/mine", "no-slash"])
def test_ids_the_catalogue_cannot_hold_are_not_looked_up(model: str) -> None:
    asked: list[str] = []

    def fetch(url: str) -> tuple[int, str]:
        asked.append(url)
        return 200, "{}"

    assert look_up(model, BASE, fetch).status == "unknown" and asked == []


def test_an_end_date_of_a_different_entry_is_not_reported() -> None:
    fetch = answer(200, {"data": {"id": "a/b", "expiration_date": "2027-01-31"}})
    assert look_up("a/b:nitro", BASE, fetch) == Listing("a/b:nitro", "listed")


def test_a_near_end_date_needs_attention() -> None:
    today = date(2026, 10, 15)
    soon = Listing("a/b:free", "listed", expires="2026-10-20")
    far = Listing("a/b:free", "listed", expires="2027-10-20")
    assert soon.days_left(today) == 5 and soon.needs_attention(today)
    assert not far.needs_attention(today)
    assert Listing("a/b:free", "not_listed").needs_attention(today)
    assert "5 days from today" in catalogue.describe(soon, today)
    assert catalogue.describe(far, today).startswith("a/b:free is listed, and OpenRouter")
    assert catalogue.describe(Listing("a/b:free", "listed")) == "a/b:free is listed."
    assert "OPENROUTER_MODELS" in catalogue.describe(Listing("a/b:free", "not_listed", note="x"))
    assert "could not be checked" in catalogue.describe(Listing("a/b:free", "unknown", note="x"))


# ---------------------------------------------------------------------------
# The whole chain, and the free models on offer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        (BASE, True),
        ("https://openrouter.ai/api/v1/", True),
        ("https://api.example.com/v1", False),
        ("http://openrouter.ai/api/v1", False),  # not the encrypted address
        ("https://openrouter.ai.example.com/api/v1", False),  # a lookalike host
        ("not an address", False),
    ],
)
def test_only_openrouters_own_address_counts(base_url: str, expected: bool) -> None:
    assert catalogue.is_openrouter(base_url) is expected


def test_the_chain_is_reported_in_order_and_other_hosts_are_never_asked() -> None:
    def fetch(url: str) -> tuple[int, str]:
        if url.endswith("gone/model:free"):
            return 404, json.dumps({"error": {"message": "Model not found: gone/model:free"}})
        return 200, json.dumps({"data": {"id": url.split("/model/")[1]}})

    listings = check_chain(["gone/model:free", "live/model:free"], BASE, fetch)
    assert [(item.model, item.status) for item in listings] == [
        ("gone/model:free", "not_listed"),
        ("live/model:free", "listed"),
    ]
    assert check_chain([], BASE, fetch) == []
    elsewhere = check_chain(["a/b:free"], "https://api.example.com/v1", fetch)
    assert elsewhere[0].status == "unknown" and "not OpenRouter" in elsewhere[0].note


def test_free_models_keeps_only_free_structured_output_models_once_each() -> None:
    listed = [
        entry("a/alpha:free"),
        entry("b/beta:free", structured=False),  # no structured output
        entry("c/gamma:free", prompt="0.000001"),  # a price after all
        entry("d/delta", structured=True),  # not a free id
        entry("a/alpha:free"),  # listed twice
        {"id": "e/epsilon:free", "pricing": {"prompt": "0"}},  # half a price
        "not an entry",
        entry("f/zeta:free", prompt="0.0", completion="0"),
        {
            "id": "@preset/mine:free",
            "pricing": {"prompt": "0", "completion": "0"},
            "supported_parameters": ["structured_outputs"],
        },
    ]
    asked: list[str] = []

    def fetch(url: str) -> tuple[int, str]:
        asked.append(url)
        return 200, json.dumps({"data": listed})

    assert free_models(BASE, fetch) == ["a/alpha:free", "f/zeta:free"]
    assert asked == [BASE + catalogue.FREE_MODELS_QUERY]


def test_no_list_is_never_taken_for_an_empty_list() -> None:
    with pytest.raises(ValueError):
        free_models(BASE, answer(503, {"error": "down"}))
    with pytest.raises(ValueError):
        free_models(BASE, answer(200, {"data": "not a list"}))
    with pytest.raises(ValueError):
        free_models("https://api.example.com/v1", answer(200, {"data": []}))
    assert free_models(BASE, answer(200, {"data": []})) == []


def test_the_free_id_rule_is_the_caches_own() -> None:
    assert catalogue.FREE_MODEL_ID is cache.FREE_MODEL_ID
    assert catalogue.is_free_model("vendor/model:free")
    assert not catalogue.is_free_model("vendor/model")
    assert not catalogue.is_free_model("vendor/model:nitro")
    assert not catalogue.is_free_model("@preset/thing:free")
    assert not catalogue.is_free_model(None)


def test_the_request_names_this_project_and_reads_a_bounded_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeAnswer:
        status = 200

        def __init__(self) -> None:
            self.asked_for: list[int | None] = []

        def read(self, amount: int | None = None) -> bytes:
            self.asked_for.append(amount)
            return b"x" * (amount or 0)

        def __enter__(self) -> FakeAnswer:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    seen: list[tuple[Any, float]] = []
    reply = FakeAnswer()

    class Opener:
        def open(self, request: Any, timeout: float) -> FakeAnswer:
            seen.append((request, timeout))
            return reply

    monkeypatch.setattr(catalogue, "_OPENER", Opener())
    status, body = catalogue._http_get(BASE + "/model/a/b:free")

    ((request, timeout),) = seen
    assert request.full_url == BASE + "/model/a/b:free"
    assert (
        request.get_header("User-agent")
        == catalogue.USER_AGENT
        == "multi-agent-ops-pipeline model-check"
    )
    assert timeout == catalogue.LOOKUP_TIMEOUT_SECONDS
    assert reply.asked_for == [catalogue.MAX_ANSWER_BYTES]
    assert status == 200 and len(body) == catalogue.MAX_ANSWER_BYTES


@pytest.mark.parametrize(
    ("target", "followed"),
    [
        ("https://openrouter.ai/api/v1/model/a/b", True),
        ("https://example.com/api/v1/model/a/b", False),
        ("http://openrouter.ai/api/v1/model/a/b", False),
        ("https://openrouter.ai.example.com/model/a/b", False),
    ],
)
def test_a_redirect_is_followed_only_on_openrouters_own_host(target: str, followed: bool) -> None:
    import urllib.request
    from email.message import Message

    handler = catalogue._SameHostRedirects()
    request = urllib.request.Request(BASE + "/model/a/b:free")
    redirected = handler.redirect_request(request, None, 302, "Found", Message(), target)
    assert (redirected is not None) == followed


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------


@pytest.fixture
def command_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Settings:
    settings = Settings(
        _env_file=None,
        _env_prefix="HERMETIC_TEST_",
        checkpointer="memory",
        openrouter_api_key="sk-or-test",
        log_level="WARNING",
    )
    monkeypatch.setattr("app.config.get_settings", lambda: settings)
    monkeypatch.setattr(cache, "PATH", tmp_path / "model_cache.json")
    cache.forget()
    return settings


@pytest.mark.parametrize(
    ("listings", "exit_code"),
    [
        ([Listing(m, "listed") for m in MODEL_CHAIN], 0),
        (
            [
                Listing(MODEL_CHAIN[0], "listed"),
                Listing(MODEL_CHAIN[1], "not_listed", note="withdrawn"),
            ],
            1,
        ),
        ([Listing(MODEL_CHAIN[0], "listed", expires=date.today().isoformat())], 1),
        ([Listing(MODEL_CHAIN[0], "listed", expires="2999-01-01")], 0),
        ([Listing(MODEL_CHAIN[0], "unknown", note="catalogue not reachable (TimeoutError)")], 1),
    ],
    ids=["all-listed", "one-withdrawn", "ends-today", "ends-far-away", "could-not-check"],
)
def test_the_command_checks_the_chain_in_use_and_exits_accordingly(
    listings: list[Listing],
    exit_code: int,
    command_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    asked: list[tuple[list[str], str]] = []

    def stand_in(models: list[str], base_url: str, fetch: Any = None) -> list[Listing]:
        asked.append((list(models), base_url))
        return listings

    monkeypatch.setattr(catalogue, "check_chain", stand_in)
    assert catalogue.main() == exit_code
    assert asked == [(MODEL_CHAIN, BASE)]
    printed = capsys.readouterr().out
    assert all(item.model in printed for item in listings)
    assert "not the built-in default" not in printed


def test_the_command_says_when_it_checks_a_saved_chain(
    command_settings: Settings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cache.use(["vendor/chosen:free"], {"kind": "own", "tested": "x", "changed": "y", "why": {}})
    monkeypatch.setattr(
        catalogue,
        "check_chain",
        lambda models, base_url, fetch=None: [Listing(models[0], "not_listed", note="withdrawn")],
    )
    assert catalogue.main() == 1
    printed = capsys.readouterr().out
    assert "saved in model_cache.json" in printed
    assert "replaces it by itself after the next request that fails" in printed
