"""The OpenAI-compatible chain against a stub ``chat.completions.create``.

Every failure a free model produces in practice is replayed here (rate limits, in-band
upstream errors, withdrawal, timeouts, truncation, empty or invalid output) and must send the
call to the next model, while a rejected key must stop the chain at once. No request leaves
the process: constructing the real SDK client is impossible in this suite.
"""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace
from typing import Any

import httpx2
import openai
import pytest

from app.errors import ConfigurationError, ProviderAuthError, ProviderError
from app.graph.prompts import json_mode_suffix
from app.schemas import AuditEntry, ChannelCopyDraft, ComplianceAuditResult
from app.services.llm_client import (
    Attempt,
    ChatOutcome,
    ModelFailure,
    OpenAICompatibleChain,
    RateLimiter,
    build_chat,
    extract_json_object,
    parse_as,
    reasoning_body,
    strict_json_schema,
)

from .conftest import GOOD_DRAFT, make_settings

ALPHA, BETA = "vendor/alpha:free", "vendor/beta:free"
GOOD = {"is_compliant": True, "risk_level": "low", "violations": [], "revision_summary": "Fine."}
MESSAGES = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]


def _completion(
    content: str | None, finish: str = "stop", error: dict[str, Any] | None = None
) -> SimpleNamespace:
    message = SimpleNamespace(content=content)
    completion = SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish)])
    if error is not None:
        completion.error = error
        completion.choices = []
    return completion


def _request() -> httpx2.Request:
    return httpx2.Request("POST", "https://llm.test/v1/chat/completions")


def _status_error(status: int, body: dict[str, Any] | None = None) -> openai.APIStatusError:
    response = httpx2.Response(status, request=_request())
    cls = {
        400: openai.BadRequestError,
        401: openai.AuthenticationError,
        403: openai.PermissionDeniedError,
        404: openai.NotFoundError,
        429: openai.RateLimitError,
        502: openai.InternalServerError,
    }[status]
    return cls(
        f"status {status}", response=response, body=body or {"error": {"message": f"s{status}"}}
    )


class StubChat:
    """Stands in for ``AsyncOpenAI().chat.completions``: outcomes consumed in order."""

    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _chain(stub: StubChat, models: Any = (ALPHA, BETA), **options: Any) -> OpenAICompatibleChain:
    settings: dict[str, Any] = {
        "provider": "openrouter",
        "api_key": "test-key",
        "base_url": "https://llm.test/v1",
        "models": list(models) if not callable(models) else models,
        "requests_per_minute": 0,
        "strict_schema": True,
        "free_only": True,
        "client": stub,
    }
    settings.update(options)
    return OpenAICompatibleChain(**settings)


async def _complete(chain: OpenAICompatibleChain) -> ChatOutcome:
    return await chain.complete(
        messages=MESSAGES,
        schema_model=ComplianceAuditResult,
        temperature=0.0,
        max_tokens=512,
        schema_name="compliance_audit_result",
    )


# ------------------------------------------------------------------------ pure helpers


def test_strict_schema_hardens_every_object_including_nested_ones() -> None:
    schema = strict_json_schema(ComplianceAuditResult)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        "is_compliant",
        "risk_level",
        "violations",
        "revision_summary",
    }
    nested = schema["$defs"]["ViolationItem"]
    assert nested["additionalProperties"] is False
    assert set(nested["required"]) == {
        "field_name",
        "violating_text",
        "regulation_rule",
        "required_fix",
    }
    draft = strict_json_schema(ChannelCopyDraft)
    assert draft["properties"]["marketplace_bullets"]["minItems"] == 5  # pydantic limits are kept
    # A model that allows extras and has defaults: pydantic leaves both out, strict mode needs both.
    plain = AuditEntry.model_json_schema()
    assert "additionalProperties" not in plain and "deterministic_findings" not in plain["required"]
    hardened = strict_json_schema(AuditEntry)
    assert hardened["additionalProperties"] is False
    assert set(hardened["required"]) == set(hardened["properties"])
    assert "additionalProperties" not in AuditEntry.model_json_schema()  # the original is untouched


@pytest.mark.parametrize(
    "raw",
    [
        json.dumps(GOOD),
        "```json\n" + json.dumps(GOOD) + "\n```",
        "```\n" + json.dumps(GOOD) + "```",
        "Sure, here is the audit:\n" + json.dumps(GOOD) + "\nHope this helps.",
        "[" + json.dumps(GOOD) + "]",
        "  \n" + json.dumps(GOOD) + "\n\n",
    ],
    ids=["plain", "fence", "bare-fence", "prose", "list-of-one", "whitespace"],
)
def test_extract_json_object_tolerates_fences_prose_and_a_list_of_one(raw: str) -> None:
    assert extract_json_object(raw) == GOOD


@pytest.mark.parametrize(
    "raw", ["", "   ", "not json", "[1, 2]", "[{}, {}]", "42", "{broken", None]
)
def test_extract_json_object_rejects_garbage(raw: str | None) -> None:
    with pytest.raises(ValueError):
        extract_json_object(raw)


def test_parse_as_validates_the_schema() -> None:
    assert parse_as(ComplianceAuditResult, json.dumps(GOOD)).is_compliant is True
    with pytest.raises(ValueError):
        parse_as(ComplianceAuditResult, json.dumps({**GOOD, "risk_level": "severe"}))


async def test_rate_limiter_spaces_request_starts() -> None:
    limiter = RateLimiter(requests_per_minute=600)  # 100 ms apart
    started = time.perf_counter()
    for _ in range(3):
        await limiter.wait()
    assert time.perf_counter() - started >= 0.19
    unlimited = RateLimiter(requests_per_minute=0)
    started = time.perf_counter()
    await asyncio.gather(*(unlimited.wait() for _ in range(50)))
    assert time.perf_counter() - started < 0.1


def test_reasoning_body_follows_the_setting() -> None:
    assert reasoning_body("default") is None
    assert reasoning_body("off") == {"reasoning": {"enabled": False}}
    assert reasoning_body("low") == {"reasoning": {"effort": "low"}}


# --------------------------------------------------------------------- the chain walker


async def test_strict_schema_answer_is_parsed_and_the_request_has_the_expected_shape() -> None:
    stub = StubChat([_completion(json.dumps(GOOD))])
    outcome = await _complete(_chain(stub, extra_body={"reasoning": {"enabled": False}}))

    assert outcome.model == ALPHA and outcome.json_mode is False and outcome.refused is False
    assert outcome.parsed == ComplianceAuditResult.model_validate(GOOD)
    assert outcome.attempts == [Attempt(ALPHA, "answered", outcome.attempts[0].seconds)]
    assert outcome.fallback_triggered is False
    (call,) = stub.calls
    assert call["model"] == ALPHA and call["stream"] is False
    assert call["temperature"] == 0.0 and call["max_tokens"] == 512
    assert call["extra_body"] == {"reasoning": {"enabled": False}}
    assert call["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "compliance_audit_result",
            "strict": True,
            "schema": strict_json_schema(ComplianceAuditResult),
        },
    }
    assert call["messages"] == MESSAGES  # no schema suffix in strict mode
    assert MESSAGES[0]["content"] == "sys"  # the caller's messages are never mutated


async def test_no_temperature_means_the_parameter_is_not_sent() -> None:
    stub = StubChat([_completion(json.dumps(GOOD))])
    await _chain(stub).complete(
        messages=MESSAGES, schema_model=ComplianceAuditResult, temperature=None, max_tokens=64
    )
    assert "temperature" not in stub.calls[0] and "extra_body" not in stub.calls[0]
    assert stub.calls[0]["response_format"]["json_schema"]["name"] == "output"


async def test_a_rejected_strict_schema_is_retried_in_json_mode_on_the_same_model() -> None:
    rejection = _status_error(400, body={"error": {"message": "response_format is not supported"}})
    stub = StubChat([rejection, _completion(json.dumps(GOOD))])
    outcome = await _complete(_chain(stub))

    assert outcome.model == ALPHA and outcome.json_mode is True
    assert outcome.parsed.is_compliant is True
    assert [call["model"] for call in stub.calls] == [ALPHA, ALPHA]
    retry = stub.calls[1]
    assert retry["response_format"] == {"type": "json_object"}
    suffix = json_mode_suffix(strict_json_schema(ComplianceAuditResult))
    assert retry["messages"][0]["content"] == "sys" + suffix
    assert retry["messages"][1] == MESSAGES[1]
    assert MESSAGES[0]["content"] == "sys"
    # The rejection is not a failure of the model: nothing is reported, no fallback is noted.
    assert [a.outcome for a in outcome.attempts] == ["answered"]


async def test_json_mode_only_when_strict_schema_is_disabled() -> None:
    stub = StubChat([_completion(json.dumps(GOOD))])
    outcome = await _complete(_chain(stub, strict_schema=False))
    assert outcome.json_mode is True
    assert stub.calls[0]["response_format"] == {"type": "json_object"}
    assert stub.calls[0]["messages"][0]["content"].startswith("sys\n\nOutput format")


@pytest.mark.parametrize(
    "first, detail",
    [
        (_status_error(429), "rate limited (429)"),
        (
            _completion(None, error={"code": 502, "message": "upstream provider error"}),
            "upstream provider error (code 502)",
        ),
        (_status_error(404), "model not found (404); withdrawn?"),
        (_status_error(403), "request forbidden (403)"),
        (_status_error(400), "request rejected (400)"),
        (_status_error(502), "HTTP 502"),
        (openai.APITimeoutError(_request()), "request timed out"),
        (openai.APIConnectionError(request=_request()), "connection error"),
        (_completion(json.dumps(GOOD)[:30], finish="length"), "response truncated at max_tokens"),
        (_completion("   "), "empty response"),
        (_completion(None), "empty response"),
        (SimpleNamespace(choices=[]), "no choices in response"),
        (_completion("not json at all"), "invalid output (ValueError)"),
        (_completion(json.dumps({"is_compliant": True})), "invalid output (ValidationError)"),
        (_completion(json.dumps({**GOOD, "extra": 1})), "invalid output (ValidationError)"),
    ],
    ids=[
        "429",
        "in-band-error",
        "404",
        "403",
        "400",
        "502",
        "timeout",
        "connection",
        "truncation",
        "empty",
        "none",
        "no-choices",
        "invalid-json",
        "schema-mismatch",
        "extra-field",
    ],
)
async def test_failures_cascade_to_the_next_model(first: Any, detail: str) -> None:
    stub = StubChat([first, _completion(json.dumps(GOOD))])
    outcome = await _complete(_chain(stub))

    assert outcome.model == BETA and outcome.parsed.is_compliant is True
    assert [call["model"] for call in stub.calls] == [ALPHA, BETA]
    assert outcome.fallback_triggered is True
    assert [a.model for a in outcome.attempts] == [ALPHA, BETA]
    assert outcome.attempts[0].outcome in {"failed", "invalid"}
    assert outcome.attempts[0].detail == detail
    assert outcome.attempts[1].outcome == "answered"


async def test_a_rejected_key_stops_the_chain_at_once() -> None:
    stub = StubChat([_status_error(401), _completion(json.dumps(GOOD))])
    with pytest.raises(ProviderAuthError) as info:
        await _complete(_chain(stub))
    assert len(stub.calls) == 1
    assert info.value.provider == "openrouter-auth" and info.value.retryable is False
    assert isinstance(info.value, ProviderError)


async def test_when_every_model_fails_the_error_is_retryable_if_any_failure_was() -> None:
    stub = StubChat([_status_error(429), _status_error(404)])
    with pytest.raises(ProviderError) as info:
        await _complete(_chain(stub))
    assert info.value.retryable is True
    assert info.value.provider == "openrouter"
    assert ALPHA in info.value.message and BETA in info.value.message
    assert "rate limited" in info.value.message and "withdrawn" in info.value.message

    stub = StubChat([_status_error(404), _completion("garbage")])
    with pytest.raises(ProviderError) as info:
        await _complete(_chain(stub))
    assert info.value.retryable is False


async def test_a_content_filter_refusal_is_returned_not_cascaded() -> None:
    stub = StubChat([_completion(None, finish="content_filter"), _completion(json.dumps(GOOD))])
    outcome = await _complete(_chain(stub))
    assert outcome.refused is True and outcome.parsed is None and outcome.model == ALPHA
    assert outcome.finish_reason == "content_filter"
    assert len(stub.calls) == 1
    assert outcome.attempts == [
        Attempt(ALPHA, "refused", outcome.attempts[0].seconds, "content filter")
    ]


async def test_free_only_skips_a_paid_id_without_calling_it() -> None:
    heard: list[tuple[list[ModelFailure], bool]] = []
    stub = StubChat([_completion(json.dumps(GOOD))])
    chain = _chain(
        stub, models=["vendor/paid-model", BETA], failure_listener=lambda f, a: heard.append((f, a))
    )
    outcome = await _complete(chain)

    assert outcome.model == BETA
    assert [call["model"] for call in stub.calls] == [BETA]
    assert outcome.attempts[0] == Attempt("vendor/paid-model", "skipped", 0.0, "not a free model")
    assert heard == [
        ([ModelFailure("vendor/paid-model", "NotFree", False, "not a free model")], True)
    ]

    lenient = _chain(
        StubChat([_completion(json.dumps(GOOD))]), models=["vendor/paid-model"], free_only=False
    )
    assert (await _complete(lenient)).model == "vendor/paid-model"


async def test_the_failure_listener_hears_every_failure_and_whether_the_call_was_answered() -> None:
    heard: list[tuple[list[ModelFailure], bool]] = []

    def listener(failures: list[ModelFailure], answered: bool) -> None:
        heard.append((list(failures), answered))

    stub = StubChat([_status_error(429), _completion(json.dumps(GOOD))])
    await _complete(_chain(stub, failure_listener=listener))
    assert heard == [([ModelFailure(ALPHA, "ProviderError", True, "rate limited (429)")], True)]

    heard.clear()
    stub = StubChat([_completion("garbage"), _status_error(404)])
    with pytest.raises(ProviderError):
        await _complete(_chain(stub, failure_listener=listener))
    (failures, answered) = heard[0]
    assert answered is False
    assert [(f.model, f.error_type, f.retryable) for f in failures] == [
        (ALPHA, "SchemaValidationError", False),
        (BETA, "ProviderError", False),
    ]

    heard.clear()
    await _complete(_chain(StubChat([_completion(json.dumps(GOOD))]), failure_listener=listener))
    assert heard == []  # nothing failed, nothing to tell


async def test_a_raising_listener_never_breaks_the_request() -> None:
    def listener(failures: list[ModelFailure], answered: bool) -> None:
        raise RuntimeError("listener bug")

    stub = StubChat([_status_error(429), _completion(json.dumps(GOOD))])
    assert (await _complete(_chain(stub, failure_listener=listener))).model == BETA


async def test_the_models_source_is_read_again_on_every_call() -> None:
    current = [ALPHA]
    stub = StubChat([_completion(json.dumps(GOOD)), _completion(json.dumps(GOOD))])
    chain = _chain(stub, models=lambda: list(current))
    assert chain.models == [ALPHA] and chain.describe() == {
        "provider": "openrouter",
        "models": [ALPHA],
    }

    assert (await _complete(chain)).model == ALPHA
    current[:] = ["vendor/gamma:free"]
    assert (await _complete(chain)).model == "vendor/gamma:free"
    assert [call["model"] for call in stub.calls] == [ALPHA, "vendor/gamma:free"]

    current.clear()
    with pytest.raises(ConfigurationError, match="empty"):
        await _complete(chain)


def test_the_chain_needs_a_key_unless_a_client_is_given() -> None:
    with pytest.raises(ConfigurationError, match="API key"):
        OpenAICompatibleChain(
            provider="openrouter", api_key=None, base_url="https://llm.test", models=[ALPHA]
        )
    with pytest.raises(ConfigurationError):
        OpenAICompatibleChain(
            provider="openrouter", api_key="", base_url="https://llm.test", models=[ALPHA]
        )


def test_constructing_the_real_sdk_client_is_impossible_in_the_suite() -> None:
    with pytest.raises(RuntimeError, match="offline"):
        OpenAICompatibleChain(
            provider="openrouter", api_key="k", base_url="https://llm.test", models=[ALPHA]
        )


# ------------------------------------------------------------------------------- factory


def test_build_chat_wires_the_openai_compatible_provider_to_one_model() -> None:
    stub = StubChat([])
    chat = build_chat(make_settings(openai_model="gpt-4o-mini"), client=stub)
    assert isinstance(chat, OpenAICompatibleChain)
    assert chat.name == "openai_compatible" and chat.models == ["gpt-4o-mini"]
    assert chat.free_only is False and chat.strict_schema is True
    assert chat.extra_body is None


def test_build_chat_wires_openrouter_to_the_settings_chain_and_reasoning() -> None:
    settings = make_settings(
        llm_provider="openrouter",
        openrouter_api_key="test-key",
        openrouter_models=f"{ALPHA},{BETA}",
        openrouter_reasoning="off",
        openrouter_requests_per_minute=0,
    )
    heard: list[Any] = []
    chat = build_chat(settings, client=StubChat([]), failure_listener=lambda f, a: heard.append(a))
    assert isinstance(chat, OpenAICompatibleChain)
    assert chat.name == "openrouter" and chat.models == [ALPHA, BETA]
    assert chat.free_only is True
    assert chat.extra_body == {"reasoning": {"enabled": False}}
    settings.openrouter_models = f"{BETA}"
    assert chat.models == [BETA]  # the chain follows the settings, call by call


async def test_the_draft_schema_round_trips_through_the_chain() -> None:
    stub = StubChat([_completion(json.dumps(GOOD_DRAFT))])
    outcome = await _chain(stub).complete(
        messages=MESSAGES, schema_model=ChannelCopyDraft, temperature=0.2, max_tokens=2048
    )
    assert outcome.parsed == ChannelCopyDraft.model_validate(GOOD_DRAFT)
