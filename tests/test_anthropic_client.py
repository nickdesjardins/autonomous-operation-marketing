"""Claude through the official SDK, against a fake ``messages.create``.

What matters here is the request shape (the schema in ``output_config``, no temperature, the
system prompts joined) and the mapping of the SDK's outcomes onto the service's own contract.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest

from app.errors import ConfigurationError, ProviderAuthError, ProviderError
from app.schemas import ComplianceAuditResult
from app.services.anthropic_client import AnthropicChat
from app.services.llm_client import build_chat, strict_json_schema

from .conftest import make_settings

GOOD = {"is_compliant": True, "risk_level": "low", "violations": [], "revision_summary": "Fine."}
MESSAGES = [
    {"role": "system", "content": "You audit."},
    {"role": "system", "content": "Be strict."},
    {"role": "user", "content": "Audit this."},
]


def _response(text: str | None = None, stop_reason: str = "end_turn") -> SimpleNamespace:
    content = [] if text is None else [SimpleNamespace(type="text", text=text)]
    return SimpleNamespace(stop_reason=stop_reason, content=content)


def _status_error(status: int) -> anthropic.APIStatusError:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(status, request=request)
    cls = {
        400: anthropic.BadRequestError,
        401: anthropic.AuthenticationError,
        404: anthropic.NotFoundError,
        429: anthropic.RateLimitError,
        529: anthropic.APIStatusError,
    }[status]
    body = {"type": "error", "error": {"type": "api_error", "message": f"status {status}"}}
    return cls(f"status {status}", response=response, body=body)


class StubMessages:
    """Stands in for ``AsyncAnthropic().messages``: answers consumed in order."""

    def __init__(self, answers: list[Any]) -> None:
        self.answers = list(answers)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer


def _chat(*answers: Any, **settings: Any) -> tuple[AnthropicChat, StubMessages]:
    stub = StubMessages(list(answers))
    client = SimpleNamespace(messages=stub)
    chat = AnthropicChat(
        make_settings(llm_provider="anthropic", anthropic_api_key="test-key", **settings),
        client=client,
    )
    return chat, stub


async def _complete(chat: AnthropicChat, messages: list[dict[str, str]] = MESSAGES) -> Any:
    return await chat.complete(
        messages=messages,
        schema_model=ComplianceAuditResult,
        temperature=0.0,
        max_tokens=1536,
        schema_name="compliance_audit_result",
    )


async def test_the_request_carries_the_schema_and_the_effort_but_no_temperature() -> None:
    chat, stub = _chat(_response(json.dumps(GOOD)), anthropic_effort="medium")
    outcome = await _complete(chat)

    (call,) = stub.calls
    assert call["model"] == "claude-opus-5-5" == chat.model
    assert call["max_tokens"] == 1536
    assert call["system"] == "You audit.\n\nBe strict."
    assert call["messages"] == [{"role": "user", "content": "Audit this."}]
    assert call["output_config"] == {
        "effort": "medium",
        "format": {"type": "json_schema", "schema": strict_json_schema(ComplianceAuditResult)},
    }
    assert "temperature" not in call
    assert outcome.model == chat.model and outcome.refused is False
    assert outcome.parsed == ComplianceAuditResult.model_validate(GOOD)
    assert outcome.finish_reason == "end_turn" and outcome.json_mode is False
    assert [a.outcome for a in outcome.attempts] == ["answered"]
    assert outcome.fallback_triggered is False


async def test_without_a_system_message_the_sdk_sentinel_is_sent() -> None:
    chat, stub = _chat(_response(json.dumps(GOOD)))
    await _complete(chat, [{"role": "user", "content": "Audit this."}])
    assert stub.calls[0]["system"] is anthropic.NOT_GIVEN


async def test_a_refusal_is_a_refused_outcome_not_an_error() -> None:
    chat, _ = _chat(_response(None, stop_reason="refusal"))
    outcome = await _complete(chat)
    assert outcome.refused is True and outcome.parsed is None and outcome.content == ""
    assert outcome.attempts[0].outcome == "refused"


async def test_truncation_empty_and_invalid_answers_are_provider_errors() -> None:
    chat, _ = _chat(_response(json.dumps(GOOD)[:40], stop_reason="max_tokens"))
    with pytest.raises(ProviderError, match="max_tokens") as info:
        await _complete(chat)
    assert info.value.retryable is False

    chat, _ = _chat(_response("   "))
    with pytest.raises(ProviderError, match="empty") as info:
        await _complete(chat)
    assert info.value.retryable is True

    chat, _ = _chat(_response('{"is_compliant": "maybe"}'))
    with pytest.raises(ProviderError, match="invalid output"):
        await _complete(chat)


async def test_a_fenced_answer_is_still_parsed() -> None:
    chat, _ = _chat(_response("```json\n" + json.dumps(GOOD) + "\n```"))
    assert (await _complete(chat)).parsed.is_compliant is True


async def test_the_key_rejection_is_an_auth_error() -> None:
    chat, _ = _chat(_status_error(401))
    with pytest.raises(ProviderAuthError) as info:
        await _complete(chat)
    assert info.value.retryable is False and info.value.provider == "anthropic-auth"


@pytest.mark.parametrize(
    "error, retryable, text",
    [
        (_status_error(429), True, "rate limited"),
        (_status_error(529), True, "HTTP 529"),
        (_status_error(404), False, "not found"),
        (_status_error(400), False, "rejected"),
        (
            anthropic.APITimeoutError(httpx2.Request("POST", "https://api.anthropic.com")),
            True,
            "timed out",
        ),
        (
            anthropic.APIConnectionError(
                request=httpx2.Request("POST", "https://api.anthropic.com")
            ),
            True,
            "connection error",
        ),
    ],
    ids=["429", "529", "404", "400", "timeout", "connection"],
)
async def test_sdk_errors_map_to_provider_errors(
    error: Exception, retryable: bool, text: str
) -> None:
    chat, _ = _chat(error)
    with pytest.raises(ProviderError, match=text) as info:
        await _complete(chat)
    assert info.value.retryable is retryable
    assert "status" not in str(info.value) or "HTTP" in str(info.value)  # no SDK body leaks


def test_the_provider_needs_a_key_unless_a_client_is_given() -> None:
    settings = make_settings(llm_provider="anthropic", openai_api_key=None)
    assert settings.llm_configured is False
    with pytest.raises(ConfigurationError, match="ANTHROPIC_API_KEY"):
        AnthropicChat(settings)
    chat = AnthropicChat(settings, client=SimpleNamespace(messages=StubMessages([])))
    assert chat.describe() == {
        "provider": "anthropic",
        "models": ["claude-opus-5-5"],
        "effort": "low",
    }


def test_build_chat_picks_the_anthropic_client_for_that_provider() -> None:
    settings = make_settings(llm_provider="anthropic", anthropic_api_key="test-key")
    chat = build_chat(settings, client=SimpleNamespace(messages=StubMessages([])))
    assert isinstance(chat, AnthropicChat) and chat.name == "anthropic"


def test_constructing_the_real_sdk_client_is_impossible_in_the_suite() -> None:
    settings = make_settings(llm_provider="anthropic", anthropic_api_key="test-key")
    with pytest.raises(RuntimeError, match="offline"):
        AnthropicChat(settings)
