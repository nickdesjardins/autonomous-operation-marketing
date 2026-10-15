"""Structured model calls: one protocol, an OpenAI-compatible chain, and the factory.

The pattern is the one Projects 1 and 2 of this portfolio settled on:

* The service talks to OpenRouter over its OpenAI-compatible Chat Completions API and gets its
  resilience from routing across *several models* rather than several vendors. The dominant
  failure of a free model is not a bad answer but a transient 429, an in-band "upstream
  provider error", or a withdrawal (404), each on one specific model.
* Structured output is requested through the strict ``json_schema`` response format first; a
  model that rejects it is retried in plain JSON mode with the schema spelled out in the system
  prompt. Either way the answer is validated against the Pydantic schema before the graph sees
  it, and an answer that does not validate sends the call to the next model.
* Request starts are paced to OpenRouter's free-model allowance (20 per minute).
* Error messages that reach callers are short and sanitised; upstream bodies go to logs only.

The same class serves any OpenAI-compatible endpoint (OpenAI itself, Gemini's compatibility
endpoint, Groq) with a single-model chain. Claude goes through the official Anthropic SDK in
:mod:`app.services.anthropic_client`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from app.config import Settings, is_free_model
from app.errors import ConfigurationError, ProviderAuthError, ProviderError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# --------------------------------------------------------------------------- schemas


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """A strict-mode JSON Schema for OpenAI-compatible structured outputs.

    Strict mode requires every property to be listed in ``required`` and
    ``additionalProperties: false`` on every object, including those under ``$defs``.
    Pydantic's ``minItems``/``maxItems`` and string length limits are kept: a provider that
    cannot honour a keyword ignores it, and validation catches the difference afterwards.
    """

    def harden(node: Any) -> Any:
        if isinstance(node, dict):
            hardened = {key: harden(value) for key, value in node.items()}
            if hardened.get("type") == "object" and "properties" in hardened:
                hardened["required"] = list(hardened["properties"].keys())
                hardened["additionalProperties"] = False
            return hardened
        if isinstance(node, list):
            return [harden(item) for item in node]
        return node

    return harden(model.model_json_schema())


# --------------------------------------------------------------------------- parsing

_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)
#: Some models emit prose or a chain of thought before the JSON despite the instructions;
#: recover the last balanced-looking object rather than failing the call.
_TRAILING_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_object(raw: str | None) -> dict[str, Any]:
    """Turn a model's raw text into a JSON object (``dict``) or raise ``ValueError``."""
    if raw is None or not raw.strip():
        raise ValueError("model returned an empty response")
    match = _CODE_FENCE_RE.match(raw)
    cleaned = match.group(1) if match else raw.strip()
    try:
        payload: Any = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        found = _TRAILING_OBJECT_RE.search(cleaned)
        if not found:
            raise ValueError(f"invalid JSON at position {exc.pos}: {exc.msg}") from exc
        try:
            payload = json.loads(found.group(0))
        except json.JSONDecodeError as inner:
            raise ValueError(f"invalid JSON at position {inner.pos}: {inner.msg}") from inner
    if isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], dict):
        payload = payload[0]
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object, got {type(payload).__name__}")
    return payload


def parse_as[M: BaseModel](model: type[M], raw: str | None) -> M:
    """Decode and validate a model's text as ``model``."""
    return model.model_validate(extract_json_object(raw))


# ------------------------------------------------------------------------ rate limit


class RateLimiter:
    """Spaces request starts so a burst can never exceed a per-minute allowance."""

    def __init__(self, requests_per_minute: int) -> None:
        self.interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        self._lock = asyncio.Lock()
        self._next_start = 0.0

    async def wait(self) -> None:
        if self.interval <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            delay = self._next_start - now
            if delay > 0:
                await asyncio.sleep(delay)
                now = time.monotonic()
            self._next_start = max(now, self._next_start) + self.interval


# ----------------------------------------------------------------------------- types


@dataclass(slots=True)
class Attempt:
    """One model's turn in a chain, for the trace the playground shows."""

    model: str
    outcome: str  # answered | failed | invalid | refused | skipped
    seconds: float
    detail: str | None = None


@dataclass(slots=True)
class ChatOutcome:
    model: str  # the model that actually answered (fallbacks may have been used)
    content: str
    finish_reason: str | None
    refused: bool = False  # the model's content filter declined
    json_mode: bool = False  # strict schema was rejected; plain JSON mode was used instead
    parsed: Any = None  # the validated Pydantic instance
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def fallback_triggered(self) -> bool:
        return any(attempt.outcome != "answered" for attempt in self.attempts)


@dataclass(frozen=True)
class ModelFailure:
    """One model of the chain that did not answer a real request (for the model chooser)."""

    model: str
    error_type: str
    retryable: bool | None
    message: str


FailureListener = Callable[[list[ModelFailure], bool], None]


class StructuredChat(Protocol):
    """What the graph needs from a model: a validated instance of a schema, or an error."""

    name: str

    async def complete(
        self,
        *,
        messages: Sequence[dict[str, str]],
        schema_model: type[T],
        temperature: float | None,
        max_tokens: int,
        schema_name: str = "output",
    ) -> ChatOutcome: ...

    def describe(self) -> dict[str, Any]: ...


class _StrictSchemaUnsupported(Exception):
    """The model rejected the strict ``json_schema`` response format."""


class _ExtraBodyRejected(Exception):
    """The model's provider rejected the vendor extension sent in ``extra_body``."""


ModelsSource = Sequence[str] | Callable[[], Sequence[str]]


# ----------------------------------------------------------------------------- chain


class OpenAICompatibleChain:
    """Walks an ordered model chain over an OpenAI-compatible Chat Completions API."""

    def __init__(
        self,
        *,
        provider: str,
        api_key: str | None,
        base_url: str,
        models: ModelsSource,
        timeout_s: float = 60.0,
        requests_per_minute: int = 0,
        strict_schema: bool = True,
        free_only: bool = False,
        headers: dict[str, str] | None = None,
        client: object | None = None,
        limiter: RateLimiter | None = None,
        failure_listener: FailureListener | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        from openai import AsyncOpenAI

        self.name = provider
        self._models = models
        #: Vendor extensions sent with every request (OpenRouter's reasoning control).
        self.extra_body = dict(extra_body) if extra_body else None
        #: Models whose provider rejected the extensions: they are called without them.
        self._without_extras: set[str] = set()
        if client is None and not api_key:
            raise ConfigurationError(f"an API key is required for the {provider} provider")
        self._client = client or AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_s,
            max_retries=0,  # the chain owns retry semantics (fallback), not the SDK
            default_headers=headers or None,
        )
        self.limiter = limiter or RateLimiter(requests_per_minute)
        self.strict_schema = strict_schema
        self.free_only = free_only
        self._listener = failure_listener

    @property
    def models(self) -> list[str]:
        source = self._models
        return list(source() if callable(source) else source)

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "models": self.models}

    async def complete(
        self,
        *,
        messages: Sequence[dict[str, str]],
        schema_model: type[T],
        temperature: float | None,
        max_tokens: int,
        schema_name: str = "output",
    ) -> ChatOutcome:
        """Try each model in order; return the first validated answer.

        A model is skipped (and the next one tried) on rate limits, timeouts, upstream errors,
        withdrawal, truncation, empty output, and when its JSON does not validate. When the
        strict schema is rejected, the same model is retried once in plain JSON mode.
        """
        models = self.models
        if not models:
            raise ConfigurationError("the model chain is empty")
        schema = strict_json_schema(schema_model)
        failures: list[ModelFailure] = []
        attempts: list[Attempt] = []
        any_retryable = False
        for model in models:
            if self.free_only and not is_free_model(model):
                # The last door a paid model could come through, whatever a list or a file
                # said. It stays shut, and the call moves on.
                logger.error("Refusing to call %s: it is not a free model", model)
                attempts.append(Attempt(model, "skipped", 0.0, "not a free model"))
                failures.append(ModelFailure(model, "NotFree", False, "not a free model"))
                continue
            modes = ["strict", "json"] if self.strict_schema else ["json"]
            for mode in modes:
                started = time.perf_counter()
                try:
                    try:
                        outcome = await self._call(
                            model, messages, max_tokens, temperature, schema, schema_name, mode
                        )
                    except _ExtraBodyRejected:
                        # The model's provider refuses the vendor extension (the reasoning
                        # control): remembered for this model, and the request goes out again
                        # without it. The strict-schema fallback below is unaffected.
                        self._without_extras.add(model)
                        logger.info("%s rejected the request extensions; retrying without", model)
                        outcome = await self._call(
                            model, messages, max_tokens, temperature, schema, schema_name, mode
                        )
                except _StrictSchemaUnsupported:
                    logger.info("%s rejected strict json_schema; retrying in JSON mode", model)
                    continue
                except ProviderAuthError:
                    raise  # no model can fix a rejected key
                except ProviderError as exc:
                    elapsed = time.perf_counter() - started
                    failures.append(
                        ModelFailure(model, type(exc).__name__, exc.retryable, exc.message)
                    )
                    attempts.append(Attempt(model, "failed", round(elapsed, 3), exc.message))
                    any_retryable = any_retryable or exc.retryable
                    logger.warning(
                        "%s model %s failed (%s); trying next", self.name, model, exc.message
                    )
                    break
                elapsed = time.perf_counter() - started
                if outcome.refused:
                    attempts.append(Attempt(model, "refused", round(elapsed, 3), "content filter"))
                    outcome.attempts = attempts
                    self._tell(failures, answered=True)
                    return outcome
                try:
                    outcome.parsed = parse_as(schema_model, outcome.content)
                except Exception as exc:  # noqa: BLE001 - any validation failure cascades
                    detail = f"invalid output ({type(exc).__name__})"
                    failures.append(ModelFailure(model, "SchemaValidationError", False, detail))
                    attempts.append(Attempt(model, "invalid", round(elapsed, 3), detail))
                    logger.warning("%s model %s returned invalid output: %s", self.name, model, exc)
                    break
                attempts.append(Attempt(model, "answered", round(elapsed, 3)))
                outcome.attempts = attempts
                self._tell(failures, answered=True)
                return outcome
        self._tell(failures, answered=False)
        raise ProviderError(
            self.name,
            "all models failed (" + "; ".join(f"{f.model}: {f.message}" for f in failures) + ")",
            retryable=any_retryable,
        )

    def _tell(self, failures: list[ModelFailure], *, answered: bool) -> None:
        if failures and self._listener is not None:
            try:
                self._listener(failures, answered)
            except Exception:  # noqa: BLE001 - the listener must never break a request
                logger.exception("The model failure listener raised")

    async def _call(
        self,
        model: str,
        messages: Sequence[dict[str, str]],
        max_tokens: int,
        temperature: float | None,
        schema: dict[str, Any],
        schema_name: str,
        mode: str,
    ) -> ChatOutcome:
        import openai

        from app.graph.prompts import json_mode_suffix

        request_messages = [dict(m) for m in messages]
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": request_messages,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        with_extras = bool(self.extra_body) and model not in self._without_extras
        if with_extras:
            kwargs["extra_body"] = dict(self.extra_body or {})
        if mode == "strict":
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema},
            }
        else:
            kwargs["response_format"] = {"type": "json_object"}
            if request_messages and request_messages[0].get("role") == "system":
                request_messages[0]["content"] += json_mode_suffix(schema)

        await self.limiter.wait()
        try:
            completion = await self._client.chat.completions.create(**kwargs)  # type: ignore[attr-defined]
        except openai.AuthenticationError as exc:
            raise ProviderAuthError(f"{self.name}-auth", "the API key was rejected (401)") from exc
        except openai.PermissionDeniedError as exc:
            raise ProviderError(model, "request forbidden (403)") from exc
        except openai.RateLimitError as exc:
            raise ProviderError(model, "rate limited (429)", retryable=True) from exc
        except openai.NotFoundError as exc:
            raise ProviderError(model, "model not found (404); withdrawn?") from exc
        except openai.BadRequestError as exc:
            body = json.dumps(getattr(exc, "body", None), default=str).lower()
            if with_extras and any(key in body for key in (self.extra_body or {})):
                raise _ExtraBodyRejected() from exc
            if mode == "strict" and any(
                marker in body for marker in ("response_format", "json_schema", "structured")
            ):
                raise _StrictSchemaUnsupported() from exc
            if with_extras:
                # A 400 with no named cause, on a request that carried an extension: the
                # extension is the first suspect, and dropping it costs one more request.
                raise _ExtraBodyRejected() from exc
            raise ProviderError(model, "request rejected (400)") from exc
        except openai.APIStatusError as exc:
            raise ProviderError(
                model, f"HTTP {exc.status_code}", retryable=exc.status_code >= 500
            ) from exc
        except openai.APITimeoutError as exc:
            raise ProviderError(model, "request timed out", retryable=True) from exc
        except openai.APIConnectionError as exc:
            raise ProviderError(model, "connection error", retryable=True) from exc

        # OpenRouter reports upstream provider failures in-band: HTTP 200 with an "error"
        # member instead of choices. This is the most common free-model failure.
        error = getattr(completion, "error", None)
        if error:
            code = error.get("code") if isinstance(error, dict) else None
            logger.warning("%s in-band error from %s: %s", self.name, model, error)
            raise ProviderError(model, f"upstream provider error (code {code})", retryable=True)

        choices = getattr(completion, "choices", None)
        if not choices:
            raise ProviderError(model, "no choices in response", retryable=True)
        choice = choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason == "length":
            raise ProviderError(model, "response truncated at max_tokens")
        if finish_reason == "content_filter":
            return ChatOutcome(model, "", finish_reason, refused=True, json_mode=(mode == "json"))
        content = getattr(getattr(choice, "message", None), "content", None) or ""
        if not content.strip():
            raise ProviderError(model, "empty response", retryable=True)
        return ChatOutcome(model, content, finish_reason, json_mode=(mode == "json"))


# ----------------------------------------------------------------------------- factory


def reasoning_body(choice: str) -> dict[str, Any] | None:
    """OpenRouter's unified ``reasoning`` parameter for ``OPENROUTER_REASONING``, or nothing.

    ``off`` asks the model not to think before answering (``enabled: false``); an effort level
    bounds the thinking. Models that have no reasoning ignore the parameter.
    """
    if choice == "default":
        return None
    if choice == "off":
        return {"reasoning": {"enabled": False}}
    return {"reasoning": {"effort": choice}}


def build_chat(
    settings: Settings,
    *,
    failure_listener: FailureListener | None = None,
    client: object | None = None,
) -> StructuredChat:
    """The provider the settings ask for. Raises ``ConfigurationError`` when its key is missing."""
    if settings.llm_provider == "openrouter":
        headers: dict[str, str] = {}
        if settings.openrouter_site_url:
            headers["HTTP-Referer"] = settings.openrouter_site_url
        if settings.openrouter_app_name:
            headers["X-Title"] = settings.openrouter_app_name
        return OpenAICompatibleChain(
            provider="openrouter",
            api_key=settings.secret("openrouter_api_key"),
            base_url=settings.openrouter_base_url,
            models=lambda: settings.model_chain,  # the chooser may change it between requests
            timeout_s=settings.openrouter_timeout_s,
            requests_per_minute=settings.openrouter_requests_per_minute,
            strict_schema=settings.openrouter_strict_json_schema,
            free_only=settings.openrouter_free_only,
            headers=headers,
            client=client,
            failure_listener=failure_listener,
            extra_body=reasoning_body(settings.openrouter_reasoning),
        )
    if settings.llm_provider == "openai_compatible":
        return OpenAICompatibleChain(
            provider="openai_compatible",
            api_key=settings.secret("openai_api_key"),
            base_url=settings.openai_base_url,
            models=[settings.openai_model],
            timeout_s=settings.openrouter_timeout_s,
            strict_schema=True,
            client=client,
        )
    from app.services.anthropic_client import AnthropicChat

    return AnthropicChat(settings, client=client)
