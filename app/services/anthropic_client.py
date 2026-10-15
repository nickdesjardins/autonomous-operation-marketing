"""Claude through the official Anthropic SDK, with the same structured contract as the chain.

Claude 5.x models reject sampling parameters (``temperature`` is not sent) and are steered with
``output_config.effort`` instead. The schema is enforced by the API itself through
``output_config.format`` (JSON schema structured outputs), and the service still validates the
answer with Pydantic before the graph sees it, exactly as it does for every other provider.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any, TypeVar

from pydantic import BaseModel

from app.config import Settings
from app.errors import ConfigurationError, ProviderAuthError, ProviderError
from app.services.llm_client import Attempt, ChatOutcome, parse_as, strict_json_schema

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class AnthropicChat:
    """One Claude model, called with JSON-schema structured output."""

    name = "anthropic"

    def __init__(self, settings: Settings, *, client: object | None = None) -> None:
        key = settings.secret("anthropic_api_key")
        if client is None and not key:
            raise ConfigurationError("ANTHROPIC_API_KEY is required for the anthropic provider")
        if client is None:
            from anthropic import AsyncAnthropic

            client = AsyncAnthropic(
                api_key=key, timeout=settings.openrouter_timeout_s, max_retries=1
            )
        self._client = client
        self.model = settings.anthropic_model
        self.effort = settings.anthropic_effort

    @property
    def models(self) -> list[str]:
        return [self.model]

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "models": self.models, "effort": self.effort}

    async def complete(
        self,
        *,
        messages: Sequence[dict[str, str]],
        schema_model: type[T],
        temperature: float | None,
        max_tokens: int,
        schema_name: str = "output",
    ) -> ChatOutcome:
        import anthropic

        system = "\n\n".join(m["content"] for m in messages if m.get("role") == "system")
        turns = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if m.get("role") in ("user", "assistant")
        ]
        started = time.perf_counter()
        try:
            response = await self._client.messages.create(  # type: ignore[attr-defined]
                model=self.model,
                max_tokens=max_tokens,
                system=system or anthropic.NOT_GIVEN,
                messages=turns,
                output_config={
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": strict_json_schema(schema_model)},
                },
            )
        except anthropic.AuthenticationError as exc:
            raise ProviderAuthError("anthropic-auth", "the API key was rejected (401)") from exc
        except anthropic.RateLimitError as exc:
            raise ProviderError(self.model, "rate limited (429)", retryable=True) from exc
        except anthropic.NotFoundError as exc:
            raise ProviderError(self.model, "model not found (404)") from exc
        except anthropic.BadRequestError as exc:
            raise ProviderError(self.model, "request rejected (400)") from exc
        except anthropic.APIStatusError as exc:
            raise ProviderError(
                self.model, f"HTTP {exc.status_code}", retryable=exc.status_code >= 500
            ) from exc
        except anthropic.APITimeoutError as exc:
            raise ProviderError(self.model, "request timed out", retryable=True) from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderError(self.model, "connection error", retryable=True) from exc

        elapsed = round(time.perf_counter() - started, 3)
        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason == "refusal":
            return ChatOutcome(
                self.model,
                "",
                stop_reason,
                refused=True,
                attempts=[Attempt(self.model, "refused", elapsed, "safety refusal")],
            )
        if stop_reason == "max_tokens":
            raise ProviderError(self.model, "response truncated at max_tokens")
        text = "".join(
            getattr(block, "text", "")
            for block in getattr(response, "content", [])
            if getattr(block, "type", "") == "text"
        )
        if not text.strip():
            raise ProviderError(self.model, "empty response", retryable=True)
        try:
            parsed = parse_as(schema_model, text)
        except Exception as exc:  # noqa: BLE001 - any validation failure is a provider failure
            raise ProviderError(self.model, f"invalid output ({type(exc).__name__})") from exc
        return ChatOutcome(
            self.model,
            text,
            stop_reason,
            parsed=parsed,
            attempts=[Attempt(self.model, "answered", elapsed)],
        )
