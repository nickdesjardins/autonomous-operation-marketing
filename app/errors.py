"""Exception hierarchy shared by the graph, the services and the API layer.

Provider SDK exceptions stay inside the service modules. The API maps these few classes to HTTP
status codes without importing any vendor SDK.
"""

from __future__ import annotations


class PipelineError(Exception):
    """Base class for every error this project raises on purpose."""


class ConfigurationError(PipelineError):
    """A required setting is missing, or two settings contradict each other."""


class InventoryError(PipelineError):
    """The inventory database could not be read."""


class ProviderError(PipelineError):
    """A language model provider failed.

    ``retryable`` separates transient failures (rate limits, 5xx, timeouts, network) from
    permanent ones (bad request, rejected key, every model of the chain gone), so the API can
    answer 503 or 502 accordingly. ``provider`` names the model or vendor for the log only.
    """

    def __init__(self, provider: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(f"{provider}: {message}")
        self.provider = provider
        self.message = message
        self.retryable = retryable


class ProviderAuthError(ProviderError):
    """The provider rejected the API key. No other model can fix that."""

    def __init__(self, provider: str, message: str) -> None:
        super().__init__(provider, message, retryable=False)


class WebhookError(PipelineError):
    """The webhook could not be delivered after every attempt."""
