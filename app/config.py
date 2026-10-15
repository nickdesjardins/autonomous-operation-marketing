"""Environment-driven configuration (pydantic-settings).

Every tunable lives here, so the graph, the services and the API never hard-code a number or
an address. Values come from the process environment first, then from ``.env`` and
``.env.local`` (both git-ignored; ``.env.example`` documents every name).

Three ways to run the stack:

* **Zero cost (the default):** ``LLM_PROVIDER=openrouter`` with an ordered chain of ``:free``
  models, PostgreSQL in Docker or on Neon, the webhook delivered to the service's own receiver.
* **Paid, drop-in:** ``LLM_PROVIDER=anthropic`` (Claude through the official SDK) or
  ``LLM_PROVIDER=openai_compatible`` (OpenAI, Gemini's OpenAI-compatible endpoint, Groq, any
  server that speaks the Chat Completions API).
* **No database at all:** ``CHECKPOINTER=memory`` keeps checkpoints and inventory in memory,
  for a first look on a laptop. A restart forgets every thread.

Free models are withdrawn often. With ``OPENROUTER_MODELS`` left blank the service looks after
its chain itself (see ``app/models``); naming models pins the chain.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["development", "test", "production"]
LlmProvider = Literal["openrouter", "openai_compatible", "anthropic"]
WebhookTarget = Literal["url", "self", "dry_run"]
CheckpointerKind = Literal["postgres", "memory"]

#: Ordered default chain of free OpenRouter models: a dated snapshot, not a ranking. Both models
#: were measured through this service's own three cases on 15 October 2026, three passes each
#: (benchmarks/2026-10-15, README "Model selection"): liquid/lfm-2.5-2.6b answered 9 of 9 cases
#: right at a 9.1 s median, nvidia/nemotron-3-super-120b-a12b 8 of 9 at 9.7 s. The first place
#: goes to the model with the fewest wrong answers; the chain moves on only when a call fails,
#: never on a wrong answer. The two sit on different upstream providers (Liquid, NVIDIA), so one
#: outage cannot take out both.
DEFAULT_MODEL_CHAIN = "liquid/lfm-2.5-2.6b:free,nvidia/nemotron-3-super-120b-a12b:free"

#: Where this project publishes the chain its scheduled test chose: one small JSON file on a
#: branch of its own, written by ``.github/workflows/model-check.yml``. A clone reads the same
#: file until it is pointed elsewhere. If the repository is renamed, change this at once.
DEFAULT_PUBLISHED_URL = (
    "https://raw.githubusercontent.com/nickdesjardins/autonomous-operation-marketing/"
    "tested-models/chain.json"
)

#: Matches docker-compose.yml.
DEFAULT_DATABASE_URL = "postgresql://ops:ops@localhost:5432/ops"

#: Placeholder values a template may ship. Treated as "not configured".
_PLACEHOLDER_KEYS: frozenset[str] = frozenset(
    {"your-openrouter-api-key", "sk-or-v1-...", "changeme", "replace-me", "todo", "xxx"}
)

#: What often comes along when a value is copied from a dashboard or a .env file.
_PASTED_PREFIXES = ("psql ", "database_url_unpooled=", "database_url=")


def parse_model_chain(raw: str) -> list[str]:
    """Split an ordered, comma-separated chain of model ids (trimmed, de-duplicated, non-empty)."""
    models: list[str] = []
    for item in raw.split(","):
        model = item.strip()
        if model and model not in models:
            models.append(model)
    return models


def is_free_model(model_id: str) -> bool:
    """True for an OpenRouter id that cannot cost money (the ``:free`` suffix)."""
    return model_id.endswith(":free")


def _strip_pasted(value: str) -> str:
    """``psql 'postgresql://...'``, ``"postgresql://..."`` or ``DATABASE_URL=...`` -> the URL."""
    text = value.strip()
    for prefix in _PASTED_PREFIXES:
        if text.lower().startswith(prefix):
            text = text[len(prefix) :].strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    return text


def normalize_database_url(url: str) -> str:
    """Make a hosted-Postgres URL (Neon, Render, Heroku) usable by psycopg.

    * Quotes, a ``psql`` prefix or a ``DATABASE_URL=`` prefix copied along are removed.
    * ``postgres://`` becomes ``postgresql://`` (libpq accepts both; one spelling is simpler).
    * libpq parameters such as ``sslmode`` and ``channel_binding`` are kept: psycopg understands
      them, unlike asyncpg.

    Error messages never include the URL: it carries the database password.
    """
    parts = urlsplit(_strip_pasted(url))
    scheme = parts.scheme.lower()
    if scheme in {"postgres", "postgresql"}:
        scheme = "postgresql"
    else:
        raise ValueError(
            "DATABASE_URL must be a PostgreSQL connection string starting with postgresql:// "
            "(paste the string itself, without quotes)"
        )
    query = [(key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True)]
    return urlunsplit((scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


class Settings(BaseSettings):
    """Runtime configuration, resolved once per process (see :func:`get_settings`)."""

    model_config = SettingsConfigDict(
        # Both are read; values in .env.local override .env. Neither is committed.
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        env_ignore_empty=True,
        env_parse_none_str="null",
        # A rejected value (a database URL, a key) must never be echoed into a start-up log.
        hide_input_in_errors=True,
        # Older pydantic-settings releases reserve every name that starts with "model_".
        protected_namespaces=("settings_",),
    )

    # ------------------------------------------------------------------ application
    service_name: str = Field(
        default="multi-agent-ops-pipeline",
        description="Name reported by /health, shown on the playground, written in the log.",
    )
    environment: Environment = Field(
        default="development",
        description="production hides /docs and makes the page's POST routes stricter.",
    )
    log_level: str = Field(default="INFO", description="DEBUG, INFO, WARNING, ERROR, CRITICAL.")
    api_key: SecretStr | None = Field(
        default=None,
        description=(
            "Inbound shared secret for THIS service, sent as X-API-Key. Outside demo mode it "
            "protects every /api/v1/workflows route. In demo mode those routes are open to "
            "visitors (the budget protects them) and the key is not needed."
        ),
    )
    cors_allow_origins: str = Field(
        default="",
        description=(
            "Comma-separated origins allowed to call the API from a browser (the GitHub Pages "
            "site). Empty disables CORS headers; * allows any origin."
        ),
    )
    playground_enabled: bool = Field(
        default=True, description="Serve the browser playground at / and /playground/*."
    )
    max_json_body_bytes: int = Field(
        default=32 * 1024,
        ge=1024,
        description="Largest request body accepted on any POST (checked before parsing).",
    )

    # --------------------------------------------------------------- public showcase
    demo_mode: bool = Field(
        default=False,
        description=(
            "Public showcase: runs are open but budgeted per visitor and per day, revisions "
            "are capped, and the webhook can only reach this service's own receiver."
        ),
    )
    demo_max_revisions: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Demo mode: the largest max_revisions a visitor's request may ask for.",
    )
    trust_proxy_headers: bool = Field(
        default=False,
        description=(
            "Behind a reverse proxy (Render), read the visitor address from CLIENT_IP_HEADER or "
            "X-Forwarded-For. Leave off when clients reach the app directly."
        ),
    )
    client_ip_header: str | None = Field(
        default=None,
        description="Header the edge proxy overwrites with the visitor address (CF-Connecting-IP).",
    )
    trusted_proxy_hops: int = Field(default=1, ge=1, le=10)
    run_rate_limit_per_minute: int = Field(
        default=2, ge=1, description="Pipeline runs per visitor address per minute (demo mode)."
    )
    run_daily_limit_per_client: int = Field(
        default=12, ge=1, description="Pipeline runs per visitor address per UTC day (demo mode)."
    )
    run_global_limit_per_minute: int = Field(
        default=6, ge=1, description="Pipeline runs per minute for everyone together (demo mode)."
    )
    run_daily_limit: int = Field(
        default=120,
        ge=1,
        description="Pipeline runs per UTC day for everyone together (demo mode).",
    )
    run_max_inflight: int = Field(
        default=2,
        ge=1,
        le=32,
        description="Demo mode: runs executed at once; more get 503 with Retry-After.",
    )
    openrouter_allowance_check: bool = Field(
        default=True,
        description="Ask OpenRouter (GET /key) how many free requests are left today, once a minute.",
    )
    openrouter_allowance_reserve: int = Field(
        default=100,
        ge=0,
        description="Refuse public runs when fewer free requests than this remain today.",
    )

    # --------------------------------------------------------------------- database
    checkpointer: CheckpointerKind = Field(
        default="postgres",
        description=(
            "postgres: checkpoints and inventory in PostgreSQL (DATABASE_URL). memory: no "
            "database at all, everything forgotten at restart (laptop mode and tests)."
        ),
    )
    database_url: str = Field(
        default=DEFAULT_DATABASE_URL,
        description=(
            "psycopg connection string. Neon/Render-style postgresql://...?sslmode=require "
            "URLs are accepted as they are."
        ),
    )
    database_url_unpooled: str | None = Field(
        default=None,
        description=(
            "Direct (non-PgBouncer) URL as Neon provides it (DATABASE_URL_UNPOOLED). Preferred "
            "over DATABASE_URL when set: the app runs its own connection pool."
        ),
    )
    db_pool_min_size: int = Field(default=1, ge=0, le=20)
    db_pool_max_size: int = Field(default=5, ge=1, le=50)
    db_connect_timeout_s: float = Field(default=10.0, gt=0, le=120)
    db_startup_retries: int = Field(
        default=3,
        ge=0,
        le=20,
        description="Retries of the start-up schema check while a serverless database wakes up.",
    )
    health_timeout_s: float = Field(default=5.0, gt=0, le=60)
    auto_create_schema: bool = Field(
        default=True,
        description="Create the inventory tables, seed the sample SKUs and set up the checkpointer.",
    )

    # -------------------------------------------------------------------- pipeline
    max_revisions: int = Field(
        default=3,
        ge=0,
        le=5,
        description="Default cap on self-correction loops when a request does not name one.",
    )
    synthesis_temperature: float | None = Field(
        default=0.2, ge=0.0, le=2.0, description="Copy synthesis model (creative but bounded)."
    )
    audit_temperature: float | None = Field(
        default=0.0, ge=0.0, le=2.0, description="Compliance auditor (deterministic)."
    )
    synthesis_max_tokens: int = Field(
        default=4096,
        ge=256,
        le=16_000,
        description=(
            "Output cap of a synthesis call. Models that reason before answering spend part of "
            "it on that; see OPENROUTER_REASONING."
        ),
    )
    audit_max_tokens: int = Field(
        default=4096,
        ge=256,
        le=16_000,
        description=(
            "Output cap of an audit call. liquid/lfm-2.5-2.6b cannot switch its reasoning off "
            "and spends about 2000 tokens thinking before a 300-token verdict."
        ),
    )

    # ------------------------------------------------------------------- providers
    llm_provider: LlmProvider = Field(
        default="openrouter",
        description="openrouter (free chain) | openai_compatible | anthropic.",
    )
    openrouter_api_key: SecretStr | None = Field(
        default=None, description="https://openrouter.ai/keys - the only key the free stack needs."
    )
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_models: str = Field(
        default="",
        description=(
            "Ordered, comma-separated chain. Left blank, the service starts from the tested "
            "default and looks after the chain itself. Naming models pins the chain."
        ),
    )
    openrouter_strict_json_schema: bool = Field(
        default=True,
        description=(
            "Request strict json_schema output first. A model that rejects it is retried in plain "
            "JSON mode with the schema spelled out in the system prompt."
        ),
    )
    openrouter_free_only: bool = Field(
        default=True,
        description="Refuse any chain entry without the :free suffix so a typo can never spend money.",
    )
    openrouter_timeout_s: float = Field(default=60.0, gt=0, le=600)
    openrouter_requests_per_minute: int = Field(
        default=20, ge=0, le=600, description="Pacing for free models (OpenRouter allows 20/min)."
    )
    openrouter_reasoning: Literal["default", "off", "minimal", "low", "medium", "high"] = Field(
        default="off",
        description=(
            "OpenRouter's unified reasoning control, sent to every model of the chain. Models "
            "that think before they answer spend the output budget on that thinking and can "
            "run out of it before the JSON (measured on 15 October 2026: "
            "nvidia/nemotron-3-super-120b-a12b truncated every answer at 2048 tokens with "
            "reasoning left on, and answered in 13 s with it off). off disables reasoning "
            "where the model allows it; a model whose endpoint refuses the switch is called "
            "without it. default sends nothing."
        ),
    )
    openrouter_site_url: str | None = Field(default=None, description="HTTP-Referer attribution.")
    openrouter_app_name: str | None = Field(
        default="multi-agent-ops-pipeline", description="X-Title attribution header."
    )
    model_published_url: str | None = Field(
        default=DEFAULT_PUBLISHED_URL,
        description=(
            "Where the service reads the chain the scheduled test published (https, no key). "
            "Read at startup and after a failed request, only with OPENROUTER_MODELS blank. "
            "null = never read it."
        ),
    )
    model_catalogue_check: bool = Field(
        default=True,
        description=(
            "At startup, read the published chain and ask OpenRouter's public catalogue whether "
            "each model is still listed. false = no request at startup (offline work)."
        ),
    )
    openai_api_key: SecretStr | None = Field(
        default=None, description="For LLM_PROVIDER=openai_compatible."
    )
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Any OpenAI-compatible Chat Completions endpoint (OpenAI, Gemini, Groq).",
    )
    openai_model: str = Field(default="gpt-4o-mini", description="Model id at OPENAI_BASE_URL.")
    anthropic_api_key: SecretStr | None = Field(
        default=None, description="For LLM_PROVIDER=anthropic (official SDK)."
    )
    anthropic_model: str = Field(default="claude-opus-5-5")
    anthropic_effort: Literal["low", "medium", "high", "xhigh", "max"] = Field(
        default="low",
        description="Claude 5.x reasoning effort. Low keeps the two tasks fast and cheap.",
    )

    # --------------------------------------------------------------------- webhook
    webhook_target: WebhookTarget = Field(
        default="dry_run",
        description=(
            "url: POST to WEBHOOK_URL. self: POST to this service's own receiver at "
            "PUBLIC_BASE_URL (the public demo). dry_run: record the delivery and send nothing "
            "(signed when WEBHOOK_SECRET is set, marked unsigned otherwise)."
        ),
    )
    webhook_url: str | None = Field(default=None, description="The external endpoint (https).")
    webhook_secret: SecretStr | None = Field(
        default=None,
        description="HMAC-SHA256 key for the X-Signature-256 header. Required for url and self.",
    )
    webhook_timeout_s: float = Field(default=10.0, gt=0, le=120)
    webhook_max_attempts: int = Field(default=3, ge=1, le=10)
    public_base_url: str | None = Field(
        default=None,
        description="This service's public address (Render fills RENDER_EXTERNAL_URL).",
    )
    render_external_url: str | None = Field(
        default=None, description="Set by Render. Used when PUBLIC_BASE_URL is empty."
    )

    # ------------------------------------------------------------------- validators
    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_case_log_level(cls, value: object) -> object:
        level = str(value).strip().upper() if isinstance(value, str) else value
        if isinstance(level, str) and level not in logging.getLevelNamesMapping():
            raise ValueError(f"Unknown log level {level!r}")
        return level

    @field_validator("database_url", mode="before")
    @classmethod
    def _normalize_database_url(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return DEFAULT_DATABASE_URL
        return normalize_database_url(value) if isinstance(value, str) else value

    @field_validator("database_url_unpooled", mode="before")
    @classmethod
    def _normalize_unpooled_url(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return normalize_database_url(value) if isinstance(value, str) else value

    @field_validator("openrouter_api_key", "openai_api_key", "anthropic_api_key", mode="before")
    @classmethod
    def _clean_provider_key(cls, value: object) -> object:
        """Strip what is copied along with a key, and refuse a value that is clearly not one.

        Pasting the database string into a key box would send the database password to a
        provider as a bearer token on every request, so start-up refuses it (without echoing
        it: hide_input_in_errors).
        """
        if not isinstance(value, str):
            return value
        text = value.strip()
        for prefix in ("openrouter_api_key=", "openai_api_key=", "anthropic_api_key="):
            if text.lower().startswith(prefix):
                text = text[len(prefix) :].strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
            text = text[1:-1].strip()
        if text.lower() in {"", "none", "null"} or text.lower() in _PLACEHOLDER_KEYS:
            return None
        if "://" in text or any(ch.isspace() for ch in text):
            raise ValueError(
                "the API key does not look like a key (was another line pasted along with it?)"
            )
        return text

    @field_validator(
        "api_key",
        "webhook_secret",
        "webhook_url",
        "openrouter_site_url",
        "client_ip_header",
        "public_base_url",
        "render_external_url",
        "synthesis_temperature",
        "audit_temperature",
        "model_published_url",
        mode="before",
    )
    @classmethod
    def _empty_string_is_none(cls, value: object) -> object:
        """``WEBHOOK_URL=`` in .env means "unset", not an empty value."""
        if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
            return None
        return value

    @model_validator(mode="after")
    def _validate_relationships(self) -> Settings:
        chain = parse_model_chain(self.openrouter_models) if self.openrouter_models else []
        if self.openrouter_free_only:
            paid = [m for m in chain if not is_free_model(m)]
            if paid:
                raise ValueError(
                    f"OPENROUTER_MODELS contains non-free models {paid}; every id must end in "
                    "':free' while OPENROUTER_FREE_ONLY=true"
                )
        if self.model_published_url is not None and not self.model_published_url.startswith(
            "https://"
        ):
            raise ValueError("MODEL_PUBLISHED_URL must be an https address, or null")
        if self.webhook_target == "url" and self.demo_mode:
            raise ValueError(
                "DEMO_MODE never dispatches to an external URL: set WEBHOOK_TARGET to self or "
                "dry_run"
            )
        if self.webhook_target == "url":
            if not self.webhook_url:
                raise ValueError("WEBHOOK_TARGET=url needs WEBHOOK_URL")
            if not self.webhook_url.startswith(("https://", "http://")):
                raise ValueError("WEBHOOK_URL must start with https:// (or http:// for a test)")
        if self.webhook_target in ("url", "self") and self.webhook_secret is None:
            raise ValueError(
                "WEBHOOK_SECRET is required to sign deliveries (any long random string)"
            )
        if self.db_pool_min_size > self.db_pool_max_size:
            raise ValueError("DB_POOL_MIN_SIZE must be <= DB_POOL_MAX_SIZE")
        return self

    # --------------------------------------------------------------- derived helpers
    @property
    def models_pinned(self) -> bool:
        """True when ``OPENROUTER_MODELS`` names the chain; False when it was left blank."""
        return bool(self.openrouter_models.strip())

    @property
    def model_chain(self) -> list[str]:
        """The ordered OpenRouter model ids to use.

        Pinned: exactly the named models. Automatic: the chain the service chose for itself
        when it has chosen one (``app.models.cache``), else the tested default. The settings
        object never changes after startup; only what this property looks up does.
        """
        if self.models_pinned:
            return parse_model_chain(self.openrouter_models)
        from app.models import cache  # imported late: app.models imports this module

        return list(cache.chain()) or parse_model_chain(DEFAULT_MODEL_CHAIN)

    @property
    def openrouter_configured(self) -> bool:
        return self.openrouter_api_key is not None

    @property
    def llm_configured(self) -> bool:
        if self.llm_provider == "openrouter":
            return self.openrouter_configured
        if self.llm_provider == "anthropic":
            return self.anthropic_api_key is not None
        return self.openai_api_key is not None

    @property
    def effective_database_url(self) -> str:
        return self.database_url_unpooled or self.database_url

    @property
    def database_is_pooled(self) -> bool:
        """Neon's PgBouncer endpoints carry ``-pooler`` in the host name."""
        return "-pooler" in (urlsplit(self.effective_database_url).hostname or "")

    @property
    def uses_database(self) -> bool:
        return self.checkpointer == "postgres"

    @property
    def docs_enabled(self) -> bool:
        """Interactive API docs are hidden on the public showcase."""
        return self.environment != "production"

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]

    @property
    def base_url(self) -> str | None:
        """This service's public address, for the self-addressed webhook."""
        value = self.public_base_url or self.render_external_url
        return value.rstrip("/") if value else None

    @property
    def effective_webhook_url(self) -> str | None:
        """Where a delivery goes: the external URL, this service's receiver, or nowhere."""
        if self.webhook_target == "url":
            return self.webhook_url
        if self.webhook_target == "self" and self.base_url:
            return f"{self.base_url}/api/v1/webhooks/receiver"
        return None

    @property
    def webhook_configured(self) -> bool:
        return self.webhook_target == "dry_run" or self.effective_webhook_url is not None

    def secret(self, field: str) -> str | None:
        """Return the plain text of a :class:`SecretStr` field, or ``None``."""
        value = getattr(self, field)
        return value.get_secret_value() if isinstance(value, SecretStr) else value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """The process-wide settings singleton (tests call ``get_settings.cache_clear()``)."""
    return Settings()
