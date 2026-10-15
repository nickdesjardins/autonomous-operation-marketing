"""Settings: what blank and pasted values mean, the cross-field rules, and the example file.

Nothing here reads the environment: every object is built with ``_env_file=None`` and a
prefix no variable carries, so the host's ``.env`` (which holds real keys) never takes part.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from app import config as config_module
from app.config import (
    DEFAULT_DATABASE_URL,
    DEFAULT_MODEL_CHAIN,
    DEFAULT_PUBLISHED_URL,
    Settings,
    get_settings,
    is_free_model,
    normalize_database_url,
    parse_model_chain,
)
from app.models import cache

from .conftest import PROJECT_ROOT, make_settings

ENV_EXAMPLE = PROJECT_ROOT / ".env.example"
COMPOSE = PROJECT_ROOT / "docker-compose.yml"
#: Documented in .env.example for docker-compose, not read by the service.
COMPOSE_ONLY = {"POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_PORT"}


# ------------------------------------------------------------------------ defaults


def test_defaults_describe_the_free_stack() -> None:
    settings = Settings(_env_file=None, _env_prefix="HERMETIC_TEST_")
    assert settings.environment == "development" and settings.demo_mode is False
    assert settings.llm_provider == "openrouter" and settings.openrouter_free_only is True
    assert settings.checkpointer == "postgres"
    assert settings.database_url == DEFAULT_DATABASE_URL
    assert settings.webhook_target == "dry_run" and settings.webhook_configured is True
    assert settings.max_revisions == 3 and settings.demo_max_revisions == 2
    assert settings.model_published_url == DEFAULT_PUBLISHED_URL
    assert settings.llm_configured is False  # no key anywhere
    assert settings.docs_enabled is True and settings.is_production is False
    assert settings.uses_database is True
    assert DEFAULT_PUBLISHED_URL.startswith("https://")
    assert all(is_free_model(model) for model in parse_model_chain(DEFAULT_MODEL_CHAIN))


def test_get_settings_is_cached_per_process(monkeypatch: pytest.MonkeyPatch) -> None:
    # The real constructor would read the project's .env; a factory that reads nothing stands in.
    monkeypatch.setattr(config_module, "Settings", lambda: make_settings())
    get_settings.cache_clear()
    try:
        first = get_settings()
        assert first is get_settings() and first.environment == "test"
    finally:
        get_settings.cache_clear()


# --------------------------------------------------------------- blank and pasted values


@pytest.mark.parametrize("raw", ["", "  ", "none", "null", "None"])
def test_empty_strings_mean_unset(raw: str) -> None:
    settings = make_settings(
        api_key=raw,
        webhook_secret=raw,
        webhook_url=raw,
        openrouter_api_key=raw,
        anthropic_api_key=raw,
        synthesis_temperature=raw,
        audit_temperature=raw,
        public_base_url=raw,
        client_ip_header=raw,
        model_published_url=raw,
    )
    assert settings.api_key is None and settings.webhook_secret is None
    assert settings.webhook_url is None and settings.public_base_url is None
    assert settings.openrouter_api_key is None and settings.anthropic_api_key is None
    assert settings.synthesis_temperature is None and settings.audit_temperature is None
    assert settings.client_ip_header is None and settings.model_published_url is None


@pytest.mark.parametrize(
    "raw", ["your-openrouter-api-key", "sk-or-v1-...", "changeme", "REPLACE-ME", "todo", "xxx"]
)
def test_placeholder_keys_are_unset(raw: str) -> None:
    settings = make_settings(openrouter_api_key=raw, openai_api_key=raw, anthropic_api_key=raw)
    assert settings.openrouter_api_key is None
    assert settings.openai_api_key is None and settings.anthropic_api_key is None
    assert settings.llm_configured is False


def test_a_key_pasted_with_its_name_or_quotes_is_cleaned() -> None:
    settings = make_settings(openai_api_key='OPENAI_API_KEY="sk-test-123"')
    assert settings.secret("openai_api_key") == "sk-test-123"
    assert make_settings(openrouter_api_key="'sk-or-x'").secret("openrouter_api_key") == "sk-or-x"
    assert isinstance(settings.openai_api_key, SecretStr)
    assert "sk-test-123" not in repr(settings)  # SecretStr never shows in a dump


@pytest.mark.parametrize(
    "raw",
    ["postgresql://ops:hunter2@db.example/ops", "https://openrouter.ai/keys", "sk-or two words"],
)
def test_a_key_that_looks_like_a_url_or_a_sentence_is_refused_without_echo(raw: str) -> None:
    with pytest.raises(ValidationError) as info:
        make_settings(openrouter_api_key=raw)
    text = str(info.value)
    assert "does not look like a key" in text
    assert "hunter2" not in text and raw not in text  # hide_input_in_errors


def test_api_key_is_inbound_auth_not_a_provider_key() -> None:
    settings = make_settings(
        api_key="shared-secret", openai_api_key=None, llm_provider="openrouter"
    )
    assert settings.api_key is not None
    assert settings.openrouter_configured is False and settings.llm_configured is False


# ------------------------------------------------------------------------ database URL


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("postgresql://u:p@h/db", "postgresql://u:p@h/db"),
        ("postgres://u:p@h/db", "postgresql://u:p@h/db"),
        ('"postgresql://u:p@h/db"', "postgresql://u:p@h/db"),
        ("'postgres://u:p@h/db'", "postgresql://u:p@h/db"),
        ("psql 'postgresql://u:p@h/db?sslmode=require'", "postgresql://u:p@h/db?sslmode=require"),
        ("DATABASE_URL=postgres://u:p@h/db", "postgresql://u:p@h/db"),
        ("database_url_unpooled=postgres://u:p@h/db", "postgresql://u:p@h/db"),
        (
            "  postgresql://u:p@ep-x-pooler.neon.tech/db?sslmode=require&channel_binding=require ",
            "postgresql://u:p@ep-x-pooler.neon.tech/db?sslmode=require&channel_binding=require",
        ),
        ("POSTGRESQL://u:p@h/db", "postgresql://u:p@h/db"),
    ],
)
def test_database_url_normalization(raw: str, expected: str) -> None:
    assert normalize_database_url(raw) == expected
    assert make_settings(database_url=raw).database_url == expected


@pytest.mark.parametrize("raw", ["mysql://u:p@h/db", "u:p@h/db", "postgresql+asyncpg://u:p@h/db"])
def test_a_database_url_of_another_kind_is_refused_without_echo(raw: str) -> None:
    with pytest.raises(ValueError, match="postgresql://") as info:
        normalize_database_url(raw)
    assert raw not in str(info.value)
    with pytest.raises(ValidationError) as settings_error:
        make_settings(database_url=raw)
    assert "u:p@h" not in str(settings_error.value)


def test_blank_database_url_falls_back_to_the_compose_default_and_unpooled_wins() -> None:
    assert make_settings(database_url="  ").database_url == DEFAULT_DATABASE_URL
    assert make_settings(database_url_unpooled="").database_url_unpooled is None
    settings = make_settings(
        database_url="postgres://u:p@ep-x-pooler.neon.tech/db",
        database_url_unpooled="postgres://u:p@ep-x.neon.tech/db",
    )
    assert settings.effective_database_url == "postgresql://u:p@ep-x.neon.tech/db"
    assert settings.database_is_pooled is False
    pooled = make_settings(database_url="postgres://u:p@ep-x-pooler.neon.tech/db")
    assert pooled.database_is_pooled is True
    with pytest.raises(ValidationError, match="DB_POOL_MIN_SIZE"):
        make_settings(db_pool_min_size=6, db_pool_max_size=5)


# --------------------------------------------------------------------------- models


def test_free_only_guard_refuses_a_paid_model_in_the_chain() -> None:
    with pytest.raises(ValidationError, match="':free'") as info:
        make_settings(openrouter_models="vendor/alpha:free, vendor/paid-model")
    assert "vendor/paid-model" in str(info.value)
    lenient = make_settings(openrouter_models="vendor/paid-model", openrouter_free_only=False)
    assert lenient.model_chain == ["vendor/paid-model"]


def test_parse_model_chain_trims_and_dedupes() -> None:
    assert parse_model_chain(" a:free , b:free,a:free,, ") == ["a:free", "b:free"]
    assert parse_model_chain("") == []
    assert is_free_model("vendor/x:free") and not is_free_model("vendor/x")


def test_model_chain_pinned_versus_automatic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cache, "PATH", tmp_path / "model_cache.json")
    cache.forget()

    automatic = make_settings(openrouter_models="")
    assert automatic.models_pinned is False
    assert automatic.model_chain == parse_model_chain(DEFAULT_MODEL_CHAIN)  # nothing chosen yet

    cache.use(["vendor/chosen:free"], {"source": "test"})
    assert (tmp_path / "model_cache.json").exists()
    assert automatic.model_chain == ["vendor/chosen:free"]  # the same object follows the cache

    pinned = make_settings(openrouter_models="vendor/alpha:free,vendor/beta:free")
    assert pinned.models_pinned is True
    assert pinned.model_chain == ["vendor/alpha:free", "vendor/beta:free"]  # the cache is ignored

    cache.forget()
    assert automatic.model_chain == ["vendor/chosen:free"]  # read back from the file
    (tmp_path / "model_cache.json").unlink()
    cache.forget()
    assert automatic.model_chain == parse_model_chain(DEFAULT_MODEL_CHAIN)


def test_published_url_must_be_https_or_null() -> None:
    assert make_settings(model_published_url="null").model_published_url is None
    assert make_settings(model_published_url="https://x.example/chain.json").model_published_url
    with pytest.raises(ValidationError, match="https"):
        make_settings(model_published_url="http://x.example/chain.json")


def test_llm_configured_follows_the_provider() -> None:
    assert make_settings().llm_configured is True  # openai_compatible with the fake key
    assert make_settings(openai_api_key=None).llm_configured is False
    assert make_settings(llm_provider="openrouter").llm_configured is False
    assert make_settings(llm_provider="openrouter", openrouter_api_key="k").llm_configured is True
    assert make_settings(llm_provider="anthropic").llm_configured is False
    assert make_settings(llm_provider="anthropic", anthropic_api_key="k").llm_configured is True
    with pytest.raises(ValidationError):
        make_settings(llm_provider="cohere")


# ---------------------------------------------------------------------------- webhook


def test_demo_mode_refuses_an_external_webhook_target() -> None:
    with pytest.raises(ValidationError, match="DEMO_MODE"):
        make_settings(
            demo_mode=True,
            webhook_target="url",
            webhook_url="https://x.example/in",
            webhook_secret="s",
        )
    assert (
        make_settings(demo_mode=True, webhook_target="self", webhook_secret="s").demo_mode is True
    )
    assert make_settings(demo_mode=True, webhook_target="dry_run").webhook_target == "dry_run"


def test_url_target_needs_an_https_url_and_a_secret() -> None:
    with pytest.raises(ValidationError, match="WEBHOOK_URL"):
        make_settings(webhook_target="url", webhook_secret="s")
    with pytest.raises(ValidationError, match="https://"):
        make_settings(webhook_target="url", webhook_url="ftp://x.example/in", webhook_secret="s")
    with pytest.raises(ValidationError, match="WEBHOOK_SECRET"):
        make_settings(webhook_target="url", webhook_url="https://x.example/in")
    with pytest.raises(ValidationError, match="WEBHOOK_SECRET"):
        make_settings(webhook_target="self", public_base_url="https://demo.example")
    ok = make_settings(
        webhook_target="url", webhook_url="http://127.0.0.1:9/in", webhook_secret="s"
    )
    assert ok.effective_webhook_url == "http://127.0.0.1:9/in"


def test_effective_webhook_url_for_the_three_targets() -> None:
    assert make_settings(webhook_target="dry_run").effective_webhook_url is None
    assert make_settings(webhook_target="dry_run").webhook_configured is True

    external = make_settings(
        webhook_target="url", webhook_url="https://x.example/in", webhook_secret="s"
    )
    assert external.effective_webhook_url == "https://x.example/in" and external.webhook_configured

    own = make_settings(
        webhook_target="self", webhook_secret="s", public_base_url="https://demo.example/"
    )
    assert own.base_url == "https://demo.example"
    assert own.effective_webhook_url == "https://demo.example/api/v1/webhooks/receiver"
    rendered = make_settings(
        webhook_target="self", webhook_secret="s", render_external_url="https://x.onrender.com"
    )
    assert rendered.effective_webhook_url == "https://x.onrender.com/api/v1/webhooks/receiver"
    nowhere = make_settings(webhook_target="self", webhook_secret="s")
    assert nowhere.effective_webhook_url is None and nowhere.webhook_configured is False


# ------------------------------------------------------------------------- the rest


def test_log_level_is_case_insensitive_but_validated() -> None:
    assert make_settings(log_level="debug").log_level == "DEBUG"
    assert make_settings(log_level=" warning ").log_level == "WARNING"
    with pytest.raises(ValidationError, match="Unknown log level"):
        make_settings(log_level="verbose")


def test_cors_origins_and_docs() -> None:
    assert make_settings().cors_origins == []
    assert make_settings(
        cors_allow_origins=" https://a.example ,, https://b.example "
    ).cors_origins == [
        "https://a.example",
        "https://b.example",
    ]
    assert make_settings(environment="production").docs_enabled is False
    assert make_settings(environment="production").is_production is True


def test_bounded_numbers_are_enforced() -> None:
    with pytest.raises(ValidationError):
        make_settings(max_revisions=6)
    with pytest.raises(ValidationError):
        make_settings(synthesis_temperature=3)
    with pytest.raises(ValidationError):
        make_settings(max_json_body_bytes=10)
    with pytest.raises(ValidationError):
        make_settings(webhook_max_attempts=0)
    assert make_settings(synthesis_temperature="0.7").synthesis_temperature == 0.7


def test_secret_helper_reads_plain_text_or_none() -> None:
    settings = make_settings(webhook_secret="s3cret")
    assert settings.secret("webhook_secret") == "s3cret"
    assert settings.secret("anthropic_api_key") is None
    assert settings.secret("service_name") == settings.service_name  # a plain field passes through


# ------------------------------------------------------------------------ .env.example


def _documented() -> dict[str, str]:
    names: dict[str, str] = {}
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^#?\s*([A-Z][A-Z0-9_]*)=(.*)$", line)
        if match:
            names[match.group(1)] = match.group(2).strip()
    return names


def test_env_example_documents_only_settings_fields() -> None:
    documented = _documented()
    fields = {name.upper() for name in Settings.model_fields}
    assert len(documented) > 40
    unknown = set(documented) - fields - COMPOSE_ONLY
    assert unknown == set(), f"not a Settings field: {sorted(unknown)}"
    compose = COMPOSE.read_text(encoding="utf-8")
    assert all(name in compose for name in COMPOSE_ONLY)
    important = {
        "OPENROUTER_API_KEY",
        "DATABASE_URL",
        "WEBHOOK_TARGET",
        "DEMO_MODE",
        "CHECKPOINTER",
    }
    assert important <= set(documented)


def test_env_example_holds_no_secret() -> None:
    documented = _documented()
    for name, value in documented.items():
        if name.endswith(("_KEY", "_SECRET")):
            assert value == "", f"{name} must be blank in .env.example"
    assert documented["POSTGRES_PASSWORD"] == "ops"  # the local compose default, not a secret
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    assert not re.search(r"sk-(?:or-v1|ant|proj)-[A-Za-z0-9_-]{16,}", text)
    assert "\u2014" not in text


def test_env_example_is_a_valid_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ENVIRONMENT", raising=False)  # conftest sets it; the file must speak
    settings = Settings(_env_file=str(ENV_EXAMPLE))
    assert settings.environment == "development"
    assert settings.llm_provider == "openrouter" and settings.llm_configured is False
    assert settings.webhook_target == "dry_run" and settings.webhook_secret is None
    assert settings.database_url == DEFAULT_DATABASE_URL
    assert settings.model_published_url == DEFAULT_PUBLISHED_URL  # the commented line stays off
