"""Hermetic fixtures: scripted model answers, in-memory inventory and checkpoints, no network.

Three things happen at import time, before any other test module loads, because they have to
come first:

* Every environment variable that ``Settings`` reads is removed from the process, so the host's
  ``.env`` (which holds real keys) and its shell can never leak into a test. The suite runs as
  ``ENVIRONMENT=test``.
* The network is closed by construction. Both HTTP transports of ``httpx`` (and of ``httpx2``,
  which the provider SDKs use) raise as soon as a request would leave the process, and the real
  ``AsyncOpenAI`` and ``AsyncAnthropic`` constructors fail loudly. Tests hand the services fake
  clients instead; a test that forgets does not hang on a socket, it fails at once. The two SDK
  classes are replaced by an import hook the moment their package loads, rather than by
  importing both packages here: together they cost two seconds to import, and the graph, gate,
  API and compliance files never need them.
* The model-chain cache is pointed at a temporary file for every test, so no test can write
  ``model_cache.json`` into the project root.

The rest of the module is the vocabulary the spec files share: ``make_settings`` (one settings
object with nothing read from the environment), ``FakeChat`` (a structured chat whose answers are
scripted per schema), the compliant and non-compliant drafts for SKU NR-MAG-200, and the builders
that compile the real graph topology over an in-memory saver and mount it in the real FastAPI
app.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import logging
import os
import sys
from collections import deque
from collections.abc import AsyncIterator, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from app.config import Settings

# The suite must not depend on whatever the host exports (API keys, LOG_LEVEL, DEMO_MODE, ...).
_SETTINGS_ENV_NAMES = {name.upper() for name in Settings.model_fields}
for _name in list(os.environ):
    if _name.upper() in _SETTINGS_ENV_NAMES:
        os.environ.pop(_name)
os.environ["ENVIRONMENT"] = "test"

import httpx  # noqa: E402
import pytest  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from app.database import InMemoryInventory, InMemoryWebhookStore  # noqa: E402
from app.graph.builder import build_graph  # noqa: E402
from app.graph.nodes import PipelineNodes  # noqa: E402
from app.graph.state import PipelineState, initial_state  # noqa: E402
from app.graph.tools import InventoryTools  # noqa: E402
from app.main import Components, create_app  # noqa: E402
from app.models import cache as model_cache  # noqa: E402
from app.schemas import (  # noqa: E402
    ALL_CHANNELS,
    ChannelCopyDraft,
    ComplianceAuditResult,
    ViolationItem,
    utc_now_iso,
)
from app.services.llm_client import Attempt, ChatOutcome  # noqa: E402
from app.services.webhook_dispatcher import WebhookDispatcher  # noqa: E402

NETWORK_CLOSED = "the test suite is offline: no request may leave the process"


class NetworkClosed(RuntimeError):
    """Raised by anything that would open a socket during the suite."""


def _closed_sync(*_args: object, **_kwargs: object) -> object:
    raise NetworkClosed(NETWORK_CLOSED)


async def _closed_async(*_args: object, **_kwargs: object) -> object:
    raise NetworkClosed(NETWORK_CLOSED)


class ClosedClient:
    """Stands in for the real SDK clients: constructing one is a test bug, so it fails loudly."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise NetworkClosed(NETWORK_CLOSED + " (construct the service with a fake client)")


#: The SDK package and the client class in it that must never be constructed for real.
_SDK_CLIENTS = {"openai": "AsyncOpenAI", "anthropic": "AsyncAnthropic"}


class _SealingLoader(importlib.abc.Loader):
    """Runs the package's own loader, then replaces the client class in the loaded module."""

    def __init__(self, inner: importlib.abc.Loader, attribute: str) -> None:
        self._inner = inner
        self._attribute = attribute

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        create = getattr(self._inner, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module: ModuleType) -> None:
        # The package runs with its real loader on record, exactly as it would otherwise.
        module.__loader__ = self._inner
        if module.__spec__ is not None:
            module.__spec__.loader = self._inner
        self._inner.exec_module(module)
        setattr(module, self._attribute, ClosedClient)


class _SealingFinder(importlib.abc.MetaPathFinder):
    """Finds the two SDK packages where they are and hands them a sealing loader."""

    def find_spec(
        self, fullname: str, path: Any = None, target: ModuleType | None = None
    ) -> importlib.machinery.ModuleSpec | None:
        attribute = _SDK_CLIENTS.get(fullname)
        if attribute is None:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _SealingLoader(spec.loader, attribute)
        return spec


httpx.AsyncHTTPTransport.handle_async_request = _closed_async  # type: ignore[method-assign]
httpx.HTTPTransport.handle_request = _closed_sync  # type: ignore[method-assign]
try:
    import httpx2
except ImportError:  # pragma: no cover - the provider SDKs bring it, but the suite must not care
    httpx2 = None
if httpx2 is not None:
    httpx2.AsyncHTTPTransport.handle_async_request = _closed_async  # type: ignore[method-assign]
    httpx2.HTTPTransport.handle_request = _closed_sync  # type: ignore[method-assign]
for _package, _attribute in _SDK_CLIENTS.items():
    if _package in sys.modules:  # already imported by a plugin: seal it now
        setattr(sys.modules[_package], _attribute, ClosedClient)
sys.meta_path.insert(0, _SealingFinder())

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------- settings


def make_settings(**overrides: Any) -> Settings:
    """A settings object that reads nothing from the environment or from any .env file.

    The defaults are the laptop configuration with a fake key for the one-model provider, so
    ``llm_configured`` is true without any real credential existing anywhere.
    """
    values: dict[str, Any] = {
        "environment": "test",
        "checkpointer": "memory",
        "llm_provider": "openai_compatible",
        "openai_api_key": "test-key",
        "webhook_target": "dry_run",
        "log_level": "WARNING",
    }
    values.update(overrides)
    return Settings(_env_file=None, _env_prefix="HERMETIC_TEST_", **values)


@pytest.fixture
def settings() -> Settings:
    return make_settings()


@pytest.fixture(autouse=True)
def isolated_model_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """No test reads or writes the project's ``model_cache.json``."""
    monkeypatch.setattr(model_cache, "PATH", tmp_path / "model_cache.json")
    model_cache.forget()
    yield
    model_cache.forget()


@pytest.fixture(autouse=True)
def restore_app_logging() -> Iterator[None]:
    """``create_app`` sets the ``app`` logger level for the process; no test keeps that change."""
    app_logger = logging.getLogger("app")
    level = app_logger.level
    yield
    app_logger.setLevel(level)


# ------------------------------------------------------------------- sample drafts (NR-MAG-200)

SKU = "NR-MAG-200"
NPN = "80098765"
MANDATORY_STATEMENTS = (
    "Consult a health care practitioner prior to use if you are pregnant or breastfeeding.",
    "Some people may experience diarrhoea.",
)
CAMPAIGN_ANGLE = "Autumn restock: calm evenings and better sleep routines for busy parents."

#: Passes every deterministic check for NR-MAG-200 (a test in test_compliance_rules holds it to that).
GOOD_DRAFT: dict[str, Any] = {
    "web_title": "Magnesium Bisglycinate 200 mg: gentle daily support for muscles and bones",
    "web_description_html": (
        "<p><strong>Magnesium Bisglycinate 200 mg</strong> gives you 200 mg of magnesium per "
        "capsule in the chelated bisglycinate form, chosen for tolerability.</p>"
        "<p>Helps to maintain proper muscle function, including the heart muscle. Helps in the "
        "development and maintenance of bones and teeth. Helps to prevent magnesium deficiency. "
        "Helps in energy metabolism and in tissue formation.</p>"
        "<ul><li>Vegetable capsule: vegan, gluten free, non-GMO.</li>"
        "<li>Take 1 capsule, 1 to 2 times daily.</li></ul>"
        "<p>NPN 80098765</p>"
        "<p>Consult a health care practitioner prior to use if you are pregnant or "
        "breastfeeding. Some people may experience diarrhoea.</p>"
    ),
    "marketplace_bullets": [
        "200 mg of magnesium per capsule, as magnesium bisglycinate (chelated for tolerability)",
        "Helps to maintain proper muscle function, including the heart muscle",
        "Helps in the development and maintenance of bones and teeth",
        "Helps to prevent magnesium deficiency and helps in energy metabolism",
        "Vegetable capsule: vegan, gluten free, non-GMO; take 1 capsule, 1 to 2 times daily",
    ],
    "social_post_copy": (
        "Calm evenings start with a steady routine. Magnesium Bisglycinate 200 mg, one capsule "
        "with dinner, helps to maintain proper muscle function and helps in the maintenance of "
        "bones and teeth. NPN 80098765 #magnesium #calmevenings"
    ),
    "image_generation_prompt": (
        "A matte white bottle of magnesium bisglycinate capsules on a wooden bedside table at "
        "dusk, warm lamplight, a folded linen blanket and a glass of water beside it. Calm, "
        "quiet evening mood with a shallow depth of field."
    ),
    "disclaimers_included": list(MANDATORY_STATEMENTS),
    "npn_displayed": "NPN 80098765",
}

#: Carries a cure claim in two fields and shows the NPN nowhere; everything else is compliant.
BAD_DRAFT: dict[str, Any] = {
    "web_title": "Magnesium Bisglycinate 200 mg: an evening ritual",
    "web_description_html": (
        "<p>Magnesium Bisglycinate 200 mg cures chronic insomnia and helps to maintain proper "
        "muscle function, including the heart muscle.</p>"
        "<p>Take 1 capsule, 1 to 2 times daily.</p>"
    ),
    "marketplace_bullets": GOOD_DRAFT["marketplace_bullets"],
    "social_post_copy": (
        "Sleep like a baby: Magnesium Bisglycinate 200 mg cures chronic insomnia. #magnesium"
    ),
    "image_generation_prompt": GOOD_DRAFT["image_generation_prompt"],
    "disclaimers_included": list(MANDATORY_STATEMENTS),
    "npn_displayed": "not shown",
}


def good_draft(**changes: Any) -> ChannelCopyDraft:
    """A fresh compliant draft, with any field replaced."""
    return ChannelCopyDraft.model_validate({**GOOD_DRAFT, **changes})


def bad_draft(**changes: Any) -> ChannelCopyDraft:
    """A fresh non-compliant draft, with any field replaced."""
    return ChannelCopyDraft.model_validate({**BAD_DRAFT, **changes})


def compliant_audit(
    summary: str = "Compliant: every claim is authorized.",
) -> ComplianceAuditResult:
    """What a (too) lenient auditor answers."""
    return ComplianceAuditResult(
        is_compliant=True, risk_level="low", violations=[], revision_summary=summary
    )


def failing_audit() -> ComplianceAuditResult:
    """An auditor's verdict with one violation the Python checks would not find by themselves."""
    return ComplianceAuditResult(
        is_compliant=False,
        risk_level="medium",
        violations=[
            ViolationItem(
                field_name="web_title",
                violating_text="an evening ritual",
                regulation_rule="Make only the claims the licence authorizes",
                required_fix="Keep the title to the product name and an authorized claim.",
            )
        ],
        revision_summary="The title implies a sleep benefit the licence does not authorize.",
    )


def refused_outcome(model: str = "fake/model:free") -> ChatOutcome:
    """A content-filter refusal, as the chain reports one."""
    return ChatOutcome(
        model,
        "",
        "content_filter",
        refused=True,
        attempts=[Attempt(model, "refused", 0.0, "content filter")],
    )


# --------------------------------------------------------------------------- fake chat


@dataclass(slots=True)
class FakeCall:
    """One ``complete`` call as the fake saw it."""

    schema_name: str
    temperature: float | None
    max_tokens: int
    messages: list[dict[str, str]]


Scripted = BaseModel | dict[str, Any] | ChatOutcome | BaseException


class FakeChat:
    """A :class:`app.services.llm_client.StructuredChat` whose answers are scripted per schema.

    Each call pops the next item scripted for its schema: a model instance (or its dict) is
    returned parsed, a ``ChatOutcome`` is returned as it is (refusals), an exception is raised.
    Running out of script is a test bug and fails at once, unless ``repeat_last`` keeps answering
    with the last item (the API tests start several runs over one fake).
    """

    name = "fake"
    model = "fake/model:free"

    def __init__(
        self,
        drafts: Iterable[Scripted] = (),
        audits: Iterable[Scripted] = (),
        *,
        repeat_last: bool = False,
    ) -> None:
        self._queues: dict[type[BaseModel], deque[Scripted]] = {
            ChannelCopyDraft: deque(drafts),
            ComplianceAuditResult: deque(audits),
        }
        self.repeat_last = repeat_last
        self.calls: list[FakeCall] = []

    def script(self, *, drafts: Iterable[Scripted] = (), audits: Iterable[Scripted] = ()) -> None:
        self._queues[ChannelCopyDraft].extend(drafts)
        self._queues[ComplianceAuditResult].extend(audits)

    @property
    def draft_calls(self) -> list[FakeCall]:
        return [call for call in self.calls if call.schema_name == "channel_copy_draft"]

    @property
    def audit_calls(self) -> list[FakeCall]:
        return [call for call in self.calls if call.schema_name == "compliance_audit_result"]

    async def complete(
        self,
        *,
        messages: Any,
        schema_model: type[BaseModel],
        temperature: float | None,
        max_tokens: int,
        schema_name: str = "output",
    ) -> ChatOutcome:
        self.calls.append(
            FakeCall(schema_name, temperature, max_tokens, [dict(m) for m in messages])
        )
        queue = self._queues.get(schema_model)
        if queue is None:
            raise AssertionError(f"FakeChat has no script for {schema_model.__name__}")
        if not queue:
            raise AssertionError(f"FakeChat ran out of scripted {schema_model.__name__} answers")
        item = queue[0] if self.repeat_last and len(queue) == 1 else queue.popleft()
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, ChatOutcome):
            return item
        parsed = item if isinstance(item, BaseModel) else schema_model.model_validate(item)
        return ChatOutcome(
            self.model,
            parsed.model_dump_json(),
            "stop",
            parsed=parsed,
            attempts=[Attempt(self.model, "answered", 0.0)],
        )

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "models": [self.model]}


class RecordingDispatcher(WebhookDispatcher):
    """The real dispatcher, remembering every payload it was asked to deliver."""

    def __init__(self, settings: Settings, **kwargs: Any) -> None:
        super().__init__(settings, **kwargs)
        self.payloads: list[dict[str, Any]] = []

    async def dispatch(self, payload: dict[str, Any]) -> Any:
        self.payloads.append(payload)
        return await super().dispatch(payload)


# ------------------------------------------------------------------------- the pipeline


def build_components(
    settings: Settings,
    chat: FakeChat | None,
    *,
    dispatcher: WebhookDispatcher | None = None,
    inventory: InMemoryInventory | None = None,
    webhook_store: InMemoryWebhookStore | None = None,
    budget: Any = None,
) -> Components:
    """The production wiring with the in-memory stand-ins: the same compiled topology."""
    inventory = inventory or InMemoryInventory()
    webhook_store = webhook_store or InMemoryWebhookStore()
    dispatcher = dispatcher or RecordingDispatcher(settings, audit=webhook_store)
    checkpointer = InMemorySaver()
    nodes = PipelineNodes(
        settings, tools=InventoryTools(inventory), chat=chat, dispatcher=dispatcher
    )
    return Components(
        settings=settings,
        db=None,
        inventory=inventory,
        webhook_store=webhook_store,
        chat=chat,
        dispatcher=dispatcher,
        checkpointer=checkpointer,
        graph=build_graph(nodes, checkpointer),
        budget=budget,
    )


def run_config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def start_input(
    thread_id: str,
    *,
    sku_code: str = SKU,
    target_channels: list[str] | None = None,
    campaign_angle: str = CAMPAIGN_ANGLE,
    max_revisions: int = 3,
) -> PipelineState:
    """The state a new thread starts from, exactly as the API builds it."""
    return initial_state(
        thread_id=thread_id,
        sku_code=sku_code,
        target_channels=list(ALL_CHANNELS) if target_channels is None else target_channels,
        campaign_angle=campaign_angle,
        max_revisions=max_revisions,
        now=utc_now_iso(),
    )


@pytest.fixture
def chat() -> FakeChat:
    """An unscripted fake: each test scripts exactly the answers it expects to be asked for."""
    return FakeChat()


@pytest.fixture
def pipeline(settings: Settings, chat: FakeChat) -> Components:
    """The compiled graph over an in-memory saver, with the fake chat and a recording dispatcher."""
    return build_components(settings, chat)


# ------------------------------------------------------------------------------- the app

START_BODY: dict[str, Any] = {
    "sku_code": SKU,
    "target_channels": list(ALL_CHANNELS),
    "campaign_angle": CAMPAIGN_ANGLE,
    "max_revisions": 2,
}


@dataclass
class Stand:
    """An app over fakes, with every collaborator reachable for assertions."""

    settings: Settings
    components: Components
    app: Any
    chat: FakeChat = field(init=False)

    def __post_init__(self) -> None:
        self.chat = self.components.chat  # type: ignore[assignment]


def build_stand(settings: Settings, chat: FakeChat | None = None, **extra: Any) -> Stand:
    """The real FastAPI app over fakes. The default fake answers every run with a clean pass."""
    if chat is None:
        chat = FakeChat([good_draft()], [compliant_audit()], repeat_last=True)
    components = build_components(settings, chat, **extra)
    return Stand(settings, components, create_app(settings=settings, components=components))


def client_for(app: Any) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


@pytest.fixture
def stand(settings: Settings) -> Stand:
    return build_stand(settings)


@pytest.fixture
def components(stand: Stand) -> Components:
    return stand.components


@pytest.fixture
def app(stand: Stand) -> Any:
    return stand.app


@pytest.fixture
async def client(app: Any) -> AsyncIterator[AsyncClient]:
    async with client_for(app) as http:
        yield http
