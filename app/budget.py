"""The one ceiling on public pipeline runs in demo mode.

The showcase calls free models on the owner's OpenRouter account, and one run can cost up to
2 * max(1, max_revisions) model calls (a draft and an audit per attempt; the API reserves one
attempt more). A public visitor must not be able to spend that account's day, or its minute,
alone. Five rules, checked in order:

1. per visitor address: ``RUN_RATE_LIMIT_PER_MINUTE`` runs a minute;
2. per visitor address: ``RUN_DAILY_LIMIT_PER_CLIENT`` runs a UTC day;
3. for everyone together: ``RUN_GLOBAL_LIMIT_PER_MINUTE``;
4. for everyone together: ``RUN_DAILY_LIMIT`` a UTC day;
5. OpenRouter's own count of today's free requests (``GET /key``, read at most once a minute):
   nothing is admitted once the count falls to a reserve.

Rules 1 and 2 are fairness between visitors and depend on knowing who the visitor is, which
behind a proxy is only as good as the header the address is read from (``client_id_for`` in
``app.main``). Rules 3 to 5 protect the account whatever the visitor claims to be.

Everything here lives in this process. A free host restarts the service with an empty memory,
which is exactly why rule 5 exists: OpenRouter's counter is the one a restart cannot erase.
The same pattern protected Projects 1 and 2 of this portfolio.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from app.config import Settings

logger = logging.getLogger(__name__)

SECONDS_PER_DAY = 86_400
WINDOW_S = 60.0
ALLOWANCE_TTL_S = 60.0


class BudgetExceeded(Exception):
    """A run was refused; ``retry_after_s`` says when to try again."""

    def __init__(self, message: str, retry_after_s: int) -> None:
        super().__init__(message)
        self.message = message
        self.retry_after_s = max(1, int(retry_after_s))


@dataclass(frozen=True, slots=True)
class Allowance:
    used: int
    limit: int
    remaining: int


AllowanceReader = Callable[[], Awaitable["Allowance | None"]]


async def read_openrouter_allowance(settings: Settings) -> Allowance | None:
    """Ask OpenRouter how many free-model requests the account has left today. Never raises.

    The key is sent to OpenRouter's own address and nowhere else; only the three numbers are
    kept or logged.
    """
    key = settings.openrouter_api_key
    if key is None or not settings.openrouter_base_url.startswith("https://openrouter.ai/"):
        return None
    import httpx

    url = settings.openrouter_base_url.rstrip("/") + "/key"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                url, headers={"Authorization": f"Bearer {key.get_secret_value()}"}
            )
    except Exception as exc:  # noqa: BLE001 - "could not be read" is an answer too
        logger.warning("OpenRouter allowance could not be read (%s)", type(exc).__name__)
        return None
    if response.status_code != 200:
        logger.warning("OpenRouter allowance request answered HTTP %d", response.status_code)
        return None
    try:
        counts = response.json()["data"]["free_model_daily_requests"]
        allowance = Allowance(
            used=int(counts["used"]), limit=int(counts["limit"]), remaining=int(counts["remaining"])
        )
    except (KeyError, TypeError, ValueError):
        logger.warning("OpenRouter allowance answer had an unexpected shape")
        return None
    logger.info(
        "OpenRouter counts %d of %d free requests used today; %d left",
        allowance.used,
        allowance.limit,
        allowance.remaining,
    )
    return allowance


class RunBudget:
    """Admission control for ``POST /api/v1/workflows/start``."""

    def __init__(
        self,
        settings: Settings,
        *,
        allowance_reader: AllowanceReader | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
    ) -> None:
        self.per_client = settings.run_rate_limit_per_minute
        self.per_client_day = settings.run_daily_limit_per_client
        self.per_minute = settings.run_global_limit_per_minute
        self.per_day = settings.run_daily_limit
        self.reserve = settings.openrouter_allowance_reserve
        if allowance_reader is not None:
            self._reader: AllowanceReader | None = allowance_reader
        elif settings.openrouter_allowance_check and settings.llm_provider == "openrouter":

            async def _default_reader() -> Allowance | None:
                return await read_openrouter_allowance(settings)

            self._reader = _default_reader
        else:
            self._reader = None
        self._clock = clock
        self._wall = wall
        self._clients: dict[str, deque[float]] = defaultdict(deque)
        self._client_days: dict[str, int] = {}
        self._global: deque[float] = deque()
        self._day: int | None = None
        self._day_count = 0
        self._allowance: Allowance | None = None
        self._allowance_at = -math.inf
        self._spent_since_read = 0
        self._lock = asyncio.Lock()

    async def admit(self, client_id: str, *, calls: int = 1) -> None:
        """Reserve one run (of up to ``calls`` model calls) for ``client_id`` or raise."""
        async with self._lock:
            now = self._clock()
            today = int(self._wall() // SECONDS_PER_DAY)
            if today != self._day:
                self._day, self._day_count = today, 0
                self._client_days.clear()
            self._prune(now)

            window = self._clients.get(client_id)
            if window is not None and len(window) >= self.per_client:
                raise BudgetExceeded(
                    "Too many runs from this address; please wait a minute.",
                    self._retry_after(window[0], now),
                )
            if self._client_days.get(client_id, 0) >= self.per_client_day:
                raise BudgetExceeded(
                    "This address has reached its runs for today; it resets at midnight UTC.",
                    self._seconds_to_midnight(),
                )
            if len(self._global) >= self.per_minute:
                raise BudgetExceeded(
                    "The demo is busy right now; please try again in a minute.",
                    self._retry_after(self._global[0], now),
                )
            if self._day_count >= self.per_day:
                raise BudgetExceeded(
                    "The demo's daily run limit is reached; it resets at midnight UTC.",
                    self._seconds_to_midnight(),
                )
            allowance = await self._current_allowance(now)
            if allowance is not None:
                left = allowance.remaining - self._spent_since_read
                if left - calls <= self._effective_reserve(allowance):
                    raise BudgetExceeded(
                        "Today's free-model allowance is used up; the demo resumes tomorrow (UTC).",
                        self._seconds_to_midnight(),
                    )

            self._clients[client_id].append(now)
            self._client_days[client_id] = self._client_days.get(client_id, 0) + 1
            self._global.append(now)
            self._day_count += 1
            self._spent_since_read += calls

    def snapshot(self) -> dict[str, int | None]:
        """What the playground shows visitors about the limits."""
        allowance = self._allowance
        return {
            "per_client_per_minute": self.per_client,
            "per_client_per_day": self.per_client_day,
            "per_minute": self.per_minute,
            "per_day": self.per_day,
            "used_today": self._day_count,
            "free_requests_remaining_today": (
                max(0, allowance.remaining - self._spent_since_read) if allowance else None
            ),
        }

    # ----------------------------------------------------------------- internals
    def _effective_reserve(self, allowance: Allowance) -> int:
        """The reserve, never more than a fifth of the day (a 50-a-day account keeps 40)."""
        return min(self.reserve, allowance.limit // 5)

    def _prune(self, now: float) -> None:
        cutoff = now - WINDOW_S
        while self._global and self._global[0] <= cutoff:
            self._global.popleft()
        for client_id in [c for c, w in self._clients.items() if not w or w[0] <= cutoff]:
            window = self._clients[client_id]
            while window and window[0] <= cutoff:
                window.popleft()
            if not window:
                del self._clients[client_id]

    @staticmethod
    def _retry_after(oldest: float, now: float) -> int:
        return int(math.ceil(max(1.0, WINDOW_S - (now - oldest))))

    def _seconds_to_midnight(self) -> int:
        return int(SECONDS_PER_DAY - (self._wall() % SECONDS_PER_DAY)) or 1

    async def _current_allowance(self, now: float) -> Allowance | None:
        if self._reader is None:
            return None
        if now - self._allowance_at < ALLOWANCE_TTL_S:
            return self._allowance
        self._allowance_at = now  # also throttles re-reads after a failed read
        fresh = await self._reader()
        if fresh is not None:
            self._allowance = fresh
            self._spent_since_read = 0
        return self._allowance
