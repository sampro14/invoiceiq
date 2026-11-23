"""Protecting the free model quotas: a per-visitor hourly cap and an app-wide daily cap."""

import time
from collections import deque
from collections.abc import Callable
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# Gemini's daily quota resets at midnight Pacific, so ours does too.
PACIFIC = ZoneInfo("America/Los_Angeles")


class RateLimiter:
    """In memory, which is enough for one process. Not shared across instances."""

    def __init__(self, per_ip_hour: int, per_day: int, now: Callable[[], float] = time.time):
        self.per_ip_hour = per_ip_hour
        self.per_day = per_day
        self._now = now
        self._hits: dict[str, deque[float]] = {}
        self._day = ""
        self._count = 0

    def _roll_day(self) -> None:
        day = datetime.fromtimestamp(self._now(), PACIFIC).date().isoformat()
        if day != self._day:
            self._day, self._count = day, 0

    def check(self, ip: str) -> tuple[str, int] | None:
        """None when allowed (and counted), otherwise (message, seconds to wait)."""
        self._roll_day()
        now = self._now()
        if self._count >= self.per_day:
            local = datetime.fromtimestamp(now, PACIFIC)
            midnight = (local + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            return "Today's free model quota is used up.", max(1, int(midnight.timestamp() - now))
        hits = self._hits.setdefault(ip, deque())
        while hits and hits[0] <= now - 3600:
            hits.popleft()
        if len(hits) >= self.per_ip_hour:
            wait = max(1, int(hits[0] + 3600 - now))
            return f"You have used your {self.per_ip_hour} runs for this hour.", wait
        hits.append(now)
        self._count += 1
        if len(self._hits) > 10_000:
            self._hits = {k: v for k, v in self._hits.items() if v and v[-1] > now - 3600}
        return None

    def remaining_today(self) -> int:
        self._roll_day()
        return max(0, self.per_day - self._count)
