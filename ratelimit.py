"""In-memory rate limiting for login attempts.

Single-replica only — switch to Redis when scaling out.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque

# IP -> deque of timestamps of failed login attempts
_login_failures: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=10))

LOGIN_WINDOW_SEC = 60
LOGIN_MAX_FAILURES = 5


def login_throttled(ip: str) -> bool:
    """Return True if this IP has exceeded the failure budget within the window."""
    now = time.time()
    cutoff = now - LOGIN_WINDOW_SEC
    failures = _login_failures[ip]
    while failures and failures[0] < cutoff:
        failures.popleft()
    return len(failures) >= LOGIN_MAX_FAILURES


def record_login_failure(ip: str) -> None:
    _login_failures[ip].append(time.time())


def reset_login_failures(ip: str) -> None:
    if ip in _login_failures:
        _login_failures[ip].clear()
