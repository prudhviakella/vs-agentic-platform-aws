"""
rate_limiter.py — In-memory rate limiter.
For production scale use ElastiCache Redis.
"""
import time
from collections import defaultdict


class RateLimiter:
    def __init__(self, requests_per_minute: int = 30):
        self.rpm    = requests_per_minute
        self._store = defaultdict(list)

    def allow(self, user_id: str) -> bool:
        now    = time.time()
        window = 60.0
        times  = [t for t in self._store[user_id] if now - t < window]
        self._store[user_id] = times
        if len(times) >= self.rpm:
            return False
        self._store[user_id].append(now)
        return True
