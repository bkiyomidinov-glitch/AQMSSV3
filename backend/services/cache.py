"""
Cache Service with Thread-Safety and Stale-While-Revalidate (SWR) support.
Abstracts caching layer for market data, indicators, and external API responses.
"""

import time
import threading
from typing import Any, Optional, Dict, Callable


class SimpleMemoryCache:
    """Thread-safe, TTL-based memory cache with Stale-While-Revalidate support."""

    def __init__(self, default_ttl: int = 60, stale_ttl: int = 3600):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._refreshing: Dict[str, bool] = {}
        self._lock = threading.Lock()
        self.default_ttl = default_ttl
        self.stale_ttl = stale_ttl

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._cache.get(key)
            if not entry:
                return None
            if time.time() > entry['expires_at']:
                self._cache.pop(key, None)
                return None
            return entry['value']

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        now = time.time()
        fresh_duration = ttl if ttl is not None else self.default_ttl
        with self._lock:
            self._cache[key] = {
                'value': value,
                'fresh_until': now + fresh_duration,
                'expires_at': now + fresh_duration + self.stale_ttl
            }

    def get_or_compute_swr(
        self,
        key: str,
        compute_fn: Callable[[], Any],
        ttl: Optional[int] = None
    ) -> Any:
        """
        Stale-While-Revalidate: Returns cached value in <1ms.
        If data is stale, triggers a background refresh worker without blocking.
        """
        now = time.time()
        fresh_duration = ttl if ttl is not None else self.default_ttl

        with self._lock:
            entry = self._cache.get(key)
            needs_background_refresh = False

            if entry is not None:
                if now <= entry['fresh_until']:
                    return entry['value']

                # Stale but still valid
                if now <= entry['expires_at']:
                    if not self._refreshing.get(key, False):
                        self._refreshing[key] = True
                        needs_background_refresh = True
                    return entry['value']

            # If no entry or hard expired, compute synchronously
            self._refreshing[key] = True

        if needs_background_refresh:
            def _async_worker():
                try:
                    val = compute_fn()
                    if val is not None:
                        self.set(key, val, ttl=fresh_duration)
                finally:
                    with self._lock:
                        self._refreshing[key] = False

            threading.Thread(target=_async_worker, daemon=True).start()
            return entry['value']

        # Synchronous computation fallback
        try:
            val = compute_fn()
            if val is not None:
                self.set(key, val, ttl=fresh_duration)
            return val
        finally:
            with self._lock:
                self._refreshing[key] = False

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._refreshing.clear()


market_cache = SimpleMemoryCache(default_ttl=90, stale_ttl=3600)
ip_cache = SimpleMemoryCache(default_ttl=86400, stale_ttl=86400 * 2)
