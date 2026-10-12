"""Redis-backed cache and fixed-window request limiter."""

import json
from typing import Any


class NullCache:
    def get(self, key: str) -> Any | None:
        return None

    def set(self, key: str, value: Any, ttl: int) -> None:
        pass

    def allow(self, identity: str, limit: int = 60, window: int = 60) -> bool:
        return True

    def ping(self) -> bool:
        return True


class RedisCache:
    def __init__(self, url: str):
        import redis

        self.client = redis.Redis.from_url(url, decode_responses=True, socket_timeout=2)

    def get(self, key: str) -> Any | None:
        value = self.client.get(key)
        return json.loads(value) if value is not None else None

    def set(self, key: str, value: Any, ttl: int) -> None:
        self.client.setex(key, ttl, json.dumps(value, ensure_ascii=False))

    def allow(self, identity: str, limit: int = 60, window: int = 60) -> bool:
        key = f"rate:{identity}"
        with self.client.pipeline() as pipe:
            pipe.incr(key)
            pipe.expire(key, window, nx=True)
            count, _ = pipe.execute()
        return int(count) <= limit

    def ping(self) -> bool:
        return bool(self.client.ping())
