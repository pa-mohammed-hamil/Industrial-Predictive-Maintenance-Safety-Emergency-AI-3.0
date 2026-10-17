# app/config/redis.py

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, AsyncIterator
from uuid import UUID

import redis.asyncio as redis
from redis.asyncio import Redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)


# =========================================================
# Configuration
# =========================================================

@dataclass(slots=True, frozen=True)
class RedisConfig:
    url: str = os.getenv(
        "REDIS_URL",
        "redis://localhost:6379/0",
    )

    max_connections: int = int(
        os.getenv(
            "REDIS_MAX_CONNECTIONS",
            "50",
        )
    )

    socket_timeout: float = float(
        os.getenv(
            "REDIS_SOCKET_TIMEOUT",
            "5",
        )
    )

    socket_connect_timeout: float = float(
        os.getenv(
            "REDIS_CONNECT_TIMEOUT",
            "5",
        )
    )

    health_check_interval: int = int(
        os.getenv(
            "REDIS_HEALTH_CHECK_INTERVAL",
            "30",
        )
    )

    default_ttl: int = int(
        os.getenv(
            "REDIS_DEFAULT_TTL",
            "300",
        )
    )

    key_prefix: str = os.getenv(
        "REDIS_KEY_PREFIX",
        "industrial-ai",
    )


# =========================================================
# Serialization
# =========================================================

def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, UUID):
        return str(value)

    if isinstance(value, Enum):
        return value.value

    if is_dataclass(value):
        return asdict(value)

    if hasattr(value, "model_dump"):
        return value.model_dump()

    raise TypeError(
        f"{type(value).__name__} is not "
        "JSON serializable"
    )


def serialize(value: Any) -> str:
    return json.dumps(
        value,
        default=_json_default,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def deserialize(
    value: str | bytes | None,
) -> Any:
    if value is None:
        return None

    if isinstance(value, bytes):
        value = value.decode("utf-8")

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


# =========================================================
# Redis Manager
# =========================================================

class RedisManager:
    """
    Central Redis integration.

    Supports:
    - generic cache
    - telemetry snapshots
    - asset health cache
    - prediction cache
    - session storage
    - distributed locks
    - counters
    - rate limiting
    - pub/sub
    """

    def __init__(
        self,
        config: RedisConfig | None = None,
    ) -> None:
        self.config = config or RedisConfig()

        self._client: Redis | None = None

        self._startup_lock = asyncio.Lock()

    # -----------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------

    async def connect(self) -> None:
        async with self._startup_lock:
            if self._client is not None:
                return

            client = redis.from_url(
                self.config.url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=(
                    self.config.max_connections
                ),
                socket_timeout=(
                    self.config.socket_timeout
                ),
                socket_connect_timeout=(
                    self.config.socket_connect_timeout
                ),
                health_check_interval=(
                    self.config.health_check_interval
                ),
            )

            try:
                await client.ping()

            except Exception:
                await client.aclose()

                logger.exception(
                    "Redis connection failed"
                )

                raise

            self._client = client

            logger.info(
                "Redis connected"
            )

    async def close(self) -> None:
        async with self._startup_lock:
            client = self._client

            self._client = None

            if client is not None:
                await client.aclose()

            logger.info(
                "Redis connection closed"
            )

    def _require_client(self) -> Redis:
        if self._client is None:
            raise RuntimeError(
                "Redis is not connected. "
                "Call await redis_manager.connect()."
            )

        return self._client

    # -----------------------------------------------------
    # Keys
    # -----------------------------------------------------

    def key(
        self,
        *parts: Any,
    ) -> str:
        cleaned = [
            str(part).strip(":")
            for part in parts
            if part is not None
        ]

        return ":".join(
            [
                self.config.key_prefix,
                *cleaned,
            ]
        )

    # -----------------------------------------------------
    # Generic cache
    # -----------------------------------------------------

    async def set(
        self,
        key: str,
        value: Any,
        *,
        ttl: int | None = None,
    ) -> bool:
        client = self._require_client()

        effective_ttl = (
            self.config.default_ttl
            if ttl is None
            else ttl
        )

        payload = serialize(value)

        if effective_ttl > 0:
            result = await client.set(
                key,
                payload,
                ex=effective_ttl,
            )
        else:
            result = await client.set(
                key,
                payload,
            )

        return bool(result)

    async def get(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        client = self._require_client()

        value = await client.get(key)

        if value is None:
            return default

        return deserialize(value)

    async def delete(
        self,
        *keys: str,
    ) -> int:
        if not keys:
            return 0

        client = self._require_client()

        return int(
            await client.delete(*keys)
        )

    async def exists(
        self,
        key: str,
    ) -> bool:
        client = self._require_client()

        return bool(
            await client.exists(key)
        )

    async def expire(
        self,
        key: str,
        ttl: int,
    ) -> bool:
        client = self._require_client()

        return bool(
            await client.expire(
                key,
                ttl,
            )
        )

    async def ttl(
        self,
        key: str,
    ) -> int:
        client = self._require_client()

        return int(
            await client.ttl(key)
        )

    # -----------------------------------------------------
    # Telemetry
    # -----------------------------------------------------

    async def set_telemetry(
        self,
        *,
        asset_id: str,
        telemetry: dict[str, Any],
        ttl: int = 60,
    ) -> bool:
        return await self.set(
            self.key(
                "telemetry",
                asset_id,
            ),
            telemetry,
            ttl=ttl,
        )

    async def get_telemetry(
        self,
        asset_id: str,
    ) -> dict[str, Any] | None:
        return await self.get(
            self.key(
                "telemetry",
                asset_id,
            )
        )

    # -----------------------------------------------------
    # Asset health
    # -----------------------------------------------------

    async def set_asset_health(
        self,
        *,
        asset_id: str,
        health: dict[str, Any],
        ttl: int = 120,
    ) -> bool:
        return await self.set(
            self.key(
                "asset-health",
                asset_id,
            ),
            health,
            ttl=ttl,
        )

    async def get_asset_health(
        self,
        asset_id: str,
    ) -> dict[str, Any] | None:
        return await self.get(
            self.key(
                "asset-health",
                asset_id,
            )
        )

    # -----------------------------------------------------
    # ML prediction cache
    # -----------------------------------------------------

    async def cache_prediction(
        self,
        *,
        asset_id: str,
        model_name: str,
        prediction: dict[str, Any],
        ttl: int = 300,
    ) -> bool:
        return await self.set(
            self.key(
                "prediction",
                model_name,
                asset_id,
            ),
            prediction,
            ttl=ttl,
        )

    async def get_prediction(
        self,
        *,
        asset_id: str,
        model_name: str,
    ) -> dict[str, Any] | None:
        return await self.get(
            self.key(
                "prediction",
                model_name,
                asset_id,
            )
        )

    # -----------------------------------------------------
    # Sessions
    # -----------------------------------------------------

    async def set_session(
        self,
        *,
        session_id: str,
        data: dict[str, Any],
        ttl: int = 3600,
    ) -> bool:
        return await self.set(
            self.key(
                "session",
                session_id,
            ),
            data,
            ttl=ttl,
        )

    async def get_session(
        self,
        session_id: str,
    ) -> dict[str, Any] | None:
        return await self.get(
            self.key(
                "session",
                session_id,
            )
        )

    async def delete_session(
        self,
        session_id: str,
    ) -> int:
        return await self.delete(
            self.key(
                "session",
                session_id,
            )
        )

    # -----------------------------------------------------
    # Counters
    # -----------------------------------------------------

    async def increment(
        self,
        key: str,
        amount: int = 1,
        *,
        ttl: int | None = None,
    ) -> int:
        client = self._require_client()

        value = await client.incrby(
            key,
            amount,
        )

        if ttl is not None:
            await client.expire(
                key,
                ttl,
            )

        return int(value)

    # -----------------------------------------------------
    # Rate limiting
    # -----------------------------------------------------

    async def check_rate_limit(
        self,
        *,
        identifier: str,
        limit: int,
        window_seconds: int,
    ) -> tuple[bool, int]:
        if limit <= 0:
            raise ValueError(
                "limit must be greater than zero"
            )

        if window_seconds <= 0:
            raise ValueError(
                "window_seconds must be positive"
            )

        client = self._require_client()

        key = self.key(
            "rate-limit",
            identifier,
        )

        # INCR + EXPIRE in one transaction.
        async with client.pipeline(
            transaction=True
        ) as pipe:
            pipe.incr(key)
            pipe.ttl(key)

            count, current_ttl = (
                await pipe.execute()
            )

        count = int(count)
        current_ttl = int(current_ttl)

        if current_ttl < 0:
            await client.expire(
                key,
                window_seconds,
            )

        remaining = max(
            0,
            limit - count,
        )

        return (
            count <= limit,
            remaining,
        )

    # -----------------------------------------------------
    # Distributed lock
    # -----------------------------------------------------

    @asynccontextmanager
    async def lock(
        self,
        name: str,
        *,
        timeout: int = 30,
        blocking_timeout: int = 10,
    ) -> AsyncIterator[None]:
        client = self._require_client()

        lock = client.lock(
            self.key(
                "lock",
                name,
            ),
            timeout=timeout,
            blocking_timeout=blocking_timeout,
        )

        acquired = await lock.acquire()

        if not acquired:
            raise TimeoutError(
                f"Unable to acquire Redis lock: "
                f"{name}"
            )

        try:
            yield

        finally:
            try:
                await lock.release()
            except RedisError:
                logger.warning(
                    "Redis lock already expired: %s",
                    name,
                )

    # -----------------------------------------------------
    # Pub/Sub
    # -----------------------------------------------------

    async def publish(
        self,
        channel: str,
        message: Any,
    ) -> int:
        client = self._require_client()

        return int(
            await client.publish(
                self.key(
                    "channel",
                    channel,
                ),
                serialize(message),
            )
        )

    async def subscribe(
        self,
        channel: str,
    ) -> AsyncIterator[Any]:
        client = self._require_client()

        pubsub = client.pubsub()

        channel_name = self.key(
            "channel",
            channel,
        )

        await pubsub.subscribe(
            channel_name
        )

        logger.info(
            "Redis subscribed channel=%s",
            channel_name,
        )

        try:
            async for message in pubsub.listen():
                if (
                    message.get("type")
                    != "message"
                ):
                    continue

                yield deserialize(
                    message.get("data")
                )

        finally:
            await pubsub.unsubscribe(
                channel_name
            )

            await pubsub.aclose()

    # -----------------------------------------------------
    # Hashes
    # -----------------------------------------------------

    async def hash_set(
        self,
        key: str,
        mapping: dict[str, Any],
    ) -> int:
        client = self._require_client()

        serialized = {
            str(field): serialize(value)
            for field, value
            in mapping.items()
        }

        return int(
            await client.hset(
                key,
                mapping=serialized,
            )
        )

    async def hash_get(
        self,
        key: str,
        field: str,
    ) -> Any:
        client = self._require_client()

        value = await client.hget(
            key,
            field,
        )

        return deserialize(value)

    async def hash_get_all(
        self,
        key: str,
    ) -> dict[str, Any]:
        client = self._require_client()

        values = await client.hgetall(
            key
        )

        return {
            field: deserialize(value)
            for field, value
            in values.items()
        }

    # -----------------------------------------------------
    # Health
    # -----------------------------------------------------

    async def health_check(
        self,
    ) -> dict[str, Any]:
        client = self._client

        if client is None:
            return {
                "status": "unhealthy",
                "connected": False,
            }

        try:
            result = await client.ping()

            pool = (
                client.connection_pool
            )

            return {
                "status": (
                    "healthy"
                    if result
                    else "unhealthy"
                ),
                "connected": bool(result),
                "max_connections": (
                    pool.max_connections
                ),
            }

        except RedisError as exc:
            logger.exception(
                "Redis health check failed"
            )

            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(exc),
            }

    # -----------------------------------------------------
    # Cache invalidation
    # -----------------------------------------------------

    async def delete_pattern(
        self,
        pattern: str,
        *,
        batch_size: int = 500,
    ) -> int:
        client = self._require_client()

        deleted = 0

        async for key in client.scan_iter(
            match=pattern,
            count=batch_size,
        ):
            deleted += int(
                await client.delete(key)
            )

        return deleted

    async def clear_asset_cache(
        self,
        asset_id: str,
    ) -> int:
        patterns = (
            self.key(
                "telemetry",
                asset_id,
            ),
            self.key(
                "asset-health",
                asset_id,
            ),
            self.key(
                "prediction",
                "*",
                asset_id,
            ),
        )

        deleted = 0

        for pattern in patterns:
            deleted += (
                await self.delete_pattern(
                    pattern
                )
            )

        return deleted


# =========================================================
# Global Redis manager
# =========================================================

redis_manager = RedisManager()


async def start_redis() -> None:
    await redis_manager.connect()


async def close_redis() -> None:
    await redis_manager.close()


async def redis_health_check() -> dict[str, Any]:
    return await redis_manager.health_check()


def get_redis() -> RedisManager:
    return redis_manager