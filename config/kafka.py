# app/config/kafka.py

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, AsyncIterator, Awaitable, Callable
from uuid import UUID

try:
    from aiokafka import (
        AIOKafkaConsumer,
        AIOKafkaProducer,
    )
    from aiokafka.errors import KafkaError
except ImportError as exc:
    raise ImportError(
        "aiokafka is required. "
        "Install it with: pip install aiokafka"
    ) from exc


logger = logging.getLogger(__name__)


# =========================================================
# Configuration
# =========================================================

@dataclass(slots=True, frozen=True)
class KafkaConfig:
    bootstrap_servers: str = os.getenv(
        "KAFKA_BOOTSTRAP_SERVERS",
        "localhost:9092",
    )

    client_id: str = os.getenv(
        "KAFKA_CLIENT_ID",
        "industrial-ai",
    )

    consumer_group: str = os.getenv(
        "KAFKA_CONSUMER_GROUP",
        "industrial-ai-workers",
    )

    auto_offset_reset: str = os.getenv(
        "KAFKA_AUTO_OFFSET_RESET",
        "latest",
    )

    enable_auto_commit: bool = (
        os.getenv(
            "KAFKA_ENABLE_AUTO_COMMIT",
            "true",
        ).lower()
        == "true"
    )

    request_timeout_ms: int = int(
        os.getenv(
            "KAFKA_REQUEST_TIMEOUT_MS",
            "30000",
        )
    )

    session_timeout_ms: int = int(
        os.getenv(
            "KAFKA_SESSION_TIMEOUT_MS",
            "10000",
        )
    )

    max_poll_records: int = int(
        os.getenv(
            "KAFKA_MAX_POLL_RECORDS",
            "500",
        )
    )

    security_protocol: str = os.getenv(
        "KAFKA_SECURITY_PROTOCOL",
        "PLAINTEXT",
    )

    sasl_mechanism: str | None = os.getenv(
        "KAFKA_SASL_MECHANISM"
    )

    sasl_username: str | None = os.getenv(
        "KAFKA_SASL_USERNAME"
    )

    sasl_password: str | None = os.getenv(
        "KAFKA_SASL_PASSWORD"
    )


# =========================================================
# Topics
# =========================================================

class KafkaTopic(str, Enum):
    TELEMETRY = "industrial.telemetry"

    TEMPERATURE = "industrial.sensor.temperature"
    VIBRATION = "industrial.sensor.vibration"
    PRESSURE = "industrial.sensor.pressure"
    CURRENT = "industrial.sensor.current"
    ENERGY = "industrial.sensor.energy"

    ANOMALIES = "industrial.anomalies"

    PREDICTIONS = "industrial.predictions"
    FAILURE_PREDICTIONS = (
        "industrial.predictions.failure"
    )
    RUL_PREDICTIONS = (
        "industrial.predictions.rul"
    )

    SAFETY_EVENTS = "industrial.safety.events"

    FIRE_EVENTS = "industrial.safety.fire"
    SMOKE_EVENTS = "industrial.safety.smoke"
    OVERHEATING_EVENTS = (
        "industrial.safety.overheating"
    )

    INCIDENTS = "industrial.incidents"

    EMERGENCY = "industrial.emergency"

    MAINTENANCE = "industrial.maintenance"
    WORK_ORDERS = "industrial.work-orders"

    COST_EVENTS = "industrial.cost"

    ASSET_HEALTH = "industrial.asset-health"

    NOTIFICATIONS = "industrial.notifications"

    AUDIT = "industrial.audit"

    DEAD_LETTER = "industrial.dead-letter"


# =========================================================
# Serialization
# =========================================================

def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, date):
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
        f"Object of type {type(value).__name__} "
        "is not JSON serializable"
    )


def serialize_json(
    value: Any,
) -> bytes:
    return json.dumps(
        value,
        default=_json_default,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def deserialize_json(
    value: bytes | None,
) -> Any:
    if value is None:
        return None

    return json.loads(
        value.decode("utf-8")
    )


def serialize_key(
    value: str | bytes | None,
) -> bytes | None:
    if value is None:
        return None

    if isinstance(value, bytes):
        return value

    return str(value).encode("utf-8")


# =========================================================
# Event envelope
# =========================================================

@dataclass(slots=True)
class KafkaEvent:
    event_type: str
    payload: Any

    asset_id: str | None = None
    site_id: str | None = None

    event_id: str | None = None
    source: str = "industrial-ai"

    timestamp: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        timestamp = (
            self.timestamp
            or datetime.now().astimezone()
        )

        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "source": self.source,
            "asset_id": self.asset_id,
            "site_id": self.site_id,
            "timestamp": timestamp.isoformat(),
            "payload": self.payload,
        }


# =========================================================
# Kafka manager
# =========================================================

class KafkaManager:

    def __init__(
        self,
        config: KafkaConfig | None = None,
    ) -> None:
        self.config = (
            config or KafkaConfig()
        )

        self._producer: (
            AIOKafkaProducer | None
        ) = None

        self._lock = asyncio.Lock()

    # -----------------------------------------------------
    # Producer
    # -----------------------------------------------------

    async def start(self) -> None:
        async with self._lock:
            if self._producer is not None:
                return

            kwargs: dict[str, Any] = {
                "bootstrap_servers": (
                    self.config.bootstrap_servers
                ),
                "client_id": (
                    self.config.client_id
                ),
                "value_serializer": (
                    serialize_json
                ),
                "key_serializer": (
                    serialize_key
                ),
                "request_timeout_ms": (
                    self.config.request_timeout_ms
                ),
                "security_protocol": (
                    self.config.security_protocol
                ),
                "acks": "all",
                "enable_idempotence": True,
            }

            self._add_security_options(
                kwargs
            )

            producer = AIOKafkaProducer(
                **kwargs
            )

            try:
                await producer.start()

            except Exception:
                logger.exception(
                    "Kafka producer startup failed"
                )
                raise

            self._producer = producer

            logger.info(
                "Kafka producer connected to %s",
                self.config.bootstrap_servers,
            )

    async def stop(self) -> None:
        async with self._lock:
            producer = self._producer

            self._producer = None

            if producer is not None:
                try:
                    await producer.stop()

                except Exception:
                    logger.exception(
                        "Kafka producer shutdown failed"
                    )

            logger.info(
                "Kafka producer stopped"
            )

    async def publish(
        self,
        topic: str | KafkaTopic,
        value: Any,
        *,
        key: str | bytes | None = None,
        headers: list[
            tuple[str, bytes]
        ] | None = None,
    ) -> dict[str, Any]:
        producer = self._require_producer()

        topic_name = (
            topic.value
            if isinstance(topic, KafkaTopic)
            else topic
        )

        try:
            metadata = (
                await producer.send_and_wait(
                    topic_name,
                    value=value,
                    key=key,
                    headers=headers,
                )
            )

            return {
                "topic": metadata.topic,
                "partition": metadata.partition,
                "offset": metadata.offset,
            }

        except KafkaError:
            logger.exception(
                "Kafka publish failed topic=%s",
                topic_name,
            )
            raise

    async def publish_event(
        self,
        topic: str | KafkaTopic,
        event: KafkaEvent,
        *,
        key: str | None = None,
    ) -> dict[str, Any]:

        event_key = (
            key
            or event.asset_id
            or event.event_id
        )

        return await self.publish(
            topic,
            event.to_dict(),
            key=event_key,
        )

    # -----------------------------------------------------
    # Industrial event helpers
    # -----------------------------------------------------

    async def publish_telemetry(
        self,
        *,
        asset_id: str,
        site_id: str,
        telemetry: dict[str, Any],
    ) -> dict[str, Any]:

        event = KafkaEvent(
            event_type="telemetry",
            asset_id=asset_id,
            site_id=site_id,
            payload=telemetry,
        )

        return await self.publish_event(
            KafkaTopic.TELEMETRY,
            event,
        )

    async def publish_anomaly(
        self,
        *,
        asset_id: str,
        site_id: str,
        anomaly: dict[str, Any],
    ) -> dict[str, Any]:

        return await self.publish_event(
            KafkaTopic.ANOMALIES,
            KafkaEvent(
                event_type="anomaly_detected",
                asset_id=asset_id,
                site_id=site_id,
                payload=anomaly,
            ),
        )

    async def publish_failure_prediction(
        self,
        *,
        asset_id: str,
        site_id: str,
        prediction: dict[str, Any],
    ) -> dict[str, Any]:

        return await self.publish_event(
            KafkaTopic.FAILURE_PREDICTIONS,
            KafkaEvent(
                event_type="failure_prediction",
                asset_id=asset_id,
                site_id=site_id,
                payload=prediction,
            ),
        )

    async def publish_rul_prediction(
        self,
        *,
        asset_id: str,
        site_id: str,
        prediction: dict[str, Any],
    ) -> dict[str, Any]:

        return await self.publish_event(
            KafkaTopic.RUL_PREDICTIONS,
            KafkaEvent(
                event_type="rul_prediction",
                asset_id=asset_id,
                site_id=site_id,
                payload=prediction,
            ),
        )

    async def publish_safety_event(
        self,
        *,
        asset_id: str,
        site_id: str,
        event_data: dict[str, Any],
    ) -> dict[str, Any]:

        return await self.publish_event(
            KafkaTopic.SAFETY_EVENTS,
            KafkaEvent(
                event_type="safety_event",
                asset_id=asset_id,
                site_id=site_id,
                payload=event_data,
            ),
        )

    async def publish_incident(
        self,
        *,
        asset_id: str | None,
        site_id: str,
        incident: dict[str, Any],
    ) -> dict[str, Any]:

        return await self.publish_event(
            KafkaTopic.INCIDENTS,
            KafkaEvent(
                event_type="incident",
                asset_id=asset_id,
                site_id=site_id,
                payload=incident,
            ),
        )

    async def publish_emergency(
        self,
        *,
        asset_id: str | None,
        site_id: str,
        emergency: dict[str, Any],
    ) -> dict[str, Any]:

        return await self.publish_event(
            KafkaTopic.EMERGENCY,
            KafkaEvent(
                event_type="emergency",
                asset_id=asset_id,
                site_id=site_id,
                payload=emergency,
            ),
        )

    async def publish_maintenance(
        self,
        *,
        asset_id: str,
        site_id: str,
        maintenance: dict[str, Any],
    ) -> dict[str, Any]:

        return await self.publish_event(
            KafkaTopic.MAINTENANCE,
            KafkaEvent(
                event_type="maintenance",
                asset_id=asset_id,
                site_id=site_id,
                payload=maintenance,
            ),
        )

    async def publish_cost_event(
        self,
        *,
        asset_id: str,
        site_id: str,
        cost: dict[str, Any],
    ) -> dict[str, Any]:

        return await self.publish_event(
            KafkaTopic.COST_EVENTS,
            KafkaEvent(
                event_type="cost_event",
                asset_id=asset_id,
                site_id=site_id,
                payload=cost,
            ),
        )

    # -----------------------------------------------------
    # Consumer creation
    # -----------------------------------------------------

    def create_consumer(
        self,
        *topics: str | KafkaTopic,
        group_id: str | None = None,
    ) -> AIOKafkaConsumer:

        if not topics:
            raise ValueError(
                "At least one Kafka topic is required"
            )

        topic_names = [
            (
                topic.value
                if isinstance(topic, KafkaTopic)
                else topic
            )
            for topic in topics
        ]

        kwargs: dict[str, Any] = {
            "bootstrap_servers": (
                self.config.bootstrap_servers
            ),
            "group_id": (
                group_id
                or self.config.consumer_group
            ),
            "client_id": (
                f"{self.config.client_id}-consumer"
            ),
            "auto_offset_reset": (
                self.config.auto_offset_reset
            ),
            "enable_auto_commit": (
                self.config.enable_auto_commit
            ),
            "value_deserializer": (
                deserialize_json
            ),
            "key_deserializer": (
                lambda value: (
                    value.decode("utf-8")
                    if value
                    else None
                )
            ),
            "session_timeout_ms": (
                self.config.session_timeout_ms
            ),
            "max_poll_records": (
                self.config.max_poll_records
            ),
            "security_protocol": (
                self.config.security_protocol
            ),
        }

        self._add_security_options(
            kwargs
        )

        return AIOKafkaConsumer(
            *topic_names,
            **kwargs,
        )

    async def consume(
        self,
        *topics: str | KafkaTopic,
        group_id: str | None = None,
    ) -> AsyncIterator[Any]:

        consumer = self.create_consumer(
            *topics,
            group_id=group_id,
        )

        await consumer.start()

        logger.info(
            "Kafka consumer started topics=%s",
            topics,
        )

        try:
            async for message in consumer:
                yield message

        finally:
            await consumer.stop()

    async def consume_with_handler(
        self,
        *topics: str | KafkaTopic,
        handler: Callable[
            [Any],
            Awaitable[None],
        ],
        group_id: str | None = None,
    ) -> None:

        async for message in self.consume(
            *topics,
            group_id=group_id,
        ):
            try:
                await handler(message)

            except asyncio.CancelledError:
                raise

            except Exception as exc:
                logger.exception(
                    "Kafka handler failed topic=%s "
                    "partition=%s offset=%s",
                    message.topic,
                    message.partition,
                    message.offset,
                )

                await self.publish(
                    KafkaTopic.DEAD_LETTER,
                    {
                        "original_topic": (
                            message.topic
                        ),
                        "partition": (
                            message.partition
                        ),
                        "offset": (
                            message.offset
                        ),
                        "error": str(exc),
                        "payload": (
                            message.value
                        ),
                    },
                    key=message.key,
                )

    # -----------------------------------------------------
    # Health
    # -----------------------------------------------------

    async def health_check(
        self,
    ) -> dict[str, Any]:

        producer = self._producer

        if producer is None:
            return {
                "status": "unhealthy",
                "connected": False,
            }

        try:
            cluster = producer.client.cluster

            brokers = cluster.brokers()

            return {
                "status": (
                    "healthy"
                    if brokers
                    else "degraded"
                ),
                "connected": bool(brokers),
                "broker_count": len(brokers),
            }

        except Exception as exc:
            logger.exception(
                "Kafka health check failed"
            )

            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(exc),
            }

    # -----------------------------------------------------
    # Internal
    # -----------------------------------------------------

    def _require_producer(
        self,
    ) -> AIOKafkaProducer:

        if self._producer is None:
            raise RuntimeError(
                "Kafka producer is not started. "
                "Call await kafka_manager.start()."
            )

        return self._producer

    def _add_security_options(
        self,
        kwargs: dict[str, Any],
    ) -> None:

        if self.config.sasl_mechanism:
            kwargs["sasl_mechanism"] = (
                self.config.sasl_mechanism
            )

        if self.config.sasl_username:
            kwargs["sasl_plain_username"] = (
                self.config.sasl_username
            )

        if self.config.sasl_password:
            kwargs["sasl_plain_password"] = (
                self.config.sasl_password
            )


# =========================================================
# Global manager
# =========================================================

kafka_manager = KafkaManager()


async def start_kafka() -> None:
    await kafka_manager.start()


async def close_kafka() -> None:
    await kafka_manager.stop()


async def kafka_health_check() -> dict[str, Any]:
    return await kafka_manager.health_check()