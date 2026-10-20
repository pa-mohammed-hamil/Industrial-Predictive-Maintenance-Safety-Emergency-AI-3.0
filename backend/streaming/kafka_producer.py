"""
Kafka producer for industrial AI events.

Responsibilities:
- Publish telemetry, safety, incident, maintenance and prediction events
- Serialize events safely
- Attach event metadata
- Support keyed messages for partition affinity
- Support async publishing
- Expose health/flush/close operations
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaError

logger = logging.getLogger(__name__)


class KafkaEventProducer:
    """Async Kafka producer for industrial AI events."""

    def __init__(
        self,
        bootstrap_servers: str,
        client_id: str = "industrial-ai-producer",
        acks: str = "all",
        compression_type: Optional[str] = "gzip",
        linger_ms: int = 5,
        request_timeout_ms: int = 30_000,
    ) -> None:
        if not bootstrap_servers:
            raise ValueError(
                "bootstrap_servers cannot be empty"
            )

        self.bootstrap_servers = bootstrap_servers
        self.client_id = client_id
        self._started = False

        self.producer = AIOKafkaProducer(
            bootstrap_servers=bootstrap_servers,
            client_id=client_id,
            acks=acks,
            compression_type=compression_type,
            linger_ms=linger_ms,
            request_timeout_ms=request_timeout_ms,
            value_serializer=self._serialize,
            key_serializer=self._serialize_key,
        )

    @staticmethod
    def _serialize(
        value: Dict[str, Any],
    ) -> bytes:
        """Serialize an event into UTF-8 JSON."""

        if not isinstance(value, dict):
            raise ValueError(
                "Kafka event must be a dictionary"
            )

        return json.dumps(
            value,
            default=str,
            separators=(",", ":"),
        ).encode("utf-8")

    @staticmethod
    def _serialize_key(
        key: Optional[str],
    ) -> Optional[bytes]:
        """Serialize a Kafka message key."""

        if key is None:
            return None

        return str(key).encode("utf-8")

    async def start(self) -> None:
        """Start the Kafka producer."""

        if self._started:
            return

        await self.producer.start()

        self._started = True

        logger.info(
            "Kafka producer started",
            extra={
                "bootstrap_servers": self.bootstrap_servers,
                "client_id": self.client_id,
            },
        )

    async def stop(self) -> None:
        """Flush and stop the producer."""

        if not self._started:
            return

        try:
            await self.flush()
            await self.producer.stop()

        finally:
            self._started = False

            logger.info(
                "Kafka producer stopped"
            )

    def _build_event(
        self,
        event: Dict[str, Any],
        event_type: str,
    ) -> Dict[str, Any]:
        """Create a normalized event envelope."""

        if not isinstance(event, dict):
            raise ValueError(
                "event must be a dictionary"
            )

        if not event_type:
            raise ValueError(
                "event_type cannot be empty"
            )

        normalized = {
            **event,
            "event_id": str(
                event.get("event_id") or uuid4()
            ),
            "event_type": event_type.lower().strip(),
            "timestamp": event.get(
                "timestamp",
                datetime.now(
                    timezone.utc
                ).isoformat(),
            ),
            "producer": self.client_id,
            "schema_version": event.get(
                "schema_version",
                "1.0",
            ),
        }

        return normalized

    async def publish(
        self,
        topic: str,
        event: Dict[str, Any],
        *,
        event_type: str,
        key: Optional[str] = None,
        headers: Optional[
            Dict[str, str]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Publish an event to Kafka.

        Returns metadata describing the published message.
        """

        if not topic:
            raise ValueError(
                "topic cannot be empty"
            )

        if not self._started:
            await self.start()

        normalized_event = self._build_event(
            event,
            event_type,
        )

        kafka_headers = None

        if headers:
            kafka_headers = [
                (
                    str(header_key),
                    str(header_value).encode("utf-8"),
                )
                for header_key, header_value
                in headers.items()
            ]

        try:
            record = await self.producer.send_and_wait(
                topic=topic,
                key=key,
                value=normalized_event,
                headers=kafka_headers,
            )

            logger.debug(
                "Kafka event published",
                extra={
                    "topic": topic,
                    "partition": record.partition,
                    "offset": record.offset,
                    "event_id": normalized_event[
                        "event_id"
                    ],
                    "event_type": event_type,
                },
            )

            return {
                "status": "published",
                "event_id": normalized_event[
                    "event_id"
                ],
                "topic": record.topic,
                "partition": record.partition,
                "offset": record.offset,
            }

        except KafkaError:
            logger.exception(
                "Kafka publishing failed",
                extra={
                    "topic": topic,
                    "event_type": event_type,
                },
            )
            raise

    async def publish_telemetry(
        self,
        event: Dict[str, Any],
        *,
        topic: str = "telemetry.events",
        key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Publish machine telemetry."""

        return await self.publish(
            topic,
            event,
            event_type="telemetry",
            key=key or event.get("asset_id"),
        )

    async def publish_safety_event(
        self,
        event: Dict[str, Any],
        *,
        topic: str = "safety.events",
        key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Publish a safety event."""

        return await self.publish(
            topic,
            event,
            event_type="safety",
            key=key or event.get("asset_id"),
        )

    async def publish_incident(
        self,
        event: Dict[str, Any],
        *,
        topic: str = "incident.events",
        key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Publish an industrial incident."""

        return await self.publish(
            topic,
            event,
            event_type="incident",
            key=key or event.get("asset_id"),
        )

    async def publish_maintenance_event(
        self,
        event: Dict[str, Any],
        *,
        topic: str = "maintenance.events",
        key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Publish a maintenance event."""

        return await self.publish(
            topic,
            event,
            event_type="maintenance",
            key=key or event.get("asset_id"),
        )

    async def publish_prediction(
        self,
        event: Dict[str, Any],
        *,
        topic: str = "prediction.events",
        key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Publish ML prediction results."""

        return await self.publish(
            topic,
            event,
            event_type="prediction",
            key=key or event.get("asset_id"),
        )

    async def publish_anomaly(
        self,
        event: Dict[str, Any],
        *,
        topic: str = "anomaly.events",
        key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Publish anomaly detection results."""

        return await self.publish(
            topic,
            event,
            event_type="anomaly",
            key=key or event.get("asset_id"),
        )

    async def flush(self) -> None:
        """Wait for buffered Kafka messages to be delivered."""

        if self._started:
            await self.producer.flush()

    async def health_check(self) -> bool:
        """Return producer availability state."""

        return self._started

    async def __aenter__(
        self,
    ) -> "KafkaEventProducer":
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        await self.stop()