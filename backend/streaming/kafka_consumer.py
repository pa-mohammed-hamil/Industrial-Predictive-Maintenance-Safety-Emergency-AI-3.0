"""
Kafka consumer for industrial telemetry and safety events.

Responsibilities:
- Connect to Kafka
- Subscribe to configured topics
- Deserialize messages
- Validate message envelopes
- Route events to the event processor
- Handle retries and failures
- Commit offsets safely
- Support graceful shutdown
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable, Dict, Iterable, Optional

from aiokafka import AIOKafkaConsumer
from aiokafka.errors import KafkaError

logger = logging.getLogger(__name__)

EventProcessor = Callable[
    [Dict[str, Any]],
    Awaitable[Dict[str, Any]],
]


class KafkaEventConsumer:
    """Async Kafka consumer for industrial event streams."""

    def __init__(
        self,
        bootstrap_servers: str,
        group_id: str,
        topics: Iterable[str],
        event_processor: EventProcessor,
        client_id: str = "industrial-ai-consumer",
        auto_offset_reset: str = "latest",
        enable_auto_commit: bool = False,
        max_poll_interval_ms: int = 300_000,
        session_timeout_ms: int = 30_000,
    ) -> None:
        if not bootstrap_servers:
            raise ValueError(
                "bootstrap_servers cannot be empty"
            )

        if not group_id:
            raise ValueError(
                "group_id cannot be empty"
            )

        topic_list = list(topics)

        if not topic_list:
            raise ValueError(
                "At least one Kafka topic is required"
            )

        if not callable(event_processor):
            raise TypeError(
                "event_processor must be callable"
            )

        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.topics = topic_list
        self.event_processor = event_processor
        self.client_id = client_id

        self.consumer = AIOKafkaConsumer(
            *self.topics,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            client_id=self.client_id,
            auto_offset_reset=auto_offset_reset,
            enable_auto_commit=enable_auto_commit,
            max_poll_interval_ms=max_poll_interval_ms,
            session_timeout_ms=session_timeout_ms,
            value_deserializer=self._deserialize,
        )

        self._running = False

    @staticmethod
    def _deserialize(
        value: bytes,
    ) -> Dict[str, Any]:
        """Deserialize Kafka JSON payload."""

        try:
            decoded = value.decode("utf-8")
            payload = json.loads(decoded)

        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Invalid Kafka JSON payload: {exc}"
            ) from exc

        if not isinstance(payload, dict):
            raise ValueError(
                "Kafka message must contain a JSON object"
            )

        return payload

    async def start(self) -> None:
        """Start the Kafka consumer."""

        if self._running:
            logger.warning(
                "Kafka consumer is already running"
            )
            return

        await self.consumer.start()

        self._running = True

        logger.info(
            "Kafka consumer started",
            extra={
                "topics": self.topics,
                "group_id": self.group_id,
            },
        )

    async def stop(self) -> None:
        """Gracefully stop the Kafka consumer."""

        if not self._running:
            return

        self._running = False

        try:
            await self.consumer.stop()

        finally:
            logger.info(
                "Kafka consumer stopped",
                extra={
                    "topics": self.topics,
                    "group_id": self.group_id,
                },
            )

    async def consume(self) -> None:
        """
        Continuously consume Kafka messages.

        Offset is committed only after successful event
        processing.
        """

        if not self._running:
            await self.start()

        logger.info("Kafka consumption loop started")

        try:
            async for message in self.consumer:

                if not self._running:
                    break

                await self._handle_message(message)

        except asyncio.CancelledError:
            logger.info(
                "Kafka consumer cancellation received"
            )
            raise

        except KafkaError:
            logger.exception(
                "Kafka error in consumption loop"
            )
            raise

        except Exception:
            logger.exception(
                "Unexpected Kafka consumption error"
            )
            raise

    async def _handle_message(
        self,
        message: Any,
    ) -> None:
        """Process an individual Kafka message."""

        metadata = {
            "topic": message.topic,
            "partition": message.partition,
            "offset": message.offset,
        }

        try:
            event = message.value

            if not isinstance(event, dict):
                raise ValueError(
                    "Kafka event must be a dictionary"
                )

            event.setdefault(
                "kafka_topic",
                message.topic,
            )

            event.setdefault(
                "kafka_partition",
                message.partition,
            )

            event.setdefault(
                "kafka_offset",
                message.offset,
            )

            result = await self.event_processor(event)

            status = (
                result.get("status")
                if isinstance(result, dict)
                else None
            )

            if status == "failed":
                logger.error(
                    "Event processor reported failure",
                    extra={
                        **metadata,
                        "result": result,
                    },
                )

                return

            await self.consumer.commit()

            logger.debug(
                "Kafka message processed and committed",
                extra={
                    **metadata,
                    "status": status,
                },
            )

        except ValueError as exc:
            logger.warning(
                "Invalid Kafka message",
                extra={
                    **metadata,
                    "error": str(exc),
                },
            )

            # Invalid messages are committed so that a permanently
            # malformed event does not block the partition.
            await self.consumer.commit()

        except Exception:
            logger.exception(
                "Kafka message processing failed",
                extra=metadata,
            )

            # Do not commit on unexpected processing failures.
            # The message can therefore be retried.
            raise

    async def run(
        self,
        retry_delay_seconds: float = 5.0,
    ) -> None:
        """
        Run the consumer with reconnect/retry behavior.
        """

        self._running = True

        while self._running:

            try:
                await self.start()
                await self.consume()

            except asyncio.CancelledError:
                self._running = False
                raise

            except Exception:
                logger.exception(
                    "Kafka consumer crashed; retrying"
                )

                if self._running:
                    await asyncio.sleep(
                        retry_delay_seconds
                    )

            finally:
                if self._running:
                    await self.stop()

    async def health_check(self) -> bool:
        """Return whether the consumer is currently running."""

        return self._running

    async def __aenter__(self) -> "KafkaEventConsumer":
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        await self.stop()


async def create_kafka_consumer(
    bootstrap_servers: str,
    group_id: str,
    topics: Iterable[str],
    event_processor: EventProcessor,
) -> KafkaEventConsumer:
    """
    Factory function for creating and starting a Kafka consumer.
    """

    consumer = KafkaEventConsumer(
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        topics=topics,
        event_processor=event_processor,
    )

    await consumer.start()

    return consumer