"""
Generic event processor for industrial predictive maintenance.

Responsibilities:
- Validate incoming events
- Normalize event metadata
- Route events by type
- Persist/process through downstream handlers
- Publish processing results
- Handle failures safely
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Optional

logger = logging.getLogger(__name__)

EventHandler = Callable[[dict[str, Any]], Awaitable[None]]


class EventProcessor:
    """Routes and processes events received from the streaming layer."""

    def __init__(
        self,
        handlers: Optional[Dict[str, EventHandler]] = None,
    ) -> None:
        self.handlers: Dict[str, EventHandler] = handlers or {}

    def register_handler(
        self,
        event_type: str,
        handler: EventHandler,
    ) -> None:
        """Register an async handler for an event type."""
        if not event_type:
            raise ValueError("event_type cannot be empty")

        if not callable(handler):
            raise TypeError("handler must be callable")

        self.handlers[event_type] = handler

        logger.info(
            "Registered event handler",
            extra={"event_type": event_type},
        )

    async def process(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate, normalize, route and process an event.

        Returns a processing result rather than raising expected
        event-processing errors.
        """
        try:
            normalized_event = self._normalize_event(event)
            event_type = normalized_event["event_type"]

            handler = self.handlers.get(event_type)

            if handler is None:
                logger.warning(
                    "No handler registered for event",
                    extra={"event_type": event_type},
                )

                return {
                    "status": "ignored",
                    "event_type": event_type,
                    "reason": "no_handler",
                }

            await handler(normalized_event)

            logger.info(
                "Event processed successfully",
                extra={
                    "event_id": normalized_event["event_id"],
                    "event_type": event_type,
                },
            )

            return {
                "status": "processed",
                "event_id": normalized_event["event_id"],
                "event_type": event_type,
            }

        except ValueError as exc:
            logger.warning(
                "Invalid event",
                extra={"error": str(exc)},
            )

            return {
                "status": "invalid",
                "reason": str(exc),
            }

        except Exception:
            logger.exception("Unexpected event processing failure")

            return {
                "status": "failed",
                "reason": "internal_processing_error",
            }

    def _normalize_event(
        self,
        event: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate and normalize the event envelope."""

        if not isinstance(event, dict):
            raise ValueError("Event must be a dictionary")

        event_type = event.get("event_type")

        if not event_type:
            raise ValueError("Missing event_type")

        event_id = event.get("event_id")

        if not event_id:
            event_id = self._generate_event_id()

        timestamp = event.get("timestamp")

        if not timestamp:
            timestamp = datetime.now(timezone.utc).isoformat()

        asset_id = event.get("asset_id")

        if not asset_id:
            logger.warning(
                "Event has no asset_id",
                extra={"event_type": event_type},
            )

        return {
            **event,
            "event_id": str(event_id),
            "event_type": str(event_type).lower().strip(),
            "timestamp": timestamp,
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _generate_event_id() -> str:
        """Generate a unique event identifier."""
        from uuid import uuid4

        return str(uuid4())


async def process_event(
    event: Dict[str, Any],
    handlers: Optional[Dict[str, EventHandler]] = None,
) -> Dict[str, Any]:
    """
    Convenience function for processing a single event.
    """
    processor = EventProcessor(handlers=handlers)
    return await processor.process(event)