"""
Industrial event stream router.

Responsibilities:
- Route incoming events to specialized processors
- Identify event categories
- Support telemetry, safety, incident, prediction,
  anomaly and maintenance streams
- Provide fallback handling for unknown events
- Track routing results
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Dict, Optional

logger = logging.getLogger(__name__)

Processor = Callable[
    [Dict[str, Any]],
    Awaitable[Dict[str, Any]],
]


class StreamRouter:
    """Routes industrial streaming events."""

    EVENT_ROUTES = {
        "telemetry": "telemetry",
        "sensor": "telemetry",
        "machine_telemetry": "telemetry",

        "safety": "safety",
        "fire": "safety",
        "smoke": "safety",
        "overheating": "safety",
        "motion": "safety",
        "restricted_zone": "safety",
        "unsafe_proximity": "safety",

        "incident": "incident",
        "emergency": "incident",

        "prediction": "prediction",
        "failure_prediction": "prediction",
        "rul_prediction": "prediction",

        "anomaly": "anomaly",
        "sensor_anomaly": "anomaly",
        "vibration_anomaly": "anomaly",

        "maintenance": "maintenance",
        "work_order": "maintenance",
    }

    def __init__(
        self,
        processors: Optional[
            Dict[str, Processor]
        ] = None,
    ) -> None:
        self.processors = processors or {}

        self.metrics = {
            "total": 0,
            "routed": 0,
            "unknown": 0,
            "failed": 0,
        }

    def register_processor(
        self,
        route: str,
        processor: Processor,
    ) -> None:
        """Register a processor for a stream route."""

        if not route:
            raise ValueError(
                "route cannot be empty"
            )

        if not callable(processor):
            raise TypeError(
                "processor must be callable"
            )

        self.processors[
            route.lower().strip()
        ] = processor

        logger.info(
            "Stream processor registered",
            extra={"route": route},
        )

    async def route(
        self,
        event: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Determine the correct stream processor."""

        self.metrics["total"] += 1

        try:
            if not isinstance(event, dict):
                raise ValueError(
                    "Event must be a dictionary"
                )

            event_type = self._get_event_type(
                event
            )

            if not event_type:
                self.metrics["unknown"] += 1

                return {
                    "status": "ignored",
                    "reason": "missing_event_type",
                }

            route = self._resolve_route(
                event_type
            )

            if route is None:
                self.metrics["unknown"] += 1

                logger.warning(
                    "Unknown stream event type",
                    extra={
                        "event_type": event_type
                    },
                )

                return {
                    "status": "ignored",
                    "event_type": event_type,
                    "reason": "unknown_event_type",
                }

            processor = self.processors.get(
                route
            )

            if processor is None:
                self.metrics["failed"] += 1

                logger.error(
                    "No processor registered",
                    extra={"route": route},
                )

                return {
                    "status": "failed",
                    "event_type": event_type,
                    "route": route,
                    "reason": "processor_not_registered",
                }

            enriched_event = {
                **event,
                "stream_route": route,
            }

            result = await processor(
                enriched_event
            )

            self.metrics["routed"] += 1

            return {
                "status": "routed",
                "event_type": event_type,
                "route": route,
                "result": result,
            }

        except ValueError as exc:
            self.metrics["failed"] += 1

            logger.warning(
                "Invalid stream event: %s",
                exc,
            )

            return {
                "status": "invalid",
                "reason": str(exc),
            }

        except Exception:
            self.metrics["failed"] += 1

            logger.exception(
                "Stream routing failed"
            )

            return {
                "status": "failed",
                "reason": "internal_routing_error",
            }

    @staticmethod
    def _get_event_type(
        event: Dict[str, Any],
    ) -> Optional[str]:
        """Extract event type from common event fields."""

        value = (
            event.get("event_type")
            or event.get("type")
            or event.get("category")
        )

        if value is None:
            return None

        return str(value).lower().strip()

    @classmethod
    def _resolve_route(
        cls,
        event_type: str,
    ) -> Optional[str]:
        """Resolve an event type to a processing route."""

        if event_type in cls.EVENT_ROUTES:
            return cls.EVENT_ROUTES[event_type]

        # Prefix-based fallback routing.
        if event_type.startswith("telemetry"):
            return "telemetry"

        if (
            event_type.startswith("safety")
            or event_type.startswith("fire")
            or event_type.startswith("smoke")
        ):
            return "safety"

        if (
            event_type.startswith("incident")
            or event_type.startswith("emergency")
        ):
            return "incident"

        if event_type.startswith("anomaly"):
            return "anomaly"

        if (
            event_type.startswith("prediction")
            or event_type.startswith("failure_prediction")
        ):
            return "prediction"

        if event_type.startswith("maintenance"):
            return "maintenance"

        return None

    async def route_batch(
        self,
        events: list[Dict[str, Any]],
    ) -> list[Dict[str, Any]]:
        """Route multiple events sequentially."""

        results = []

        for event in events:
            results.append(
                await self.route(event)
            )

        return results

    def get_metrics(self) -> Dict[str, int]:
        """Return routing metrics."""

        return dict(self.metrics)

    def reset_metrics(self) -> None:
        """Reset routing counters."""

        for key in self.metrics:
            self.metrics[key] = 0


async def create_stream_router(
    telemetry_processor: Optional[
        Processor
    ] = None,
    safety_processor: Optional[
        Processor
    ] = None,
    incident_processor: Optional[
        Processor
    ] = None,
    prediction_processor: Optional[
        Processor
    ] = None,
    anomaly_processor: Optional[
        Processor
    ] = None,
    maintenance_processor: Optional[
        Processor
    ] = None,
) -> StreamRouter:
    """Create a configured stream router."""

    processors: Dict[
        str,
        Processor,
    ] = {}

    if telemetry_processor:
        processors["telemetry"] = (
            telemetry_processor
        )

    if safety_processor:
        processors["safety"] = (
            safety_processor
        )

    if incident_processor:
        processors["incident"] = (
            incident_processor
        )

    if prediction_processor:
        processors["prediction"] = (
            prediction_processor
        )

    if anomaly_processor:
        processors["anomaly"] = (
            anomaly_processor
        )

    if maintenance_processor:
        processors["maintenance"] = (
            maintenance_processor
        )

    return StreamRouter(
        processors=processors
    )