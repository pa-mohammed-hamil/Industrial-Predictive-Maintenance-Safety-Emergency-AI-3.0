"""
Industrial safety event processor.

Responsibilities:
- Process real-time safety events
- Evaluate fire, smoke, overheating, motion and zone violations
- Calculate safety severity
- Correlate multiple safety signals
- Trigger incident creation
- Trigger emergency escalation
- Publish notifications
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

SafetyHandler = Callable[
    [Dict[str, Any]],
    Awaitable[Optional[Dict[str, Any]]],
]


class SafetyProcessor:
    """Processes real-time industrial safety events."""

    SEVERITY_LEVELS = {
        "info": 0,
        "low": 1,
        "medium": 2,
        "high": 3,
        "critical": 4,
    }

    EVENT_SEVERITY = {
        "motion": "low",
        "person_detected": "low",
        "restricted_zone": "high",
        "unsafe_proximity": "high",
        "smoke": "high",
        "overheating": "high",
        "fire": "critical",
        "explosion": "critical",
        "gas_leak": "critical",
        "emergency_stop": "critical",
        "multi_signal_emergency": "critical",
    }

    def __init__(
        self,
        handlers: Optional[
            Dict[str, SafetyHandler]
        ] = None,
        incident_handler: Optional[
            SafetyHandler
        ] = None,
        emergency_handler: Optional[
            SafetyHandler
        ] = None,
        notification_handler: Optional[
            SafetyHandler
        ] = None,
    ) -> None:
        self.handlers = handlers or {}
        self.incident_handler = incident_handler
        self.emergency_handler = emergency_handler
        self.notification_handler = notification_handler

    def register_handler(
        self,
        event_type: str,
        handler: SafetyHandler,
    ) -> None:
        """Register a safety-specific handler."""

        if not event_type:
            raise ValueError(
                "event_type cannot be empty"
            )

        if not callable(handler):
            raise TypeError(
                "handler must be callable"
            )

        self.handlers[
            event_type.lower().strip()
        ] = handler

    async def process(
        self,
        event: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Process a single safety event."""

        try:
            safety_event = self._normalize_event(event)

            self._validate_event(safety_event)

            safety_event = self._determine_severity(
                safety_event
            )

            safety_event = self._evaluate_thresholds(
                safety_event
            )

            handler = self.handlers.get(
                safety_event["event_type"]
            )

            handler_result = None

            if handler:
                handler_result = await handler(
                    safety_event
                )

            safety_event = (
                self._apply_handler_result(
                    safety_event,
                    handler_result,
                )
            )

            await self._handle_actions(
                safety_event
            )

            logger.info(
                "Safety event processed",
                extra={
                    "event_id": safety_event["event_id"],
                    "event_type": safety_event[
                        "event_type"
                    ],
                    "severity": safety_event[
                        "severity"
                    ],
                    "requires_incident": safety_event[
                        "requires_incident"
                    ],
                    "requires_emergency": safety_event[
                        "requires_emergency"
                    ],
                },
            )

            return {
                "status": "processed",
                "event_id": safety_event[
                    "event_id"
                ],
                "event_type": safety_event[
                    "event_type"
                ],
                "severity": safety_event[
                    "severity"
                ],
                "requires_incident": safety_event[
                    "requires_incident"
                ],
                "requires_emergency": safety_event[
                    "requires_emergency"
                ],
            }

        except ValueError as exc:
            logger.warning(
                "Invalid safety event: %s",
                exc,
            )

            return {
                "status": "invalid",
                "reason": str(exc),
            }

        except Exception:
            logger.exception(
                "Safety event processing failed"
            )

            return {
                "status": "failed",
                "reason": "internal_processing_error",
            }

    def _normalize_event(
        self,
        event: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Normalize the safety event envelope."""

        if not isinstance(event, dict):
            raise ValueError(
                "Safety event must be a dictionary"
            )

        event_type = (
            event.get("event_type")
            or event.get("type")
        )

        if not event_type:
            raise ValueError(
                "Missing safety event type"
            )

        timestamp = event.get("timestamp")

        if not timestamp:
            timestamp = datetime.now(
                timezone.utc
            ).isoformat()

        return {
            **event,
            "event_id": str(
                event.get("event_id")
                or uuid4()
            ),
            "event_type": str(
                event_type
            ).lower().strip(),
            "timestamp": timestamp,
            "processed_at": datetime.now(
                timezone.utc
            ).isoformat(),
        }

    def _validate_event(
        self,
        event: Dict[str, Any],
    ) -> None:
        """Validate required safety information."""

        if not event.get("event_id"):
            raise ValueError(
                "Missing event_id"
            )

        if not event.get("event_type"):
            raise ValueError(
                "Missing event_type"
            )

        confidence = event.get(
            "confidence"
        )

        if confidence is not None:
            try:
                confidence = float(
                    confidence
                )
            except (TypeError, ValueError):
                raise ValueError(
                    "confidence must be numeric"
                )

            if not 0 <= confidence <= 1:
                raise ValueError(
                    "confidence must be between 0 and 1"
                )

            event["confidence"] = confidence

    def _determine_severity(
        self,
        event: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Determine severity using explicit severity or event type."""

        explicit = event.get("severity")

        if explicit:
            severity = str(
                explicit
            ).lower().strip()

            if severity not in self.SEVERITY_LEVELS:
                raise ValueError(
                    f"Invalid severity: {severity}"
                )

            event["severity"] = severity
            return event

        event_type = event["event_type"]

        event["severity"] = self.EVENT_SEVERITY.get(
            event_type,
            "medium",
        )

        return event

    def _evaluate_thresholds(
        self,
        event: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Apply deterministic safety thresholds.

        ML/CV models can provide signals, while these rules
        provide a deterministic safety layer.
        """

        event_type = event["event_type"]

        temperature = event.get(
            "temperature"
        )

        smoke_probability = event.get(
            "smoke_probability"
        )

        fire_probability = event.get(
            "fire_probability"
        )

        vibration = event.get(
            "vibration"
        )

        if (
            fire_probability is not None
            and float(fire_probability) >= 0.80
        ):
            event["severity"] = "critical"

        elif (
            smoke_probability is not None
            and float(smoke_probability) >= 0.80
        ):
            event["severity"] = "high"

        elif (
            temperature is not None
            and float(temperature) >= 100
        ):
            event["severity"] = "critical"

        elif (
            temperature is not None
            and float(temperature) >= 85
        ):
            event["severity"] = max(
                event["severity"],
                "high",
                key=self.SEVERITY_LEVELS.get,
            )

        elif (
            vibration is not None
            and float(vibration) >= 10
        ):
            event["severity"] = max(
                event["severity"],
                "high",
                key=self.SEVERITY_LEVELS.get,
            )

        event["safety_score"] = self._calculate_safety_score(
            event
        )

        event["requires_incident"] = (
            self.SEVERITY_LEVELS[
                event["severity"]
            ] >= self.SEVERITY_LEVELS["high"]
        )

        event["requires_emergency"] = (
            event["severity"] == "critical"
        )

        event["requires_notification"] = (
            event["severity"] in {
                "medium",
                "high",
                "critical",
            }
        )

        return event

    def _calculate_safety_score(
        self,
        event: Dict[str, Any],
    ) -> float:
        """
        Calculate a normalized safety risk score.

        0 = normal
        100 = extreme risk
        """

        severity_score = (
            self.SEVERITY_LEVELS[
                event["severity"]
            ]
            / 4
        ) * 100

        confidence = event.get(
            "confidence",
            1.0,
        )

        return round(
            severity_score * float(confidence),
            2,
        )

    @staticmethod
    def _apply_handler_result(
        event: Dict[str, Any],
        result: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Merge optional ML/CV safety-engine results."""

        if not result:
            return event

        if "severity" in result:
            event["severity"] = result[
                "severity"
            ]

        if "confidence" in result:
            event["confidence"] = result[
                "confidence"
            ]

        if "safety_score" in result:
            event["safety_score"] = result[
                "safety_score"
            ]

        if "reason" in result:
            event["reason"] = result["reason"]

        if "evidence" in result:
            event["evidence"] = result["evidence"]

        return event

    async def _handle_actions(
        self,
        event: Dict[str, Any],
    ) -> None:
        """Trigger downstream safety actions."""

        if (
            event["requires_notification"]
            and self.notification_handler
        ):
            try:
                await self.notification_handler(
                    event
                )

            except Exception:
                logger.exception(
                    "Safety notification failed",
                    extra={
                        "event_id": event[
                            "event_id"
                        ]
                    },
                )

        if (
            event["requires_incident"]
            and self.incident_handler
        ):
            try:
                await self.incident_handler(
                    event
                )

            except Exception:
                logger.exception(
                    "Incident creation failed",
                    extra={
                        "event_id": event[
                            "event_id"
                        ]
                    },
                )

        if (
            event["requires_emergency"]
            and self.emergency_handler
        ):
            try:
                await self.emergency_handler(
                    event
                )

            except Exception:
                logger.exception(
                    "Emergency escalation failed",
                    extra={
                        "event_id": event[
                            "event_id"
                        ]
                    },
                )

    async def process_batch(
        self,
        events: list[Dict[str, Any]],
    ) -> list[Dict[str, Any]]:
        """Process multiple safety events."""

        results = []

        for event in events:
            results.append(
                await self.process(event)
            )

        return results


async def process_safety_event(
    event: Dict[str, Any],
    handlers: Optional[
        Dict[str, SafetyHandler]
    ] = None,
    incident_handler: Optional[
        SafetyHandler
    ] = None,
    emergency_handler: Optional[
        SafetyHandler
    ] = None,
    notification_handler: Optional[
        SafetyHandler
    ] = None,
) -> Dict[str, Any]:
    """Convenience function for processing one safety event."""

    processor = SafetyProcessor(
        handlers=handlers,
        incident_handler=incident_handler,
        emergency_handler=emergency_handler,
        notification_handler=notification_handler,
    )

    return await processor.process(event)