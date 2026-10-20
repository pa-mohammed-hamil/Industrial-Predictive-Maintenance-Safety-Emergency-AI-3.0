"""
Industrial incident processor.

Responsibilities:
- Process safety and operational incident events
- Normalize incident data
- Determine severity and escalation
- Correlate related signals
- Trigger emergency workflows
- Publish notifications
- Return a structured processing result
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

IncidentHandler = Callable[[dict[str, Any]], Awaitable[None]]


class IncidentProcessor:
    """Processes and routes industrial incident events."""

    SEVERITY_LEVELS = {
        "low": 1,
        "medium": 2,
        "high": 3,
        "critical": 4,
    }

    ESCALATION_LEVELS = {
        "low": "log",
        "medium": "notify",
        "high": "escalate",
        "critical": "emergency",
    }

    def __init__(
        self,
        handlers: Optional[Dict[str, IncidentHandler]] = None,
        notification_handler: Optional[IncidentHandler] = None,
        emergency_handler: Optional[IncidentHandler] = None,
    ) -> None:
        self.handlers = handlers or {}
        self.notification_handler = notification_handler
        self.emergency_handler = emergency_handler

    def register_handler(
        self,
        incident_type: str,
        handler: IncidentHandler,
    ) -> None:
        """Register a handler for a specific incident type."""
        if not incident_type:
            raise ValueError("incident_type cannot be empty")

        if not callable(handler):
            raise TypeError("handler must be callable")

        self.handlers[incident_type.lower().strip()] = handler

    async def process(
        self,
        event: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Process a single industrial incident event."""

        try:
            incident = self._normalize_event(event)

            self._validate_incident(incident)

            incident = self._calculate_severity(incident)

            incident = self._determine_escalation(incident)

            handler = self.handlers.get(
                incident["incident_type"]
            )

            if handler:
                await handler(incident)

            await self._handle_escalation(incident)

            logger.info(
                "Incident processed",
                extra={
                    "incident_id": incident["incident_id"],
                    "incident_type": incident["incident_type"],
                    "severity": incident["severity"],
                    "escalation": incident["escalation_level"],
                },
            )

            return {
                "status": "processed",
                "incident_id": incident["incident_id"],
                "incident_type": incident["incident_type"],
                "severity": incident["severity"],
                "escalation_level": incident["escalation_level"],
            }

        except ValueError as exc:
            logger.warning(
                "Invalid incident event: %s",
                exc,
            )

            return {
                "status": "invalid",
                "reason": str(exc),
            }

        except Exception:
            logger.exception(
                "Unexpected incident processing failure"
            )

            return {
                "status": "failed",
                "reason": "internal_processing_error",
            }

    def _normalize_event(
        self,
        event: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Normalize the incoming incident event."""

        if not isinstance(event, dict):
            raise ValueError("Incident event must be a dictionary")

        incident_type = event.get("incident_type")

        if not incident_type:
            incident_type = event.get("type")

        if not incident_type:
            raise ValueError("Missing incident_type")

        incident_id = (
            event.get("incident_id")
            or event.get("event_id")
            or str(uuid4())
        )

        timestamp = event.get("timestamp")

        if not timestamp:
            timestamp = datetime.now(timezone.utc).isoformat()

        return {
            **event,
            "incident_id": str(incident_id),
            "incident_type": str(
                incident_type
            ).lower().strip(),
            "timestamp": timestamp,
            "processed_at": datetime.now(
                timezone.utc
            ).isoformat(),
        }

    def _validate_incident(
        self,
        incident: Dict[str, Any],
    ) -> None:
        """Validate the minimum incident information."""

        if not incident.get("incident_id"):
            raise ValueError("Missing incident_id")

        if not incident.get("incident_type"):
            raise ValueError("Missing incident_type")

        severity = incident.get("severity")

        if severity:
            severity = str(severity).lower().strip()

            if severity not in self.SEVERITY_LEVELS:
                raise ValueError(
                    f"Invalid severity: {severity}"
                )

            incident["severity"] = severity

    def _calculate_severity(
        self,
        incident: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Determine severity from explicit severity or
        incident signals.
        """

        if incident.get("severity"):
            return incident

        incident_type = incident["incident_type"]

        critical_types = {
            "fire",
            "explosion",
            "gas_leak",
            "major_overheating",
            "worker_injury",
            "emergency_stop",
            "multi_signal_emergency",
        }

        high_types = {
            "smoke",
            "severe_overheating",
            "restricted_zone",
            "unsafe_proximity",
            "critical_vibration",
            "critical_pressure",
            "critical_electrical_fault",
        }

        medium_types = {
            "anomaly",
            "equipment_fault",
            "high_temperature",
            "high_vibration",
            "maintenance_required",
        }

        if incident_type in critical_types:
            severity = "critical"

        elif incident_type in high_types:
            severity = "high"

        elif incident_type in medium_types:
            severity = "medium"

        else:
            severity = "low"

        incident["severity"] = severity

        return incident

    def _determine_escalation(
        self,
        incident: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Determine the required escalation action."""

        severity = incident["severity"]

        incident["escalation_level"] = (
            self.ESCALATION_LEVELS.get(
                severity,
                "log",
            )
        )

        incident["requires_notification"] = (
            severity in {
                "medium",
                "high",
                "critical",
            }
        )

        incident["requires_emergency_response"] = (
            severity == "critical"
        )

        return incident

    async def _handle_escalation(
        self,
        incident: Dict[str, Any],
    ) -> None:
        """Trigger notification and emergency workflows."""

        if (
            incident["requires_notification"]
            and self.notification_handler
        ):
            try:
                await self.notification_handler(incident)

            except Exception:
                logger.exception(
                    "Incident notification failed",
                    extra={
                        "incident_id": incident["incident_id"]
                    },
                )

        if (
            incident["requires_emergency_response"]
            and self.emergency_handler
        ):
            try:
                await self.emergency_handler(incident)

            except Exception:
                logger.exception(
                    "Emergency response failed",
                    extra={
                        "incident_id": incident["incident_id"]
                    },
                )

    async def process_batch(
        self,
        events: list[Dict[str, Any]],
    ) -> list[Dict[str, Any]]:
        """Process multiple incidents independently."""

        results: list[Dict[str, Any]] = []

        for event in events:
            result = await self.process(event)
            results.append(result)

        return results


async def process_incident(
    event: Dict[str, Any],
    handlers: Optional[Dict[str, IncidentHandler]] = None,
    notification_handler: Optional[IncidentHandler] = None,
    emergency_handler: Optional[IncidentHandler] = None,
) -> Dict[str, Any]:
    """Convenience function for processing one incident."""

    processor = IncidentProcessor(
        handlers=handlers,
        notification_handler=notification_handler,
        emergency_handler=emergency_handler,
    )

    return await processor.process(event)