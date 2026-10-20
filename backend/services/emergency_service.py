
"""
app/services/emergency_service.py

Industrial Predictive Maintenance, Safety & Emergency AI

Emergency Management Service

Features:
- Emergency creation and validation
- Fire, smoke and overheating events
- Electrical and hazardous incidents
- Emergency severity classification
- Emergency status lifecycle
- Alert acknowledgment
- Escalation tracking
- Emergency response assignment
- Event timeline and audit metadata
- Optional notification integration
- Repository-based persistence

SAFETY:
This module coordinates alerts and workflows.
It does not directly control machinery, emergency
shutdown systems, fire suppression, or evacuation
equipment.

Certified independent safety systems and trained
personnel must remain responsible for safety-critical
actions.
"""

from __future__ import annotations

import inspect
import logging

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, Sequence
from uuid import uuid4


logger = logging.getLogger(__name__)


# ============================================================
# ENUMS
# ============================================================

class EmergencyType(str, Enum):
    FIRE = "fire"
    SMOKE = "smoke"
    OVERHEATING = "overheating"
    ELECTRICAL = "electrical"
    GAS_LEAK = "gas_leak"
    CHEMICAL_LEAK = "chemical_leak"
    MECHANICAL_FAILURE = "mechanical_failure"
    PERSON_IN_DANGER = "person_in_danger"
    RESTRICTED_ZONE = "restricted_zone"
    ENVIRONMENTAL = "environmental"
    OTHER = "other"


class EmergencySeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EmergencyStatus(str, Enum):
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESPONDING = "responding"
    CONTAINED = "contained"
    RESOLVED = "resolved"
    CLOSED = "closed"


# ============================================================
# EXCEPTIONS
# ============================================================

class EmergencyServiceError(Exception):
    pass


class EmergencyValidationError(EmergencyServiceError):
    pass


class EmergencyNotFoundError(EmergencyServiceError):
    pass


class EmergencyTransitionError(EmergencyServiceError):
    pass


# ============================================================
# HELPERS
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def validate_text(
    value: Any,
    field_name: str,
    max_length: int = 1000,
) -> str:

    if not isinstance(value, str):
        raise EmergencyValidationError(
            f"{field_name} must be a string"
        )

    cleaned = value.strip()

    if not cleaned:
        raise EmergencyValidationError(
            f"{field_name} cannot be empty"
        )

    if len(cleaned) > max_length:
        raise EmergencyValidationError(
            f"{field_name} exceeds maximum length"
        )

    return cleaned


def validate_probability(value: Any) -> float:
    if isinstance(value, bool):
        raise EmergencyValidationError(
            "Probability must be numeric"
        )

    try:
        probability = float(value)
    except (TypeError, ValueError) as exc:
        raise EmergencyValidationError(
            "Invalid probability"
        ) from exc

    import math

    if not math.isfinite(probability):
        raise EmergencyValidationError(
            "Probability must be finite"
        )

    if not 0 <= probability <= 1:
        raise EmergencyValidationError(
            "Probability must be between 0 and 1"
        )

    return probability


# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class EmergencyEvent:
    emergency_id: str
    company_id: str
    site_id: str
    zone_id: Optional[str]
    asset_id: Optional[str]

    emergency_type: str
    severity: str
    status: str

    title: str
    description: str
    source: str

    confidence: Optional[float] = None
    assigned_to: Optional[str] = None
    acknowledged_by: Optional[str] = None

    created_at: datetime = field(
        default_factory=utc_now
    )
    updated_at: datetime = field(
        default_factory=utc_now
    )

    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None

    escalation_level: int = 0
    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)

        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()

        return data


# ============================================================
# STATUS TRANSITIONS
# ============================================================

ALLOWED_TRANSITIONS = {
    EmergencyStatus.OPEN: {
        EmergencyStatus.ACKNOWLEDGED,
        EmergencyStatus.RESPONDING,
    },
    EmergencyStatus.ACKNOWLEDGED: {
        EmergencyStatus.RESPONDING,
    },
    EmergencyStatus.RESPONDING: {
        EmergencyStatus.CONTAINED,
    },
    EmergencyStatus.CONTAINED: {
        EmergencyStatus.RESOLVED,
        EmergencyStatus.RESPONDING,
    },
    EmergencyStatus.RESOLVED: {
        EmergencyStatus.CLOSED,
        EmergencyStatus.RESPONDING,
    },
    EmergencyStatus.CLOSED: set(),
}


# ============================================================
# EMERGENCY SERVICE
# ============================================================

class EmergencyService:
    """
    Emergency workflow orchestration.

    Expected repository methods:

    async create_emergency(data: dict)
    async get_emergency(emergency_id: str)
    async update_emergency(
        emergency_id: str,
        updates: dict
    )
    async list_emergencies(filters: dict)

    Optional notification adapter:

    async send_emergency_alert(data: dict)

    Optional audit adapter:

    async record_event(data: dict)

    Adapters may expose synchronous or async methods.
    """

    def __init__(
        self,
        emergency_repository: Any,
        notification_service: Any = None,
        audit_service: Any = None,
    ):

        if emergency_repository is None:
            raise ValueError(
                "emergency_repository is required"
            )

        self.repository = emergency_repository
        self.notifications = notification_service
        self.audit = audit_service

    @staticmethod
    async def _call(
        method: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:

        result = method(*args, **kwargs)

        if inspect.isawaitable(result):
            return await result

        return result

    async def _audit(
        self,
        *,
        emergency_id: str,
        action: str,
        actor_id: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:

        if self.audit is None:
            return

        method = getattr(
            self.audit,
            "record_event",
            None,
        )

        if method is None:
            return

        try:
            await self._call(
                method,
                {
                    "event_id": str(uuid4()),
                    "entity_type": "emergency",
                    "entity_id": emergency_id,
                    "action": action,
                    "actor_id": actor_id,
                    "details": details or {},
                    "timestamp": utc_now().isoformat(),
                },
            )
        except Exception:
            logger.exception(
                "Emergency audit delivery failed: %s",
                emergency_id,
            )

    async def _notify(
        self,
        event: dict[str, Any],
    ) -> None:

        if self.notifications is None:
            return

        method = getattr(
            self.notifications,
            "send_emergency_alert",
            None,
        )

        if method is None:
            return

        try:
            await self._call(method, event)
        except Exception:
            logger.exception(
                "Emergency notification failed: %s",
                event.get("emergency_id"),
            )

    @staticmethod
    def classify_severity(
        emergency_type: str | EmergencyType,
        confidence: Optional[float] = None,
    ) -> EmergencySeverity:
        """
        Conservative demonstration classification.

        AI confidence is not a safety severity measure.
        Real severity requires approved site-specific
        hazard rules and human verification.
        """

        try:
            kind = EmergencyType(emergency_type)
        except ValueError as exc:
            raise EmergencyValidationError(
                "Invalid emergency type"
            ) from exc

        if confidence is not None:
            validate_probability(confidence)

        critical_types = {
            EmergencyType.FIRE,
            EmergencyType.GAS_LEAK,
            EmergencyType.CHEMICAL_LEAK,
            EmergencyType.PERSON_IN_DANGER,
        }

        high_types = {
            EmergencyType.SMOKE,
            EmergencyType.ELECTRICAL,
            EmergencyType.OVERHEATING,
        }

        if kind in critical_types:
            return EmergencySeverity.CRITICAL

        if kind in high_types:
            return EmergencySeverity.HIGH

        if kind in {
            EmergencyType.MECHANICAL_FAILURE,
            EmergencyType.RESTRICTED_ZONE,
        }:
            return EmergencySeverity.MEDIUM

        return EmergencySeverity.LOW

    async def create_emergency(
        self,
        *,
        company_id: str,
        site_id: str,
        emergency_type: str | EmergencyType,
        title: str,
        description: str,
        source: str,
        zone_id: Optional[str] = None,
        asset_id: Optional[str] = None,
        severity: Optional[
            str | EmergencySeverity
        ] = None,
        confidence: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Any:

        company_id = validate_text(
            company_id, "company_id", 100
        )
        site_id = validate_text(
            site_id, "site_id", 100
        )

        title = validate_text(
            title, "title", 255
        )
        description = validate_text(
            description, "description", 5000
        )
        source = validate_text(
            source, "source", 100
        )

        try:
            kind = EmergencyType(emergency_type)
        except ValueError as exc:
            raise EmergencyValidationError(
                "Invalid emergency type"
            ) from exc

        if confidence is not None:
            confidence = validate_probability(
                confidence
            )

        if severity is None:
            resolved_severity = (
                self.classify_severity(
                    kind,
                    confidence,
                )
            )
        else:
            try:
                resolved_severity = (
                    EmergencySeverity(severity)
                )
            except ValueError as exc:
                raise EmergencyValidationError(
                    "Invalid emergency severity"
                ) from exc

        now = utc_now()

        event = EmergencyEvent(
            emergency_id=str(uuid4()),
            company_id=company_id,
            site_id=site_id,
            zone_id=zone_id,
            asset_id=asset_id,
            emergency_type=kind.value,
            severity=resolved_severity.value,
            status=EmergencyStatus.OPEN.value,
            title=title,
            description=description,
            source=source,
            confidence=confidence,
            created_at=now,
            updated_at=now,
            metadata=dict(metadata or {}),
        )

        payload = event.to_dict()

        created = await self._call(
            self.repository.create_emergency,
            payload,
        )

        await self._audit(
            emergency_id=event.emergency_id,
            action="created",
            details={
                "type": kind.value,
                "severity": resolved_severity.value,
            },
        )

        await self._notify(payload)

        logger.warning(
            "Emergency created id=%s type=%s "
            "severity=%s site=%s",
            event.emergency_id,
            kind.value,
            resolved_severity.value,
            site_id,
        )

        return created

    async def get_emergency(
        self,
        emergency_id: str,
    ) -> Any:

        emergency_id = validate_text(
            emergency_id,
            "emergency_id",
            100,
        )

        event = await self._call(
            self.repository.get_emergency,
            emergency_id,
        )

        if event is None:
            raise EmergencyNotFoundError(
                f"Emergency {emergency_id} not found"
            )

        return event

    @staticmethod
    def _field(
        event: Any,
        name: str,
        default: Any = None,
    ) -> Any:

        if isinstance(event, dict):
            return event.get(name, default)

        return getattr(event, name, default)

    async def transition_status(
        self,
        *,
        emergency_id: str,
        new_status: str | EmergencyStatus,
        actor_id: str,
        notes: Optional[str] = None,
    ) -> Any:

        event = await self.get_emergency(
            emergency_id
        )

        try:
            current = EmergencyStatus(
                self._field(event, "status")
            )
            target = EmergencyStatus(
                new_status
            )
        except ValueError as exc:
            raise EmergencyValidationError(
                "Invalid emergency status"
            ) from exc

        if target not in ALLOWED_TRANSITIONS[current]:
            raise EmergencyTransitionError(
                f"Transition {current.value} -> "
                f"{target.value} is not allowed"
            )

        actor_id = validate_text(
            actor_id, "actor_id", 100
        )

        updates: dict[str, Any] = {
            "status": target.value,
            "updated_at": utc_now().isoformat(),
        }

        if target == EmergencyStatus.ACKNOWLEDGED:
            updates["acknowledged_by"] = actor_id
            updates["acknowledged_at"] = (
                utc_now().isoformat()
            )

        if target == EmergencyStatus.RESOLVED:
            updates["resolved_at"] = (
                utc_now().isoformat()
            )

        if target == EmergencyStatus.CLOSED:
            updates["closed_at"] = (
                utc_now().isoformat()
            )

        updated = await self._call(
            self.repository.update_emergency,
            emergency_id,
            updates,
        )

        await self._audit(
            emergency_id=emergency_id,
            action="status_changed",
            actor_id=actor_id,
            details={
                "previous_status": current.value,
                "new_status": target.value,
                "notes": notes,
            },
        )

        if target in {
            EmergencyStatus.RESPONDING,
            EmergencyStatus.RESOLVED,
        }:
            await self._notify({
                "emergency_id": emergency_id,
                "status": target.value,
                "actor_id": actor_id,
            })

        return updated

    async def acknowledge_emergency(
        self,
        emergency_id: str,
        actor_id: str,
    ) -> Any:

        return await self.transition_status(
            emergency_id=emergency_id,
            new_status=(
                EmergencyStatus.ACKNOWLEDGED
            ),
            actor_id=actor_id,
        )

    async def start_response(
        self,
        emergency_id: str,
        actor_id: str,
    ) -> Any:

        return await self.transition_status(
            emergency_id=emergency_id,
            new_status=EmergencyStatus.RESPONDING,
            actor_id=actor_id,
        )

    async def contain_emergency(
        self,
        emergency_id: str,
        actor_id: str,
    ) -> Any:

        return await self.transition_status(
            emergency_id=emergency_id,
            new_status=EmergencyStatus.CONTAINED,
            actor_id=actor_id,
        )

    async def resolve_emergency(
        self,
        emergency_id: str,
        actor_id: str,
    ) -> Any:

        return await self.transition_status(
            emergency_id=emergency_id,
            new_status=EmergencyStatus.RESOLVED,
            actor_id=actor_id,
        )

    async def close_emergency(
        self,
        emergency_id: str,
        actor_id: str,
    ) -> Any:

        return await self.transition_status(
            emergency_id=emergency_id,
            new_status=EmergencyStatus.CLOSED,
            actor_id=actor_id,
        )

    async def assign_responder(
        self,
        *,
        emergency_id: str,
        responder_id: str,
        actor_id: str,
    ) -> Any:

        await self.get_emergency(emergency_id)

        responder_id = validate_text(
            responder_id,
            "responder_id",
            100,
        )

        updated = await self._call(
            self.repository.update_emergency,
            emergency_id,
            {
                "assigned_to": responder_id,
                "updated_at": utc_now().isoformat(),
            },
        )

        await self._audit(
            emergency_id=emergency_id,
            action="responder_assigned",
            actor_id=actor_id,
            details={
                "responder_id": responder_id,
            },
        )

        await self._notify({
            "emergency_id": emergency_id,
            "assigned_to": responder_id,
            "action": "responder_assigned",
        })

        return updated

    async def escalate_emergency(
        self,
        *,
        emergency_id: str,
        actor_id: str,
        reason: str,
    ) -> Any:

        event = await self.get_emergency(
            emergency_id
        )

        status = EmergencyStatus(
            self._field(event, "status")
        )

        if status == EmergencyStatus.CLOSED:
            raise EmergencyTransitionError(
                "Cannot escalate a closed emergency"
            )

        current_level = int(
            self._field(
                event,
                "escalation_level",
                0,
            )
        )

        next_level = min(
            current_level + 1,
            5,
        )

        reason = validate_text(
            reason, "reason", 1000
        )

        updated = await self._call(
            self.repository.update_emergency,
            emergency_id,
            {
                "escalation_level": next_level,
                "updated_at": utc_now().isoformat(),
            },
        )

        await self._audit(
            emergency_id=emergency_id,
            action="escalated",
            actor_id=actor_id,
            details={
                "escalation_level": next_level,
                "reason": reason,
            },
        )

        await self._notify({
            "emergency_id": emergency_id,
            "action": "escalation",
            "escalation_level": next_level,
            "reason": reason,
        })

        logger.warning(
            "Emergency escalated id=%s level=%s",
            emergency_id,
            next_level,
        )

        return updated

    async def list_emergencies(
        self,
        *,
        company_id: Optional[str] = None,
        site_id: Optional[str] = None,
        status: Optional[str] = None,
        severity: Optional[str] = None,
    ) -> Any:

        filters: dict[str, Any] = {}

        if company_id:
            filters["company_id"] = company_id

        if site_id:
            filters["site_id"] = site_id

        if status:
            filters["status"] = (
                EmergencyStatus(status).value
            )

        if severity:
            filters["severity"] = (
                EmergencySeverity(severity).value
            )

        return await self._call(
            self.repository.list_emergencies,
            filters,
        )

    async def get_active_emergencies(
        self,
        site_id: str,
    ) -> list[Any]:

        records = await self.list_emergencies(
            site_id=site_id
        )

        terminal = {
            EmergencyStatus.RESOLVED.value,
            EmergencyStatus.CLOSED.value,
        }

        return [
            record
            for record in records
            if self._field(
                record, "status"
            ) not in terminal
        ]


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":

    import asyncio
    import json

    class InMemoryEmergencyRepository:
        def __init__(self):
            self.records = {}

        async def create_emergency(self, data):
            self.records[data["emergency_id"]] = data
            return data

        async def get_emergency(self, emergency_id):
            return self.records.get(emergency_id)

        async def update_emergency(
            self,
            emergency_id,
            updates,
        ):
            record = self.records[emergency_id]
            record.update(updates)
            return record

        async def list_emergencies(self, filters):
            return [
                item
                for item in self.records.values()
                if all(
                    item.get(key) == value
                    for key, value in filters.items()
                )
            ]

    async def main():
        repository = InMemoryEmergencyRepository()

        service = EmergencyService(
            emergency_repository=repository
        )

        event = await service.create_emergency(
            company_id="COMPANY-001",
            site_id="SITE-001",
            zone_id="ZONE-A",
            asset_id="MACHINE-001",
            emergency_type="fire",
            title="Fire detected near machine",
            description=(
                "Camera reported visible flames "
                "near industrial equipment."
            ),
            source="computer_vision",
            confidence=0.96,
        )

        emergency_id = event["emergency_id"]

        await service.acknowledge_emergency(
            emergency_id,
            actor_id="SAFETY-OFFICER-001",
        )

        await service.start_response(
            emergency_id,
            actor_id="SAFETY-OFFICER-001",
        )

        print(
            json.dumps(
                await service.get_emergency(
                    emergency_id
                ),
                indent=2,
            )
        )

    asyncio.run(main())
