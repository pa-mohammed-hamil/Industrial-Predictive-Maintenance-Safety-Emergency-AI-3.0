
"""
app/services/incident_service.py

Industrial Predictive Maintenance, Safety & Emergency AI

Incident Management Service

Features:
- Incident creation and reporting
- Industrial incident classification
- Severity assessment
- Incident lifecycle management
- Investigator assignment
- Investigation findings
- Corrective action tracking
- Incident escalation
- Incident event timeline
- Incident statistics
- Optional notifications
- Optional audit logging
- Async repository integration

SAFETY:
This service supports incident workflows.
It does not replace certified safety systems,
emergency response procedures, or trained personnel.

Production deployments require RBAC, tenant isolation,
database transactions, and durable audit records.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Optional, Sequence
from uuid import uuid4


logger = logging.getLogger(__name__)


# ============================================================
# EXCEPTIONS
# ============================================================

class IncidentServiceError(Exception):
    """Base incident service exception."""


class IncidentValidationError(IncidentServiceError):
    """Invalid incident data."""


class IncidentNotFoundError(IncidentServiceError):
    """Incident record was not found."""


class IncidentTransitionError(IncidentServiceError):
    """Invalid incident status transition."""


class IncidentAuthorizationError(IncidentServiceError):
    """Actor is not authorized for the operation."""


# ============================================================
# ENUMS
# ============================================================

class IncidentType(str, Enum):
    MACHINE_FAILURE = "machine_failure"
    FIRE = "fire"
    SMOKE = "smoke"
    OVERHEATING = "overheating"
    ELECTRICAL = "electrical"
    GAS_LEAK = "gas_leak"
    CHEMICAL_LEAK = "chemical_leak"
    EQUIPMENT_DAMAGE = "equipment_damage"
    WORKPLACE_INJURY = "workplace_injury"
    NEAR_MISS = "near_miss"
    RESTRICTED_ZONE = "restricted_zone"
    UNSAFE_PROXIMITY = "unsafe_proximity"
    ENVIRONMENTAL = "environmental"
    OTHER = "other"


class IncidentSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentStatus(str, Enum):
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    INVESTIGATING = "investigating"
    ACTION_REQUIRED = "action_required"
    MONITORING = "monitoring"
    RESOLVED = "resolved"
    CLOSED = "closed"


class CorrectiveActionStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    VERIFIED = "verified"


class IncidentSource(str, Enum):
    MANUAL = "manual"
    SENSOR = "sensor"
    COMPUTER_VISION = "computer_vision"
    AI_MODEL = "ai_model"
    SAFETY_ENGINE = "safety_engine"
    EXTERNAL_SYSTEM = "external_system"


# ============================================================
# HELPERS
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def required_text(
    value: Any,
    name: str,
    max_length: int = 1000,
) -> str:

    if not isinstance(value, str):
        raise IncidentValidationError(
            f"{name} must be a string"
        )

    cleaned = value.strip()

    if not cleaned:
        raise IncidentValidationError(
            f"{name} cannot be empty"
        )

    if len(cleaned) > max_length:
        raise IncidentValidationError(
            f"{name} exceeds {max_length} characters"
        )

    return cleaned


def optional_text(
    value: Optional[str],
    name: str,
    max_length: int = 1000,
) -> Optional[str]:

    if value is None:
        return None

    return required_text(
        value,
        name,
        max_length,
    )


def parse_enum(
    enum_class: type[Enum],
    value: Any,
    field_name: str,
) -> Enum:

    try:
        return enum_class(value)
    except (ValueError, TypeError) as exc:
        raise IncidentValidationError(
            f"Invalid {field_name}: {value}"
        ) from exc


def serialize(value: Any) -> Any:
    """Convert service data into JSON-compatible values."""

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if hasattr(value, "__dataclass_fields__"):
        return serialize(asdict(value))

    if isinstance(value, dict):
        return {
            str(key): serialize(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            serialize(item)
            for item in value
        ]

    return value


async def invoke(
    method: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:

    result = method(*args, **kwargs)

    if inspect.isawaitable(result):
        return await result

    return result


# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class CorrectiveAction:
    action_id: str
    description: str
    assigned_to: str
    status: str = CorrectiveActionStatus.OPEN.value
    due_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    verified_at: Optional[datetime] = None
    verified_by: Optional[str] = None
    created_at: datetime = field(
        default_factory=utc_now
    )


@dataclass
class IncidentEvent:
    event_id: str
    action: str
    actor_id: str
    timestamp: datetime
    details: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class IncidentRecord:
    incident_id: str
    company_id: str
    site_id: str
    incident_type: str
    severity: str
    status: str
    title: str
    description: str
    source: str
    reported_by: str

    zone_id: Optional[str] = None
    asset_id: Optional[str] = None
    emergency_id: Optional[str] = None

    investigator_id: Optional[str] = None
    root_cause: Optional[str] = None
    investigation_findings: Optional[str] = None

    escalation_level: int = 0

    corrective_actions: list[CorrectiveAction] = field(
        default_factory=list
    )

    timeline: list[IncidentEvent] = field(
        default_factory=list
    )

    occurred_at: datetime = field(
        default_factory=utc_now
    )

    created_at: datetime = field(
        default_factory=utc_now
    )

    updated_at: datetime = field(
        default_factory=utc_now
    )

    resolved_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


# ============================================================
# WORKFLOW RULES
# ============================================================

ALLOWED_TRANSITIONS = {
    IncidentStatus.OPEN: {
        IncidentStatus.ACKNOWLEDGED,
        IncidentStatus.INVESTIGATING,
    },
    IncidentStatus.ACKNOWLEDGED: {
        IncidentStatus.INVESTIGATING,
    },
    IncidentStatus.INVESTIGATING: {
        IncidentStatus.ACTION_REQUIRED,
        IncidentStatus.MONITORING,
    },
    IncidentStatus.ACTION_REQUIRED: {
        IncidentStatus.MONITORING,
        IncidentStatus.INVESTIGATING,
    },
    IncidentStatus.MONITORING: {
        IncidentStatus.RESOLVED,
        IncidentStatus.INVESTIGATING,
        IncidentStatus.ACTION_REQUIRED,
    },
    IncidentStatus.RESOLVED: {
        IncidentStatus.CLOSED,
        IncidentStatus.INVESTIGATING,
    },
    IncidentStatus.CLOSED: set(),
}


ACTION_TRANSITIONS = {
    CorrectiveActionStatus.OPEN: {
        CorrectiveActionStatus.IN_PROGRESS,
    },
    CorrectiveActionStatus.IN_PROGRESS: {
        CorrectiveActionStatus.COMPLETED,
    },
    CorrectiveActionStatus.COMPLETED: {
        CorrectiveActionStatus.VERIFIED,
        CorrectiveActionStatus.IN_PROGRESS,
    },
    CorrectiveActionStatus.VERIFIED: set(),
}


# ============================================================
# INCIDENT SERVICE
# ============================================================

class IncidentService:
    """
    Incident management application service.

    Expected repository methods:

        async create_incident(data: dict)
        async get_incident(incident_id: str)
        async update_incident(
            incident_id: str,
            updates: dict
        )
        async list_incidents(filters: dict)

    Optional notification service:

        async send_incident_alert(data: dict)

    Optional audit service:

        async record_event(data: dict)

    Authorization and tenant access must be enforced
    by the application boundary and repository.
    """

    def __init__(
        self,
        incident_repository: Any,
        notification_service: Any = None,
        audit_service: Any = None,
    ):

        if incident_repository is None:
            raise IncidentValidationError(
                "incident_repository is required"
            )

        self.repository = incident_repository
        self.notifications = notification_service
        self.audit = audit_service

    # --------------------------------------------------------
    # RECORD ACCESS
    # --------------------------------------------------------

    @staticmethod
    def _field(
        record: Any,
        key: str,
        default: Any = None,
    ) -> Any:

        if isinstance(record, Mapping):
            return record.get(key, default)

        return getattr(record, key, default)

    async def get_incident(
        self,
        incident_id: str,
    ) -> Any:

        incident_id = required_text(
            incident_id,
            "incident_id",
            100,
        )

        record = await invoke(
            self.repository.get_incident,
            incident_id,
        )

        if record is None:
            raise IncidentNotFoundError(
                f"Incident not found: {incident_id}"
            )

        return record

    # --------------------------------------------------------
    # SEVERITY CLASSIFICATION
    # --------------------------------------------------------

    @staticmethod
    def classify_severity(
        incident_type: str | IncidentType,
    ) -> IncidentSeverity:
        """
        Illustrative conservative classification.

        Real severity must consider actual harm,
        exposure, consequence and site policies.
        """

        kind = parse_enum(
            IncidentType,
            incident_type,
            "incident_type",
        )

        critical = {
            IncidentType.FIRE,
            IncidentType.GAS_LEAK,
            IncidentType.CHEMICAL_LEAK,
            IncidentType.WORKPLACE_INJURY,
        }

        high = {
            IncidentType.SMOKE,
            IncidentType.OVERHEATING,
            IncidentType.ELECTRICAL,
            IncidentType.UNSAFE_PROXIMITY,
        }

        medium = {
            IncidentType.MACHINE_FAILURE,
            IncidentType.EQUIPMENT_DAMAGE,
            IncidentType.RESTRICTED_ZONE,
        }

        if kind in critical:
            return IncidentSeverity.CRITICAL

        if kind in high:
            return IncidentSeverity.HIGH

        if kind in medium:
            return IncidentSeverity.MEDIUM

        return IncidentSeverity.LOW

    # --------------------------------------------------------
    # AUDIT AND NOTIFICATION
    # --------------------------------------------------------

    async def _audit(
        self,
        *,
        incident_id: str,
        action: str,
        actor_id: str,
        details: Optional[
            dict[str, Any]
        ] = None,
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
            await invoke(
                method,
                {
                    "event_id": str(uuid4()),
                    "entity_type": "incident",
                    "entity_id": incident_id,
                    "action": action,
                    "actor_id": actor_id,
                    "details": details or {},
                    "timestamp": utc_now().isoformat(),
                },
            )
        except Exception:
            logger.exception(
                "Incident audit delivery failed: %s",
                incident_id,
            )

    async def _notify(
        self,
        payload: dict[str, Any],
    ) -> None:

        if self.notifications is None:
            return

        method = getattr(
            self.notifications,
            "send_incident_alert",
            None,
        )

        if method is None:
            return

        try:
            await invoke(method, payload)
        except Exception:
            logger.exception(
                "Incident notification failed: %s",
                payload.get("incident_id"),
            )

    @staticmethod
    def _timeline_event(
        action: str,
        actor_id: str,
        details: Optional[
            dict[str, Any]
        ] = None,
    ) -> dict[str, Any]:

        return {
            "event_id": str(uuid4()),
            "action": action,
            "actor_id": actor_id,
            "timestamp": utc_now().isoformat(),
            "details": details or {},
        }

    # --------------------------------------------------------
    # CREATE INCIDENT
    # --------------------------------------------------------

    async def create_incident(
        self,
        *,
        company_id: str,
        site_id: str,
        incident_type: str | IncidentType,
        title: str,
        description: str,
        reported_by: str,
        source: str | IncidentSource = (
            IncidentSource.MANUAL
        ),
        severity: Optional[
            str | IncidentSeverity
        ] = None,
        zone_id: Optional[str] = None,
        asset_id: Optional[str] = None,
        emergency_id: Optional[str] = None,
        occurred_at: Optional[datetime] = None,
        metadata: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> Any:

        company_id = required_text(
            company_id, "company_id", 100
        )

        site_id = required_text(
            site_id, "site_id", 100
        )

        title = required_text(
            title, "title", 255
        )

        description = required_text(
            description, "description", 5000
        )

        reported_by = required_text(
            reported_by, "reported_by", 100
        )

        kind = parse_enum(
            IncidentType,
            incident_type,
            "incident_type",
        )

        incident_source = parse_enum(
            IncidentSource,
            source,
            "source",
        )

        resolved_severity = (
            parse_enum(
                IncidentSeverity,
                severity,
                "severity",
            )
            if severity is not None
            else self.classify_severity(kind)
        )

        if occurred_at is None:
            occurred_at = utc_now()

        if (
            occurred_at.tzinfo is None
            or occurred_at.utcoffset() is None
        ):
            raise IncidentValidationError(
                "occurred_at must be timezone-aware"
            )

        incident_id = str(uuid4())

        timeline = [
            IncidentEvent(
                event_id=str(uuid4()),
                action="incident_created",
                actor_id=reported_by,
                timestamp=utc_now(),
                details={
                    "incident_type": kind.value,
                    "severity": resolved_severity.value,
                },
            )
        ]

        record = IncidentRecord(
            incident_id=incident_id,
            company_id=company_id,
            site_id=site_id,
            zone_id=zone_id,
            asset_id=asset_id,
            emergency_id=emergency_id,
            incident_type=kind.value,
            severity=resolved_severity.value,
            status=IncidentStatus.OPEN.value,
            title=title,
            description=description,
            source=incident_source.value,
            reported_by=reported_by,
            occurred_at=occurred_at,
            timeline=timeline,
            metadata=dict(metadata or {}),
        )

        created = await invoke(
            self.repository.create_incident,
            record.to_dict(),
        )

        await self._audit(
            incident_id=incident_id,
            action="incident_created",
            actor_id=reported_by,
            details={
                "severity": resolved_severity.value,
                "type": kind.value,
            },
        )

        await self._notify({
            "incident_id": incident_id,
            "company_id": company_id,
            "site_id": site_id,
            "severity": resolved_severity.value,
            "incident_type": kind.value,
            "status": IncidentStatus.OPEN.value,
            "title": title,
        })

        logger.warning(
            "Incident created id=%s type=%s "
            "severity=%s",
            incident_id,
            kind.value,
            resolved_severity.value,
        )

        return created

    # --------------------------------------------------------
    # STATUS TRANSITIONS
    # --------------------------------------------------------

    async def change_status(
        self,
        *,
        incident_id: str,
        new_status: str | IncidentStatus,
        actor_id: str,
        notes: Optional[str] = None,
    ) -> Any:

        incident = await self.get_incident(
            incident_id
        )

        current = parse_enum(
            IncidentStatus,
            self._field(incident, "status"),
            "current_status",
        )

        target = parse_enum(
            IncidentStatus,
            new_status,
            "new_status",
        )

        if target not in ALLOWED_TRANSITIONS[current]:
            raise IncidentTransitionError(
                f"Cannot transition from "
                f"{current.value} to {target.value}"
            )

        actor_id = required_text(
            actor_id, "actor_id", 100
        )

        if target in {
            IncidentStatus.RESOLVED,
            IncidentStatus.CLOSED,
        }:
            actions = self._field(
                incident,
                "corrective_actions",
                [],
            )

            for action in actions:
                action_status = self._field(
                    action,
                    "status",
                )

                if (
                    action_status
                    != CorrectiveActionStatus.VERIFIED.value
                ):
                    raise IncidentTransitionError(
                        "All corrective actions must "
                        "be verified before resolution "
                        "or closure"
                    )

        now = utc_now()

        updates: dict[str, Any] = {
            "status": target.value,
            "updated_at": now.isoformat(),
        }

        if target == IncidentStatus.RESOLVED:
            updates["resolved_at"] = (
                now.isoformat()
            )

        if target == IncidentStatus.CLOSED:
            updates["closed_at"] = (
                now.isoformat()
            )

        timeline = list(
            self._field(
                incident,
                "timeline",
                [],
            )
        )

        timeline.append(
            self._timeline_event(
                action="status_changed",
                actor_id=actor_id,
                details={
                    "previous_status": current.value,
                    "new_status": target.value,
                    "notes": notes,
                },
            )
        )

        updates["timeline"] = [
            serialize(event)
            for event in timeline
        ]

        updated = await invoke(
            self.repository.update_incident,
            incident_id,
            updates,
        )

        await self._audit(
            incident_id=incident_id,
            action="status_changed",
            actor_id=actor_id,
            details={
                "from": current.value,
                "to": target.value,
            },
        )

        return updated

    # --------------------------------------------------------
    # INVESTIGATOR ASSIGNMENT
    # --------------------------------------------------------

    async def assign_investigator(
        self,
        *,
        incident_id: str,
        investigator_id: str,
        actor_id: str,
    ) -> Any:

        incident = await self.get_incident(
            incident_id
        )

        investigator_id = required_text(
            investigator_id,
            "investigator_id",
            100,
        )

        actor_id = required_text(
            actor_id,
            "actor_id",
            100,
        )

        if (
            self._field(incident, "status")
            == IncidentStatus.CLOSED.value
        ):
            raise IncidentTransitionError(
                "Cannot modify a closed incident"
            )

        timeline = list(
            self._field(
                incident,
                "timeline",
                [],
            )
        )

        timeline.append(
            self._timeline_event(
                "investigator_assigned",
                actor_id,
                {
                    "investigator_id": investigator_id,
                },
            )
        )

        updated = await invoke(
            self.repository.update_incident,
            incident_id,
            {
                "investigator_id": investigator_id,
                "timeline": [
                    serialize(item)
                    for item in timeline
                ],
                "updated_at": utc_now().isoformat(),
            },
        )

        await self._audit(
            incident_id=incident_id,
            action="investigator_assigned",
            actor_id=actor_id,
            details={
                "investigator_id": investigator_id,
            },
        )

        return updated

    # --------------------------------------------------------
    # INVESTIGATION FINDINGS
    # --------------------------------------------------------

    async def record_investigation(
        self,
        *,
        incident_id: str,
        investigator_id: str,
        root_cause: str,
        findings: str,
    ) -> Any:

        incident = await self.get_incident(
            incident_id
        )

        root_cause = required_text(
            root_cause,
            "root_cause",
            3000,
        )

        findings = required_text(
            findings,
            "findings",
            10000,
        )

        investigator_id = required_text(
            investigator_id,
            "investigator_id",
            100,
        )

        if (
            self._field(incident, "status")
            == IncidentStatus.CLOSED.value
        ):
            raise IncidentTransitionError(
                "Cannot investigate a closed incident"
            )

        timeline = list(
            self._field(
                incident,
                "timeline",
                [],
            )
        )

        timeline.append(
            self._timeline_event(
                "investigation_recorded",
                investigator_id,
                {
                    "root_cause": root_cause,
                },
            )
        )

        updated = await invoke(
            self.repository.update_incident,
            incident_id,
            {
                "root_cause": root_cause,
                "investigation_findings": findings,
                "investigator_id": investigator_id,
                "timeline": [
                    serialize(item)
                    for item in timeline
                ],
                "updated_at": utc_now().isoformat(),
            },
        )

        await self._audit(
            incident_id=incident_id,
            action="investigation_recorded",
            actor_id=investigator_id,
        )

        return updated

    # --------------------------------------------------------
    # CORRECTIVE ACTIONS
    # --------------------------------------------------------

    async def add_corrective_action(
        self,
        *,
        incident_id: str,
        description: str,
        assigned_to: str,
        actor_id: str,
        due_at: Optional[datetime] = None,
    ) -> Any:

        incident = await self.get_incident(
            incident_id
        )

        if (
            self._field(incident, "status")
            == IncidentStatus.CLOSED.value
        ):
            raise IncidentTransitionError(
                "Cannot add actions to closed incident"
            )

        if due_at is not None and (
            due_at.tzinfo is None
            or due_at.utcoffset() is None
        ):
            raise IncidentValidationError(
                "due_at must be timezone-aware"
            )

        action = CorrectiveAction(
            action_id=str(uuid4()),
            description=required_text(
                description,
                "description",
                3000,
            ),
            assigned_to=required_text(
                assigned_to,
                "assigned_to",
                100,
            ),
            due_at=due_at,
        )

        actions = [
            serialize(item)
            for item in self._field(
                incident,
                "corrective_actions",
                [],
            )
        ]

        actions.append(
            serialize(action)
        )

        timeline = [
            serialize(item)
            for item in self._field(
                incident,
                "timeline",
                [],
            )
        ]

        timeline.append(
            self._timeline_event(
                "corrective_action_created",
                actor_id,
                {
                    "action_id": action.action_id,
                    "assigned_to": assigned_to,
                },
            )
        )

        updated = await invoke(
            self.repository.update_incident,
            incident_id,
            {
                "corrective_actions": actions,
                "timeline": timeline,
                "updated_at": utc_now().isoformat(),
            },
        )

        await self._audit(
            incident_id=incident_id,
            action="corrective_action_created",
            actor_id=actor_id,
            details={
                "action_id": action.action_id,
            },
        )

        return updated

    async def update_corrective_action(
        self,
        *,
        incident_id: str,
        action_id: str,
        new_status: str | CorrectiveActionStatus,
        actor_id: str,
    ) -> Any:

        incident = await self.get_incident(
            incident_id
        )

        if (
            self._field(incident, "status")
            == IncidentStatus.CLOSED.value
        ):
            raise IncidentTransitionError(
                "Cannot modify a closed incident"
            )

        target = parse_enum(
            CorrectiveActionStatus,
            new_status,
            "new_status",
        )

        actions = [
            serialize(item)
            for item in self._field(
                incident,
                "corrective_actions",
                [],
            )
        ]

        matching = next(
            (
                item
                for item in actions
                if item["action_id"] == action_id
            ),
            None,
        )

        if matching is None:
            raise IncidentNotFoundError(
                f"Corrective action not found: "
                f"{action_id}"
            )

        current = parse_enum(
            CorrectiveActionStatus,
            matching["status"],
            "current_action_status",
        )

        if target not in ACTION_TRANSITIONS[current]:
            raise IncidentTransitionError(
                f"Invalid action transition: "
                f"{current.value} -> {target.value}"
            )

        now = utc_now().isoformat()

        matching["status"] = target.value

        if target == CorrectiveActionStatus.COMPLETED:
            matching["completed_at"] = now

        if target == CorrectiveActionStatus.VERIFIED:
            matching["verified_at"] = now
            matching["verified_by"] = actor_id

        if target == CorrectiveActionStatus.IN_PROGRESS:
            matching["completed_at"] = None
            matching["verified_at"] = None
            matching["verified_by"] = None

        timeline = [
            serialize(item)
            for item in self._field(
                incident,
                "timeline",
                [],
            )
        ]

        timeline.append(
            self._timeline_event(
                "corrective_action_updated",
                actor_id,
                {
                    "action_id": action_id,
                    "status": target.value,
                },
            )
        )

        updated = await invoke(
            self.repository.update_incident,
            incident_id,
            {
                "corrective_actions": actions,
                "timeline": timeline,
                "updated_at": now,
            },
        )

        await self._audit(
            incident_id=incident_id,
            action="corrective_action_updated",
            actor_id=actor_id,
            details={
                "action_id": action_id,
                "status": target.value,
            },
        )

        return updated

    # --------------------------------------------------------
    # ESCALATION
    # --------------------------------------------------------

    async def escalate_incident(
        self,
        *,
        incident_id: str,
        actor_id: str,
        reason: str,
    ) -> Any:

        incident = await self.get_incident(
            incident_id
        )

        if (
            self._field(incident, "status")
            == IncidentStatus.CLOSED.value
        ):
            raise IncidentTransitionError(
                "Cannot escalate a closed incident"
            )

        reason = required_text(
            reason,
            "reason",
            2000,
        )

        current_level = int(
            self._field(
                incident,
                "escalation_level",
                0,
            )
        )

        next_level = min(
            current_level + 1,
            5,
        )

        timeline = [
            serialize(item)
            for item in self._field(
                incident,
                "timeline",
                [],
            )
        ]

        timeline.append(
            self._timeline_event(
                "incident_escalated",
                actor_id,
                {
                    "level": next_level,
                    "reason": reason,
                },
            )
        )

        updated = await invoke(
            self.repository.update_incident,
            incident_id,
            {
                "escalation_level": next_level,
                "timeline": timeline,
                "updated_at": utc_now().isoformat(),
            },
        )

        await self._audit(
            incident_id=incident_id,
            action="incident_escalated",
            actor_id=actor_id,
            details={
                "level": next_level,
                "reason": reason,
            },
        )

        await self._notify({
            "incident_id": incident_id,
            "action": "escalation",
            "escalation_level": next_level,
            "reason": reason,
        })

        return updated

    # --------------------------------------------------------
    # LIST INCIDENTS
    # --------------------------------------------------------

    async def list_incidents(
        self,
        *,
        company_id: Optional[str] = None,
        site_id: Optional[str] = None,
        status: Optional[str] = None,
        severity: Optional[str] = None,
        incident_type: Optional[str] = None,
    ) -> Any:

        filters = {}

        if company_id:
            filters["company_id"] = company_id

        if site_id:
            filters["site_id"] = site_id

        if status:
            filters["status"] = parse_enum(
                IncidentStatus,
                status,
                "status",
            ).value

        if severity:
            filters["severity"] = parse_enum(
                IncidentSeverity,
                severity,
                "severity",
            ).value

        if incident_type:
            filters["incident_type"] = parse_enum(
                IncidentType,
                incident_type,
                "incident_type",
            ).value

        return await invoke(
            self.repository.list_incidents,
            filters,
        )

    async def get_active_incidents(
        self,
        *,
        site_id: str,
    ) -> list[Any]:

        incidents = await self.list_incidents(
            site_id=site_id
        )

        terminal = {
            IncidentStatus.RESOLVED.value,
            IncidentStatus.CLOSED.value,
        }

        return [
            incident
            for incident in incidents
            if self._field(
                incident,
                "status",
            ) not in terminal
        ]

    # --------------------------------------------------------
    # INCIDENT STATISTICS
    # --------------------------------------------------------

    async def get_incident_statistics(
        self,
        *,
        company_id: Optional[str] = None,
        site_id: Optional[str] = None,
    ) -> dict[str, Any]:

        incidents = await self.list_incidents(
            company_id=company_id,
            site_id=site_id,
        )

        by_status = {
            item.value: 0
            for item in IncidentStatus
        }

        by_severity = {
            item.value: 0
            for item in IncidentSeverity
        }

        by_type = {
            item.value: 0
            for item in IncidentType
        }

        for incident in incidents:

            status = self._field(
                incident, "status"
            )

            severity = self._field(
                incident, "severity"
            )

            incident_type = self._field(
                incident, "incident_type"
            )

            if status in by_status:
                by_status[status] += 1

            if severity in by_severity:
                by_severity[severity] += 1

            if incident_type in by_type:
                by_type[incident_type] += 1

        active = sum(
            count
            for status, count in by_status.items()
            if status not in {
                IncidentStatus.RESOLVED.value,
                IncidentStatus.CLOSED.value,
            }
        )

        return {
            "total_incidents": len(incidents),
            "active_incidents": active,
            "by_status": by_status,
            "by_severity": by_severity,
            "by_type": by_type,
            "generated_at": utc_now().isoformat(),
        }


# ============================================================
# IN-MEMORY DEMONSTRATION REPOSITORY
# ============================================================

class InMemoryIncidentRepository:
    """
    Demonstration repository.

    Not suitable for concurrent production writes.
    """

    def __init__(self):
        self.records: dict[str, dict] = {}

    async def create_incident(
        self,
        data: dict,
    ) -> dict:

        self.records[data["incident_id"]] = data
        return data

    async def get_incident(
        self,
        incident_id: str,
    ) -> Optional[dict]:

        return self.records.get(
            incident_id
        )

    async def update_incident(
        self,
        incident_id: str,
        updates: dict,
    ) -> dict:

        if incident_id not in self.records:
            raise IncidentNotFoundError(
                "Incident not found"
            )

        self.records[incident_id].update(
            updates
        )

        return self.records[incident_id]

    async def list_incidents(
        self,
        filters: dict,
    ) -> list[dict]:

        return [
            incident
            for incident in self.records.values()
            if all(
                incident.get(key) == value
                for key, value in filters.items()
            )
        ]


# ============================================================
# DEMONSTRATION
# ============================================================

if __name__ == "__main__":

    import asyncio
    import json

    async def main():

        repository = InMemoryIncidentRepository()

        service = IncidentService(
            incident_repository=repository
        )

        incident = await service.create_incident(
            company_id="COMPANY-001",
            site_id="SITE-001",
            zone_id="ZONE-A",
            asset_id="MACHINE-001",
            incident_type="overheating",
            title="Industrial motor overheating",
            description=(
                "Motor temperature exceeded "
                "the configured warning limit."
            ),
            reported_by="SENSOR-SYSTEM",
            source="sensor",
        )

        incident_id = incident["incident_id"]

        await service.assign_investigator(
            incident_id=incident_id,
            investigator_id="ENGINEER-001",
            actor_id="SUPERVISOR-001",
        )

        await service.change_status(
            incident_id=incident_id,
            new_status="investigating",
            actor_id="ENGINEER-001",
        )

        await service.record_investigation(
            incident_id=incident_id,
            investigator_id="ENGINEER-001",
            root_cause=(
                "Possible cooling system malfunction"
            ),
            findings=(
                "Temperature readings require "
                "physical inspection and verification."
            ),
        )

        await service.add_corrective_action(
            incident_id=incident_id,
            description=(
                "Inspect cooling fans and "
                "temperature sensors."
            ),
            assigned_to="TECHNICIAN-001",
            actor_id="ENGINEER-001",
        )

        print(
            json.dumps(
                await service.get_incident(
                    incident_id
                ),
                indent=2,
            )
        )

    asyncio.run(main())
