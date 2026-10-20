"""
app/services/safety_service.py

Industrial Predictive Maintenance, Safety & Emergency AI
=========================================================

Safety monitoring and incident coordination service.

Features:
    - Safety event detection and classification
    - Fire, smoke, overheating, electrical and motion hazards
    - Restricted-zone and unsafe-proximity events
    - Severity classification
    - Risk scoring
    - Safety rule evaluation
    - Event lifecycle management
    - Acknowledgment and investigation
    - Escalation
    - Optional notification integration
    - Optional incident integration
    - Optional emergency integration
    - Optional audit logging
    - Safety statistics
    - In-memory development repository

Expected repository interface:
    create_safety_event(data)
    get_safety_event(event_id)
    update_safety_event(event_id, updates)
    list_safety_events(filters)

Optional integrations:
    notification_service.send_safety_alert(data)
    incident_service.create_incident(...)
    emergency_service.create_emergency(...)
    audit_service.record_event(data)

Safety limitations:
    This software is advisory and is not a certified
    industrial safety system.

    It must not independently authorize equipment restart,
    override safety interlocks, actuate machinery or replace
    emergency procedures.

    Critical notifications must be handled by separately
    validated and monitored delivery infrastructure.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import logging
import math

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Optional, Sequence
from uuid import uuid4


logger = logging.getLogger(__name__)


# ============================================================
# EXCEPTIONS
# ============================================================

class SafetyServiceError(Exception):
    """Base safety service exception."""


class SafetyValidationError(SafetyServiceError):
    """Invalid safety event input."""


class SafetyEventNotFoundError(SafetyServiceError):
    """Safety event was not found."""


class SafetyTransitionError(SafetyServiceError):
    """Invalid safety event status transition."""


class SafetyRepositoryError(SafetyServiceError):
    """Safety event persistence failed."""


class SafetyAuthorizationError(SafetyServiceError):
    """Safety action is not authorized."""


# ============================================================
# ENUMS
# ============================================================

class SafetyEventType(str, Enum):
    FIRE = "fire"
    SMOKE = "smoke"
    OVERHEATING = "overheating"
    ELECTRICAL_HAZARD = "electrical_hazard"
    GAS_LEAK = "gas_leak"
    CHEMICAL_LEAK = "chemical_leak"
    PRESSURE_HAZARD = "pressure_hazard"
    VIBRATION_HAZARD = "vibration_hazard"
    UNSAFE_PROXIMITY = "unsafe_proximity"
    RESTRICTED_ZONE = "restricted_zone"
    PERSON_DETECTED = "person_detected"
    FALL_DETECTED = "fall_detected"
    MOTION_ANOMALY = "motion_anomaly"
    PPE_VIOLATION = "ppe_violation"
    MACHINE_GUARD_VIOLATION = "machine_guard_violation"
    COLLISION_RISK = "collision_risk"
    EQUIPMENT_FAILURE = "equipment_failure"
    EMERGENCY_STOP = "emergency_stop"
    OTHER = "other"


class SafetySeverity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SafetyStatus(str, Enum):
    DETECTED = "detected"
    ACKNOWLEDGED = "acknowledged"
    INVESTIGATING = "investigating"
    MITIGATED = "mitigated"
    RESOLVED = "resolved"
    CLOSED = "closed"
    FALSE_POSITIVE = "false_positive"


class SafetySource(str, Enum):
    SENSOR = "sensor"
    CAMERA = "camera"
    COMPUTER_VISION = "computer_vision"
    TELEMETRY = "telemetry"
    MULTIMODAL_FUSION = "multimodal_fusion"
    SAFETY_RULE = "safety_rule"
    MANUAL = "manual"
    EXTERNAL_SYSTEM = "external_system"


class SafetyRiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class SafetyRuleOperator(str, Enum):
    GREATER_THAN = "gt"
    GREATER_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_EQUAL = "lte"
    EQUAL = "eq"
    NOT_EQUAL = "neq"


# ============================================================
# CONSTANTS
# ============================================================

SAFETY_TRANSITIONS = {
    SafetyStatus.DETECTED: {
        SafetyStatus.ACKNOWLEDGED,
        SafetyStatus.INVESTIGATING,
        SafetyStatus.FALSE_POSITIVE,
    },
    SafetyStatus.ACKNOWLEDGED: {
        SafetyStatus.INVESTIGATING,
        SafetyStatus.MITIGATED,
        SafetyStatus.FALSE_POSITIVE,
    },
    SafetyStatus.INVESTIGATING: {
        SafetyStatus.MITIGATED,
        SafetyStatus.FALSE_POSITIVE,
    },
    SafetyStatus.MITIGATED: {
        SafetyStatus.RESOLVED,
        SafetyStatus.INVESTIGATING,
    },
    SafetyStatus.RESOLVED: {
        SafetyStatus.CLOSED,
        SafetyStatus.INVESTIGATING,
    },
    SafetyStatus.CLOSED: set(),
    SafetyStatus.FALSE_POSITIVE: set(),
}

SEVERITY_SCORES = {
    SafetySeverity.INFO: 0,
    SafetySeverity.LOW: 20,
    SafetySeverity.MEDIUM: 45,
    SafetySeverity.HIGH: 75,
    SafetySeverity.CRITICAL: 100,
}

DEFAULT_EVENT_SEVERITIES = {
    SafetyEventType.FIRE: SafetySeverity.CRITICAL,
    SafetyEventType.SMOKE: SafetySeverity.HIGH,
    SafetyEventType.OVERHEATING: SafetySeverity.HIGH,
    SafetyEventType.ELECTRICAL_HAZARD: SafetySeverity.HIGH,
    SafetyEventType.GAS_LEAK: SafetySeverity.CRITICAL,
    SafetyEventType.CHEMICAL_LEAK: SafetySeverity.CRITICAL,
    SafetyEventType.PRESSURE_HAZARD: SafetySeverity.HIGH,
    SafetyEventType.VIBRATION_HAZARD: SafetySeverity.MEDIUM,
    SafetyEventType.UNSAFE_PROXIMITY: SafetySeverity.HIGH,
    SafetyEventType.RESTRICTED_ZONE: SafetySeverity.HIGH,
    SafetyEventType.PERSON_DETECTED: SafetySeverity.INFO,
    SafetyEventType.FALL_DETECTED: SafetySeverity.CRITICAL,
    SafetyEventType.MOTION_ANOMALY: SafetySeverity.MEDIUM,
    SafetyEventType.PPE_VIOLATION: SafetySeverity.HIGH,
    SafetyEventType.MACHINE_GUARD_VIOLATION: SafetySeverity.CRITICAL,
    SafetyEventType.COLLISION_RISK: SafetySeverity.CRITICAL,
    SafetyEventType.EQUIPMENT_FAILURE: SafetySeverity.HIGH,
    SafetyEventType.EMERGENCY_STOP: SafetySeverity.CRITICAL,
    SafetyEventType.OTHER: SafetySeverity.MEDIUM,
}


# ============================================================
# HELPERS
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def validate_text(
    value: Any,
    name: str,
    max_length: int = 255,
) -> str:
    if not isinstance(value, str):
        raise SafetyValidationError(
            f"{name} must be a string"
        )

    result = value.strip()

    if not result:
        raise SafetyValidationError(
            f"{name} cannot be empty"
        )

    if len(result) > max_length:
        raise SafetyValidationError(
            f"{name} exceeds {max_length} characters"
        )

    return result


def optional_text(
    value: Optional[str],
    name: str,
) -> Optional[str]:
    if value is None:
        return None

    return validate_text(value, name)


def validate_number(
    value: Any,
    name: str,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    if isinstance(value, bool):
        raise SafetyValidationError(
            f"{name} must be numeric"
        )

    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SafetyValidationError(
            f"{name} must be numeric"
        ) from exc

    if not math.isfinite(result):
        raise SafetyValidationError(
            f"{name} must be finite"
        )

    if minimum is not None and result < minimum:
        raise SafetyValidationError(
            f"{name} must be >= {minimum}"
        )

    if maximum is not None and result > maximum:
        raise SafetyValidationError(
            f"{name} must be <= {maximum}"
        )

    return result


def parse_enum(
    enum_class: type[Enum],
    value: Any,
    name: str,
) -> Enum:
    try:
        return enum_class(value)
    except (TypeError, ValueError) as exc:
        raise SafetyValidationError(
            f"Invalid {name}: {value}"
        ) from exc


def validate_datetime(
    value: datetime,
    name: str,
) -> datetime:
    if not isinstance(value, datetime):
        raise SafetyValidationError(
            f"{name} must be datetime"
        )

    if value.tzinfo is None or value.utcoffset() is None:
        raise SafetyValidationError(
            f"{name} must be timezone-aware"
        )

    return value.astimezone(timezone.utc)


def serialize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if hasattr(value, "__dataclass_fields__"):
        return serialize(asdict(value))

    if isinstance(value, Mapping):
        return {
            str(key): serialize(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [serialize(item) for item in value]

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

@dataclass(frozen=True)
class SafetyEvidence:
    evidence_id: str
    evidence_type: str
    reference: str
    description: Optional[str] = None
    captured_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


@dataclass(frozen=True)
class SafetyRule:
    rule_id: str
    name: str
    event_type: str
    sensor_field: str
    operator: str
    threshold: float
    severity: str
    enabled: bool = True
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


@dataclass(frozen=True)
class SafetyRuleResult:
    rule_id: str
    rule_name: str
    triggered: bool
    observed_value: Optional[float]
    threshold: float
    event_type: str
    severity: str

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


@dataclass(frozen=True)
class SafetyRiskAssessment:
    event_type: str
    severity: str
    probability: float
    exposure: float
    vulnerability: float
    risk_score: float
    risk_level: str
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


@dataclass
class SafetyEvent:
    event_id: str
    company_id: str
    site_id: str
    zone_id: Optional[str]
    asset_id: Optional[str]
    machine_id: Optional[str]

    event_type: str
    severity: str
    status: str
    source: str

    title: str
    description: str

    risk_score: float
    confidence: Optional[float]

    detected_at: datetime
    created_at: datetime
    updated_at: datetime

    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None

    assigned_to: Optional[str] = None

    mitigated_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None

    resolution_notes: Optional[str] = None

    evidence: list[dict[str, Any]] = field(
        default_factory=list
    )

    recommendations: list[str] = field(
        default_factory=list
    )

    incident_id: Optional[str] = None
    emergency_id: Optional[str] = None

    escalation_level: int = 0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


# ============================================================
# SAFETY SERVICE
# ============================================================

class SafetyService:
    """
    Industrial safety event management.

    Rule thresholds are supplied by the caller.
    No universal industrial safety limits are assumed.

    Authorization and tenant isolation must also be
    enforced by the API and repository layers.
    """

    def __init__(
        self,
        safety_repository: Any,
        notification_service: Any = None,
        incident_service: Any = None,
        emergency_service: Any = None,
        audit_service: Any = None,
    ):
        if safety_repository is None:
            raise SafetyValidationError(
                "safety_repository is required"
            )

        self.repository = safety_repository
        self.notifications = notification_service
        self.incidents = incident_service
        self.emergencies = emergency_service
        self.audit = audit_service

    # ========================================================
    # DEFAULT SEVERITY
    # ========================================================

    @staticmethod
    def classify_severity(
        event_type: str | SafetyEventType,
    ) -> SafetySeverity:
        selected = parse_enum(
            SafetyEventType,
            event_type,
            "event_type",
        )

        return DEFAULT_EVENT_SEVERITIES[selected]

    # ========================================================
    # RISK ASSESSMENT
    # ========================================================

    @staticmethod
    def calculate_risk(
        *,
        event_type: str | SafetyEventType,
        severity: str | SafetySeverity,
        probability: float,
        exposure: float = 1.0,
        vulnerability: float = 1.0,
    ) -> SafetyRiskAssessment:
        selected_type = parse_enum(
            SafetyEventType,
            event_type,
            "event_type",
        )

        selected_severity = parse_enum(
            SafetySeverity,
            severity,
            "severity",
        )

        probability_value = validate_number(
            probability,
            "probability",
            0,
            1,
        )

        exposure_value = validate_number(
            exposure,
            "exposure",
            0,
            1,
        )

        vulnerability_value = validate_number(
            vulnerability,
            "vulnerability",
            0,
            1,
        )

        severity_score = SEVERITY_SCORES[
            selected_severity
        ]

        risk_score = (
            severity_score
            * probability_value
            * exposure_value
            * vulnerability_value
        )

        risk_score = round(
            max(0.0, min(100.0, risk_score)),
            2,
        )

        if risk_score >= 75:
            risk_level = SafetyRiskLevel.CRITICAL
        elif risk_score >= 50:
            risk_level = SafetyRiskLevel.HIGH
        elif risk_score >= 25:
            risk_level = SafetyRiskLevel.MODERATE
        else:
            risk_level = SafetyRiskLevel.LOW

        return SafetyRiskAssessment(
            event_type=selected_type.value,
            severity=selected_severity.value,
            probability=probability_value,
            exposure=exposure_value,
            vulnerability=vulnerability_value,
            risk_score=risk_score,
            risk_level=risk_level.value,
            explanation=(
                "Illustrative risk score based on "
                "severity, estimated probability, "
                "exposure and vulnerability. "
                "Not a validated safety risk assessment."
            ),
        )

    # ========================================================
    # RULE EVALUATION
    # ========================================================

    @staticmethod
    def evaluate_rule(
        rule: SafetyRule,
        telemetry: Mapping[str, Any],
    ) -> SafetyRuleResult:
        if not isinstance(rule, SafetyRule):
            raise SafetyValidationError(
                "Expected SafetyRule"
            )

        if not isinstance(telemetry, Mapping):
            raise SafetyValidationError(
                "telemetry must be a mapping"
            )

        operator = parse_enum(
            SafetyRuleOperator,
            rule.operator,
            "operator",
        )

        threshold = validate_number(
            rule.threshold,
            "threshold",
        )

        observed = telemetry.get(
            rule.sensor_field
        )

        if observed is None or not rule.enabled:
            return SafetyRuleResult(
                rule_id=rule.rule_id,
                rule_name=rule.name,
                triggered=False,
                observed_value=None,
                threshold=threshold,
                event_type=rule.event_type,
                severity=rule.severity,
            )

        value = validate_number(
            observed,
            rule.sensor_field,
        )

        comparisons = {
            SafetyRuleOperator.GREATER_THAN: (
                value > threshold
            ),
            SafetyRuleOperator.GREATER_EQUAL: (
                value >= threshold
            ),
            SafetyRuleOperator.LESS_THAN: (
                value < threshold
            ),
            SafetyRuleOperator.LESS_EQUAL: (
                value <= threshold
            ),
            SafetyRuleOperator.EQUAL: (
                value == threshold
            ),
            SafetyRuleOperator.NOT_EQUAL: (
                value != threshold
            ),
        }

        return SafetyRuleResult(
            rule_id=rule.rule_id,
            rule_name=rule.name,
            triggered=comparisons[operator],
            observed_value=value,
            threshold=threshold,
            event_type=rule.event_type,
            severity=rule.severity,
        )

    @classmethod
    def evaluate_rules(
        cls,
        rules: Sequence[SafetyRule],
        telemetry: Mapping[str, Any],
    ) -> list[SafetyRuleResult]:
        return [
            cls.evaluate_rule(rule, telemetry)
            for rule in rules
        ]

    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    @staticmethod
    def generate_recommendations(
        event_type: SafetyEventType,
        severity: SafetySeverity,
    ) -> list[str]:
        recommendations = [
            "Verify the event using approved "
            "site monitoring and inspection procedures.",
        ]

        if event_type in {
            SafetyEventType.FIRE,
            SafetyEventType.SMOKE,
        }:
            recommendations.extend([
                "Follow the site fire and evacuation plan.",
                "Notify trained emergency personnel "
                "through approved emergency channels.",
                "Do not approach the hazard without "
                "appropriate authorization and protection.",
            ])

        elif event_type in {
            SafetyEventType.GAS_LEAK,
            SafetyEventType.CHEMICAL_LEAK,
        }:
            recommendations.extend([
                "Follow the approved hazardous-material "
                "response and evacuation procedures.",
                "Avoid entering the affected area.",
                "Request assessment by qualified responders.",
            ])

        elif event_type == SafetyEventType.OVERHEATING:
            recommendations.extend([
                "Review temperature trends and sensor health.",
                "Request qualified equipment inspection.",
                "Follow approved thermal hazard procedures.",
            ])

        elif event_type in {
            SafetyEventType.UNSAFE_PROXIMITY,
            SafetyEventType.RESTRICTED_ZONE,
            SafetyEventType.COLLISION_RISK,
        }:
            recommendations.extend([
                "Alert authorized site safety personnel.",
                "Review access controls and exclusion zones.",
                "Verify detection accuracy before "
                "recording a confirmed violation.",
            ])

        elif event_type == SafetyEventType.FALL_DETECTED:
            recommendations.extend([
                "Request immediate assessment by "
                "trained on-site responders.",
                "Follow approved medical emergency procedures.",
            ])

        elif event_type == SafetyEventType.ELECTRICAL_HAZARD:
            recommendations.extend([
                "Keep unauthorized personnel away "
                "from the suspected electrical hazard.",
                "Request assessment by qualified "
                "electrical safety personnel.",
            ])

        if severity == SafetySeverity.CRITICAL:
            recommendations.append(
                "Escalate through the approved "
                "site emergency response process."
            )

        elif severity == SafetySeverity.HIGH:
            recommendations.append(
                "Prioritize investigation and "
                "supervisor review."
            )

        return recommendations

    # ========================================================
    # CREATE SAFETY EVENT
    # ========================================================

    async def create_safety_event(
        self,
        *,
        company_id: str,
        site_id: str,
        event_type: str | SafetyEventType,
        title: str,
        description: str,
        source: str | SafetySource,
        severity: Optional[
            str | SafetySeverity
        ] = None,
        zone_id: Optional[str] = None,
        asset_id: Optional[str] = None,
        machine_id: Optional[str] = None,
        confidence: Optional[float] = None,
        probability: Optional[float] = None,
        exposure: float = 1.0,
        vulnerability: float = 1.0,
        evidence: Optional[
            Sequence[Mapping[str, Any]]
        ] = None,
        detected_at: Optional[datetime] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        send_notification: bool = True,
    ) -> SafetyEvent:
        selected_type = parse_enum(
            SafetyEventType,
            event_type,
            "event_type",
        )

        selected_source = parse_enum(
            SafetySource,
            source,
            "source",
        )

        selected_severity = (
            parse_enum(
                SafetySeverity,
                severity,
                "severity",
            )
            if severity is not None
            else self.classify_severity(selected_type)
        )

        selected_confidence = (
            validate_number(
                confidence,
                "confidence",
                0,
                1,
            )
            if confidence is not None
            else None
        )

        # Detection confidence is not automatically
        # a calibrated hazard probability.
        risk_probability = (
            validate_number(
                probability,
                "probability",
                0,
                1,
            )
            if probability is not None
            else None
        )

        risk_assessment = (
            self.calculate_risk(
                event_type=selected_type,
                severity=selected_severity,
                probability=risk_probability,
                exposure=exposure,
                vulnerability=vulnerability,
            )
            if risk_probability is not None
            else None
        )

        timestamp = utc_now()

        event = SafetyEvent(
            event_id=str(uuid4()),
            company_id=validate_text(
                company_id,
                "company_id",
            ),
            site_id=validate_text(
                site_id,
                "site_id",
            ),
            zone_id=optional_text(
                zone_id,
                "zone_id",
            ),
            asset_id=optional_text(
                asset_id,
                "asset_id",
            ),
            machine_id=optional_text(
                machine_id,
                "machine_id",
            ),
            event_type=selected_type.value,
            severity=selected_severity.value,
            status=SafetyStatus.DETECTED.value,
            source=selected_source.value,
            title=validate_text(
                title,
                "title",
            ),
            description=validate_text(
                description,
                "description",
                5000,
            ),
            risk_score=(
                risk_assessment.risk_score
                if risk_assessment is not None
                else 0.0
            ),
            confidence=selected_confidence,
            detected_at=(
                validate_datetime(
                    detected_at,
                    "detected_at",
                )
                if detected_at is not None
                else timestamp
            ),
            created_at=timestamp,
            updated_at=timestamp,
            evidence=[
                serialize(dict(item))
                for item in (evidence or [])
            ],
            recommendations=(
                self.generate_recommendations(
                    selected_type,
                    selected_severity,
                )
            ),
            metadata={
                **dict(metadata or {}),
                "risk_assessment_available": (
                    risk_assessment is not None
                ),
                "risk_score_note": (
                    "Risk score is zero when no "
                    "calibrated probability is supplied; "
                    "zero does not mean no hazard."
                    if risk_assessment is None
                    else risk_assessment.explanation
                ),
            },
        )

        try:
            await invoke(
                self.repository.create_safety_event,
                event.to_dict(),
            )
        except Exception as exc:
            raise SafetyRepositoryError(
                "Unable to create safety event"
            ) from exc

        await self._audit(
            event_id=event.event_id,
            action="safety_event_created",
            actor_id="system",
            details={
                "event_type": event.event_type,
                "severity": event.severity,
                "company_id": event.company_id,
                "site_id": event.site_id,
            },
        )

        if send_notification:
            await self._notify(event)

        return event

    # ========================================================
    # GET SAFETY EVENT
    # ========================================================

    async def get_safety_event(
        self,
        event_id: str,
    ) -> dict[str, Any]:
        identifier = validate_text(
            event_id,
            "event_id",
        )

        record = await invoke(
            self.repository.get_safety_event,
            identifier,
        )

        if record is None:
            raise SafetyEventNotFoundError(
                f"Safety event not found: {identifier}"
            )

        return record

    # ========================================================
    # CHANGE STATUS
    # ========================================================

    async def change_status(
        self,
        *,
        event_id: str,
        new_status: str | SafetyStatus,
        actor_id: str,
        notes: Optional[str] = None,
    ) -> dict[str, Any]:
        record = await self.get_safety_event(
            event_id
        )

        current_status = parse_enum(
            SafetyStatus,
            record["status"],
            "current_status",
        )

        target_status = parse_enum(
            SafetyStatus,
            new_status,
            "new_status",
        )

        if target_status not in SAFETY_TRANSITIONS[
            current_status
        ]:
            raise SafetyTransitionError(
                f"Cannot transition from "
                f"{current_status.value} to "
                f"{target_status.value}"
            )

        actor = validate_text(
            actor_id,
            "actor_id",
        )

        timestamp = utc_now()

        updates: dict[str, Any] = {
            "status": target_status.value,
            "updated_at": timestamp.isoformat(),
        }

        if notes is not None:
            updates["resolution_notes"] = validate_text(
                notes,
                "notes",
                5000,
            )

        if target_status == SafetyStatus.ACKNOWLEDGED:
            updates["acknowledged_at"] = (
                timestamp.isoformat()
            )
            updates["acknowledged_by"] = actor

        elif target_status == SafetyStatus.MITIGATED:
            updates["mitigated_at"] = (
                timestamp.isoformat()
            )

        elif target_status == SafetyStatus.RESOLVED:
            updates["resolved_at"] = (
                timestamp.isoformat()
            )

        elif target_status == SafetyStatus.CLOSED:
            updates["closed_at"] = (
                timestamp.isoformat()
            )

        updated = await invoke(
            self.repository.update_safety_event,
            event_id,
            updates,
        )

        await self._audit(
            event_id=event_id,
            action="safety_status_changed",
            actor_id=actor,
            details={
                "previous_status": current_status.value,
                "new_status": target_status.value,
                "notes": notes,
            },
        )

        return updated

    # ========================================================
    # ACKNOWLEDGE
    # ========================================================

    async def acknowledge_event(
        self,
        *,
        event_id: str,
        actor_id: str,
        notes: Optional[str] = None,
    ) -> dict[str, Any]:
        return await self.change_status(
            event_id=event_id,
            new_status=SafetyStatus.ACKNOWLEDGED,
            actor_id=actor_id,
            notes=notes,
        )

    # ========================================================
    # START INVESTIGATION
    # ========================================================

    async def start_investigation(
        self,
        *,
        event_id: str,
        actor_id: str,
        notes: Optional[str] = None,
    ) -> dict[str, Any]:
        return await self.change_status(
            event_id=event_id,
            new_status=SafetyStatus.INVESTIGATING,
            actor_id=actor_id,
            notes=notes,
        )

    # ========================================================
    # ASSIGN INVESTIGATOR
    # ========================================================

    async def assign_investigator(
        self,
        *,
        event_id: str,
        investigator_id: str,
        actor_id: str,
    ) -> dict[str, Any]:
        record = await self.get_safety_event(
            event_id
        )

        status = parse_enum(
            SafetyStatus,
            record["status"],
            "status",
        )

        if status in {
            SafetyStatus.CLOSED,
            SafetyStatus.FALSE_POSITIVE,
        }:
            raise SafetyTransitionError(
                "Cannot assign a terminal safety event"
            )

        investigator = validate_text(
            investigator_id,
            "investigator_id",
        )

        actor = validate_text(
            actor_id,
            "actor_id",
        )

        updated = await invoke(
            self.repository.update_safety_event,
            event_id,
            {
                "assigned_to": investigator,
                "updated_at": utc_now().isoformat(),
            },
        )

        await self._audit(
            event_id=event_id,
            action="safety_investigator_assigned",
            actor_id=actor,
            details={
                "investigator_id": investigator,
            },
        )

        return updated

    # ========================================================
    # MITIGATE
    # ========================================================

    async def mitigate_event(
        self,
        *,
        event_id: str,
        actor_id: str,
        mitigation_notes: str,
    ) -> dict[str, Any]:
        return await self.change_status(
            event_id=event_id,
            new_status=SafetyStatus.MITIGATED,
            actor_id=actor_id,
            notes=mitigation_notes,
        )

    # ========================================================
    # RESOLVE
    # ========================================================

    async def resolve_event(
        self,
        *,
        event_id: str,
        actor_id: str,
        resolution_notes: str,
    ) -> dict[str, Any]:
        return await self.change_status(
            event_id=event_id,
            new_status=SafetyStatus.RESOLVED,
            actor_id=actor_id,
            notes=resolution_notes,
        )

    # ========================================================
    # CLOSE
    # ========================================================

    async def close_event(
        self,
        *,
        event_id: str,
        actor_id: str,
        closure_notes: str,
    ) -> dict[str, Any]:
        return await self.change_status(
            event_id=event_id,
            new_status=SafetyStatus.CLOSED,
            actor_id=actor_id,
            notes=closure_notes,
        )

    # ========================================================
    # FALSE POSITIVE
    # ========================================================

    async def mark_false_positive(
        self,
        *,
        event_id: str,
        actor_id: str,
        reason: str,
    ) -> dict[str, Any]:
        return await self.change_status(
            event_id=event_id,
            new_status=SafetyStatus.FALSE_POSITIVE,
            actor_id=actor_id,
            notes=reason,
        )

    # ========================================================
    # ADD EVIDENCE
    # ========================================================

    async def add_evidence(
        self,
        *,
        event_id: str,
        evidence_type: str,
        reference: str,
        actor_id: str,
        description: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        record = await self.get_safety_event(
            event_id
        )

        status = parse_enum(
            SafetyStatus,
            record["status"],
            "status",
        )

        if status in {
            SafetyStatus.CLOSED,
            SafetyStatus.FALSE_POSITIVE,
        }:
            raise SafetyTransitionError(
                "Cannot modify terminal safety event"
            )

        evidence = SafetyEvidence(
            evidence_id=str(uuid4()),
            evidence_type=validate_text(
                evidence_type,
                "evidence_type",
            ),
            reference=validate_text(
                reference,
                "reference",
                2000,
            ),
            description=optional_text(
                description,
                "description",
            ),
            metadata=dict(metadata or {}),
        )

        existing = list(
            record.get("evidence", [])
        )

        existing.append(
            evidence.to_dict()
        )

        updated = await invoke(
            self.repository.update_safety_event,
            event_id,
            {
                "evidence": existing,
                "updated_at": utc_now().isoformat(),
            },
        )

        await self._audit(
            event_id=event_id,
            action="safety_evidence_added",
            actor_id=validate_text(
                actor_id,
                "actor_id",
            ),
            details={
                "evidence_id": evidence.evidence_id,
                "evidence_type": evidence.evidence_type,
            },
        )

        return updated

    # ========================================================
    # ESCALATION
    # ========================================================

    async def escalate_event(
        self,
        *,
        event_id: str,
        actor_id: str,
        reason: str,
        maximum_level: int = 5,
    ) -> dict[str, Any]:
        record = await self.get_safety_event(
            event_id
        )

        status = parse_enum(
            SafetyStatus,
            record["status"],
            "status",
        )

        if status in {
            SafetyStatus.CLOSED,
            SafetyStatus.FALSE_POSITIVE,
        }:
            raise SafetyTransitionError(
                "Cannot escalate terminal safety event"
            )

        current_level = int(
            record.get("escalation_level", 0)
        )

        if current_level >= maximum_level:
            raise SafetyTransitionError(
                "Maximum escalation level reached"
            )

        new_level = current_level + 1

        updated = await invoke(
            self.repository.update_safety_event,
            event_id,
            {
                "escalation_level": new_level,
                "updated_at": utc_now().isoformat(),
            },
        )

        await self._audit(
            event_id=event_id,
            action="safety_event_escalated",
            actor_id=validate_text(
                actor_id,
                "actor_id",
            ),
            details={
                "escalation_level": new_level,
                "reason": validate_text(
                    reason,
                    "reason",
                    2000,
                ),
            },
        )

        await self._notify(updated)

        return updated

    # ========================================================
    # NOTIFICATIONS
    # ========================================================

    async def _notify(
        self,
        event: SafetyEvent | Mapping[str, Any],
    ) -> bool:
        if self.notifications is None:
            return False

        method = getattr(
            self.notifications,
            "send_safety_alert",
            None,
        )

        if method is None:
            logger.warning(
                "Notification service does not implement "
                "send_safety_alert"
            )
            return False

        payload = (
            event.to_dict()
            if isinstance(event, SafetyEvent)
            else dict(event)
        )

        try:
            await invoke(
                method,
                payload,
            )
            return True

        except Exception:
            logger.exception(
                "Safety notification delivery failed "
                "for event %s",
                payload.get("event_id"),
            )
            return False

    # ========================================================
    # AUDIT
    # ========================================================

    async def _audit(
        self,
        *,
        event_id: str,
        action: str,
        actor_id: str,
        details: Optional[
            Mapping[str, Any]
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
                    "entity_type": "safety_event",
                    "entity_id": event_id,
                    "actor_id": actor_id,
                    "action": action,
                    "details": dict(details or {}),
                    "timestamp": utc_now().isoformat(),
                },
            )

        except Exception:
            logger.exception(
                "Safety audit failed for event %s",
                event_id,
            )

    # ========================================================
    # LIST EVENTS
    # ========================================================

    async def list_safety_events(
        self,
        *,
        company_id: Optional[str] = None,
        site_id: Optional[str] = None,
        zone_id: Optional[str] = None,
        asset_id: Optional[str] = None,
        event_type: Optional[str] = None,
        severity: Optional[str] = None,
        status: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        filters: dict[str, Any] = {}

        for name, value in {
            "company_id": company_id,
            "site_id": site_id,
            "zone_id": zone_id,
            "asset_id": asset_id,
        }.items():
            if value is not None:
                filters[name] = validate_text(
                    value,
                    name,
                )

        if event_type is not None:
            filters["event_type"] = parse_enum(
                SafetyEventType,
                event_type,
                "event_type",
            ).value

        if severity is not None:
            filters["severity"] = parse_enum(
                SafetySeverity,
                severity,
                "severity",
            ).value

        if status is not None:
            filters["status"] = parse_enum(
                SafetyStatus,
                status,
                "status",
            ).value

        return await invoke(
            self.repository.list_safety_events,
            filters,
        )

    # ========================================================
    # ACTIVE EVENTS
    # ========================================================

    async def get_active_events(
        self,
        *,
        company_id: str,
        site_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        records = await self.list_safety_events(
            company_id=company_id,
            site_id=site_id,
        )

        terminal_statuses = {
            SafetyStatus.RESOLVED.value,
            SafetyStatus.CLOSED.value,
            SafetyStatus.FALSE_POSITIVE.value,
        }

        return [
            record
            for record in records
            if record.get("status")
            not in terminal_statuses
        ]

    # ========================================================
    # SAFETY STATISTICS
    # ========================================================

    async def get_safety_statistics(
        self,
        *,
        company_id: str,
        site_id: Optional[str] = None,
    ) -> dict[str, Any]:
        records = await self.list_safety_events(
            company_id=company_id,
            site_id=site_id,
        )

        severity_counts: dict[str, int] = {}
        status_counts: dict[str, int] = {}
        type_counts: dict[str, int] = {}

        risk_scores = []

        for record in records:
            severity = str(
                record.get("severity", "unknown")
            )

            status = str(
                record.get("status", "unknown")
            )

            event_type = str(
                record.get("event_type", "unknown")
            )

            severity_counts[severity] = (
                severity_counts.get(severity, 0) + 1
            )

            status_counts[status] = (
                status_counts.get(status, 0) + 1
            )

            type_counts[event_type] = (
                type_counts.get(event_type, 0) + 1
            )

            if (
                record.get("metadata", {}).get(
                    "risk_assessment_available"
                )
            ):
                risk_scores.append(
                    validate_number(
                        record.get("risk_score", 0),
                        "risk_score",
                        0,
                        100,
                    )
                )

        active_events = [
            record
            for record in records
            if record.get("status")
            not in {
                SafetyStatus.RESOLVED.value,
                SafetyStatus.CLOSED.value,
                SafetyStatus.FALSE_POSITIVE.value,
            }
        ]

        return {
            "company_id": company_id,
            "site_id": site_id,
            "total_events": len(records),
            "active_events": len(active_events),
            "critical_events": severity_counts.get(
                SafetySeverity.CRITICAL.value,
                0,
            ),
            "high_severity_events": severity_counts.get(
                SafetySeverity.HIGH.value,
                0,
            ),
            "severity_counts": severity_counts,
            "status_counts": status_counts,
            "event_type_counts": type_counts,
            "average_assessed_risk_score": (
                round(
                    sum(risk_scores) / len(risk_scores),
                    2,
                )
                if risk_scores
                else None
            ),
            "assessed_event_count": len(risk_scores),
            "generated_at": utc_now().isoformat(),
        }


# ============================================================
# IN-MEMORY REPOSITORY
# ============================================================

class InMemorySafetyRepository:
    """
    Development-only safety repository.

    Not durable, transactional or concurrency-safe.
    """

    def __init__(self):
        self.records: dict[
            str,
            dict[str, Any],
        ] = {}

    async def create_safety_event(
        self,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        record = copy.deepcopy(data)

        event_id = record["event_id"]

        if event_id in self.records:
            raise SafetyRepositoryError(
                "Duplicate safety event ID"
            )

        self.records[event_id] = record

        return copy.deepcopy(record)

    async def get_safety_event(
        self,
        event_id: str,
    ) -> Optional[dict[str, Any]]:
        record = self.records.get(
            event_id
        )

        return (
            copy.deepcopy(record)
            if record is not None
            else None
        )

    async def update_safety_event(
        self,
        event_id: str,
        updates: dict[str, Any],
    ) -> dict[str, Any]:
        if event_id not in self.records:
            raise SafetyEventNotFoundError(
                f"Safety event not found: {event_id}"
            )

        self.records[event_id].update(
            copy.deepcopy(updates)
        )

        return copy.deepcopy(
            self.records[event_id]
        )

    async def list_safety_events(
        self,
        filters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return [
            copy.deepcopy(record)
            for record in self.records.values()
            if all(
                record.get(key) == value
                for key, value in filters.items()
            )
        ]


# ============================================================
# DEMONSTRATION
# ============================================================

async def main() -> None:
    logging.basicConfig(
        level=logging.INFO
    )

    repository = InMemorySafetyRepository()

    service = SafetyService(
        safety_repository=repository,
    )

    # --------------------------------------------------------
    # CONFIGURABLE RULES
    # --------------------------------------------------------

    rules = [
        SafetyRule(
            rule_id="RULE-TEMP-001",
            name="Motor Temperature Threshold",
            event_type="overheating",
            sensor_field="temperature",
            operator="gt",
            threshold=90.0,
            severity="high",
            description=(
                "Demonstration threshold only"
            ),
        ),
        SafetyRule(
            rule_id="RULE-VIB-001",
            name="Motor Vibration Threshold",
            event_type="vibration_hazard",
            sensor_field="vibration",
            operator="gt",
            threshold=10.0,
            severity="medium",
            description=(
                "Demonstration threshold only"
            ),
        ),
    ]

    telemetry = {
        "temperature": 96.5,
        "vibration": 11.2,
        "pressure": 180.0,
    }

    rule_results = service.evaluate_rules(
        rules,
        telemetry,
    )

    print("\nSAFETY RULE RESULTS")

    print(
        json.dumps(
            [
                result.to_dict()
                for result in rule_results
            ],
            indent=2,
        )
    )

    # --------------------------------------------------------
    # CREATE EVENT
    # --------------------------------------------------------

    event = await service.create_safety_event(
        company_id="COMPANY-001",
        site_id="SITE-001",
        zone_id="ZONE-001",
        asset_id="MACHINE-001",
        machine_id="MOTOR-001",
        event_type="overheating",
        title="Industrial Motor Overheating",
        description=(
            "Motor temperature exceeded the "
            "configured demonstration threshold."
        ),
        source="telemetry",
        severity="high",
        confidence=0.95,
        probability=0.80,
        exposure=0.90,
        vulnerability=0.85,
        evidence=[
            {
                "type": "telemetry",
                "temperature": 96.5,
                "threshold": 90.0,
            }
        ],
        metadata={
            "sensor_id": "TEMP-001",
            "environment": "development",
        },
    )

    print("\nCREATED SAFETY EVENT")

    print(
        json.dumps(
            event.to_dict(),
            indent=2,
        )
    )

    # --------------------------------------------------------
    # ACKNOWLEDGE
    # --------------------------------------------------------

    await service.acknowledge_event(
        event_id=event.event_id,
        actor_id="SAFETY-OFFICER-001",
        notes="Alert acknowledged.",
    )

    # --------------------------------------------------------
    # ASSIGN INVESTIGATOR
    # --------------------------------------------------------

    await service.assign_investigator(
        event_id=event.event_id,
        investigator_id="ENGINEER-001",
        actor_id="SAFETY-OFFICER-001",
    )

    # --------------------------------------------------------
    # INVESTIGATION
    # --------------------------------------------------------

    await service.start_investigation(
        event_id=event.event_id,
        actor_id="ENGINEER-001",
        notes="Engineering review initiated.",
    )

    # --------------------------------------------------------
    # ADD EVIDENCE
    # --------------------------------------------------------

    await service.add_evidence(
        event_id=event.event_id,
        evidence_type="inspection",
        reference="INSPECTION-001",
        actor_id="ENGINEER-001",
        description="Inspection evidence reference.",
    )

    # --------------------------------------------------------
    # MITIGATE
    # --------------------------------------------------------

    await service.mitigate_event(
        event_id=event.event_id,
        actor_id="ENGINEER-001",
        mitigation_notes=(
            "Mitigation documented by authorized personnel."
        ),
    )

    # --------------------------------------------------------
    # RESOLVE
    # --------------------------------------------------------

    await service.resolve_event(
        event_id=event.event_id,
        actor_id="SAFETY-OFFICER-001",
        resolution_notes=(
            "Investigation and mitigation reviewed."
        ),
    )

    # --------------------------------------------------------
    # CLOSE
    # --------------------------------------------------------

    closed_event = await service.close_event(
        event_id=event.event_id,
        actor_id="SAFETY-MANAGER-001",
        closure_notes=(
            "Administrative event closure approved."
        ),
    )

    print("\nCLOSED SAFETY EVENT")

    print(
        json.dumps(
            closed_event,
            indent=2,
        )
    )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    statistics_result = (
        await service.get_safety_statistics(
            company_id="COMPANY-001",
            site_id="SITE-001",
        )
    )

    print("\nSAFETY STATISTICS")

    print(
        json.dumps(
            statistics_result,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())