"""
app/ai/ml/models/emergency.py

Industrial Emergency Intelligence Models
========================================

Python 3.11+
Pydantic v2

Install:
    pip install "pydantic>=2,<3"

Features:
- Industrial emergency event schemas
- Fire, smoke, gas and thermal incidents
- Electrical and mechanical emergencies
- Worker injury and restricted-zone events
- Multi-tenant emergency management
- Emergency severity and status tracking
- Sensor and computer-vision evidence
- Emergency alerts and notifications
- Evacuation planning records
- Emergency response coordination
- Shutdown authorization requests
- Responder assignments
- Incident escalation
- Emergency cost and recovery information
- Event timeline and audit metadata
- Create, update and response schemas

SAFETY:
    AI detections are preliminary evidence,
    not certified emergency detection.

    These models never actuate machinery,
    trigger physical shutdowns or replace
    certified emergency systems.

    Emergency actions must follow approved
    procedures and independent safety controls.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# ============================================================
# ENUMS
# ============================================================

class EmergencyType(str, Enum):
    FIRE = "fire"
    SMOKE = "smoke"
    OVERHEATING = "overheating"
    GAS_LEAK = "gas_leak"
    CHEMICAL_LEAK = "chemical_leak"
    LIQUID_LEAK = "liquid_leak"
    ELECTRICAL_FAULT = "electrical_fault"
    ELECTRICAL_ARC = "electrical_arc"
    EXPLOSION = "explosion"
    EQUIPMENT_FAILURE = "equipment_failure"
    PRESSURE_FAILURE = "pressure_failure"
    STRUCTURAL_FAILURE = "structural_failure"
    WORKER_INJURY = "worker_injury"
    PERSON_TRAPPED = "person_trapped"
    RESTRICTED_ZONE_INTRUSION = (
        "restricted_zone_intrusion"
    )
    HAZARDOUS_MATERIAL = "hazardous_material"
    FLOOD = "flood"
    OTHER = "other"


class EmergencySeverity(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    CATASTROPHIC = "catastrophic"


class EmergencyStatus(str, Enum):
    DETECTED = "detected"
    PENDING_VERIFICATION = "pending_verification"
    CONFIRMED = "confirmed"
    ESCALATED = "escalated"
    RESPONDING = "responding"
    CONTAINED = "contained"
    RESOLVED = "resolved"
    CLOSED = "closed"
    FALSE_ALARM = "false_alarm"


class EmergencySource(str, Enum):
    SENSOR = "sensor"
    COMPUTER_VISION = "computer_vision"
    ML_MODEL = "ml_model"
    AI_AGENT = "ai_agent"
    HUMAN_REPORT = "human_report"
    SCADA = "scada"
    PLC = "plc"
    FIRE_ALARM_SYSTEM = "fire_alarm_system"
    GAS_DETECTION_SYSTEM = "gas_detection_system"
    OTHER = "other"


class EmergencyAlertChannel(str, Enum):
    DASHBOARD = "dashboard"
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    WEBHOOK = "webhook"
    CONTROL_ROOM = "control_room"


class EmergencyAlertStatus(str, Enum):
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    ACKNOWLEDGED = "acknowledged"
    FAILED = "failed"


class EmergencyActionType(str, Enum):
    NOTIFY = "notify"
    EVACUATE = "evacuate"
    ISOLATE_AREA = "isolate_area"
    REQUEST_SHUTDOWN = "request_shutdown"
    REQUEST_MAINTENANCE = "request_maintenance"
    REQUEST_MEDICAL_RESPONSE = (
        "request_medical_response"
    )
    REQUEST_FIRE_RESPONSE = (
        "request_fire_response"
    )
    INSPECT = "inspect"
    MONITOR = "monitor"
    ESCALATE = "escalate"


class EmergencyActionStatus(str, Enum):
    PROPOSED = "proposed"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class EvacuationStatus(str, Enum):
    NOT_REQUIRED = "not_required"
    RECOMMENDED = "recommended"
    ORDERED = "ordered"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ResponderType(str, Enum):
    SAFETY_OFFICER = "safety_officer"
    FIRE_TEAM = "fire_team"
    MEDICAL_TEAM = "medical_team"
    MAINTENANCE_TEAM = "maintenance_team"
    SECURITY_TEAM = "security_team"
    CONTROL_ROOM = "control_room"
    EXTERNAL_EMERGENCY_SERVICE = (
        "external_emergency_service"
    )


class EvidenceType(str, Enum):
    SENSOR_READING = "sensor_reading"
    IMAGE = "image"
    VIDEO = "video"
    THERMAL_IMAGE = "thermal_image"
    MODEL_PREDICTION = "model_prediction"
    HUMAN_REPORT = "human_report"
    DOCUMENT = "document"


# ============================================================
# BASE MODEL
# ============================================================

class EmergencyBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True,
        str_strip_whitespace=True,
    )


# ============================================================
# LOCATION
# ============================================================

class EmergencyLocation(EmergencyBaseModel):
    site_id: str = Field(
        min_length=1,
        max_length=100,
    )

    site_name: str | None = None
    building: str | None = None
    floor: str | None = None
    production_line: str | None = None
    zone_id: str | None = None
    zone_name: str | None = None

    latitude: float | None = Field(
        default=None,
        ge=-90,
        le=90,
    )

    longitude: float | None = Field(
        default=None,
        ge=-180,
        le=180,
    )

    location_description: str | None = None


# ============================================================
# EVIDENCE
# ============================================================

class EmergencyEvidence(EmergencyBaseModel):
    evidence_id: UUID = Field(
        default_factory=uuid4
    )

    evidence_type: EvidenceType

    source: EmergencySource

    source_id: str | None = None

    reference_uri: str | None = None

    model_id: str | None = None
    model_version: str | None = None
    prediction_id: str | None = None

    confidence: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    description: str | None = None

    captured_at: datetime = Field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


# ============================================================
# SENSOR TRIGGER
# ============================================================

class EmergencySensorTrigger(EmergencyBaseModel):
    sensor_id: str = Field(
        min_length=1,
        max_length=100,
    )

    sensor_type: str = Field(
        min_length=1,
        max_length=100,
    )

    observed_value: float

    unit: str = Field(
        min_length=1,
        max_length=30,
    )

    configured_threshold: float | None = None

    threshold_exceeded: bool

    measured_at: datetime

    @field_validator("measured_at")
    @classmethod
    def validate_measured_at(
        cls,
        value: datetime,
    ) -> datetime:
        if value.tzinfo is None:
            raise ValueError(
                "measured_at must be timezone-aware"
            )

        return value


# ============================================================
# ALERT
# ============================================================

class EmergencyAlert(EmergencyBaseModel):
    alert_id: UUID = Field(
        default_factory=uuid4
    )

    channel: EmergencyAlertChannel

    recipient_id: str | None = None

    recipient_address: str | None = None

    message: str = Field(
        min_length=1,
        max_length=4000,
    )

    status: EmergencyAlertStatus = (
        EmergencyAlertStatus.PENDING
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )

    sent_at: datetime | None = None
    delivered_at: datetime | None = None
    acknowledged_at: datetime | None = None

    delivery_reference: str | None = None

    failure_reason: str | None = None


# ============================================================
# EMERGENCY RESPONDER
# ============================================================

class EmergencyResponder(EmergencyBaseModel):
    responder_id: str = Field(
        min_length=1,
        max_length=100,
    )

    responder_type: ResponderType

    team_name: str | None = None

    assigned_at: datetime = Field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )

    acknowledged_at: datetime | None = None
    arrived_at: datetime | None = None

    completed_at: datetime | None = None

    notes: str | None = None


# ============================================================
# EMERGENCY ACTION
# ============================================================

class EmergencyAction(EmergencyBaseModel):
    action_id: UUID = Field(
        default_factory=uuid4
    )

    action_type: EmergencyActionType

    description: str = Field(
        min_length=1,
        max_length=2000,
    )

    status: EmergencyActionStatus = (
        EmergencyActionStatus.PROPOSED
    )

    assigned_to: str | None = None

    proposed_by: str | None = None

    approved_by: str | None = None

    approval_reference: str | None = None

    proposed_at: datetime = Field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )

    approved_at: datetime | None = None

    started_at: datetime | None = None
    completed_at: datetime | None = None

    requires_human_approval: bool = True

    notes: str | None = None

    @model_validator(mode="after")
    def validate_shutdown_approval(
        self,
    ) -> "EmergencyAction":
        if (
            self.action_type
            == EmergencyActionType.REQUEST_SHUTDOWN.value
        ):
            if not self.requires_human_approval:
                raise ValueError(
                    "Shutdown requests require "
                    "an approved external workflow"
                )

            if self.status in {
                EmergencyActionStatus.APPROVED.value,
                EmergencyActionStatus.IN_PROGRESS.value,
                EmergencyActionStatus.COMPLETED.value,
            }:
                if not (
                    self.approved_by
                    and self.approval_reference
                    and self.approved_at
                ):
                    raise ValueError(
                        "Shutdown request requires "
                        "documented approval"
                    )

        return self


# ============================================================
# EVACUATION
# ============================================================

class EmergencyEvacuation(EmergencyBaseModel):
    evacuation_id: UUID = Field(
        default_factory=uuid4
    )

    status: EvacuationStatus = (
        EvacuationStatus.NOT_REQUIRED
    )

    affected_zone_ids: list[str] = Field(
        default_factory=list
    )

    designated_assembly_point: str | None = None

    expected_person_count: int | None = Field(
        default=None,
        ge=0,
    )

    confirmed_safe_person_count: int | None = Field(
        default=None,
        ge=0,
    )

    ordered_by: str | None = None

    ordered_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None

    notes: str | None = None

    @model_validator(mode="after")
    def validate_counts(
        self,
    ) -> "EmergencyEvacuation":
        if (
            self.expected_person_count is not None
            and self.confirmed_safe_person_count
            is not None
            and self.confirmed_safe_person_count
            > self.expected_person_count
        ):
            raise ValueError(
                "Confirmed safe count exceeds "
                "expected person count"
            )

        return self


# ============================================================
# EMERGENCY COST
# ============================================================

class EmergencyCost(EmergencyBaseModel):
    currency: str = Field(
        default="INR",
        pattern=r"^[A-Z]{3}$",
    )

    equipment_damage_cost: float = Field(
        default=0,
        ge=0,
    )

    repair_cost: float = Field(
        default=0,
        ge=0,
    )

    replacement_cost: float = Field(
        default=0,
        ge=0,
    )

    downtime_cost: float = Field(
        default=0,
        ge=0,
    )

    production_loss_cost: float = Field(
        default=0,
        ge=0,
    )

    emergency_response_cost: float = Field(
        default=0,
        ge=0,
    )

    cleanup_cost: float = Field(
        default=0,
        ge=0,
    )

    other_cost: float = Field(
        default=0,
        ge=0,
    )

    @property
    def total_estimated_cost(self) -> float:
        return round(
            self.equipment_damage_cost
            + self.repair_cost
            + self.replacement_cost
            + self.downtime_cost
            + self.production_loss_cost
            + self.emergency_response_cost
            + self.cleanup_cost
            + self.other_cost,
            2,
        )

    def to_dict(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")
        data["total_estimated_cost"] = (
            self.total_estimated_cost
        )
        return data


# ============================================================
# TIMELINE EVENT
# ============================================================

class EmergencyTimelineEvent(EmergencyBaseModel):
    timeline_event_id: UUID = Field(
        default_factory=uuid4
    )

    event_type: str = Field(
        min_length=1,
        max_length=100,
    )

    description: str = Field(
        min_length=1,
        max_length=2000,
    )

    actor_id: str | None = None

    occurred_at: datetime = Field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )


# ============================================================
# MAIN EMERGENCY MODEL
# ============================================================

class IndustrialEmergency(EmergencyBaseModel):
    emergency_id: UUID = Field(
        default_factory=uuid4
    )

    company_id: str = Field(
        min_length=1,
        max_length=100,
    )

    emergency_code: str = Field(
        min_length=1,
        max_length=100,
    )

    emergency_type: EmergencyType

    title: str = Field(
        min_length=1,
        max_length=250,
    )

    description: str | None = None

    severity: EmergencySeverity

    status: EmergencyStatus = (
        EmergencyStatus.DETECTED
    )

    source: EmergencySource

    location: EmergencyLocation

    asset_id: str | None = None
    equipment_id: str | None = None
    component_id: str | None = None

    detected_at: datetime = Field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )

    confirmed_at: datetime | None = None
    contained_at: datetime | None = None
    resolved_at: datetime | None = None
    closed_at: datetime | None = None

    confirmed_by: str | None = None
    resolved_by: str | None = None

    requires_human_review: bool = True

    safety_escalation_recommended: bool = False

    sensor_triggers: list[
        EmergencySensorTrigger
    ] = Field(
        default_factory=list
    )

    evidence: list[
        EmergencyEvidence
    ] = Field(
        default_factory=list
    )

    alerts: list[
        EmergencyAlert
    ] = Field(
        default_factory=list
    )

    responders: list[
        EmergencyResponder
    ] = Field(
        default_factory=list
    )

    actions: list[
        EmergencyAction
    ] = Field(
        default_factory=list
    )

    evacuation: EmergencyEvacuation | None = None

    estimated_cost: EmergencyCost | None = None

    timeline: list[
        EmergencyTimelineEvent
    ] = Field(
        default_factory=list
    )

    root_cause: str | None = None

    corrective_actions: list[str] = Field(
        default_factory=list
    )

    preventive_actions: list[str] = Field(
        default_factory=list
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )

    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )

    @field_validator(
        "detected_at",
        "confirmed_at",
        "contained_at",
        "resolved_at",
        "closed_at",
    )
    @classmethod
    def validate_timestamp(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        if (
            value is not None
            and value.tzinfo is None
        ):
            raise ValueError(
                "Emergency timestamps must "
                "be timezone-aware"
            )

        return value

    @model_validator(mode="after")
    def validate_emergency(
        self,
    ) -> "IndustrialEmergency":
        if (
            self.closed_at is not None
            and self.resolved_at is not None
            and self.closed_at < self.resolved_at
        ):
            raise ValueError(
                "Emergency cannot close "
                "before resolution"
            )

        return self

    # --------------------------------------------------------
    # SERIALIZATION
    # --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(
            mode="json"
        )

    # --------------------------------------------------------
    # ADD EVIDENCE
    # --------------------------------------------------------

    def add_evidence(
        self,
        evidence: EmergencyEvidence,
    ) -> None:
        if any(
            item.evidence_id == evidence.evidence_id
            for item in self.evidence
        ):
            raise ValueError(
                "Duplicate evidence ID"
            )

        self.evidence.append(evidence)

        self._touch()

    # --------------------------------------------------------
    # ADD ALERT
    # --------------------------------------------------------

    def add_alert(
        self,
        alert: EmergencyAlert,
    ) -> None:
        if any(
            item.alert_id == alert.alert_id
            for item in self.alerts
        ):
            raise ValueError(
                "Duplicate alert ID"
            )

        self.alerts.append(alert)

        self._touch()

    # --------------------------------------------------------
    # ADD RESPONDER
    # --------------------------------------------------------

    def add_responder(
        self,
        responder: EmergencyResponder,
    ) -> None:
        if any(
            item.responder_id == responder.responder_id
            for item in self.responders
        ):
            raise ValueError(
                "Responder already assigned"
            )

        self.responders.append(responder)

        self._touch()

    # --------------------------------------------------------
    # ADD ACTION
    # --------------------------------------------------------

    def add_action(
        self,
        action: EmergencyAction,
    ) -> None:
        if any(
            item.action_id == action.action_id
            for item in self.actions
        ):
            raise ValueError(
                "Duplicate action ID"
            )

        self.actions.append(action)

        self._touch()

    # --------------------------------------------------------
    # RECORD TIMELINE
    # --------------------------------------------------------

    def record_event(
        self,
        event_type: str,
        description: str,
        actor_id: str | None = None,
    ) -> None:
        self.timeline.append(
            EmergencyTimelineEvent(
                event_type=event_type,
                description=description,
                actor_id=actor_id,
            )
        )

        self._touch()

    # --------------------------------------------------------
    # CONFIRM
    # --------------------------------------------------------

    def confirm(
        self,
        confirmed_by: str,
    ) -> None:
        if self.status not in {
            EmergencyStatus.DETECTED.value,
            EmergencyStatus.PENDING_VERIFICATION.value,
        }:
            raise ValueError(
                "Emergency cannot be confirmed "
                "from its current status"
            )

        self.status = EmergencyStatus.CONFIRMED
        self.confirmed_by = confirmed_by
        self.confirmed_at = datetime.now(
            timezone.utc
        )

        self.record_event(
            event_type="confirmed",
            description="Emergency confirmed",
            actor_id=confirmed_by,
        )

    # --------------------------------------------------------
    # ESCALATE
    # --------------------------------------------------------

    def escalate(
        self,
        actor_id: str,
        reason: str,
    ) -> None:
        if self.status in {
            EmergencyStatus.CLOSED.value,
            EmergencyStatus.FALSE_ALARM.value,
        }:
            raise ValueError(
                "Cannot escalate closed or "
                "false-alarm emergency"
            )

        self.status = EmergencyStatus.ESCALATED

        self.safety_escalation_recommended = True

        self.record_event(
            event_type="escalated",
            description=reason,
            actor_id=actor_id,
        )

    # --------------------------------------------------------
    # RESOLVE
    # --------------------------------------------------------

    def resolve(
        self,
        resolved_by: str,
        notes: str | None = None,
    ) -> None:
        if self.status not in {
            EmergencyStatus.CONTAINED.value,
            EmergencyStatus.RESPONDING.value,
        }:
            raise ValueError(
                "Emergency must be contained "
                "or responding before resolution"
            )

        self.status = EmergencyStatus.RESOLVED
        self.resolved_by = resolved_by
        self.resolved_at = datetime.now(
            timezone.utc
        )

        self.record_event(
            event_type="resolved",
            description=(
                notes or "Emergency resolved"
            ),
            actor_id=resolved_by,
        )

    # --------------------------------------------------------
    # CLOSE
    # --------------------------------------------------------

    def close(
        self,
        actor_id: str,
    ) -> None:
        if (
            self.status
            != EmergencyStatus.RESOLVED.value
        ):
            raise ValueError(
                "Only resolved emergencies "
                "can be closed"
            )

        self.status = EmergencyStatus.CLOSED

        self.closed_at = datetime.now(
            timezone.utc
        )

        self.record_event(
            event_type="closed",
            description="Emergency record closed",
            actor_id=actor_id,
        )

    # --------------------------------------------------------
    # CRITICAL FLAG
    # --------------------------------------------------------

    def is_critical(self) -> bool:
        return self.severity in {
            EmergencySeverity.CRITICAL.value,
            EmergencySeverity.CATASTROPHIC.value,
        }

    # --------------------------------------------------------
    # INTERNAL
    # --------------------------------------------------------

    def _touch(self) -> None:
        self.updated_at = datetime.now(
            timezone.utc
        )


# ============================================================
# CREATE SCHEMA
# ============================================================

class EmergencyCreate(EmergencyBaseModel):
    company_id: str = Field(
        min_length=1,
        max_length=100,
    )

    emergency_code: str = Field(
        min_length=1,
        max_length=100,
    )

    emergency_type: EmergencyType

    title: str = Field(
        min_length=1,
        max_length=250,
    )

    description: str | None = None

    severity: EmergencySeverity

    source: EmergencySource

    location: EmergencyLocation

    asset_id: str | None = None
    equipment_id: str | None = None
    component_id: str | None = None

    sensor_triggers: list[
        EmergencySensorTrigger
    ] = Field(
        default_factory=list
    )

    evidence: list[
        EmergencyEvidence
    ] = Field(
        default_factory=list
    )


# ============================================================
# UPDATE SCHEMA
# ============================================================

class EmergencyUpdate(EmergencyBaseModel):
    title: str | None = None
    description: str | None = None

    severity: EmergencySeverity | None = None

    status: EmergencyStatus | None = None

    root_cause: str | None = None

    corrective_actions: list[str] | None = None

    preventive_actions: list[str] | None = None


# ============================================================
# RESPONSE SCHEMA
# ============================================================

class EmergencyResponse(IndustrialEmergency):
    pass


# ============================================================
# SUMMARY SCHEMA
# ============================================================

class EmergencySummary(EmergencyBaseModel):
    emergency_id: UUID
    company_id: str
    emergency_code: str

    emergency_type: EmergencyType
    severity: EmergencySeverity
    status: EmergencyStatus

    title: str

    asset_id: str | None
    equipment_id: str | None

    detected_at: datetime

    alert_count: int
    responder_count: int
    action_count: int

    requires_human_review: bool
    safety_escalation_recommended: bool

    @classmethod
    def from_emergency(
        cls,
        emergency: IndustrialEmergency,
    ) -> "EmergencySummary":
        return cls(
            emergency_id=emergency.emergency_id,
            company_id=emergency.company_id,
            emergency_code=emergency.emergency_code,
            emergency_type=emergency.emergency_type,
            severity=emergency.severity,
            status=emergency.status,
            title=emergency.title,
            asset_id=emergency.asset_id,
            equipment_id=emergency.equipment_id,
            detected_at=emergency.detected_at,
            alert_count=len(emergency.alerts),
            responder_count=len(
                emergency.responders
            ),
            action_count=len(emergency.actions),
            requires_human_review=(
                emergency.requires_human_review
            ),
            safety_escalation_recommended=(
                emergency.safety_escalation_recommended
            ),
        )


# ============================================================
# EXAMPLE
# ============================================================

def example() -> None:
    import json

    emergency = IndustrialEmergency(
        company_id="company-001",
        emergency_code="EMG-2026-001",
        emergency_type=EmergencyType.OVERHEATING,
        title="Pump Motor Overheating",
        severity=EmergencySeverity.HIGH,
        source=EmergencySource.SENSOR,
        location=EmergencyLocation(
            site_id="SITE-01",
            site_name="Manufacturing Plant A",
            production_line="LINE-01",
            zone_name="Pump Room",
        ),
        asset_id="ASSET-101",
        equipment_id="PUMP-101",
        sensor_triggers=[
            EmergencySensorTrigger(
                sensor_id="TEMP-101",
                sensor_type="temperature",
                observed_value=105.0,
                configured_threshold=90.0,
                threshold_exceeded=True,
                unit="C",
                measured_at=datetime.now(
                    timezone.utc
                ),
            )
        ],
        requires_human_review=True,
        safety_escalation_recommended=True,
    )

    emergency.add_evidence(
        EmergencyEvidence(
            evidence_type=EvidenceType.SENSOR_READING,
            source=EmergencySource.SENSOR,
            source_id="TEMP-101",
            description=(
                "Temperature threshold exceeded"
            ),
        )
    )

    emergency.add_alert(
        EmergencyAlert(
            channel=EmergencyAlertChannel.DASHBOARD,
            message=(
                "High-temperature event detected "
                "at PUMP-101. Verify immediately."
            ),
        )
    )

    emergency.add_action(
        EmergencyAction(
            action_type=(
                EmergencyActionType.REQUEST_SHUTDOWN
            ),
            description=(
                "Request authorized assessment "
                "for equipment isolation or shutdown"
            ),
            status=(
                EmergencyActionStatus.PENDING_APPROVAL
            ),
            requires_human_approval=True,
        )
    )

    summary = EmergencySummary.from_emergency(
        emergency
    )

    print(
        json.dumps(
            summary.model_dump(mode="json"),
            indent=2,
        )
    )


if __name__ == "__main__":
    example()