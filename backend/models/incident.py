"""
app/ai/ml/models/incident.py

Industrial Incident Intelligence Models
=======================================

Python 3.11+
Pydantic v2

Install:
    pip install "pydantic>=2,<3"

Features:
- Multi-tenant incident management
- Safety and equipment incidents
- AI, sensor and human incident reports
- Incident severity and status tracking
- Asset and component associations
- Evidence and model provenance
- Worker impact and injury records
- Investigation and root-cause analysis
- Corrective and preventive actions (CAPA)
- Emergency and maintenance links
- Financial and production impact
- Incident timelines
- Create, update and response schemas

This module contains domain models only.
It does not perform database operations,
run AI inference or control equipment.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
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
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def money(
    value: Decimal | int | float | str,
) -> Decimal:
    return Decimal(str(value)).quantize(
        Decimal("0.01"),
        rounding=ROUND_HALF_UP,
    )


# ============================================================
# ENUMS
# ============================================================

class IncidentType(str, Enum):
    EQUIPMENT_FAILURE = "equipment_failure"
    MACHINE_BREAKDOWN = "machine_breakdown"
    FIRE = "fire"
    SMOKE = "smoke"
    OVERHEATING = "overheating"
    GAS_LEAK = "gas_leak"
    CHEMICAL_LEAK = "chemical_leak"
    LIQUID_LEAK = "liquid_leak"
    ELECTRICAL_FAULT = "electrical_fault"
    ELECTRICAL_ARC = "electrical_arc"
    EXPLOSION = "explosion"
    WORKER_INJURY = "worker_injury"
    FALL = "fall"
    COLLISION = "collision"
    NEAR_MISS = "near_miss"
    PPE_VIOLATION = "ppe_violation"
    RESTRICTED_ZONE = "restricted_zone"
    STRUCTURAL_DAMAGE = "structural_damage"
    ENVIRONMENTAL = "environmental"
    SECURITY = "security"
    PRODUCTION_DISRUPTION = "production_disruption"
    OTHER = "other"


class IncidentSeverity(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    CATASTROPHIC = "catastrophic"


class IncidentStatus(str, Enum):
    REPORTED = "reported"
    PENDING_VERIFICATION = "pending_verification"
    CONFIRMED = "confirmed"
    UNDER_INVESTIGATION = "under_investigation"
    ACTION_REQUIRED = "action_required"
    MONITORING = "monitoring"
    RESOLVED = "resolved"
    CLOSED = "closed"
    FALSE_POSITIVE = "false_positive"


class IncidentSource(str, Enum):
    HUMAN = "human"
    SENSOR = "sensor"
    COMPUTER_VISION = "computer_vision"
    ML_MODEL = "ml_model"
    AI_AGENT = "ai_agent"
    SCADA = "scada"
    PLC = "plc"
    CCTV = "cctv"
    EXTERNAL_SYSTEM = "external_system"


class IncidentPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class EvidenceType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    SENSOR_READING = "sensor_reading"
    TELEMETRY = "telemetry"
    THERMAL_IMAGE = "thermal_image"
    DOCUMENT = "document"
    HUMAN_STATEMENT = "human_statement"
    MODEL_PREDICTION = "model_prediction"
    OTHER = "other"


class InvestigationStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REVIEWED = "reviewed"


class RootCauseCategory(str, Enum):
    MECHANICAL = "mechanical"
    ELECTRICAL = "electrical"
    HUMAN_FACTOR = "human_factor"
    PROCESS = "process"
    MAINTENANCE = "maintenance"
    DESIGN = "design"
    ENVIRONMENTAL = "environmental"
    SOFTWARE = "software"
    SENSOR = "sensor"
    MATERIAL = "material"
    UNKNOWN = "unknown"


class CorrectiveActionType(str, Enum):
    INSPECTION = "inspection"
    REPAIR = "repair"
    REPLACEMENT = "replacement"
    MAINTENANCE = "maintenance"
    TRAINING = "training"
    PROCEDURE_UPDATE = "procedure_update"
    SAFETY_CONTROL = "safety_control"
    ENGINEERING_REVIEW = "engineering_review"
    MONITORING = "monitoring"
    OTHER = "other"


class CorrectiveActionStatus(str, Enum):
    OPEN = "open"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    PENDING_VERIFICATION = "pending_verification"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


# ============================================================
# BASE MODEL
# ============================================================

class IncidentBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True,
        str_strip_whitespace=True,
    )


# ============================================================
# CONTEXT
# ============================================================

class IncidentContext(IncidentBaseModel):
    company_id: str = Field(
        min_length=1,
        max_length=100,
    )

    site_id: str = Field(
        min_length=1,
        max_length=100,
    )

    asset_id: str | None = None
    equipment_id: str | None = None
    component_id: str | None = None

    production_line_id: str | None = None
    zone_id: str | None = None

    building: str | None = None
    floor: str | None = None

    location_description: str | None = None


# ============================================================
# MODEL REFERENCE
# ============================================================

class IncidentModelReference(IncidentBaseModel):
    model_id: str = Field(
        min_length=1,
        max_length=200,
    )

    model_version: str = Field(
        min_length=1,
        max_length=100,
    )

    prediction_id: str | None = None

    confidence: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    explanation: str | None = None


# ============================================================
# EVIDENCE
# ============================================================

class IncidentEvidence(IncidentBaseModel):
    evidence_id: UUID = Field(
        default_factory=uuid4
    )

    evidence_type: EvidenceType

    description: str | None = None

    storage_reference: str | None = None

    source_id: str | None = None

    model_reference: (
        IncidentModelReference | None
    ) = None

    captured_at: datetime = Field(
        default_factory=utc_now
    )

    added_by: str | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    @field_validator("captured_at")
    @classmethod
    def validate_timestamp(
        cls,
        value: datetime,
    ) -> datetime:
        if value.tzinfo is None:
            raise ValueError(
                "captured_at must be timezone-aware"
            )

        return value


# ============================================================
# PERSONNEL IMPACT
# ============================================================

class IncidentPersonnelImpact(IncidentBaseModel):
    affected_person_count: int = Field(
        default=0,
        ge=0,
    )

    injury_count: int = Field(
        default=0,
        ge=0,
    )

    fatality_count: int = Field(
        default=0,
        ge=0,
    )

    missing_person_count: int = Field(
        default=0,
        ge=0,
    )

    medical_response_required: bool = False

    evacuation_required: bool = False

    description: str | None = None

    @model_validator(mode="after")
    def validate_counts(
        self,
    ) -> "IncidentPersonnelImpact":
        if (
            self.injury_count
            + self.fatality_count
            + self.missing_person_count
            > self.affected_person_count
        ):
            raise ValueError(
                "Injured, deceased and missing "
                "person counts exceed total affected"
            )

        return self


# ============================================================
# FINANCIAL IMPACT
# ============================================================

class IncidentFinancialImpact(IncidentBaseModel):
    currency: str = Field(
        default="INR",
        pattern=r"^[A-Z]{3}$",
    )

    repair_cost: Decimal = Field(
        default=Decimal("0"),
        ge=0,
    )

    replacement_cost: Decimal = Field(
        default=Decimal("0"),
        ge=0,
    )

    downtime_cost: Decimal = Field(
        default=Decimal("0"),
        ge=0,
    )

    production_loss_cost: Decimal = Field(
        default=Decimal("0"),
        ge=0,
    )

    emergency_response_cost: Decimal = Field(
        default=Decimal("0"),
        ge=0,
    )

    cleanup_cost: Decimal = Field(
        default=Decimal("0"),
        ge=0,
    )

    other_cost: Decimal = Field(
        default=Decimal("0"),
        ge=0,
    )

    downtime_hours: float = Field(
        default=0,
        ge=0,
    )

    lost_production_units: float = Field(
        default=0,
        ge=0,
    )

    is_estimate: bool = True

    @property
    def total_cost(self) -> Decimal:
        return money(
            self.repair_cost
            + self.replacement_cost
            + self.downtime_cost
            + self.production_loss_cost
            + self.emergency_response_cost
            + self.cleanup_cost
            + self.other_cost
        )

    def to_dict(self) -> dict[str, Any]:
        result = self.model_dump(mode="json")
        result["total_cost"] = str(
            self.total_cost
        )
        return result


# ============================================================
# ROOT CAUSE
# ============================================================

class IncidentRootCause(IncidentBaseModel):
    root_cause_id: UUID = Field(
        default_factory=uuid4
    )

    category: RootCauseCategory

    description: str = Field(
        min_length=1,
        max_length=3000,
    )

    contributing_factors: list[str] = Field(
        default_factory=list
    )

    evidence_ids: list[UUID] = Field(
        default_factory=list
    )

    verified: bool = False

    verified_by: str | None = None

    verified_at: datetime | None = None

    @model_validator(mode="after")
    def validate_verification(
        self,
    ) -> "IncidentRootCause":
        if self.verified and not (
            self.verified_by and self.verified_at
        ):
            raise ValueError(
                "Verified root cause requires "
                "reviewer and verification timestamp"
            )

        return self


# ============================================================
# INVESTIGATION
# ============================================================

class IncidentInvestigation(IncidentBaseModel):
    investigation_id: UUID = Field(
        default_factory=uuid4
    )

    status: InvestigationStatus = (
        InvestigationStatus.NOT_STARTED
    )

    lead_investigator_id: str | None = None

    investigator_ids: list[str] = Field(
        default_factory=list
    )

    findings: list[str] = Field(
        default_factory=list
    )

    root_causes: list[IncidentRootCause] = Field(
        default_factory=list
    )

    started_at: datetime | None = None
    completed_at: datetime | None = None

    reviewed_by: str | None = None

    notes: str | None = None

    @model_validator(mode="after")
    def validate_investigation(
        self,
    ) -> "IncidentInvestigation":
        if (
            self.started_at is not None
            and self.completed_at is not None
            and self.completed_at < self.started_at
        ):
            raise ValueError(
                "Investigation completion precedes start"
            )

        return self


# ============================================================
# CORRECTIVE / PREVENTIVE ACTION
# ============================================================

class IncidentCorrectiveAction(IncidentBaseModel):
    action_id: UUID = Field(
        default_factory=uuid4
    )

    action_type: CorrectiveActionType

    description: str = Field(
        min_length=1,
        max_length=3000,
    )

    priority: IncidentPriority = (
        IncidentPriority.MEDIUM
    )

    status: CorrectiveActionStatus = (
        CorrectiveActionStatus.OPEN
    )

    assigned_to: str | None = None

    due_at: datetime | None = None

    completed_at: datetime | None = None

    verified_by: str | None = None
    verified_at: datetime | None = None

    work_order_id: str | None = None

    requires_approval: bool = True

    notes: str | None = None

    created_at: datetime = Field(
        default_factory=utc_now
    )

    @model_validator(mode="after")
    def validate_action(
        self,
    ) -> "IncidentCorrectiveAction":
        if (
            self.status
            == CorrectiveActionStatus.COMPLETED.value
            and self.completed_at is None
        ):
            raise ValueError(
                "Completed action requires completed_at"
            )

        if (
            self.verified_at is not None
            and self.verified_by is None
        ):
            raise ValueError(
                "Verification requires verified_by"
            )

        return self


# ============================================================
# TIMELINE
# ============================================================

class IncidentTimelineEvent(IncidentBaseModel):
    event_id: UUID = Field(
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
        default_factory=utc_now
    )


# ============================================================
# MAIN INCIDENT MODEL
# ============================================================

class IndustrialIncident(IncidentBaseModel):
    incident_id: UUID = Field(
        default_factory=uuid4
    )

    incident_code: str = Field(
        min_length=1,
        max_length=100,
    )

    context: IncidentContext

    incident_type: IncidentType

    title: str = Field(
        min_length=1,
        max_length=250,
    )

    description: str | None = None

    severity: IncidentSeverity

    priority: IncidentPriority = (
        IncidentPriority.MEDIUM
    )

    status: IncidentStatus = (
        IncidentStatus.REPORTED
    )

    source: IncidentSource

    reported_by: str | None = None

    assigned_to: str | None = None

    occurred_at: datetime | None = None

    detected_at: datetime = Field(
        default_factory=utc_now
    )

    reported_at: datetime = Field(
        default_factory=utc_now
    )

    confirmed_at: datetime | None = None
    resolved_at: datetime | None = None
    closed_at: datetime | None = None

    confirmed_by: str | None = None
    resolved_by: str | None = None
    closed_by: str | None = None

    model_reference: (
        IncidentModelReference | None
    ) = None

    evidence: list[IncidentEvidence] = Field(
        default_factory=list
    )

    personnel_impact: IncidentPersonnelImpact = Field(
        default_factory=IncidentPersonnelImpact
    )

    financial_impact: IncidentFinancialImpact = Field(
        default_factory=IncidentFinancialImpact
    )

    investigation: (
        IncidentInvestigation | None
    ) = None

    corrective_actions: list[
        IncidentCorrectiveAction
    ] = Field(
        default_factory=list
    )

    preventive_actions: list[
        IncidentCorrectiveAction
    ] = Field(
        default_factory=list
    )

    timeline: list[IncidentTimelineEvent] = Field(
        default_factory=list
    )

    emergency_id: str | None = None
    failure_id: str | None = None
    maintenance_id: str | None = None
    work_order_id: str | None = None

    requires_human_review: bool = True

    safety_escalation_required: bool = False

    regulatory_reporting_required: bool = False

    tags: list[str] = Field(
        default_factory=list
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    created_at: datetime = Field(
        default_factory=utc_now
    )

    updated_at: datetime = Field(
        default_factory=utc_now
    )

    @field_validator(
        "occurred_at",
        "detected_at",
        "reported_at",
        "confirmed_at",
        "resolved_at",
        "closed_at",
        "created_at",
        "updated_at",
    )
    @classmethod
    def validate_timestamps(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        if (
            value is not None
            and value.tzinfo is None
        ):
            raise ValueError(
                "Incident timestamps must "
                "be timezone-aware"
            )

        return value

    @model_validator(mode="after")
    def validate_lifecycle(
        self,
    ) -> "IndustrialIncident":
        if (
            self.closed_at is not None
            and self.resolved_at is not None
            and self.closed_at < self.resolved_at
        ):
            raise ValueError(
                "Incident cannot close "
                "before resolution"
            )

        return self

    # --------------------------------------------------------
    # SERIALIZATION
    # --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        result = self.model_dump(mode="json")

        result["total_financial_impact"] = str(
            self.financial_impact.total_cost
        )

        result["evidence_count"] = len(
            self.evidence
        )

        result["corrective_action_count"] = len(
            self.corrective_actions
        )

        return result

    # --------------------------------------------------------
    # INTERNAL TIMESTAMP
    # --------------------------------------------------------

    def _touch(self) -> None:
        self.updated_at = utc_now()

    # --------------------------------------------------------
    # TIMELINE
    # --------------------------------------------------------

    def record_event(
        self,
        *,
        event_type: str,
        description: str,
        actor_id: str | None = None,
    ) -> None:
        self.timeline.append(
            IncidentTimelineEvent(
                event_type=event_type,
                description=description,
                actor_id=actor_id,
            )
        )

        self._touch()

    # --------------------------------------------------------
    # ADD EVIDENCE
    # --------------------------------------------------------

    def add_evidence(
        self,
        evidence: IncidentEvidence,
    ) -> None:
        if any(
            item.evidence_id == evidence.evidence_id
            for item in self.evidence
        ):
            raise ValueError(
                "Duplicate evidence ID"
            )

        self.evidence.append(evidence)

        self.record_event(
            event_type="evidence_added",
            description=(
                f"Evidence added: "
                f"{evidence.evidence_type}"
            ),
            actor_id=evidence.added_by,
        )

    # --------------------------------------------------------
    # ADD CORRECTIVE ACTION
    # --------------------------------------------------------

    def add_corrective_action(
        self,
        action: IncidentCorrectiveAction,
    ) -> None:
        all_actions = (
            self.corrective_actions
            + self.preventive_actions
        )

        if any(
            item.action_id == action.action_id
            for item in all_actions
        ):
            raise ValueError(
                "Duplicate action ID"
            )

        self.corrective_actions.append(action)

        self.record_event(
            event_type="corrective_action_added",
            description=action.description,
        )

    # --------------------------------------------------------
    # ADD PREVENTIVE ACTION
    # --------------------------------------------------------

    def add_preventive_action(
        self,
        action: IncidentCorrectiveAction,
    ) -> None:
        all_actions = (
            self.corrective_actions
            + self.preventive_actions
        )

        if any(
            item.action_id == action.action_id
            for item in all_actions
        ):
            raise ValueError(
                "Duplicate action ID"
            )

        self.preventive_actions.append(action)

        self.record_event(
            event_type="preventive_action_added",
            description=action.description,
        )

    # --------------------------------------------------------
    # CONFIRM
    # --------------------------------------------------------

    def confirm(
        self,
        confirmed_by: str,
    ) -> None:
        if self.status not in {
            IncidentStatus.REPORTED.value,
            IncidentStatus.PENDING_VERIFICATION.value,
        }:
            raise ValueError(
                "Incident cannot be confirmed "
                "from its current status"
            )

        if not confirmed_by.strip():
            raise ValueError(
                "confirmed_by is required"
            )

        self.status = IncidentStatus.CONFIRMED

        self.confirmed_by = confirmed_by
        self.confirmed_at = utc_now()

        self.record_event(
            event_type="confirmed",
            description="Incident confirmed",
            actor_id=confirmed_by,
        )

    # --------------------------------------------------------
    # START INVESTIGATION
    # --------------------------------------------------------

    def start_investigation(
        self,
        lead_investigator_id: str,
    ) -> None:
        if self.status not in {
            IncidentStatus.CONFIRMED.value,
            IncidentStatus.ACTION_REQUIRED.value,
        }:
            raise ValueError(
                "Incident must be confirmed "
                "before investigation"
            )

        if self.investigation is not None:
            raise ValueError(
                "Investigation already exists"
            )

        self.investigation = (
            IncidentInvestigation(
                status=InvestigationStatus.IN_PROGRESS,
                lead_investigator_id=(
                    lead_investigator_id
                ),
                started_at=utc_now(),
            )
        )

        self.status = (
            IncidentStatus.UNDER_INVESTIGATION
        )

        self.record_event(
            event_type="investigation_started",
            description="Investigation started",
            actor_id=lead_investigator_id,
        )

    # --------------------------------------------------------
    # RESOLVE
    # --------------------------------------------------------

    def resolve(
        self,
        resolved_by: str,
        resolution_notes: str,
    ) -> None:
        if self.status not in {
            IncidentStatus.CONFIRMED.value,
            IncidentStatus.UNDER_INVESTIGATION.value,
            IncidentStatus.ACTION_REQUIRED.value,
            IncidentStatus.MONITORING.value,
        }:
            raise ValueError(
                "Incident cannot be resolved "
                "from its current status"
            )

        if not resolved_by.strip():
            raise ValueError(
                "resolved_by is required"
            )

        if not resolution_notes.strip():
            raise ValueError(
                "Resolution notes are required"
            )

        self.status = IncidentStatus.RESOLVED

        self.resolved_by = resolved_by
        self.resolved_at = utc_now()

        self.record_event(
            event_type="resolved",
            description=resolution_notes,
            actor_id=resolved_by,
        )

    # --------------------------------------------------------
    # CLOSE
    # --------------------------------------------------------

    def close(
        self,
        closed_by: str,
    ) -> None:
        if (
            self.status
            != IncidentStatus.RESOLVED.value
        ):
            raise ValueError(
                "Only resolved incidents "
                "can be closed"
            )

        pending_actions = [
            action
            for action in (
                self.corrective_actions
                + self.preventive_actions
            )
            if action.status not in {
                CorrectiveActionStatus.COMPLETED.value,
                CorrectiveActionStatus.CANCELLED.value,
            }
        ]

        if pending_actions:
            raise ValueError(
                "Cannot close incident with "
                "unfinished corrective or "
                "preventive actions"
            )

        if not closed_by.strip():
            raise ValueError(
                "closed_by is required"
            )

        self.status = IncidentStatus.CLOSED

        self.closed_by = closed_by
        self.closed_at = utc_now()

        self.record_event(
            event_type="closed",
            description="Incident closed",
            actor_id=closed_by,
        )

    # --------------------------------------------------------
    # RISK HELPERS
    # --------------------------------------------------------

    def is_critical(self) -> bool:
        return self.severity in {
            IncidentSeverity.CRITICAL.value,
            IncidentSeverity.CATASTROPHIC.value,
        }

    def has_personnel_impact(self) -> bool:
        return (
            self.personnel_impact.affected_person_count
            > 0
        )

    def requires_escalation(self) -> bool:
        return (
            self.is_critical()
            or self.safety_escalation_required
            or self.personnel_impact.fatality_count > 0
        )

    def open_action_count(self) -> int:
        return sum(
            action.status not in {
                CorrectiveActionStatus.COMPLETED.value,
                CorrectiveActionStatus.CANCELLED.value,
            }
            for action in (
                self.corrective_actions
                + self.preventive_actions
            )
        )


# ============================================================
# CREATE SCHEMA
# ============================================================

class IncidentCreate(IncidentBaseModel):
    incident_code: str = Field(
        min_length=1,
        max_length=100,
    )

    context: IncidentContext

    incident_type: IncidentType

    title: str = Field(
        min_length=1,
        max_length=250,
    )

    description: str | None = None

    severity: IncidentSeverity

    priority: IncidentPriority = (
        IncidentPriority.MEDIUM
    )

    source: IncidentSource

    reported_by: str | None = None

    occurred_at: datetime | None = None

    model_reference: (
        IncidentModelReference | None
    ) = None

    emergency_id: str | None = None
    failure_id: str | None = None


# ============================================================
# UPDATE SCHEMA
# ============================================================

class IncidentUpdate(IncidentBaseModel):
    title: str | None = None
    description: str | None = None

    severity: IncidentSeverity | None = None
    priority: IncidentPriority | None = None

    assigned_to: str | None = None

    safety_escalation_required: bool | None = None

    regulatory_reporting_required: bool | None = None

    tags: list[str] | None = None


# ============================================================
# RESPONSE SCHEMA
# ============================================================

class IncidentResponse(IndustrialIncident):
    pass


# ============================================================
# SUMMARY
# ============================================================

class IncidentSummary(IncidentBaseModel):
    incident_id: UUID

    incident_code: str

    company_id: str
    site_id: str

    asset_id: str | None

    incident_type: IncidentType

    title: str

    severity: IncidentSeverity
    priority: IncidentPriority
    status: IncidentStatus

    evidence_count: int
    open_action_count: int

    total_financial_impact: Decimal

    requires_escalation: bool

    reported_at: datetime

    @classmethod
    def from_incident(
        cls,
        incident: IndustrialIncident,
    ) -> "IncidentSummary":
        return cls(
            incident_id=incident.incident_id,
            incident_code=incident.incident_code,
            company_id=incident.context.company_id,
            site_id=incident.context.site_id,
            asset_id=incident.context.asset_id,
            incident_type=incident.incident_type,
            title=incident.title,
            severity=incident.severity,
            priority=incident.priority,
            status=incident.status,
            evidence_count=len(incident.evidence),
            open_action_count=(
                incident.open_action_count()
            ),
            total_financial_impact=(
                incident.financial_impact.total_cost
            ),
            requires_escalation=(
                incident.requires_escalation()
            ),
            reported_at=incident.reported_at,
        )


# ============================================================
# EXAMPLE
# ============================================================

def example() -> None:
    import json

    incident = IndustrialIncident(
        incident_code="INC-2026-001",
        context=IncidentContext(
            company_id="company-001",
            site_id="SITE-01",
            asset_id="ASSET-101",
            equipment_id="PUMP-101",
            component_id="BEARING-101",
            production_line_id="LINE-01",
            location_description="Pump Room A",
        ),
        incident_type=IncidentType.EQUIPMENT_FAILURE,
        title="Industrial Pump Bearing Failure",
        description=(
            "Abnormal vibration followed by "
            "pump bearing failure."
        ),
        severity=IncidentSeverity.HIGH,
        priority=IncidentPriority.HIGH,
        source=IncidentSource.SENSOR,
        financial_impact=IncidentFinancialImpact(
            repair_cost=Decimal("25000"),
            downtime_cost=Decimal("15000"),
            production_loss_cost=Decimal("30000"),
            downtime_hours=6,
        ),
    )

    incident.add_evidence(
        IncidentEvidence(
            evidence_type=EvidenceType.TELEMETRY,
            description=(
                "Vibration sensor recorded "
                "abnormal operating conditions."
            ),
            source_id="VIB-101",
        )
    )

    incident.confirm(
        confirmed_by="safety-officer-001"
    )

    incident.start_investigation(
        lead_investigator_id="engineer-001"
    )

    incident.add_corrective_action(
        IncidentCorrectiveAction(
            action_type=CorrectiveActionType.REPAIR,
            description=(
                "Inspect and repair the affected "
                "bearing assembly."
            ),
            priority=IncidentPriority.HIGH,
            assigned_to="maintenance-team-001",
        )
    )

    summary = IncidentSummary.from_incident(
        incident
    )

    print(
        json.dumps(
            summary.model_dump(mode="json"),
            indent=2,
        )
    )


if __name__ == "__main__":
    example()