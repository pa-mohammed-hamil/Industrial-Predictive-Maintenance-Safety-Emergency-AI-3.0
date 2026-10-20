"""
app/ai/ml/models/maintenance.py

Industrial Maintenance Intelligence Models
==========================================

Python 3.11+
Pydantic v2

Install:
    pip install "pydantic>=2,<3"

Features:
- Multi-tenant maintenance management
- Preventive, predictive and corrective maintenance
- Maintenance work orders
- Machine and component relationships
- Failure, anomaly and RUL prediction references
- Maintenance scheduling
- Technician assignments
- Spare parts and inventory references
- Labor, repair and downtime costs
- Safety permits and approval workflows
- Maintenance checklists
- Maintenance history
- Repair and replacement decisions
- Completion and verification
- Create, update, response and summary schemas

This module defines validated domain models.
Database persistence, notifications, ML inference
and physical equipment control are separate.
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


def validate_aware(
    value: datetime | None,
) -> datetime | None:
    if value is not None and (
        value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(
            "Datetime must be timezone-aware"
        )

    return value


# ============================================================
# ENUMS
# ============================================================

class MaintenanceType(str, Enum):
    PREVENTIVE = "preventive"
    PREDICTIVE = "predictive"
    CORRECTIVE = "corrective"
    CONDITION_BASED = "condition_based"
    EMERGENCY = "emergency"
    INSPECTION = "inspection"
    CALIBRATION = "calibration"
    OVERHAUL = "overhaul"


class MaintenanceStatus(str, Enum):
    DRAFT = "draft"
    PLANNED = "planned"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    ON_HOLD = "on_hold"
    PENDING_VERIFICATION = "pending_verification"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class MaintenancePriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"
    EMERGENCY = "emergency"


class MaintenanceActionType(str, Enum):
    INSPECTION = "inspection"
    CLEANING = "cleaning"
    LUBRICATION = "lubrication"
    CALIBRATION = "calibration"
    ADJUSTMENT = "adjustment"
    REPAIR = "repair"
    REPLACEMENT = "replacement"
    OVERHAUL = "overhaul"
    TESTING = "testing"
    SOFTWARE_UPDATE = "software_update"
    SAFETY_REVIEW = "safety_review"


class MaintenanceDecision(str, Enum):
    REPAIR = "repair"
    REPLACE = "replace"
    INSPECT = "inspect"
    MONITOR = "monitor"
    ENGINEERING_REVIEW = "engineering_review"


class TechnicianRole(str, Enum):
    TECHNICIAN = "technician"
    ELECTRICIAN = "electrician"
    MECHANICAL_ENGINEER = "mechanical_engineer"
    MAINTENANCE_ENGINEER = "maintenance_engineer"
    SAFETY_OFFICER = "safety_officer"
    SUPERVISOR = "supervisor"
    EXTERNAL_CONTRACTOR = "external_contractor"


class ChecklistStatus(str, Enum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"


class MaintenanceSource(str, Enum):
    MANUAL = "manual"
    SCHEDULE = "schedule"
    SENSOR_ALERT = "sensor_alert"
    ANOMALY_MODEL = "anomaly_model"
    FAILURE_MODEL = "failure_model"
    RUL_MODEL = "rul_model"
    AI_AGENT = "ai_agent"
    INSPECTION = "inspection"
    INCIDENT = "incident"


class MaintenanceSafetyStatus(str, Enum):
    NOT_ASSESSED = "not_assessed"
    REVIEW_REQUIRED = "review_required"
    APPROVED = "approved"
    BLOCKED = "blocked"


# ============================================================
# BASE MODEL
# ============================================================

class MaintenanceBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True,
        str_strip_whitespace=True,
    )


# ============================================================
# MAINTENANCE CONTEXT
# ============================================================

class MaintenanceContext(MaintenanceBaseModel):
    company_id: str = Field(min_length=1)
    site_id: str = Field(min_length=1)
    asset_id: str = Field(min_length=1)

    equipment_id: str | None = None
    machine_id: str | None = None
    component_id: str | None = None

    production_line_id: str | None = None
    zone_id: str | None = None


# ============================================================
# PREDICTION REFERENCE
# ============================================================

class MaintenancePredictionReference(
    MaintenanceBaseModel
):
    prediction_id: str = Field(min_length=1)

    model_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)

    failure_probability: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    prediction_horizon_hours: float | None = Field(
        default=None,
        gt=0,
    )

    remaining_useful_life_hours: float | None = Field(
        default=None,
        ge=0,
    )

    anomaly_score: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    predicted_at: datetime = Field(
        default_factory=utc_now
    )

    @field_validator("predicted_at")
    @classmethod
    def validate_date(
        cls,
        value: datetime,
    ) -> datetime:
        return validate_aware(value)

    @model_validator(mode="after")
    def validate_horizon(
        self,
    ) -> "MaintenancePredictionReference":
        if (
            self.failure_probability is not None
            and self.prediction_horizon_hours is None
        ):
            raise ValueError(
                "Failure probability requires "
                "a prediction horizon"
            )

        return self


# ============================================================
# TECHNICIAN
# ============================================================

class MaintenanceTechnician(MaintenanceBaseModel):
    technician_id: str = Field(min_length=1)

    name: str = Field(min_length=1)

    role: TechnicianRole

    skills: list[str] = Field(
        default_factory=list
    )

    assigned_at: datetime = Field(
        default_factory=utc_now
    )

    hourly_rate: Decimal | None = Field(
        default=None,
        ge=0,
    )

    assigned_hours: float = Field(
        default=0,
        ge=0,
    )

    @property
    def estimated_labor_cost(self) -> Decimal:
        return money(
            (self.hourly_rate or Decimal("0"))
            * Decimal(str(self.assigned_hours))
        )


# ============================================================
# SPARE PART
# ============================================================

class MaintenanceSparePart(MaintenanceBaseModel):
    part_id: str = Field(min_length=1)

    part_name: str = Field(min_length=1)

    part_number: str | None = None

    quantity: float = Field(gt=0)

    unit: str = "unit"

    unit_cost: Decimal = Field(ge=0)

    supplier_id: str | None = None

    inventory_reference_id: str | None = None

    available_quantity: float | None = Field(
        default=None,
        ge=0,
    )

    @property
    def total_cost(self) -> Decimal:
        return money(
            Decimal(str(self.quantity))
            * self.unit_cost
        )


# ============================================================
# MAINTENANCE CHECKLIST
# ============================================================

class MaintenanceChecklistItem(
    MaintenanceBaseModel
):
    checklist_item_id: UUID = Field(
        default_factory=uuid4
    )

    description: str = Field(min_length=1)

    status: ChecklistStatus = (
        ChecklistStatus.PENDING
    )

    mandatory: bool = True

    completed_by: str | None = None
    completed_at: datetime | None = None

    notes: str | None = None

    @field_validator("completed_at")
    @classmethod
    def validate_completed_at(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        return validate_aware(value)

    @model_validator(mode="after")
    def validate_completion(
        self,
    ) -> "MaintenanceChecklistItem":
        if self.status in {
            ChecklistStatus.PASSED.value,
            ChecklistStatus.FAILED.value,
            ChecklistStatus.NOT_APPLICABLE.value,
        }:
            if not self.completed_by or not self.completed_at:
                raise ValueError(
                    "Completed checklist decisions require "
                    "an actor and timestamp"
                )

        return self


# ============================================================
# MAINTENANCE SAFETY
# ============================================================

class MaintenanceSafety(MaintenanceBaseModel):
    safety_status: MaintenanceSafetyStatus = (
        MaintenanceSafetyStatus.NOT_ASSESSED
    )

    permit_required: bool = False
    permit_id: str | None = None

    lockout_tagout_required: bool = False
    lockout_tagout_verified: bool = False

    safety_reviewed_by: str | None = None
    safety_reviewed_at: datetime | None = None

    hazard_notes: list[str] = Field(
        default_factory=list
    )

    required_ppe: list[str] = Field(
        default_factory=list
    )

    @field_validator("safety_reviewed_at")
    @classmethod
    def validate_reviewed_at(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        return validate_aware(value)

    @model_validator(mode="after")
    def validate_approval(
        self,
    ) -> "MaintenanceSafety":
        if (
            self.safety_status
            == MaintenanceSafetyStatus.APPROVED.value
        ):
            if not (
                self.safety_reviewed_by
                and self.safety_reviewed_at
            ):
                raise ValueError(
                    "Safety approval requires "
                    "reviewer and timestamp"
                )

            if self.permit_required and not self.permit_id:
                raise ValueError(
                    "Required safety permit is missing"
                )

            if (
                self.lockout_tagout_required
                and not self.lockout_tagout_verified
            ):
                raise ValueError(
                    "Required lockout/tagout "
                    "verification is missing"
                )

        return self


# ============================================================
# MAINTENANCE COST
# ============================================================

class MaintenanceCost(MaintenanceBaseModel):
    currency: str = Field(
        default="INR",
        pattern=r"^[A-Z]{3}$",
    )

    spare_parts_cost: Decimal = Field(
        default=Decimal("0"),
        ge=0,
    )

    labor_cost: Decimal = Field(
        default=Decimal("0"),
        ge=0,
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

    inspection_cost: Decimal = Field(
        default=Decimal("0"),
        ge=0,
    )

    other_cost: Decimal = Field(
        default=Decimal("0"),
        ge=0,
    )

    estimated: bool = True

    @property
    def total_cost(self) -> Decimal:
        return money(
            self.spare_parts_cost
            + self.labor_cost
            + self.repair_cost
            + self.replacement_cost
            + self.downtime_cost
            + self.production_loss_cost
            + self.inspection_cost
            + self.other_cost
        )


# ============================================================
# MAINTENANCE SCHEDULE
# ============================================================

class MaintenanceSchedule(MaintenanceBaseModel):
    scheduled_start_at: datetime | None = None
    scheduled_end_at: datetime | None = None

    actual_start_at: datetime | None = None
    actual_end_at: datetime | None = None

    estimated_duration_hours: float | None = Field(
        default=None,
        gt=0,
    )

    recurrence_days: int | None = Field(
        default=None,
        gt=0,
    )

    next_maintenance_at: datetime | None = None

    @field_validator(
        "scheduled_start_at",
        "scheduled_end_at",
        "actual_start_at",
        "actual_end_at",
        "next_maintenance_at",
    )
    @classmethod
    def validate_dates(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        return validate_aware(value)

    @model_validator(mode="after")
    def validate_schedule(
        self,
    ) -> "MaintenanceSchedule":
        if (
            self.scheduled_start_at is not None
            and self.scheduled_end_at is not None
            and self.scheduled_end_at
            < self.scheduled_start_at
        ):
            raise ValueError(
                "Scheduled end precedes start"
            )

        if (
            self.actual_start_at is not None
            and self.actual_end_at is not None
            and self.actual_end_at
            < self.actual_start_at
        ):
            raise ValueError(
                "Actual end precedes start"
            )

        return self


# ============================================================
# MAINTENANCE ACTION
# ============================================================

class MaintenanceAction(MaintenanceBaseModel):
    action_id: UUID = Field(
        default_factory=uuid4
    )

    action_type: MaintenanceActionType

    description: str = Field(
        min_length=1,
        max_length=3000,
    )

    component_id: str | None = None

    assigned_to: str | None = None

    completed: bool = False

    completed_at: datetime | None = None

    notes: str | None = None

    @model_validator(mode="after")
    def validate_action(
        self,
    ) -> "MaintenanceAction":
        if self.completed and self.completed_at is None:
            raise ValueError(
                "Completed action requires timestamp"
            )

        return self


# ============================================================
# MAIN MAINTENANCE MODEL
# ============================================================

class IndustrialMaintenance(MaintenanceBaseModel):
    maintenance_id: UUID = Field(
        default_factory=uuid4
    )

    work_order_code: str = Field(
        min_length=1,
        max_length=100,
    )

    context: MaintenanceContext

    maintenance_type: MaintenanceType

    source: MaintenanceSource

    title: str = Field(
        min_length=1,
        max_length=250,
    )

    description: str | None = None

    status: MaintenanceStatus = (
        MaintenanceStatus.DRAFT
    )

    priority: MaintenancePriority = (
        MaintenancePriority.MEDIUM
    )

    decision: MaintenanceDecision | None = None

    prediction_reference: (
        MaintenancePredictionReference | None
    ) = None

    failure_id: str | None = None
    anomaly_id: str | None = None
    inspection_id: str | None = None
    incident_id: str | None = None

    schedule: MaintenanceSchedule = Field(
        default_factory=MaintenanceSchedule
    )

    technicians: list[
        MaintenanceTechnician
    ] = Field(default_factory=list)

    spare_parts: list[
        MaintenanceSparePart
    ] = Field(default_factory=list)

    checklist: list[
        MaintenanceChecklistItem
    ] = Field(default_factory=list)

    actions: list[MaintenanceAction] = Field(
        default_factory=list
    )

    safety: MaintenanceSafety = Field(
        default_factory=MaintenanceSafety
    )

    estimated_cost: MaintenanceCost = Field(
        default_factory=MaintenanceCost
    )

    actual_cost: MaintenanceCost | None = None

    approval_required: bool = True

    approved_by: str | None = None
    approved_at: datetime | None = None

    verified_by: str | None = None
    verified_at: datetime | None = None

    completion_notes: str | None = None

    created_by: str | None = None

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
        "approved_at",
        "verified_at",
        "created_at",
        "updated_at",
    )
    @classmethod
    def validate_dates(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        return validate_aware(value)

    @model_validator(mode="after")
    def validate_work_order(
        self,
    ) -> "IndustrialMaintenance":
        technician_ids = [
            item.technician_id
            for item in self.technicians
        ]

        if len(technician_ids) != len(
            set(technician_ids)
        ):
            raise ValueError(
                "Duplicate technician assignments"
            )

        if self.approved_at and not self.approved_by:
            raise ValueError(
                "Approval timestamp requires approver"
            )

        return self

    # --------------------------------------------------------
    # SERIALIZATION
    # --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        result = self.model_dump(mode="json")

        result["estimated_total_cost"] = str(
            self.estimated_cost.total_cost
        )

        result["actual_total_cost"] = (
            str(self.actual_cost.total_cost)
            if self.actual_cost is not None
            else None
        )

        return result

    # --------------------------------------------------------
    # INTERNAL
    # --------------------------------------------------------

    def _touch(self) -> None:
        self.updated_at = utc_now()

    # --------------------------------------------------------
    # TECHNICIANS
    # --------------------------------------------------------

    def add_technician(
        self,
        technician: MaintenanceTechnician,
    ) -> None:
        if any(
            item.technician_id
            == technician.technician_id
            for item in self.technicians
        ):
            raise ValueError(
                "Technician already assigned"
            )

        self.technicians.append(technician)
        self._touch()

    # --------------------------------------------------------
    # SPARE PARTS
    # --------------------------------------------------------

    def add_spare_part(
        self,
        part: MaintenanceSparePart,
    ) -> None:
        if any(
            item.part_id == part.part_id
            for item in self.spare_parts
        ):
            raise ValueError(
                "Spare part already added"
            )

        self.spare_parts.append(part)
        self._touch()

    # --------------------------------------------------------
    # CHECKLIST
    # --------------------------------------------------------

    def add_checklist_item(
        self,
        item: MaintenanceChecklistItem,
    ) -> None:
        if any(
            existing.checklist_item_id
            == item.checklist_item_id
            for existing in self.checklist
        ):
            raise ValueError(
                "Duplicate checklist item"
            )

        self.checklist.append(item)
        self._touch()

    # --------------------------------------------------------
    # ACTIONS
    # --------------------------------------------------------

    def add_action(
        self,
        action: MaintenanceAction,
    ) -> None:
        if any(
            item.action_id == action.action_id
            for item in self.actions
        ):
            raise ValueError(
                "Duplicate maintenance action"
            )

        self.actions.append(action)
        self._touch()

    # --------------------------------------------------------
    # APPROVAL
    # --------------------------------------------------------

    def approve(
        self,
        approved_by: str,
    ) -> None:
        if self.status not in {
            MaintenanceStatus.DRAFT.value,
            MaintenanceStatus.PLANNED.value,
            MaintenanceStatus.PENDING_APPROVAL.value,
        }:
            raise ValueError(
                "Maintenance cannot be approved "
                "from its current status"
            )

        if not approved_by.strip():
            raise ValueError(
                "approved_by is required"
            )

        self.status = MaintenanceStatus.APPROVED
        self.approved_by = approved_by
        self.approved_at = utc_now()
        self._touch()

    # --------------------------------------------------------
    # SCHEDULE
    # --------------------------------------------------------

    def schedule_maintenance(
        self,
        start_at: datetime,
        end_at: datetime,
    ) -> None:
        if (
            self.status
            != MaintenanceStatus.APPROVED.value
        ):
            raise ValueError(
                "Maintenance must be approved "
                "before scheduling"
            )

        start_at = validate_aware(start_at)
        end_at = validate_aware(end_at)

        if end_at <= start_at:
            raise ValueError(
                "End time must follow start time"
            )

        self.schedule.scheduled_start_at = start_at
        self.schedule.scheduled_end_at = end_at

        self.status = MaintenanceStatus.SCHEDULED
        self._touch()

    # --------------------------------------------------------
    # START
    # --------------------------------------------------------

    def start(self) -> None:
        if (
            self.status
            != MaintenanceStatus.SCHEDULED.value
        ):
            raise ValueError(
                "Only scheduled maintenance "
                "can be started"
            )

        if not self.technicians:
            raise ValueError(
                "At least one technician is required"
            )

        if (
            self.safety.safety_status
            != MaintenanceSafetyStatus.APPROVED.value
        ):
            raise ValueError(
                "Safety approval is required "
                "before maintenance begins"
            )

        self.schedule.actual_start_at = utc_now()

        self.status = MaintenanceStatus.IN_PROGRESS
        self._touch()

    # --------------------------------------------------------
    # SUBMIT FOR VERIFICATION
    # --------------------------------------------------------

    def submit_for_verification(
        self,
        completion_notes: str,
    ) -> None:
        if (
            self.status
            != MaintenanceStatus.IN_PROGRESS.value
        ):
            raise ValueError(
                "Maintenance must be in progress"
            )

        if not completion_notes.strip():
            raise ValueError(
                "Completion notes are required"
            )

        pending_checklist = [
            item
            for item in self.checklist
            if item.mandatory
            and item.status
            not in {
                ChecklistStatus.PASSED.value,
                ChecklistStatus.NOT_APPLICABLE.value,
            }
        ]

        if pending_checklist:
            raise ValueError(
                "Mandatory checklist items "
                "are not satisfied"
            )

        incomplete_actions = [
            action
            for action in self.actions
            if not action.completed
        ]

        if incomplete_actions:
            raise ValueError(
                "Maintenance actions are incomplete"
            )

        self.completion_notes = completion_notes

        self.schedule.actual_end_at = utc_now()

        self.status = (
            MaintenanceStatus.PENDING_VERIFICATION
        )

        self._touch()

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    def complete(
        self,
        verified_by: str,
    ) -> None:
        if (
            self.status
            != MaintenanceStatus.PENDING_VERIFICATION.value
        ):
            raise ValueError(
                "Maintenance must be pending verification"
            )

        if not verified_by.strip():
            raise ValueError(
                "verified_by is required"
            )

        self.verified_by = verified_by
        self.verified_at = utc_now()

        self.status = MaintenanceStatus.COMPLETED
        self._touch()

    # --------------------------------------------------------
    # FINANCIAL HELPERS
    # --------------------------------------------------------

    def spare_parts_total(self) -> Decimal:
        return money(
            sum(
                (
                    part.total_cost
                    for part in self.spare_parts
                ),
                Decimal("0"),
            )
        )

    def assigned_labor_total(self) -> Decimal:
        return money(
            sum(
                (
                    technician.estimated_labor_cost
                    for technician in self.technicians
                ),
                Decimal("0"),
            )
        )

    def cost_variance(self) -> Decimal | None:
        if self.actual_cost is None:
            return None

        return money(
            self.actual_cost.total_cost
            - self.estimated_cost.total_cost
        )

    # --------------------------------------------------------
    # STATUS HELPERS
    # --------------------------------------------------------

    def is_overdue(
        self,
        reference_time: datetime | None = None,
    ) -> bool:
        if self.status in {
            MaintenanceStatus.COMPLETED.value,
            MaintenanceStatus.CANCELLED.value,
        }:
            return False

        if self.schedule.scheduled_end_at is None:
            return False

        now = reference_time or utc_now()
        validate_aware(now)

        return now > self.schedule.scheduled_end_at


# ============================================================
# CREATE SCHEMA
# ============================================================

class MaintenanceCreate(MaintenanceBaseModel):
    work_order_code: str = Field(min_length=1)

    context: MaintenanceContext

    maintenance_type: MaintenanceType

    source: MaintenanceSource

    title: str = Field(min_length=1)

    description: str | None = None

    priority: MaintenancePriority = (
        MaintenancePriority.MEDIUM
    )

    decision: MaintenanceDecision | None = None

    prediction_reference: (
        MaintenancePredictionReference | None
    ) = None

    created_by: str | None = None


# ============================================================
# UPDATE SCHEMA
# ============================================================

class MaintenanceUpdate(MaintenanceBaseModel):
    title: str | None = None
    description: str | None = None

    priority: MaintenancePriority | None = None

    decision: MaintenanceDecision | None = None

    completion_notes: str | None = None

    tags: list[str] | None = None


# ============================================================
# RESPONSE SCHEMA
# ============================================================

class MaintenanceResponse(IndustrialMaintenance):
    pass


# ============================================================
# SUMMARY
# ============================================================

class MaintenanceSummary(MaintenanceBaseModel):
    maintenance_id: UUID

    work_order_code: str

    company_id: str
    asset_id: str

    equipment_id: str | None
    component_id: str | None

    maintenance_type: MaintenanceType

    title: str

    status: MaintenanceStatus
    priority: MaintenancePriority

    technician_count: int

    estimated_total_cost: Decimal
    actual_total_cost: Decimal | None

    scheduled_start_at: datetime | None
    scheduled_end_at: datetime | None

    is_overdue: bool

    created_at: datetime

    @classmethod
    def from_maintenance(
        cls,
        maintenance: IndustrialMaintenance,
    ) -> "MaintenanceSummary":
        return cls(
            maintenance_id=maintenance.maintenance_id,
            work_order_code=maintenance.work_order_code,
            company_id=maintenance.context.company_id,
            asset_id=maintenance.context.asset_id,
            equipment_id=maintenance.context.equipment_id,
            component_id=maintenance.context.component_id,
            maintenance_type=maintenance.maintenance_type,
            title=maintenance.title,
            status=maintenance.status,
            priority=maintenance.priority,
            technician_count=len(maintenance.technicians),
            estimated_total_cost=(
                maintenance.estimated_cost.total_cost
            ),
            actual_total_cost=(
                maintenance.actual_cost.total_cost
                if maintenance.actual_cost is not None
                else None
            ),
            scheduled_start_at=(
                maintenance.schedule.scheduled_start_at
            ),
            scheduled_end_at=(
                maintenance.schedule.scheduled_end_at
            ),
            is_overdue=maintenance.is_overdue(),
            created_at=maintenance.created_at,
        )


# ============================================================
# EXAMPLE
# ============================================================

def example() -> None:
    import json
    from datetime import timedelta

    work_order = IndustrialMaintenance(
        work_order_code="WO-2026-001",
        context=MaintenanceContext(
            company_id="company-001",
            site_id="SITE-01",
            asset_id="ASSET-101",
            equipment_id="PUMP-101",
            machine_id="PUMP-101",
            component_id="BEARING-101",
        ),
        maintenance_type=MaintenanceType.PREDICTIVE,
        source=MaintenanceSource.FAILURE_MODEL,
        title="Pump Bearing Preventive Replacement",
        description=(
            "Inspect the bearing and replace it "
            "if confirmed wear exceeds limits."
        ),
        priority=MaintenancePriority.HIGH,
        decision=MaintenanceDecision.ENGINEERING_REVIEW,
        prediction_reference=MaintenancePredictionReference(
            prediction_id="PRED-101",
            model_id="bearing-failure-model",
            model_version="1.0.0",
            failure_probability=0.82,
            prediction_horizon_hours=168,
            remaining_useful_life_hours=450,
        ),
        estimated_cost=MaintenanceCost(
            spare_parts_cost=Decimal("12000"),
            labor_cost=Decimal("5000"),
            downtime_cost=Decimal("8000"),
        ),
    )

    work_order.add_technician(
        MaintenanceTechnician(
            technician_id="TECH-001",
            name="Maintenance Engineer",
            role=TechnicianRole.MAINTENANCE_ENGINEER,
            hourly_rate=Decimal("500"),
            assigned_hours=4,
        )
    )

    work_order.add_spare_part(
        MaintenanceSparePart(
            part_id="PART-001",
            part_name="Pump Bearing",
            quantity=1,
            unit_cost=Decimal("12000"),
        )
    )

    work_order.add_checklist_item(
        MaintenanceChecklistItem(
            description="Inspect bearing condition",
        )
    )

    work_order.approve(
        approved_by="manager-001"
    )

    start_at = utc_now() + timedelta(days=1)

    work_order.schedule_maintenance(
        start_at=start_at,
        end_at=start_at + timedelta(hours=4),
    )

    print(
        json.dumps(
            MaintenanceSummary.from_maintenance(
                work_order
            ).model_dump(mode="json"),
            indent=2,
        )
    )


if __name__ == "__main__":
    example()