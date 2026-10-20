"""
app/ai/ml/models/inspection.py

Industrial Inspection Intelligence Models
=========================================

Python 3.11+
Pydantic v2

Install:
    pip install "pydantic>=2,<3"

Features:
- Industrial asset and component inspections
- Manual, sensor, CV and thermal inspections
- Multi-tenant inspection records
- Inspection scheduling and lifecycle
- AI detection and model provenance
- Crack, corrosion, damage and defect findings
- Bounding boxes and image references
- Inspection measurements
- Equipment visual health scoring
- Inspection checklists
- Repair and replacement recommendations
- Inspector review and approval
- Follow-up inspection tracking
- Create, update and response schemas

This module defines domain models only.
Computer vision inference, persistence,
notifications and equipment control
belong in separate application modules.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
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


# ============================================================
# ENUMS
# ============================================================

class InspectionType(str, Enum):
    VISUAL = "visual"
    AI_VISION = "ai_vision"
    THERMAL = "thermal"
    VIBRATION = "vibration"
    ULTRASONIC = "ultrasonic"
    ELECTRICAL = "electrical"
    MECHANICAL = "mechanical"
    PRESSURE = "pressure"
    LEAKAGE = "leakage"
    LUBRICATION = "lubrication"
    STRUCTURAL = "structural"
    SAFETY = "safety"
    PPE = "ppe"
    GENERAL = "general"


class InspectionStatus(str, Enum):
    SCHEDULED = "scheduled"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class InspectionSource(str, Enum):
    HUMAN = "human"
    COMPUTER_VISION = "computer_vision"
    SENSOR = "sensor"
    ROBOT = "robot"
    DRONE = "drone"
    AUTOMATED_SYSTEM = "automated_system"


class InspectionSeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class InspectionFindingType(str, Enum):
    CRACK = "crack"
    CORROSION = "corrosion"
    DAMAGE = "damage"
    DEFECT = "defect"
    LEAKAGE = "leakage"
    OVERHEATING = "overheating"
    WEAR = "wear"
    MISALIGNMENT = "misalignment"
    IMBALANCE = "imbalance"
    LUBRICATION_ISSUE = "lubrication_issue"
    ELECTRICAL_FAULT = "electrical_fault"
    STRUCTURAL_DAMAGE = "structural_damage"
    CONTAMINATION = "contamination"
    PPE_VIOLATION = "ppe_violation"
    OTHER = "other"


class InspectionHealthStatus(str, Enum):
    HEALTHY = "healthy"
    WATCH = "watch"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    NOT_ASSESSED = "not_assessed"


class InspectionRecommendationType(str, Enum):
    MONITOR = "monitor"
    REINSPECT = "reinspect"
    CLEAN = "clean"
    LUBRICATE = "lubricate"
    CALIBRATE = "calibrate"
    ADJUST = "adjust"
    REPAIR = "repair"
    REPLACE = "replace"
    ENGINEERING_REVIEW = "engineering_review"
    REQUEST_SHUTDOWN_REVIEW = (
        "request_shutdown_review"
    )


class InspectionPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class ChecklistResult(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"
    NOT_CHECKED = "not_checked"


# ============================================================
# BASE MODEL
# ============================================================

class InspectionBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True,
        str_strip_whitespace=True,
    )


# ============================================================
# INSPECTION CONTEXT
# ============================================================

class InspectionContext(InspectionBaseModel):
    company_id: str = Field(
        min_length=1,
        max_length=100,
    )

    site_id: str = Field(
        min_length=1,
        max_length=100,
    )

    asset_id: str = Field(
        min_length=1,
        max_length=100,
    )

    equipment_id: str | None = None
    component_id: str | None = None

    production_line_id: str | None = None
    zone_id: str | None = None

    work_order_id: str | None = None
    maintenance_id: str | None = None

    location_description: str | None = None


# ============================================================
# AI MODEL REFERENCE
# ============================================================

class InspectionModelReference(InspectionBaseModel):
    model_id: str = Field(
        min_length=1,
        max_length=200,
    )

    model_name: str = Field(
        min_length=1,
        max_length=200,
    )

    model_version: str = Field(
        min_length=1,
        max_length=100,
    )

    prediction_id: str | None = None

    confidence_threshold: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    inference_duration_ms: float | None = Field(
        default=None,
        ge=0,
    )


# ============================================================
# BOUNDING BOX
# ============================================================

class InspectionBoundingBox(InspectionBaseModel):
    x_min: float = Field(ge=0, le=1)
    y_min: float = Field(ge=0, le=1)
    x_max: float = Field(ge=0, le=1)
    y_max: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_box(
        self,
    ) -> "InspectionBoundingBox":
        if self.x_max <= self.x_min:
            raise ValueError(
                "x_max must exceed x_min"
            )

        if self.y_max <= self.y_min:
            raise ValueError(
                "y_max must exceed y_min"
            )

        return self


# ============================================================
# IMAGE EVIDENCE
# ============================================================

class InspectionImage(InspectionBaseModel):
    image_id: UUID = Field(
        default_factory=uuid4
    )

    image_reference: str = Field(
        min_length=1,
        max_length=2048,
    )

    image_type: str = "rgb"

    captured_at: datetime = Field(
        default_factory=utc_now
    )

    camera_id: str | None = None

    description: str | None = None

    @field_validator("captured_at")
    @classmethod
    def validate_captured_at(
        cls,
        value: datetime,
    ) -> datetime:
        if value.tzinfo is None:
            raise ValueError(
                "captured_at must be timezone-aware"
            )

        return value


# ============================================================
# MEASUREMENT
# ============================================================

class InspectionMeasurement(InspectionBaseModel):
    measurement_id: UUID = Field(
        default_factory=uuid4
    )

    measurement_type: str = Field(
        min_length=1,
        max_length=100,
    )

    value: float

    unit: str = Field(
        min_length=1,
        max_length=50,
    )

    lower_limit: float | None = None
    upper_limit: float | None = None

    within_limits: bool | None = None

    sensor_id: str | None = None

    measured_at: datetime = Field(
        default_factory=utc_now
    )

    @model_validator(mode="after")
    def validate_limits(
        self,
    ) -> "InspectionMeasurement":
        import math

        numbers = (
            self.value,
            self.lower_limit,
            self.upper_limit,
        )

        if any(
            number is not None
            and not math.isfinite(number)
            for number in numbers
        ):
            raise ValueError(
                "Measurement values must be finite"
            )

        if (
            self.lower_limit is not None
            and self.upper_limit is not None
            and self.lower_limit > self.upper_limit
        ):
            raise ValueError(
                "lower_limit exceeds upper_limit"
            )

        if (
            self.lower_limit is not None
            or self.upper_limit is not None
        ):
            computed = (
                (
                    self.lower_limit is None
                    or self.value >= self.lower_limit
                )
                and (
                    self.upper_limit is None
                    or self.value <= self.upper_limit
                )
            )

            if (
                self.within_limits is not None
                and self.within_limits != computed
            ):
                raise ValueError(
                    "within_limits conflicts "
                    "with configured limits"
                )

            self.within_limits = computed

        return self


# ============================================================
# INSPECTION FINDING
# ============================================================

class InspectionFinding(InspectionBaseModel):
    finding_id: UUID = Field(
        default_factory=uuid4
    )

    finding_type: InspectionFindingType

    severity: InspectionSeverity

    description: str = Field(
        min_length=1,
        max_length=3000,
    )

    confidence: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    affected_area_percent: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    bounding_box: InspectionBoundingBox | None = None

    image_id: UUID | None = None

    component_id: str | None = None

    model_reference: (
        InspectionModelReference | None
    ) = None

    verified: bool = False

    verified_by: str | None = None
    verified_at: datetime | None = None

    requires_human_review: bool = True

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def validate_verification(
        self,
    ) -> "InspectionFinding":
        if self.verified and not (
            self.verified_by
            and self.verified_at
        ):
            raise ValueError(
                "Verified findings require "
                "reviewer and timestamp"
            )

        return self


# ============================================================
# INSPECTION CHECKLIST
# ============================================================

class InspectionChecklistItem(InspectionBaseModel):
    checklist_item_id: UUID = Field(
        default_factory=uuid4
    )

    description: str = Field(
        min_length=1,
        max_length=1000,
    )

    result: ChecklistResult = (
        ChecklistResult.NOT_CHECKED
    )

    notes: str | None = None

    checked_by: str | None = None

    checked_at: datetime | None = None


# ============================================================
# HEALTH ASSESSMENT
# ============================================================

class InspectionHealthAssessment(InspectionBaseModel):
    health_score: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    health_status: InspectionHealthStatus = (
        InspectionHealthStatus.NOT_ASSESSED
    )

    inspection_coverage_percent: float = Field(
        default=0,
        ge=0,
        le=100,
    )

    scoring_method: str | None = None

    scoring_model_version: str | None = None

    explanation: str | None = None

    assessed_at: datetime = Field(
        default_factory=utc_now
    )

    @model_validator(mode="after")
    def validate_assessment(
        self,
    ) -> "InspectionHealthAssessment":
        if (
            self.health_status
            == InspectionHealthStatus.NOT_ASSESSED.value
            and self.health_score is not None
        ):
            raise ValueError(
                "Unassessed inspections cannot "
                "have a health score"
            )

        if (
            self.health_status
            != InspectionHealthStatus.NOT_ASSESSED.value
            and self.health_score is None
        ):
            raise ValueError(
                "Assessed inspections require "
                "a health score"
            )

        return self


# ============================================================
# RECOMMENDATION
# ============================================================

class InspectionRecommendation(InspectionBaseModel):
    recommendation_id: UUID = Field(
        default_factory=uuid4
    )

    recommendation_type: (
        InspectionRecommendationType
    )

    priority: InspectionPriority

    description: str = Field(
        min_length=1,
        max_length=3000,
    )

    related_finding_ids: list[UUID] = Field(
        default_factory=list
    )

    recommended_within_hours: float | None = Field(
        default=None,
        ge=0,
    )

    estimated_cost: Decimal | None = Field(
        default=None,
        ge=0,
    )

    estimated_downtime_hours: float | None = Field(
        default=None,
        ge=0,
    )

    currency: str = Field(
        default="INR",
        pattern=r"^[A-Z]{3}$",
    )

    requires_approval: bool = True

    approved_by: str | None = None

    work_order_id: str | None = None

    @model_validator(mode="after")
    def validate_shutdown_request(
        self,
    ) -> "InspectionRecommendation":
        if (
            self.recommendation_type
            == InspectionRecommendationType
            .REQUEST_SHUTDOWN_REVIEW.value
            and not self.requires_approval
        ):
            raise ValueError(
                "Shutdown review requests "
                "must require approval"
            )

        return self


# ============================================================
# INSPECTION REVIEW
# ============================================================

class InspectionReview(InspectionBaseModel):
    review_id: UUID = Field(
        default_factory=uuid4
    )

    reviewer_id: str = Field(
        min_length=1,
        max_length=100,
    )

    approved: bool

    notes: str | None = None

    reviewed_at: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# MAIN INSPECTION MODEL
# ============================================================

class IndustrialInspection(InspectionBaseModel):
    inspection_id: UUID = Field(
        default_factory=uuid4
    )

    inspection_code: str = Field(
        min_length=1,
        max_length=100,
    )

    context: InspectionContext

    inspection_type: InspectionType

    source: InspectionSource

    title: str = Field(
        min_length=1,
        max_length=250,
    )

    description: str | None = None

    status: InspectionStatus = (
        InspectionStatus.SCHEDULED
    )

    priority: InspectionPriority = (
        InspectionPriority.MEDIUM
    )

    assigned_inspector_id: str | None = None

    performed_by: str | None = None

    scheduled_at: datetime | None = None

    started_at: datetime | None = None

    completed_at: datetime | None = None

    approved_at: datetime | None = None

    images: list[InspectionImage] = Field(
        default_factory=list
    )

    measurements: list[
        InspectionMeasurement
    ] = Field(
        default_factory=list
    )

    findings: list[InspectionFinding] = Field(
        default_factory=list
    )

    checklist: list[
        InspectionChecklistItem
    ] = Field(
        default_factory=list
    )

    health_assessment: (
        InspectionHealthAssessment
    ) = Field(
        default_factory=InspectionHealthAssessment
    )

    recommendations: list[
        InspectionRecommendation
    ] = Field(
        default_factory=list
    )

    reviews: list[InspectionReview] = Field(
        default_factory=list
    )

    follow_up_required: bool = False

    next_inspection_at: datetime | None = None

    requires_human_review: bool = True

    notes: str | None = None

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
        "scheduled_at",
        "started_at",
        "completed_at",
        "approved_at",
        "next_inspection_at",
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
                "Inspection timestamps "
                "must be timezone-aware"
            )

        return value

    @model_validator(mode="after")
    def validate_dates(
        self,
    ) -> "IndustrialInspection":
        if (
            self.started_at is not None
            and self.completed_at is not None
            and self.completed_at < self.started_at
        ):
            raise ValueError(
                "Inspection completion precedes start"
            )

        return self

    # --------------------------------------------------------
    # SERIALIZATION
    # --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")

        data["finding_count"] = (
            self.finding_count
        )

        data["critical_finding_count"] = (
            self.critical_finding_count
        )

        return data

    # --------------------------------------------------------
    # DERIVED PROPERTIES
    # --------------------------------------------------------

    @property
    def finding_count(self) -> int:
        return len(self.findings)

    @property
    def critical_finding_count(self) -> int:
        return sum(
            finding.severity
            == InspectionSeverity.CRITICAL.value
            for finding in self.findings
        )

    @property
    def high_severity_finding_count(self) -> int:
        return sum(
            finding.severity in {
                InspectionSeverity.HIGH.value,
                InspectionSeverity.CRITICAL.value,
            }
            for finding in self.findings
        )

    # --------------------------------------------------------
    # INTERNAL
    # --------------------------------------------------------

    def _touch(self) -> None:
        self.updated_at = utc_now()

    # --------------------------------------------------------
    # ADD IMAGE
    # --------------------------------------------------------

    def add_image(
        self,
        image: InspectionImage,
    ) -> None:
        if any(
            item.image_id == image.image_id
            for item in self.images
        ):
            raise ValueError(
                "Duplicate image ID"
            )

        self.images.append(image)

        self._touch()

    # --------------------------------------------------------
    # ADD FINDING
    # --------------------------------------------------------

    def add_finding(
        self,
        finding: InspectionFinding,
    ) -> None:
        if any(
            item.finding_id == finding.finding_id
            for item in self.findings
        ):
            raise ValueError(
                "Duplicate finding ID"
            )

        if (
            finding.image_id is not None
            and not any(
                image.image_id == finding.image_id
                for image in self.images
            )
        ):
            raise ValueError(
                "Finding references an image "
                "not attached to this inspection"
            )

        self.findings.append(finding)

        self._touch()

    # --------------------------------------------------------
    # ADD MEASUREMENT
    # --------------------------------------------------------

    def add_measurement(
        self,
        measurement: InspectionMeasurement,
    ) -> None:
        if any(
            item.measurement_id
            == measurement.measurement_id
            for item in self.measurements
        ):
            raise ValueError(
                "Duplicate measurement ID"
            )

        self.measurements.append(measurement)

        self._touch()

    # --------------------------------------------------------
    # ADD RECOMMENDATION
    # --------------------------------------------------------

    def add_recommendation(
        self,
        recommendation: InspectionRecommendation,
    ) -> None:
        if any(
            item.recommendation_id
            == recommendation.recommendation_id
            for item in self.recommendations
        ):
            raise ValueError(
                "Duplicate recommendation ID"
            )

        known_findings = {
            finding.finding_id
            for finding in self.findings
        }

        if not set(
            recommendation.related_finding_ids
        ).issubset(known_findings):
            raise ValueError(
                "Recommendation references "
                "unknown findings"
            )

        self.recommendations.append(
            recommendation
        )

        self._touch()

    # --------------------------------------------------------
    # START INSPECTION
    # --------------------------------------------------------

    def start(
        self,
        inspector_id: str,
    ) -> None:
        if self.status not in {
            InspectionStatus.SCHEDULED.value,
            InspectionStatus.ASSIGNED.value,
        }:
            raise ValueError(
                "Inspection cannot be started "
                "from its current status"
            )

        if not inspector_id.strip():
            raise ValueError(
                "inspector_id is required"
            )

        self.status = (
            InspectionStatus.IN_PROGRESS
        )

        self.performed_by = inspector_id
        self.started_at = utc_now()

        self._touch()

    # --------------------------------------------------------
    # SUBMIT FOR REVIEW
    # --------------------------------------------------------

    def submit_for_review(self) -> None:
        if (
            self.status
            != InspectionStatus.IN_PROGRESS.value
        ):
            raise ValueError(
                "Only in-progress inspections "
                "can be submitted"
            )

        self.status = (
            InspectionStatus.PENDING_REVIEW
        )

        self.completed_at = utc_now()

        self._touch()

    # --------------------------------------------------------
    # APPROVE
    # --------------------------------------------------------

    def approve(
        self,
        reviewer_id: str,
        notes: str | None = None,
    ) -> None:
        if (
            self.status
            != InspectionStatus.PENDING_REVIEW.value
        ):
            raise ValueError(
                "Inspection must be pending review"
            )

        if not reviewer_id.strip():
            raise ValueError(
                "reviewer_id is required"
            )

        self.reviews.append(
            InspectionReview(
                reviewer_id=reviewer_id,
                approved=True,
                notes=notes,
            )
        )

        self.status = InspectionStatus.APPROVED
        self.approved_at = utc_now()

        self._touch()

    # --------------------------------------------------------
    # REJECT
    # --------------------------------------------------------

    def reject(
        self,
        reviewer_id: str,
        reason: str,
    ) -> None:
        if (
            self.status
            != InspectionStatus.PENDING_REVIEW.value
        ):
            raise ValueError(
                "Inspection must be pending review"
            )

        if not reason.strip():
            raise ValueError(
                "Rejection reason is required"
            )

        self.reviews.append(
            InspectionReview(
                reviewer_id=reviewer_id,
                approved=False,
                notes=reason,
            )
        )

        self.status = InspectionStatus.REJECTED

        self._touch()

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    def complete(self) -> None:
        if (
            self.status
            != InspectionStatus.APPROVED.value
        ):
            raise ValueError(
                "Inspection requires approval "
                "before completion"
            )

        self.status = InspectionStatus.COMPLETED

        self._touch()

    # --------------------------------------------------------
    # SAFETY REVIEW
    # --------------------------------------------------------

    def needs_engineering_review(self) -> bool:
        return (
            self.high_severity_finding_count > 0
            or self.health_assessment.health_status
            == InspectionHealthStatus.CRITICAL.value
            or self.follow_up_required
        )


# ============================================================
# CREATE SCHEMA
# ============================================================

class InspectionCreate(InspectionBaseModel):
    inspection_code: str = Field(
        min_length=1,
        max_length=100,
    )

    context: InspectionContext

    inspection_type: InspectionType

    source: InspectionSource

    title: str = Field(
        min_length=1,
        max_length=250,
    )

    description: str | None = None

    priority: InspectionPriority = (
        InspectionPriority.MEDIUM
    )

    assigned_inspector_id: str | None = None

    scheduled_at: datetime | None = None


# ============================================================
# UPDATE SCHEMA
# ============================================================

class InspectionUpdate(InspectionBaseModel):
    title: str | None = None
    description: str | None = None

    priority: InspectionPriority | None = None

    assigned_inspector_id: str | None = None

    follow_up_required: bool | None = None

    next_inspection_at: datetime | None = None

    notes: str | None = None

    tags: list[str] | None = None


# ============================================================
# RESPONSE SCHEMA
# ============================================================

class InspectionResponse(IndustrialInspection):
    pass


# ============================================================
# SUMMARY
# ============================================================

class InspectionSummary(InspectionBaseModel):
    inspection_id: UUID

    inspection_code: str

    company_id: str
    asset_id: str

    equipment_id: str | None
    component_id: str | None

    inspection_type: InspectionType

    title: str

    status: InspectionStatus
    priority: InspectionPriority

    finding_count: int
    critical_finding_count: int

    health_score: float | None
    health_status: InspectionHealthStatus

    needs_engineering_review: bool

    created_at: datetime

    @classmethod
    def from_inspection(
        cls,
        inspection: IndustrialInspection,
    ) -> "InspectionSummary":
        return cls(
            inspection_id=inspection.inspection_id,
            inspection_code=inspection.inspection_code,
            company_id=(
                inspection.context.company_id
            ),
            asset_id=inspection.context.asset_id,
            equipment_id=(
                inspection.context.equipment_id
            ),
            component_id=(
                inspection.context.component_id
            ),
            inspection_type=inspection.inspection_type,
            title=inspection.title,
            status=inspection.status,
            priority=inspection.priority,
            finding_count=inspection.finding_count,
            critical_finding_count=(
                inspection.critical_finding_count
            ),
            health_score=(
                inspection.health_assessment.health_score
            ),
            health_status=(
                inspection.health_assessment.health_status
            ),
            needs_engineering_review=(
                inspection.needs_engineering_review()
            ),
            created_at=inspection.created_at,
        )


# ============================================================
# EXAMPLE
# ============================================================

def example() -> None:
    import json

    inspection = IndustrialInspection(
        inspection_code="INSP-2026-001",
        context=InspectionContext(
            company_id="company-001",
            site_id="SITE-01",
            asset_id="ASSET-101",
            equipment_id="PUMP-101",
            component_id="BEARING-101",
        ),
        inspection_type=InspectionType.AI_VISION,
        source=InspectionSource.COMPUTER_VISION,
        title="Pump Bearing Visual Inspection",
        priority=InspectionPriority.HIGH,
    )

    inspection.start(
        inspector_id="inspector-001"
    )

    image = InspectionImage(
        image_reference=(
            "company-001/inspections/"
            "bearing-101.jpg"
        ),
    )

    inspection.add_image(image)

    # Illustrative CV output only.
    finding = InspectionFinding(
        finding_type=InspectionFindingType.CRACK,
        severity=InspectionSeverity.HIGH,
        description="Candidate surface crack",
        confidence=0.91,
        image_id=image.image_id,
        bounding_box=InspectionBoundingBox(
            x_min=0.20,
            y_min=0.25,
            x_max=0.50,
            y_max=0.60,
        ),
        model_reference=InspectionModelReference(
            model_id="crack-detector",
            model_name="Industrial Crack Detector",
            model_version="1.0.0",
        ),
    )

    inspection.add_finding(finding)

    inspection.add_recommendation(
        InspectionRecommendation(
            recommendation_type=(
                InspectionRecommendationType
                .ENGINEERING_REVIEW
            ),
            priority=InspectionPriority.HIGH,
            description=(
                "Request qualified engineering "
                "inspection of the suspected crack."
            ),
            related_finding_ids=[
                finding.finding_id
            ],
        )
    )

    inspection.submit_for_review()

    inspection.approve(
        reviewer_id="engineer-001",
        notes="Inspection report reviewed",
    )

    inspection.complete()

    print(
        json.dumps(
            InspectionSummary.from_inspection(
                inspection
            ).model_dump(mode="json"),
            indent=2,
        )
    )


if __name__ == "__main__":
    example()