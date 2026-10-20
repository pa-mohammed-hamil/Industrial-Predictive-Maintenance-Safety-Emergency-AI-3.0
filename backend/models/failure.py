"""
app/ai/ml/models/failure.py

Industrial Equipment Failure Intelligence Models
================================================

Python 3.11+
Pydantic v2

Install:
    pip install "pydantic>=2,<3"

Features:
- Multi-tenant failure records
- Asset, machine and component relationships
- Failure prediction results
- Failure probability and prediction horizon
- Failure modes and root-cause analysis
- Failure severity and criticality
- Sensor and visual evidence
- Failure risk assessments
- Remaining useful life (RUL)
- Maintenance recommendations
- Repair and replacement estimates
- Actual failure event tracking
- Model version and prediction provenance
- Create, update and response schemas

NOTE:
    This module defines validated domain models.
    It does not train or execute ML models.

    Predicted probabilities and RUL estimates
    must come from separately validated models.
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

class FailureType(str, Enum):
    MECHANICAL = "mechanical"
    ELECTRICAL = "electrical"
    THERMAL = "thermal"
    HYDRAULIC = "hydraulic"
    PNEUMATIC = "pneumatic"
    STRUCTURAL = "structural"
    SOFTWARE = "software"
    CONTROL_SYSTEM = "control_system"
    SENSOR = "sensor"
    PROCESS = "process"
    OTHER = "other"


class FailureMode(str, Enum):
    BEARING_FAILURE = "bearing_failure"
    GEAR_FAILURE = "gear_failure"
    MOTOR_FAILURE = "motor_failure"
    SHAFT_FAILURE = "shaft_failure"
    SEAL_FAILURE = "seal_failure"
    PUMP_FAILURE = "pump_failure"
    COMPRESSOR_FAILURE = "compressor_failure"
    VALVE_FAILURE = "valve_failure"
    OVERHEATING = "overheating"
    OVERLOAD = "overload"
    VIBRATION = "vibration"
    MISALIGNMENT = "misalignment"
    IMBALANCE = "imbalance"
    LUBRICATION_FAILURE = "lubrication_failure"
    CORROSION = "corrosion"
    FATIGUE = "fatigue"
    CRACK = "crack"
    LEAKAGE = "leakage"
    SHORT_CIRCUIT = "short_circuit"
    INSULATION_FAILURE = "insulation_failure"
    POWER_FAILURE = "power_failure"
    SENSOR_FAILURE = "sensor_failure"
    CONTROLLER_FAILURE = "controller_failure"
    BLOCKAGE = "blockage"
    CONTAMINATION = "contamination"
    UNKNOWN = "unknown"


class FailureSeverity(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    CATASTROPHIC = "catastrophic"


class FailureStatus(str, Enum):
    SUSPECTED = "suspected"
    PREDICTED = "predicted"
    DETECTED = "detected"
    CONFIRMED = "confirmed"
    UNDER_INVESTIGATION = "under_investigation"
    UNDER_REPAIR = "under_repair"
    RESOLVED = "resolved"
    CLOSED = "closed"
    FALSE_POSITIVE = "false_positive"


class FailureRiskLevel(str, Enum):
    UNKNOWN = "unknown"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class FailureEvidenceType(str, Enum):
    SENSOR = "sensor"
    TELEMETRY = "telemetry"
    VISUAL_INSPECTION = "visual_inspection"
    THERMAL_IMAGE = "thermal_image"
    VIBRATION_ANALYSIS = "vibration_analysis"
    MAINTENANCE_HISTORY = "maintenance_history"
    MODEL_PREDICTION = "model_prediction"
    HUMAN_INSPECTION = "human_inspection"


class FailureActionType(str, Enum):
    MONITOR = "monitor"
    INSPECT = "inspect"
    LUBRICATE = "lubricate"
    CALIBRATE = "calibrate"
    ADJUST = "adjust"
    REPAIR = "repair"
    REPLACE = "replace"
    OVERHAUL = "overhaul"
    ENGINEERING_REVIEW = "engineering_review"
    REQUEST_SAFE_SHUTDOWN = "request_safe_shutdown"


class FailureActionPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class PredictionSource(str, Enum):
    ML_MODEL = "ml_model"
    DEEP_LEARNING = "deep_learning"
    ENSEMBLE = "ensemble"
    RULE_ENGINE = "rule_engine"
    HUMAN_ASSESSMENT = "human_assessment"


# ============================================================
# BASE MODEL
# ============================================================

class FailureBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True,
        str_strip_whitespace=True,
    )


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ============================================================
# FAILURE CONTEXT
# ============================================================

class FailureContext(FailureBaseModel):
    company_id: str = Field(
        min_length=1,
        max_length=100,
    )

    asset_id: str = Field(
        min_length=1,
        max_length=100,
    )

    equipment_id: str | None = None
    component_id: str | None = None
    site_id: str | None = None
    production_line_id: str | None = None

    operating_hours: float | None = Field(
        default=None,
        ge=0,
    )

    operating_cycles: int | None = Field(
        default=None,
        ge=0,
    )


# ============================================================
# FAILURE EVIDENCE
# ============================================================

class FailureEvidence(FailureBaseModel):
    evidence_id: UUID = Field(
        default_factory=uuid4
    )

    evidence_type: FailureEvidenceType

    source_id: str | None = None

    description: str | None = None

    observed_value: float | None = None
    expected_value: float | None = None

    unit: str | None = None

    confidence: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    reference_uri: str | None = None

    observed_at: datetime = Field(
        default_factory=utc_now
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


# ============================================================
# FAILURE PREDICTION
# ============================================================

class FailurePrediction(FailureBaseModel):
    prediction_id: UUID = Field(
        default_factory=uuid4
    )

    context: FailureContext

    failure_type: FailureType

    predicted_failure_mode: FailureMode = (
        FailureMode.UNKNOWN
    )

    failure_probability: float = Field(
        ge=0,
        le=1,
    )

    prediction_horizon_hours: float = Field(
        gt=0,
    )

    remaining_useful_life_hours: float | None = Field(
        default=None,
        ge=0,
    )

    predicted_time_to_failure_hours: (
        float | None
    ) = Field(
        default=None,
        ge=0,
    )

    model_id: str = Field(
        min_length=1,
        max_length=200,
    )

    model_version: str = Field(
        min_length=1,
        max_length=100,
    )

    source: PredictionSource = (
        PredictionSource.ML_MODEL
    )

    risk_level: FailureRiskLevel = (
        FailureRiskLevel.UNKNOWN
    )

    decision_threshold: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    confidence_score: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    evidence: list[FailureEvidence] = Field(
        default_factory=list
    )

    feature_importance: dict[str, float] = Field(
        default_factory=dict
    )

    explanation: str | None = None

    predicted_at: datetime = Field(
        default_factory=utc_now
    )

    requires_human_review: bool = True

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    @field_validator("predicted_at")
    @classmethod
    def validate_predicted_at(
        cls,
        value: datetime,
    ) -> datetime:
        if value.tzinfo is None:
            raise ValueError(
                "predicted_at must be timezone-aware"
            )

        return value

    @field_validator("feature_importance")
    @classmethod
    def validate_importance(
        cls,
        values: dict[str, float],
    ) -> dict[str, float]:
        import math

        if any(
            not math.isfinite(value)
            for value in values.values()
        ):
            raise ValueError(
                "Feature importance must be finite"
            )

        return values

    def exceeds_threshold(self) -> bool | None:
        if self.decision_threshold is None:
            return None

        return (
            self.failure_probability
            >= self.decision_threshold
        )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


# ============================================================
# FAILURE RISK ASSESSMENT
# ============================================================

class FailureRiskAssessment(FailureBaseModel):
    assessment_id: UUID = Field(
        default_factory=uuid4
    )

    context: FailureContext

    prediction_id: UUID | None = None

    failure_probability: float = Field(
        ge=0,
        le=1,
    )

    consequence_severity: FailureSeverity

    risk_level: FailureRiskLevel

    estimated_failure_cost: float | None = Field(
        default=None,
        ge=0,
    )

    expected_failure_cost: float | None = Field(
        default=None,
        ge=0,
    )

    expected_downtime_hours: float | None = Field(
        default=None,
        ge=0,
    )

    personnel_safety_impact: bool = False
    environmental_impact: bool = False
    production_impact: bool = False

    assessment_method: str | None = None

    assumptions: list[str] = Field(
        default_factory=list
    )

    assessed_at: datetime = Field(
        default_factory=utc_now
    )

    requires_engineering_review: bool = True

    @model_validator(mode="after")
    def validate_expected_cost(
        self,
    ) -> "FailureRiskAssessment":
        if (
            self.expected_failure_cost is not None
            and self.estimated_failure_cost is None
        ):
            raise ValueError(
                "estimated_failure_cost is required "
                "when expected_failure_cost is supplied"
            )

        return self

    @classmethod
    def from_prediction(
        cls,
        prediction: FailurePrediction,
        *,
        consequence_severity: FailureSeverity,
        risk_level: FailureRiskLevel,
        estimated_failure_cost: float | None = None,
    ) -> "FailureRiskAssessment":
        expected_cost = None

        if estimated_failure_cost is not None:
            expected_cost = round(
                prediction.failure_probability
                * estimated_failure_cost,
                2,
            )

        return cls(
            context=prediction.context,
            prediction_id=prediction.prediction_id,
            failure_probability=(
                prediction.failure_probability
            ),
            consequence_severity=consequence_severity,
            risk_level=risk_level,
            estimated_failure_cost=(
                estimated_failure_cost
            ),
            expected_failure_cost=expected_cost,
            assessment_method=(
                "prediction_with_engineering_inputs"
            ),
        )


# ============================================================
# MAINTENANCE RECOMMENDATION
# ============================================================

class FailureMaintenanceRecommendation(
    FailureBaseModel
):
    recommendation_id: UUID = Field(
        default_factory=uuid4
    )

    prediction_id: UUID | None = None

    action: FailureActionType

    priority: FailureActionPriority

    description: str = Field(
        min_length=1,
        max_length=2000,
    )

    recommended_within_hours: float | None = Field(
        default=None,
        ge=0,
    )

    estimated_cost: float | None = Field(
        default=None,
        ge=0,
    )

    estimated_downtime_hours: float | None = Field(
        default=None,
        ge=0,
    )

    required_skills: list[str] = Field(
        default_factory=list
    )

    spare_parts: list[str] = Field(
        default_factory=list
    )

    reasoning: str | None = None

    requires_approval: bool = True

    approved_by: str | None = None

    created_at: datetime = Field(
        default_factory=utc_now
    )

    @model_validator(mode="after")
    def validate_safe_shutdown(
        self,
    ) -> "FailureMaintenanceRecommendation":
        if (
            self.action
            == FailureActionType.REQUEST_SAFE_SHUTDOWN.value
            and not self.requires_approval
        ):
            raise ValueError(
                "Safe-shutdown requests must "
                "require approved procedures"
            )

        return self


# ============================================================
# ACTUAL FAILURE EVENT
# ============================================================

class FailureEvent(FailureBaseModel):
    failure_id: UUID = Field(
        default_factory=uuid4
    )

    context: FailureContext

    failure_code: str = Field(
        min_length=1,
        max_length=100,
    )

    failure_type: FailureType

    failure_mode: FailureMode

    severity: FailureSeverity

    status: FailureStatus = (
        FailureStatus.DETECTED
    )

    title: str = Field(
        min_length=1,
        max_length=250,
    )

    description: str | None = None

    detected_at: datetime = Field(
        default_factory=utc_now
    )

    occurred_at: datetime | None = None

    confirmed_at: datetime | None = None
    resolved_at: datetime | None = None
    closed_at: datetime | None = None

    confirmed_by: str | None = None
    resolved_by: str | None = None

    root_cause: str | None = None

    related_prediction_id: UUID | None = None

    evidence: list[FailureEvidence] = Field(
        default_factory=list
    )

    maintenance_recommendations: list[
        FailureMaintenanceRecommendation
    ] = Field(
        default_factory=list
    )

    downtime_hours: float = Field(
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

    production_loss_cost: float = Field(
        default=0,
        ge=0,
    )

    currency: str = Field(
        default="INR",
        pattern=r"^[A-Z]{3}$",
    )

    corrective_actions: list[str] = Field(
        default_factory=list
    )

    preventive_actions: list[str] = Field(
        default_factory=list
    )

    requires_human_review: bool = True

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
        "detected_at",
        "occurred_at",
        "confirmed_at",
        "resolved_at",
        "closed_at",
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
                "Failure timestamps must "
                "be timezone-aware"
            )

        return value

    @model_validator(mode="after")
    def validate_lifecycle(
        self,
    ) -> "FailureEvent":
        if (
            self.closed_at is not None
            and self.resolved_at is not None
            and self.closed_at < self.resolved_at
        ):
            raise ValueError(
                "closed_at cannot precede resolved_at"
            )

        return self

    @property
    def total_recorded_cost(self) -> float:
        return round(
            self.repair_cost
            + self.replacement_cost
            + self.production_loss_cost,
            2,
        )

    def to_dict(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")

        data["total_recorded_cost"] = (
            self.total_recorded_cost
        )

        return data

    def add_evidence(
        self,
        evidence: FailureEvidence,
    ) -> None:
        if any(
            existing.evidence_id == evidence.evidence_id
            for existing in self.evidence
        ):
            raise ValueError(
                "Duplicate evidence ID"
            )

        self.evidence.append(evidence)
        self.updated_at = utc_now()

    def add_recommendation(
        self,
        recommendation: (
            FailureMaintenanceRecommendation
        ),
    ) -> None:
        if any(
            existing.recommendation_id
            == recommendation.recommendation_id
            for existing in self.maintenance_recommendations
        ):
            raise ValueError(
                "Duplicate recommendation ID"
            )

        self.maintenance_recommendations.append(
            recommendation
        )

        self.updated_at = utc_now()

    def confirm(
        self,
        *,
        confirmed_by: str,
        root_cause: str | None = None,
    ) -> None:
        if self.status not in {
            FailureStatus.SUSPECTED.value,
            FailureStatus.DETECTED.value,
            FailureStatus.UNDER_INVESTIGATION.value,
        }:
            raise ValueError(
                "Failure cannot be confirmed "
                "from its current status"
            )

        if not confirmed_by.strip():
            raise ValueError(
                "confirmed_by is required"
            )

        self.status = FailureStatus.CONFIRMED
        self.confirmed_by = confirmed_by
        self.confirmed_at = utc_now()

        if root_cause is not None:
            self.root_cause = root_cause

        self.updated_at = utc_now()

    def resolve(
        self,
        *,
        resolved_by: str,
    ) -> None:
        if self.status not in {
            FailureStatus.CONFIRMED.value,
            FailureStatus.UNDER_REPAIR.value,
        }:
            raise ValueError(
                "Only confirmed or under-repair "
                "failures can be resolved"
            )

        if not resolved_by.strip():
            raise ValueError(
                "resolved_by is required"
            )

        self.status = FailureStatus.RESOLVED
        self.resolved_by = resolved_by
        self.resolved_at = utc_now()
        self.updated_at = utc_now()

    def close(self) -> None:
        if (
            self.status
            != FailureStatus.RESOLVED.value
        ):
            raise ValueError(
                "Only resolved failures can be closed"
            )

        self.status = FailureStatus.CLOSED
        self.closed_at = utc_now()
        self.updated_at = utc_now()


# ============================================================
# FAILURE PREDICTION REQUEST
# ============================================================

class FailurePredictionRequest(FailureBaseModel):
    company_id: str = Field(
        min_length=1,
        max_length=100,
    )

    asset_id: str = Field(
        min_length=1,
        max_length=100,
    )

    equipment_id: str | None = None
    component_id: str | None = None

    prediction_horizon_hours: float = Field(
        default=168,
        gt=0,
    )

    model_id: str | None = None

    include_explanations: bool = True

    include_maintenance_recommendations: bool = True


# ============================================================
# FAILURE PREDICTION RESPONSE
# ============================================================

class FailurePredictionResponse(FailureBaseModel):
    prediction: FailurePrediction

    risk_assessment: (
        FailureRiskAssessment | None
    ) = None

    recommendations: list[
        FailureMaintenanceRecommendation
    ] = Field(
        default_factory=list
    )

    generated_at: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# CREATE / UPDATE SCHEMAS
# ============================================================

class FailureCreate(FailureBaseModel):
    context: FailureContext

    failure_code: str = Field(
        min_length=1,
        max_length=100,
    )

    failure_type: FailureType
    failure_mode: FailureMode

    severity: FailureSeverity

    title: str = Field(
        min_length=1,
        max_length=250,
    )

    description: str | None = None

    occurred_at: datetime | None = None

    related_prediction_id: UUID | None = None


class FailureUpdate(FailureBaseModel):
    title: str | None = None

    description: str | None = None

    severity: FailureSeverity | None = None

    root_cause: str | None = None

    corrective_actions: list[str] | None = None

    preventive_actions: list[str] | None = None


class FailureResponse(FailureEvent):
    pass


# ============================================================
# FAILURE SUMMARY
# ============================================================

class FailureSummary(FailureBaseModel):
    failure_id: UUID

    company_id: str
    asset_id: str

    equipment_id: str | None
    component_id: str | None

    failure_code: str

    failure_type: FailureType
    failure_mode: FailureMode

    severity: FailureSeverity
    status: FailureStatus

    title: str

    downtime_hours: float

    total_recorded_cost: float

    detected_at: datetime

    @classmethod
    def from_failure(
        cls,
        failure: FailureEvent,
    ) -> "FailureSummary":
        return cls(
            failure_id=failure.failure_id,
            company_id=failure.context.company_id,
            asset_id=failure.context.asset_id,
            equipment_id=(
                failure.context.equipment_id
            ),
            component_id=(
                failure.context.component_id
            ),
            failure_code=failure.failure_code,
            failure_type=failure.failure_type,
            failure_mode=failure.failure_mode,
            severity=failure.severity,
            status=failure.status,
            title=failure.title,
            downtime_hours=failure.downtime_hours,
            total_recorded_cost=(
                failure.total_recorded_cost
            ),
            detected_at=failure.detected_at,
        )


# ============================================================
# EXAMPLE
# ============================================================

def example() -> None:
    import json

    context = FailureContext(
        company_id="company-001",
        asset_id="ASSET-101",
        equipment_id="PUMP-101",
        component_id="BEARING-101",
        site_id="SITE-01",
        operating_hours=6500,
    )

    # Illustrative model output only.
    prediction = FailurePrediction(
        context=context,
        failure_type=FailureType.MECHANICAL,
        predicted_failure_mode=(
            FailureMode.BEARING_FAILURE
        ),
        failure_probability=0.82,
        prediction_horizon_hours=168,
        remaining_useful_life_hours=450,
        model_id="bearing-failure-model",
        model_version="1.0.0",
        source=PredictionSource.ML_MODEL,
        risk_level=FailureRiskLevel.HIGH,
        decision_threshold=0.75,
        explanation=(
            "Elevated vibration and temperature "
            "features contributed to the prediction."
        ),
    )

    assessment = FailureRiskAssessment.from_prediction(
        prediction,
        consequence_severity=FailureSeverity.HIGH,
        risk_level=FailureRiskLevel.HIGH,
        estimated_failure_cost=150000,
    )

    recommendation = FailureMaintenanceRecommendation(
        prediction_id=prediction.prediction_id,
        action=FailureActionType.INSPECT,
        priority=FailureActionPriority.HIGH,
        description=(
            "Arrange a qualified bearing inspection "
            "and review lubrication and alignment."
        ),
        recommended_within_hours=24,
        requires_approval=True,
    )

    response = FailurePredictionResponse(
        prediction=prediction,
        risk_assessment=assessment,
        recommendations=[recommendation],
    )

    print(
        json.dumps(
            response.model_dump(mode="json"),
            indent=2,
        )
    )


if __name__ == "__main__":
    example()