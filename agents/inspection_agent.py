# app/agents/inspection_agent.py

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence
from uuid import uuid4


# =========================================================
# Utilities
# =========================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid4())


def _finite(value: float, *, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def _probability(value: float, *, name: str) -> float:
    value = _finite(value, name=name)

    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{name} must be between 0 and 1."
        )

    return value


def _clamp(
    value: float,
    low: float = 0.0,
    high: float = 1.0,
) -> float:
    return max(low, min(high, float(value)))


# =========================================================
# Enums
# =========================================================

class InspectionType(str, Enum):
    ROUTINE = "routine"
    PREVENTIVE = "preventive"
    PREDICTIVE = "predictive"
    CONDITION_BASED = "condition_based"
    SAFETY = "safety"
    EMERGENCY = "emergency"
    POST_FAILURE = "post_failure"
    POST_REPAIR = "post_repair"
    VISUAL = "visual"
    THERMAL = "thermal"
    MULTIMODAL = "multimodal"


class InspectionStatus(str, Enum):
    PASSED = "passed"
    PASSED_WITH_OBSERVATIONS = (
        "passed_with_observations"
    )
    ATTENTION_REQUIRED = "attention_required"
    MAINTENANCE_REQUIRED = "maintenance_required"
    CRITICAL = "critical"
    INCONCLUSIVE = "inconclusive"


class InspectionSeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class FindingType(str, Enum):
    CRACK = "crack"
    CORROSION = "corrosion"
    WEAR = "wear"
    DENT = "dent"
    SCRATCH = "scratch"
    DEFORMATION = "deformation"

    LEAKAGE = "leakage"

    LOOSE_COMPONENT = "loose_component"
    MISSING_COMPONENT = "missing_component"

    SURFACE_DAMAGE = "surface_damage"
    STRUCTURAL_DAMAGE = "structural_damage"

    OVERHEATING = "overheating"
    BURN_DAMAGE = "burn_damage"

    ELECTRICAL_DAMAGE = "electrical_damage"

    CONTAMINATION = "contamination"

    ABNORMAL_TEMPERATURE = (
        "abnormal_temperature"
    )

    ABNORMAL_VIBRATION = (
        "abnormal_vibration"
    )

    ABNORMAL_PRESSURE = (
        "abnormal_pressure"
    )

    ABNORMAL_CURRENT = (
        "abnormal_current"
    )

    HIGH_ENERGY_CONSUMPTION = (
        "high_energy_consumption"
    )

    LOW_HEALTH_SCORE = "low_health_score"
    LOW_RUL = "low_rul"

    FAILURE_RISK = "failure_risk"

    SAFETY_HAZARD = "safety_hazard"

    OTHER = "other"


class FindingSource(str, Enum):
    COMPUTER_VISION = "computer_vision"
    THERMAL_CAMERA = "thermal_camera"

    TEMPERATURE_SENSOR = (
        "temperature_sensor"
    )

    VIBRATION_SENSOR = (
        "vibration_sensor"
    )

    PRESSURE_SENSOR = (
        "pressure_sensor"
    )

    CURRENT_SENSOR = "current_sensor"
    ENERGY_SENSOR = "energy_sensor"

    ANOMALY_MODEL = "anomaly_model"
    FAILURE_MODEL = "failure_model"
    RUL_MODEL = "rul_model"

    ASSET_HEALTH_AGENT = (
        "asset_health_agent"
    )

    FAILURE_ANALYSIS_AGENT = (
        "failure_analysis_agent"
    )

    TECHNICIAN = "technician"
    OPERATOR = "operator"

    HISTORICAL_INSPECTION = (
        "historical_inspection"
    )

    OTHER = "other"


class InspectionAction(str, Enum):
    CONTINUE_OPERATION = "continue_operation"

    MONITOR = "monitor"

    REINSPECT = "reinspect"

    MANUAL_INSPECTION = (
        "manual_inspection"
    )

    THERMAL_INSPECTION = (
        "thermal_inspection"
    )

    VIBRATION_ANALYSIS = (
        "vibration_analysis"
    )

    ELECTRICAL_INSPECTION = (
        "electrical_inspection"
    )

    LUBRICATION_CHECK = (
        "lubrication_check"
    )

    LEAK_CHECK = "leak_check"

    CLEAN = "clean"

    TIGHTEN = "tighten"

    REPAIR = "repair"

    REPLACE_COMPONENT = (
        "replace_component"
    )

    CREATE_WORK_ORDER = (
        "create_work_order"
    )

    FAILURE_ANALYSIS = (
        "failure_analysis"
    )

    SAFETY_REVIEW = "safety_review"

    REQUEST_MACHINE_STOP = (
        "request_machine_stop"
    )

    EMERGENCY_REVIEW = (
        "emergency_review"
    )


class InspectionPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    IMMEDIATE = "immediate"


# =========================================================
# Bounding box
# =========================================================

@dataclass(slots=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 < self.x1:
            raise ValueError(
                "x2 must be >= x1."
            )

        if self.y2 < self.y1:
            raise ValueError(
                "y2 must be >= y1."
            )

    @property
    def area(self) -> float:
        return max(
            0.0,
            self.x2 - self.x1,
        ) * max(
            0.0,
            self.y2 - self.y1,
        )


# =========================================================
# Visual detection
# =========================================================

@dataclass(slots=True)
class VisualDetection:
    detection_id: str

    finding_type: FindingType

    confidence: float

    severity_score: float

    bounding_box: BoundingBox | None = None

    image_id: str | None = None

    image_path: str | None = None

    component: str | None = None

    description: str | None = None

    model_name: str | None = None
    model_version: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.confidence = _probability(
            self.confidence,
            name="confidence",
        )

        self.severity_score = _probability(
            self.severity_score,
            name="severity_score",
        )


# =========================================================
# Sensor context
# =========================================================

@dataclass(slots=True)
class InspectionSensorContext:
    temperature: float | None = None
    temperature_warning: float | None = None
    temperature_critical: float | None = None

    vibration_rms: float | None = None
    vibration_warning: float | None = None
    vibration_critical: float | None = None

    pressure: float | None = None
    nominal_pressure: float | None = None

    current: float | None = None
    nominal_current: float | None = None

    power_kw: float | None = None

    energy_efficiency: float | None = None

    anomaly_score: float | None = None

    timestamp: datetime = field(
        default_factory=utc_now
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# =========================================================
# Asset context
# =========================================================

@dataclass(slots=True)
class InspectionAssetContext:
    asset_id: str

    asset_name: str | None = None
    asset_type: str | None = None

    site_id: str | None = None
    zone_id: str | None = None

    manufacturer: str | None = None
    model_number: str | None = None
    serial_number: str | None = None

    operating: bool = True

    safety_critical: bool = False

    criticality: float = 0.5

    operating_hours: float | None = None

    health_score: float | None = None

    failure_probability: float | None = None

    predicted_rul_hours: float | None = None

    replacement_cost: float | None = None
    estimated_repair_cost: float | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.criticality = _probability(
            self.criticality,
            name="criticality",
        )

        if self.failure_probability is not None:
            self.failure_probability = (
                _probability(
                    self.failure_probability,
                    name="failure_probability",
                )
            )

        if (
            self.health_score is not None
            and not 0.0
            <= self.health_score
            <= 100.0
        ):
            raise ValueError(
                "health_score must be between "
                "0 and 100."
            )


# =========================================================
# Human inspection observation
# =========================================================

@dataclass(slots=True)
class InspectionObservation:
    observation_id: str

    finding_type: FindingType

    description: str

    severity: InspectionSeverity

    component: str | None = None

    inspector_id: str | None = None

    image_id: str | None = None

    confirmed: bool = True

    timestamp: datetime = field(
        default_factory=utc_now
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# =========================================================
# Agent input
# =========================================================

@dataclass(slots=True)
class InspectionAgentInput:
    asset: InspectionAssetContext

    inspection_type: InspectionType = (
        InspectionType.MULTIMODAL
    )

    visual_detections: list[
        VisualDetection
    ] = field(default_factory=list)

    sensors: InspectionSensorContext = field(
        default_factory=InspectionSensorContext
    )

    observations: list[
        InspectionObservation
    ] = field(default_factory=list)

    previous_findings: list[
        Mapping[str, Any]
    ] = field(default_factory=list)

    inspection_id: str | None = None

    work_order_id: str | None = None

    incident_id: str | None = None

    prediction_id: str | None = None

    request_id: str | None = None
    trace_id: str | None = None

    data_drift_detected: bool = False
    model_drift_detected: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# =========================================================
# Finding
# =========================================================

@dataclass(slots=True)
class InspectionFinding:
    finding_id: str

    finding_type: FindingType

    source: FindingSource

    severity: InspectionSeverity

    severity_score: float

    confidence: float

    description: str

    component: str | None = None

    value: float | None = None
    unit: str | None = None

    image_id: str | None = None

    bounding_box: BoundingBox | None = None

    requires_action: bool = False

    critical: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# =========================================================
# Action
# =========================================================

@dataclass(slots=True)
class InspectionRecommendation:
    recommendation_id: str

    action: InspectionAction

    priority: InspectionPriority

    reason: str

    due_within_hours: float | None = None

    component: str | None = None

    requires_human_approval: bool = True

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# =========================================================
# Result
# =========================================================

@dataclass(slots=True)
class InspectionResult:
    inspection_id: str

    asset_id: str
    asset_name: str | None

    site_id: str | None
    zone_id: str | None

    inspection_type: InspectionType

    status: InspectionStatus

    overall_severity: InspectionSeverity

    inspection_score: float

    confidence: float

    findings: list[
        InspectionFinding
    ]

    recommendations: list[
        InspectionRecommendation
    ]

    total_findings: int

    critical_findings: int
    high_findings: int
    moderate_findings: int

    visual_findings: int
    sensor_findings: int

    asset_health_score: float | None

    failure_probability: float | None

    predicted_rul_hours: float | None

    maintenance_required: bool

    safety_review_required: bool

    shutdown_review_required: bool

    failure_analysis_required: bool

    human_review_required: bool

    model_reliability_warning: bool

    summary: str

    inspected_at: datetime

    created_at: datetime = field(
        default_factory=utc_now
    )

    work_order_id: str | None = None
    incident_id: str | None = None
    prediction_id: str | None = None

    trace_id: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        def bbox_dict(
            box: BoundingBox | None,
        ) -> dict[str, float] | None:
            return (
                asdict(box)
                if box is not None
                else None
            )

        return {
            "inspection_id":
                self.inspection_id,

            "asset_id":
                self.asset_id,

            "asset_name":
                self.asset_name,

            "site_id":
                self.site_id,

            "zone_id":
                self.zone_id,

            "inspection_type":
                self.inspection_type.value,

            "status":
                self.status.value,

            "overall_severity":
                self.overall_severity.value,

            "inspection_score":
                self.inspection_score,

            "confidence":
                self.confidence,

            "findings": [
                {
                    "finding_id":
                        finding.finding_id,

                    "finding_type":
                        finding
                        .finding_type.value,

                    "source":
                        finding.source.value,

                    "severity":
                        finding.severity.value,

                    "severity_score":
                        finding.severity_score,

                    "confidence":
                        finding.confidence,

                    "description":
                        finding.description,

                    "component":
                        finding.component,

                    "value":
                        finding.value,

                    "unit":
                        finding.unit,

                    "image_id":
                        finding.image_id,

                    "bounding_box":
                        bbox_dict(
                            finding.bounding_box
                        ),

                    "requires_action":
                        finding.requires_action,

                    "critical":
                        finding.critical,

                    "metadata":
                        dict(
                            finding.metadata
                        ),
                }
                for finding in self.findings
            ],

            "recommendations": [
                {
                    **asdict(item),
                    "action":
                        item.action.value,
                    "priority":
                        item.priority.value,
                }
                for item
                in self.recommendations
            ],

            "total_findings":
                self.total_findings,

            "critical_findings":
                self.critical_findings,

            "high_findings":
                self.high_findings,

            "moderate_findings":
                self.moderate_findings,

            "visual_findings":
                self.visual_findings,

            "sensor_findings":
                self.sensor_findings,

            "asset_health_score":
                self.asset_health_score,

            "failure_probability":
                self.failure_probability,

            "predicted_rul_hours":
                self.predicted_rul_hours,

            "maintenance_required":
                self.maintenance_required,

            "safety_review_required":
                self.safety_review_required,

            "shutdown_review_required":
                self.shutdown_review_required,

            "failure_analysis_required":
                self.failure_analysis_required,

            "human_review_required":
                self.human_review_required,

            "model_reliability_warning":
                self.model_reliability_warning,

            "summary":
                self.summary,

            "inspected_at":
                self.inspected_at.isoformat(),

            "created_at":
                self.created_at.isoformat(),

            "work_order_id":
                self.work_order_id,

            "incident_id":
                self.incident_id,

            "prediction_id":
                self.prediction_id,

            "trace_id":
                self.trace_id,

            "metadata":
                dict(self.metadata),
        }


# =========================================================
# Integration protocols
# =========================================================

class InspectionRepository(Protocol):

    async def save_inspection(
        self,
        inspection: InspectionResult,
    ) -> Any:
        ...


class WorkOrderService(Protocol):

    async def create_work_order(
        self,
        *,
        asset_id: str,
        title: str,
        description: str,
        priority: str,
        metadata: Mapping[
            str,
            Any,
        ] | None = None,
    ) -> Any:
        ...


class NotificationService(Protocol):

    async def notify(
        self,
        *,
        title: str,
        message: str,
        severity: str,
        metadata: Mapping[
            str,
            Any,
        ] | None = None,
    ) -> Any:
        ...


class FailureAnalysisService(Protocol):

    async def request_analysis(
        self,
        *,
        asset_id: str,
        inspection_id: str,
        findings: Sequence[
            Mapping[str, Any]
        ],
        metadata: Mapping[
            str,
            Any,
        ] | None = None,
    ) -> Any:
        ...


# =========================================================
# Config
# =========================================================

@dataclass(slots=True)
class InspectionAgentConfig:
    minimum_detection_confidence: float = 0.50

    moderate_threshold: float = 0.40
    high_threshold: float = 0.65
    critical_threshold: float = 0.85

    low_health_threshold: float = 50.0
    critical_health_threshold: float = 25.0

    failure_warning_threshold: float = 0.60
    failure_critical_threshold: float = 0.85

    warning_rul_hours: float = 168.0
    critical_rul_hours: float = 24.0

    anomaly_threshold: float = 0.70

    auto_create_work_order: bool = False

    auto_request_failure_analysis: bool = True

    enable_notifications: bool = True


# =========================================================
# Inspection Agent
# =========================================================

class InspectionAgent:

    def __init__(
        self,
        *,
        config: InspectionAgentConfig | None = None,
        repository: InspectionRepository | None = None,
        work_order_service: WorkOrderService | None = None,
        notification_service: NotificationService | None = None,
        failure_analysis_service: FailureAnalysisService | None = None,
    ) -> None:

        self.config = (
            config
            or InspectionAgentConfig()
        )

        self.repository = repository

        self.work_order_service = (
            work_order_service
        )

        self.notification_service = (
            notification_service
        )

        self.failure_analysis_service = (
            failure_analysis_service
        )

    # -----------------------------------------------------
    # Main execution
    # -----------------------------------------------------

    async def run(
        self,
        data: InspectionAgentInput,
    ) -> InspectionResult:

        findings: list[
            InspectionFinding
        ] = []

        findings.extend(
            self._visual_findings(data)
        )

        findings.extend(
            self._sensor_findings(data)
        )

        findings.extend(
            self._model_findings(data)
        )

        findings.extend(
            self._human_findings(data)
        )

        findings = (
            self._deduplicate_findings(
                findings
            )
        )

        overall_severity = (
            self._overall_severity(
                findings
            )
        )

        inspection_score = (
            self._inspection_score(
                data,
                findings,
            )
        )

        confidence = (
            self._confidence(
                data,
                findings,
            )
        )

        reliability_warning = (
            data.data_drift_detected
            or data.model_drift_detected
        )

        status = self._status(
            findings=findings,
            severity=overall_severity,
            reliability_warning=(
                reliability_warning
            ),
        )

        maintenance_required = any(
            finding.requires_action
            for finding in findings
        ) or overall_severity in {
            InspectionSeverity.HIGH,
            InspectionSeverity.CRITICAL,
        }

        safety_review = (
            self._safety_review_required(
                data,
                findings,
            )
        )

        shutdown_review = (
            self._shutdown_review_required(
                data,
                findings,
                overall_severity,
            )
        )

        failure_analysis = (
            self._failure_analysis_required(
                data,
                findings,
            )
        )

        human_review = (
            reliability_warning
            or overall_severity
            in {
                InspectionSeverity.HIGH,
                InspectionSeverity.CRITICAL,
            }
            or confidence < 0.60
        )

        recommendations = (
            self._recommendations(
                data=data,
                findings=findings,
                severity=overall_severity,
                maintenance_required=(
                    maintenance_required
                ),
                safety_review=(
                    safety_review
                ),
                shutdown_review=(
                    shutdown_review
                ),
                failure_analysis=(
                    failure_analysis
                ),
                reliability_warning=(
                    reliability_warning
                ),
            )
        )

        critical_count = sum(
            finding.severity
            == InspectionSeverity.CRITICAL
            for finding in findings
        )

        high_count = sum(
            finding.severity
            == InspectionSeverity.HIGH
            for finding in findings
        )

        moderate_count = sum(
            finding.severity
            == InspectionSeverity.MODERATE
            for finding in findings
        )

        visual_count = sum(
            finding.source
            in {
                FindingSource.COMPUTER_VISION,
                FindingSource.THERMAL_CAMERA,
            }
            for finding in findings
        )

        sensor_count = sum(
            finding.source
            in {
                FindingSource
                .TEMPERATURE_SENSOR,
                FindingSource
                .VIBRATION_SENSOR,
                FindingSource
                .PRESSURE_SENSOR,
                FindingSource
                .CURRENT_SENSOR,
                FindingSource
                .ENERGY_SENSOR,
            }
            for finding in findings
        )

        inspection_id = (
            data.inspection_id
            or new_id()
        )

        summary = self._summary(
            data=data,
            status=status,
            severity=overall_severity,
            findings=findings,
            inspection_score=(
                inspection_score
            ),
        )

        result = InspectionResult(
            inspection_id=inspection_id,

            asset_id=(
                data.asset.asset_id
            ),

            asset_name=(
                data.asset.asset_name
            ),

            site_id=(
                data.asset.site_id
            ),

            zone_id=(
                data.asset.zone_id
            ),

            inspection_type=(
                data.inspection_type
            ),

            status=status,

            overall_severity=(
                overall_severity
            ),

            inspection_score=round(
                inspection_score,
                2,
            ),

            confidence=round(
                confidence,
                4,
            ),

            findings=findings,

            recommendations=(
                recommendations
            ),

            total_findings=(
                len(findings)
            ),

            critical_findings=(
                critical_count
            ),

            high_findings=(
                high_count
            ),

            moderate_findings=(
                moderate_count
            ),

            visual_findings=(
                visual_count
            ),

            sensor_findings=(
                sensor_count
            ),

            asset_health_score=(
                data.asset.health_score
            ),

            failure_probability=(
                data.asset
                .failure_probability
            ),

            predicted_rul_hours=(
                data.asset
                .predicted_rul_hours
            ),

            maintenance_required=(
                maintenance_required
            ),

            safety_review_required=(
                safety_review
            ),

            shutdown_review_required=(
                shutdown_review
            ),

            failure_analysis_required=(
                failure_analysis
            ),

            human_review_required=(
                human_review
            ),

            model_reliability_warning=(
                reliability_warning
            ),

            summary=summary,

            inspected_at=utc_now(),

            work_order_id=(
                data.work_order_id
            ),

            incident_id=(
                data.incident_id
            ),

            prediction_id=(
                data.prediction_id
            ),

            trace_id=data.trace_id,

            metadata={
                **data.metadata,

                "request_id":
                    data.request_id,
            },
        )

        await self._post_process(
            result
        )

        return result

    # -----------------------------------------------------
    # Visual findings
    # -----------------------------------------------------

    def _visual_findings(
        self,
        data: InspectionAgentInput,
    ) -> list[InspectionFinding]:

        results: list[
            InspectionFinding
        ] = []

        for detection in (
            data.visual_detections
        ):

            if (
                detection.confidence
                < self.config
                .minimum_detection_confidence
            ):
                continue

            severity = (
                self._severity_from_score(
                    detection
                    .severity_score
                )
            )

            results.append(
                InspectionFinding(
                    finding_id=(
                        detection
                        .detection_id
                        or new_id()
                    ),

                    finding_type=(
                        detection
                        .finding_type
                    ),

                    source=(
                        FindingSource
                        .COMPUTER_VISION
                    ),

                    severity=severity,

                    severity_score=(
                        detection
                        .severity_score
                    ),

                    confidence=(
                        detection.confidence
                    ),

                    description=(
                        detection.description
                        or (
                            f"Visual inspection "
                            f"detected "
                            f"{detection.finding_type.value}."
                        )
                    ),

                    component=(
                        detection.component
                    ),

                    image_id=(
                        detection.image_id
                    ),

                    bounding_box=(
                        detection
                        .bounding_box
                    ),

                    requires_action=(
                        severity
                        in {
                            InspectionSeverity
                            .MODERATE,

                            InspectionSeverity
                            .HIGH,

                            InspectionSeverity
                            .CRITICAL,
                        }
                    ),

                    critical=(
                        severity
                        == InspectionSeverity
                        .CRITICAL
                    ),

                    metadata={
                        **detection.metadata,

                        "model_name":
                            detection
                            .model_name,

                        "model_version":
                            detection
                            .model_version,
                    },
                )
            )

        return results

    # -----------------------------------------------------
    # Sensor findings
    # -----------------------------------------------------

    def _sensor_findings(
        self,
        data: InspectionAgentInput,
    ) -> list[InspectionFinding]:

        sensor = data.sensors

        results: list[
            InspectionFinding
        ] = []

        # Temperature
        if sensor.temperature is not None:

            if (
                sensor.temperature_critical
                is not None
                and sensor.temperature
                >= sensor.temperature_critical
            ):
                results.append(
                    self._sensor_finding(
                        FindingType
                        .ABNORMAL_TEMPERATURE,

                        FindingSource
                        .TEMPERATURE_SENSOR,

                        InspectionSeverity
                        .CRITICAL,

                        1.0,

                        0.98,

                        (
                            "Temperature exceeds "
                            "critical threshold."
                        ),

                        sensor.temperature,

                        "°C",
                    )
                )

            elif (
                sensor.temperature_warning
                is not None
                and sensor.temperature
                >= sensor.temperature_warning
            ):
                results.append(
                    self._sensor_finding(
                        FindingType
                        .ABNORMAL_TEMPERATURE,

                        FindingSource
                        .TEMPERATURE_SENSOR,

                        InspectionSeverity.HIGH,

                        0.75,

                        0.95,

                        (
                            "Temperature exceeds "
                            "warning threshold."
                        ),

                        sensor.temperature,

                        "°C",
                    )
                )

        # Vibration
        if sensor.vibration_rms is not None:

            if (
                sensor.vibration_critical
                is not None
                and sensor.vibration_rms
                >= sensor.vibration_critical
            ):
                results.append(
                    self._sensor_finding(
                        FindingType
                        .ABNORMAL_VIBRATION,

                        FindingSource
                        .VIBRATION_SENSOR,

                        InspectionSeverity
                        .CRITICAL,

                        1.0,

                        0.98,

                        (
                            "Vibration exceeds "
                            "critical threshold."
                        ),

                        sensor.vibration_rms,

                        "mm/s",
                    )
                )

            elif (
                sensor.vibration_warning
                is not None
                and sensor.vibration_rms
                >= sensor.vibration_warning
            ):
                results.append(
                    self._sensor_finding(
                        FindingType
                        .ABNORMAL_VIBRATION,

                        FindingSource
                        .VIBRATION_SENSOR,

                        InspectionSeverity.HIGH,

                        0.75,

                        0.95,

                        (
                            "Vibration exceeds "
                            "warning threshold."
                        ),

                        sensor.vibration_rms,

                        "mm/s",
                    )
                )

        # Pressure
        if (
            sensor.pressure is not None
            and sensor.nominal_pressure
            and sensor.nominal_pressure > 0
        ):
            deviation = (
                abs(
                    sensor.pressure
                    - sensor.nominal_pressure
                )
                / sensor.nominal_pressure
            )

            if deviation >= 0.30:
                results.append(
                    self._sensor_finding(
                        FindingType
                        .ABNORMAL_PRESSURE,

                        FindingSource
                        .PRESSURE_SENSOR,

                        InspectionSeverity.HIGH,

                        _clamp(
                            deviation / 0.50
                        ),

                        0.90,

                        (
                            "Significant pressure "
                            "deviation detected."
                        ),

                        sensor.pressure,

                        None,
                    )
                )

            elif deviation >= 0.15:
                results.append(
                    self._sensor_finding(
                        FindingType
                        .ABNORMAL_PRESSURE,

                        FindingSource
                        .PRESSURE_SENSOR,

                        InspectionSeverity
                        .MODERATE,

                        _clamp(
                            deviation / 0.50
                        ),

                        0.85,

                        (
                            "Pressure deviation "
                            "detected."
                        ),

                        sensor.pressure,

                        None,
                    )
                )

        # Current
        if (
            sensor.current is not None
            and sensor.nominal_current
            and sensor.nominal_current > 0
        ):
            ratio = (
                sensor.current
                / sensor.nominal_current
            )

            if ratio >= 1.30:
                results.append(
                    self._sensor_finding(
                        FindingType
                        .ABNORMAL_CURRENT,

                        FindingSource
                        .CURRENT_SENSOR,

                        InspectionSeverity.HIGH,

                        _clamp(
                            (ratio - 1.0)
                            / 0.50
                        ),

                        0.92,

                        (
                            "Current is "
                            "significantly above "
                            "nominal level."
                        ),

                        sensor.current,

                        "A",
                    )
                )

            elif ratio >= 1.15:
                results.append(
                    self._sensor_finding(
                        FindingType
                        .ABNORMAL_CURRENT,

                        FindingSource
                        .CURRENT_SENSOR,

                        InspectionSeverity
                        .MODERATE,

                        _clamp(
                            (ratio - 1.0)
                            / 0.50
                        ),

                        0.88,

                        (
                            "Current exceeds "
                            "nominal level."
                        ),

                        sensor.current,

                        "A",
                    )
                )

        # Energy efficiency
        if (
            sensor.energy_efficiency
            is not None
            and sensor.energy_efficiency
            < 0.70
        ):
            results.append(
                self._sensor_finding(
                    FindingType
                    .HIGH_ENERGY_CONSUMPTION,

                    FindingSource
                    .ENERGY_SENSOR,

                    InspectionSeverity.MODERATE,

                    _clamp(
                        1.0
                        - sensor
                        .energy_efficiency
                    ),

                    0.80,

                    (
                        "Low operating energy "
                        "efficiency detected."
                    ),

                    sensor.energy_efficiency,

                    "ratio",
                )
            )

        return results

    @staticmethod
    def _sensor_finding(
        finding_type: FindingType,
        source: FindingSource,
        severity: InspectionSeverity,
        severity_score: float,
        confidence: float,
        description: str,
        value: float,
        unit: str | None,
    ) -> InspectionFinding:

        return InspectionFinding(
            finding_id=new_id(),

            finding_type=finding_type,

            source=source,

            severity=severity,

            severity_score=(
                _clamp(severity_score)
            ),

            confidence=(
                _clamp(confidence)
            ),

            description=description,

            value=value,

            unit=unit,

            requires_action=(
                severity
                in {
                    InspectionSeverity
                    .MODERATE,

                    InspectionSeverity.HIGH,

                    InspectionSeverity
                    .CRITICAL,
                }
            ),

            critical=(
                severity
                == InspectionSeverity.CRITICAL
            ),
        )

    # -----------------------------------------------------
    # ML findings
    # -----------------------------------------------------

    def _model_findings(
        self,
        data: InspectionAgentInput,
    ) -> list[InspectionFinding]:

        results: list[
            InspectionFinding
        ] = []

        asset = data.asset
        sensor = data.sensors

        if (
            sensor.anomaly_score is not None
            and sensor.anomaly_score
            >= self.config
            .anomaly_threshold
        ):
            results.append(
                InspectionFinding(
                    finding_id=new_id(),

                    finding_type=(
                        FindingType
                        .FAILURE_RISK
                    ),

                    source=(
                        FindingSource
                        .ANOMALY_MODEL
                    ),

                    severity=(
                        self._severity_from_score(
                            sensor
                            .anomaly_score
                        )
                    ),

                    severity_score=(
                        sensor.anomaly_score
                    ),

                    confidence=0.85,

                    description=(
                        "Anomaly model detected "
                        "abnormal asset behaviour."
                    ),

                    requires_action=True,

                    critical=(
                        sensor.anomaly_score
                        >= 0.90
                    ),
                )
            )

        if (
            asset.failure_probability
            is not None
            and asset.failure_probability
            >= self.config
            .failure_warning_threshold
        ):

            severity = (
                InspectionSeverity.CRITICAL
                if (
                    asset
                    .failure_probability
                    >= self.config
                    .failure_critical_threshold
                )
                else InspectionSeverity.HIGH
            )

            results.append(
                InspectionFinding(
                    finding_id=new_id(),

                    finding_type=(
                        FindingType
                        .FAILURE_RISK
                    ),

                    source=(
                        FindingSource
                        .FAILURE_MODEL
                    ),

                    severity=severity,

                    severity_score=(
                        asset
                        .failure_probability
                    ),

                    confidence=0.85,

                    description=(
                        "Predictive model indicates "
                        "elevated equipment failure "
                        "risk."
                    ),

                    value=(
                        asset
                        .failure_probability
                    ),

                    unit="probability",

                    requires_action=True,

                    critical=(
                        severity
                        == InspectionSeverity
                        .CRITICAL
                    ),
                )
            )

        if (
            asset.predicted_rul_hours
            is not None
            and asset.predicted_rul_hours
            <= self.config
            .warning_rul_hours
        ):

            severity = (
                InspectionSeverity.CRITICAL
                if (
                    asset
                    .predicted_rul_hours
                    <= self.config
                    .critical_rul_hours
                )
                else InspectionSeverity.HIGH
            )

            results.append(
                InspectionFinding(
                    finding_id=new_id(),

                    finding_type=(
                        FindingType.LOW_RUL
                    ),

                    source=(
                        FindingSource.RUL_MODEL
                    ),

                    severity=severity,

                    severity_score=(
                        1.0
                        if severity
                        == InspectionSeverity
                        .CRITICAL
                        else 0.75
                    ),

                    confidence=0.85,

                    description=(
                        "Remaining useful life is "
                        "below the configured "
                        "inspection threshold."
                    ),

                    value=(
                        asset
                        .predicted_rul_hours
                    ),

                    unit="hours",

                    requires_action=True,

                    critical=(
                        severity
                        == InspectionSeverity
                        .CRITICAL
                    ),
                )
            )

        if (
            asset.health_score is not None
            and asset.health_score
            <= self.config
            .low_health_threshold
        ):

            severity = (
                InspectionSeverity.CRITICAL
                if (
                    asset.health_score
                    <= self.config
                    .critical_health_threshold
                )
                else InspectionSeverity.HIGH
            )

            results.append(
                InspectionFinding(
                    finding_id=new_id(),

                    finding_type=(
                        FindingType
                        .LOW_HEALTH_SCORE
                    ),

                    source=(
                        FindingSource
                        .ASSET_HEALTH_AGENT
                    ),

                    severity=severity,

                    severity_score=(
                        _clamp(
                            1.0
                            - (
                                asset
                                .health_score
                                / 100.0
                            )
                        )
                    ),

                    confidence=0.90,

                    description=(
                        "Asset health score is "
                        "below the acceptable "
                        "inspection threshold."
                    ),

                    value=(
                        asset.health_score
                    ),

                    unit="score",

                    requires_action=True,

                    critical=(
                        severity
                        == InspectionSeverity
                        .CRITICAL
                    ),
                )
            )

        return results

    # -----------------------------------------------------
    # Human findings
    # -----------------------------------------------------

    @staticmethod
    def _human_findings(
        data: InspectionAgentInput,
    ) -> list[InspectionFinding]:

        results: list[
            InspectionFinding
        ] = []

        for observation in (
            data.observations
        ):

            if not observation.confirmed:
                continue

            score_map = {
                InspectionSeverity.NONE: 0.0,
                InspectionSeverity.LOW: 0.25,
                InspectionSeverity.MODERATE: 0.50,
                InspectionSeverity.HIGH: 0.75,
                InspectionSeverity.CRITICAL: 1.0,
            }

            results.append(
                InspectionFinding(
                    finding_id=(
                        observation
                        .observation_id
                    ),

                    finding_type=(
                        observation
                        .finding_type
                    ),

                    source=(
                        FindingSource
                        .TECHNICIAN
                    ),

                    severity=(
                        observation.severity
                    ),

                    severity_score=(
                        score_map[
                            observation
                            .severity
                        ]
                    ),

                    confidence=1.0,

                    description=(
                        observation.description
                    ),

                    component=(
                        observation.component
                    ),

                    image_id=(
                        observation.image_id
                    ),

                    requires_action=(
                        observation.severity
                        in {
                            InspectionSeverity
                            .MODERATE,

                            InspectionSeverity.HIGH,

                            InspectionSeverity
                            .CRITICAL,
                        }
                    ),

                    critical=(
                        observation.severity
                        == InspectionSeverity
                        .CRITICAL
                    ),

                    metadata={
                        **observation.metadata,

                        "inspector_id":
                            observation
                            .inspector_id,
                    },
                )
            )

        return results

    # -----------------------------------------------------
    # Dedup
    # -----------------------------------------------------

    @staticmethod
    def _deduplicate_findings(
        findings: Sequence[
            InspectionFinding
        ],
    ) -> list[InspectionFinding]:

        # Different sources are deliberately
        # preserved because independent evidence
        # improves inspection confidence.
        seen: set[
            tuple[
                FindingType,
                FindingSource,
                str | None,
            ]
        ] = set()

        result: list[
            InspectionFinding
        ] = []

        ordered = sorted(
            findings,
            key=lambda item: (
                item.severity_score,
                item.confidence,
            ),
            reverse=True,
        )

        for finding in ordered:

            key = (
                finding.finding_type,
                finding.source,
                finding.component,
            )

            if key in seen:
                continue

            seen.add(key)
            result.append(finding)

        return result

    # -----------------------------------------------------
    # Severity
    # -----------------------------------------------------

    def _severity_from_score(
        self,
        score: float,
    ) -> InspectionSeverity:

        score = _clamp(score)

        if (
            score
            >= self.config
            .critical_threshold
        ):
            return (
                InspectionSeverity.CRITICAL
            )

        if (
            score
            >= self.config.high_threshold
        ):
            return InspectionSeverity.HIGH

        if (
            score
            >= self.config
            .moderate_threshold
        ):
            return (
                InspectionSeverity.MODERATE
            )

        if score > 0:
            return InspectionSeverity.LOW

        return InspectionSeverity.NONE

    @staticmethod
    def _overall_severity(
        findings: Sequence[
            InspectionFinding
        ],
    ) -> InspectionSeverity:

        rank = {
            InspectionSeverity.NONE: 0,
            InspectionSeverity.LOW: 1,
            InspectionSeverity.MODERATE: 2,
            InspectionSeverity.HIGH: 3,
            InspectionSeverity.CRITICAL: 4,
        }

        if not findings:
            return InspectionSeverity.NONE

        return max(
            (
                finding.severity
                for finding in findings
            ),
            key=lambda item: rank[item],
        )

    # -----------------------------------------------------
    # Inspection score
    # -----------------------------------------------------

    @staticmethod
    def _inspection_score(
        data: InspectionAgentInput,
        findings: Sequence[
            InspectionFinding
        ],
    ) -> float:

        if not findings:
            base = 100.0
        else:
            risk_values = [
                (
                    finding.severity_score
                    * finding.confidence
                )
                for finding in findings
            ]

            max_risk = max(
                risk_values,
                default=0.0,
            )

            avg_risk = (
                sum(risk_values)
                / len(risk_values)
            )

            risk = (
                0.70 * max_risk
                + 0.30 * avg_risk
            )

            base = (
                100.0
                * (1.0 - risk)
            )

        if (
            data.asset.health_score
            is not None
        ):
            base = (
                0.70 * base
                + 0.30
                * data.asset.health_score
            )

        return max(
            0.0,
            min(
                100.0,
                base,
            ),
        )

    # -----------------------------------------------------
    # Confidence
    # -----------------------------------------------------

    @staticmethod
    def _confidence(
        data: InspectionAgentInput,
        findings: Sequence[
            InspectionFinding
        ],
    ) -> float:

        if not findings:
            # No findings does not imply perfect
            # certainty; it means no abnormal
            # evidence was observed.
            confidence = 0.80
        else:
            confidence = (
                sum(
                    item.confidence
                    for item in findings
                )
                / len(findings)
            )

            independent_sources = {
                item.source
                for item in findings
            }

            if len(independent_sources) >= 3:
                confidence += 0.05

        if (
            data.data_drift_detected
            or data.model_drift_detected
        ):
            confidence *= 0.80

        return _clamp(confidence)

    # -----------------------------------------------------
    # Status
    # -----------------------------------------------------

    @staticmethod
    def _status(
        *,
        findings: Sequence[
            InspectionFinding
        ],
        severity: InspectionSeverity,
        reliability_warning: bool,
    ) -> InspectionStatus:

        if reliability_warning and not findings:
            return (
                InspectionStatus.INCONCLUSIVE
            )

        if severity == InspectionSeverity.CRITICAL:
            return InspectionStatus.CRITICAL

        if severity == InspectionSeverity.HIGH:
            return (
                InspectionStatus
                .MAINTENANCE_REQUIRED
            )

        if (
            severity
            == InspectionSeverity.MODERATE
        ):
            return (
                InspectionStatus
                .ATTENTION_REQUIRED
            )

        if severity == InspectionSeverity.LOW:
            return (
                InspectionStatus
                .PASSED_WITH_OBSERVATIONS
            )

        return InspectionStatus.PASSED

    # -----------------------------------------------------
    # Decision helpers
    # -----------------------------------------------------

    @staticmethod
    def _safety_review_required(
        data: InspectionAgentInput,
        findings: Sequence[
            InspectionFinding
        ],
    ) -> bool:

        safety_types = {
            FindingType.STRUCTURAL_DAMAGE,
            FindingType.BURN_DAMAGE,
            FindingType.ELECTRICAL_DAMAGE,
            FindingType.OVERHEATING,
            FindingType.SAFETY_HAZARD,
        }

        if (
            data.asset.safety_critical
            and any(
                finding.severity
                in {
                    InspectionSeverity.HIGH,
                    InspectionSeverity.CRITICAL,
                }
                for finding in findings
            )
        ):
            return True

        return any(
            finding.finding_type
            in safety_types
            and finding.severity
            in {
                InspectionSeverity.HIGH,
                InspectionSeverity.CRITICAL,
            }
            for finding in findings
        )

    @staticmethod
    def _shutdown_review_required(
        data: InspectionAgentInput,
        findings: Sequence[
            InspectionFinding
        ],
        severity: InspectionSeverity,
    ) -> bool:

        if not data.asset.operating:
            return False

        if severity == InspectionSeverity.CRITICAL:
            return True

        critical_types = {
            FindingType.STRUCTURAL_DAMAGE,
            FindingType.BURN_DAMAGE,
            FindingType.ELECTRICAL_DAMAGE,
            FindingType.OVERHEATING,
        }

        return any(
            finding.finding_type
            in critical_types
            and finding.severity
            == InspectionSeverity.HIGH
            for finding in findings
        )

    def _failure_analysis_required(
        self,
        data: InspectionAgentInput,
        findings: Sequence[
            InspectionFinding
        ],
    ) -> bool:

        failure_related = {
            FindingType
            .ABNORMAL_TEMPERATURE,

            FindingType
            .ABNORMAL_VIBRATION,

            FindingType
            .ABNORMAL_PRESSURE,

            FindingType
            .ABNORMAL_CURRENT,

            FindingType
            .LOW_HEALTH_SCORE,

            FindingType.LOW_RUL,

            FindingType.FAILURE_RISK,

            FindingType.WEAR,

            FindingType.CRACK,

            FindingType.CORROSION,

            FindingType.LEAKAGE,
        }

        if (
            data.asset.failure_probability
            is not None
            and data.asset.failure_probability
            >= self.config
            .failure_warning_threshold
        ):
            return True

        return any(
            finding.finding_type
            in failure_related
            and finding.severity
            in {
                InspectionSeverity.HIGH,
                InspectionSeverity.CRITICAL,
            }
            for finding in findings
        )

    # -----------------------------------------------------
    # Recommendations
    # -----------------------------------------------------

    def _recommendations(
        self,
        *,
        data: InspectionAgentInput,
        findings: Sequence[
            InspectionFinding
        ],
        severity: InspectionSeverity,
        maintenance_required: bool,
        safety_review: bool,
        shutdown_review: bool,
        failure_analysis: bool,
        reliability_warning: bool,
    ) -> list[
        InspectionRecommendation
    ]:

        recommendations: list[
            InspectionRecommendation
        ] = []

        if not findings:
            recommendations.append(
                self._recommendation(
                    InspectionAction
                    .CONTINUE_OPERATION,

                    InspectionPriority.LOW,

                    (
                        "No abnormal inspection "
                        "findings were identified."
                    ),

                    approval=False,
                )
            )

        if reliability_warning:
            recommendations.append(
                self._recommendation(
                    InspectionAction
                    .MANUAL_INSPECTION,

                    InspectionPriority.HIGH,

                    (
                        "Data/model drift reduces "
                        "AI inspection reliability; "
                        "perform independent manual "
                        "verification."
                    ),
                )
            )

        finding_types = {
            item.finding_type
            for item in findings
        }

        if FindingType.CRACK in finding_types:
            recommendations.append(
                self._recommendation(
                    InspectionAction.REPAIR,

                    InspectionPriority.HIGH,

                    (
                        "Inspect crack dimensions "
                        "and structural impact "
                        "before repair approval."
                    ),
                )
            )

        if (
            FindingType.CORROSION
            in finding_types
        ):
            recommendations.append(
                self._recommendation(
                    InspectionAction.REPAIR,

                    InspectionPriority.NORMAL,

                    (
                        "Assess corrosion depth, "
                        "affected area and component "
                        "integrity."
                    ),
                )
            )

        if (
            FindingType.LEAKAGE
            in finding_types
            or FindingType
            .ABNORMAL_PRESSURE
            in finding_types
        ):
            recommendations.append(
                self._recommendation(
                    InspectionAction.LEAK_CHECK,

                    InspectionPriority.HIGH,

                    (
                        "Inspect seals, hoses, "
                        "valves and connections for "
                        "leakage."
                    ),
                )
            )

        if (
            FindingType
            .ABNORMAL_VIBRATION
            in finding_types
        ):
            recommendations.append(
                self._recommendation(
                    InspectionAction
                    .VIBRATION_ANALYSIS,

                    InspectionPriority.HIGH,

                    (
                        "Perform detailed vibration "
                        "analysis for bearing, "
                        "alignment, imbalance and "
                        "looseness faults."
                    ),
                )
            )

            recommendations.append(
                self._recommendation(
                    InspectionAction
                    .LUBRICATION_CHECK,

                    InspectionPriority.NORMAL,

                    (
                        "Verify bearing lubrication "
                        "condition and lubrication "
                        "schedule."
                    ),
                )
            )

        if (
            FindingType
            .ABNORMAL_CURRENT
            in finding_types
            or FindingType
            .ELECTRICAL_DAMAGE
            in finding_types
        ):
            recommendations.append(
                self._recommendation(
                    InspectionAction
                    .ELECTRICAL_INSPECTION,

                    InspectionPriority.HIGH,

                    (
                        "Inspect electrical loading, "
                        "connections, insulation and "
                        "motor condition."
                    ),
                )
            )

        if (
            FindingType.OVERHEATING
            in finding_types
            or FindingType
            .ABNORMAL_TEMPERATURE
            in finding_types
        ):
            recommendations.append(
                self._recommendation(
                    InspectionAction
                    .THERMAL_INSPECTION,

                    InspectionPriority.HIGH,

                    (
                        "Perform thermal inspection "
                        "and verify cooling, "
                        "lubrication and electrical "
                        "loading."
                    ),
                )
            )

        if failure_analysis:
            recommendations.append(
                self._recommendation(
                    InspectionAction
                    .FAILURE_ANALYSIS,

                    InspectionPriority.URGENT,

                    (
                        "Forward inspection evidence "
                        "to the failure analysis "
                        "agent for root-cause "
                        "analysis."
                    ),
                )
            )

        if maintenance_required:
            recommendations.append(
                self._recommendation(
                    InspectionAction
                    .CREATE_WORK_ORDER,

                    (
                        InspectionPriority.URGENT
                        if severity
                        in {
                            InspectionSeverity.HIGH,
                            InspectionSeverity
                            .CRITICAL,
                        }
                        else
                        InspectionPriority.NORMAL
                    ),

                    (
                        "Create a work order linked "
                        "to inspection findings."
                    ),
                )
            )

        if safety_review:
            recommendations.append(
                self._recommendation(
                    InspectionAction
                    .SAFETY_REVIEW,

                    InspectionPriority.URGENT,

                    (
                        "Qualified safety personnel "
                        "should review the inspection "
                        "before continued operation."
                    ),
                )
            )

        if shutdown_review:
            recommendations.append(
                self._recommendation(
                    InspectionAction
                    .REQUEST_MACHINE_STOP,

                    InspectionPriority.IMMEDIATE,

                    (
                        "Request authorized review "
                        "for controlled or "
                        "safety-rated machine stop."
                    ),

                    hours=0.0,
                )
            )

        if (
            severity
            == InspectionSeverity.CRITICAL
        ):
            recommendations.append(
                self._recommendation(
                    InspectionAction
                    .EMERGENCY_REVIEW,

                    InspectionPriority.IMMEDIATE,

                    (
                        "Critical inspection "
                        "findings require immediate "
                        "qualified human review."
                    ),

                    hours=0.0,
                )
            )

        return self._deduplicate_recommendations(
            recommendations
        )

    @staticmethod
    def _recommendation(
        action: InspectionAction,
        priority: InspectionPriority,
        reason: str,
        *,
        hours: float | None = None,
        component: str | None = None,
        approval: bool = True,
    ) -> InspectionRecommendation:

        default_hours = {
            InspectionPriority.LOW: 168.0,
            InspectionPriority.NORMAL: 72.0,
            InspectionPriority.HIGH: 24.0,
            InspectionPriority.URGENT: 4.0,
            InspectionPriority.IMMEDIATE: 0.0,
        }

        return InspectionRecommendation(
            recommendation_id=new_id(),

            action=action,

            priority=priority,

            reason=reason,

            due_within_hours=(
                hours
                if hours is not None
                else default_hours[
                    priority
                ]
            ),

            component=component,

            requires_human_approval=(
                approval
            ),
        )

    @staticmethod
    def _deduplicate_recommendations(
        recommendations: Sequence[
            InspectionRecommendation
        ],
    ) -> list[
        InspectionRecommendation
    ]:

        rank = {
            InspectionPriority.LOW: 1,
            InspectionPriority.NORMAL: 2,
            InspectionPriority.HIGH: 3,
            InspectionPriority.URGENT: 4,
            InspectionPriority.IMMEDIATE: 5,
        }

        unique: dict[
            InspectionAction,
            InspectionRecommendation,
        ] = {}

        for item in recommendations:

            existing = unique.get(
                item.action
            )

            if (
                existing is None
                or rank[item.priority]
                > rank[existing.priority]
            ):
                unique[item.action] = item

        return list(unique.values())

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    @staticmethod
    def _summary(
        *,
        data: InspectionAgentInput,
        status: InspectionStatus,
        severity: InspectionSeverity,
        findings: Sequence[
            InspectionFinding
        ],
        inspection_score: float,
    ) -> str:

        asset = (
            data.asset.asset_name
            or data.asset.asset_id
        )

        critical = sum(
            item.severity
            == InspectionSeverity.CRITICAL
            for item in findings
        )

        high = sum(
            item.severity
            == InspectionSeverity.HIGH
            for item in findings
        )

        return (
            f"Inspection of {asset} completed "
            f"with status {status.value}. "
            f"{len(findings)} finding(s) were "
            f"identified, including {critical} "
            f"critical and {high} high-severity "
            f"finding(s). Overall severity is "
            f"{severity.value}; inspection health "
            f"score is {inspection_score:.1f}/100."
        )

    # -----------------------------------------------------
    # Post processing
    # -----------------------------------------------------

    async def _post_process(
        self,
        result: InspectionResult,
    ) -> None:

        self._record_metrics(result)
        self._log_result(result)

        if self.repository is not None:
            try:
                await (
                    self.repository
                    .save_inspection(result)
                )
            except Exception:
                pass

        if (
            self.config
            .auto_create_work_order
            and self.work_order_service
            and result
            .maintenance_required
        ):
            await self._create_work_order(
                result
            )

        if (
            self.config
            .auto_request_failure_analysis
            and self.failure_analysis_service
            and result
            .failure_analysis_required
        ):
            await self._request_failure_analysis(
                result
            )

        if (
            self.config.enable_notifications
            and self.notification_service
            and result.overall_severity
            in {
                InspectionSeverity.HIGH,
                InspectionSeverity.CRITICAL,
            }
        ):
            await self._notify(result)

    # -----------------------------------------------------
    # Work order
    # -----------------------------------------------------

    async def _create_work_order(
        self,
        result: InspectionResult,
    ) -> None:

        if self.work_order_service is None:
            return

        try:
            await (
                self.work_order_service
                .create_work_order(
                    asset_id=(
                        result.asset_id
                    ),

                    title=(
                        "Inspection Findings: "
                        f"{result.asset_name or result.asset_id}"
                    ),

                    description=(
                        result.summary
                    ),

                    priority=(
                        self._work_order_priority(
                            result
                            .overall_severity
                        )
                    ),

                    metadata={
                        "inspection_id":
                            result.inspection_id,

                        "incident_id":
                            result.incident_id,

                        "finding_count":
                            result.total_findings,

                        "critical_findings":
                            result
                            .critical_findings,

                        "inspection_score":
                            result
                            .inspection_score,
                    },
                )
            )

        except Exception:
            pass

    @staticmethod
    def _work_order_priority(
        severity: InspectionSeverity,
    ) -> str:

        mapping = {
            InspectionSeverity.NONE:
                "low",

            InspectionSeverity.LOW:
                "low",

            InspectionSeverity.MODERATE:
                "normal",

            InspectionSeverity.HIGH:
                "urgent",

            InspectionSeverity.CRITICAL:
                "immediate",
        }

        return mapping[severity]

    # -----------------------------------------------------
    # Failure analysis
    # -----------------------------------------------------

    async def _request_failure_analysis(
        self,
        result: InspectionResult,
    ) -> None:

        if (
            self.failure_analysis_service
            is None
        ):
            return

        findings = [
            {
                "finding_id":
                    item.finding_id,

                "finding_type":
                    item.finding_type.value,

                "source":
                    item.source.value,

                "severity":
                    item.severity.value,

                "severity_score":
                    item.severity_score,

                "confidence":
                    item.confidence,

                "component":
                    item.component,

                "description":
                    item.description,
            }
            for item in result.findings
        ]

        try:
            await (
                self.failure_analysis_service
                .request_analysis(
                    asset_id=(
                        result.asset_id
                    ),

                    inspection_id=(
                        result.inspection_id
                    ),

                    findings=findings,

                    metadata={
                        "incident_id":
                            result.incident_id,

                        "inspection_score":
                            result
                            .inspection_score,

                        "rul_hours":
                            result
                            .predicted_rul_hours,

                        "failure_probability":
                            result
                            .failure_probability,
                    },
                )
            )

        except Exception:
            pass

    # -----------------------------------------------------
    # Notification
    # -----------------------------------------------------

    async def _notify(
        self,
        result: InspectionResult,
    ) -> None:

        if self.notification_service is None:
            return

        try:
            await (
                self.notification_service
                .notify(
                    title=(
                        "Industrial Inspection Alert"
                    ),

                    message=result.summary,

                    severity=(
                        result
                        .overall_severity
                        .value
                    ),

                    metadata={
                        "inspection_id":
                            result.inspection_id,

                        "asset_id":
                            result.asset_id,

                        "site_id":
                            result.site_id,

                        "critical_findings":
                            result
                            .critical_findings,

                        "shutdown_review":
                            result
                            .shutdown_review_required,
                    },
                )
            )

        except Exception:
            pass

    # -----------------------------------------------------
    # Observability
    # -----------------------------------------------------

    @staticmethod
    def _record_metrics(
        result: InspectionResult,
    ) -> None:

        try:
            from app.observability.metrics import (
                registry,
            )

            registry.counter(
                "inspection_agent_runs_total"
            ).inc(
                labels={
                    "type":
                        result
                        .inspection_type.value,

                    "status":
                        result.status.value,
                }
            )

            registry.histogram(
                "inspection_agent_score"
            ).observe(
                result.inspection_score,

                labels={
                    "type":
                        result
                        .inspection_type.value,
                },
            )

            registry.histogram(
                "inspection_agent_findings"
            ).observe(
                result.total_findings,

                labels={
                    "severity":
                        result
                        .overall_severity.value,
                },
            )

            registry.counter(
                "inspection_findings_total"
            ).inc(
                amount=(
                    result.total_findings
                ),

                labels={
                    "severity":
                        result
                        .overall_severity.value,
                },
            )

        except Exception:
            pass

    @staticmethod
    def _log_result(
        result: InspectionResult,
    ) -> None:

        try:
            from app.observability.logging import (
                get_logger,
            )

            logger = get_logger(
                "industrial.inspection_agent"
            )

            payload = {
                "inspection_id":
                    result.inspection_id,

                "asset_id":
                    result.asset_id,

                "status":
                    result.status.value,

                "severity":
                    result
                    .overall_severity.value,

                "score":
                    result.inspection_score,

                "findings":
                    result.total_findings,

                "critical_findings":
                    result.critical_findings,

                "maintenance_required":
                    result
                    .maintenance_required,

                "shutdown_review":
                    result
                    .shutdown_review_required,
            }

            if (
                result.overall_severity
                == InspectionSeverity.CRITICAL
            ):
                logger.critical(
                    "Critical asset inspection "
                    "findings detected.",

                    extra={
                        "inspection":
                            payload
                    },
                )

            elif (
                result.overall_severity
                == InspectionSeverity.HIGH
            ):
                logger.warning(
                    "High-severity asset "
                    "inspection findings.",

                    extra={
                        "inspection":
                            payload
                    },
                )

            else:
                logger.info(
                    "Asset inspection completed.",

                    extra={
                        "inspection":
                            payload
                    },
                )

        except Exception:
            pass


# =========================================================
# Factory
# =========================================================

def create_inspection_agent(
    *,
    config: InspectionAgentConfig | None = None,
    repository: InspectionRepository | None = None,
    work_order_service: WorkOrderService | None = None,
    notification_service: NotificationService | None = None,
    failure_analysis_service: FailureAnalysisService | None = None,
) -> InspectionAgent:

    return InspectionAgent(
        config=config,

        repository=repository,

        work_order_service=(
            work_order_service
        ),

        notification_service=(
            notification_service
        ),

        failure_analysis_service=(
            failure_analysis_service
        ),
    )


inspection_agent = InspectionAgent()


__all__ = [
    "InspectionType",
    "InspectionStatus",
    "InspectionSeverity",

    "FindingType",
    "FindingSource",

    "InspectionAction",
    "InspectionPriority",

    "BoundingBox",
    "VisualDetection",

    "InspectionSensorContext",
    "InspectionAssetContext",

    "InspectionObservation",

    "InspectionAgentInput",

    "InspectionFinding",
    "InspectionRecommendation",

    "InspectionResult",

    "InspectionRepository",
    "WorkOrderService",
    "NotificationService",
    "FailureAnalysisService",

    "InspectionAgentConfig",
    "InspectionAgent",

    "create_inspection_agent",
    "inspection_agent",
]