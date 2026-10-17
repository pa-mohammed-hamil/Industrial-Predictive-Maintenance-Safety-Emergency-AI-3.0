# app/agents/failure_analysis_agent.py

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
        raise ValueError(f"{name} must be between 0 and 1.")
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

class FailureMode(str, Enum):
    BEARING_FAILURE = "bearing_failure"
    MOTOR_FAILURE = "motor_failure"
    GEAR_FAILURE = "gear_failure"
    SHAFT_FAILURE = "shaft_failure"

    PUMP_FAILURE = "pump_failure"
    VALVE_FAILURE = "valve_failure"

    ELECTRICAL_FAILURE = "electrical_failure"

    OVERHEATING = "overheating"

    LUBRICATION_FAILURE = "lubrication_failure"
    COOLING_FAILURE = "cooling_failure"

    PRESSURE_FAILURE = "pressure_failure"

    HYDRAULIC_FAILURE = "hydraulic_failure"
    PNEUMATIC_FAILURE = "pneumatic_failure"

    SENSOR_FAILURE = "sensor_failure"
    CONTROL_FAILURE = "control_failure"

    STRUCTURAL_FAILURE = "structural_failure"

    ALIGNMENT_FAILURE = "alignment_failure"
    IMBALANCE = "imbalance"

    LOOSENESS = "looseness"
    FATIGUE = "fatigue"

    CORROSION = "corrosion"
    LEAKAGE = "leakage"

    UNKNOWN = "unknown"


class FailureSeverity(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class FailureUrgency(str, Enum):
    MONITOR = "monitor"
    SCHEDULE = "schedule"
    URGENT = "urgent"
    IMMEDIATE = "immediate"


class FailureTrend(str, Enum):
    STABLE = "stable"
    IMPROVING = "improving"
    DEGRADING = "degrading"
    RAPIDLY_DEGRADING = "rapidly_degrading"
    UNKNOWN = "unknown"


class EvidenceType(str, Enum):
    TEMPERATURE = "temperature"
    VIBRATION = "vibration"
    PRESSURE = "pressure"
    CURRENT = "current"
    ENERGY = "energy"

    ANOMALY = "anomaly"
    RUL = "rul"

    FAILURE_MODEL = "failure_model"

    COMPUTER_VISION = "computer_vision"

    MAINTENANCE_HISTORY = "maintenance_history"

    WORK_ORDER = "work_order"

    OPERATOR_REPORT = "operator_report"

    INSPECTION = "inspection"

    TELEMETRY = "telemetry"

    OTHER = "other"


class RecommendedAction(str, Enum):
    CONTINUE_MONITORING = "continue_monitoring"

    INSPECT = "inspect"

    LUBRICATE = "lubricate"

    ALIGN = "align"

    BALANCE = "balance"

    TIGHTEN = "tighten"

    COOL_DOWN = "cool_down"

    REDUCE_LOAD = "reduce_load"

    REPAIR = "repair"

    REPLACE_COMPONENT = "replace_component"

    SCHEDULE_MAINTENANCE = "schedule_maintenance"

    CREATE_WORK_ORDER = "create_work_order"

    SHUTDOWN_RECOMMENDED = "shutdown_recommended"

    EMERGENCY_REVIEW = "emergency_review"

    HUMAN_REVIEW = "human_review"


# =========================================================
# Evidence
# =========================================================

@dataclass(slots=True)
class FailureEvidence:
    evidence_type: EvidenceType

    source: str

    description: str

    confidence: float = 1.0

    severity_score: float = 0.0

    value: float | None = None
    unit: str | None = None

    timestamp: datetime = field(
        default_factory=utc_now
    )

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

        if self.value is not None:
            self.value = _finite(
                self.value,
                name="value",
            )


# =========================================================
# Sensor context
# =========================================================

@dataclass(slots=True)
class SensorContext:
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

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# =========================================================
# Model context
# =========================================================

@dataclass(slots=True)
class FailureModelContext:
    failure_probability: float | None = None

    predicted_failure_mode: FailureMode | None = None

    failure_confidence: float | None = None

    anomaly_score: float | None = None

    predicted_rul_hours: float | None = None

    rul_confidence: float | None = None

    asset_health_score: float | None = None

    trend: FailureTrend = FailureTrend.UNKNOWN

    model_name: str | None = None
    model_version: str | None = None

    data_drift_detected: bool = False
    model_drift_detected: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        for name in (
            "failure_probability",
            "failure_confidence",
            "anomaly_score",
            "rul_confidence",
        ):
            value = getattr(self, name)

            if value is not None:
                setattr(
                    self,
                    name,
                    _probability(
                        value,
                        name=name,
                    ),
                )

        if (
            self.predicted_rul_hours is not None
            and self.predicted_rul_hours < 0
        ):
            raise ValueError(
                "predicted_rul_hours cannot be negative."
            )

        if (
            self.asset_health_score is not None
            and not 0.0 <= self.asset_health_score <= 100.0
        ):
            raise ValueError(
                "asset_health_score must be between 0 and 100."
            )


# =========================================================
# Asset context
# =========================================================

@dataclass(slots=True)
class AssetContext:
    asset_id: str

    asset_name: str | None = None
    asset_type: str | None = None

    site_id: str | None = None

    operating: bool = True

    load_percent: float | None = None

    criticality: float = 0.5

    age_hours: float | None = None

    last_maintenance_hours_ago: float | None = None

    safety_critical: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.criticality = _probability(
            self.criticality,
            name="criticality",
        )


# =========================================================
# Input
# =========================================================

@dataclass(slots=True)
class FailureAnalysisInput:
    asset: AssetContext

    sensors: SensorContext = field(
        default_factory=SensorContext
    )

    model: FailureModelContext = field(
        default_factory=FailureModelContext
    )

    evidence: list[FailureEvidence] = field(
        default_factory=list
    )

    maintenance_history: list[
        Mapping[str, Any]
    ] = field(
        default_factory=list
    )

    inspection_findings: list[
        Mapping[str, Any]
    ] = field(
        default_factory=list
    )

    prediction_id: str | None = None

    anomaly_id: str | None = None

    request_id: str | None = None
    trace_id: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# =========================================================
# Root cause models
# =========================================================

@dataclass(slots=True)
class RootCauseCandidate:
    failure_mode: FailureMode

    probability: float

    confidence: float

    score: float

    supporting_evidence: list[str]

    contradicting_evidence: list[str] = field(
        default_factory=list
    )

    explanation: str = ""


@dataclass(slots=True)
class FailureAction:
    action: RecommendedAction

    priority: FailureUrgency

    reason: str

    due_within_hours: float | None = None

    requires_human_approval: bool = True

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class FailureAnalysisResult:
    analysis_id: str

    asset_id: str

    asset_name: str | None
    site_id: str | None

    failure_detected: bool

    primary_failure_mode: FailureMode

    failure_probability: float

    confidence: float

    severity: FailureSeverity

    urgency: FailureUrgency

    predicted_rul_hours: float | None

    health_score: float | None

    trend: FailureTrend

    root_causes: list[
        RootCauseCandidate
    ]

    evidence: list[
        FailureEvidence
    ]

    actions: list[
        FailureAction
    ]

    model_reliability_warning: bool

    shutdown_review_required: bool

    summary: str

    created_at: datetime = field(
        default_factory=utc_now
    )

    prediction_id: str | None = None

    anomaly_id: str | None = None

    trace_id: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,

            "asset_id": self.asset_id,

            "asset_name": self.asset_name,

            "site_id": self.site_id,

            "failure_detected":
                self.failure_detected,

            "primary_failure_mode":
                self.primary_failure_mode.value,

            "failure_probability":
                self.failure_probability,

            "confidence":
                self.confidence,

            "severity":
                self.severity.value,

            "urgency":
                self.urgency.value,

            "predicted_rul_hours":
                self.predicted_rul_hours,

            "health_score":
                self.health_score,

            "trend":
                self.trend.value,

            "root_causes": [
                {
                    **asdict(item),

                    "failure_mode":
                        item.failure_mode.value,
                }
                for item in self.root_causes
            ],

            "evidence": [
                {
                    **asdict(item),

                    "evidence_type":
                        item.evidence_type.value,

                    "timestamp":
                        item.timestamp.isoformat(),
                }
                for item in self.evidence
            ],

            "actions": [
                {
                    **asdict(item),

                    "action":
                        item.action.value,

                    "priority":
                        item.priority.value,
                }
                for item in self.actions
            ],

            "model_reliability_warning":
                self.model_reliability_warning,

            "shutdown_review_required":
                self.shutdown_review_required,

            "summary":
                self.summary,

            "created_at":
                self.created_at.isoformat(),

            "prediction_id":
                self.prediction_id,

            "anomaly_id":
                self.anomaly_id,

            "trace_id":
                self.trace_id,

            "metadata":
                dict(self.metadata),
        }


# =========================================================
# Integration protocols
# =========================================================

class FailureRepository(Protocol):

    async def save_analysis(
        self,
        result: FailureAnalysisResult,
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


# =========================================================
# Configuration
# =========================================================

@dataclass(slots=True)
class FailureAnalysisConfig:
    detection_threshold: float = 0.55

    high_probability_threshold: float = 0.70

    critical_probability_threshold: float = 0.90

    anomaly_warning_threshold: float = 0.70

    critical_rul_hours: float = 24.0

    urgent_rul_hours: float = 72.0

    warning_rul_hours: float = 168.0

    poor_health_threshold: float = 40.0

    critical_health_threshold: float = 20.0

    minimum_model_confidence: float = 0.60

    max_root_causes: int = 5

    auto_create_work_order: bool = False

    enable_notifications: bool = True


# =========================================================
# Agent
# =========================================================

class FailureAnalysisAgent:

    def __init__(
        self,
        *,
        config: FailureAnalysisConfig | None = None,
        repository: FailureRepository | None = None,
        work_order_service: WorkOrderService | None = None,
        notification_service: NotificationService | None = None,
    ) -> None:

        self.config = (
            config
            or FailureAnalysisConfig()
        )

        self.repository = repository

        self.work_order_service = (
            work_order_service
        )

        self.notification_service = (
            notification_service
        )

    # -----------------------------------------------------
    # Main execution
    # -----------------------------------------------------

    async def run(
        self,
        data: FailureAnalysisInput,
    ) -> FailureAnalysisResult:

        evidence = list(
            data.evidence
        )

        evidence.extend(
            self._sensor_evidence(data)
        )

        evidence.extend(
            self._model_evidence(data)
        )

        root_causes = (
            self._analyse_root_causes(
                data,
                evidence,
            )
        )

        primary_mode = (
            root_causes[0].failure_mode
            if root_causes
            else (
                data.model
                .predicted_failure_mode
                or FailureMode.UNKNOWN
            )
        )

        probability = (
            self._failure_probability(
                data,
                root_causes,
            )
        )

        confidence = (
            self._analysis_confidence(
                data,
                evidence,
                root_causes,
            )
        )

        severity = (
            self._severity(
                data,
                probability,
            )
        )

        urgency = (
            self._urgency(
                data,
                severity,
            )
        )

        reliability_warning = (
            self._reliability_warning(
                data
            )
        )

        failure_detected = (
            probability
            >= self.config
            .detection_threshold
        )

        actions = (
            self._recommend_actions(
                data=data,
                mode=primary_mode,
                probability=probability,
                severity=severity,
                urgency=urgency,
                reliability_warning=(
                    reliability_warning
                ),
            )
        )

        shutdown_review = any(
            action.action
            in {
                RecommendedAction
                .SHUTDOWN_RECOMMENDED,

                RecommendedAction
                .EMERGENCY_REVIEW,
            }
            for action in actions
        )

        summary = self._summary(
            data=data,
            mode=primary_mode,
            probability=probability,
            severity=severity,
            urgency=urgency,
        )

        result = FailureAnalysisResult(
            analysis_id=new_id(),

            asset_id=data.asset.asset_id,

            asset_name=(
                data.asset.asset_name
            ),

            site_id=(
                data.asset.site_id
            ),

            failure_detected=(
                failure_detected
            ),

            primary_failure_mode=(
                primary_mode
            ),

            failure_probability=round(
                probability,
                4,
            ),

            confidence=round(
                confidence,
                4,
            ),

            severity=severity,

            urgency=urgency,

            predicted_rul_hours=(
                data.model
                .predicted_rul_hours
            ),

            health_score=(
                data.model
                .asset_health_score
            ),

            trend=data.model.trend,

            root_causes=root_causes,

            evidence=evidence,

            actions=actions,

            model_reliability_warning=(
                reliability_warning
            ),

            shutdown_review_required=(
                shutdown_review
            ),

            summary=summary,

            prediction_id=(
                data.prediction_id
            ),

            anomaly_id=(
                data.anomaly_id
            ),

            trace_id=(
                data.trace_id
            ),

            metadata={
                **data.metadata,

                "model_name":
                    data.model.model_name,

                "model_version":
                    data.model.model_version,

                "request_id":
                    data.request_id,
            },
        )

        await self._post_process(
            result
        )

        return result

    # -----------------------------------------------------
    # Sensor evidence
    # -----------------------------------------------------

    def _sensor_evidence(
        self,
        data: FailureAnalysisInput,
    ) -> list[FailureEvidence]:

        sensor = data.sensors

        evidence: list[
            FailureEvidence
        ] = []

        # Temperature
        if sensor.temperature is not None:

            severity = 0.0

            if (
                sensor.temperature_critical
                is not None
                and sensor.temperature
                >= sensor.temperature_critical
            ):
                severity = 1.0

            elif (
                sensor.temperature_warning
                is not None
                and sensor.temperature
                >= sensor.temperature_warning
            ):
                severity = 0.70

            if severity > 0:
                evidence.append(
                    FailureEvidence(
                        evidence_type=(
                            EvidenceType
                            .TEMPERATURE
                        ),

                        source=(
                            "temperature_sensor"
                        ),

                        description=(
                            "Elevated asset "
                            "temperature detected."
                        ),

                        confidence=0.95,

                        severity_score=severity,

                        value=(
                            sensor.temperature
                        ),

                        unit="°C",
                    )
                )

        # Vibration
        if sensor.vibration_rms is not None:

            severity = 0.0

            if (
                sensor.vibration_critical
                is not None
                and sensor.vibration_rms
                >= sensor.vibration_critical
            ):
                severity = 1.0

            elif (
                sensor.vibration_warning
                is not None
                and sensor.vibration_rms
                >= sensor.vibration_warning
            ):
                severity = 0.75

            if severity > 0:
                evidence.append(
                    FailureEvidence(
                        evidence_type=(
                            EvidenceType
                            .VIBRATION
                        ),

                        source=(
                            "vibration_sensor"
                        ),

                        description=(
                            "Abnormal vibration "
                            "detected."
                        ),

                        confidence=0.95,

                        severity_score=severity,

                        value=(
                            sensor.vibration_rms
                        ),

                        unit="mm/s",
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

            if ratio >= 1.15:
                evidence.append(
                    FailureEvidence(
                        evidence_type=(
                            EvidenceType.CURRENT
                        ),

                        source=(
                            "current_sensor"
                        ),

                        description=(
                            "Electrical current is "
                            "above nominal level."
                        ),

                        confidence=0.90,

                        severity_score=(
                            _clamp(
                                (
                                    ratio
                                    - 1.0
                                )
                                / 0.50
                            )
                        ),

                        value=sensor.current,

                        unit="A",

                        metadata={
                            "nominal_current":
                                sensor
                                .nominal_current,

                            "ratio": ratio,
                        },
                    )
                )

        # Pressure
        if (
            sensor.pressure is not None
            and sensor.nominal_pressure
            and sensor.nominal_pressure > 0
        ):
            deviation = abs(
                sensor.pressure
                - sensor.nominal_pressure
            ) / sensor.nominal_pressure

            if deviation >= 0.15:
                evidence.append(
                    FailureEvidence(
                        evidence_type=(
                            EvidenceType.PRESSURE
                        ),

                        source=(
                            "pressure_sensor"
                        ),

                        description=(
                            "Pressure deviation "
                            "from nominal operating "
                            "range detected."
                        ),

                        confidence=0.90,

                        severity_score=(
                            _clamp(
                                deviation
                                / 0.50
                            )
                        ),

                        value=sensor.pressure,

                        metadata={
                            "nominal_pressure":
                                sensor
                                .nominal_pressure,

                            "deviation_ratio":
                                deviation,
                        },
                    )
                )

        return evidence

    # -----------------------------------------------------
    # Model evidence
    # -----------------------------------------------------

    def _model_evidence(
        self,
        data: FailureAnalysisInput,
    ) -> list[FailureEvidence]:

        model = data.model

        evidence: list[
            FailureEvidence
        ] = []

        if (
            model.failure_probability
            is not None
        ):
            evidence.append(
                FailureEvidence(
                    evidence_type=(
                        EvidenceType
                        .FAILURE_MODEL
                    ),

                    source=(
                        model.model_name
                        or "failure_model"
                    ),

                    description=(
                        "Predictive model "
                        "estimated failure "
                        "probability."
                    ),

                    confidence=(
                        model.failure_confidence
                        if (
                            model
                            .failure_confidence
                            is not None
                        )
                        else 0.80
                    ),

                    severity_score=(
                        model
                        .failure_probability
                    ),

                    value=(
                        model
                        .failure_probability
                    ),

                    unit="probability",
                )
            )

        if model.anomaly_score is not None:
            evidence.append(
                FailureEvidence(
                    evidence_type=(
                        EvidenceType.ANOMALY
                    ),

                    source="anomaly_model",

                    description=(
                        "Anomaly model score "
                        "available."
                    ),

                    confidence=0.85,

                    severity_score=(
                        model.anomaly_score
                    ),

                    value=(
                        model.anomaly_score
                    ),

                    unit="score",
                )
            )

        if (
            model.predicted_rul_hours
            is not None
        ):
            rul = (
                model.predicted_rul_hours
            )

            if rul <= self.config.critical_rul_hours:
                severity = 1.0

            elif rul <= self.config.urgent_rul_hours:
                severity = 0.85

            elif rul <= self.config.warning_rul_hours:
                severity = 0.60

            else:
                severity = 0.20

            evidence.append(
                FailureEvidence(
                    evidence_type=(
                        EvidenceType.RUL
                    ),

                    source="rul_model",

                    description=(
                        "Remaining useful life "
                        "prediction available."
                    ),

                    confidence=(
                        model.rul_confidence
                        if (
                            model.rul_confidence
                            is not None
                        )
                        else 0.80
                    ),

                    severity_score=severity,

                    value=rul,

                    unit="hours",
                )
            )

        return evidence

    # -----------------------------------------------------
    # Root cause analysis
    # -----------------------------------------------------

    def _analyse_root_causes(
        self,
        data: FailureAnalysisInput,
        evidence: Sequence[
            FailureEvidence
        ],
    ) -> list[RootCauseCandidate]:

        scores: dict[
            FailureMode,
            float,
        ] = {}

        support: dict[
            FailureMode,
            list[str],
        ] = {}

        def add(
            mode: FailureMode,
            score: float,
            reason: str,
        ) -> None:

            scores[mode] = (
                scores.get(mode, 0.0)
                + score
            )

            support.setdefault(
                mode,
                [],
            ).append(reason)

        # Model prediction receives strong
        # but not exclusive weight.
        if (
            data.model
            .predicted_failure_mode
            is not None
        ):
            add(
                data.model
                .predicted_failure_mode,

                (
                    data.model
                    .failure_probability
                    or 0.50
                )
                * 1.5,

                "Failure prediction model "
                "identified this mode.",
            )

        for item in evidence:

            strength = (
                item.confidence
                * item.severity_score
            )

            if (
                item.evidence_type
                == EvidenceType.VIBRATION
            ):
                add(
                    FailureMode
                    .BEARING_FAILURE,

                    strength * 1.00,

                    "Abnormal vibration can "
                    "indicate bearing degradation.",
                )

                add(
                    FailureMode.IMBALANCE,

                    strength * 0.70,

                    "Vibration pattern may "
                    "indicate imbalance.",
                )

                add(
                    FailureMode
                    .ALIGNMENT_FAILURE,

                    strength * 0.65,

                    "Vibration can indicate "
                    "shaft/coupling misalignment.",
                )

                add(
                    FailureMode.LOOSENESS,

                    strength * 0.55,

                    "Mechanical looseness is "
                    "consistent with elevated "
                    "vibration.",
                )

            elif (
                item.evidence_type
                == EvidenceType.TEMPERATURE
            ):
                add(
                    FailureMode.OVERHEATING,

                    strength * 1.10,

                    "High temperature directly "
                    "supports overheating.",
                )

                add(
                    FailureMode
                    .LUBRICATION_FAILURE,

                    strength * 0.65,

                    "Poor lubrication can increase "
                    "friction and temperature.",
                )

                add(
                    FailureMode
                    .BEARING_FAILURE,

                    strength * 0.55,

                    "Bearing degradation can "
                    "increase temperature.",
                )

                add(
                    FailureMode
                    .COOLING_FAILURE,

                    strength * 0.60,

                    "Cooling-system degradation "
                    "can cause overheating.",
                )

            elif (
                item.evidence_type
                == EvidenceType.CURRENT
            ):
                add(
                    FailureMode
                    .ELECTRICAL_FAILURE,

                    strength * 0.90,

                    "Abnormal current supports "
                    "electrical degradation or "
                    "overload.",
                )

                add(
                    FailureMode.MOTOR_FAILURE,

                    strength * 0.75,

                    "Current increase can indicate "
                    "motor loading or winding "
                    "problems.",
                )

                add(
                    FailureMode
                    .BEARING_FAILURE,

                    strength * 0.35,

                    "Mechanical resistance can "
                    "increase motor current.",
                )

            elif (
                item.evidence_type
                == EvidenceType.PRESSURE
            ):
                add(
                    FailureMode
                    .PRESSURE_FAILURE,

                    strength * 0.90,

                    "Abnormal pressure directly "
                    "supports pressure-system "
                    "degradation.",
                )

                add(
                    FailureMode.PUMP_FAILURE,

                    strength * 0.65,

                    "Pump degradation can cause "
                    "pressure deviation.",
                )

                add(
                    FailureMode
                    .VALVE_FAILURE,

                    strength * 0.55,

                    "Valve malfunction can cause "
                    "pressure instability.",
                )

                add(
                    FailureMode.LEAKAGE,

                    strength * 0.45,

                    "Pressure loss may indicate "
                    "leakage.",
                )

        if not scores:
            return []

        total = sum(
            max(score, 0.0)
            for score in scores.values()
        )

        candidates: list[
            RootCauseCandidate
        ] = []

        for mode, score in scores.items():

            probability = (
                score / total
                if total > 0
                else 0.0
            )

            evidence_count = len(
                support.get(mode, [])
            )

            confidence = _clamp(
                0.50
                + min(
                    evidence_count,
                    4,
                )
                * 0.10
                + min(
                    score,
                    1.0,
                )
                * 0.10
            )

            candidates.append(
                RootCauseCandidate(
                    failure_mode=mode,

                    probability=round(
                        probability,
                        4,
                    ),

                    confidence=round(
                        confidence,
                        4,
                    ),

                    score=round(
                        score,
                        4,
                    ),

                    supporting_evidence=(
                        support.get(
                            mode,
                            [],
                        )
                    ),

                    explanation=(
                        f"{mode.value} received "
                        f"{evidence_count} supporting "
                        "evidence signal(s)."
                    ),
                )
            )

        candidates.sort(
            key=lambda item: (
                item.score,
                item.confidence,
            ),
            reverse=True,
        )

        return candidates[
            : self.config.max_root_causes
        ]

    # -----------------------------------------------------
    # Probability
    # -----------------------------------------------------

    @staticmethod
    def _failure_probability(
        data: FailureAnalysisInput,
        causes: Sequence[
            RootCauseCandidate
        ],
    ) -> float:

        values: list[
            tuple[float, float]
        ] = []

        if (
            data.model
            .failure_probability
            is not None
        ):
            values.append(
                (
                    data.model
                    .failure_probability,
                    0.55,
                )
            )

        if data.model.anomaly_score is not None:
            values.append(
                (
                    data.model.anomaly_score,
                    0.20,
                )
            )

        if causes:
            root_signal = _clamp(
                causes[0].score
            )

            values.append(
                (
                    root_signal,
                    0.15,
                )
            )

        if (
            data.model
            .asset_health_score
            is not None
        ):
            health_risk = (
                1.0
                - (
                    data.model
                    .asset_health_score
                    / 100.0
                )
            )

            values.append(
                (
                    health_risk,
                    0.10,
                )
            )

        if not values:
            return 0.0

        weight = sum(
            item[1]
            for item in values
        )

        return _clamp(
            sum(
                value * w
                for value, w in values
            )
            / weight
        )

    # -----------------------------------------------------
    # Confidence
    # -----------------------------------------------------

    def _analysis_confidence(
        self,
        data: FailureAnalysisInput,
        evidence: Sequence[
            FailureEvidence
        ],
        causes: Sequence[
            RootCauseCandidate
        ],
    ) -> float:

        values = [
            item.confidence
            for item in evidence
        ]

        if causes:
            values.append(
                causes[0].confidence
            )

        if not values:
            return 0.0

        confidence = (
            sum(values)
            / len(values)
        )

        # Independent evidence improves
        # confidence slightly.
        independent_sources = {
            item.source
            for item in evidence
        }

        if len(independent_sources) >= 3:
            confidence += 0.05

        if (
            data.model.data_drift_detected
            or data.model
            .model_drift_detected
        ):
            confidence *= 0.80

        return _clamp(confidence)

    # -----------------------------------------------------
    # Severity
    # -----------------------------------------------------

    def _severity(
        self,
        data: FailureAnalysisInput,
        probability: float,
    ) -> FailureSeverity:

        model = data.model
        asset = data.asset

        if (
            probability
            >= self.config
            .critical_probability_threshold
        ):
            return FailureSeverity.CRITICAL

        if (
            model.predicted_rul_hours
            is not None
            and model.predicted_rul_hours
            <= self.config
            .critical_rul_hours
        ):
            return FailureSeverity.CRITICAL

        if (
            model.asset_health_score
            is not None
            and model.asset_health_score
            <= self.config
            .critical_health_threshold
        ):
            return FailureSeverity.CRITICAL

        if (
            asset.safety_critical
            and probability >= 0.70
        ):
            return FailureSeverity.CRITICAL

        if (
            probability
            >= self.config
            .high_probability_threshold
        ):
            return FailureSeverity.HIGH

        if (
            model.predicted_rul_hours
            is not None
            and model.predicted_rul_hours
            <= self.config
            .urgent_rul_hours
        ):
            return FailureSeverity.HIGH

        if probability >= 0.40:
            return (
                FailureSeverity.MODERATE
            )

        return FailureSeverity.LOW

    # -----------------------------------------------------
    # Urgency
    # -----------------------------------------------------

    def _urgency(
        self,
        data: FailureAnalysisInput,
        severity: FailureSeverity,
    ) -> FailureUrgency:

        rul = (
            data.model
            .predicted_rul_hours
        )

        if (
            severity
            == FailureSeverity.CRITICAL
        ):
            return FailureUrgency.IMMEDIATE

        if (
            rul is not None
            and rul
            <= self.config
            .critical_rul_hours
        ):
            return FailureUrgency.IMMEDIATE

        if (
            severity
            == FailureSeverity.HIGH
            or (
                rul is not None
                and rul
                <= self.config
                .urgent_rul_hours
            )
        ):
            return FailureUrgency.URGENT

        if (
            severity
            == FailureSeverity.MODERATE
        ):
            return FailureUrgency.SCHEDULE

        return FailureUrgency.MONITOR

    # -----------------------------------------------------
    # Reliability
    # -----------------------------------------------------

    def _reliability_warning(
        self,
        data: FailureAnalysisInput,
    ) -> bool:

        model = data.model

        if (
            model.data_drift_detected
            or model.model_drift_detected
        ):
            return True

        if (
            model.failure_confidence
            is not None
            and model.failure_confidence
            < self.config
            .minimum_model_confidence
        ):
            return True

        return False

    # -----------------------------------------------------
    # Actions
    # -----------------------------------------------------

    def _recommend_actions(
        self,
        *,
        data: FailureAnalysisInput,
        mode: FailureMode,
        probability: float,
        severity: FailureSeverity,
        urgency: FailureUrgency,
        reliability_warning: bool,
    ) -> list[FailureAction]:

        actions: list[
            FailureAction
        ] = []

        if reliability_warning:
            actions.append(
                FailureAction(
                    action=(
                        RecommendedAction
                        .HUMAN_REVIEW
                    ),

                    priority=(
                        FailureUrgency.URGENT
                    ),

                    reason=(
                        "Prediction reliability is "
                        "reduced because of drift or "
                        "low model confidence."
                    ),

                    due_within_hours=4.0,
                )
            )

        if severity == FailureSeverity.LOW:
            actions.append(
                FailureAction(
                    action=(
                        RecommendedAction
                        .CONTINUE_MONITORING
                    ),

                    priority=(
                        FailureUrgency.MONITOR
                    ),

                    reason=(
                        "Current evidence does not "
                        "indicate an urgent failure."
                    ),

                    requires_human_approval=False,
                )
            )

        elif severity == FailureSeverity.MODERATE:
            actions.extend([
                FailureAction(
                    action=(
                        RecommendedAction.INSPECT
                    ),

                    priority=(
                        FailureUrgency.SCHEDULE
                    ),

                    reason=(
                        "Inspect the asset to verify "
                        "the developing failure mode."
                    ),

                    due_within_hours=72.0,
                ),

                FailureAction(
                    action=(
                        RecommendedAction
                        .SCHEDULE_MAINTENANCE
                    ),

                    priority=(
                        FailureUrgency.SCHEDULE
                    ),

                    reason=(
                        "Schedule condition-based "
                        "maintenance."
                    ),

                    due_within_hours=168.0,
                ),
            ])

        elif severity == FailureSeverity.HIGH:
            actions.extend([
                FailureAction(
                    action=(
                        RecommendedAction.INSPECT
                    ),

                    priority=(
                        FailureUrgency.URGENT
                    ),

                    reason=(
                        "High failure risk requires "
                        "priority inspection."
                    ),

                    due_within_hours=24.0,
                ),

                FailureAction(
                    action=(
                        RecommendedAction
                        .CREATE_WORK_ORDER
                    ),

                    priority=(
                        FailureUrgency.URGENT
                    ),

                    reason=(
                        "Create a maintenance work "
                        "order for the predicted "
                        "failure."
                    ),

                    due_within_hours=24.0,
                ),
            ])

        else:
            actions.extend([
                FailureAction(
                    action=(
                        RecommendedAction
                        .EMERGENCY_REVIEW
                    ),

                    priority=(
                        FailureUrgency.IMMEDIATE
                    ),

                    reason=(
                        "Critical failure indicators "
                        "require immediate operator "
                        "and maintenance review."
                    ),

                    due_within_hours=0.0,
                ),

                FailureAction(
                    action=(
                        RecommendedAction
                        .SHUTDOWN_RECOMMENDED
                    ),

                    priority=(
                        FailureUrgency.IMMEDIATE
                    ),

                    reason=(
                        "A controlled shutdown should "
                        "be considered through the "
                        "authorized safety process."
                    ),

                    due_within_hours=0.0,

                    requires_human_approval=True,
                ),
            ])

        actions.extend(
            self._mode_actions(
                mode,
                urgency,
            )
        )

        return self._deduplicate_actions(
            actions
        )

    @staticmethod
    def _mode_actions(
        mode: FailureMode,
        urgency: FailureUrgency,
    ) -> list[FailureAction]:

        mapping = {
            FailureMode.LUBRICATION_FAILURE:
                RecommendedAction.LUBRICATE,

            FailureMode.ALIGNMENT_FAILURE:
                RecommendedAction.ALIGN,

            FailureMode.IMBALANCE:
                RecommendedAction.BALANCE,

            FailureMode.LOOSENESS:
                RecommendedAction.TIGHTEN,

            FailureMode.OVERHEATING:
                RecommendedAction.COOL_DOWN,

            FailureMode.BEARING_FAILURE:
                RecommendedAction
                .REPLACE_COMPONENT,

            FailureMode.MOTOR_FAILURE:
                RecommendedAction.REPAIR,

            FailureMode.PUMP_FAILURE:
                RecommendedAction.REPAIR,

            FailureMode.VALVE_FAILURE:
                RecommendedAction.REPAIR,

            FailureMode
            .ELECTRICAL_FAILURE:
                RecommendedAction.INSPECT,
        }

        action = mapping.get(mode)

        if action is None:
            return []

        return [
            FailureAction(
                action=action,

                priority=urgency,

                reason=(
                    f"Recommended response for "
                    f"suspected {mode.value}."
                ),

                due_within_hours={
                    FailureUrgency.IMMEDIATE: 0.0,
                    FailureUrgency.URGENT: 24.0,
                    FailureUrgency.SCHEDULE: 72.0,
                    FailureUrgency.MONITOR: None,
                }[urgency],
            )
        ]

    @staticmethod
    def _deduplicate_actions(
        actions: Sequence[
            FailureAction
        ],
    ) -> list[FailureAction]:

        urgency_rank = {
            FailureUrgency.MONITOR: 1,
            FailureUrgency.SCHEDULE: 2,
            FailureUrgency.URGENT: 3,
            FailureUrgency.IMMEDIATE: 4,
        }

        unique: dict[
            RecommendedAction,
            FailureAction,
        ] = {}

        for action in actions:

            current = unique.get(
                action.action
            )

            if (
                current is None
                or urgency_rank[
                    action.priority
                ]
                > urgency_rank[
                    current.priority
                ]
            ):
                unique[action.action] = (
                    action
                )

        return list(unique.values())

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    @staticmethod
    def _summary(
        *,
        data: FailureAnalysisInput,
        mode: FailureMode,
        probability: float,
        severity: FailureSeverity,
        urgency: FailureUrgency,
    ) -> str:

        asset = (
            data.asset.asset_name
            or data.asset.asset_id
        )

        rul = (
            f" Predicted RUL is "
            f"{data.model.predicted_rul_hours:.1f} "
            f"hours."
            if (
                data.model
                .predicted_rul_hours
                is not None
            )
            else ""
        )

        return (
            f"{asset}: suspected "
            f"{mode.value} with "
            f"{probability:.1%} estimated failure "
            f"risk. Severity is {severity.value} "
            f"and response urgency is "
            f"{urgency.value}.{rul}"
        )

    # -----------------------------------------------------
    # Post-processing
    # -----------------------------------------------------

    async def _post_process(
        self,
        result: FailureAnalysisResult,
    ) -> None:

        self._record_metrics(result)
        self._log_result(result)

        if self.repository is not None:
            try:
                await (
                    self.repository
                    .save_analysis(result)
                )
            except Exception:
                pass

        if (
            self.config.auto_create_work_order
            and self.work_order_service
            and result.urgency
            in {
                FailureUrgency.URGENT,
                FailureUrgency.IMMEDIATE,
            }
        ):
            await self._create_work_order(
                result
            )

        if (
            self.config.enable_notifications
            and self.notification_service
            and result.severity
            in {
                FailureSeverity.HIGH,
                FailureSeverity.CRITICAL,
            }
        ):
            await self._notify(result)

    # -----------------------------------------------------
    # Work order
    # -----------------------------------------------------

    async def _create_work_order(
        self,
        result: FailureAnalysisResult,
    ) -> None:

        if self.work_order_service is None:
            return

        try:
            await (
                self.work_order_service
                .create_work_order(
                    asset_id=result.asset_id,

                    title=(
                        "AI Failure Analysis: "
                        f"{result.primary_failure_mode.value}"
                    ),

                    description=result.summary,

                    priority=(
                        result.urgency.value
                    ),

                    metadata={
                        "analysis_id":
                            result.analysis_id,

                        "failure_probability":
                            result
                            .failure_probability,

                        "failure_mode":
                            result
                            .primary_failure_mode
                            .value,

                        "rul_hours":
                            result
                            .predicted_rul_hours,
                    },
                )
            )

        except Exception:
            return

    # -----------------------------------------------------
    # Notifications
    # -----------------------------------------------------

    async def _notify(
        self,
        result: FailureAnalysisResult,
    ) -> None:

        if self.notification_service is None:
            return

        try:
            await (
                self.notification_service
                .notify(
                    title=(
                        "Industrial Failure Alert"
                    ),

                    message=result.summary,

                    severity=(
                        result.severity.value
                    ),

                    metadata={
                        "analysis_id":
                            result.analysis_id,

                        "asset_id":
                            result.asset_id,

                        "failure_mode":
                            result
                            .primary_failure_mode
                            .value,
                    },
                )
            )

        except Exception:
            return

    # -----------------------------------------------------
    # Observability
    # -----------------------------------------------------

    @staticmethod
    def _record_metrics(
        result: FailureAnalysisResult,
    ) -> None:

        try:
            from app.observability.metrics import (
                registry,
            )

            registry.counter(
                "failure_analysis_total"
            ).inc(
                labels={
                    "failure_mode":
                        result
                        .primary_failure_mode
                        .value,

                    "severity":
                        result.severity.value,
                }
            )

            registry.histogram(
                "failure_analysis_probability"
            ).observe(
                result.failure_probability,

                labels={
                    "failure_mode":
                        result
                        .primary_failure_mode
                        .value,
                },
            )

            if (
                result.predicted_rul_hours
                is not None
            ):
                registry.histogram(
                    "failure_analysis_rul_hours"
                ).observe(
                    result
                    .predicted_rul_hours,

                    labels={
                        "failure_mode":
                            result
                            .primary_failure_mode
                            .value,
                    },
                )

        except Exception:
            pass

    @staticmethod
    def _log_result(
        result: FailureAnalysisResult,
    ) -> None:

        try:
            from app.observability.logging import (
                get_logger,
            )

            logger = get_logger(
                "industrial.failure_analysis_agent"
            )

            logger.info(
                "Failure analysis completed.",
                extra={
                    "failure_analysis": {
                        "analysis_id":
                            result.analysis_id,

                        "asset_id":
                            result.asset_id,

                        "failure_mode":
                            result
                            .primary_failure_mode
                            .value,

                        "probability":
                            result
                            .failure_probability,

                        "severity":
                            result.severity.value,

                        "urgency":
                            result.urgency.value,
                    }
                },
            )

        except Exception:
            pass


# =========================================================
# Factory
# =========================================================

def create_failure_analysis_agent(
    *,
    config: FailureAnalysisConfig | None = None,
    repository: FailureRepository | None = None,
    work_order_service: WorkOrderService | None = None,
    notification_service: NotificationService | None = None,
) -> FailureAnalysisAgent:

    return FailureAnalysisAgent(
        config=config,
        repository=repository,

        work_order_service=(
            work_order_service
        ),

        notification_service=(
            notification_service
        ),
    )


failure_analysis_agent = (
    FailureAnalysisAgent()
)


__all__ = [
    "FailureMode",
    "FailureSeverity",
    "FailureUrgency",
    "FailureTrend",

    "EvidenceType",
    "RecommendedAction",

    "FailureEvidence",

    "SensorContext",
    "FailureModelContext",
    "AssetContext",

    "FailureAnalysisInput",

    "RootCauseCandidate",
    "FailureAction",
    "FailureAnalysisResult",

    "FailureRepository",
    "WorkOrderService",
    "NotificationService",

    "FailureAnalysisConfig",
    "FailureAnalysisAgent",

    "create_failure_analysis_agent",
    "failure_analysis_agent",
]