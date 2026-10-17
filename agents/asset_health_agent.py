# app/agents/asset_health_agent.py

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


def _finite(
    value: float,
    *,
    name: str,
) -> float:
    value = float(value)

    if not math.isfinite(value):
        raise ValueError(
            f"{name} must be finite."
        )

    return value


def _clamp(
    value: float,
    low: float = 0.0,
    high: float = 1.0,
) -> float:
    return max(
        low,
        min(high, value),
    )


def _probability(
    value: float,
    *,
    name: str,
) -> float:
    value = _finite(
        value,
        name=name,
    )

    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{name} must be between 0 and 1."
        )

    return value


# =========================================================
# Enums
# =========================================================

class AssetHealthStatus(str, Enum):
    HEALTHY = "healthy"
    WATCH = "watch"
    DEGRADED = "degraded"
    HIGH_RISK = "high_risk"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class ActionPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class ActionType(str, Enum):
    CONTINUE_OPERATION = "continue_operation"

    MONITOR = "monitor"
    INSPECT = "inspect"

    SCHEDULE_MAINTENANCE = (
        "schedule_maintenance"
    )

    CREATE_WORK_ORDER = (
        "create_work_order"
    )

    REPAIR = "repair"

    REPLACE_COMPONENT = (
        "replace_component"
    )

    REDUCE_LOAD = "reduce_load"

    SHUTDOWN = "shutdown"

    EMERGENCY_SHUTDOWN = (
        "emergency_shutdown"
    )

    HUMAN_REVIEW = "human_review"

    RETRAIN_MODEL = "retrain_model"

    CHECK_SENSOR = "check_sensor"


class SignalType(str, Enum):
    TEMPERATURE = "temperature"
    VIBRATION = "vibration"
    PRESSURE = "pressure"
    CURRENT = "current"
    ENERGY = "energy"

    ANOMALY = "anomaly"
    FAILURE = "failure"
    RUL = "rul"

    DEFECT = "defect"

    SAFETY = "safety"

    DATA_DRIFT = "data_drift"
    MODEL_DRIFT = "model_drift"

    MAINTENANCE = "maintenance"


# =========================================================
# Input models
# =========================================================

@dataclass(slots=True)
class SensorHealthSignal:
    signal_type: SignalType

    value: float

    normalized_risk: float

    unit: str | None = None

    threshold: float | None = None

    sensor_id: str | None = None

    status: str | None = None

    timestamp: datetime = field(
        default_factory=utc_now
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.value = _finite(
            self.value,
            name="value",
        )

        self.normalized_risk = (
            _probability(
                self.normalized_risk,
                name="normalized_risk",
            )
        )


@dataclass(slots=True)
class AnomalySignal:
    detected: bool

    score: float

    anomaly_type: str | None = None

    severity: str | None = None

    model_name: str | None = None
    model_version: str | None = None

    timestamp: datetime = field(
        default_factory=utc_now
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.score = _probability(
            self.score,
            name="anomaly score",
        )


@dataclass(slots=True)
class FailureSignal:
    failure_probability: float

    predicted_failure_mode: str | None = None

    time_to_failure_hours: float | None = None

    confidence: float | None = None

    model_name: str | None = None
    model_version: str | None = None

    timestamp: datetime = field(
        default_factory=utc_now
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.failure_probability = (
            _probability(
                self.failure_probability,
                name="failure_probability",
            )
        )

        if self.confidence is not None:
            self.confidence = (
                _probability(
                    self.confidence,
                    name="confidence",
                )
            )

        if (
            self.time_to_failure_hours
            is not None
            and self.time_to_failure_hours
            < 0
        ):
            raise ValueError(
                "time_to_failure_hours "
                "cannot be negative."
            )


@dataclass(slots=True)
class RULSignal:
    predicted_rul_hours: float

    confidence: float | None = None

    lower_bound_hours: float | None = None
    upper_bound_hours: float | None = None

    model_name: str | None = None
    model_version: str | None = None

    timestamp: datetime = field(
        default_factory=utc_now
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.predicted_rul_hours < 0:
            raise ValueError(
                "predicted_rul_hours "
                "cannot be negative."
            )

        if self.confidence is not None:
            self.confidence = (
                _probability(
                    self.confidence,
                    name="confidence",
                )
            )


@dataclass(slots=True)
class InspectionSignal:
    defect_detected: bool

    defect_type: str | None = None

    severity: str | None = None

    confidence: float | None = None

    component: str | None = None

    image_id: str | None = None

    timestamp: datetime = field(
        default_factory=utc_now
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.confidence is not None:
            self.confidence = (
                _probability(
                    self.confidence,
                    name="confidence",
                )
            )


@dataclass(slots=True)
class SafetySignal:
    risk_score: float

    emergency: bool = False

    hazard_type: str | None = None

    severity: str | None = None

    timestamp: datetime = field(
        default_factory=utc_now
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.risk_score = (
            _probability(
                self.risk_score,
                name="risk_score",
            )
        )


@dataclass(slots=True)
class DriftSignal:
    data_drift_detected: bool = False
    model_drift_detected: bool = False

    data_drift_score: float | None = None
    model_drift_score: float | None = None

    safety_critical_model_drift: bool = False

    timestamp: datetime = field(
        default_factory=utc_now
    )


@dataclass(slots=True)
class MaintenanceContext:
    hours_since_last_maintenance: float | None = None

    overdue: bool = False

    open_work_orders: int = 0

    repeated_failure_count: int = 0

    last_failure_mode: str | None = None

    maintenance_due_hours: float | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class AssetHealthInput:
    asset_id: str

    asset_name: str | None = None
    asset_type: str | None = None

    site_id: str | None = None
    location: str | None = None

    sensor_signals: list[
        SensorHealthSignal
    ] = field(
        default_factory=list
    )

    anomaly: AnomalySignal | None = None

    failure: FailureSignal | None = None

    rul: RULSignal | None = None

    inspections: list[
        InspectionSignal
    ] = field(
        default_factory=list
    )

    safety: SafetySignal | None = None

    drift: DriftSignal | None = None

    maintenance: (
        MaintenanceContext | None
    ) = None

    operating_load_percent: float | None = None

    production_criticality: float = 0.5

    asset_criticality: float = 0.5

    request_id: str | None = None
    trace_id: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.production_criticality = (
            _probability(
                self.production_criticality,
                name="production_criticality",
            )
        )

        self.asset_criticality = (
            _probability(
                self.asset_criticality,
                name="asset_criticality",
            )
        )


# =========================================================
# Agent output
# =========================================================

@dataclass(slots=True)
class HealthFactor:
    name: str

    risk_score: float

    weight: float

    contribution: float

    explanation: str

    critical: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class RecommendedAction:
    action: ActionType

    priority: ActionPriority

    reason: str

    component: str | None = None

    due_within_hours: float | None = None

    requires_human_approval: bool = True

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class AssetHealthAssessment:
    assessment_id: str

    asset_id: str
    asset_name: str | None

    site_id: str | None

    health_score: float
    risk_score: float

    status: AssetHealthStatus
    risk_level: RiskLevel

    confidence: float

    factors: list[HealthFactor]

    primary_risk_drivers: list[str]

    recommended_actions: list[
        RecommendedAction
    ]

    anomaly_detected: bool

    failure_probability: float | None

    predicted_failure_mode: str | None

    predicted_rul_hours: float | None

    defect_count: int

    safety_risk_score: float | None

    data_drift_detected: bool
    model_drift_detected: bool

    human_review_required: bool

    summary: str

    created_at: datetime = field(
        default_factory=utc_now
    )

    trace_id: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)

        data["status"] = (
            self.status.value
        )

        data["risk_level"] = (
            self.risk_level.value
        )

        data["created_at"] = (
            self.created_at.isoformat()
        )

        for factor in data["factors"]:
            pass

        for action in data[
            "recommended_actions"
        ]:
            if isinstance(
                action["action"],
                Enum,
            ):
                action["action"] = (
                    action["action"].value
                )

            if isinstance(
                action["priority"],
                Enum,
            ):
                action["priority"] = (
                    action["priority"].value
                )

        return data


# =========================================================
# Optional dependency protocols
# =========================================================

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


class KnowledgeService(Protocol):

    async def retrieve(
        self,
        *,
        query: str,
        asset_id: str | None = None,
        top_k: int = 5,
    ) -> Sequence[Any]:
        ...


# =========================================================
# Agent configuration
# =========================================================

@dataclass(slots=True)
class AssetHealthAgentConfig:
    healthy_max_risk: float = 0.20
    watch_max_risk: float = 0.40
    degraded_max_risk: float = 0.60
    high_risk_max_risk: float = 0.80

    critical_failure_probability: float = 0.85
    high_failure_probability: float = 0.65

    critical_rul_hours: float = 24.0
    warning_rul_hours: float = 168.0

    critical_safety_risk: float = 0.80

    sensor_weight: float = 0.20
    anomaly_weight: float = 0.15
    failure_weight: float = 0.25
    rul_weight: float = 0.15
    inspection_weight: float = 0.10
    safety_weight: float = 0.10
    maintenance_weight: float = 0.05

    minimum_confidence: float = 0.60

    auto_create_work_order: bool = False

    require_human_approval_for_shutdown: bool = True


# =========================================================
# Asset Health Agent
# =========================================================

class AssetHealthAgent:

    def __init__(
        self,
        *,
        config: AssetHealthAgentConfig | None = None,
        work_order_service: WorkOrderService | None = None,
        knowledge_service: KnowledgeService | None = None,
    ) -> None:

        self.config = (
            config
            or AssetHealthAgentConfig()
        )

        self.work_order_service = (
            work_order_service
        )

        self.knowledge_service = (
            knowledge_service
        )

    # -----------------------------------------------------
    # Main execution
    # -----------------------------------------------------

    async def run(
        self,
        input_data: AssetHealthInput,
    ) -> AssetHealthAssessment:

        with self._trace(
            input_data
        ) as span:

            factors = (
                self._build_health_factors(
                    input_data
                )
            )

            risk_score = (
                self._calculate_risk_score(
                    factors
                )
            )

            risk_score = (
                self._apply_criticality(
                    risk_score,
                    input_data,
                )
            )

            risk_score = (
                self._apply_critical_overrides(
                    risk_score,
                    input_data,
                )
            )

            health_score = (
                (1.0 - risk_score)
                * 100.0
            )

            status = (
                self._classify_status(
                    risk_score,
                    input_data,
                )
            )

            risk_level = (
                self._classify_risk(
                    risk_score
                )
            )

            confidence = (
                self._calculate_confidence(
                    input_data
                )
            )

            actions = (
                self._recommend_actions(
                    input_data,
                    risk_score,
                    status,
                    confidence,
                )
            )

            drivers = (
                self._primary_drivers(
                    factors
                )
            )

            human_review = (
                self._requires_human_review(
                    input_data,
                    status,
                    confidence,
                    actions,
                )
            )

            summary = self._build_summary(
                input_data,
                health_score,
                status,
                risk_level,
                drivers,
            )

            assessment = (
                AssetHealthAssessment(
                    assessment_id=new_id(),

                    asset_id=(
                        input_data.asset_id
                    ),

                    asset_name=(
                        input_data.asset_name
                    ),

                    site_id=(
                        input_data.site_id
                    ),

                    health_score=round(
                        health_score,
                        2,
                    ),

                    risk_score=round(
                        risk_score,
                        4,
                    ),

                    status=status,
                    risk_level=risk_level,

                    confidence=round(
                        confidence,
                        4,
                    ),

                    factors=factors,

                    primary_risk_drivers=(
                        drivers
                    ),

                    recommended_actions=(
                        actions
                    ),

                    anomaly_detected=(
                        bool(
                            input_data.anomaly
                            and input_data
                            .anomaly.detected
                        )
                    ),

                    failure_probability=(
                        input_data.failure
                        .failure_probability
                        if input_data.failure
                        else None
                    ),

                    predicted_failure_mode=(
                        input_data.failure
                        .predicted_failure_mode
                        if input_data.failure
                        else None
                    ),

                    predicted_rul_hours=(
                        input_data.rul
                        .predicted_rul_hours
                        if input_data.rul
                        else None
                    ),

                    defect_count=sum(
                        inspection
                        .defect_detected
                        for inspection
                        in input_data
                        .inspections
                    ),

                    safety_risk_score=(
                        input_data.safety
                        .risk_score
                        if input_data.safety
                        else None
                    ),

                    data_drift_detected=(
                        bool(
                            input_data.drift
                            and input_data
                            .drift
                            .data_drift_detected
                        )
                    ),

                    model_drift_detected=(
                        bool(
                            input_data.drift
                            and input_data
                            .drift
                            .model_drift_detected
                        )
                    ),

                    human_review_required=(
                        human_review
                    ),

                    summary=summary,

                    trace_id=(
                        input_data.trace_id
                    ),

                    metadata={
                        **input_data.metadata,

                        "request_id":
                            input_data
                            .request_id,
                    },
                )
            )

            if span is not None:
                try:
                    span.set_attributes({
                        "asset.health_score":
                            assessment
                            .health_score,

                        "asset.risk_score":
                            assessment
                            .risk_score,

                        "asset.status":
                            assessment
                            .status.value,

                        "asset.action_count":
                            len(actions),
                    })
                except Exception:
                    pass

            await self._post_process(
                assessment
            )

            return assessment

    # -----------------------------------------------------
    # Factors
    # -----------------------------------------------------

    def _build_health_factors(
        self,
        data: AssetHealthInput,
    ) -> list[HealthFactor]:

        factors: list[
            HealthFactor
        ] = []

        # Sensor health
        if data.sensor_signals:
            sensor_risk = max(
                signal.normalized_risk
                for signal
                in data.sensor_signals
            )

            worst_sensor = max(
                data.sensor_signals,
                key=lambda item:
                    item.normalized_risk,
            )

            factors.append(
                HealthFactor(
                    name="sensor_health",

                    risk_score=(
                        sensor_risk
                    ),

                    weight=(
                        self.config
                        .sensor_weight
                    ),

                    contribution=(
                        sensor_risk
                        * self.config
                        .sensor_weight
                    ),

                    explanation=(
                        "Highest sensor risk "
                        f"is {worst_sensor.signal_type.value} "
                        f"at {sensor_risk:.1%}."
                    ),

                    critical=(
                        sensor_risk
                        >= 0.90
                    ),
                )
            )

        # Anomaly
        if data.anomaly is not None:
            risk = (
                data.anomaly.score
                if data.anomaly.detected
                else 0.0
            )

            factors.append(
                HealthFactor(
                    name="anomaly",

                    risk_score=risk,

                    weight=(
                        self.config
                        .anomaly_weight
                    ),

                    contribution=(
                        risk
                        * self.config
                        .anomaly_weight
                    ),

                    explanation=(
                        "Anomaly model risk "
                        f"{risk:.1%}"
                    ),

                    critical=(
                        risk >= 0.90
                    ),
                )
            )

        # Failure
        if data.failure is not None:
            risk = (
                data.failure
                .failure_probability
            )

            factors.append(
                HealthFactor(
                    name="failure_probability",

                    risk_score=risk,

                    weight=(
                        self.config
                        .failure_weight
                    ),

                    contribution=(
                        risk
                        * self.config
                        .failure_weight
                    ),

                    explanation=(
                        "Predicted failure "
                        f"probability {risk:.1%}"
                    ),

                    critical=(
                        risk
                        >= self.config
                        .critical_failure_probability
                    ),

                    metadata={
                        "failure_mode":
                            data.failure
                            .predicted_failure_mode,
                    },
                )
            )

        # RUL
        if data.rul is not None:
            rul = (
                data.rul
                .predicted_rul_hours
            )

            risk = self._rul_to_risk(
                rul
            )

            factors.append(
                HealthFactor(
                    name="remaining_useful_life",

                    risk_score=risk,

                    weight=(
                        self.config
                        .rul_weight
                    ),

                    contribution=(
                        risk
                        * self.config
                        .rul_weight
                    ),

                    explanation=(
                        "Predicted RUL "
                        f"{rul:.1f} hours."
                    ),

                    critical=(
                        rul
                        <= self.config
                        .critical_rul_hours
                    ),
                )
            )

        # CV / inspection
        defects = [
            item
            for item
            in data.inspections
            if item.defect_detected
        ]

        if data.inspections:
            inspection_risk = (
                self._inspection_risk(
                    defects
                )
            )

            factors.append(
                HealthFactor(
                    name="inspection",

                    risk_score=(
                        inspection_risk
                    ),

                    weight=(
                        self.config
                        .inspection_weight
                    ),

                    contribution=(
                        inspection_risk
                        * self.config
                        .inspection_weight
                    ),

                    explanation=(
                        f"{len(defects)} "
                        "inspection defect(s) "
                        "detected."
                    ),

                    critical=any(
                        self._critical_severity(
                            defect.severity
                        )
                        for defect in defects
                    ),
                )
            )

        # Safety
        if data.safety is not None:
            factors.append(
                HealthFactor(
                    name="safety",

                    risk_score=(
                        data.safety
                        .risk_score
                    ),

                    weight=(
                        self.config
                        .safety_weight
                    ),

                    contribution=(
                        data.safety
                        .risk_score
                        * self.config
                        .safety_weight
                    ),

                    explanation=(
                        "Safety risk score "
                        f"{data.safety.risk_score:.1%}."
                    ),

                    critical=(
                        data.safety.emergency
                        or data.safety
                        .risk_score
                        >= self.config
                        .critical_safety_risk
                    ),
                )
            )

        # Maintenance
        if data.maintenance is not None:
            maintenance_risk = (
                self._maintenance_risk(
                    data.maintenance
                )
            )

            factors.append(
                HealthFactor(
                    name="maintenance",

                    risk_score=(
                        maintenance_risk
                    ),

                    weight=(
                        self.config
                        .maintenance_weight
                    ),

                    contribution=(
                        maintenance_risk
                        * self.config
                        .maintenance_weight
                    ),

                    explanation=(
                        "Maintenance risk "
                        f"{maintenance_risk:.1%}."
                    ),

                    critical=False,
                )
            )

        return factors

    # -----------------------------------------------------
    # Risk
    # -----------------------------------------------------

    @staticmethod
    def _calculate_risk_score(
        factors: Sequence[
            HealthFactor
        ],
    ) -> float:

        if not factors:
            return 0.0

        weight_sum = sum(
            factor.weight
            for factor in factors
            if factor.weight > 0
        )

        if weight_sum <= 0:
            return 0.0

        weighted = sum(
            factor.risk_score
            * factor.weight
            for factor in factors
        )

        return _clamp(
            weighted / weight_sum
        )

    @staticmethod
    def _apply_criticality(
        risk: float,
        data: AssetHealthInput,
    ) -> float:

        criticality = (
            0.60
            * data.asset_criticality
            + 0.40
            * data.production_criticality
        )

        multiplier = (
            0.90
            + 0.20
            * criticality
        )

        return _clamp(
            risk * multiplier
        )

    def _apply_critical_overrides(
        self,
        risk: float,
        data: AssetHealthInput,
    ) -> float:

        # Safety emergency has highest
        # priority.
        if (
            data.safety
            and data.safety.emergency
        ):
            return max(
                risk,
                0.98,
            )

        if (
            data.safety
            and data.safety.risk_score
            >= self.config
            .critical_safety_risk
        ):
            risk = max(
                risk,
                0.90,
            )

        if (
            data.failure
            and data.failure
            .failure_probability
            >= self.config
            .critical_failure_probability
        ):
            risk = max(
                risk,
                0.88,
            )

        if (
            data.rul
            and data.rul
            .predicted_rul_hours
            <= self.config
            .critical_rul_hours
        ):
            risk = max(
                risk,
                0.85,
            )

        critical_defect = any(
            inspection.defect_detected
            and self._critical_severity(
                inspection.severity
            )
            for inspection
            in data.inspections
        )

        if critical_defect:
            risk = max(
                risk,
                0.88,
            )

        return _clamp(risk)

    # -----------------------------------------------------
    # RUL
    # -----------------------------------------------------

    def _rul_to_risk(
        self,
        rul_hours: float,
    ) -> float:

        critical = (
            self.config
            .critical_rul_hours
        )

        warning = (
            self.config
            .warning_rul_hours
        )

        if rul_hours <= critical:
            return 1.0

        if rul_hours >= warning:
            return 0.10

        fraction = (
            rul_hours - critical
        ) / max(
            warning - critical,
            1e-9,
        )

        return _clamp(
            1.0 - 0.90 * fraction
        )

    # -----------------------------------------------------
    # Inspection risk
    # -----------------------------------------------------

    @staticmethod
    def _inspection_risk(
        defects: Sequence[
            InspectionSignal
        ],
    ) -> float:

        if not defects:
            return 0.0

        severity_scores = {
            "low": 0.20,
            "minor": 0.20,

            "medium": 0.45,
            "moderate": 0.45,

            "high": 0.75,
            "severe": 0.85,

            "critical": 1.0,
        }

        scores: list[float] = []

        for defect in defects:
            severity = (
                defect.severity
                or "medium"
            ).lower()

            score = (
                severity_scores.get(
                    severity,
                    0.50,
                )
            )

            if (
                defect.confidence
                is not None
            ):
                score *= (
                    0.50
                    + 0.50
                    * defect.confidence
                )

            scores.append(
                _clamp(score)
            )

        return max(scores)

    # -----------------------------------------------------
    # Maintenance risk
    # -----------------------------------------------------

    @staticmethod
    def _maintenance_risk(
        context: MaintenanceContext,
    ) -> float:

        risk = 0.0

        if context.overdue:
            risk += 0.45

        if (
            context.open_work_orders
            > 0
        ):
            risk += min(
                0.25,
                0.05
                * context.open_work_orders,
            )

        if (
            context.repeated_failure_count
            > 0
        ):
            risk += min(
                0.30,
                0.10
                * context
                .repeated_failure_count,
            )

        return _clamp(risk)

    # -----------------------------------------------------
    # Classification
    # -----------------------------------------------------

    def _classify_status(
        self,
        risk: float,
        data: AssetHealthInput,
    ) -> AssetHealthStatus:

        if (
            data.safety
            and data.safety.emergency
        ):
            return (
                AssetHealthStatus.CRITICAL
            )

        if (
            risk
            >= self.config
            .high_risk_max_risk
        ):
            return (
                AssetHealthStatus.CRITICAL
            )

        if (
            risk
            >= self.config
            .degraded_max_risk
        ):
            return (
                AssetHealthStatus.HIGH_RISK
            )

        if (
            risk
            >= self.config
            .watch_max_risk
        ):
            return (
                AssetHealthStatus.DEGRADED
            )

        if (
            risk
            >= self.config
            .healthy_max_risk
        ):
            return (
                AssetHealthStatus.WATCH
            )

        return AssetHealthStatus.HEALTHY

    @staticmethod
    def _classify_risk(
        risk: float,
    ) -> RiskLevel:

        if risk >= 0.80:
            return RiskLevel.CRITICAL

        if risk >= 0.60:
            return RiskLevel.HIGH

        if risk >= 0.30:
            return RiskLevel.MODERATE

        return RiskLevel.LOW

    # -----------------------------------------------------
    # Confidence
    # -----------------------------------------------------

    @staticmethod
    def _calculate_confidence(
        data: AssetHealthInput,
    ) -> float:

        values: list[float] = []

        if (
            data.anomaly
            and data.anomaly.detected
        ):
            values.append(
                max(
                    0.50,
                    data.anomaly.score,
                )
            )

        if (
            data.failure
            and data.failure.confidence
            is not None
        ):
            values.append(
                data.failure.confidence
            )

        if (
            data.rul
            and data.rul.confidence
            is not None
        ):
            values.append(
                data.rul.confidence
            )

        values.extend(
            inspection.confidence
            for inspection
            in data.inspections
            if inspection.confidence
            is not None
        )

        if not values:
            # No model confidence should not be
            # represented as high confidence.
            return 0.50

        return _clamp(
            sum(values) / len(values)
        )

    # -----------------------------------------------------
    # Recommendations
    # -----------------------------------------------------

    def _recommend_actions(
        self,
        data: AssetHealthInput,
        risk: float,
        status: AssetHealthStatus,
        confidence: float,
    ) -> list[RecommendedAction]:

        actions: list[
            RecommendedAction
        ] = []

        # Emergency condition
        if (
            data.safety
            and data.safety.emergency
        ):
            actions.append(
                RecommendedAction(
                    action=(
                        ActionType
                        .EMERGENCY_SHUTDOWN
                    ),

                    priority=(
                        ActionPriority
                        .EMERGENCY
                    ),

                    reason=(
                        "Emergency safety "
                        "condition detected."
                    ),

                    due_within_hours=0.0,

                    requires_human_approval=(
                        self.config
                        .require_human_approval_for_shutdown
                    ),
                )
            )

            return actions

        # Critical failure risk
        if (
            data.failure
            and data.failure
            .failure_probability
            >= self.config
            .critical_failure_probability
        ):
            actions.append(
                RecommendedAction(
                    action=(
                        ActionType
                        .CREATE_WORK_ORDER
                    ),

                    priority=(
                        ActionPriority.CRITICAL
                    ),

                    reason=(
                        "Critical predicted "
                        "failure probability."
                    ),

                    due_within_hours=4.0,
                )
            )

            actions.append(
                RecommendedAction(
                    action=(
                        ActionType
                        .REDUCE_LOAD
                    ),

                    priority=(
                        ActionPriority.HIGH
                    ),

                    reason=(
                        "Reduce operating load "
                        "until inspection."
                    ),

                    due_within_hours=0.0,
                )
            )

        # RUL
        if (
            data.rul
            and data.rul
            .predicted_rul_hours
            <= self.config
            .critical_rul_hours
        ):
            actions.append(
                RecommendedAction(
                    action=(
                        ActionType
                        .SCHEDULE_MAINTENANCE
                    ),

                    priority=(
                        ActionPriority.CRITICAL
                    ),

                    reason=(
                        "Remaining useful life "
                        "is within critical window."
                    ),

                    due_within_hours=min(
                        data.rul
                        .predicted_rul_hours,
                        self.config
                        .critical_rul_hours,
                    ),
                )
            )

        # Defects
        for inspection in (
            data.inspections
        ):
            if not (
                inspection.defect_detected
            ):
                continue

            priority = (
                ActionPriority.CRITICAL
                if self._critical_severity(
                    inspection.severity
                )
                else ActionPriority.HIGH
            )

            actions.append(
                RecommendedAction(
                    action=(
                        ActionType.INSPECT
                    ),

                    priority=priority,

                    component=(
                        inspection.component
                    ),

                    reason=(
                        f"Detected "
                        f"{inspection.defect_type or 'defect'}"
                        + (
                            f" on {inspection.component}"
                            if inspection.component
                            else ""
                        )
                        + "."
                    ),

                    due_within_hours=(
                        4.0
                        if priority
                        == ActionPriority.CRITICAL
                        else 24.0
                    ),
                )
            )

        # Anomaly
        if (
            data.anomaly
            and data.anomaly.detected
            and data.anomaly.score
            >= 0.70
        ):
            actions.append(
                RecommendedAction(
                    action=(
                        ActionType.INSPECT
                    ),

                    priority=(
                        ActionPriority.HIGH
                    ),

                    reason=(
                        "High-confidence abnormal "
                        "equipment behavior detected."
                    ),

                    due_within_hours=24.0,
                )
            )

        # Model drift
        if (
            data.drift
            and data.drift
            .model_drift_detected
        ):
            actions.append(
                RecommendedAction(
                    action=(
                        ActionType
                        .HUMAN_REVIEW
                    ),

                    priority=(
                        ActionPriority.HIGH
                        if data.drift
                        .safety_critical_model_drift
                        else ActionPriority.MEDIUM
                    ),

                    reason=(
                        "Model performance drift "
                        "reduces confidence in "
                        "automated recommendations."
                    ),
                )
            )

            actions.append(
                RecommendedAction(
                    action=(
                        ActionType
                        .RETRAIN_MODEL
                    ),

                    priority=(
                        ActionPriority.MEDIUM
                    ),

                    reason=(
                        "Investigate model drift "
                        "and validate retraining "
                        "requirements."
                    ),
                )
            )

        # Data drift
        if (
            data.drift
            and data.drift
            .data_drift_detected
        ):
            actions.append(
                RecommendedAction(
                    action=(
                        ActionType
                        .CHECK_SENSOR
                    ),

                    priority=(
                        ActionPriority.MEDIUM
                    ),

                    reason=(
                        "Input distribution drift "
                        "detected. Validate sensors "
                        "and operating conditions."
                    ),
                )
            )

        # Low confidence
        if (
            confidence
            < self.config
            .minimum_confidence
            and risk >= 0.40
        ):
            actions.append(
                RecommendedAction(
                    action=(
                        ActionType
                        .HUMAN_REVIEW
                    ),

                    priority=(
                        ActionPriority.HIGH
                    ),

                    reason=(
                        "Risk is elevated but "
                        "prediction confidence is "
                        "below the required level."
                    ),
                )
            )

        # Maintenance
        if (
            data.maintenance
            and data.maintenance.overdue
        ):
            actions.append(
                RecommendedAction(
                    action=(
                        ActionType
                        .SCHEDULE_MAINTENANCE
                    ),

                    priority=(
                        ActionPriority.HIGH
                    ),

                    reason=(
                        "Scheduled maintenance "
                        "is overdue."
                    ),

                    due_within_hours=24.0,
                )
            )

        if not actions:
            if (
                status
                == AssetHealthStatus.HEALTHY
            ):
                actions.append(
                    RecommendedAction(
                        action=(
                            ActionType
                            .CONTINUE_OPERATION
                        ),

                        priority=(
                            ActionPriority.LOW
                        ),

                        reason=(
                            "No significant asset "
                            "health risk detected."
                        ),

                        requires_human_approval=False,
                    )
                )

            else:
                actions.append(
                    RecommendedAction(
                        action=(
                            ActionType.MONITOR
                        ),

                        priority=(
                            ActionPriority.MEDIUM
                        ),

                        reason=(
                            "Continue enhanced "
                            "condition monitoring."
                        ),

                        requires_human_approval=False,
                    )
                )

        return self._deduplicate_actions(
            actions
        )

    # -----------------------------------------------------
    # Human review
    # -----------------------------------------------------

    def _requires_human_review(
        self,
        data: AssetHealthInput,
        status: AssetHealthStatus,
        confidence: float,
        actions: Sequence[
            RecommendedAction
        ],
    ) -> bool:

        if status in {
            AssetHealthStatus.HIGH_RISK,
            AssetHealthStatus.CRITICAL,
        }:
            return True

        if (
            confidence
            < self.config
            .minimum_confidence
        ):
            return True

        if (
            data.drift
            and data.drift
            .safety_critical_model_drift
        ):
            return True

        return any(
            action.action
            in {
                ActionType.SHUTDOWN,
                ActionType.EMERGENCY_SHUTDOWN,
                ActionType.REPLACE_COMPONENT,
            }
            for action in actions
        )

    # -----------------------------------------------------
    # Drivers
    # -----------------------------------------------------

    @staticmethod
    def _primary_drivers(
        factors: Sequence[
            HealthFactor
        ],
        limit: int = 5,
    ) -> list[str]:

        ordered = sorted(
            factors,
            key=lambda factor:
                factor.contribution,
            reverse=True,
        )

        return [
            factor.name
            for factor in ordered[:limit]
            if factor.risk_score > 0
        ]

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    @staticmethod
    def _build_summary(
        data: AssetHealthInput,
        health_score: float,
        status: AssetHealthStatus,
        risk_level: RiskLevel,
        drivers: Sequence[str],
    ) -> str:

        asset = (
            data.asset_name
            or data.asset_id
        )

        driver_text = (
            ", ".join(drivers)
            if drivers
            else "no major risk driver"
        )

        return (
            f"{asset} health score is "
            f"{health_score:.1f}/100 with "
            f"{status.value} status and "
            f"{risk_level.value} risk. "
            f"Primary drivers: {driver_text}."
        )

    # -----------------------------------------------------
    # Post-processing
    # -----------------------------------------------------

    async def _post_process(
        self,
        assessment: AssetHealthAssessment,
    ) -> None:

        self._publish_metrics(
            assessment
        )

        self._log_assessment(
            assessment
        )

        self._publish_alert(
            assessment
        )

        if (
            self.config
            .auto_create_work_order
            and self.work_order_service
            and assessment.status
            in {
                AssetHealthStatus.HIGH_RISK,
                AssetHealthStatus.CRITICAL,
            }
        ):
            await self._create_work_order(
                assessment
            )

    # -----------------------------------------------------
    # Work order
    # -----------------------------------------------------

    async def _create_work_order(
        self,
        assessment: AssetHealthAssessment,
    ) -> None:

        if self.work_order_service is None:
            return

        try:
            await (
                self.work_order_service
                .create_work_order(
                    asset_id=(
                        assessment.asset_id
                    ),

                    title=(
                        "AI Asset Health "
                        "Inspection"
                    ),

                    description=(
                        assessment.summary
                    ),

                    priority=(
                        assessment
                        .risk_level.value
                    ),

                    metadata={
                        "assessment_id":
                            assessment
                            .assessment_id,

                        "health_score":
                            assessment
                            .health_score,

                        "risk_score":
                            assessment
                            .risk_score,

                        "failure_probability":
                            assessment
                            .failure_probability,

                        "predicted_rul_hours":
                            assessment
                            .predicted_rul_hours,
                    },
                )
            )

        except Exception:
            # Agent assessment should not fail
            # because an integration is unavailable.
            return

    # -----------------------------------------------------
    # Observability
    # -----------------------------------------------------

    @staticmethod
    def _publish_metrics(
        assessment: AssetHealthAssessment,
    ) -> None:

        try:
            from app.observability.metrics import (
                registry,
            )

            registry.gauge(
                "asset_health_score"
            ).set(
                assessment.health_score,

                labels={
                    "status":
                        assessment
                        .status.value,
                },
            )

            registry.gauge(
                "asset_risk_score"
            ).set(
                assessment.risk_score,

                labels={
                    "risk_level":
                        assessment
                        .risk_level.value,
                },
            )

            registry.counter(
                "asset_health_assessments_total"
            ).inc(
                labels={
                    "status":
                        assessment
                        .status.value,
                }
            )

        except Exception:
            return

    @staticmethod
    def _log_assessment(
        assessment: AssetHealthAssessment,
    ) -> None:

        try:
            from app.observability.logging import (
                get_logger,
            )

            logger = get_logger(
                "industrial.asset_health_agent"
            )

            payload = {
                "assessment_id":
                    assessment
                    .assessment_id,

                "asset_id":
                    assessment.asset_id,

                "health_score":
                    assessment
                    .health_score,

                "risk_score":
                    assessment
                    .risk_score,

                "status":
                    assessment
                    .status.value,

                "risk_level":
                    assessment
                    .risk_level.value,

                "drivers":
                    assessment
                    .primary_risk_drivers,
            }

            if (
                assessment.status
                == AssetHealthStatus.CRITICAL
            ):
                logger.critical(
                    "Critical asset health "
                    "assessment.",
                    extra={
                        "asset_health":
                            payload
                    },
                )

            elif assessment.status in {
                AssetHealthStatus.HIGH_RISK,
                AssetHealthStatus.DEGRADED,
            }:
                logger.warning(
                    "Asset health degradation "
                    "detected.",
                    extra={
                        "asset_health":
                            payload
                    },
                )

            else:
                logger.info(
                    "Asset health assessment "
                    "completed.",
                    extra={
                        "asset_health":
                            payload
                    },
                )

        except Exception:
            return

    @staticmethod
    def _publish_alert(
        assessment: AssetHealthAssessment,
    ) -> None:

        if assessment.status not in {
            AssetHealthStatus.HIGH_RISK,
            AssetHealthStatus.CRITICAL,
        }:
            return

        try:
            from app.observability.alerting import (
                create_failure_alert,
                alert_manager,
            )

            probability = (
                assessment
                .failure_probability
                or assessment.risk_score
            )

            alert = create_failure_alert(
                asset_id=(
                    assessment.asset_id
                ),

                failure_probability=(
                    probability
                ),

                failure_mode=(
                    assessment
                    .predicted_failure_mode
                    or "asset_health_degradation"
                ),
            )

            emit = getattr(
                alert_manager,
                "emit",
                None,
            )

            if callable(emit):
                result = emit(alert)

                # If alert manager is async,
                # caller integration should await
                # this explicitly. Avoid blocking
                # the health assessment here.
                _ = result

        except Exception:
            return

    # -----------------------------------------------------
    # Trace integration
    # -----------------------------------------------------

    @staticmethod
    def _trace(
        data: AssetHealthInput,
    ) -> Any:

        try:
            from app.observability.tracing import (
                TraceComponent,
                tracer,
            )

            return tracer.start_span(
                "agent.asset_health",

                component=(
                    TraceComponent.AGENT
                ),

                attributes={
                    "agent.name":
                        "asset_health_agent",

                    "industrial.asset_id":
                        data.asset_id,

                    "industrial.site_id":
                        data.site_id,
                },
            )

        except Exception:
            return _NullSpan()

    # -----------------------------------------------------
    # Helpers
    # -----------------------------------------------------

    @staticmethod
    def _critical_severity(
        severity: str | None,
    ) -> bool:

        if not severity:
            return False

        return severity.lower() in {
            "critical",
            "emergency",
            "severe",
        }

    @staticmethod
    def _deduplicate_actions(
        actions: Sequence[
            RecommendedAction
        ],
    ) -> list[RecommendedAction]:

        priority_rank = {
            ActionPriority.LOW: 1,
            ActionPriority.MEDIUM: 2,
            ActionPriority.HIGH: 3,
            ActionPriority.CRITICAL: 4,
            ActionPriority.EMERGENCY: 5,
        }

        result: dict[
            tuple[
                ActionType,
                str | None,
            ],
            RecommendedAction,
        ] = {}

        for action in actions:
            key = (
                action.action,
                action.component,
            )

            existing = (
                result.get(key)
            )

            if (
                existing is None
                or priority_rank[
                    action.priority
                ]
                > priority_rank[
                    existing.priority
                ]
            ):
                result[key] = action

        return list(
            result.values()
        )


# =========================================================
# Null tracing
# =========================================================

class _NullSpan:

    def __enter__(
        self,
    ) -> "_NullSpan":
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> bool:
        return False

    def set_attribute(
        self,
        *_: Any,
        **__: Any,
    ) -> None:
        return None

    def set_attributes(
        self,
        *_: Any,
        **__: Any,
    ) -> None:
        return None


# =========================================================
# Factory
# =========================================================

def create_asset_health_agent(
    *,
    config: AssetHealthAgentConfig | None = None,
    work_order_service: WorkOrderService | None = None,
    knowledge_service: KnowledgeService | None = None,
) -> AssetHealthAgent:

    return AssetHealthAgent(
        config=config,

        work_order_service=(
            work_order_service
        ),

        knowledge_service=(
            knowledge_service
        ),
    )


# =========================================================
# Global agent
# =========================================================

asset_health_agent = (
    AssetHealthAgent()
)


# =========================================================
# Public API
# =========================================================

__all__ = [
    "AssetHealthStatus",
    "RiskLevel",
    "ActionPriority",
    "ActionType",
    "SignalType",

    "SensorHealthSignal",
    "AnomalySignal",
    "FailureSignal",
    "RULSignal",
    "InspectionSignal",
    "SafetySignal",
    "DriftSignal",
    "MaintenanceContext",

    "AssetHealthInput",

    "HealthFactor",
    "RecommendedAction",
    "AssetHealthAssessment",

    "WorkOrderService",
    "KnowledgeService",

    "AssetHealthAgentConfig",
    "AssetHealthAgent",

    "create_asset_health_agent",
    "asset_health_agent",
]