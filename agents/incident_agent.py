
# app/agents/incident_agent.py

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
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

class IncidentType(str, Enum):
    ANOMALY = "anomaly"

    EQUIPMENT_FAILURE = "equipment_failure"
    PREDICTED_FAILURE = "predicted_failure"

    OVERHEATING = "overheating"

    FIRE = "fire"
    SMOKE = "smoke"

    ELECTRICAL = "electrical"
    ARC_FLASH = "arc_flash"

    GAS_LEAK = "gas_leak"
    CHEMICAL_LEAK = "chemical_leak"
    FLUID_LEAK = "fluid_leak"

    PRESSURE = "pressure"

    HYDRAULIC = "hydraulic"
    PNEUMATIC = "pneumatic"

    STRUCTURAL = "structural"

    WORKER_INJURY = "worker_injury"
    WORKER_FALL = "worker_fall"

    UNSAFE_PROXIMITY = "unsafe_proximity"
    RESTRICTED_ZONE = "restricted_zone"
    PPE_VIOLATION = "ppe_violation"

    QUALITY = "quality"
    PRODUCTION = "production"

    CYBER_SECURITY = "cyber_security"

    MULTI_HAZARD = "multi_hazard"

    OTHER = "other"


class IncidentSeverity(str, Enum):
    INFO = "info"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class IncidentPriority(str, Enum):
    P4_LOW = "p4_low"
    P3_MEDIUM = "p3_medium"
    P2_HIGH = "p2_high"
    P1_CRITICAL = "p1_critical"
    P0_EMERGENCY = "p0_emergency"


class IncidentStatus(str, Enum):
    DETECTED = "detected"
    OPEN = "open"

    ACKNOWLEDGED = "acknowledged"

    INVESTIGATING = "investigating"

    RESPONDING = "responding"

    CONTAINED = "contained"

    MONITORING = "monitoring"

    RESOLVED = "resolved"

    CLOSED = "closed"

    ESCALATED = "escalated"

    FALSE_POSITIVE = "false_positive"


class IncidentSource(str, Enum):
    SENSOR = "sensor"
    TELEMETRY = "telemetry"

    ANOMALY_MODEL = "anomaly_model"
    FAILURE_MODEL = "failure_model"
    RUL_MODEL = "rul_model"

    COMPUTER_VISION = "computer_vision"

    SAFETY_ENGINE = "safety_engine"

    FIRE_ENGINE = "fire_engine"
    SMOKE_ENGINE = "smoke_engine"
    OVERHEATING_ENGINE = "overheating_engine"

    MOTION_ENGINE = "motion_engine"

    FAILURE_ANALYSIS_AGENT = (
        "failure_analysis_agent"
    )

    EMERGENCY_RESPONSE_AGENT = (
        "emergency_response_agent"
    )

    OPERATOR = "operator"

    INSPECTION = "inspection"

    PLC = "plc"
    SCADA = "scada"

    OTHER = "other"


class IncidentActionType(str, Enum):
    MONITOR = "monitor"

    VERIFY = "verify"

    ACKNOWLEDGE = "acknowledge"

    INVESTIGATE = "investigate"

    INSPECT_ASSET = "inspect_asset"

    NOTIFY_OPERATOR = "notify_operator"

    NOTIFY_MAINTENANCE = "notify_maintenance"

    NOTIFY_SAFETY_TEAM = "notify_safety_team"

    NOTIFY_EMERGENCY_TEAM = (
        "notify_emergency_team"
    )

    CREATE_WORK_ORDER = "create_work_order"

    RESTRICT_AREA = "restrict_area"

    REQUEST_MACHINE_STOP = (
        "request_machine_stop"
    )

    REQUEST_ENERGY_ISOLATION = (
        "request_energy_isolation"
    )

    EVACUATION_REVIEW = "evacuation_review"

    EMERGENCY_RESPONSE = (
        "emergency_response"
    )

    ROOT_CAUSE_ANALYSIS = (
        "root_cause_analysis"
    )

    PRESERVE_EVIDENCE = (
        "preserve_evidence"
    )


class IncidentActionStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# =========================================================
# Incident signal
# =========================================================

@dataclass(slots=True)
class IncidentSignal:
    signal_id: str

    incident_type: IncidentType

    source: IncidentSource

    source_name: str

    detected: bool

    confidence: float

    severity_score: float

    asset_id: str | None = None
    site_id: str | None = None
    zone_id: str | None = None

    sensor_id: str | None = None
    camera_id: str | None = None

    metric_name: str | None = None

    value: float | None = None
    unit: str | None = None

    message: str | None = None

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
# Context
# =========================================================

@dataclass(slots=True)
class IncidentAssetContext:
    asset_id: str

    asset_name: str | None = None
    asset_type: str | None = None

    site_id: str | None = None
    zone_id: str | None = None

    operating: bool = True

    criticality: float = 0.5

    safety_critical: bool = False

    health_score: float | None = None

    failure_probability: float | None = None

    predicted_rul_hours: float | None = None

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
            and not 0 <= self.health_score <= 100
        ):
            raise ValueError(
                "health_score must be between 0 and 100."
            )


@dataclass(slots=True)
class IncidentWorkerContext:
    workers_present: bool = False

    worker_count: int = 0

    injury_detected: bool = False
    fall_detected: bool = False

    unsafe_proximity: bool = False

    zone_id: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.worker_count < 0:
            raise ValueError(
                "worker_count cannot be negative."
            )


# =========================================================
# Input
# =========================================================

@dataclass(slots=True)
class IncidentAgentInput:
    signals: list[IncidentSignal]

    asset: IncidentAssetContext | None = None

    workers: IncidentWorkerContext | None = None

    existing_incident_id: str | None = None

    correlation_id: str | None = None

    request_id: str | None = None
    trace_id: str | None = None

    data_drift_detected: bool = False
    model_drift_detected: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# =========================================================
# Correlation
# =========================================================

@dataclass(slots=True)
class IncidentCorrelation:
    signal_count: int

    independent_source_count: int

    correlated_types: list[IncidentType]

    multi_source_confirmed: bool

    multi_hazard: bool

    correlation_score: float

    time_span_seconds: float

    explanation: str


# =========================================================
# Risk factors
# =========================================================

@dataclass(slots=True)
class IncidentRiskFactor:
    name: str

    score: float

    weight: float

    contribution: float

    explanation: str

    critical: bool = False


# =========================================================
# Actions
# =========================================================

@dataclass(slots=True)
class IncidentAction:
    action_id: str

    action: IncidentActionType

    priority: IncidentPriority

    reason: str

    status: IncidentActionStatus = (
        IncidentActionStatus.PENDING
    )

    due_within_minutes: float | None = None

    requires_human_approval: bool = True

    asset_id: str | None = None
    site_id: str | None = None
    zone_id: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# =========================================================
# Timeline
# =========================================================

@dataclass(slots=True)
class IncidentTimelineEvent:
    event_id: str

    event_type: str

    description: str

    timestamp: datetime = field(
        default_factory=utc_now
    )

    actor: str = "incident_agent"

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# =========================================================
# Incident result
# =========================================================

@dataclass(slots=True)
class IncidentResult:
    incident_id: str

    incident_number: str

    incident_type: IncidentType

    severity: IncidentSeverity

    priority: IncidentPriority

    status: IncidentStatus

    confidence: float

    risk_score: float

    asset_id: str | None

    asset_name: str | None

    site_id: str | None
    zone_id: str | None

    signals: list[IncidentSignal]

    correlation: IncidentCorrelation

    risk_factors: list[
        IncidentRiskFactor
    ]

    actions: list[IncidentAction]

    timeline: list[
        IncidentTimelineEvent
    ]

    workers_at_risk: bool

    worker_count: int

    emergency_response_required: bool

    shutdown_review_required: bool

    escalation_required: bool

    human_review_required: bool

    model_reliability_warning: bool

    summary: str

    detected_at: datetime

    created_at: datetime = field(
        default_factory=utc_now
    )

    correlation_id: str | None = None

    trace_id: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id":
                self.incident_id,

            "incident_number":
                self.incident_number,

            "incident_type":
                self.incident_type.value,

            "severity":
                self.severity.value,

            "priority":
                self.priority.value,

            "status":
                self.status.value,

            "confidence":
                self.confidence,

            "risk_score":
                self.risk_score,

            "asset_id":
                self.asset_id,

            "asset_name":
                self.asset_name,

            "site_id":
                self.site_id,

            "zone_id":
                self.zone_id,

            "signals": [
                {
                    **asdict(signal),

                    "incident_type":
                        signal
                        .incident_type
                        .value,

                    "source":
                        signal.source.value,

                    "timestamp":
                        signal
                        .timestamp
                        .isoformat(),
                }
                for signal in self.signals
            ],

            "correlation": {
                **asdict(
                    self.correlation
                ),

                "correlated_types": [
                    item.value
                    for item
                    in self.correlation
                    .correlated_types
                ],
            },

            "risk_factors": [
                asdict(item)
                for item
                in self.risk_factors
            ],

            "actions": [
                {
                    **asdict(action),

                    "action":
                        action.action.value,

                    "priority":
                        action.priority.value,

                    "status":
                        action.status.value,
                }
                for action in self.actions
            ],

            "timeline": [
                {
                    **asdict(event),

                    "timestamp":
                        event
                        .timestamp
                        .isoformat(),
                }
                for event in self.timeline
            ],

            "workers_at_risk":
                self.workers_at_risk,

            "worker_count":
                self.worker_count,

            "emergency_response_required":
                self.emergency_response_required,

            "shutdown_review_required":
                self.shutdown_review_required,

            "escalation_required":
                self.escalation_required,

            "human_review_required":
                self.human_review_required,

            "model_reliability_warning":
                self.model_reliability_warning,

            "summary":
                self.summary,

            "detected_at":
                self.detected_at.isoformat(),

            "created_at":
                self.created_at.isoformat(),

            "correlation_id":
                self.correlation_id,

            "trace_id":
                self.trace_id,

            "metadata":
                dict(self.metadata),
        }


# =========================================================
# Repository / integrations
# =========================================================

class IncidentRepository(Protocol):

    async def save_incident(
        self,
        incident: IncidentResult,
    ) -> Any:
        ...

    async def update_status(
        self,
        incident_id: str,
        status: str,
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


class EmergencyResponseService(Protocol):

    async def request_response(
        self,
        *,
        incident_id: str,
        incident_type: str,
        severity: str,
        asset_id: str | None,
        site_id: str | None,
        zone_id: str | None,
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
class IncidentAgentConfig:
    minimum_signal_confidence: float = 0.50

    moderate_threshold: float = 0.40

    high_threshold: float = 0.60

    critical_threshold: float = 0.80

    emergency_threshold: float = 0.95

    correlation_window_seconds: float = 300.0

    minimum_multi_source_count: int = 2

    critical_rul_hours: float = 24.0

    poor_health_threshold: float = 30.0

    minimum_model_confidence: float = 0.60

    auto_create_work_order: bool = False

    auto_request_emergency_response: bool = True

    enable_notifications: bool = True


# =========================================================
# Incident Agent
# =========================================================

class IncidentAgent:

    def __init__(
        self,
        *,
        config: IncidentAgentConfig | None = None,
        repository: IncidentRepository | None = None,
        notification_service: NotificationService | None = None,
        work_order_service: WorkOrderService | None = None,
        emergency_service: EmergencyResponseService | None = None,
    ) -> None:

        self.config = (
            config
            or IncidentAgentConfig()
        )

        self.repository = repository

        self.notification_service = (
            notification_service
        )

        self.work_order_service = (
            work_order_service
        )

        self.emergency_service = (
            emergency_service
        )

    # -----------------------------------------------------
    # Main
    # -----------------------------------------------------

    async def run(
        self,
        data: IncidentAgentInput,
    ) -> IncidentResult:

        signals = self._valid_signals(
            data.signals
        )

        correlation = (
            self._correlate(signals)
        )

        risk_factors = (
            self._risk_factors(
                data,
                signals,
                correlation,
            )
        )

        risk_score = (
            self._risk_score(
                risk_factors
            )
        )

        risk_score = (
            self._apply_critical_overrides(
                risk_score,
                data,
                signals,
            )
        )

        incident_type = (
            self._incident_type(
                signals
            )
        )

        severity = (
            self._severity(
                risk_score
            )
        )

        priority = (
            self._priority(
                severity
            )
        )

        confidence = (
            self._confidence(
                data,
                signals,
                correlation,
            )
        )

        reliability_warning = (
            data.data_drift_detected
            or data.model_drift_detected
        )

        workers_at_risk = (
            self._workers_at_risk(data)
        )

        emergency_required = (
            self._requires_emergency_response(
                incident_type,
                severity,
                workers_at_risk,
            )
        )

        shutdown_review = (
            self._requires_shutdown_review(
                data,
                incident_type,
                severity,
            )
        )

        escalation_required = (
            severity
            in {
                IncidentSeverity.CRITICAL,
                IncidentSeverity.EMERGENCY,
            }
        )

        human_review = (
            reliability_warning
            or severity
            in {
                IncidentSeverity.HIGH,
                IncidentSeverity.CRITICAL,
                IncidentSeverity.EMERGENCY,
            }
        )

        incident_id = (
            data.existing_incident_id
            or new_id()
        )

        actions = self._actions(
            data=data,

            incident_id=incident_id,

            incident_type=incident_type,

            severity=severity,

            priority=priority,

            emergency_required=(
                emergency_required
            ),

            shutdown_review=(
                shutdown_review
            ),

            reliability_warning=(
                reliability_warning
            ),
        )

        detected_at = (
            min(
                (
                    signal.timestamp
                    for signal in signals
                ),
                default=utc_now(),
            )
        )

        timeline = [
            IncidentTimelineEvent(
                event_id=new_id(),

                event_type="incident_detected",

                description=(
                    f"{incident_type.value} "
                    "incident detected."
                ),

                timestamp=detected_at,

                metadata={
                    "signal_count":
                        len(signals),

                    "risk_score":
                        risk_score,
                },
            )
        ]

        if correlation.multi_source_confirmed:
            timeline.append(
                IncidentTimelineEvent(
                    event_id=new_id(),

                    event_type=(
                        "incident_correlated"
                    ),

                    description=(
                        "Incident confirmed by "
                        "multiple independent "
                        "sources."
                    ),
                )
            )

        if escalation_required:
            timeline.append(
                IncidentTimelineEvent(
                    event_id=new_id(),

                    event_type="escalation",

                    description=(
                        "Incident requires "
                        "priority escalation."
                    ),
                )
            )

        summary = self._summary(
            data=data,
            incident_type=incident_type,
            severity=severity,
            risk_score=risk_score,
            correlation=correlation,
        )

        result = IncidentResult(
            incident_id=incident_id,

            incident_number=(
                self._incident_number()
            ),

            incident_type=incident_type,

            severity=severity,

            priority=priority,

            status=IncidentStatus.OPEN,

            confidence=round(
                confidence,
                4,
            ),

            risk_score=round(
                risk_score,
                4,
            ),

            asset_id=(
                data.asset.asset_id
                if data.asset
                else self._asset_id(
                    signals
                )
            ),

            asset_name=(
                data.asset.asset_name
                if data.asset
                else None
            ),

            site_id=(
                data.asset.site_id
                if (
                    data.asset
                    and data.asset.site_id
                )
                else self._site_id(
                    signals
                )
            ),

            zone_id=(
                data.asset.zone_id
                if (
                    data.asset
                    and data.asset.zone_id
                )
                else self._zone_id(
                    signals
                )
            ),

            signals=signals,

            correlation=correlation,

            risk_factors=risk_factors,

            actions=actions,

            timeline=timeline,

            workers_at_risk=(
                workers_at_risk
            ),

            worker_count=(
                data.workers.worker_count
                if data.workers
                else 0
            ),

            emergency_response_required=(
                emergency_required
            ),

            shutdown_review_required=(
                shutdown_review
            ),

            escalation_required=(
                escalation_required
            ),

            human_review_required=(
                human_review
            ),

            model_reliability_warning=(
                reliability_warning
            ),

            summary=summary,

            detected_at=detected_at,

            correlation_id=(
                data.correlation_id
            ),

            trace_id=data.trace_id,

            metadata={
                **data.metadata,

                "request_id":
                    data.request_id,

                "data_drift_detected":
                    data
                    .data_drift_detected,

                "model_drift_detected":
                    data
                    .model_drift_detected,
            },
        )

        await self._post_process(
            result
        )

        return result

    # -----------------------------------------------------
    # Signals
    # -----------------------------------------------------

    def _valid_signals(
        self,
        signals: Sequence[
            IncidentSignal
        ],
    ) -> list[IncidentSignal]:

        return [
            signal
            for signal in signals
            if (
                signal.detected
                and signal.confidence
                >= self.config
                .minimum_signal_confidence
            )
        ]

    # -----------------------------------------------------
    # Correlation
    # -----------------------------------------------------

    def _correlate(
        self,
        signals: Sequence[
            IncidentSignal
        ],
    ) -> IncidentCorrelation:

        if not signals:
            return IncidentCorrelation(
                signal_count=0,

                independent_source_count=0,

                correlated_types=[],

                multi_source_confirmed=False,

                multi_hazard=False,

                correlation_score=0.0,

                time_span_seconds=0.0,

                explanation=(
                    "No valid incident signals."
                ),
            )

        sources = {
            (
                signal.source.value,
                signal.source_name,
            )
            for signal in signals
        }

        types = list({
            signal.incident_type
            for signal in signals
        })

        timestamps = [
            signal.timestamp
            for signal in signals
        ]

        time_span = (
            max(timestamps)
            - min(timestamps)
        ).total_seconds()

        source_score = min(
            len(sources) / 3.0,
            1.0,
        )

        confidence_score = (
            sum(
                signal.confidence
                for signal in signals
            )
            / len(signals)
        )

        temporal_score = (
            1.0
            if time_span
            <= self.config
            .correlation_window_seconds
            else max(
                0.0,
                1.0
                - (
                    time_span
                    - self.config
                    .correlation_window_seconds
                )
                / max(
                    self.config
                    .correlation_window_seconds,
                    1.0,
                )
            )
        )

        correlation_score = (
            0.40 * source_score
            + 0.40 * confidence_score
            + 0.20 * temporal_score
        )

        multi_source = (
            len(sources)
            >= self.config
            .minimum_multi_source_count
        )

        multi_hazard = len(types) > 1

        return IncidentCorrelation(
            signal_count=len(signals),

            independent_source_count=(
                len(sources)
            ),

            correlated_types=types,

            multi_source_confirmed=(
                multi_source
            ),

            multi_hazard=multi_hazard,

            correlation_score=round(
                _clamp(
                    correlation_score
                ),
                4,
            ),

            time_span_seconds=max(
                0.0,
                time_span,
            ),

            explanation=(
                f"{len(signals)} signal(s) "
                f"from {len(sources)} independent "
                "source(s) were correlated."
            ),
        )

    # -----------------------------------------------------
    # Risk factors
    # -----------------------------------------------------

    def _risk_factors(
        self,
        data: IncidentAgentInput,
        signals: Sequence[
            IncidentSignal
        ],
        correlation: IncidentCorrelation,
    ) -> list[IncidentRiskFactor]:

        factors: list[
            IncidentRiskFactor
        ] = []

        for signal in signals:

            score = _clamp(
                (
                    0.60
                    * signal.severity_score
                )
                + (
                    0.40
                    * signal.confidence
                )
            )

            weight = (
                self._incident_weight(
                    signal.incident_type
                )
            )

            factors.append(
                IncidentRiskFactor(
                    name=(
                        signal
                        .incident_type
                        .value
                    ),

                    score=score,

                    weight=weight,

                    contribution=(
                        score * weight
                    ),

                    explanation=(
                        signal.message
                        or (
                            f"Incident signal from "
                            f"{signal.source_name}."
                        )
                    ),

                    critical=(
                        score >= 0.90
                    ),
                )
            )

        if correlation.signal_count > 1:
            factors.append(
                IncidentRiskFactor(
                    name=(
                        "event_correlation"
                    ),

                    score=(
                        correlation
                        .correlation_score
                    ),

                    weight=0.60,

                    contribution=(
                        correlation
                        .correlation_score
                        * 0.60
                    ),

                    explanation=(
                        correlation
                        .explanation
                    ),
                )
            )

        if data.asset is not None:

            factors.append(
                IncidentRiskFactor(
                    name="asset_criticality",

                    score=(
                        data.asset.criticality
                    ),

                    weight=0.70,

                    contribution=(
                        data.asset.criticality
                        * 0.70
                    ),

                    explanation=(
                        "Asset criticality "
                        "contributes to incident "
                        "priority."
                    ),

                    critical=(
                        data.asset
                        .safety_critical
                    ),
                )
            )

            if (
                data.asset
                .failure_probability
                is not None
            ):
                factors.append(
                    IncidentRiskFactor(
                        name=(
                            "failure_probability"
                        ),

                        score=(
                            data.asset
                            .failure_probability
                        ),

                        weight=0.80,

                        contribution=(
                            data.asset
                            .failure_probability
                            * 0.80
                        ),

                        explanation=(
                            "Predicted equipment "
                            "failure risk."
                        ),
                    )
                )

        if self._workers_at_risk(data):
            worker_score = (
                1.0
                if (
                    data.workers
                    and (
                        data.workers
                        .injury_detected
                        or data.workers
                        .fall_detected
                    )
                )
                else 0.80
            )

            factors.append(
                IncidentRiskFactor(
                    name="worker_exposure",

                    score=worker_score,

                    weight=1.40,

                    contribution=(
                        worker_score * 1.40
                    ),

                    explanation=(
                        "Personnel may be exposed "
                        "to the incident."
                    ),

                    critical=(
                        worker_score >= 0.90
                    ),
                )
            )

        return factors

    @staticmethod
    def _incident_weight(
        incident_type: IncidentType,
    ) -> float:

        weights = {
            IncidentType.FIRE: 1.60,

            IncidentType.ARC_FLASH:
                1.60,

            IncidentType.GAS_LEAK:
                1.50,

            IncidentType.CHEMICAL_LEAK:
                1.50,

            IncidentType.WORKER_INJURY:
                1.60,

            IncidentType.WORKER_FALL:
                1.40,

            IncidentType.SMOKE:
                1.30,

            IncidentType.OVERHEATING:
                1.20,

            IncidentType.ELECTRICAL:
                1.30,

            IncidentType.STRUCTURAL:
                1.50,

            IncidentType
            .EQUIPMENT_FAILURE:
                1.10,

            IncidentType
            .PREDICTED_FAILURE:
                0.90,
        }

        return weights.get(
            incident_type,
            1.0,
        )

    @staticmethod
    def _risk_score(
        factors: Sequence[
            IncidentRiskFactor
        ],
    ) -> float:

        if not factors:
            return 0.0

        total_weight = sum(
            factor.weight
            for factor in factors
        )

        if total_weight <= 0:
            return 0.0

        return _clamp(
            sum(
                factor.score
                * factor.weight
                for factor in factors
            )
            / total_weight
        )

    # -----------------------------------------------------
    # Critical overrides
    # -----------------------------------------------------

    def _apply_critical_overrides(
        self,
        score: float,
        data: IncidentAgentInput,
        signals: Sequence[
            IncidentSignal
        ],
    ) -> float:

        critical_types = {
            IncidentType.FIRE,
            IncidentType.ARC_FLASH,
            IncidentType.GAS_LEAK,
            IncidentType.CHEMICAL_LEAK,
            IncidentType.WORKER_INJURY,
        }

        if any(
            signal.incident_type
            in critical_types
            and signal.severity_score >= 0.85
            for signal in signals
        ):
            score = max(
                score,
                0.90,
            )

        if (
            data.workers
            and (
                data.workers.injury_detected
                or data.workers.fall_detected
            )
        ):
            score = max(
                score,
                0.95,
            )

        if (
            data.asset
            and data.asset
            .predicted_rul_hours
            is not None
            and data.asset
            .predicted_rul_hours
            <= self.config
            .critical_rul_hours
            and data.asset
            .failure_probability
            is not None
            and data.asset
            .failure_probability
            >= 0.80
        ):
            score = max(
                score,
                0.85,
            )

        return _clamp(score)

    # -----------------------------------------------------
    # Incident classification
    # -----------------------------------------------------

    @staticmethod
    def _incident_type(
        signals: Sequence[
            IncidentSignal
        ],
    ) -> IncidentType:

        if not signals:
            return IncidentType.OTHER

        types = {
            signal.incident_type
            for signal in signals
        }

        if len(types) > 1:
            return (
                IncidentType.MULTI_HAZARD
            )

        return signals[0].incident_type

    def _severity(
        self,
        score: float,
    ) -> IncidentSeverity:

        if (
            score
            >= self.config
            .emergency_threshold
        ):
            return (
                IncidentSeverity.EMERGENCY
            )

        if (
            score
            >= self.config
            .critical_threshold
        ):
            return (
                IncidentSeverity.CRITICAL
            )

        if (
            score
            >= self.config
            .high_threshold
        ):
            return IncidentSeverity.HIGH

        if (
            score
            >= self.config
            .moderate_threshold
        ):
            return (
                IncidentSeverity.MODERATE
            )

        if score >= 0.20:
            return IncidentSeverity.LOW

        return IncidentSeverity.INFO

    @staticmethod
    def _priority(
        severity: IncidentSeverity,
    ) -> IncidentPriority:

        mapping = {
            IncidentSeverity.INFO:
                IncidentPriority.P4_LOW,

            IncidentSeverity.LOW:
                IncidentPriority.P4_LOW,

            IncidentSeverity.MODERATE:
                IncidentPriority.P3_MEDIUM,

            IncidentSeverity.HIGH:
                IncidentPriority.P2_HIGH,

            IncidentSeverity.CRITICAL:
                IncidentPriority.P1_CRITICAL,

            IncidentSeverity.EMERGENCY:
                IncidentPriority.P0_EMERGENCY,
        }

        return mapping[severity]

    # -----------------------------------------------------
    # Confidence
    # -----------------------------------------------------

    @staticmethod
    def _confidence(
        data: IncidentAgentInput,
        signals: Sequence[
            IncidentSignal
        ],
        correlation: IncidentCorrelation,
    ) -> float:

        if not signals:
            return 0.0

        confidence = (
            sum(
                signal.confidence
                for signal in signals
            )
            / len(signals)
        )

        confidence = (
            0.75 * confidence
            + 0.25
            * correlation
            .correlation_score
        )

        if (
            data.data_drift_detected
            or data.model_drift_detected
        ):
            confidence *= 0.85

        return _clamp(confidence)

    # -----------------------------------------------------
    # Safety decisions
    # -----------------------------------------------------

    @staticmethod
    def _workers_at_risk(
        data: IncidentAgentInput,
    ) -> bool:

        if data.workers is None:
            return False

        return (
            data.workers.workers_present
            or data.workers.worker_count > 0
            or data.workers.injury_detected
            or data.workers.fall_detected
            or data.workers.unsafe_proximity
        )

    @staticmethod
    def _requires_emergency_response(
        incident_type: IncidentType,
        severity: IncidentSeverity,
        workers_at_risk: bool,
    ) -> bool:

        if severity in {
            IncidentSeverity.CRITICAL,
            IncidentSeverity.EMERGENCY,
        }:
            return True

        if incident_type in {
            IncidentType.FIRE,
            IncidentType.ARC_FLASH,
            IncidentType.GAS_LEAK,
            IncidentType.CHEMICAL_LEAK,
            IncidentType.WORKER_INJURY,
        }:
            return True

        return (
            workers_at_risk
            and severity
            == IncidentSeverity.HIGH
        )

    @staticmethod
    def _requires_shutdown_review(
        data: IncidentAgentInput,
        incident_type: IncidentType,
        severity: IncidentSeverity,
    ) -> bool:

        if (
            data.asset is None
            or not data.asset.operating
        ):
            return False

        if severity in {
            IncidentSeverity.CRITICAL,
            IncidentSeverity.EMERGENCY,
        }:
            return True

        return incident_type in {
            IncidentType.FIRE,
            IncidentType.ARC_FLASH,
            IncidentType.ELECTRICAL,
            IncidentType.OVERHEATING,
            IncidentType
            .EQUIPMENT_FAILURE,
        }

    # -----------------------------------------------------
    # Actions
    # -----------------------------------------------------

    def _actions(
        self,
        *,
        data: IncidentAgentInput,
        incident_id: str,
        incident_type: IncidentType,
        severity: IncidentSeverity,
        priority: IncidentPriority,
        emergency_required: bool,
        shutdown_review: bool,
        reliability_warning: bool,
    ) -> list[IncidentAction]:

        asset_id = (
            data.asset.asset_id
            if data.asset
            else None
        )

        site_id = (
            data.asset.site_id
            if data.asset
            else None
        )

        zone_id = (
            data.asset.zone_id
            if data.asset
            else None
        )

        actions: list[
            IncidentAction
        ] = [
            self._action(
                IncidentActionType
                .NOTIFY_OPERATOR,

                priority,

                "Notify the responsible operator "
                "about the incident.",

                asset_id,
                site_id,
                zone_id,

                approval=False,
            ),

            self._action(
                IncidentActionType
                .PRESERVE_EVIDENCE,

                priority,

                "Preserve telemetry, sensor data, "
                "images, model outputs and logs "
                "for investigation.",

                asset_id,
                site_id,
                zone_id,

                approval=False,
            ),
        ]

        if severity in {
            IncidentSeverity.MODERATE,
            IncidentSeverity.HIGH,
            IncidentSeverity.CRITICAL,
            IncidentSeverity.EMERGENCY,
        }:
            actions.append(
                self._action(
                    IncidentActionType
                    .INVESTIGATE,

                    priority,

                    "Investigate the incident and "
                    "verify the suspected cause.",

                    asset_id,
                    site_id,
                    zone_id,
                )
            )

        if incident_type in {
            IncidentType
            .EQUIPMENT_FAILURE,

            IncidentType
            .PREDICTED_FAILURE,

            IncidentType.OVERHEATING,

            IncidentType.ELECTRICAL,

            IncidentType.PRESSURE,
        }:
            actions.extend([
                self._action(
                    IncidentActionType
                    .NOTIFY_MAINTENANCE,

                    priority,

                    "Notify the maintenance team.",

                    asset_id,
                    site_id,
                    zone_id,

                    approval=False,
                ),

                self._action(
                    IncidentActionType
                    .INSPECT_ASSET,

                    priority,

                    "Perform a qualified asset "
                    "inspection.",

                    asset_id,
                    site_id,
                    zone_id,
                ),

                self._action(
                    IncidentActionType
                    .ROOT_CAUSE_ANALYSIS,

                    priority,

                    "Run failure/root-cause "
                    "analysis.",

                    asset_id,
                    site_id,
                    zone_id,
                ),
            ])

        if emergency_required:
            actions.extend([
                self._action(
                    IncidentActionType
                    .NOTIFY_SAFETY_TEAM,

                    priority,

                    "Notify the trained safety "
                    "response team.",

                    asset_id,
                    site_id,
                    zone_id,

                    minutes=0,

                    approval=False,
                ),

                self._action(
                    IncidentActionType
                    .EMERGENCY_RESPONSE,

                    priority,

                    "Escalate the incident to the "
                    "emergency response workflow.",

                    asset_id,
                    site_id,
                    zone_id,

                    minutes=0,
                ),
            ])

        if (
            self._workers_at_risk(data)
            and severity
            in {
                IncidentSeverity.CRITICAL,
                IncidentSeverity.EMERGENCY,
            }
        ):
            actions.extend([
                self._action(
                    IncidentActionType
                    .RESTRICT_AREA,

                    priority,

                    "Restrict access to the "
                    "affected area according to "
                    "site safety procedures.",

                    asset_id,
                    site_id,
                    zone_id,

                    minutes=0,
                ),

                self._action(
                    IncidentActionType
                    .EVACUATION_REVIEW,

                    priority,

                    "Authorized safety personnel "
                    "should assess whether the "
                    "approved evacuation procedure "
                    "must be initiated.",

                    asset_id,
                    site_id,
                    zone_id,

                    minutes=0,
                ),
            ])

        if shutdown_review:
            actions.append(
                self._action(
                    IncidentActionType
                    .REQUEST_MACHINE_STOP,

                    priority,

                    "Request review for a "
                    "controlled or safety-rated "
                    "machine stop through the "
                    "authorized safety system.",

                    asset_id,
                    site_id,
                    zone_id,

                    minutes=0,

                    approval=True,
                )
            )

        if reliability_warning:
            actions.append(
                self._action(
                    IncidentActionType.VERIFY,

                    IncidentPriority.P1_CRITICAL,

                    "Model or data drift was "
                    "detected; verify the incident "
                    "using independent sensors or "
                    "qualified personnel.",

                    asset_id,
                    site_id,
                    zone_id,

                    approval=False,
                )
            )

        if severity in {
            IncidentSeverity.HIGH,
            IncidentSeverity.CRITICAL,
            IncidentSeverity.EMERGENCY,
        }:
            actions.append(
                self._action(
                    IncidentActionType
                    .CREATE_WORK_ORDER,

                    priority,

                    "Create a maintenance/safety "
                    "work order linked to this "
                    "incident.",

                    asset_id,
                    site_id,
                    zone_id,
                )
            )

        return self._deduplicate_actions(
            actions
        )

    @staticmethod
    def _action(
        action: IncidentActionType,
        priority: IncidentPriority,
        reason: str,
        asset_id: str | None,
        site_id: str | None,
        zone_id: str | None,
        *,
        minutes: float | None = None,
        approval: bool = True,
    ) -> IncidentAction:

        return IncidentAction(
            action_id=new_id(),

            action=action,

            priority=priority,

            reason=reason,

            due_within_minutes=minutes,

            requires_human_approval=(
                approval
            ),

            asset_id=asset_id,

            site_id=site_id,

            zone_id=zone_id,
        )

    @staticmethod
    def _deduplicate_actions(
        actions: Sequence[
            IncidentAction
        ],
    ) -> list[IncidentAction]:

        priority_rank = {
            IncidentPriority.P4_LOW: 1,
            IncidentPriority.P3_MEDIUM: 2,
            IncidentPriority.P2_HIGH: 3,
            IncidentPriority.P1_CRITICAL: 4,
            IncidentPriority.P0_EMERGENCY: 5,
        }

        result: dict[
            IncidentActionType,
            IncidentAction,
        ] = {}

        for action in actions:

            current = result.get(
                action.action
            )

            if (
                current is None
                or priority_rank[
                    action.priority
                ]
                > priority_rank[
                    current.priority
                ]
            ):
                result[action.action] = (
                    action
                )

        return list(result.values())

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    @staticmethod
    def _summary(
        *,
        data: IncidentAgentInput,
        incident_type: IncidentType,
        severity: IncidentSeverity,
        risk_score: float,
        correlation: IncidentCorrelation,
    ) -> str:

        asset = (
            data.asset.asset_name
            if (
                data.asset
                and data.asset.asset_name
            )
            else (
                data.asset.asset_id
                if data.asset
                else "industrial site"
            )
        )

        return (
            f"{severity.value} "
            f"{incident_type.value} incident "
            f"identified for {asset}. "
            f"Risk score {risk_score:.1%}. "
            f"{correlation.signal_count} signal(s) "
            f"from "
            f"{correlation.independent_source_count} "
            "independent source(s) were correlated."
        )

    # -----------------------------------------------------
    # Post-processing
    # -----------------------------------------------------

    async def _post_process(
        self,
        result: IncidentResult,
    ) -> None:

        self._record_metrics(result)
        self._log_incident(result)

        if self.repository is not None:
            try:
                await (
                    self.repository
                    .save_incident(result)
                )
            except Exception:
                pass

        if (
            self.config.enable_notifications
            and self.notification_service
            and result.severity
            in {
                IncidentSeverity.HIGH,
                IncidentSeverity.CRITICAL,
                IncidentSeverity.EMERGENCY,
            }
        ):
            await self._notify(result)

        if (
            self.config.auto_create_work_order
            and self.work_order_service
            and result.asset_id
            and result.severity
            in {
                IncidentSeverity.HIGH,
                IncidentSeverity.CRITICAL,
                IncidentSeverity.EMERGENCY,
            }
        ):
            await self._create_work_order(
                result
            )

        if (
            self.config
            .auto_request_emergency_response
            and self.emergency_service
            and result
            .emergency_response_required
        ):
            await (
                self._request_emergency_response(
                    result
                )
            )

    # -----------------------------------------------------
    # Work order
    # -----------------------------------------------------

    async def _create_work_order(
        self,
        result: IncidentResult,
    ) -> None:

        if (
            self.work_order_service is None
            or result.asset_id is None
        ):
            return

        try:
            await (
                self.work_order_service
                .create_work_order(
                    asset_id=(
                        result.asset_id
                    ),

                    title=(
                        f"Incident: "
                        f"{result.incident_type.value}"
                    ),

                    description=(
                        result.summary
                    ),

                    priority=(
                        result.priority.value
                    ),

                    metadata={
                        "incident_id":
                            result.incident_id,

                        "incident_number":
                            result
                            .incident_number,

                        "severity":
                            result
                            .severity.value,

                        "risk_score":
                            result.risk_score,
                    },
                )
            )

        except Exception:
            pass

    # -----------------------------------------------------
    # Emergency response
    # -----------------------------------------------------

    async def _request_emergency_response(
        self,
        result: IncidentResult,
    ) -> None:

        if self.emergency_service is None:
            return

        try:
            await (
                self.emergency_service
                .request_response(
                    incident_id=(
                        result.incident_id
                    ),

                    incident_type=(
                        result
                        .incident_type.value
                    ),

                    severity=(
                        result.severity.value
                    ),

                    asset_id=(
                        result.asset_id
                    ),

                    site_id=(
                        result.site_id
                    ),

                    zone_id=(
                        result.zone_id
                    ),

                    metadata={
                        "risk_score":
                            result.risk_score,

                        "workers_at_risk":
                            result
                            .workers_at_risk,

                        "shutdown_review":
                            result
                            .shutdown_review_required,
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
        result: IncidentResult,
    ) -> None:

        if self.notification_service is None:
            return

        try:
            await (
                self.notification_service
                .notify(
                    title=(
                        "Industrial Incident Alert"
                    ),

                    message=result.summary,

                    severity=(
                        result.severity.value
                    ),

                    metadata={
                        "incident_id":
                            result.incident_id,

                        "incident_number":
                            result
                            .incident_number,

                        "asset_id":
                            result.asset_id,

                        "site_id":
                            result.site_id,
                    },
                )
            )

        except Exception:
            pass

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    @staticmethod
    def _record_metrics(
        result: IncidentResult,
    ) -> None:

        try:
            from app.observability.metrics import (
                registry,
            )

            registry.counter(
                "incident_agent_incidents_total"
            ).inc(
                labels={
                    "type":
                        result
                        .incident_type.value,

                    "severity":
                        result.severity.value,
                }
            )

            registry.histogram(
                "incident_agent_risk_score"
            ).observe(
                result.risk_score,

                labels={
                    "type":
                        result
                        .incident_type.value,
                },
            )

            registry.histogram(
                "incident_agent_signal_count"
            ).observe(
                len(result.signals),

                labels={
                    "severity":
                        result.severity.value,
                },
            )

        except Exception:
            pass

    # -----------------------------------------------------
    # Logging
    # -----------------------------------------------------

    @staticmethod
    def _log_incident(
        result: IncidentResult,
    ) -> None:

        try:
            from app.observability.logging import (
                get_logger,
            )

            logger = get_logger(
                "industrial.incident_agent"
            )

            payload = {
                "incident_id":
                    result.incident_id,

                "incident_number":
                    result.incident_number,

                "type":
                    result.incident_type.value,

                "severity":
                    result.severity.value,

                "priority":
                    result.priority.value,

                "asset_id":
                    result.asset_id,

                "risk_score":
                    result.risk_score,

                "signal_count":
                    len(result.signals),
            }

            if result.severity in {
                IncidentSeverity.CRITICAL,
                IncidentSeverity.EMERGENCY,
            }:
                logger.critical(
                    "Critical industrial incident.",
                    extra={
                        "incident":
                            payload
                    },
                )

            else:
                logger.warning(
                    "Industrial incident created.",
                    extra={
                        "incident":
                            payload
                    },
                )

        except Exception:
            pass

    # -----------------------------------------------------
    # Location helpers
    # -----------------------------------------------------

    @staticmethod
    def _asset_id(
        signals: Sequence[
            IncidentSignal
        ],
    ) -> str | None:

        for signal in signals:
            if signal.asset_id:
                return signal.asset_id

        return None

    @staticmethod
    def _site_id(
        signals: Sequence[
            IncidentSignal
        ],
    ) -> str | None:

        for signal in signals:
            if signal.site_id:
                return signal.site_id

        return None

    @staticmethod
    def _zone_id(
        signals: Sequence[
            IncidentSignal
        ],
    ) -> str | None:

        for signal in signals:
            if signal.zone_id:
                return signal.zone_id

        return None

    @staticmethod
    def _incident_number() -> str:
        now = utc_now()

        return (
            f"INC-"
            f"{now:%Y%m%d}-"
            f"{uuid4().hex[:8].upper()}"
        )


# =========================================================
# Factory
# =========================================================

def create_incident_agent(
    *,
    config: IncidentAgentConfig | None = None,
    repository: IncidentRepository | None = None,
    notification_service: NotificationService | None = None,
    work_order_service: WorkOrderService | None = None,
    emergency_service: EmergencyResponseService | None = None,
) -> IncidentAgent:

    return IncidentAgent(
        config=config,

        repository=repository,

        notification_service=(
            notification_service
        ),

        work_order_service=(
            work_order_service
        ),

        emergency_service=(
            emergency_service
        ),
    )


incident_agent = IncidentAgent()


__all__ = [
    "IncidentType",
    "IncidentSeverity",
    "IncidentPriority",
    "IncidentStatus",
    "IncidentSource",

    "IncidentActionType",
    "IncidentActionStatus",

    "IncidentSignal",

    "IncidentAssetContext",
    "IncidentWorkerContext",

    "IncidentAgentInput",

    "IncidentCorrelation",
    "IncidentRiskFactor",

    "IncidentAction",
    "IncidentTimelineEvent",

    "IncidentResult",

    "IncidentRepository",
    "NotificationService",
    "WorkOrderService",
    "EmergencyResponseService",

    "IncidentAgentConfig",
    "IncidentAgent",

    "create_incident_agent",
    "incident_agent",
]