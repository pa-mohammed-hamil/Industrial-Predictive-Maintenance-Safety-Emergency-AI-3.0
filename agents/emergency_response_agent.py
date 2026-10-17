# app/agents/emergency_response_agent.py

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
    return max(low, min(high, value))


# =========================================================
# Enums
# =========================================================

class EmergencyType(str, Enum):
    FIRE = "fire"
    SMOKE = "smoke"

    OVERHEATING = "overheating"

    ELECTRICAL = "electrical"
    ARC_FLASH = "arc_flash"

    GAS_LEAK = "gas_leak"
    CHEMICAL_LEAK = "chemical_leak"
    FLUID_LEAK = "fluid_leak"

    EXPLOSION_RISK = "explosion_risk"

    PRESSURE = "pressure"
    HYDRAULIC = "hydraulic"

    MACHINE_FAILURE = "machine_failure"

    STRUCTURAL = "structural"

    WORKER_INJURY = "worker_injury"
    WORKER_FALL = "worker_fall"

    UNSAFE_PROXIMITY = "unsafe_proximity"

    RESTRICTED_ZONE = "restricted_zone"

    PPE_VIOLATION = "ppe_violation"

    EVACUATION = "evacuation"

    MULTI_HAZARD = "multi_hazard"

    UNKNOWN = "unknown"


class EmergencySeverity(str, Enum):
    INFO = "info"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    CATASTROPHIC = "catastrophic"


class EmergencyStatus(str, Enum):
    DETECTED = "detected"

    VALIDATING = "validating"

    CONFIRMED = "confirmed"

    RESPONDING = "responding"

    CONTAINED = "contained"

    RESOLVED = "resolved"

    FALSE_ALARM = "false_alarm"

    ESCALATED = "escalated"


class ResponseActionType(str, Enum):
    CONTINUE_MONITORING = "continue_monitoring"

    VERIFY_EVENT = "verify_event"

    HUMAN_REVIEW = "human_review"

    ALERT_OPERATOR = "alert_operator"

    ALERT_SAFETY_TEAM = "alert_safety_team"

    ALERT_MAINTENANCE = "alert_maintenance"

    ALERT_EMERGENCY_TEAM = "alert_emergency_team"

    CREATE_INCIDENT = "create_incident"

    CREATE_WORK_ORDER = "create_work_order"

    RESTRICT_AREA = "restrict_area"

    EVACUATE_ZONE = "evacuate_zone"

    EVACUATE_SITE = "evacuate_site"

    REDUCE_LOAD = "reduce_load"

    STOP_MACHINE = "stop_machine"

    ISOLATE_ASSET = "isolate_asset"

    ISOLATE_ENERGY = "isolate_energy"

    ACTIVATE_ALARM = "activate_alarm"

    REQUEST_FIRE_RESPONSE = "request_fire_response"

    REQUEST_MEDICAL_RESPONSE = "request_medical_response"

    REQUEST_EXTERNAL_EMERGENCY_SERVICES = (
        "request_external_emergency_services"
    )

    INSPECT_ASSET = "inspect_asset"

    PRESERVE_EVIDENCE = "preserve_evidence"


class ResponsePriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    IMMEDIATE = "immediate"


class VerificationState(str, Enum):
    UNVERIFIED = "unverified"
    SINGLE_SOURCE = "single_source"
    MULTI_SOURCE = "multi_source"
    CONFIRMED = "confirmed"


class ResponseExecutionStatus(str, Enum):
    PENDING = "pending"
    RECOMMENDED = "recommended"
    APPROVED = "approved"
    EXECUTED = "executed"
    FAILED = "failed"
    DENIED = "denied"


# =========================================================
# Signals
# =========================================================

@dataclass(slots=True)
class HazardSignal:
    hazard_type: EmergencyType

    detected: bool

    confidence: float

    severity_score: float

    source: str

    sensor_id: str | None = None
    camera_id: str | None = None

    value: float | None = None
    unit: str | None = None

    zone_id: str | None = None

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


@dataclass(slots=True)
class WorkerSafetySignal:
    worker_detected: bool = False

    worker_count: int = 0

    injury_detected: bool = False
    fall_detected: bool = False

    unsafe_proximity: bool = False

    restricted_zone_entry: bool = False

    ppe_violation: bool = False

    confidence: float | None = None

    zone_id: str | None = None

    timestamp: datetime = field(
        default_factory=utc_now
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.worker_count < 0:
            raise ValueError(
                "worker_count cannot be negative."
            )

        if self.confidence is not None:
            self.confidence = _probability(
                self.confidence,
                name="confidence",
            )


@dataclass(slots=True)
class AssetEmergencyContext:
    asset_id: str

    asset_name: str | None = None

    operating: bool = True

    load_percent: float | None = None

    temperature: float | None = None

    failure_probability: float | None = None

    predicted_failure_mode: str | None = None

    predicted_rul_hours: float | None = None

    health_score: float | None = None

    safety_critical: bool = False

    emergency_stop_available: bool = False

    isolation_available: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.failure_probability is not None:
            self.failure_probability = _probability(
                self.failure_probability,
                name="failure_probability",
            )

        if (
            self.health_score is not None
            and not 0.0 <= self.health_score <= 100.0
        ):
            raise ValueError(
                "health_score must be between 0 and 100."
            )


@dataclass(slots=True)
class SiteEmergencyContext:
    site_id: str | None = None
    zone_id: str | None = None

    occupied: bool = False
    worker_count: int = 0

    evacuation_route_available: bool = True

    fire_suppression_available: bool = False

    medical_team_available: bool = False

    emergency_team_available: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class EmergencyResponseInput:
    hazards: list[HazardSignal]

    asset: AssetEmergencyContext | None = None

    workers: WorkerSafetySignal | None = None

    site: SiteEmergencyContext | None = None

    incident_id: str | None = None

    request_id: str | None = None
    trace_id: str | None = None

    data_drift_detected: bool = False
    model_drift_detected: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# =========================================================
# Response models
# =========================================================

@dataclass(slots=True)
class EmergencyFactor:
    name: str

    risk_score: float

    weight: float

    contribution: float

    explanation: str

    critical: bool = False


@dataclass(slots=True)
class ResponseAction:
    action_id: str

    action: ResponseActionType

    priority: ResponsePriority

    reason: str

    execution_status: ResponseExecutionStatus = (
        ResponseExecutionStatus.RECOMMENDED
    )

    due_within_minutes: float | None = None

    zone_id: str | None = None
    asset_id: str | None = None

    requires_human_approval: bool = True

    safety_controller_required: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class EmergencyResponsePlan:
    response_id: str

    incident_id: str

    emergency_type: EmergencyType

    severity: EmergencySeverity

    status: EmergencyStatus

    verification_state: VerificationState

    emergency_score: float

    confidence: float

    factors: list[EmergencyFactor]

    actions: list[ResponseAction]

    affected_asset_id: str | None

    site_id: str | None
    zone_id: str | None

    workers_at_risk: bool
    estimated_workers_at_risk: int

    evacuation_recommended: bool

    machine_stop_recommended: bool

    energy_isolation_recommended: bool

    external_services_recommended: bool

    human_authorization_required: bool

    primary_hazards: list[str]

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

        data["emergency_type"] = (
            self.emergency_type.value
        )

        data["severity"] = (
            self.severity.value
        )

        data["status"] = (
            self.status.value
        )

        data["verification_state"] = (
            self.verification_state.value
        )

        data["created_at"] = (
            self.created_at.isoformat()
        )

        for action in data["actions"]:
            action["action"] = (
                action["action"].value
                if isinstance(
                    action["action"],
                    Enum,
                )
                else action["action"]
            )

            action["priority"] = (
                action["priority"].value
                if isinstance(
                    action["priority"],
                    Enum,
                )
                else action["priority"]
            )

            action["execution_status"] = (
                action["execution_status"].value
                if isinstance(
                    action["execution_status"],
                    Enum,
                )
                else action[
                    "execution_status"
                ]
            )

        return data


# =========================================================
# Integration protocols
# =========================================================

class IncidentService(Protocol):

    async def create_incident(
        self,
        *,
        incident_type: str,
        severity: str,
        asset_id: str | None,
        site_id: str | None,
        zone_id: str | None,
        summary: str,
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
        channel: str,
        title: str,
        message: str,
        severity: str,
        metadata: Mapping[
            str,
            Any,
        ] | None = None,
    ) -> Any:
        ...


class SafetyController(Protocol):
    """
    Deterministic, independently authorized
    safety-controller interface.

    This agent should not directly manipulate
    PLC/relay hardware.
    """

    async def request_machine_stop(
        self,
        *,
        asset_id: str,
        incident_id: str,
        reason: str,
    ) -> Any:
        ...

    async def request_energy_isolation(
        self,
        *,
        asset_id: str,
        incident_id: str,
        reason: str,
    ) -> Any:
        ...


class EmergencyRepository(Protocol):

    async def save_response(
        self,
        response: EmergencyResponsePlan,
    ) -> Any:
        ...


# =========================================================
# Configuration
# =========================================================

@dataclass(slots=True)
class EmergencyResponseConfig:
    minimum_signal_confidence: float = 0.55

    high_risk_threshold: float = 0.60
    critical_risk_threshold: float = 0.80
    catastrophic_risk_threshold: float = 0.95

    fire_critical_score: float = 0.80
    smoke_critical_score: float = 0.85
    overheating_critical_score: float = 0.85

    minimum_confirmation_sources: int = 2

    require_human_authorization: bool = True

    auto_create_incident: bool = True

    enable_notifications: bool = True

    # Never execute physical shutdown from the
    # reasoning agent unless a separately configured
    # safety controller accepts the request.
    allow_safety_controller_requests: bool = False


# =========================================================
# Emergency Response Agent
# =========================================================

class EmergencyResponseAgent:

    def __init__(
        self,
        *,
        config: EmergencyResponseConfig | None = None,
        incident_service: IncidentService | None = None,
        notification_service: NotificationService | None = None,
        safety_controller: SafetyController | None = None,
        repository: EmergencyRepository | None = None,
    ) -> None:

        self.config = (
            config
            or EmergencyResponseConfig()
        )

        self.incident_service = (
            incident_service
        )

        self.notification_service = (
            notification_service
        )

        self.safety_controller = (
            safety_controller
        )

        self.repository = repository

    # -----------------------------------------------------
    # Main execution
    # -----------------------------------------------------

    async def run(
        self,
        data: EmergencyResponseInput,
    ) -> EmergencyResponsePlan:

        with self._trace(data) as span:

            valid_hazards = (
                self._valid_hazards(
                    data.hazards
                )
            )

            verification = (
                self._verification_state(
                    valid_hazards
                )
            )

            factors = (
                self._build_factors(
                    data,
                    valid_hazards,
                )
            )

            emergency_score = (
                self._calculate_risk(
                    factors
                )
            )

            emergency_score = (
                self._apply_overrides(
                    emergency_score,
                    data,
                    valid_hazards,
                )
            )

            severity = (
                self._classify_severity(
                    emergency_score
                )
            )

            emergency_type = (
                self._primary_emergency_type(
                    valid_hazards
                )
            )

            confidence = (
                self._confidence(
                    valid_hazards,
                    data,
                )
            )

            incident_id = (
                data.incident_id
                or new_id()
            )

            actions = (
                self._build_actions(
                    data=data,
                    hazards=valid_hazards,
                    incident_id=incident_id,
                    emergency_type=emergency_type,
                    severity=severity,
                    verification=verification,
                    confidence=confidence,
                )
            )

            workers_at_risk = (
                self._workers_at_risk(
                    data
                )
            )

            worker_count = (
                self._worker_count(data)
            )

            evacuation = any(
                action.action
                in {
                    ResponseActionType
                    .EVACUATE_ZONE,

                    ResponseActionType
                    .EVACUATE_SITE,
                }
                for action in actions
            )

            stop = any(
                action.action
                == ResponseActionType
                .STOP_MACHINE
                for action in actions
            )

            isolation = any(
                action.action
                == ResponseActionType
                .ISOLATE_ENERGY
                for action in actions
            )

            external_services = any(
                action.action
                == ResponseActionType
                .REQUEST_EXTERNAL_EMERGENCY_SERVICES
                for action in actions
            )

            authorization_required = any(
                action
                .requires_human_approval
                for action in actions
            )

            status = (
                EmergencyStatus.CONFIRMED
                if verification
                in {
                    VerificationState
                    .MULTI_SOURCE,

                    VerificationState
                    .CONFIRMED,
                }
                else EmergencyStatus
                .VALIDATING
            )

            summary = self._summary(
                data=data,
                emergency_type=(
                    emergency_type
                ),
                severity=severity,
                score=emergency_score,
                verification=verification,
            )

            plan = EmergencyResponsePlan(
                response_id=new_id(),

                incident_id=incident_id,

                emergency_type=(
                    emergency_type
                ),

                severity=severity,

                status=status,

                verification_state=(
                    verification
                ),

                emergency_score=round(
                    emergency_score,
                    4,
                ),

                confidence=round(
                    confidence,
                    4,
                ),

                factors=factors,

                actions=actions,

                affected_asset_id=(
                    data.asset.asset_id
                    if data.asset
                    else None
                ),

                site_id=(
                    data.site.site_id
                    if data.site
                    else None
                ),

                zone_id=(
                    data.site.zone_id
                    if data.site
                    else self._hazard_zone(
                        valid_hazards
                    )
                ),

                workers_at_risk=(
                    workers_at_risk
                ),

                estimated_workers_at_risk=(
                    worker_count
                ),

                evacuation_recommended=(
                    evacuation
                ),

                machine_stop_recommended=(
                    stop
                ),

                energy_isolation_recommended=(
                    isolation
                ),

                external_services_recommended=(
                    external_services
                ),

                human_authorization_required=(
                    authorization_required
                ),

                primary_hazards=[
                    hazard.hazard_type.value
                    for hazard
                    in valid_hazards
                ],

                summary=summary,

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

            if span is not None:
                try:
                    span.set_attributes({
                        "emergency.type":
                            plan
                            .emergency_type
                            .value,

                        "emergency.severity":
                            plan.severity.value,

                        "emergency.score":
                            plan
                            .emergency_score,

                        "emergency.actions":
                            len(
                                plan.actions
                            ),

                        "emergency.workers_at_risk":
                            plan
                            .workers_at_risk,
                    })
                except Exception:
                    pass

            await self._post_process(
                plan
            )

            return plan

    # -----------------------------------------------------
    # Signal validation
    # -----------------------------------------------------

    def _valid_hazards(
        self,
        hazards: Sequence[
            HazardSignal
        ],
    ) -> list[HazardSignal]:

        return [
            hazard
            for hazard in hazards
            if (
                hazard.detected
                and hazard.confidence
                >= self.config
                .minimum_signal_confidence
            )
        ]

    # -----------------------------------------------------
    # Verification
    # -----------------------------------------------------

    def _verification_state(
        self,
        hazards: Sequence[
            HazardSignal
        ],
    ) -> VerificationState:

        if not hazards:
            return (
                VerificationState
                .UNVERIFIED
            )

        sources = {
            hazard.source
            for hazard in hazards
        }

        if (
            len(sources)
            >= self.config
            .minimum_confirmation_sources
        ):
            return (
                VerificationState
                .CONFIRMED
            )

        if len(hazards) >= 2:
            return (
                VerificationState
                .MULTI_SOURCE
            )

        return (
            VerificationState
            .SINGLE_SOURCE
        )

    # -----------------------------------------------------
    # Risk factors
    # -----------------------------------------------------

    def _build_factors(
        self,
        data: EmergencyResponseInput,
        hazards: Sequence[
            HazardSignal
        ],
    ) -> list[EmergencyFactor]:

        factors: list[
            EmergencyFactor
        ] = []

        for hazard in hazards:

            score = (
                0.60
                * hazard.severity_score
                + 0.40
                * hazard.confidence
            )

            weight = (
                self._hazard_weight(
                    hazard.hazard_type
                )
            )

            critical = (
                self._critical_hazard(
                    hazard
                )
            )

            factors.append(
                EmergencyFactor(
                    name=(
                        hazard
                        .hazard_type
                        .value
                    ),

                    risk_score=(
                        _clamp(score)
                    ),

                    weight=weight,

                    contribution=(
                        score * weight
                    ),

                    explanation=(
                        f"{hazard.hazard_type.value} "
                        f"detected by "
                        f"{hazard.source} "
                        f"with "
                        f"{hazard.confidence:.1%} "
                        f"confidence."
                    ),

                    critical=critical,
                )
            )

        if data.workers is not None:
            worker_risk = (
                self._worker_risk(
                    data.workers
                )
            )

            if worker_risk > 0:
                factors.append(
                    EmergencyFactor(
                        name=(
                            "worker_safety"
                        ),

                        risk_score=(
                            worker_risk
                        ),

                        weight=1.30,

                        contribution=(
                            worker_risk
                            * 1.30
                        ),

                        explanation=(
                            "Worker safety "
                            "exposure detected."
                        ),

                        critical=(
                            data.workers
                            .injury_detected
                            or data.workers
                            .fall_detected
                        ),
                    )
                )

        if data.asset is not None:

            if (
                data.asset
                .failure_probability
                is not None
            ):
                failure_risk = (
                    data.asset
                    .failure_probability
                )

                factors.append(
                    EmergencyFactor(
                        name=(
                            "asset_failure"
                        ),

                        risk_score=(
                            failure_risk
                        ),

                        weight=0.80,

                        contribution=(
                            failure_risk
                            * 0.80
                        ),

                        explanation=(
                            "Asset failure "
                            "probability "
                            f"{failure_risk:.1%}."
                        ),

                        critical=(
                            failure_risk
                            >= 0.90
                        ),
                    )
                )

        return factors

    @staticmethod
    def _hazard_weight(
        hazard_type: EmergencyType,
    ) -> float:

        weights = {
            EmergencyType.FIRE: 1.50,

            EmergencyType.EXPLOSION_RISK:
                1.60,

            EmergencyType.ARC_FLASH:
                1.50,

            EmergencyType.GAS_LEAK:
                1.40,

            EmergencyType.CHEMICAL_LEAK:
                1.40,

            EmergencyType.WORKER_INJURY:
                1.50,

            EmergencyType.WORKER_FALL:
                1.30,

            EmergencyType.SMOKE:
                1.20,

            EmergencyType.OVERHEATING:
                1.10,

            EmergencyType.ELECTRICAL:
                1.20,

            EmergencyType.PRESSURE:
                1.20,

            EmergencyType.MACHINE_FAILURE:
                1.00,
        }

        return weights.get(
            hazard_type,
            1.0,
        )

    def _critical_hazard(
        self,
        hazard: HazardSignal,
    ) -> bool:

        if (
            hazard.hazard_type
            == EmergencyType.FIRE
        ):
            return (
                hazard.severity_score
                >= self.config
                .fire_critical_score
            )

        if (
            hazard.hazard_type
            == EmergencyType.SMOKE
        ):
            return (
                hazard.severity_score
                >= self.config
                .smoke_critical_score
            )

        if (
            hazard.hazard_type
            == EmergencyType.OVERHEATING
        ):
            return (
                hazard.severity_score
                >= self.config
                .overheating_critical_score
            )

        return (
            hazard.severity_score
            >= 0.90
        )

    # -----------------------------------------------------
    # Worker risk
    # -----------------------------------------------------

    @staticmethod
    def _worker_risk(
        workers: WorkerSafetySignal,
    ) -> float:

        risk = 0.0

        if workers.injury_detected:
            risk = max(risk, 1.0)

        if workers.fall_detected:
            risk = max(risk, 0.90)

        if workers.unsafe_proximity:
            risk = max(risk, 0.80)

        if workers.restricted_zone_entry:
            risk = max(risk, 0.70)

        if workers.ppe_violation:
            risk = max(risk, 0.50)

        return risk

    # -----------------------------------------------------
    # Risk calculation
    # -----------------------------------------------------

    @staticmethod
    def _calculate_risk(
        factors: Sequence[
            EmergencyFactor
        ],
    ) -> float:

        if not factors:
            return 0.0

        weights = sum(
            factor.weight
            for factor in factors
        )

        if weights <= 0:
            return 0.0

        score = sum(
            factor.risk_score
            * factor.weight
            for factor in factors
        ) / weights

        return _clamp(score)

    def _apply_overrides(
        self,
        score: float,
        data: EmergencyResponseInput,
        hazards: Sequence[
            HazardSignal
        ],
    ) -> float:

        if any(
            factor
            for factor in hazards
            if (
                factor.hazard_type
                in {
                    EmergencyType.FIRE,
                    EmergencyType.EXPLOSION_RISK,
                    EmergencyType.ARC_FLASH,
                }
                and self._critical_hazard(
                    factor
                )
            )
        ):
            score = max(
                score,
                0.95,
            )

        if (
            data.workers
            and data.workers
            .injury_detected
        ):
            score = max(
                score,
                0.95,
            )

        if (
            data.workers
            and data.workers
            .fall_detected
        ):
            score = max(
                score,
                0.90,
            )

        critical_count = sum(
            self._critical_hazard(
                hazard
            )
            for hazard in hazards
        )

        if critical_count >= 2:
            score = max(
                score,
                0.98,
            )

        return _clamp(score)

    # -----------------------------------------------------
    # Severity
    # -----------------------------------------------------

    def _classify_severity(
        self,
        score: float,
    ) -> EmergencySeverity:

        if (
            score
            >= self.config
            .catastrophic_risk_threshold
        ):
            return (
                EmergencySeverity
                .CATASTROPHIC
            )

        if (
            score
            >= self.config
            .critical_risk_threshold
        ):
            return (
                EmergencySeverity.CRITICAL
            )

        if (
            score
            >= self.config
            .high_risk_threshold
        ):
            return EmergencySeverity.HIGH

        if score >= 0.40:
            return (
                EmergencySeverity.MODERATE
            )

        if score >= 0.20:
            return EmergencySeverity.LOW

        return EmergencySeverity.INFO

    # -----------------------------------------------------
    # Primary type
    # -----------------------------------------------------

    @staticmethod
    def _primary_emergency_type(
        hazards: Sequence[
            HazardSignal
        ],
    ) -> EmergencyType:

        if not hazards:
            return EmergencyType.UNKNOWN

        unique = {
            hazard.hazard_type
            for hazard in hazards
        }

        if len(unique) > 1:
            return (
                EmergencyType.MULTI_HAZARD
            )

        return hazards[0].hazard_type

    # -----------------------------------------------------
    # Confidence
    # -----------------------------------------------------

    @staticmethod
    def _confidence(
        hazards: Sequence[
            HazardSignal
        ],
        data: EmergencyResponseInput,
    ) -> float:

        values = [
            hazard.confidence
            for hazard in hazards
        ]

        if (
            data.workers
            and data.workers.confidence
            is not None
        ):
            values.append(
                data.workers.confidence
            )

        if not values:
            return 0.0

        confidence = (
            sum(values)
            / len(values)
        )

        if (
            data.data_drift_detected
            or data.model_drift_detected
        ):
            confidence *= 0.85

        return _clamp(confidence)

    # -----------------------------------------------------
    # Response actions
    # -----------------------------------------------------

    def _build_actions(
        self,
        *,
        data: EmergencyResponseInput,
        hazards: Sequence[
            HazardSignal
        ],
        incident_id: str,
        emergency_type: EmergencyType,
        severity: EmergencySeverity,
        verification: VerificationState,
        confidence: float,
    ) -> list[ResponseAction]:

        actions: list[
            ResponseAction
        ] = []

        asset_id = (
            data.asset.asset_id
            if data.asset
            else None
        )

        zone_id = (
            data.site.zone_id
            if data.site
            else self._hazard_zone(
                hazards
            )
        )

        # Always establish incident awareness.
        actions.append(
            self._action(
                ResponseActionType
                .ALERT_OPERATOR,

                ResponsePriority.HIGH
                if severity
                in {
                    EmergencySeverity.HIGH,
                    EmergencySeverity.CRITICAL,
                    EmergencySeverity.CATASTROPHIC,
                }
                else ResponsePriority.MEDIUM,

                "Notify the control-room/operator "
                "of the detected emergency.",

                asset_id=asset_id,
                zone_id=zone_id,

                approval=False,
            )
        )

        if (
            verification
            == VerificationState
            .SINGLE_SOURCE
            and severity
            not in {
                EmergencySeverity.CRITICAL,
                EmergencySeverity.CATASTROPHIC,
            }
        ):
            actions.append(
                self._action(
                    ResponseActionType
                    .VERIFY_EVENT,

                    ResponsePriority.HIGH,

                    "Verify the event using an "
                    "independent sensor, camera, "
                    "operator or safety system.",

                    asset_id=asset_id,
                    zone_id=zone_id,

                    approval=False,
                )
            )

        actions.append(
            self._action(
                ResponseActionType
                .CREATE_INCIDENT,

                ResponsePriority.HIGH,

                "Create an auditable emergency "
                "incident record.",

                asset_id=asset_id,
                zone_id=zone_id,

                approval=False,
            )
        )

        hazard_types = {
            hazard.hazard_type
            for hazard in hazards
        }

        # Fire / smoke / explosion.
        if hazard_types & {
            EmergencyType.FIRE,
            EmergencyType.SMOKE,
            EmergencyType.EXPLOSION_RISK,
        }:
            actions.append(
                self._action(
                    ResponseActionType
                    .ACTIVATE_ALARM,

                    ResponsePriority.IMMEDIATE,

                    "Fire/smoke/explosion hazard "
                    "requires immediate alarm and "
                    "safety-team response.",

                    asset_id=asset_id,
                    zone_id=zone_id,

                    minutes=0,
                    approval=False,
                )
            )

            actions.append(
                self._action(
                    ResponseActionType
                    .ALERT_SAFETY_TEAM,

                    ResponsePriority.IMMEDIATE,

                    "Dispatch the trained safety "
                    "response team.",

                    asset_id=asset_id,
                    zone_id=zone_id,

                    minutes=0,
                    approval=False,
                )
            )

            actions.append(
                self._action(
                    ResponseActionType
                    .REQUEST_FIRE_RESPONSE,

                    ResponsePriority.IMMEDIATE,

                    "Initiate the site's approved "
                    "fire emergency procedure.",

                    asset_id=asset_id,
                    zone_id=zone_id,

                    minutes=0,
                    approval=True,
                )
            )

        # Worker injury/fall.
        if (
            data.workers
            and (
                data.workers
                .injury_detected
                or data.workers
                .fall_detected
            )
        ):
            actions.append(
                self._action(
                    ResponseActionType
                    .REQUEST_MEDICAL_RESPONSE,

                    ResponsePriority.IMMEDIATE,

                    "Possible worker injury "
                    "requires trained medical "
                    "response.",

                    asset_id=asset_id,
                    zone_id=zone_id,

                    minutes=0,
                    approval=False,
                )
            )

        # Area restriction.
        if severity in {
            EmergencySeverity.HIGH,
            EmergencySeverity.CRITICAL,
            EmergencySeverity.CATASTROPHIC,
        }:
            actions.append(
                self._action(
                    ResponseActionType
                    .RESTRICT_AREA,

                    ResponsePriority.IMMEDIATE,

                    "Restrict personnel access "
                    "to the affected zone.",

                    asset_id=asset_id,
                    zone_id=zone_id,

                    minutes=0,
                    approval=False,
                )
            )

        # Evacuation recommendation.
        workers_at_risk = (
            self._workers_at_risk(
                data
            )
        )

        if (
            workers_at_risk
            and severity
            in {
                EmergencySeverity.CRITICAL,
                EmergencySeverity.CATASTROPHIC,
            }
        ):
            actions.append(
                self._action(
                    ResponseActionType
                    .EVACUATE_ZONE,

                    ResponsePriority.IMMEDIATE,

                    "Critical hazard may threaten "
                    "personnel in the affected zone; "
                    "follow the site's approved "
                    "evacuation procedure.",

                    asset_id=asset_id,
                    zone_id=zone_id,

                    minutes=0,
                    approval=True,
                )
            )

        # Machine stop/isolation are recommendations
        # requiring an independent safety path.
        if (
            data.asset
            and data.asset.operating
            and severity
            in {
                EmergencySeverity.CRITICAL,
                EmergencySeverity.CATASTROPHIC,
            }
        ):
            actions.append(
                self._action(
                    ResponseActionType
                    .STOP_MACHINE,

                    ResponsePriority.IMMEDIATE,

                    "Critical hazard detected while "
                    "the asset is operating. Request "
                    "a controlled/safety-rated stop "
                    "through the authorized safety "
                    "controller.",

                    asset_id=asset_id,
                    zone_id=zone_id,

                    minutes=0,

                    approval=True,

                    safety_controller=True,
                )
            )

            if (
                data.asset
                .isolation_available
            ):
                actions.append(
                    self._action(
                        ResponseActionType
                        .ISOLATE_ENERGY,

                        ResponsePriority.IMMEDIATE,

                        "Request authorized energy "
                        "isolation according to the "
                        "site's lockout/tagout and "
                        "safety procedure.",

                        asset_id=asset_id,
                        zone_id=zone_id,

                        minutes=0,

                        approval=True,

                        safety_controller=True,
                    )
                )

        if (
            severity
            == EmergencySeverity.CATASTROPHIC
        ):
            actions.append(
                self._action(
                    ResponseActionType
                    .REQUEST_EXTERNAL_EMERGENCY_SERVICES,

                    ResponsePriority.IMMEDIATE,

                    "Escalate according to the "
                    "site emergency plan and local "
                    "emergency-service procedures.",

                    asset_id=asset_id,
                    zone_id=zone_id,

                    minutes=0,

                    approval=True,
                )
            )

        if (
            confidence < 0.60
            or data.model_drift_detected
        ):
            actions.append(
                self._action(
                    ResponseActionType
                    .HUMAN_REVIEW,

                    ResponsePriority.HIGH,

                    "Automated confidence is limited; "
                    "require human/safety-system "
                    "verification.",

                    asset_id=asset_id,
                    zone_id=zone_id,

                    approval=False,
                )
            )

        actions.append(
            self._action(
                ResponseActionType
                .PRESERVE_EVIDENCE,

                ResponsePriority.MEDIUM,

                "Preserve relevant telemetry, "
                "alarms, images, logs and model "
                "outputs for investigation.",

                asset_id=asset_id,
                zone_id=zone_id,

                approval=False,
            )
        )

        return self._deduplicate_actions(
            actions
        )

    # -----------------------------------------------------
    # Action factory
    # -----------------------------------------------------

    @staticmethod
    def _action(
        action: ResponseActionType,
        priority: ResponsePriority,
        reason: str,
        *,
        asset_id: str | None,
        zone_id: str | None,
        minutes: float | None = None,
        approval: bool = True,
        safety_controller: bool = False,
    ) -> ResponseAction:

        return ResponseAction(
            action_id=new_id(),

            action=action,
            priority=priority,

            reason=reason,

            due_within_minutes=minutes,

            asset_id=asset_id,
            zone_id=zone_id,

            requires_human_approval=approval,

            safety_controller_required=(
                safety_controller
            ),
        )

    # -----------------------------------------------------
    # Workers
    # -----------------------------------------------------

    @staticmethod
    def _workers_at_risk(
        data: EmergencyResponseInput,
    ) -> bool:

        if data.workers:
            if (
                data.workers.worker_detected
                or data.workers.worker_count > 0
            ):
                return True

        if data.site:
            return (
                data.site.occupied
                or data.site.worker_count > 0
            )

        return False

    @staticmethod
    def _worker_count(
        data: EmergencyResponseInput,
    ) -> int:

        counts = [0]

        if data.workers:
            counts.append(
                data.workers.worker_count
            )

        if data.site:
            counts.append(
                data.site.worker_count
            )

        return max(counts)

    # -----------------------------------------------------
    # Helpers
    # -----------------------------------------------------

    @staticmethod
    def _hazard_zone(
        hazards: Sequence[
            HazardSignal
        ],
    ) -> str | None:

        for hazard in hazards:
            if hazard.zone_id:
                return hazard.zone_id

        return None

    @staticmethod
    def _deduplicate_actions(
        actions: Sequence[
            ResponseAction
        ],
    ) -> list[ResponseAction]:

        priority_rank = {
            ResponsePriority.LOW: 1,
            ResponsePriority.MEDIUM: 2,
            ResponsePriority.HIGH: 3,
            ResponsePriority.CRITICAL: 4,
            ResponsePriority.IMMEDIATE: 5,
        }

        result: dict[
            tuple[
                ResponseActionType,
                str | None,
                str | None,
            ],
            ResponseAction,
        ] = {}

        for action in actions:

            key = (
                action.action,
                action.asset_id,
                action.zone_id,
            )

            current = result.get(key)

            if (
                current is None
                or priority_rank[
                    action.priority
                ]
                > priority_rank[
                    current.priority
                ]
            ):
                result[key] = action

        return list(
            result.values()
        )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    @staticmethod
    def _summary(
        *,
        data: EmergencyResponseInput,
        emergency_type: EmergencyType,
        severity: EmergencySeverity,
        score: float,
        verification: VerificationState,
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
                else "site"
            )
        )

        return (
            f"{severity.value} "
            f"{emergency_type.value} emergency "
            f"identified for {asset}. "
            f"Emergency risk score "
            f"{score:.1%}; verification state "
            f"{verification.value}. "
            "Execute only approved site safety "
            "and emergency procedures."
        )

    # -----------------------------------------------------
    # Post-processing
    # -----------------------------------------------------

    async def _post_process(
        self,
        plan: EmergencyResponsePlan,
    ) -> None:

        self._publish_metrics(plan)
        self._log_plan(plan)
        self._publish_alert(plan)

        if self.repository is not None:
            try:
                await (
                    self.repository
                    .save_response(plan)
                )
            except Exception:
                pass

        if (
            self.config.auto_create_incident
            and self.incident_service
        ):
            await self._create_incident(
                plan
            )

        if (
            self.config
            .enable_notifications
            and self.notification_service
        ):
            await self._send_notifications(
                plan
            )

        if (
            self.config
            .allow_safety_controller_requests
            and self.safety_controller
        ):
            await (
                self._request_authorized_controls(
                    plan
                )
            )

    # -----------------------------------------------------
    # Incident
    # -----------------------------------------------------

    async def _create_incident(
        self,
        plan: EmergencyResponsePlan,
    ) -> None:

        if self.incident_service is None:
            return

        try:
            await (
                self.incident_service
                .create_incident(
                    incident_type=(
                        plan
                        .emergency_type
                        .value
                    ),

                    severity=(
                        plan.severity.value
                    ),

                    asset_id=(
                        plan
                        .affected_asset_id
                    ),

                    site_id=(
                        plan.site_id
                    ),

                    zone_id=(
                        plan.zone_id
                    ),

                    summary=(
                        plan.summary
                    ),

                    metadata={
                        "response_id":
                            plan.response_id,

                        "emergency_score":
                            plan
                            .emergency_score,

                        "verification":
                            plan
                            .verification_state
                            .value,
                    },
                )
            )

        except Exception:
            return

    # -----------------------------------------------------
    # Notifications
    # -----------------------------------------------------

    async def _send_notifications(
        self,
        plan: EmergencyResponsePlan,
    ) -> None:

        if self.notification_service is None:
            return

        try:
            await (
                self.notification_service
                .notify(
                    channel="emergency",

                    title=(
                        "Industrial Emergency Alert"
                    ),

                    message=(
                        plan.summary
                    ),

                    severity=(
                        plan.severity.value
                    ),

                    metadata={
                        "incident_id":
                            plan.incident_id,

                        "response_id":
                            plan.response_id,

                        "asset_id":
                            plan
                            .affected_asset_id,

                        "site_id":
                            plan.site_id,

                        "zone_id":
                            plan.zone_id,
                    },
                )
            )

        except Exception:
            return

    # -----------------------------------------------------
    # Authorized control requests
    # -----------------------------------------------------

    async def _request_authorized_controls(
        self,
        plan: EmergencyResponsePlan,
    ) -> None:
        """
        Sends requests to a separate safety
        controller.

        The controller itself must enforce:
        - hardware interlocks
        - authorization
        - safe-state rules
        - PLC/relay validation
        - site safety policy
        """

        if (
            self.safety_controller is None
            or not plan.affected_asset_id
        ):
            return

        for action in plan.actions:

            if (
                not action
                .safety_controller_required
            ):
                continue

            # Human authorization remains required.
            # Therefore do not mark as EXECUTED here.
            try:
                if (
                    action.action
                    == ResponseActionType
                    .STOP_MACHINE
                ):
                    await (
                        self.safety_controller
                        .request_machine_stop(
                            asset_id=(
                                plan
                                .affected_asset_id
                            ),

                            incident_id=(
                                plan.incident_id
                            ),

                            reason=(
                                action.reason
                            ),
                        )
                    )

                    action.execution_status = (
                        ResponseExecutionStatus
                        .PENDING
                    )

                elif (
                    action.action
                    == ResponseActionType
                    .ISOLATE_ENERGY
                ):
                    await (
                        self.safety_controller
                        .request_energy_isolation(
                            asset_id=(
                                plan
                                .affected_asset_id
                            ),

                            incident_id=(
                                plan.incident_id
                            ),

                            reason=(
                                action.reason
                            ),
                        )
                    )

                    action.execution_status = (
                        ResponseExecutionStatus
                        .PENDING
                    )

            except Exception:
                action.execution_status = (
                    ResponseExecutionStatus
                    .FAILED
                )

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    @staticmethod
    def _publish_metrics(
        plan: EmergencyResponsePlan,
    ) -> None:

        try:
            from app.observability.metrics import (
                registry,
            )

            registry.counter(
                "emergency_responses_total"
            ).inc(
                labels={
                    "type":
                        plan
                        .emergency_type
                        .value,

                    "severity":
                        plan
                        .severity
                        .value,
                }
            )

            registry.gauge(
                "emergency_risk_score"
            ).set(
                plan.emergency_score,

                labels={
                    "type":
                        plan
                        .emergency_type
                        .value,
                },
            )

            if plan.workers_at_risk:
                registry.gauge(
                    "emergency_workers_at_risk"
                ).set(
                    plan
                    .estimated_workers_at_risk
                )

        except Exception:
            return

    # -----------------------------------------------------
    # Logging
    # -----------------------------------------------------

    @staticmethod
    def _log_plan(
        plan: EmergencyResponsePlan,
    ) -> None:

        try:
            from app.observability.logging import (
                get_logger,
            )

            logger = get_logger(
                "industrial.emergency_response_agent"
            )

            payload = {
                "response_id":
                    plan.response_id,

                "incident_id":
                    plan.incident_id,

                "asset_id":
                    plan
                    .affected_asset_id,

                "emergency_type":
                    plan
                    .emergency_type
                    .value,

                "severity":
                    plan.severity.value,

                "risk_score":
                    plan
                    .emergency_score,

                "workers_at_risk":
                    plan.workers_at_risk,

                "actions":
                    len(plan.actions),
            }

            if plan.severity in {
                EmergencySeverity.CRITICAL,
                EmergencySeverity.CATASTROPHIC,
            }:
                logger.critical(
                    "Critical industrial "
                    "emergency response generated.",
                    extra={
                        "emergency":
                            payload
                    },
                )

            else:
                logger.warning(
                    "Industrial emergency "
                    "response generated.",
                    extra={
                        "emergency":
                            payload
                    },
                )

        except Exception:
            return

    # -----------------------------------------------------
    # Alerting
    # -----------------------------------------------------

    @staticmethod
    def _publish_alert(
        plan: EmergencyResponsePlan,
    ) -> None:

        try:
            from app.observability.alerting import (
                create_safety_alert,
                alert_manager,
            )

            if not plan.affected_asset_id:
                return

            alert = create_safety_alert(
                asset_id=(
                    plan
                    .affected_asset_id
                ),

                hazard_type=(
                    plan
                    .emergency_type
                    .value
                ),

                risk_score=(
                    plan
                    .emergency_score
                ),
            )

            emit = getattr(
                alert_manager,
                "emit",
                None,
            )

            if callable(emit):
                emit(alert)

        except Exception:
            return

    # -----------------------------------------------------
    # Tracing
    # -----------------------------------------------------

    @staticmethod
    def _trace(
        data: EmergencyResponseInput,
    ) -> Any:

        try:
            from app.observability.tracing import (
                TraceComponent,
                tracer,
            )

            asset_id = (
                data.asset.asset_id
                if data.asset
                else None
            )

            return tracer.start_span(
                "agent.emergency_response",

                component=(
                    TraceComponent
                    .EMERGENCY
                ),

                attributes={
                    "agent.name":
                        "emergency_response_agent",

                    "industrial.asset_id":
                        asset_id,

                    "emergency.signal_count":
                        len(
                            data.hazards
                        ),
                },
            )

        except Exception:
            return _NullSpan()


# =========================================================
# Null span
# =========================================================

class _NullSpan:

    def __enter__(self) -> "_NullSpan":
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> bool:
        return False

    def set_attributes(
        self,
        *_: Any,
        **__: Any,
    ) -> None:
        return None


# =========================================================
# Factory
# =========================================================

def create_emergency_response_agent(
    *,
    config: EmergencyResponseConfig | None = None,
    incident_service: IncidentService | None = None,
    notification_service: NotificationService | None = None,
    safety_controller: SafetyController | None = None,
    repository: EmergencyRepository | None = None,
) -> EmergencyResponseAgent:

    return EmergencyResponseAgent(
        config=config,

        incident_service=(
            incident_service
        ),

        notification_service=(
            notification_service
        ),

        safety_controller=(
            safety_controller
        ),

        repository=repository,
    )


emergency_response_agent = (
    EmergencyResponseAgent()
)


__all__ = [
    "EmergencyType",
    "EmergencySeverity",
    "EmergencyStatus",

    "ResponseActionType",
    "ResponsePriority",
    "VerificationState",
    "ResponseExecutionStatus",

    "HazardSignal",
    "WorkerSafetySignal",

    "AssetEmergencyContext",
    "SiteEmergencyContext",

    "EmergencyResponseInput",

    "EmergencyFactor",
    "ResponseAction",
    "EmergencyResponsePlan",

    "IncidentService",
    "NotificationService",
    "SafetyController",
    "EmergencyRepository",

    "EmergencyResponseConfig",
    "EmergencyResponseAgent",

    "create_emergency_response_agent",
    "emergency_response_agent",
]