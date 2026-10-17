# app/agents/tools/maintenance_tools.py

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence
from uuid import UUID, uuid4


# =========================================================
# Utilities
# =========================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid4())


def clamp(
    value: float,
    low: float = 0.0,
    high: float = 1.0,
) -> float:
    return max(low, min(high, float(value)))


def finite(
    value: Any,
    *,
    default: float = 0.0,
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    if not math.isfinite(result):
        return default

    return result


def non_negative(
    value: Any,
    *,
    default: float = 0.0,
) -> float:
    return max(
        0.0,
        finite(value, default=default),
    )


def probability(
    value: Any,
    *,
    default: float = 0.0,
) -> float:
    return clamp(
        finite(value, default=default)
    )


def ensure_utc(
    value: datetime | None,
) -> datetime | None:
    if value is None:
        return None

    if value.tzinfo is None:
        return value.replace(
            tzinfo=timezone.utc
        )

    return value.astimezone(timezone.utc)


def json_safe(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, UUID):
        return str(value)

    if is_dataclass(value):
        return json_safe(asdict(value))

    if hasattr(value, "model_dump"):
        try:
            return json_safe(
                value.model_dump(mode="json")
            )
        except Exception:
            return json_safe(
                value.model_dump()
            )

    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            json_safe(item)
            for item in value
        ]

    if (
        isinstance(value, float)
        and not math.isfinite(value)
    ):
        return None

    return value


# =========================================================
# Enums
# =========================================================

class MaintenanceStrategy(str, Enum):
    REACTIVE = "reactive"
    CORRECTIVE = "corrective"
    PREVENTIVE = "preventive"
    PREDICTIVE = "predictive"
    CONDITION_BASED = "condition_based"
    INSPECTION = "inspection"
    OVERHAUL = "overhaul"
    REPLACEMENT = "replacement"
    EMERGENCY = "emergency"


class MaintenancePriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    IMMEDIATE = "immediate"


class MaintenanceAction(str, Enum):
    MONITOR = "monitor"
    INSPECT = "inspect"

    CLEAN = "clean"
    LUBRICATE = "lubricate"
    ALIGN = "align"
    BALANCE = "balance"
    TIGHTEN = "tighten"
    CALIBRATE = "calibrate"

    CHECK_BEARING = "check_bearing"
    CHECK_GEARBOX = "check_gearbox"
    CHECK_MOTOR = "check_motor"
    CHECK_PUMP = "check_pump"
    CHECK_COOLING = "check_cooling"
    CHECK_LUBRICATION = "check_lubrication"
    CHECK_ELECTRICAL = "check_electrical"
    CHECK_HYDRAULIC = "check_hydraulic"
    CHECK_PNEUMATIC = "check_pneumatic"
    CHECK_SEALS = "check_seals"

    LEAK_INSPECTION = "leak_inspection"
    VIBRATION_ANALYSIS = "vibration_analysis"
    THERMAL_INSPECTION = "thermal_inspection"

    REPAIR = "repair"
    REPLACE_COMPONENT = "replace_component"
    OVERHAUL = "overhaul"

    SAFETY_REVIEW = "safety_review"
    REQUEST_MACHINE_STOP = "request_machine_stop"

    CREATE_WORK_ORDER = "create_work_order"
    POST_MAINTENANCE_VERIFY = "post_maintenance_verify"


class MaintenanceReason(str, Enum):
    SCHEDULED = "scheduled"
    OVERDUE = "overdue"

    LOW_HEALTH = "low_health"
    FAILURE_RISK = "failure_risk"
    LOW_RUL = "low_rul"
    ANOMALY = "anomaly"
    DEGRADATION = "degradation"

    TEMPERATURE = "temperature"
    VIBRATION = "vibration"
    PRESSURE = "pressure"
    CURRENT = "current"
    ENERGY = "energy"

    INSPECTION_FINDING = "inspection_finding"
    INCIDENT = "incident"
    SAFETY = "safety"

    REPEAT_FAILURE = "repeat_failure"

    OTHER = "other"


class MaintenanceReadiness(str, Enum):
    READY = "ready"
    PARTIALLY_READY = "partially_ready"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"


class DataCoverage(str, Enum):
    COMPLETE = "complete"
    GOOD = "good"
    LIMITED = "limited"
    INSUFFICIENT = "insufficient"


# =========================================================
# Evidence
# =========================================================

@dataclass(slots=True)
class MaintenanceEvidence:
    reason: MaintenanceReason

    description: str

    severity: float
    confidence: float

    source: str

    component: str | None = None

    value: float | None = None
    unit: str | None = None

    timestamp: datetime | None = None

    critical: bool = False

    provenance_id: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.severity = probability(
            self.severity
        )
        self.confidence = probability(
            self.confidence
        )
        self.timestamp = ensure_utc(
            self.timestamp
        )


# =========================================================
# Asset maintenance context
# =========================================================

@dataclass(slots=True)
class MaintenanceAssetContext:
    asset_id: str

    asset_name: str | None = None
    asset_type: str | None = None

    site_id: str | None = None
    zone_id: str | None = None

    safety_critical: bool = False
    production_critical: bool = False

    criticality_score: float = 0.5

    operating: bool | None = None

    operating_hours: float | None = None

    health_score: float | None = None

    failure_probability: float | None = None

    anomaly_score: float | None = None

    degradation_score: float | None = None

    rul_hours: float | None = None

    last_maintenance_at: datetime | None = None
    next_maintenance_at: datetime | None = None

    repeat_failure_count: int = 0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.criticality_score = probability(
            self.criticality_score
        )

        if self.health_score is not None:
            self.health_score = max(
                0.0,
                min(
                    100.0,
                    finite(self.health_score),
                ),
            )

        if self.failure_probability is not None:
            self.failure_probability = (
                probability(
                    self.failure_probability
                )
            )

        if self.anomaly_score is not None:
            self.anomaly_score = probability(
                self.anomaly_score
            )

        if self.degradation_score is not None:
            self.degradation_score = (
                probability(
                    self.degradation_score
                )
            )

        if self.rul_hours is not None:
            self.rul_hours = non_negative(
                self.rul_hours
            )

        self.last_maintenance_at = (
            ensure_utc(
                self.last_maintenance_at
            )
        )

        self.next_maintenance_at = (
            ensure_utc(
                self.next_maintenance_at
            )
        )

        self.repeat_failure_count = max(
            0,
            int(self.repeat_failure_count),
        )


# =========================================================
# Resource constraints
# =========================================================

@dataclass(slots=True)
class MaintenanceResources:
    technician_available: bool | None = None

    required_skill: str | None = None
    technician_skills: list[str] = field(
        default_factory=list
    )

    required_parts: list[str] = field(
        default_factory=list
    )

    available_parts: list[str] = field(
        default_factory=list
    )

    required_tools: list[str] = field(
        default_factory=list
    )

    available_tools: list[str] = field(
        default_factory=list
    )

    production_window_available: bool | None = None

    estimated_duration_hours: float | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# =========================================================
# Readiness
# =========================================================

@dataclass(slots=True)
class MaintenanceReadinessResult:
    readiness: MaintenanceReadiness

    ready: bool

    missing_parts: list[str]
    missing_tools: list[str]

    missing_skill: bool

    technician_available: bool | None
    production_window_available: bool | None

    warnings: list[str]

    score: float

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# =========================================================
# Risk
# =========================================================

@dataclass(slots=True)
class MaintenanceRiskResult:
    physical_risk_score: float

    business_priority_score: float

    priority: MaintenancePriority

    coverage: DataCoverage

    confidence: float

    critical_reasons: list[str]

    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# =========================================================
# Maintenance window
# =========================================================

@dataclass(slots=True)
class MaintenanceWindow:
    recommended_start: datetime

    deadline: datetime

    maximum_delay_hours: float

    based_on_rul: bool

    based_on_schedule: bool

    emergency: bool

    reason: str

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# =========================================================
# Action item
# =========================================================

@dataclass(slots=True)
class MaintenanceActionItem:
    action_id: str

    action: MaintenanceAction

    reason: MaintenanceReason

    description: str

    priority: MaintenancePriority

    component: str | None = None

    due_at: datetime | None = None

    estimated_duration_hours: float | None = None

    requires_shutdown: bool = False
    requires_human_approval: bool = False

    completed: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# =========================================================
# Work order
# =========================================================

@dataclass(slots=True)
class WorkOrderRequest:
    idempotency_key: str

    asset_id: str

    title: str
    description: str

    priority: MaintenancePriority

    strategy: MaintenanceStrategy

    actions: list[MaintenanceActionItem]

    scheduled_start: datetime | None = None
    deadline: datetime | None = None

    estimated_duration_hours: float | None = None

    required_parts: list[str] = field(
        default_factory=list
    )

    required_tools: list[str] = field(
        default_factory=list
    )

    required_skill: str | None = None

    source_agent: str = "maintenance_agent"

    request_id: str | None = None
    trace_id: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# =========================================================
# Maintenance plan
# =========================================================

@dataclass(slots=True)
class MaintenancePlan:
    plan_id: str

    asset_id: str

    strategy: MaintenanceStrategy
    priority: MaintenancePriority

    risk: MaintenanceRiskResult

    window: MaintenanceWindow

    readiness: MaintenanceReadinessResult

    actions: list[MaintenanceActionItem]

    evidence: list[MaintenanceEvidence]

    maintenance_required: bool

    machine_stop_review_required: bool

    human_review_required: bool

    work_order: WorkOrderRequest | None

    generated_at: datetime = field(
        default_factory=utc_now
    )

    warnings: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# =========================================================
# Configuration
# =========================================================

@dataclass(slots=True)
class MaintenanceToolsConfig:
    health_warning: float = 60.0
    health_critical: float = 30.0

    failure_warning: float = 0.50
    failure_high: float = 0.70
    failure_critical: float = 0.90

    anomaly_warning: float = 0.60
    anomaly_critical: float = 0.90

    degradation_warning: float = 0.60
    degradation_critical: float = 0.90

    rul_monitor_hours: float = 720.0
    rul_warning_hours: float = 168.0
    rul_urgent_hours: float = 72.0
    rul_critical_hours: float = 24.0

    repeat_failure_high: int = 2
    repeat_failure_critical: int = 4

    high_risk_threshold: float = 0.60
    urgent_risk_threshold: float = 0.75
    critical_risk_threshold: float = 0.90

    evidence_stale_hours: float = 24.0

    minimum_confidence: float = 0.50

    immediate_delay_hours: float = 1.0
    urgent_delay_hours: float = 24.0
    high_delay_hours: float = 72.0
    normal_delay_hours: float = 168.0
    low_delay_hours: float = 720.0


# =========================================================
# Optional external service contracts
# =========================================================

class WorkOrderService(Protocol):

    async def create_work_order(
        self,
        request: WorkOrderRequest,
    ) -> Any:
        ...


class MaintenanceRepository(Protocol):

    async def save_plan(
        self,
        plan: MaintenancePlan,
    ) -> Any:
        ...


# =========================================================
# Maintenance Tools
# =========================================================

class MaintenanceTools:

    def __init__(
        self,
        *,
        config: MaintenanceToolsConfig | None = None,
    ) -> None:
        self.config = (
            config
            or MaintenanceToolsConfig()
        )

    # =====================================================
    # Evidence extraction
    # =====================================================

    def build_evidence(
        self,
        asset: MaintenanceAssetContext,
        *,
        external_evidence: Sequence[
            MaintenanceEvidence
        ] = (),
    ) -> list[MaintenanceEvidence]:

        evidence: list[MaintenanceEvidence] = []

        if asset.health_score is not None:
            if (
                asset.health_score
                <= self.config.health_critical
            ):
                evidence.append(
                    MaintenanceEvidence(
                        reason=(
                            MaintenanceReason
                            .LOW_HEALTH
                        ),
                        description=(
                            "Asset health score is "
                            "critically low."
                        ),
                        severity=0.95,
                        confidence=0.90,
                        source="asset_health",
                        value=asset.health_score,
                        unit="score",
                        critical=True,
                    )
                )

            elif (
                asset.health_score
                <= self.config.health_warning
            ):
                evidence.append(
                    MaintenanceEvidence(
                        reason=(
                            MaintenanceReason
                            .LOW_HEALTH
                        ),
                        description=(
                            "Asset health score is "
                            "below the maintenance "
                            "warning threshold."
                        ),
                        severity=0.65,
                        confidence=0.85,
                        source="asset_health",
                        value=asset.health_score,
                        unit="score",
                    )
                )

        if asset.failure_probability is not None:
            p = asset.failure_probability

            if p >= self.config.failure_critical:
                severity = 1.0
                critical = True

            elif p >= self.config.failure_high:
                severity = 0.85
                critical = False

            elif p >= self.config.failure_warning:
                severity = 0.65
                critical = False

            else:
                severity = 0.0
                critical = False

            if severity:
                evidence.append(
                    MaintenanceEvidence(
                        reason=(
                            MaintenanceReason
                            .FAILURE_RISK
                        ),
                        description=(
                            "Predicted failure "
                            "probability exceeds the "
                            "maintenance threshold."
                        ),
                        severity=severity,
                        confidence=0.90,
                        source="failure_prediction",
                        value=p,
                        unit="probability",
                        critical=critical,
                    )
                )

        if asset.rul_hours is not None:
            rul = asset.rul_hours

            if rul <= self.config.rul_critical_hours:
                severity = 1.0
                critical = True

            elif rul <= self.config.rul_urgent_hours:
                severity = 0.90
                critical = False

            elif rul <= self.config.rul_warning_hours:
                severity = 0.75
                critical = False

            elif rul <= self.config.rul_monitor_hours:
                severity = 0.45
                critical = False

            else:
                severity = 0.0
                critical = False

            if severity:
                evidence.append(
                    MaintenanceEvidence(
                        reason=MaintenanceReason.LOW_RUL,
                        description=(
                            "Predicted remaining useful "
                            "life indicates maintenance "
                            "planning is required."
                        ),
                        severity=severity,
                        confidence=0.90,
                        source="rul_prediction",
                        value=rul,
                        unit="hours",
                        critical=critical,
                    )
                )

        if asset.anomaly_score is not None:
            if (
                asset.anomaly_score
                >= self.config.anomaly_critical
            ):
                severity = 0.95
                critical = True

            elif (
                asset.anomaly_score
                >= self.config.anomaly_warning
            ):
                severity = 0.65
                critical = False

            else:
                severity = 0.0
                critical = False

            if severity:
                evidence.append(
                    MaintenanceEvidence(
                        reason=MaintenanceReason.ANOMALY,
                        description=(
                            "Anomaly score exceeds the "
                            "configured maintenance "
                            "threshold."
                        ),
                        severity=severity,
                        confidence=0.80,
                        source="anomaly_detection",
                        value=asset.anomaly_score,
                        unit="score",
                        critical=critical,
                    )
                )

        if asset.degradation_score is not None:
            if (
                asset.degradation_score
                >= self.config.degradation_critical
            ):
                severity = 0.95
                critical = True

            elif (
                asset.degradation_score
                >= self.config.degradation_warning
            ):
                severity = 0.65
                critical = False

            else:
                severity = 0.0
                critical = False

            if severity:
                evidence.append(
                    MaintenanceEvidence(
                        reason=(
                            MaintenanceReason
                            .DEGRADATION
                        ),
                        description=(
                            "Asset degradation exceeds "
                            "the maintenance threshold."
                        ),
                        severity=severity,
                        confidence=0.80,
                        source="degradation_model",
                        value=asset.degradation_score,
                        unit="score",
                        critical=critical,
                    )
                )

        if (
            asset.repeat_failure_count
            >= self.config.repeat_failure_high
        ):
            critical = (
                asset.repeat_failure_count
                >= self.config
                .repeat_failure_critical
            )

            evidence.append(
                MaintenanceEvidence(
                    reason=(
                        MaintenanceReason
                        .REPEAT_FAILURE
                    ),
                    description=(
                        "Repeated failures indicate "
                        "possible unresolved root cause."
                    ),
                    severity=(
                        0.95
                        if critical
                        else 0.70
                    ),
                    confidence=0.90,
                    source="maintenance_history",
                    value=float(
                        asset.repeat_failure_count
                    ),
                    unit="failures",
                    critical=critical,
                )
            )

        now = utc_now()

        if asset.next_maintenance_at:
            if asset.next_maintenance_at <= now:
                evidence.append(
                    MaintenanceEvidence(
                        reason=(
                            MaintenanceReason.OVERDUE
                        ),
                        description=(
                            "Scheduled maintenance is "
                            "overdue."
                        ),
                        severity=0.70,
                        confidence=1.0,
                        source="maintenance_schedule",
                        timestamp=(
                            asset.next_maintenance_at
                        ),
                    )
                )

            elif (
                asset.next_maintenance_at
                <= now + timedelta(days=7)
            ):
                evidence.append(
                    MaintenanceEvidence(
                        reason=(
                            MaintenanceReason
                            .SCHEDULED
                        ),
                        description=(
                            "Scheduled maintenance is "
                            "due within seven days."
                        ),
                        severity=0.40,
                        confidence=1.0,
                        source="maintenance_schedule",
                        timestamp=(
                            asset.next_maintenance_at
                        ),
                    )
                )

        evidence.extend(external_evidence)

        return self.deduplicate_evidence(
            evidence
        )

    # =====================================================
    # Evidence deduplication
    # =====================================================

    @staticmethod
    def deduplicate_evidence(
        evidence: Sequence[MaintenanceEvidence],
    ) -> list[MaintenanceEvidence]:

        best: dict[
            tuple[str, str, str | None],
            MaintenanceEvidence,
        ] = {}

        provenance_seen: set[str] = set()

        for item in evidence:
            if item.provenance_id:
                if (
                    item.provenance_id
                    in provenance_seen
                ):
                    continue

                provenance_seen.add(
                    item.provenance_id
                )

            key = (
                item.reason.value,
                item.source,
                item.component,
            )

            current = best.get(key)

            if current is None:
                best[key] = item
                continue

            current_score = (
                current.severity
                * current.confidence
            )

            new_score = (
                item.severity
                * item.confidence
            )

            if new_score > current_score:
                best[key] = item

        return list(best.values())

    # =====================================================
    # Data coverage
    # =====================================================

    @staticmethod
    def calculate_data_coverage(
        asset: MaintenanceAssetContext,
        evidence: Sequence[MaintenanceEvidence],
    ) -> tuple[DataCoverage, float]:

        fields = (
            asset.health_score,
            asset.failure_probability,
            asset.rul_hours,
            asset.anomaly_score,
            asset.degradation_score,
        )

        available = sum(
            value is not None
            for value in fields
        )

        coverage_score = available / len(fields)

        if evidence:
            coverage_score = min(
                1.0,
                coverage_score + 0.10,
            )

        if coverage_score >= 0.90:
            level = DataCoverage.COMPLETE

        elif coverage_score >= 0.60:
            level = DataCoverage.GOOD

        elif coverage_score >= 0.30:
            level = DataCoverage.LIMITED

        else:
            level = DataCoverage.INSUFFICIENT

        return level, round(
            coverage_score,
            4,
        )

    # =====================================================
    # Risk
    # =====================================================

    def calculate_risk(
        self,
        asset: MaintenanceAssetContext,
        evidence: Sequence[MaintenanceEvidence],
    ) -> MaintenanceRiskResult:

        coverage, coverage_score = (
            self.calculate_data_coverage(
                asset,
                evidence,
            )
        )

        warnings: list[str] = []

        if (
            coverage
            == DataCoverage.INSUFFICIENT
        ):
            warnings.append(
                "Maintenance risk cannot be "
                "reliably classified because "
                "condition data is insufficient."
            )

        weighted_sum = 0.0
        total_weight = 0.0

        critical_reasons: list[str] = []

        for item in evidence:
            weight = max(
                0.10,
                item.confidence,
            )

            weighted_sum += (
                item.severity * weight
            )

            total_weight += weight

            if item.critical:
                critical_reasons.append(
                    item.description
                )

        physical_risk = (
            weighted_sum / total_weight
            if total_weight > 0
            else 0.0
        )

        # Deterministic critical overrides.
        if (
            asset.failure_probability is not None
            and asset.failure_probability
            >= self.config.failure_critical
        ):
            physical_risk = max(
                physical_risk,
                0.95,
            )

        if (
            asset.rul_hours is not None
            and asset.rul_hours
            <= self.config.rul_critical_hours
        ):
            physical_risk = max(
                physical_risk,
                0.95,
            )

        if any(
            item.critical
            and item.reason
            in {
                MaintenanceReason.SAFETY,
                MaintenanceReason.INCIDENT,
            }
            for item in evidence
        ):
            physical_risk = max(
                physical_risk,
                1.0,
            )

        physical_risk = clamp(
            physical_risk
        )

        # Business priority is separate from
        # physical failure/safety risk.
        criticality = asset.criticality_score

        if asset.production_critical:
            criticality = max(
                criticality,
                0.80,
            )

        if asset.safety_critical:
            criticality = max(
                criticality,
                0.95,
            )

        business_priority = clamp(
            0.75 * physical_risk
            + 0.25 * criticality
        )

        if physical_risk >= (
            self.config.critical_risk_threshold
        ):
            priority = (
                MaintenancePriority.IMMEDIATE
            )

        elif business_priority >= (
            self.config.urgent_risk_threshold
        ):
            priority = (
                MaintenancePriority.URGENT
            )

        elif business_priority >= (
            self.config.high_risk_threshold
        ):
            priority = MaintenancePriority.HIGH

        elif business_priority >= 0.30:
            priority = MaintenancePriority.NORMAL

        else:
            priority = MaintenancePriority.LOW

        if coverage_score <= 0:
            confidence = 0.0

        elif evidence:
            evidence_confidence = (
                sum(
                    item.confidence
                    for item in evidence
                )
                / len(evidence)
            )

            confidence = (
                0.70 * evidence_confidence
                + 0.30 * coverage_score
            )

        else:
            confidence = (
                0.50 * coverage_score
            )

        return MaintenanceRiskResult(
            physical_risk_score=round(
                physical_risk,
                4,
            ),

            business_priority_score=round(
                business_priority,
                4,
            ),

            priority=priority,

            coverage=coverage,

            confidence=round(
                clamp(confidence),
                4,
            ),

            critical_reasons=(
                critical_reasons
            ),

            warnings=warnings,
        )

    # =====================================================
    # Strategy
    # =====================================================

    def choose_strategy(
        self,
        asset: MaintenanceAssetContext,
        risk: MaintenanceRiskResult,
        evidence: Sequence[MaintenanceEvidence],
    ) -> MaintenanceStrategy:

        if any(
            item.critical
            and item.reason
            in {
                MaintenanceReason.SAFETY,
                MaintenanceReason.INCIDENT,
            }
            for item in evidence
        ):
            return MaintenanceStrategy.EMERGENCY

        if (
            risk.priority
            == MaintenancePriority.IMMEDIATE
        ):
            return MaintenanceStrategy.CORRECTIVE

        if (
            asset.repeat_failure_count
            >= self.config.repeat_failure_critical
        ):
            return MaintenanceStrategy.OVERHAUL

        if (
            asset.rul_hours is not None
            or asset.failure_probability is not None
            or asset.degradation_score is not None
        ):
            return MaintenanceStrategy.PREDICTIVE

        if (
            asset.anomaly_score is not None
            or evidence
        ):
            return (
                MaintenanceStrategy.CONDITION_BASED
            )

        if asset.next_maintenance_at:
            return MaintenanceStrategy.PREVENTIVE

        return MaintenanceStrategy.INSPECTION

    # =====================================================
    # Maintenance window
    # =====================================================

    def calculate_window(
        self,
        asset: MaintenanceAssetContext,
        risk: MaintenanceRiskResult,
    ) -> MaintenanceWindow:

        now = utc_now()

        delay_by_priority = {
            MaintenancePriority.IMMEDIATE:
                self.config.immediate_delay_hours,

            MaintenancePriority.URGENT:
                self.config.urgent_delay_hours,

            MaintenancePriority.HIGH:
                self.config.high_delay_hours,

            MaintenancePriority.NORMAL:
                self.config.normal_delay_hours,

            MaintenancePriority.LOW:
                self.config.low_delay_hours,
        }

        max_delay = delay_by_priority[
            risk.priority
        ]

        based_on_rul = False
        based_on_schedule = False

        deadline = (
            now + timedelta(
                hours=max_delay
            )
        )

        reasons = [
            f"priority={risk.priority.value}"
        ]

        if asset.rul_hours is not None:
            # Maintain safety margin: do not schedule
            # at the predicted end of useful life.
            rul_deadline = (
                now
                + timedelta(
                    hours=max(
                        1.0,
                        asset.rul_hours * 0.50,
                    )
                )
            )

            if rul_deadline < deadline:
                deadline = rul_deadline
                based_on_rul = True

                reasons.append(
                    "RUL safety margin"
                )

        if asset.next_maintenance_at:
            if (
                asset.next_maintenance_at
                < deadline
            ):
                deadline = (
                    asset.next_maintenance_at
                )

                based_on_schedule = True

                reasons.append(
                    "scheduled maintenance"
                )

        if deadline < now:
            deadline = now

        available_window = max(
            0.0,
            (
                deadline - now
            ).total_seconds()
            / 3600.0,
        )

        if (
            risk.priority
            == MaintenancePriority.IMMEDIATE
        ):
            recommended_start = now
            emergency = True

        else:
            recommended_start = (
                now
                + timedelta(
                    hours=(
                        available_window
                        * 0.50
                    )
                )
            )

            emergency = False

        return MaintenanceWindow(
            recommended_start=(
                recommended_start
            ),

            deadline=deadline,

            maximum_delay_hours=round(
                available_window,
                2,
            ),

            based_on_rul=based_on_rul,

            based_on_schedule=(
                based_on_schedule
            ),

            emergency=emergency,

            reason=", ".join(reasons),
        )

    # =====================================================
    # Resource readiness
    # =====================================================

    @staticmethod
    def check_readiness(
        resources: MaintenanceResources,
    ) -> MaintenanceReadinessResult:

        required_parts = {
            item.strip().lower()
            for item in resources.required_parts
            if item.strip()
        }

        available_parts = {
            item.strip().lower()
            for item in resources.available_parts
            if item.strip()
        }

        missing_parts = sorted(
            required_parts
            - available_parts
        )

        required_tools = {
            item.strip().lower()
            for item in resources.required_tools
            if item.strip()
        }

        available_tools = {
            item.strip().lower()
            for item in resources.available_tools
            if item.strip()
        }

        missing_tools = sorted(
            required_tools
            - available_tools
        )

        skills = {
            item.strip().lower()
            for item in resources.technician_skills
            if item.strip()
        }

        missing_skill = False

        if resources.required_skill:
            missing_skill = (
                resources.required_skill
                .strip()
                .lower()
                not in skills
            )

        checks: list[bool] = []

        if (
            resources.technician_available
            is not None
        ):
            checks.append(
                resources.technician_available
            )

        if required_parts:
            checks.append(
                not missing_parts
            )

        if required_tools:
            checks.append(
                not missing_tools
            )

        if resources.required_skill:
            checks.append(
                not missing_skill
            )

        if (
            resources.production_window_available
            is not None
        ):
            checks.append(
                resources
                .production_window_available
            )

        warnings: list[str] = []

        if missing_parts:
            warnings.append(
                "Required spare parts are missing."
            )

        if missing_tools:
            warnings.append(
                "Required maintenance tools are "
                "missing."
            )

        if missing_skill:
            warnings.append(
                "Required technician skill is not "
                "available."
            )

        if (
            resources.technician_available
            is False
        ):
            warnings.append(
                "No technician is currently "
                "available."
            )

        if (
            resources.production_window_available
            is False
        ):
            warnings.append(
                "No approved production maintenance "
                "window is currently available."
            )

        if not checks:
            return MaintenanceReadinessResult(
                readiness=(
                    MaintenanceReadiness.UNKNOWN
                ),
                ready=False,
                missing_parts=missing_parts,
                missing_tools=missing_tools,
                missing_skill=missing_skill,
                technician_available=(
                    resources
                    .technician_available
                ),
                production_window_available=(
                    resources
                    .production_window_available
                ),
                warnings=[
                    "Maintenance resource readiness "
                    "cannot be determined."
                ],
                score=0.0,
            )

        score = sum(checks) / len(checks)

        if all(checks):
            readiness = MaintenanceReadiness.READY
            ready = True

        elif score >= 0.50:
            readiness = (
                MaintenanceReadiness
                .PARTIALLY_READY
            )
            ready = False

        else:
            readiness = (
                MaintenanceReadiness.BLOCKED
            )
            ready = False

        return MaintenanceReadinessResult(
            readiness=readiness,
            ready=ready,

            missing_parts=missing_parts,
            missing_tools=missing_tools,
            missing_skill=missing_skill,

            technician_available=(
                resources.technician_available
            ),

            production_window_available=(
                resources
                .production_window_available
            ),

            warnings=warnings,

            score=round(score, 4),
        )

    # =====================================================
    # Recommended actions
    # =====================================================

    def recommend_actions(
        self,
        asset: MaintenanceAssetContext,
        risk: MaintenanceRiskResult,
        evidence: Sequence[MaintenanceEvidence],
        window: MaintenanceWindow,
    ) -> list[MaintenanceActionItem]:

        actions: list[MaintenanceActionItem] = []

        def add(
            action: MaintenanceAction,
            reason: MaintenanceReason,
            description: str,
            *,
            component: str | None = None,
            shutdown: bool = False,
            approval: bool = False,
        ) -> None:
            actions.append(
                MaintenanceActionItem(
                    action_id=new_id(),

                    action=action,
                    reason=reason,

                    description=description,

                    priority=risk.priority,

                    component=component,

                    due_at=window.deadline,

                    requires_shutdown=shutdown,

                    requires_human_approval=approval,
                )
            )

        reasons = {
            item.reason
            for item in evidence
        }

        if (
            risk.coverage
            == DataCoverage.INSUFFICIENT
        ):
            add(
                MaintenanceAction.INSPECT,
                MaintenanceReason.OTHER,
                (
                    "Perform manual inspection and "
                    "collect additional condition "
                    "data before making a major "
                    "maintenance decision."
                ),
            )

        if MaintenanceReason.VIBRATION in reasons:
            add(
                MaintenanceAction.VIBRATION_ANALYSIS,
                MaintenanceReason.VIBRATION,
                (
                    "Perform detailed vibration "
                    "analysis."
                ),
            )

            add(
                MaintenanceAction.CHECK_BEARING,
                MaintenanceReason.VIBRATION,
                (
                    "Inspect bearings for wear, "
                    "lubrication issues and damage."
                ),
            )

            add(
                MaintenanceAction.ALIGN,
                MaintenanceReason.VIBRATION,
                (
                    "Verify shaft and coupling "
                    "alignment."
                ),
            )

            add(
                MaintenanceAction.BALANCE,
                MaintenanceReason.VIBRATION,
                (
                    "Check rotating assembly "
                    "balance."
                ),
            )

        if MaintenanceReason.TEMPERATURE in reasons:
            add(
                MaintenanceAction.THERMAL_INSPECTION,
                MaintenanceReason.TEMPERATURE,
                (
                    "Perform thermal inspection and "
                    "identify abnormal heat sources."
                ),
            )

            add(
                MaintenanceAction.CHECK_COOLING,
                MaintenanceReason.TEMPERATURE,
                (
                    "Inspect cooling and ventilation "
                    "systems."
                ),
            )

        if MaintenanceReason.CURRENT in reasons:
            add(
                MaintenanceAction.CHECK_ELECTRICAL,
                MaintenanceReason.CURRENT,
                (
                    "Inspect electrical load, "
                    "connections and motor condition."
                ),
            )

        if MaintenanceReason.PRESSURE in reasons:
            add(
                MaintenanceAction.LEAK_INSPECTION,
                MaintenanceReason.PRESSURE,
                (
                    "Inspect pressure system for "
                    "leaks, restrictions and valve "
                    "faults."
                ),
            )

        if (
            MaintenanceReason.FAILURE_RISK
            in reasons
            or MaintenanceReason.LOW_RUL
            in reasons
            or MaintenanceReason.REPEAT_FAILURE
            in reasons
        ):
            add(
                MaintenanceAction.REPAIR,
                MaintenanceReason.FAILURE_RISK,
                (
                    "Inspect and repair the predicted "
                    "failure mechanism."
                ),
                approval=True,
            )

        if MaintenanceReason.REPEAT_FAILURE in reasons:
            add(
                MaintenanceAction.OVERHAUL,
                MaintenanceReason.REPEAT_FAILURE,
                (
                    "Evaluate overhaul or component "
                    "replacement because failures "
                    "are recurring."
                ),
                approval=True,
            )

        if (
            MaintenanceReason.SAFETY in reasons
            or MaintenanceReason.INCIDENT
            in reasons
        ):
            add(
                MaintenanceAction.SAFETY_REVIEW,
                MaintenanceReason.SAFETY,
                (
                    "Require qualified safety review "
                    "before continued operation."
                ),
                approval=True,
            )

        if (
            asset.safety_critical
            and risk.priority
            in {
                MaintenancePriority.IMMEDIATE,
                MaintenancePriority.URGENT,
            }
        ):
            add(
                MaintenanceAction.REQUEST_MACHINE_STOP,
                MaintenanceReason.SAFETY,
                (
                    "Request deterministic safety "
                    "controller / authorized operator "
                    "review for safe machine stop."
                ),
                shutdown=True,
                approval=True,
            )

        if not actions:
            if (
                risk.priority
                == MaintenancePriority.LOW
            ):
                add(
                    MaintenanceAction.MONITOR,
                    MaintenanceReason.OTHER,
                    (
                        "Continue condition monitoring "
                        "and trend analysis."
                    ),
                )
            else:
                add(
                    MaintenanceAction.INSPECT,
                    MaintenanceReason.OTHER,
                    (
                        "Perform targeted maintenance "
                        "inspection."
                    ),
                )

        if risk.priority in {
            MaintenancePriority.HIGH,
            MaintenancePriority.URGENT,
            MaintenancePriority.IMMEDIATE,
        }:
            add(
                MaintenanceAction.CREATE_WORK_ORDER,
                MaintenanceReason.OTHER,
                (
                    "Create a maintenance work order "
                    "for the recommended actions."
                ),
            )

        # Always verify after an actual repair,
        # overhaul or component replacement.
        if any(
            item.action
            in {
                MaintenanceAction.REPAIR,
                MaintenanceAction.OVERHAUL,
                MaintenanceAction.REPLACE_COMPONENT,
            }
            for item in actions
        ):
            add(
                MaintenanceAction.POST_MAINTENANCE_VERIFY,
                MaintenanceReason.OTHER,
                (
                    "Verify telemetry, vibration, "
                    "temperature, failure risk and "
                    "asset health after maintenance."
                ),
            )

        return self._deduplicate_actions(
            actions
        )

    # =====================================================
    # Work order
    # =====================================================

    def build_work_order(
        self,
        *,
        asset: MaintenanceAssetContext,
        strategy: MaintenanceStrategy,
        risk: MaintenanceRiskResult,
        window: MaintenanceWindow,
        actions: Sequence[MaintenanceActionItem],
        resources: MaintenanceResources,
        request_id: str | None = None,
        trace_id: str | None = None,
    ) -> WorkOrderRequest:

        action_signature = "|".join(
            sorted(
                item.action.value
                for item in actions
            )
        )

        key_material = (
            f"{asset.asset_id}|"
            f"{strategy.value}|"
            f"{risk.priority.value}|"
            f"{action_signature}|"
            f"{window.deadline.date().isoformat()}"
        )

        import hashlib

        idempotency_key = hashlib.sha256(
            key_material.encode("utf-8")
        ).hexdigest()

        return WorkOrderRequest(
            idempotency_key=idempotency_key,

            asset_id=asset.asset_id,

            title=(
                f"{risk.priority.value.upper()} "
                f"maintenance — "
                f"{asset.asset_name or asset.asset_id}"
            ),

            description=(
                "Maintenance plan generated from "
                "asset condition, failure risk, RUL, "
                "inspection/incident evidence and "
                "maintenance schedule."
            ),

            priority=risk.priority,

            strategy=strategy,

            actions=list(actions),

            scheduled_start=(
                window.recommended_start
            ),

            deadline=window.deadline,

            estimated_duration_hours=(
                resources
                .estimated_duration_hours
            ),

            required_parts=list(
                resources.required_parts
            ),

            required_tools=list(
                resources.required_tools
            ),

            required_skill=(
                resources.required_skill
            ),

            request_id=request_id,
            trace_id=trace_id,

            metadata={
                "physical_risk_score":
                    risk.physical_risk_score,

                "business_priority_score":
                    risk.business_priority_score,

                "confidence":
                    risk.confidence,

                "data_coverage":
                    risk.coverage.value,
            },
        )

    # =====================================================
    # Full plan
    # =====================================================

    def build_plan(
        self,
        *,
        asset: MaintenanceAssetContext,
        resources: MaintenanceResources | None = None,
        evidence: Sequence[
            MaintenanceEvidence
        ] = (),
        request_id: str | None = None,
        trace_id: str | None = None,
    ) -> MaintenancePlan:

        resources = (
            resources
            or MaintenanceResources()
        )

        all_evidence = self.build_evidence(
            asset,
            external_evidence=evidence,
        )

        risk = self.calculate_risk(
            asset,
            all_evidence,
        )

        strategy = self.choose_strategy(
            asset,
            risk,
            all_evidence,
        )

        window = self.calculate_window(
            asset,
            risk,
        )

        readiness = self.check_readiness(
            resources
        )

        actions = self.recommend_actions(
            asset,
            risk,
            all_evidence,
            window,
        )

        maintenance_required = (
            risk.priority
            != MaintenancePriority.LOW
            or bool(all_evidence)
        )

        stop_review_required = any(
            item.action
            == MaintenanceAction
            .REQUEST_MACHINE_STOP
            for item in actions
        )

        human_review_required = (
            stop_review_required
            or risk.priority
            in {
                MaintenancePriority.URGENT,
                MaintenancePriority.IMMEDIATE,
            }
            or risk.confidence
            < self.config.minimum_confidence
            or risk.coverage
            == DataCoverage.INSUFFICIENT
        )

        work_order: WorkOrderRequest | None = None

        if maintenance_required:
            work_order = self.build_work_order(
                asset=asset,
                strategy=strategy,
                risk=risk,
                window=window,
                actions=actions,
                resources=resources,
                request_id=request_id,
                trace_id=trace_id,
            )

        warnings = [
            *risk.warnings,
            *readiness.warnings,
        ]

        if (
            risk.coverage
            == DataCoverage.INSUFFICIENT
        ):
            warnings.append(
                "Do not interpret low calculated "
                "risk as evidence that the asset is "
                "healthy."
            )

        if (
            stop_review_required
        ):
            warnings.append(
                "Machine stop is a recommendation "
                "only. Physical actuation must be "
                "performed by approved deterministic "
                "safety controls or authorized "
                "personnel."
            )

        return MaintenancePlan(
            plan_id=new_id(),

            asset_id=asset.asset_id,

            strategy=strategy,
            priority=risk.priority,

            risk=risk,

            window=window,

            readiness=readiness,

            actions=actions,

            evidence=all_evidence,

            maintenance_required=(
                maintenance_required
            ),

            machine_stop_review_required=(
                stop_review_required
            ),

            human_review_required=(
                human_review_required
            ),

            work_order=work_order,

            warnings=warnings,

            metadata={
                "request_id": request_id,
                "trace_id": trace_id,
            },
        )

    # =====================================================
    # Post-maintenance verification
    # =====================================================

    @staticmethod
    def verify_improvement(
        *,
        before_health_score: float | None,
        after_health_score: float | None,

        before_failure_probability: float | None,
        after_failure_probability: float | None,

        before_anomaly_score: float | None,
        after_anomaly_score: float | None,
    ) -> dict[str, Any]:

        checks: dict[str, Any] = {}

        improved = 0
        total = 0

        if (
            before_health_score is not None
            and after_health_score is not None
        ):
            total += 1

            delta = (
                after_health_score
                - before_health_score
            )

            checks["health_score_delta"] = (
                round(delta, 4)
            )

            if delta > 0:
                improved += 1

        if (
            before_failure_probability
            is not None
            and after_failure_probability
            is not None
        ):
            total += 1

            delta = (
                before_failure_probability
                - after_failure_probability
            )

            checks[
                "failure_probability_reduction"
            ] = round(delta, 6)

            if delta > 0:
                improved += 1

        if (
            before_anomaly_score is not None
            and after_anomaly_score is not None
        ):
            total += 1

            delta = (
                before_anomaly_score
                - after_anomaly_score
            )

            checks[
                "anomaly_score_reduction"
            ] = round(delta, 6)

            if delta > 0:
                improved += 1

        effectiveness = (
            improved / total
            if total
            else 0.0
        )

        return {
            "verified": total > 0,
            "improved_checks": improved,
            "total_checks": total,
            "effectiveness_score": round(
                effectiveness,
                4,
            ),
            "effective": (
                total > 0
                and effectiveness >= 0.67
            ),
            "checks": checks,
        }

    # =====================================================
    # Action dedupe
    # =====================================================

    @staticmethod
    def _deduplicate_actions(
        actions: Sequence[
            MaintenanceActionItem
        ],
    ) -> list[MaintenanceActionItem]:

        seen: set[
            tuple[str, str | None]
        ] = set()

        result: list[
            MaintenanceActionItem
        ] = []

        for item in actions:
            key = (
                item.action.value,
                item.component,
            )

            if key in seen:
                continue

            seen.add(key)
            result.append(item)

        return result


# =========================================================
# Default tools instance
# =========================================================

maintenance_tools = MaintenanceTools()


# =========================================================
# Agent-facing functions
# =========================================================

def calculate_maintenance_risk(
    asset: MaintenanceAssetContext,
    *,
    evidence: Sequence[
        MaintenanceEvidence
    ] = (),
) -> MaintenanceRiskResult:

    all_evidence = (
        maintenance_tools.build_evidence(
            asset,
            external_evidence=evidence,
        )
    )

    return maintenance_tools.calculate_risk(
        asset,
        all_evidence,
    )


def calculate_maintenance_window(
    asset: MaintenanceAssetContext,
    risk: MaintenanceRiskResult,
) -> MaintenanceWindow:

    return maintenance_tools.calculate_window(
        asset,
        risk,
    )


def check_maintenance_readiness(
    resources: MaintenanceResources,
) -> MaintenanceReadinessResult:

    return maintenance_tools.check_readiness(
        resources
    )


def create_maintenance_plan(
    *,
    asset: MaintenanceAssetContext,
    resources: MaintenanceResources | None = None,
    evidence: Sequence[
        MaintenanceEvidence
    ] = (),
    request_id: str | None = None,
    trace_id: str | None = None,
) -> MaintenancePlan:

    return maintenance_tools.build_plan(
        asset=asset,
        resources=resources,
        evidence=evidence,
        request_id=request_id,
        trace_id=trace_id,
    )


def verify_maintenance_effectiveness(
    *,
    before_health_score: float | None,
    after_health_score: float | None,

    before_failure_probability: float | None,
    after_failure_probability: float | None,

    before_anomaly_score: float | None,
    after_anomaly_score: float | None,
) -> dict[str, Any]:

    return (
        maintenance_tools.verify_improvement(
            before_health_score=(
                before_health_score
            ),
            after_health_score=(
                after_health_score
            ),

            before_failure_probability=(
                before_failure_probability
            ),
            after_failure_probability=(
                after_failure_probability
            ),

            before_anomaly_score=(
                before_anomaly_score
            ),
            after_anomaly_score=(
                after_anomaly_score
            ),
        )
    )


# =========================================================
# Exports
# =========================================================

__all__ = [
    "MaintenanceStrategy",
    "MaintenancePriority",
    "MaintenanceAction",
    "MaintenanceReason",
    "MaintenanceReadiness",
    "DataCoverage",

    "MaintenanceEvidence",
    "MaintenanceAssetContext",
    "MaintenanceResources",

    "MaintenanceReadinessResult",
    "MaintenanceRiskResult",
    "MaintenanceWindow",
    "MaintenanceActionItem",
    "WorkOrderRequest",
    "MaintenancePlan",

    "MaintenanceToolsConfig",
    "MaintenanceTools",

    "WorkOrderService",
    "MaintenanceRepository",

    "maintenance_tools",

    "calculate_maintenance_risk",
    "calculate_maintenance_window",
    "check_maintenance_readiness",
    "create_maintenance_plan",
    "verify_maintenance_effectiveness",
]