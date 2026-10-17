# app/agents/repair_replacement_agent.py

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


def _non_negative(value: float, *, name: str) -> float:
    value = _finite(value, name=name)
    if value < 0:
        raise ValueError(f"{name} cannot be negative.")
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

class RepairReplacementDecision(str, Enum):
    REPAIR = "repair"
    REPLACE = "replace"
    MONITOR = "monitor"
    INSPECT = "inspect"
    NO_ACTION = "no_action"
    HUMAN_REVIEW = "human_review"


class DecisionPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    IMMEDIATE = "immediate"


class DecisionConfidence(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class AssetCondition(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class DecisionReason(str, Enum):
    LOW_FAILURE_RISK = "low_failure_risk"
    HIGH_FAILURE_RISK = "high_failure_risk"

    LOW_RUL = "low_rul"

    POOR_HEALTH = "poor_health"

    CRITICAL_DEFECT = "critical_defect"

    REPEATED_FAILURE = "repeated_failure"

    HIGH_REPAIR_COST = "high_repair_cost"

    REPLACEMENT_ECONOMICALLY_BETTER = (
        "replacement_economically_better"
    )

    REPAIR_ECONOMICALLY_BETTER = (
        "repair_economically_better"
    )

    SAFETY_CRITICAL = "safety_critical"

    HIGH_DOWNTIME_COST = "high_downtime_cost"

    OBSOLESCENCE = "obsolescence"

    PARTS_UNAVAILABLE = "parts_unavailable"

    WARRANTY = "warranty"

    INSUFFICIENT_DATA = "insufficient_data"

    OTHER = "other"


class ActionType(str, Enum):
    CONTINUE_OPERATION = "continue_operation"
    MONITOR = "monitor"
    INSPECT = "inspect"

    REPAIR = "repair"
    REPLACE_COMPONENT = "replace_component"
    REPLACE_ASSET = "replace_asset"

    CREATE_WORK_ORDER = "create_work_order"

    COST_REVIEW = "cost_review"
    SAFETY_REVIEW = "safety_review"

    PROCUREMENT_REVIEW = "procurement_review"

    FAILURE_ANALYSIS = "failure_analysis"

    REQUEST_MACHINE_STOP = "request_machine_stop"

    HUMAN_APPROVAL = "human_approval"


# =========================================================
# Asset context
# =========================================================

@dataclass(slots=True)
class AssetContext:
    asset_id: str

    asset_name: str | None = None
    asset_type: str | None = None

    site_id: str | None = None
    zone_id: str | None = None

    component_id: str | None = None
    component_name: str | None = None

    manufacturer: str | None = None
    model_number: str | None = None
    serial_number: str | None = None

    age_years: float | None = None
    expected_life_years: float | None = None

    operating_hours: float | None = None

    safety_critical: bool = False
    production_critical: bool = False

    criticality: float = 0.5

    obsolete: bool = False
    parts_available: bool = True

    warranty_active: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.criticality = _probability(
            self.criticality,
            name="criticality",
        )

        for name in (
            "age_years",
            "expected_life_years",
            "operating_hours",
        ):
            value = getattr(self, name)

            if value is not None:
                setattr(
                    self,
                    name,
                    _non_negative(
                        value,
                        name=name,
                    ),
                )


# =========================================================
# Health / predictive context
# =========================================================

@dataclass(slots=True)
class HealthContext:
    health_score: float | None = None

    failure_probability: float | None = None

    predicted_rul_hours: float | None = None

    anomaly_score: float | None = None

    degradation_score: float | None = None

    repeated_failure_count: int = 0

    critical_defect: bool = False

    critical_safety_risk: bool = False

    emergency_condition: bool = False

    data_drift_detected: bool = False
    model_drift_detected: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.health_score is not None:
            if not 0 <= self.health_score <= 100:
                raise ValueError(
                    "health_score must be "
                    "between 0 and 100."
                )

        for name in (
            "failure_probability",
            "anomaly_score",
            "degradation_score",
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

        if self.predicted_rul_hours is not None:
            self.predicted_rul_hours = (
                _non_negative(
                    self.predicted_rul_hours,
                    name="predicted_rul_hours",
                )
            )

        if self.repeated_failure_count < 0:
            raise ValueError(
                "repeated_failure_count "
                "cannot be negative."
            )


# =========================================================
# Cost context
# =========================================================

@dataclass(slots=True)
class DecisionCostContext:
    repair_cost: float | None = None

    replacement_cost: float | None = None

    repair_labor_cost: float = 0.0
    repair_parts_cost: float = 0.0

    replacement_labor_cost: float = 0.0
    installation_cost: float = 0.0

    repair_downtime_hours: float = 0.0
    replacement_downtime_hours: float = 0.0

    downtime_cost_per_hour: float = 0.0

    expected_failure_cost: float = 0.0

    emergency_failure_cost: float = 0.0

    salvage_value: float = 0.0

    repair_life_extension_hours: float | None = None
    replacement_life_hours: float | None = None

    annual_maintenance_cost_current: float = 0.0
    annual_maintenance_cost_new: float = 0.0

    energy_savings_per_year: float = 0.0

    currency: str = "INR"

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        fields = (
            "repair_cost",
            "replacement_cost",
            "repair_labor_cost",
            "repair_parts_cost",
            "replacement_labor_cost",
            "installation_cost",
            "repair_downtime_hours",
            "replacement_downtime_hours",
            "downtime_cost_per_hour",
            "expected_failure_cost",
            "emergency_failure_cost",
            "salvage_value",
            "repair_life_extension_hours",
            "replacement_life_hours",
            "annual_maintenance_cost_current",
            "annual_maintenance_cost_new",
            "energy_savings_per_year",
        )

        for name in fields:
            value = getattr(self, name)

            if value is not None:
                setattr(
                    self,
                    name,
                    _non_negative(
                        value,
                        name=name,
                    ),
                )


# =========================================================
# Evidence
# =========================================================

@dataclass(slots=True)
class DecisionEvidence:
    evidence_id: str

    source: str
    description: str

    severity_score: float
    confidence: float

    reason: DecisionReason = DecisionReason.OTHER

    component: str | None = None

    critical: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.severity_score = _probability(
            self.severity_score,
            name="severity_score",
        )

        self.confidence = _probability(
            self.confidence,
            name="confidence",
        )


# =========================================================
# Agent input
# =========================================================

@dataclass(slots=True)
class RepairReplacementAgentInput:
    asset: AssetContext

    health: HealthContext = field(
        default_factory=HealthContext
    )

    cost: DecisionCostContext = field(
        default_factory=DecisionCostContext
    )

    evidence: list[
        DecisionEvidence
    ] = field(default_factory=list)

    inspection_result: Mapping[
        str,
        Any,
    ] | None = None

    failure_analysis: Mapping[
        str,
        Any,
    ] | None = None

    maintenance_plan: Mapping[
        str,
        Any,
    ] | None = None

    incident: Mapping[
        str,
        Any,
    ] | None = None

    prediction_id: str | None = None
    inspection_id: str | None = None
    maintenance_plan_id: str | None = None
    incident_id: str | None = None

    request_id: str | None = None
    trace_id: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# =========================================================
# Cost comparison
# =========================================================

@dataclass(slots=True)
class CostComparison:
    repair_direct_cost: float

    repair_downtime_cost: float

    repair_expected_failure_cost: float

    total_repair_expected_cost: float

    replacement_direct_cost: float

    replacement_downtime_cost: float

    replacement_net_cost: float

    repair_to_replacement_ratio: float | None

    replacement_savings: float

    currency: str


# =========================================================
# Decision factor
# =========================================================

@dataclass(slots=True)
class DecisionFactor:
    name: str

    repair_score: float
    replacement_score: float

    weight: float

    explanation: str

    critical: bool = False


# =========================================================
# Recommended action
# =========================================================

@dataclass(slots=True)
class RecommendedAction:
    action_id: str

    action: ActionType

    priority: DecisionPriority

    reason: str

    component: str | None = None

    due_within_hours: float | None = None

    requires_human_approval: bool = True

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# =========================================================
# Result
# =========================================================

@dataclass(slots=True)
class RepairReplacementAgentResult:
    decision_id: str

    asset_id: str
    asset_name: str | None

    component_id: str | None
    component_name: str | None

    site_id: str | None

    decision: RepairReplacementDecision

    priority: DecisionPriority

    confidence: float

    confidence_level: DecisionConfidence

    condition: AssetCondition

    repair_score: float

    replacement_score: float

    health_score: float | None

    failure_probability: float | None

    predicted_rul_hours: float | None

    cost_comparison: CostComparison

    factors: list[DecisionFactor]

    reasons: list[DecisionReason]

    actions: list[RecommendedAction]

    safety_review_required: bool

    shutdown_review_required: bool

    procurement_review_required: bool

    human_review_required: bool

    reliability_warning: bool

    summary: str

    generated_at: datetime = field(
        default_factory=utc_now
    )

    prediction_id: str | None = None
    inspection_id: str | None = None
    maintenance_plan_id: str | None = None
    incident_id: str | None = None

    trace_id: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id":
                self.decision_id,

            "asset_id":
                self.asset_id,

            "asset_name":
                self.asset_name,

            "component_id":
                self.component_id,

            "component_name":
                self.component_name,

            "site_id":
                self.site_id,

            "decision":
                self.decision.value,

            "priority":
                self.priority.value,

            "confidence":
                self.confidence,

            "confidence_level":
                self.confidence_level.value,

            "condition":
                self.condition.value,

            "repair_score":
                self.repair_score,

            "replacement_score":
                self.replacement_score,

            "health_score":
                self.health_score,

            "failure_probability":
                self.failure_probability,

            "predicted_rul_hours":
                self.predicted_rul_hours,

            "cost_comparison":
                asdict(
                    self.cost_comparison
                ),

            "factors": [
                asdict(item)
                for item in self.factors
            ],

            "reasons": [
                item.value
                for item in self.reasons
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

            "safety_review_required":
                self.safety_review_required,

            "shutdown_review_required":
                self.shutdown_review_required,

            "procurement_review_required":
                self.procurement_review_required,

            "human_review_required":
                self.human_review_required,

            "reliability_warning":
                self.reliability_warning,

            "summary":
                self.summary,

            "generated_at":
                self.generated_at.isoformat(),

            "prediction_id":
                self.prediction_id,

            "inspection_id":
                self.inspection_id,

            "maintenance_plan_id":
                self.maintenance_plan_id,

            "incident_id":
                self.incident_id,

            "trace_id":
                self.trace_id,

            "metadata":
                dict(self.metadata),
        }


# =========================================================
# Protocols
# =========================================================

class DecisionRepository(Protocol):

    async def save_decision(
        self,
        decision: RepairReplacementAgentResult,
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
        metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        ...


class CostAgentService(Protocol):

    async def evaluate(
        self,
        *,
        asset_id: str,
        decision: str,
        repair_cost: float,
        replacement_cost: float,
        metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        ...


class NotificationService(Protocol):

    async def notify(
        self,
        *,
        title: str,
        message: str,
        severity: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        ...


# =========================================================
# Config
# =========================================================

@dataclass(slots=True)
class RepairReplacementAgentConfig:
    minimum_confidence: float = 0.60

    health_repair_threshold: float = 65.0
    health_replace_threshold: float = 30.0

    failure_repair_threshold: float = 0.50
    failure_replace_threshold: float = 0.80
    failure_critical_threshold: float = 0.90

    rul_repair_hours: float = 720.0
    rul_replace_hours: float = 168.0
    rul_critical_hours: float = 24.0

    repeated_failure_replace_count: int = 3

    repair_cost_ratio_threshold: float = 0.65
    strong_replace_cost_ratio: float = 0.85

    decision_margin: float = 0.10

    auto_create_work_order: bool = False

    enable_cost_agent: bool = True

    enable_notifications: bool = True


# =========================================================
# Agent
# =========================================================

class RepairReplacementAgent:

    def __init__(
        self,
        *,
        config: RepairReplacementAgentConfig | None = None,
        repository: DecisionRepository | None = None,
        work_order_service: WorkOrderService | None = None,
        cost_agent_service: CostAgentService | None = None,
        notification_service: NotificationService | None = None,
    ) -> None:

        self.config = (
            config
            or RepairReplacementAgentConfig()
        )

        self.repository = repository

        self.work_order_service = (
            work_order_service
        )

        self.cost_agent_service = (
            cost_agent_service
        )

        self.notification_service = (
            notification_service
        )

    # -----------------------------------------------------
    # Main
    # -----------------------------------------------------

    async def run(
        self,
        data: RepairReplacementAgentInput,
    ) -> RepairReplacementAgentResult:

        cost_comparison = (
            self._cost_comparison(data)
        )

        factors = self._build_factors(
            data,
            cost_comparison,
        )

        repair_score, replacement_score = (
            self._score(factors)
        )

        decision = self._decision(
            data=data,
            repair_score=repair_score,
            replacement_score=(
                replacement_score
            ),
        )

        reliability_warning = (
            data.health.data_drift_detected
            or data.health.model_drift_detected
        )

        confidence = self._confidence(
            data=data,
            repair_score=repair_score,
            replacement_score=(
                replacement_score
            ),
            reliability_warning=(
                reliability_warning
            ),
        )

        if (
            confidence
            < self.config.minimum_confidence
            and decision
            not in {
                RepairReplacementDecision.REPLACE,
            }
        ):
            decision = (
                RepairReplacementDecision
                .HUMAN_REVIEW
            )

        priority = self._priority(
            data,
            decision,
        )

        condition = self._condition(
            data
        )

        reasons = self._reasons(
            data,
            decision,
            cost_comparison,
        )

        safety_review = (
            data.asset.safety_critical
            or data.health
            .critical_safety_risk
            or data.health
            .emergency_condition
        )

        shutdown_review = (
            self._shutdown_review(
                data,
                decision,
                priority,
            )
        )

        procurement_review = (
            decision
            == RepairReplacementDecision.REPLACE
        )

        human_review = (
            reliability_warning
            or safety_review
            or procurement_review
            or confidence
            < self.config.minimum_confidence
            or priority
            in {
                DecisionPriority.URGENT,
                DecisionPriority.IMMEDIATE,
            }
        )

        actions = self._actions(
            data=data,
            decision=decision,
            priority=priority,
            safety_review=safety_review,
            shutdown_review=shutdown_review,
            procurement_review=(
                procurement_review
            ),
        )

        summary = self._summary(
            data=data,
            decision=decision,
            priority=priority,
            repair_score=repair_score,
            replacement_score=(
                replacement_score
            ),
            cost_comparison=(
                cost_comparison
            ),
        )

        result = RepairReplacementAgentResult(
            decision_id=new_id(),

            asset_id=data.asset.asset_id,

            asset_name=data.asset.asset_name,

            component_id=(
                data.asset.component_id
            ),

            component_name=(
                data.asset.component_name
            ),

            site_id=data.asset.site_id,

            decision=decision,

            priority=priority,

            confidence=round(
                confidence,
                4,
            ),

            confidence_level=(
                self._confidence_level(
                    confidence
                )
            ),

            condition=condition,

            repair_score=round(
                repair_score,
                4,
            ),

            replacement_score=round(
                replacement_score,
                4,
            ),

            health_score=(
                data.health.health_score
            ),

            failure_probability=(
                data.health
                .failure_probability
            ),

            predicted_rul_hours=(
                data.health
                .predicted_rul_hours
            ),

            cost_comparison=(
                cost_comparison
            ),

            factors=factors,

            reasons=reasons,

            actions=actions,

            safety_review_required=(
                safety_review
            ),

            shutdown_review_required=(
                shutdown_review
            ),

            procurement_review_required=(
                procurement_review
            ),

            human_review_required=(
                human_review
            ),

            reliability_warning=(
                reliability_warning
            ),

            summary=summary,

            prediction_id=(
                data.prediction_id
            ),

            inspection_id=(
                data.inspection_id
            ),

            maintenance_plan_id=(
                data.maintenance_plan_id
            ),

            incident_id=(
                data.incident_id
            ),

            trace_id=data.trace_id,

            metadata={
                **data.metadata,
                "request_id":
                    data.request_id,
            },
        )

        await self._post_process(result)

        return result

    # -----------------------------------------------------
    # Cost comparison
    # -----------------------------------------------------

    @staticmethod
    def _cost_comparison(
        data: RepairReplacementAgentInput,
    ) -> CostComparison:

        cost = data.cost

        repair_direct = (
            cost.repair_cost
            if cost.repair_cost is not None
            else (
                cost.repair_labor_cost
                + cost.repair_parts_cost
            )
        )

        replacement_direct = (
            cost.replacement_cost
            if cost.replacement_cost is not None
            else (
                cost.replacement_labor_cost
                + cost.installation_cost
            )
        )

        repair_downtime = (
            cost.repair_downtime_hours
            * cost.downtime_cost_per_hour
        )

        replacement_downtime = (
            cost.replacement_downtime_hours
            * cost.downtime_cost_per_hour
        )

        failure_probability = (
            data.health.failure_probability
            or 0.0
        )

        repair_expected_failure_cost = (
            failure_probability
            * cost.expected_failure_cost
        )

        if data.health.emergency_condition:
            repair_expected_failure_cost += (
                cost.emergency_failure_cost
            )

        total_repair = (
            repair_direct
            + repair_downtime
            + repair_expected_failure_cost
        )

        replacement_net = max(
            0.0,
            (
                replacement_direct
                + replacement_downtime
                - cost.salvage_value
                - cost.energy_savings_per_year
            ),
        )

        ratio = None

        if replacement_net > 0:
            ratio = (
                total_repair
                / replacement_net
            )

        replacement_savings = (
            total_repair
            - replacement_net
        )

        return CostComparison(
            repair_direct_cost=round(
                repair_direct,
                2,
            ),

            repair_downtime_cost=round(
                repair_downtime,
                2,
            ),

            repair_expected_failure_cost=round(
                repair_expected_failure_cost,
                2,
            ),

            total_repair_expected_cost=round(
                total_repair,
                2,
            ),

            replacement_direct_cost=round(
                replacement_direct,
                2,
            ),

            replacement_downtime_cost=round(
                replacement_downtime,
                2,
            ),

            replacement_net_cost=round(
                replacement_net,
                2,
            ),

            repair_to_replacement_ratio=(
                round(ratio, 4)
                if ratio is not None
                else None
            ),

            replacement_savings=round(
                replacement_savings,
                2,
            ),

            currency=cost.currency,
        )

    # -----------------------------------------------------
    # Decision factors
    # -----------------------------------------------------

    def _build_factors(
        self,
        data: RepairReplacementAgentInput,
        cost: CostComparison,
    ) -> list[DecisionFactor]:

        factors: list[DecisionFactor] = []

        health = data.health
        asset = data.asset

        # Asset health
        if health.health_score is not None:
            deterioration = _clamp(
                1.0
                - health.health_score / 100.0
            )

            factors.append(
                DecisionFactor(
                    name="asset_health",

                    repair_score=_clamp(
                        1.0
                        - abs(
                            deterioration - 0.55
                        )
                    ),

                    replacement_score=(
                        deterioration
                    ),

                    weight=1.2,

                    explanation=(
                        "Current asset health is "
                        f"{health.health_score:.1f}/100."
                    ),

                    critical=(
                        health.health_score
                        <= self.config
                        .health_replace_threshold
                    ),
                )
            )

        # Failure probability
        if (
            health.failure_probability
            is not None
        ):
            probability = (
                health.failure_probability
            )

            factors.append(
                DecisionFactor(
                    name="failure_probability",

                    repair_score=_clamp(
                        1.0
                        - abs(
                            probability - 0.55
                        )
                    ),

                    replacement_score=(
                        probability
                    ),

                    weight=1.5,

                    explanation=(
                        "Predicted failure "
                        f"probability is "
                        f"{probability:.1%}."
                    ),

                    critical=(
                        probability
                        >= self.config
                        .failure_critical_threshold
                    ),
                )
            )

        # RUL
        if (
            health.predicted_rul_hours
            is not None
        ):
            rul = health.predicted_rul_hours

            replacement_risk = _clamp(
                1.0
                - (
                    rul
                    / max(
                        self.config
                        .rul_repair_hours,
                        1.0,
                    )
                )
            )

            factors.append(
                DecisionFactor(
                    name="remaining_useful_life",

                    repair_score=_clamp(
                        rul
                        / max(
                            self.config
                            .rul_repair_hours,
                            1.0,
                        )
                    ),

                    replacement_score=(
                        replacement_risk
                    ),

                    weight=1.4,

                    explanation=(
                        "Predicted RUL is "
                        f"{rul:.1f} hours."
                    ),

                    critical=(
                        rul
                        <= self.config
                        .rul_critical_hours
                    ),
                )
            )

        # Repeated failures
        if (
            health.repeated_failure_count > 0
        ):
            replacement_score = _clamp(
                health.repeated_failure_count
                / max(
                    self.config
                    .repeated_failure_replace_count,
                    1,
                )
            )

            factors.append(
                DecisionFactor(
                    name="repeated_failures",

                    repair_score=_clamp(
                        1.0 - replacement_score
                    ),

                    replacement_score=(
                        replacement_score
                    ),

                    weight=1.2,

                    explanation=(
                        "Asset has "
                        f"{health.repeated_failure_count} "
                        "repeated failure(s)."
                    ),

                    critical=(
                        health.repeated_failure_count
                        >= self.config
                        .repeated_failure_replace_count
                    ),
                )
            )

        # Cost
        ratio = (
            cost.repair_to_replacement_ratio
        )

        if ratio is not None:
            factors.append(
                DecisionFactor(
                    name="lifecycle_cost",

                    repair_score=_clamp(
                        1.0 - ratio
                    ),

                    replacement_score=_clamp(
                        ratio
                    ),

                    weight=1.5,

                    explanation=(
                        "Expected repair cost is "
                        f"{ratio:.1%} of the "
                        "replacement option."
                    ),

                    critical=(
                        ratio
                        >= self.config
                        .strong_replace_cost_ratio
                    ),
                )
            )

        # Age
        if (
            asset.age_years is not None
            and asset.expected_life_years
            is not None
            and asset.expected_life_years > 0
        ):
            age_ratio = _clamp(
                asset.age_years
                / asset.expected_life_years
            )

            factors.append(
                DecisionFactor(
                    name="asset_age",

                    repair_score=_clamp(
                        1.0 - age_ratio
                    ),

                    replacement_score=age_ratio,

                    weight=0.8,

                    explanation=(
                        "Asset age is "
                        f"{asset.age_years:.1f} years "
                        "against expected life of "
                        f"{asset.expected_life_years:.1f} "
                        "years."
                    ),
                )
            )

        # Obsolescence
        if asset.obsolete:
            factors.append(
                DecisionFactor(
                    name="obsolescence",
                    repair_score=0.10,
                    replacement_score=0.95,
                    weight=1.2,
                    explanation=(
                        "Asset/component is marked "
                        "obsolete."
                    ),
                    critical=False,
                )
            )

        # Parts unavailable
        if not asset.parts_available:
            factors.append(
                DecisionFactor(
                    name="parts_availability",
                    repair_score=0.05,
                    replacement_score=0.95,
                    weight=1.3,
                    explanation=(
                        "Required repair parts are "
                        "not available."
                    ),
                )
            )

        # Safety
        if (
            asset.safety_critical
            or health.critical_safety_risk
        ):
            factors.append(
                DecisionFactor(
                    name="safety_criticality",

                    repair_score=0.35,

                    replacement_score=0.85,

                    weight=1.6,

                    explanation=(
                        "Safety criticality requires "
                        "a conservative lifecycle "
                        "decision."
                    ),

                    critical=(
                        health
                        .critical_safety_risk
                    ),
                )
            )

        # Inspection/failure evidence
        for evidence in data.evidence:
            weighted = (
                evidence.severity_score
                * evidence.confidence
            )

            factors.append(
                DecisionFactor(
                    name=(
                        f"evidence:"
                        f"{evidence.reason.value}"
                    ),

                    repair_score=_clamp(
                        1.0 - weighted
                    ),

                    replacement_score=weighted,

                    weight=(
                        1.3
                        if evidence.critical
                        else 0.7
                    ),

                    explanation=(
                        evidence.description
                    ),

                    critical=evidence.critical,
                )
            )

        return factors

    @staticmethod
    def _score(
        factors: Sequence[DecisionFactor],
    ) -> tuple[float, float]:

        if not factors:
            return 0.0, 0.0

        total_weight = sum(
            factor.weight
            for factor in factors
        )

        if total_weight <= 0:
            return 0.0, 0.0

        repair = sum(
            factor.repair_score
            * factor.weight
            for factor in factors
        ) / total_weight

        replacement = sum(
            factor.replacement_score
            * factor.weight
            for factor in factors
        ) / total_weight

        return (
            _clamp(repair),
            _clamp(replacement),
        )

    # -----------------------------------------------------
    # Decision
    # -----------------------------------------------------

    def _decision(
        self,
        *,
        data: RepairReplacementAgentInput,
        repair_score: float,
        replacement_score: float,
    ) -> RepairReplacementDecision:

        health = data.health
        asset = data.asset

        # Safety/emergency override.
        if (
            health.emergency_condition
            or health.critical_safety_risk
        ):
            return (
                RepairReplacementDecision.REPLACE
            )

        if health.critical_defect:
            return (
                RepairReplacementDecision.REPLACE
            )

        if (
            health.failure_probability
            is not None
            and health.failure_probability
            >= self.config
            .failure_critical_threshold
            and health.predicted_rul_hours
            is not None
            and health.predicted_rul_hours
            <= self.config.rul_replace_hours
        ):
            return (
                RepairReplacementDecision.REPLACE
            )

        if (
            health.predicted_rul_hours
            is not None
            and health.predicted_rul_hours
            <= self.config.rul_critical_hours
        ):
            return (
                RepairReplacementDecision.REPLACE
            )

        if (
            health.repeated_failure_count
            >= self.config
            .repeated_failure_replace_count
        ):
            return (
                RepairReplacementDecision.REPLACE
            )

        if asset.obsolete:
            return (
                RepairReplacementDecision.REPLACE
            )

        if not asset.parts_available:
            return (
                RepairReplacementDecision.REPLACE
            )

        ratio = (
            data.cost.repair_cost
            / data.cost.replacement_cost
            if (
                data.cost.repair_cost
                is not None
                and data.cost.replacement_cost
                is not None
                and data.cost.replacement_cost > 0
            )
            else None
        )

        if (
            ratio is not None
            and ratio
            >= self.config
            .strong_replace_cost_ratio
        ):
            return (
                RepairReplacementDecision.REPLACE
            )

        margin = (
            replacement_score
            - repair_score
        )

        if (
            margin
            >= self.config.decision_margin
        ):
            return (
                RepairReplacementDecision.REPLACE
            )

        if (
            -margin
            >= self.config.decision_margin
        ):
            return (
                RepairReplacementDecision.REPAIR
            )

        if (
            health.failure_probability
            is not None
            and health.failure_probability
            >= self.config
            .failure_repair_threshold
        ):
            return (
                RepairReplacementDecision.REPAIR
            )

        if (
            health.health_score is not None
            and health.health_score
            < self.config
            .health_repair_threshold
        ):
            return (
                RepairReplacementDecision.REPAIR
            )

        if (
            health.anomaly_score is not None
            and health.anomaly_score >= 0.60
        ):
            return (
                RepairReplacementDecision.INSPECT
            )

        if (
            health.health_score is None
            and health.failure_probability is None
            and health.predicted_rul_hours is None
            and not data.evidence
        ):
            return (
                RepairReplacementDecision
                .HUMAN_REVIEW
            )

        return (
            RepairReplacementDecision.MONITOR
        )

    # -----------------------------------------------------
    # Confidence
    # -----------------------------------------------------

    @staticmethod
    def _confidence(
        *,
        data: RepairReplacementAgentInput,
        repair_score: float,
        replacement_score: float,
        reliability_warning: bool,
    ) -> float:

        evidence_confidence = [
            item.confidence
            for item in data.evidence
        ]

        available = sum([
            data.health.health_score
            is not None,

            data.health.failure_probability
            is not None,

            data.health.predicted_rul_hours
            is not None,

            data.cost.repair_cost
            is not None,

            data.cost.replacement_cost
            is not None,

            bool(data.inspection_result),

            bool(data.failure_analysis),
        ])

        coverage = min(
            available / 7.0,
            1.0,
        )

        margin = abs(
            replacement_score
            - repair_score
        )

        confidence = (
            0.40
            + 0.30 * coverage
            + 0.20 * min(
                margin / 0.50,
                1.0,
            )
        )

        if evidence_confidence:
            confidence += (
                0.10
                * (
                    sum(evidence_confidence)
                    / len(evidence_confidence)
                )
            )

        if reliability_warning:
            confidence *= 0.75

        return _clamp(confidence)

    @staticmethod
    def _confidence_level(
        confidence: float,
    ) -> DecisionConfidence:

        if confidence >= 0.90:
            return DecisionConfidence.VERY_HIGH

        if confidence >= 0.75:
            return DecisionConfidence.HIGH

        if confidence >= 0.60:
            return DecisionConfidence.MODERATE

        return DecisionConfidence.LOW

    # -----------------------------------------------------
    # Condition
    # -----------------------------------------------------

    @staticmethod
    def _condition(
        data: RepairReplacementAgentInput,
    ) -> AssetCondition:

        health = data.health

        if health.emergency_condition:
            return AssetCondition.CRITICAL

        if (
            health.health_score is not None
        ):
            score = health.health_score

            if score >= 85:
                return AssetCondition.EXCELLENT

            if score >= 70:
                return AssetCondition.GOOD

            if score >= 50:
                return AssetCondition.FAIR

            if score >= 30:
                return AssetCondition.POOR

            return AssetCondition.CRITICAL

        if (
            health.failure_probability
            is not None
        ):
            if health.failure_probability >= 0.85:
                return AssetCondition.CRITICAL

            if health.failure_probability >= 0.65:
                return AssetCondition.POOR

            if health.failure_probability >= 0.40:
                return AssetCondition.FAIR

            return AssetCondition.GOOD

        return AssetCondition.UNKNOWN

    # -----------------------------------------------------
    # Priority
    # -----------------------------------------------------

    def _priority(
        self,
        data: RepairReplacementAgentInput,
        decision: RepairReplacementDecision,
    ) -> DecisionPriority:

        health = data.health

        if (
            health.emergency_condition
            or health.critical_safety_risk
        ):
            return DecisionPriority.IMMEDIATE

        if (
            health.predicted_rul_hours
            is not None
            and health.predicted_rul_hours
            <= self.config.rul_critical_hours
        ):
            return DecisionPriority.IMMEDIATE

        if (
            health.failure_probability
            is not None
            and health.failure_probability
            >= self.config
            .failure_critical_threshold
        ):
            return DecisionPriority.IMMEDIATE

        if (
            decision
            == RepairReplacementDecision.REPLACE
        ):
            return DecisionPriority.URGENT

        if (
            decision
            == RepairReplacementDecision.REPAIR
        ):
            return DecisionPriority.HIGH

        if (
            decision
            in {
                RepairReplacementDecision.INSPECT,
                RepairReplacementDecision.HUMAN_REVIEW,
            }
        ):
            return DecisionPriority.NORMAL

        return DecisionPriority.LOW

    # -----------------------------------------------------
    # Reasons
    # -----------------------------------------------------

    def _reasons(
        self,
        data: RepairReplacementAgentInput,
        decision: RepairReplacementDecision,
        cost: CostComparison,
    ) -> list[DecisionReason]:

        reasons: list[DecisionReason] = []

        health = data.health

        if (
            health.failure_probability
            is not None
            and health.failure_probability
            >= self.config
            .failure_replace_threshold
        ):
            reasons.append(
                DecisionReason.HIGH_FAILURE_RISK
            )

        if (
            health.predicted_rul_hours
            is not None
            and health.predicted_rul_hours
            <= self.config.rul_replace_hours
        ):
            reasons.append(
                DecisionReason.LOW_RUL
            )

        if (
            health.health_score is not None
            and health.health_score
            <= self.config
            .health_replace_threshold
        ):
            reasons.append(
                DecisionReason.POOR_HEALTH
            )

        if health.critical_defect:
            reasons.append(
                DecisionReason.CRITICAL_DEFECT
            )

        if (
            health.repeated_failure_count
            >= self.config
            .repeated_failure_replace_count
        ):
            reasons.append(
                DecisionReason.REPEATED_FAILURE
            )

        if (
            cost.repair_to_replacement_ratio
            is not None
        ):
            if (
                cost.repair_to_replacement_ratio
                >= self.config
                .repair_cost_ratio_threshold
            ):
                reasons.append(
                    DecisionReason
                    .HIGH_REPAIR_COST
                )

            if (
                cost.replacement_net_cost
                < cost.total_repair_expected_cost
            ):
                reasons.append(
                    DecisionReason
                    .REPLACEMENT_ECONOMICALLY_BETTER
                )
            else:
                reasons.append(
                    DecisionReason
                    .REPAIR_ECONOMICALLY_BETTER
                )

        if (
            data.asset.safety_critical
            or health.critical_safety_risk
        ):
            reasons.append(
                DecisionReason.SAFETY_CRITICAL
            )

        if data.asset.obsolete:
            reasons.append(
                DecisionReason.OBSOLESCENCE
            )

        if not data.asset.parts_available:
            reasons.append(
                DecisionReason.PARTS_UNAVAILABLE
            )

        if data.asset.warranty_active:
            reasons.append(
                DecisionReason.WARRANTY
            )

        if (
            decision
            == RepairReplacementDecision.HUMAN_REVIEW
        ):
            reasons.append(
                DecisionReason.INSUFFICIENT_DATA
            )

        return list(dict.fromkeys(reasons))

    # -----------------------------------------------------
    # Shutdown
    # -----------------------------------------------------

    @staticmethod
    def _shutdown_review(
        data: RepairReplacementAgentInput,
        decision: RepairReplacementDecision,
        priority: DecisionPriority,
    ) -> bool:

        if data.health.emergency_condition:
            return True

        if data.health.critical_safety_risk:
            return True

        if priority == DecisionPriority.IMMEDIATE:
            return True

        if (
            decision
            == RepairReplacementDecision.REPLACE
            and data.asset.safety_critical
        ):
            return True

        return False

    # -----------------------------------------------------
    # Actions
    # -----------------------------------------------------

    def _actions(
        self,
        *,
        data: RepairReplacementAgentInput,
        decision: RepairReplacementDecision,
        priority: DecisionPriority,
        safety_review: bool,
        shutdown_review: bool,
        procurement_review: bool,
    ) -> list[RecommendedAction]:

        actions: list[RecommendedAction] = []

        component = (
            data.asset.component_name
            or data.asset.component_id
        )

        if (
            decision
            == RepairReplacementDecision.REPAIR
        ):
            actions.extend([
                self._action(
                    ActionType.REPAIR,
                    priority,
                    (
                        "Repair is preferred based "
                        "on current condition, risk "
                        "and lifecycle cost."
                    ),
                    component=component,
                ),
                self._action(
                    ActionType.CREATE_WORK_ORDER,
                    priority,
                    (
                        "Create repair work order."
                    ),
                    component=component,
                ),
            ])

        elif (
            decision
            == RepairReplacementDecision.REPLACE
        ):
            actions.extend([
                self._action(
                    (
                        ActionType.REPLACE_COMPONENT
                        if data.asset.component_id
                        else ActionType.REPLACE_ASSET
                    ),
                    priority,
                    (
                        "Replacement is preferred "
                        "based on health, failure "
                        "risk, RUL and lifecycle "
                        "economics."
                    ),
                    component=component,
                ),
                self._action(
                    ActionType.PROCUREMENT_REVIEW,
                    priority,
                    (
                        "Validate replacement part/"
                        "asset availability, lead "
                        "time and procurement cost."
                    ),
                    component=component,
                ),
                self._action(
                    ActionType.CREATE_WORK_ORDER,
                    priority,
                    (
                        "Create replacement work "
                        "order after authorization."
                    ),
                    component=component,
                ),
            ])

        elif (
            decision
            == RepairReplacementDecision.INSPECT
        ):
            actions.append(
                self._action(
                    ActionType.INSPECT,
                    priority,
                    (
                        "Collect additional physical "
                        "inspection evidence before "
                        "repair/replacement decision."
                    ),
                    component=component,
                )
            )

        elif (
            decision
            == RepairReplacementDecision.MONITOR
        ):
            actions.append(
                self._action(
                    ActionType.MONITOR,
                    priority,
                    (
                        "Continue condition "
                        "monitoring and reassess if "
                        "health or failure risk "
                        "deteriorates."
                    ),
                    component=component,
                    approval=False,
                )
            )

        elif (
            decision
            == RepairReplacementDecision.NO_ACTION
        ):
            actions.append(
                self._action(
                    ActionType.CONTINUE_OPERATION,
                    DecisionPriority.LOW,
                    (
                        "No repair or replacement "
                        "action is currently required."
                    ),
                    component=component,
                    approval=False,
                )
            )

        else:
            actions.append(
                self._action(
                    ActionType.HUMAN_APPROVAL,
                    priority,
                    (
                        "Insufficient or conflicting "
                        "evidence requires qualified "
                        "human review."
                    ),
                    component=component,
                )
            )

        if safety_review:
            actions.append(
                self._action(
                    ActionType.SAFETY_REVIEW,
                    (
                        DecisionPriority.IMMEDIATE
                        if shutdown_review
                        else DecisionPriority.URGENT
                    ),
                    (
                        "Qualified safety review is "
                        "required before executing "
                        "the lifecycle decision."
                    ),
                    component=component,
                )
            )

        if shutdown_review:
            actions.append(
                self._action(
                    ActionType.REQUEST_MACHINE_STOP,
                    DecisionPriority.IMMEDIATE,
                    (
                        "Request authorized review "
                        "for controlled or safety-"
                        "rated equipment shutdown."
                    ),
                    component=component,
                    hours=0.0,
                )
            )

        if procurement_review:
            actions.append(
                self._action(
                    ActionType.COST_REVIEW,
                    DecisionPriority.HIGH,
                    (
                        "Validate total replacement "
                        "cost, downtime, installation "
                        "and lifecycle economics."
                    ),
                    component=component,
                )
            )

        return self._deduplicate_actions(actions)

    @staticmethod
    def _action(
        action: ActionType,
        priority: DecisionPriority,
        reason: str,
        *,
        component: str | None = None,
        hours: float | None = None,
        approval: bool = True,
    ) -> RecommendedAction:

        default_hours = {
            DecisionPriority.LOW: 720.0,
            DecisionPriority.NORMAL: 168.0,
            DecisionPriority.HIGH: 72.0,
            DecisionPriority.URGENT: 24.0,
            DecisionPriority.IMMEDIATE: 0.0,
        }

        return RecommendedAction(
            action_id=new_id(),

            action=action,

            priority=priority,

            reason=reason,

            component=component,

            due_within_hours=(
                hours
                if hours is not None
                else default_hours[priority]
            ),

            requires_human_approval=approval,
        )

    @staticmethod
    def _deduplicate_actions(
        actions: Sequence[RecommendedAction],
    ) -> list[RecommendedAction]:

        rank = {
            DecisionPriority.LOW: 1,
            DecisionPriority.NORMAL: 2,
            DecisionPriority.HIGH: 3,
            DecisionPriority.URGENT: 4,
            DecisionPriority.IMMEDIATE: 5,
        }

        unique: dict[
            ActionType,
            RecommendedAction,
        ] = {}

        for item in actions:
            current = unique.get(item.action)

            if (
                current is None
                or rank[item.priority]
                > rank[current.priority]
            ):
                unique[item.action] = item

        return list(unique.values())

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    @staticmethod
    def _summary(
        *,
        data: RepairReplacementAgentInput,
        decision: RepairReplacementDecision,
        priority: DecisionPriority,
        repair_score: float,
        replacement_score: float,
        cost_comparison: CostComparison,
    ) -> str:

        asset = (
            data.asset.component_name
            or data.asset.asset_name
            or data.asset.asset_id
        )

        parts = [
            (
                f"Repair/replacement assessment "
                f"for {asset} recommends "
                f"{decision.value} with "
                f"{priority.value} priority."
            ),
            (
                f"Repair suitability score is "
                f"{repair_score:.1%}; replacement "
                f"suitability score is "
                f"{replacement_score:.1%}."
            ),
        ]

        if (
            cost_comparison
            .repair_to_replacement_ratio
            is not None
        ):
            parts.append(
                "Expected repair cost is "
                f"{cost_comparison.repair_to_replacement_ratio:.1%} "
                "of the evaluated replacement "
                "cost."
            )

        if (
            data.health.predicted_rul_hours
            is not None
        ):
            parts.append(
                "Predicted RUL is "
                f"{data.health.predicted_rul_hours:.1f} "
                "hours."
            )

        if (
            data.health.failure_probability
            is not None
        ):
            parts.append(
                "Failure probability is "
                f"{data.health.failure_probability:.1%}."
            )

        return " ".join(parts)

    # -----------------------------------------------------
    # Post processing
    # -----------------------------------------------------

    async def _post_process(
        self,
        result: RepairReplacementAgentResult,
    ) -> None:

        self._record_metrics(result)
        self._log_result(result)

        if self.repository is not None:
            try:
                await self.repository.save_decision(
                    result
                )
            except Exception:
                pass

        if (
            self.config.enable_cost_agent
            and self.cost_agent_service
            and result.decision
            in {
                RepairReplacementDecision.REPAIR,
                RepairReplacementDecision.REPLACE,
            }
        ):
            await self._run_cost_agent(result)

        if (
            self.config.auto_create_work_order
            and self.work_order_service
            and result.decision
            in {
                RepairReplacementDecision.REPAIR,
                RepairReplacementDecision.REPLACE,
            }
        ):
            await self._create_work_order(result)

        if (
            self.config.enable_notifications
            and self.notification_service
            and result.priority
            in {
                DecisionPriority.URGENT,
                DecisionPriority.IMMEDIATE,
            }
        ):
            await self._notify(result)

    # -----------------------------------------------------
    # Cost agent
    # -----------------------------------------------------

    async def _run_cost_agent(
        self,
        result: RepairReplacementAgentResult,
    ) -> None:

        if self.cost_agent_service is None:
            return

        try:
            await self.cost_agent_service.evaluate(
                asset_id=result.asset_id,

                decision=result.decision.value,

                repair_cost=(
                    result.cost_comparison
                    .total_repair_expected_cost
                ),

                replacement_cost=(
                    result.cost_comparison
                    .replacement_net_cost
                ),

                metadata={
                    "decision_id":
                        result.decision_id,

                    "component_id":
                        result.component_id,

                    "priority":
                        result.priority.value,

                    "currency":
                        result.cost_comparison
                        .currency,
                },
            )

        except Exception:
            pass

    # -----------------------------------------------------
    # Work order
    # -----------------------------------------------------

    async def _create_work_order(
        self,
        result: RepairReplacementAgentResult,
    ) -> None:

        if self.work_order_service is None:
            return

        try:
            await self.work_order_service.create_work_order(
                asset_id=result.asset_id,

                title=(
                    f"{result.decision.value.title()} "
                    f"Decision - "
                    f"{result.component_name or result.asset_name or result.asset_id}"
                ),

                description=result.summary,

                priority=result.priority.value,

                metadata={
                    "decision_id":
                        result.decision_id,

                    "decision":
                        result.decision.value,

                    "component_id":
                        result.component_id,

                    "repair_score":
                        result.repair_score,

                    "replacement_score":
                        result.replacement_score,

                    "inspection_id":
                        result.inspection_id,

                    "maintenance_plan_id":
                        result.maintenance_plan_id,

                    "prediction_id":
                        result.prediction_id,
                },
            )

        except Exception:
            pass

    # -----------------------------------------------------
    # Notification
    # -----------------------------------------------------

    async def _notify(
        self,
        result: RepairReplacementAgentResult,
    ) -> None:

        if self.notification_service is None:
            return

        try:
            await self.notification_service.notify(
                title=(
                    "Repair / Replacement "
                    "Decision Required"
                ),

                message=result.summary,

                severity=result.priority.value,

                metadata={
                    "decision_id":
                        result.decision_id,

                    "asset_id":
                        result.asset_id,

                    "decision":
                        result.decision.value,

                    "shutdown_review":
                        result
                        .shutdown_review_required,

                    "safety_review":
                        result
                        .safety_review_required,
                },
            )

        except Exception:
            pass

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    @staticmethod
    def _record_metrics(
        result: RepairReplacementAgentResult,
    ) -> None:

        try:
            from app.observability.metrics import (
                registry,
            )

            registry.counter(
                "repair_replacement_decisions_total"
            ).inc(
                labels={
                    "decision":
                        result.decision.value,

                    "priority":
                        result.priority.value,
                }
            )

            registry.histogram(
                "repair_replacement_confidence"
            ).observe(
                result.confidence,

                labels={
                    "decision":
                        result.decision.value,
                },
            )

            registry.histogram(
                "repair_suitability_score"
            ).observe(
                result.repair_score
            )

            registry.histogram(
                "replacement_suitability_score"
            ).observe(
                result.replacement_score
            )

        except Exception:
            pass

    # -----------------------------------------------------
    # Logging
    # -----------------------------------------------------

    @staticmethod
    def _log_result(
        result: RepairReplacementAgentResult,
    ) -> None:

        try:
            from app.observability.logging import (
                get_logger,
            )

            logger = get_logger(
                "industrial.repair_replacement_agent"
            )

            payload = {
                "decision_id":
                    result.decision_id,

                "asset_id":
                    result.asset_id,

                "component_id":
                    result.component_id,

                "decision":
                    result.decision.value,

                "priority":
                    result.priority.value,

                "confidence":
                    result.confidence,

                "repair_score":
                    result.repair_score,

                "replacement_score":
                    result.replacement_score,

                "failure_probability":
                    result.failure_probability,

                "predicted_rul_hours":
                    result.predicted_rul_hours,

                "safety_review":
                    result.safety_review_required,

                "shutdown_review":
                    result.shutdown_review_required,
            }

            if (
                result.priority
                == DecisionPriority.IMMEDIATE
            ):
                logger.critical(
                    "Immediate repair/replacement "
                    "decision required.",
                    extra={
                        "repair_replacement":
                            payload
                    },
                )

            elif (
                result.priority
                in {
                    DecisionPriority.URGENT,
                    DecisionPriority.HIGH,
                }
            ):
                logger.warning(
                    "Repair/replacement decision "
                    "generated.",
                    extra={
                        "repair_replacement":
                            payload
                    },
                )

            else:
                logger.info(
                    "Repair/replacement assessment "
                    "completed.",
                    extra={
                        "repair_replacement":
                            payload
                    },
                )

        except Exception:
            pass


# =========================================================
# Factory
# =========================================================

def create_repair_replacement_agent(
    *,
    config: RepairReplacementAgentConfig | None = None,
    repository: DecisionRepository | None = None,
    work_order_service: WorkOrderService | None = None,
    cost_agent_service: CostAgentService | None = None,
    notification_service: NotificationService | None = None,
) -> RepairReplacementAgent:

    return RepairReplacementAgent(
        config=config,
        repository=repository,
        work_order_service=work_order_service,
        cost_agent_service=cost_agent_service,
        notification_service=notification_service,
    )


repair_replacement_agent = (
    RepairReplacementAgent()
)


__all__ = [
    "RepairReplacementDecision",
    "DecisionPriority",
    "DecisionConfidence",
    "AssetCondition",
    "DecisionReason",
    "ActionType",

    "AssetContext",
    "HealthContext",
    "DecisionCostContext",
    "DecisionEvidence",

    "RepairReplacementAgentInput",

    "CostComparison",
    "DecisionFactor",
    "RecommendedAction",

    "RepairReplacementAgentResult",

    "DecisionRepository",
    "WorkOrderService",
    "CostAgentService",
    "NotificationService",

    "RepairReplacementAgentConfig",
    "RepairReplacementAgent",

    "create_repair_replacement_agent",
    "repair_replacement_agent",
]