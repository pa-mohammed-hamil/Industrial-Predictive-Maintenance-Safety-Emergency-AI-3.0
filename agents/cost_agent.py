# app/agents/cost_agent.py

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


def _safe_divide(
    numerator: float,
    denominator: float,
    default: float = 0.0,
) -> float:
    if abs(denominator) <= 1e-12:
        return default

    return numerator / denominator


# =========================================================
# Enums
# =========================================================

class CostDecision(str, Enum):
    MONITOR = "monitor"
    INSPECT = "inspect"

    PREVENTIVE_MAINTENANCE = (
        "preventive_maintenance"
    )

    PREDICTIVE_MAINTENANCE = (
        "predictive_maintenance"
    )

    REPAIR = "repair"
    REPLACE = "replace"

    REDUCE_LOAD = "reduce_load"

    SHUTDOWN = "shutdown"

    HUMAN_REVIEW = "human_review"


class CostRiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class RecommendationPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CostCategory(str, Enum):
    REPAIR = "repair"
    REPLACEMENT = "replacement"

    LABOR = "labor"
    SPARE_PARTS = "spare_parts"

    DOWNTIME = "downtime"
    PRODUCTION_LOSS = "production_loss"

    SCRAP = "scrap"
    QUALITY = "quality"

    ENERGY = "energy"

    SAFETY = "safety"
    EMERGENCY = "emergency"

    LOGISTICS = "logistics"
    CONTRACTOR = "contractor"

    MAINTENANCE = "maintenance"

    OTHER = "other"


# =========================================================
# Cost models
# =========================================================

@dataclass(slots=True)
class CostBreakdown:
    repair: float = 0.0
    replacement: float = 0.0

    labor: float = 0.0
    spare_parts: float = 0.0

    downtime: float = 0.0
    production_loss: float = 0.0

    scrap: float = 0.0
    quality: float = 0.0

    energy: float = 0.0

    safety: float = 0.0
    emergency: float = 0.0

    logistics: float = 0.0
    contractor: float = 0.0

    other: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "repair",
            "replacement",
            "labor",
            "spare_parts",
            "downtime",
            "production_loss",
            "scrap",
            "quality",
            "energy",
            "safety",
            "emergency",
            "logistics",
            "contractor",
            "other",
        ):
            setattr(
                self,
                name,
                _non_negative(
                    getattr(self, name),
                    name=name,
                ),
            )

    @property
    def total(self) -> float:
        return sum(
            (
                self.repair,
                self.replacement,
                self.labor,
                self.spare_parts,
                self.downtime,
                self.production_loss,
                self.scrap,
                self.quality,
                self.energy,
                self.safety,
                self.emergency,
                self.logistics,
                self.contractor,
                self.other,
            )
        )

    def to_dict(self) -> dict[str, float]:
        data = asdict(self)
        data["total"] = self.total
        return data


@dataclass(slots=True)
class RepairOption:
    repair_cost: float

    labor_cost: float = 0.0
    parts_cost: float = 0.0

    downtime_hours: float = 0.0

    post_repair_failure_probability: float = 0.10

    life_extension_hours: float | None = None

    success_probability: float = 0.90

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.repair_cost = _non_negative(
            self.repair_cost,
            name="repair_cost",
        )

        self.labor_cost = _non_negative(
            self.labor_cost,
            name="labor_cost",
        )

        self.parts_cost = _non_negative(
            self.parts_cost,
            name="parts_cost",
        )

        self.downtime_hours = _non_negative(
            self.downtime_hours,
            name="downtime_hours",
        )

        self.post_repair_failure_probability = (
            _probability(
                self.post_repair_failure_probability,
                name=(
                    "post_repair_failure_probability"
                ),
            )
        )

        self.success_probability = _probability(
            self.success_probability,
            name="success_probability",
        )


@dataclass(slots=True)
class ReplacementOption:
    replacement_cost: float

    installation_cost: float = 0.0
    logistics_cost: float = 0.0

    downtime_hours: float = 0.0

    salvage_value: float = 0.0

    post_replacement_failure_probability: float = 0.02

    expected_life_hours: float | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.replacement_cost = _non_negative(
            self.replacement_cost,
            name="replacement_cost",
        )

        self.installation_cost = _non_negative(
            self.installation_cost,
            name="installation_cost",
        )

        self.logistics_cost = _non_negative(
            self.logistics_cost,
            name="logistics_cost",
        )

        self.downtime_hours = _non_negative(
            self.downtime_hours,
            name="downtime_hours",
        )

        self.salvage_value = _non_negative(
            self.salvage_value,
            name="salvage_value",
        )

        self.post_replacement_failure_probability = (
            _probability(
                self.post_replacement_failure_probability,
                name=(
                    "post_replacement_failure_probability"
                ),
            )
        )


@dataclass(slots=True)
class MaintenanceOption:
    maintenance_cost: float

    downtime_hours: float = 0.0

    post_maintenance_failure_probability: float = 0.15

    life_extension_hours: float | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.maintenance_cost = _non_negative(
            self.maintenance_cost,
            name="maintenance_cost",
        )

        self.downtime_hours = _non_negative(
            self.downtime_hours,
            name="downtime_hours",
        )

        self.post_maintenance_failure_probability = (
            _probability(
                self.post_maintenance_failure_probability,
                name=(
                    "post_maintenance_failure_probability"
                ),
            )
        )


# =========================================================
# Input
# =========================================================

@dataclass(slots=True)
class CostAgentInput:
    asset_id: str

    asset_name: str | None = None
    asset_type: str | None = None
    site_id: str | None = None

    currency: str = "INR"

    failure_probability: float = 0.0

    predicted_failure_mode: str | None = None

    predicted_rul_hours: float | None = None

    asset_health_score: float | None = None

    anomaly_score: float | None = None

    # Failure consequence
    direct_failure_cost: float = 0.0

    emergency_failure_cost: float = 0.0

    safety_consequence_cost: float = 0.0

    # Downtime
    expected_failure_downtime_hours: float = 0.0

    downtime_cost_per_hour: float = 0.0

    # Production
    production_units_per_hour: float = 0.0

    contribution_margin_per_unit: float = 0.0

    # Quality / scrap
    expected_scrap_cost: float = 0.0
    expected_quality_cost: float = 0.0

    # Energy
    expected_excess_energy_cost: float = 0.0

    # Options
    repair: RepairOption | None = None

    replacement: ReplacementOption | None = None

    maintenance: MaintenanceOption | None = None

    # Business importance
    asset_criticality: float = 0.5
    production_criticality: float = 0.5

    # Model reliability
    prediction_confidence: float | None = None

    data_drift_detected: bool = False
    model_drift_detected: bool = False

    work_order_id: str | None = None
    prediction_id: str | None = None

    trace_id: str | None = None
    request_id: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.failure_probability = _probability(
            self.failure_probability,
            name="failure_probability",
        )

        self.asset_criticality = _probability(
            self.asset_criticality,
            name="asset_criticality",
        )

        self.production_criticality = _probability(
            self.production_criticality,
            name="production_criticality",
        )

        if self.prediction_confidence is not None:
            self.prediction_confidence = (
                _probability(
                    self.prediction_confidence,
                    name="prediction_confidence",
                )
            )

        if (
            self.predicted_rul_hours is not None
            and self.predicted_rul_hours < 0
        ):
            raise ValueError(
                "predicted_rul_hours cannot "
                "be negative."
            )

        if (
            self.asset_health_score is not None
            and not 0 <= self.asset_health_score <= 100
        ):
            raise ValueError(
                "asset_health_score must be "
                "between 0 and 100."
            )

        if self.anomaly_score is not None:
            self.anomaly_score = _probability(
                self.anomaly_score,
                name="anomaly_score",
            )

        for name in (
            "direct_failure_cost",
            "emergency_failure_cost",
            "safety_consequence_cost",
            "expected_failure_downtime_hours",
            "downtime_cost_per_hour",
            "production_units_per_hour",
            "contribution_margin_per_unit",
            "expected_scrap_cost",
            "expected_quality_cost",
            "expected_excess_energy_cost",
        ):
            setattr(
                self,
                name,
                _non_negative(
                    getattr(self, name),
                    name=name,
                ),
            )


# =========================================================
# Option evaluation
# =========================================================

@dataclass(slots=True)
class CostOptionEvaluation:
    decision: CostDecision

    intervention_cost: float

    downtime_cost: float

    production_loss: float

    residual_expected_failure_cost: float

    total_expected_cost: float

    expected_savings: float

    roi_percent: float | None

    benefit_cost_ratio: float | None

    risk_after_action: float

    feasible: bool = True

    reason: str = ""

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)

        data["decision"] = (
            self.decision.value
        )

        return data


@dataclass(slots=True)
class CostRecommendation:
    decision: CostDecision

    priority: RecommendationPriority

    expected_cost: float

    expected_savings: float

    roi_percent: float | None

    reason: str

    due_within_hours: float | None = None

    requires_human_approval: bool = True


# =========================================================
# Output
# =========================================================

@dataclass(slots=True)
class CostAgentResult:
    analysis_id: str

    asset_id: str
    asset_name: str | None
    site_id: str | None

    currency: str

    failure_probability: float

    expected_failure_cost: float

    expected_loss_without_action: float

    failure_cost_breakdown: CostBreakdown

    recommended_decision: CostDecision

    recommendation: CostRecommendation

    option_evaluations: list[
        CostOptionEvaluation
    ]

    gross_avoided_cost: float

    net_avoided_cost: float

    risk_level: CostRiskLevel

    human_review_required: bool

    model_reliability_warning: bool

    summary: str

    created_at: datetime = field(
        default_factory=utc_now
    )

    trace_id: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_id":
                self.analysis_id,

            "asset_id":
                self.asset_id,

            "asset_name":
                self.asset_name,

            "site_id":
                self.site_id,

            "currency":
                self.currency,

            "failure_probability":
                self.failure_probability,

            "expected_failure_cost":
                self.expected_failure_cost,

            "expected_loss_without_action":
                self.expected_loss_without_action,

            "failure_cost_breakdown":
                self.failure_cost_breakdown.to_dict(),

            "recommended_decision":
                self.recommended_decision.value,

            "recommendation": {
                **asdict(
                    self.recommendation
                ),
                "decision":
                    self.recommendation
                    .decision.value,
                "priority":
                    self.recommendation
                    .priority.value,
            },

            "option_evaluations": [
                option.to_dict()
                for option
                in self.option_evaluations
            ],

            "gross_avoided_cost":
                self.gross_avoided_cost,

            "net_avoided_cost":
                self.net_avoided_cost,

            "risk_level":
                self.risk_level.value,

            "human_review_required":
                self.human_review_required,

            "model_reliability_warning":
                self.model_reliability_warning,

            "summary":
                self.summary,

            "created_at":
                self.created_at.isoformat(),

            "trace_id":
                self.trace_id,

            "metadata":
                dict(self.metadata),
        }


# =========================================================
# External integration protocols
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


class CostRepository(Protocol):

    async def save_analysis(
        self,
        result: CostAgentResult,
    ) -> Any:
        ...


# =========================================================
# Configuration
# =========================================================

@dataclass(slots=True)
class CostAgentConfig:
    critical_failure_probability: float = 0.85
    high_failure_probability: float = 0.65

    critical_rul_hours: float = 24.0
    warning_rul_hours: float = 168.0

    minimum_prediction_confidence: float = 0.60

    minimum_repair_roi_percent: float = 10.0

    replacement_cost_advantage_ratio: float = 0.90

    high_cost_threshold: float = 500_000.0

    auto_create_work_order: bool = False

    require_human_review_for_replacement: bool = True


# =========================================================
# Cost Agent
# =========================================================

class CostAgent:

    def __init__(
        self,
        *,
        config: CostAgentConfig | None = None,
        work_order_service: WorkOrderService | None = None,
        repository: CostRepository | None = None,
    ) -> None:

        self.config = (
            config
            or CostAgentConfig()
        )

        self.work_order_service = (
            work_order_service
        )

        self.repository = repository

    # -----------------------------------------------------
    # Main execution
    # -----------------------------------------------------

    async def run(
        self,
        data: CostAgentInput,
    ) -> CostAgentResult:

        with self._trace(data) as span:

            failure_breakdown = (
                self._failure_cost_breakdown(
                    data
                )
            )

            failure_cost = (
                failure_breakdown.total
            )

            expected_loss = (
                data.failure_probability
                * failure_cost
            )

            options = (
                self._evaluate_options(
                    data=data,
                    failure_cost=(
                        failure_cost
                    ),
                    baseline_expected_loss=(
                        expected_loss
                    ),
                )
            )

            selected = (
                self._select_option(
                    data,
                    options,
                )
            )

            risk_level = (
                self._risk_level(
                    data,
                    expected_loss,
                )
            )

            priority = (
                self._priority(
                    data,
                    risk_level,
                )
            )

            gross_avoided_cost = max(
                0.0,
                expected_loss
                - selected
                .residual_expected_failure_cost,
            )

            net_avoided_cost = max(
                0.0,
                expected_loss
                - selected.total_expected_cost,
            )

            reliability_warning = (
                self._model_reliability_warning(
                    data
                )
            )

            human_review = (
                reliability_warning
                or selected.decision
                in {
                    CostDecision.REPLACE,
                    CostDecision.SHUTDOWN,
                }
                or risk_level
                == CostRiskLevel.CRITICAL
            )

            recommendation = (
                CostRecommendation(
                    decision=(
                        selected.decision
                    ),

                    priority=priority,

                    expected_cost=(
                        selected
                        .total_expected_cost
                    ),

                    expected_savings=(
                        selected
                        .expected_savings
                    ),

                    roi_percent=(
                        selected.roi_percent
                    ),

                    reason=(
                        selected.reason
                    ),

                    due_within_hours=(
                        self._due_hours(
                            data,
                            priority,
                        )
                    ),

                    requires_human_approval=(
                        human_review
                    ),
                )
            )

            summary = self._summary(
                data=data,

                expected_loss=(
                    expected_loss
                ),

                selected=selected,

                currency=data.currency,
            )

            result = CostAgentResult(
                analysis_id=new_id(),

                asset_id=data.asset_id,
                asset_name=data.asset_name,
                site_id=data.site_id,

                currency=data.currency,

                failure_probability=(
                    data.failure_probability
                ),

                expected_failure_cost=round(
                    failure_cost,
                    2,
                ),

                expected_loss_without_action=round(
                    expected_loss,
                    2,
                ),

                failure_cost_breakdown=(
                    failure_breakdown
                ),

                recommended_decision=(
                    selected.decision
                ),

                recommendation=(
                    recommendation
                ),

                option_evaluations=(
                    options
                ),

                gross_avoided_cost=round(
                    gross_avoided_cost,
                    2,
                ),

                net_avoided_cost=round(
                    net_avoided_cost,
                    2,
                ),

                risk_level=risk_level,

                human_review_required=(
                    human_review
                ),

                model_reliability_warning=(
                    reliability_warning
                ),

                summary=summary,

                trace_id=data.trace_id,

                metadata={
                    **data.metadata,

                    "prediction_id":
                        data.prediction_id,

                    "work_order_id":
                        data.work_order_id,

                    "request_id":
                        data.request_id,
                },
            )

            if span is not None:
                try:
                    span.set_attributes({
                        "cost.expected_loss":
                            expected_loss,

                        "cost.selected_action":
                            selected
                            .decision.value,

                        "cost.expected_savings":
                            selected
                            .expected_savings,

                        "cost.currency":
                            data.currency,
                    })
                except Exception:
                    pass

            await self._post_process(
                result
            )

            return result

    # -----------------------------------------------------
    # Failure consequence
    # -----------------------------------------------------

    @staticmethod
    def _failure_cost_breakdown(
        data: CostAgentInput,
    ) -> CostBreakdown:

        downtime = (
            data
            .expected_failure_downtime_hours
            * data
            .downtime_cost_per_hour
        )

        production_loss = (
            data
            .expected_failure_downtime_hours
            * data
            .production_units_per_hour
            * data
            .contribution_margin_per_unit
        )

        return CostBreakdown(
            repair=(
                data.direct_failure_cost
            ),

            downtime=downtime,

            production_loss=(
                production_loss
            ),

            scrap=(
                data.expected_scrap_cost
            ),

            quality=(
                data.expected_quality_cost
            ),

            energy=(
                data
                .expected_excess_energy_cost
            ),

            safety=(
                data
                .safety_consequence_cost
            ),

            emergency=(
                data
                .emergency_failure_cost
            ),
        )

    # -----------------------------------------------------
    # Options
    # -----------------------------------------------------

    def _evaluate_options(
        self,
        *,
        data: CostAgentInput,
        failure_cost: float,
        baseline_expected_loss: float,
    ) -> list[CostOptionEvaluation]:

        options: list[
            CostOptionEvaluation
        ] = []

        options.append(
            self._monitor_option(
                data,
                baseline_expected_loss,
            )
        )

        if data.maintenance is not None:
            options.append(
                self._maintenance_option(
                    data,
                    failure_cost,
                    baseline_expected_loss,
                )
            )

        if data.repair is not None:
            options.append(
                self._repair_option(
                    data,
                    failure_cost,
                    baseline_expected_loss,
                )
            )

        if data.replacement is not None:
            options.append(
                self._replacement_option(
                    data,
                    failure_cost,
                    baseline_expected_loss,
                )
            )

        return options

    @staticmethod
    def _monitor_option(
        data: CostAgentInput,
        baseline_expected_loss: float,
    ) -> CostOptionEvaluation:

        return CostOptionEvaluation(
            decision=(
                CostDecision.MONITOR
            ),

            intervention_cost=0.0,
            downtime_cost=0.0,
            production_loss=0.0,

            residual_expected_failure_cost=(
                baseline_expected_loss
            ),

            total_expected_cost=(
                baseline_expected_loss
            ),

            expected_savings=0.0,

            roi_percent=None,

            benefit_cost_ratio=None,

            risk_after_action=(
                data.failure_probability
            ),

            reason=(
                "Continue monitoring without "
                "immediate intervention."
            ),
        )

    def _repair_option(
        self,
        data: CostAgentInput,
        failure_cost: float,
        baseline_expected_loss: float,
    ) -> CostOptionEvaluation:

        option = data.repair

        assert option is not None

        downtime_cost = (
            option.downtime_hours
            * data.downtime_cost_per_hour
        )

        production_loss = (
            option.downtime_hours
            * data.production_units_per_hour
            * data.contribution_margin_per_unit
        )

        intervention = (
            option.repair_cost
            + option.labor_cost
            + option.parts_cost
        )

        residual_failure_cost = (
            option
            .post_repair_failure_probability
            * failure_cost
        )

        total = (
            intervention
            + downtime_cost
            + production_loss
            + residual_failure_cost
        )

        savings = (
            baseline_expected_loss
            - total
        )

        roi = (
            savings / intervention * 100.0
            if intervention > 0
            else None
        )

        bcr = (
            baseline_expected_loss
            - residual_failure_cost
        )

        bcr = (
            bcr / intervention
            if intervention > 0
            else None
        )

        return CostOptionEvaluation(
            decision=CostDecision.REPAIR,

            intervention_cost=(
                intervention
            ),

            downtime_cost=(
                downtime_cost
            ),

            production_loss=(
                production_loss
            ),

            residual_expected_failure_cost=(
                residual_failure_cost
            ),

            total_expected_cost=total,

            expected_savings=savings,

            roi_percent=roi,

            benefit_cost_ratio=bcr,

            risk_after_action=(
                option
                .post_repair_failure_probability
            ),

            reason=(
                "Repair option evaluated using "
                "intervention cost, downtime, "
                "production loss and residual "
                "failure risk."
            ),

            metadata={
                "success_probability":
                    option
                    .success_probability,

                "life_extension_hours":
                    option
                    .life_extension_hours,
            },
        )

    def _replacement_option(
        self,
        data: CostAgentInput,
        failure_cost: float,
        baseline_expected_loss: float,
    ) -> CostOptionEvaluation:

        option = data.replacement

        assert option is not None

        downtime_cost = (
            option.downtime_hours
            * data.downtime_cost_per_hour
        )

        production_loss = (
            option.downtime_hours
            * data.production_units_per_hour
            * data.contribution_margin_per_unit
        )

        intervention = max(
            0.0,
            (
                option.replacement_cost
                + option.installation_cost
                + option.logistics_cost
                - option.salvage_value
            ),
        )

        residual_failure_cost = (
            option
            .post_replacement_failure_probability
            * failure_cost
        )

        total = (
            intervention
            + downtime_cost
            + production_loss
            + residual_failure_cost
        )

        savings = (
            baseline_expected_loss
            - total
        )

        roi = (
            savings / intervention * 100.0
            if intervention > 0
            else None
        )

        bcr = (
            (
                baseline_expected_loss
                - residual_failure_cost
            )
            / intervention
            if intervention > 0
            else None
        )

        return CostOptionEvaluation(
            decision=CostDecision.REPLACE,

            intervention_cost=intervention,

            downtime_cost=downtime_cost,

            production_loss=production_loss,

            residual_expected_failure_cost=(
                residual_failure_cost
            ),

            total_expected_cost=total,

            expected_savings=savings,

            roi_percent=roi,

            benefit_cost_ratio=bcr,

            risk_after_action=(
                option
                .post_replacement_failure_probability
            ),

            reason=(
                "Replacement option evaluated "
                "using purchase, installation, "
                "logistics, salvage value, "
                "downtime and residual risk."
            ),

            metadata={
                "expected_life_hours":
                    option
                    .expected_life_hours,

                "salvage_value":
                    option.salvage_value,
            },
        )

    def _maintenance_option(
        self,
        data: CostAgentInput,
        failure_cost: float,
        baseline_expected_loss: float,
    ) -> CostOptionEvaluation:

        option = data.maintenance

        assert option is not None

        downtime_cost = (
            option.downtime_hours
            * data.downtime_cost_per_hour
        )

        production_loss = (
            option.downtime_hours
            * data.production_units_per_hour
            * data.contribution_margin_per_unit
        )

        intervention = (
            option.maintenance_cost
        )

        residual_failure_cost = (
            option
            .post_maintenance_failure_probability
            * failure_cost
        )

        total = (
            intervention
            + downtime_cost
            + production_loss
            + residual_failure_cost
        )

        savings = (
            baseline_expected_loss
            - total
        )

        roi = (
            savings / intervention * 100.0
            if intervention > 0
            else None
        )

        bcr = (
            (
                baseline_expected_loss
                - residual_failure_cost
            )
            / intervention
            if intervention > 0
            else None
        )

        return CostOptionEvaluation(
            decision=(
                CostDecision
                .PREDICTIVE_MAINTENANCE
            ),

            intervention_cost=intervention,

            downtime_cost=downtime_cost,

            production_loss=production_loss,

            residual_expected_failure_cost=(
                residual_failure_cost
            ),

            total_expected_cost=total,

            expected_savings=savings,

            roi_percent=roi,

            benefit_cost_ratio=bcr,

            risk_after_action=(
                option
                .post_maintenance_failure_probability
            ),

            reason=(
                "Predictive maintenance option "
                "evaluated against expected "
                "failure consequence."
            ),

            metadata={
                "life_extension_hours":
                    option
                    .life_extension_hours,
            },
        )

    # -----------------------------------------------------
    # Decision selection
    # -----------------------------------------------------

    def _select_option(
        self,
        data: CostAgentInput,
        options: Sequence[
            CostOptionEvaluation
        ],
    ) -> CostOptionEvaluation:

        feasible = [
            option
            for option in options
            if option.feasible
        ]

        if not feasible:
            raise RuntimeError(
                "No feasible cost option."
            )

        # Safety-critical / imminent failure:
        # do not choose "monitor" purely because
        # it looks cheaper.
        imminent_failure = (
            data.failure_probability
            >= self.config
            .critical_failure_probability
            or (
                data.predicted_rul_hours
                is not None
                and data.predicted_rul_hours
                <= self.config
                .critical_rul_hours
            )
        )

        if imminent_failure:
            intervention_options = [
                option
                for option in feasible
                if option.decision
                != CostDecision.MONITOR
            ]

            if intervention_options:
                return min(
                    intervention_options,
                    key=lambda option: (
                        option
                        .total_expected_cost,
                        option
                        .risk_after_action,
                    ),
                )

        return min(
            feasible,
            key=lambda option: (
                option.total_expected_cost,
                option.risk_after_action,
            ),
        )

    # -----------------------------------------------------
    # Risk
    # -----------------------------------------------------

    def _risk_level(
        self,
        data: CostAgentInput,
        expected_loss: float,
    ) -> CostRiskLevel:

        if (
            data.failure_probability
            >= self.config
            .critical_failure_probability
            or expected_loss
            >= self.config
            .high_cost_threshold
        ):
            return CostRiskLevel.CRITICAL

        if (
            data.failure_probability
            >= self.config
            .high_failure_probability
        ):
            return CostRiskLevel.HIGH

        if data.failure_probability >= 0.30:
            return CostRiskLevel.MODERATE

        return CostRiskLevel.LOW

    @staticmethod
    def _priority(
        data: CostAgentInput,
        risk: CostRiskLevel,
    ) -> RecommendationPriority:

        if risk == CostRiskLevel.CRITICAL:
            return (
                RecommendationPriority.CRITICAL
            )

        if risk == CostRiskLevel.HIGH:
            return RecommendationPriority.HIGH

        if risk == CostRiskLevel.MODERATE:
            return (
                RecommendationPriority.MEDIUM
            )

        return RecommendationPriority.LOW

    # -----------------------------------------------------
    # Reliability
    # -----------------------------------------------------

    def _model_reliability_warning(
        self,
        data: CostAgentInput,
    ) -> bool:

        if data.model_drift_detected:
            return True

        if (
            data.prediction_confidence
            is not None
            and data.prediction_confidence
            < self.config
            .minimum_prediction_confidence
        ):
            return True

        return False

    # -----------------------------------------------------
    # Due time
    # -----------------------------------------------------

    def _due_hours(
        self,
        data: CostAgentInput,
        priority: RecommendationPriority,
    ) -> float | None:

        if (
            data.predicted_rul_hours
            is not None
            and data.predicted_rul_hours
            <= self.config
            .critical_rul_hours
        ):
            return max(
                0.0,
                data.predicted_rul_hours,
            )

        return {
            RecommendationPriority.CRITICAL: 4.0,
            RecommendationPriority.HIGH: 24.0,
            RecommendationPriority.MEDIUM: 72.0,
            RecommendationPriority.LOW: None,
        }[priority]

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    @staticmethod
    def _summary(
        *,
        data: CostAgentInput,
        expected_loss: float,
        selected: CostOptionEvaluation,
        currency: str,
    ) -> str:

        asset = (
            data.asset_name
            or data.asset_id
        )

        return (
            f"{asset}: expected failure loss is "
            f"{currency} {expected_loss:,.2f}. "
            f"Recommended action is "
            f"{selected.decision.value} with "
            f"expected total cost "
            f"{currency} "
            f"{selected.total_expected_cost:,.2f} "
            f"and estimated savings "
            f"{currency} "
            f"{selected.expected_savings:,.2f}."
        )

    # -----------------------------------------------------
    # Post processing
    # -----------------------------------------------------

    async def _post_process(
        self,
        result: CostAgentResult,
    ) -> None:

        self._publish_metrics(result)
        self._log_result(result)

        if self.repository is not None:
            try:
                await self.repository.save_analysis(
                    result
                )
            except Exception:
                pass

        if (
            self.config.auto_create_work_order
            and self.work_order_service
            and result.recommended_decision
            != CostDecision.MONITOR
        ):
            await self._create_work_order(
                result
            )

    async def _create_work_order(
        self,
        result: CostAgentResult,
    ) -> None:

        if self.work_order_service is None:
            return

        try:
            await (
                self.work_order_service
                .create_work_order(
                    asset_id=result.asset_id,

                    title=(
                        "AI Cost-Optimized "
                        "Maintenance Action"
                    ),

                    description=(
                        result.summary
                    ),

                    priority=(
                        result.recommendation
                        .priority.value
                    ),

                    metadata={
                        "analysis_id":
                            result.analysis_id,

                        "recommended_decision":
                            result
                            .recommended_decision
                            .value,

                        "expected_failure_cost":
                            result
                            .expected_failure_cost,

                        "expected_loss":
                            result
                            .expected_loss_without_action,

                        "expected_savings":
                            result
                            .recommendation
                            .expected_savings,

                        "currency":
                            result.currency,
                    },
                )
            )

        except Exception:
            return

    # -----------------------------------------------------
    # Observability
    # -----------------------------------------------------

    @staticmethod
    def _publish_metrics(
        result: CostAgentResult,
    ) -> None:

        try:
            from app.observability.metrics import (
                registry,
            )

            registry.counter(
                "cost_agent_analyses_total"
            ).inc(
                labels={
                    "decision":
                        result
                        .recommended_decision
                        .value,

                    "risk":
                        result
                        .risk_level.value,
                }
            )

            registry.histogram(
                "cost_agent_expected_loss"
            ).observe(
                result
                .expected_loss_without_action,

                labels={
                    "currency":
                        result.currency,
                },
            )

            registry.histogram(
                "cost_agent_avoided_cost"
            ).observe(
                result.net_avoided_cost,

                labels={
                    "currency":
                        result.currency,
                },
            )

        except Exception:
            return

    @staticmethod
    def _log_result(
        result: CostAgentResult,
    ) -> None:

        try:
            from app.observability.logging import (
                get_logger,
            )

            logger = get_logger(
                "industrial.cost_agent"
            )

            logger.info(
                "Cost agent analysis completed.",
                extra={
                    "cost_analysis": {
                        "analysis_id":
                            result.analysis_id,

                        "asset_id":
                            result.asset_id,

                        "decision":
                            result
                            .recommended_decision
                            .value,

                        "expected_loss":
                            result
                            .expected_loss_without_action,

                        "net_avoided_cost":
                            result
                            .net_avoided_cost,

                        "currency":
                            result.currency,
                    }
                },
            )

        except Exception:
            return

    # -----------------------------------------------------
    # Tracing
    # -----------------------------------------------------

    @staticmethod
    def _trace(
        data: CostAgentInput,
    ) -> Any:

        try:
            from app.observability.tracing import (
                TraceComponent,
                tracer,
            )

            return tracer.start_span(
                "agent.cost_analysis",

                component=(
                    TraceComponent.COST
                ),

                attributes={
                    "agent.name":
                        "cost_agent",

                    "industrial.asset_id":
                        data.asset_id,

                    "industrial.site_id":
                        data.site_id,

                    "cost.currency":
                        data.currency,
                },
            )

        except Exception:
            return _NullSpan()


# =========================================================
# Null tracing
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
# Factory + global
# =========================================================

def create_cost_agent(
    *,
    config: CostAgentConfig | None = None,
    work_order_service: WorkOrderService | None = None,
    repository: CostRepository | None = None,
) -> CostAgent:

    return CostAgent(
        config=config,

        work_order_service=(
            work_order_service
        ),

        repository=repository,
    )


cost_agent = CostAgent()


__all__ = [
    "CostDecision",
    "CostRiskLevel",
    "RecommendationPriority",
    "CostCategory",

    "CostBreakdown",
    "RepairOption",
    "ReplacementOption",
    "MaintenanceOption",

    "CostAgentInput",

    "CostOptionEvaluation",
    "CostRecommendation",
    "CostAgentResult",

    "WorkOrderService",
    "CostRepository",

    "CostAgentConfig",
    "CostAgent",

    "create_cost_agent",
    "cost_agent",
]