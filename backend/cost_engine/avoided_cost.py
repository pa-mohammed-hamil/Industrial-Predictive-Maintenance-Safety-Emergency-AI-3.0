# app/agents/tools/avoided_cost.py

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence
from uuid import UUID, uuid4


# =========================================================
# Utilities
# =========================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid4())


def finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    if not math.isfinite(result):
        return default

    return result


def non_negative(value: Any, default: float = 0.0) -> float:
    return max(0.0, finite(value, default))


def clamp(
    value: float,
    low: float = 0.0,
    high: float = 1.0,
) -> float:
    return max(low, min(high, finite(value)))


def safe_divide(
    numerator: float,
    denominator: float,
    default: float = 0.0,
) -> float:
    if abs(denominator) <= 1e-12:
        return default

    return numerator / denominator


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

    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


# =========================================================
# Enums
# =========================================================

class CostCategory(str, Enum):
    REPAIR = "repair"
    REPLACEMENT = "replacement"
    LABOR = "labor"
    PARTS = "parts"
    SPARES = "spares"

    DOWNTIME = "downtime"
    PRODUCTION_LOSS = "production_loss"

    SCRAP = "scrap"
    QUALITY = "quality"

    ENERGY = "energy"

    SAFETY = "safety"
    EMERGENCY = "emergency"

    ENVIRONMENTAL = "environmental"
    PENALTY = "penalty"

    LOGISTICS = "logistics"
    CONTRACTOR = "contractor"

    WARRANTY = "warranty"

    OTHER = "other"


class AvoidedCostType(str, Enum):
    FAILURE_PREVENTION = "failure_prevention"
    DOWNTIME_REDUCTION = "downtime_reduction"
    PRODUCTION_RECOVERY = "production_recovery"

    REPAIR_AVOIDANCE = "repair_avoidance"
    REPLACEMENT_AVOIDANCE = "replacement_avoidance"

    SCRAP_REDUCTION = "scrap_reduction"
    ENERGY_SAVINGS = "energy_savings"

    SAFETY_LOSS_AVOIDANCE = "safety_loss_avoidance"
    EMERGENCY_AVOIDANCE = "emergency_avoidance"

    MAINTENANCE_OPTIMIZATION = "maintenance_optimization"

    TOTAL = "total"


class InterventionType(str, Enum):
    MONITORING = "monitoring"
    INSPECTION = "inspection"

    PREVENTIVE_MAINTENANCE = "preventive_maintenance"
    PREDICTIVE_MAINTENANCE = "predictive_maintenance"

    REPAIR = "repair"
    OVERHAUL = "overhaul"
    REPLACEMENT = "replacement"

    LOAD_REDUCTION = "load_reduction"
    PROCESS_CHANGE = "process_change"

    SAFETY_INTERVENTION = "safety_intervention"

    OTHER = "other"


class AvoidedCostConfidence(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class CostBasis(str, Enum):
    COMPONENTIZED = "componentized"
    ALL_IN = "all_in"


# =========================================================
# Cost component
# =========================================================

@dataclass(slots=True)
class CostComponent:
    category: CostCategory
    amount: float

    description: str | None = None

    confidence: float = 1.0

    source: str | None = None

    timestamp: datetime | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.amount = non_negative(self.amount)
        self.confidence = clamp(self.confidence)


# =========================================================
# Failure consequence
# =========================================================

@dataclass(slots=True)
class FailureConsequence:
    repair_cost: float = 0.0
    replacement_cost: float = 0.0

    labor_cost: float = 0.0
    parts_cost: float = 0.0
    spares_cost: float = 0.0

    downtime_cost: float = 0.0
    production_loss: float = 0.0

    scrap_cost: float = 0.0
    quality_loss: float = 0.0

    energy_loss: float = 0.0

    safety_loss: float = 0.0
    emergency_cost: float = 0.0

    environmental_cost: float = 0.0
    penalty_cost: float = 0.0

    logistics_cost: float = 0.0
    contractor_cost: float = 0.0

    other_cost: float = 0.0

    cost_basis: CostBasis = CostBasis.COMPONENTIZED

    all_in_failure_cost: float | None = None

    currency: str = "INR"

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        for name in (
            "repair_cost",
            "replacement_cost",
            "labor_cost",
            "parts_cost",
            "spares_cost",
            "downtime_cost",
            "production_loss",
            "scrap_cost",
            "quality_loss",
            "energy_loss",
            "safety_loss",
            "emergency_cost",
            "environmental_cost",
            "penalty_cost",
            "logistics_cost",
            "contractor_cost",
            "other_cost",
        ):
            setattr(
                self,
                name,
                non_negative(getattr(self, name)),
            )

        if self.all_in_failure_cost is not None:
            self.all_in_failure_cost = non_negative(
                self.all_in_failure_cost
            )

    @property
    def component_total(self) -> float:
        return sum(
            (
                self.repair_cost,
                self.replacement_cost,
                self.labor_cost,
                self.parts_cost,
                self.spares_cost,
                self.downtime_cost,
                self.production_loss,
                self.scrap_cost,
                self.quality_loss,
                self.energy_loss,
                self.safety_loss,
                self.emergency_cost,
                self.environmental_cost,
                self.penalty_cost,
                self.logistics_cost,
                self.contractor_cost,
                self.other_cost,
            )
        )

    @property
    def total(self) -> float:
        if (
            self.cost_basis == CostBasis.ALL_IN
            and self.all_in_failure_cost is not None
        ):
            return self.all_in_failure_cost

        return self.component_total


# =========================================================
# Baseline / intervention
# =========================================================

@dataclass(slots=True)
class FailureRisk:
    probability: float

    consequence: FailureConsequence

    horizon_hours: float | None = None

    confidence: float = 1.0

    def __post_init__(self) -> None:
        self.probability = clamp(self.probability)
        self.confidence = clamp(self.confidence)

        if self.horizon_hours is not None:
            self.horizon_hours = non_negative(
                self.horizon_hours
            )


@dataclass(slots=True)
class Intervention:
    intervention_type: InterventionType

    intervention_cost: float

    post_intervention_failure_probability: float

    success_probability: float = 1.0

    expected_downtime_cost: float = 0.0
    expected_production_loss: float = 0.0

    additional_operating_cost: float = 0.0

    expected_direct_savings: float = 0.0

    description: str | None = None

    confidence: float = 1.0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.intervention_cost = non_negative(
            self.intervention_cost
        )

        self.post_intervention_failure_probability = clamp(
            self.post_intervention_failure_probability
        )

        self.success_probability = clamp(
            self.success_probability
        )

        self.expected_downtime_cost = non_negative(
            self.expected_downtime_cost
        )

        self.expected_production_loss = non_negative(
            self.expected_production_loss
        )

        self.additional_operating_cost = non_negative(
            self.additional_operating_cost
        )

        self.expected_direct_savings = non_negative(
            self.expected_direct_savings
        )

        self.confidence = clamp(self.confidence)


# =========================================================
# Breakdown
# =========================================================

@dataclass(slots=True)
class AvoidedCostBreakdown:
    repair: float = 0.0
    replacement: float = 0.0

    labor: float = 0.0
    parts: float = 0.0
    spares: float = 0.0

    downtime: float = 0.0
    production_loss: float = 0.0

    scrap: float = 0.0
    quality: float = 0.0

    energy: float = 0.0

    safety: float = 0.0
    emergency: float = 0.0

    environmental: float = 0.0
    penalties: float = 0.0

    logistics: float = 0.0
    contractor: float = 0.0

    other: float = 0.0

    @property
    def total(self) -> float:
        return sum(
            (
                self.repair,
                self.replacement,
                self.labor,
                self.parts,
                self.spares,
                self.downtime,
                self.production_loss,
                self.scrap,
                self.quality,
                self.energy,
                self.safety,
                self.emergency,
                self.environmental,
                self.penalties,
                self.logistics,
                self.contractor,
                self.other,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        data = json_safe(self)
        data["total"] = round(self.total, 2)
        return data


# =========================================================
# Result
# =========================================================

@dataclass(slots=True)
class AvoidedCostResult:
    calculation_id: str

    asset_id: str | None

    baseline_failure_probability: float

    post_intervention_failure_probability: float

    effective_post_intervention_probability: float

    baseline_expected_loss: float

    post_intervention_expected_loss: float

    gross_avoided_failure_cost: float

    direct_savings: float

    gross_avoided_cost: float

    intervention_cost: float

    intervention_downtime_cost: float

    intervention_production_loss: float

    additional_operating_cost: float

    total_intervention_cost: float

    net_avoided_cost: float

    benefit_cost_ratio: float | None

    roi_percent: float | None

    avoided_cost_percent: float

    risk_reduction_absolute: float

    risk_reduction_percent: float

    breakdown: AvoidedCostBreakdown

    confidence: float

    confidence_level: AvoidedCostConfidence

    currency: str

    economically_positive: bool

    warnings: list[str]

    generated_at: datetime = field(
        default_factory=utc_now
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# =========================================================
# Scenario
# =========================================================

@dataclass(slots=True)
class AvoidedCostScenario:
    name: str

    baseline_probability: float
    post_probability: float

    consequence_multiplier: float = 1.0

    probability_weight: float = 1.0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.baseline_probability = clamp(
            self.baseline_probability
        )

        self.post_probability = clamp(
            self.post_probability
        )

        self.consequence_multiplier = non_negative(
            self.consequence_multiplier
        )

        self.probability_weight = non_negative(
            self.probability_weight
        )


@dataclass(slots=True)
class AvoidedCostScenarioResult:
    scenario_name: str

    baseline_expected_loss: float

    post_expected_loss: float

    avoided_cost: float

    probability_weight: float

    weighted_avoided_cost: float

    currency: str


# =========================================================
# Config
# =========================================================

@dataclass(slots=True)
class AvoidedCostConfig:
    minimum_confidence: float = 0.50

    meaningful_savings_ratio: float = 0.05

    high_savings_ratio: float = 0.25

    warn_on_componentized_double_count: bool = True

    default_currency: str = "INR"


# =========================================================
# Engine
# =========================================================

class AvoidedCostCalculator:

    def __init__(
        self,
        *,
        config: AvoidedCostConfig | None = None,
    ) -> None:
        self.config = (
            config or AvoidedCostConfig()
        )

    # =====================================================
    # Expected loss
    # =====================================================

    @staticmethod
    def expected_loss(
        *,
        failure_probability: float,
        consequence_cost: float,
    ) -> float:
        return (
            clamp(failure_probability)
            * non_negative(consequence_cost)
        )

    # =====================================================
    # Effective post-intervention risk
    # =====================================================

    @staticmethod
    def effective_post_probability(
        *,
        baseline_probability: float,
        target_post_probability: float,
        intervention_success_probability: float,
    ) -> float:
        baseline = clamp(
            baseline_probability
        )

        target = clamp(
            target_post_probability
        )

        success = clamp(
            intervention_success_probability
        )

        # If the intervention succeeds, use target risk.
        # If it fails, baseline risk remains.
        return clamp(
            success * target
            + (1.0 - success) * baseline
        )

    # =====================================================
    # Category expected avoided cost
    # =====================================================

    @staticmethod
    def category_avoided_cost(
        *,
        baseline_probability: float,
        post_probability: float,
        category_cost: float,
    ) -> float:
        risk_reduction = max(
            0.0,
            clamp(baseline_probability)
            - clamp(post_probability),
        )

        return (
            risk_reduction
            * non_negative(category_cost)
        )

    # =====================================================
    # Breakdown
    # =====================================================

    def calculate_breakdown(
        self,
        *,
        consequence: FailureConsequence,
        baseline_probability: float,
        post_probability: float,
    ) -> AvoidedCostBreakdown:

        if consequence.cost_basis == CostBasis.ALL_IN:
            # Do not distribute an all-in value across
            # component categories because doing so would
            # create artificial precision/double counting.
            return AvoidedCostBreakdown(
                other=self.category_avoided_cost(
                    baseline_probability=(
                        baseline_probability
                    ),
                    post_probability=post_probability,
                    category_cost=consequence.total,
                )
            )

        def avoided(amount: float) -> float:
            return self.category_avoided_cost(
                baseline_probability=(
                    baseline_probability
                ),
                post_probability=post_probability,
                category_cost=amount,
            )

        return AvoidedCostBreakdown(
            repair=avoided(
                consequence.repair_cost
            ),

            replacement=avoided(
                consequence.replacement_cost
            ),

            labor=avoided(
                consequence.labor_cost
            ),

            parts=avoided(
                consequence.parts_cost
            ),

            spares=avoided(
                consequence.spares_cost
            ),

            downtime=avoided(
                consequence.downtime_cost
            ),

            production_loss=avoided(
                consequence.production_loss
            ),

            scrap=avoided(
                consequence.scrap_cost
            ),

            quality=avoided(
                consequence.quality_loss
            ),

            energy=avoided(
                consequence.energy_loss
            ),

            safety=avoided(
                consequence.safety_loss
            ),

            emergency=avoided(
                consequence.emergency_cost
            ),

            environmental=avoided(
                consequence.environmental_cost
            ),

            penalties=avoided(
                consequence.penalty_cost
            ),

            logistics=avoided(
                consequence.logistics_cost
            ),

            contractor=avoided(
                consequence.contractor_cost
            ),

            other=avoided(
                consequence.other_cost
            ),
        )

    # =====================================================
    # Confidence
    # =====================================================

    @staticmethod
    def confidence(
        *,
        baseline_confidence: float,
        intervention_confidence: float,
        consequence: FailureConsequence,
    ) -> tuple[
        float,
        AvoidedCostConfidence,
    ]:

        base = clamp(
            baseline_confidence
        )

        intervention = clamp(
            intervention_confidence
        )

        data_coverage = 0.0

        if consequence.total > 0:
            data_coverage += 0.40

        if consequence.downtime_cost > 0:
            data_coverage += 0.15

        if consequence.production_loss > 0:
            data_coverage += 0.15

        if (
            consequence.repair_cost > 0
            or consequence.replacement_cost > 0
        ):
            data_coverage += 0.15

        if (
            consequence.safety_loss > 0
            or consequence.emergency_cost > 0
        ):
            data_coverage += 0.15

        score = clamp(
            0.40 * base
            + 0.35 * intervention
            + 0.25 * clamp(data_coverage)
        )

        if score >= 0.90:
            level = (
                AvoidedCostConfidence.VERY_HIGH
            )

        elif score >= 0.75:
            level = AvoidedCostConfidence.HIGH

        elif score >= 0.50:
            level = (
                AvoidedCostConfidence.MODERATE
            )

        else:
            level = AvoidedCostConfidence.LOW

        return round(score, 4), level

    # =====================================================
    # Main calculation
    # =====================================================

    def calculate(
        self,
        *,
        baseline: FailureRisk,
        intervention: Intervention,
        asset_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> AvoidedCostResult:

        warnings: list[str] = []

        baseline_probability = (
            baseline.probability
        )

        effective_post_probability = (
            self.effective_post_probability(
                baseline_probability=(
                    baseline_probability
                ),

                target_post_probability=(
                    intervention
                    .post_intervention_failure_probability
                ),

                intervention_success_probability=(
                    intervention.success_probability
                ),
            )
        )

        consequence_total = (
            baseline.consequence.total
        )

        baseline_expected_loss = (
            self.expected_loss(
                failure_probability=(
                    baseline_probability
                ),
                consequence_cost=(
                    consequence_total
                ),
            )
        )

        post_expected_loss = (
            self.expected_loss(
                failure_probability=(
                    effective_post_probability
                ),
                consequence_cost=(
                    consequence_total
                ),
            )
        )

        gross_avoided_failure_cost = max(
            0.0,
            baseline_expected_loss
            - post_expected_loss,
        )

        breakdown = (
            self.calculate_breakdown(
                consequence=baseline.consequence,

                baseline_probability=(
                    baseline_probability
                ),

                post_probability=(
                    effective_post_probability
                ),
            )
        )

        direct_savings = (
            intervention.expected_direct_savings
        )

        gross_avoided_cost = (
            gross_avoided_failure_cost
            + direct_savings
        )

        total_intervention_cost = sum(
            (
                intervention.intervention_cost,

                intervention
                .expected_downtime_cost,

                intervention
                .expected_production_loss,

                intervention
                .additional_operating_cost,
            )
        )

        net_avoided_cost = (
            gross_avoided_cost
            - total_intervention_cost
        )

        benefit_cost_ratio = None
        roi_percent = None

        if total_intervention_cost > 0:
            benefit_cost_ratio = (
                gross_avoided_cost
                / total_intervention_cost
            )

            roi_percent = (
                net_avoided_cost
                / total_intervention_cost
                * 100.0
            )

        avoided_cost_percent = (
            safe_divide(
                gross_avoided_failure_cost,
                baseline_expected_loss,
                default=0.0,
            )
            * 100.0
        )

        absolute_risk_reduction = max(
            0.0,
            baseline_probability
            - effective_post_probability,
        )

        risk_reduction_percent = (
            safe_divide(
                absolute_risk_reduction,
                baseline_probability,
                default=0.0,
            )
            * 100.0
        )

        confidence, confidence_level = (
            self.confidence(
                baseline_confidence=(
                    baseline.confidence
                ),

                intervention_confidence=(
                    intervention.confidence
                ),

                consequence=(
                    baseline.consequence
                ),
            )
        )

        if consequence_total <= 0:
            warnings.append(
                "Failure consequence cost is zero or "
                "unavailable; avoided-cost output is "
                "not economically informative."
            )

        if (
            confidence
            < self.config.minimum_confidence
        ):
            warnings.append(
                "Avoided-cost estimate has low "
                "confidence and should be reviewed."
            )

        if (
            effective_post_probability
            >= baseline_probability
        ):
            warnings.append(
                "The intervention does not reduce "
                "effective failure probability."
            )

        if net_avoided_cost < 0:
            warnings.append(
                "Expected intervention cost exceeds "
                "the estimated avoided cost."
            )

        if (
            baseline.consequence.cost_basis
            == CostBasis.COMPONENTIZED
            and self.config
            .warn_on_componentized_double_count
        ):
            if (
                baseline.consequence.repair_cost > 0
                and (
                    baseline.consequence.labor_cost > 0
                    or baseline.consequence.parts_cost > 0
                )
            ):
                warnings.append(
                    "Verify that repair_cost does not "
                    "already include labor or parts; "
                    "otherwise failure consequence may "
                    "be double counted."
                )

            if (
                baseline.consequence.downtime_cost > 0
                and baseline.consequence
                .production_loss > 0
            ):
                warnings.append(
                    "Downtime cost and production loss "
                    "are both included. Confirm that "
                    "downtime_cost represents only "
                    "direct downtime overhead and does "
                    "not already include lost production."
                )

        return AvoidedCostResult(
            calculation_id=new_id(),

            asset_id=asset_id,

            baseline_failure_probability=round(
                baseline_probability,
                4,
            ),

            post_intervention_failure_probability=round(
                intervention
                .post_intervention_failure_probability,
                4,
            ),

            effective_post_intervention_probability=round(
                effective_post_probability,
                4,
            ),

            baseline_expected_loss=round(
                baseline_expected_loss,
                2,
            ),

            post_intervention_expected_loss=round(
                post_expected_loss,
                2,
            ),

            gross_avoided_failure_cost=round(
                gross_avoided_failure_cost,
                2,
            ),

            direct_savings=round(
                direct_savings,
                2,
            ),

            gross_avoided_cost=round(
                gross_avoided_cost,
                2,
            ),

            intervention_cost=round(
                intervention.intervention_cost,
                2,
            ),

            intervention_downtime_cost=round(
                intervention
                .expected_downtime_cost,
                2,
            ),

            intervention_production_loss=round(
                intervention
                .expected_production_loss,
                2,
            ),

            additional_operating_cost=round(
                intervention
                .additional_operating_cost,
                2,
            ),

            total_intervention_cost=round(
                total_intervention_cost,
                2,
            ),

            net_avoided_cost=round(
                net_avoided_cost,
                2,
            ),

            benefit_cost_ratio=(
                round(benefit_cost_ratio, 4)
                if benefit_cost_ratio is not None
                else None
            ),

            roi_percent=(
                round(roi_percent, 2)
                if roi_percent is not None
                else None
            ),

            avoided_cost_percent=round(
                avoided_cost_percent,
                2,
            ),

            risk_reduction_absolute=round(
                absolute_risk_reduction,
                4,
            ),

            risk_reduction_percent=round(
                risk_reduction_percent,
                2,
            ),

            breakdown=breakdown,

            confidence=confidence,

            confidence_level=confidence_level,

            currency=(
                baseline.consequence.currency
                or self.config.default_currency
            ),

            economically_positive=(
                net_avoided_cost > 0
            ),

            warnings=warnings,

            metadata=dict(metadata or {}),
        )

    # =====================================================
    # Scenario analysis
    # =====================================================

    def scenario_analysis(
        self,
        *,
        consequence: FailureConsequence,
        scenarios: Sequence[
            AvoidedCostScenario
        ],
    ) -> list[AvoidedCostScenarioResult]:

        results: list[
            AvoidedCostScenarioResult
        ] = []

        for scenario in scenarios:
            consequence_cost = (
                consequence.total
                * scenario.consequence_multiplier
            )

            baseline_loss = (
                self.expected_loss(
                    failure_probability=(
                        scenario
                        .baseline_probability
                    ),
                    consequence_cost=(
                        consequence_cost
                    ),
                )
            )

            post_loss = (
                self.expected_loss(
                    failure_probability=(
                        scenario.post_probability
                    ),
                    consequence_cost=(
                        consequence_cost
                    ),
                )
            )

            avoided = max(
                0.0,
                baseline_loss - post_loss,
            )

            weighted = (
                avoided
                * scenario.probability_weight
            )

            results.append(
                AvoidedCostScenarioResult(
                    scenario_name=(
                        scenario.name
                    ),

                    baseline_expected_loss=round(
                        baseline_loss,
                        2,
                    ),

                    post_expected_loss=round(
                        post_loss,
                        2,
                    ),

                    avoided_cost=round(
                        avoided,
                        2,
                    ),

                    probability_weight=round(
                        scenario
                        .probability_weight,
                        4,
                    ),

                    weighted_avoided_cost=round(
                        weighted,
                        2,
                    ),

                    currency=(
                        consequence.currency
                    ),
                )
            )

        return results

    # =====================================================
    # Weighted scenarios
    # =====================================================

    def weighted_scenario_avoided_cost(
        self,
        *,
        consequence: FailureConsequence,
        scenarios: Sequence[
            AvoidedCostScenario
        ],
    ) -> float:

        results = self.scenario_analysis(
            consequence=consequence,
            scenarios=scenarios,
        )

        total_weight = sum(
            result.probability_weight
            for result in results
        )

        if total_weight <= 0:
            return 0.0

        return round(
            sum(
                result.weighted_avoided_cost
                for result in results
            )
            / total_weight,
            2,
        )

    # =====================================================
    # Break-even intervention cost
    # =====================================================

    def break_even_intervention_cost(
        self,
        *,
        baseline: FailureRisk,
        post_probability: float,
        success_probability: float = 1.0,
        direct_savings: float = 0.0,
    ) -> float:

        effective_post = (
            self.effective_post_probability(
                baseline_probability=(
                    baseline.probability
                ),
                target_post_probability=(
                    post_probability
                ),
                intervention_success_probability=(
                    success_probability
                ),
            )
        )

        baseline_loss = self.expected_loss(
            failure_probability=(
                baseline.probability
            ),
            consequence_cost=(
                baseline.consequence.total
            ),
        )

        post_loss = self.expected_loss(
            failure_probability=effective_post,
            consequence_cost=(
                baseline.consequence.total
            ),
        )

        return round(
            max(
                0.0,
                baseline_loss
                - post_loss
                + non_negative(direct_savings),
            ),
            2,
        )

    # =====================================================
    # Required risk reduction
    # =====================================================

    @staticmethod
    def required_risk_reduction(
        *,
        intervention_cost: float,
        consequence_cost: float,
    ) -> float | None:

        intervention_cost = non_negative(
            intervention_cost
        )

        consequence_cost = non_negative(
            consequence_cost
        )

        if consequence_cost <= 0:
            return None

        return clamp(
            intervention_cost
            / consequence_cost
        )


# =========================================================
# Default instance
# =========================================================

avoided_cost_calculator = (
    AvoidedCostCalculator()
)


# =========================================================
# Agent-facing helpers
# =========================================================

def calculate_avoided_cost(
    *,
    baseline: FailureRisk,
    intervention: Intervention,
    asset_id: str | None = None,
) -> AvoidedCostResult:
    return avoided_cost_calculator.calculate(
        baseline=baseline,
        intervention=intervention,
        asset_id=asset_id,
    )


def calculate_expected_failure_loss(
    *,
    failure_probability: float,
    consequence: FailureConsequence,
) -> float:
    return round(
        avoided_cost_calculator.expected_loss(
            failure_probability=(
                failure_probability
            ),
            consequence_cost=(
                consequence.total
            ),
        ),
        2,
    )


def calculate_break_even_cost(
    *,
    baseline: FailureRisk,
    post_probability: float,
    success_probability: float = 1.0,
) -> float:
    return (
        avoided_cost_calculator
        .break_even_intervention_cost(
            baseline=baseline,
            post_probability=post_probability,
            success_probability=(
                success_probability
            ),
        )
    )


def calculate_risk_reduction_value(
    *,
    baseline_probability: float,
    post_probability: float,
    failure_cost: float,
) -> float:
    return round(
        avoided_cost_calculator
        .category_avoided_cost(
            baseline_probability=(
                baseline_probability
            ),
            post_probability=(
                post_probability
            ),
            category_cost=failure_cost,
        ),
        2,
    )


# =========================================================
# Exports
# =========================================================

__all__ = [
    "CostCategory",
    "AvoidedCostType",
    "InterventionType",
    "AvoidedCostConfidence",
    "CostBasis",

    "CostComponent",
    "FailureConsequence",
    "FailureRisk",
    "Intervention",

    "AvoidedCostBreakdown",
    "AvoidedCostResult",

    "AvoidedCostScenario",
    "AvoidedCostScenarioResult",

    "AvoidedCostConfig",
    "AvoidedCostCalculator",

    "avoided_cost_calculator",

    "calculate_avoided_cost",
    "calculate_expected_failure_loss",
    "calculate_break_even_cost",
    "calculate_risk_reduction_value",
]