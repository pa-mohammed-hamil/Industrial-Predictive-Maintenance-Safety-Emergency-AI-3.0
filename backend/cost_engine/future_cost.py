# app/agents/tools/future_cost.py

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
    return result if math.isfinite(result) else default


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

class FutureCostCategory(str, Enum):
    MAINTENANCE = "maintenance"
    REPAIR = "repair"
    OVERHAUL = "overhaul"
    REPLACEMENT = "replacement"

    FAILURE = "failure"
    DOWNTIME = "downtime"
    PRODUCTION_LOSS = "production_loss"

    ENERGY = "energy"
    SPARES = "spares"
    LABOR = "labor"

    INSPECTION = "inspection"
    CONTRACTOR = "contractor"

    SAFETY = "safety"
    EMERGENCY = "emergency"
    ENVIRONMENTAL = "environmental"

    REGULATORY = "regulatory"
    SOFTWARE = "software"
    LICENSING = "licensing"

    OTHER = "other"


class CostTiming(str, Enum):
    ONE_TIME = "one_time"
    ANNUAL = "annual"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    EVENT_BASED = "event_based"


class ForecastScenario(str, Enum):
    OPTIMISTIC = "optimistic"
    BASE = "base"
    PESSIMISTIC = "pessimistic"
    STRESS = "stress"


class FutureEconomicRisk(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    EXTREME = "extreme"


class FutureCostDecision(str, Enum):
    RETAIN = "retain"
    MONITOR = "monitor"
    MAINTAIN = "maintain"
    REPAIR = "repair"
    OVERHAUL = "overhaul"
    REPLACE = "replace"
    HUMAN_REVIEW = "human_review"


class ForecastConfidence(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


# =========================================================
# Cost item
# =========================================================

@dataclass(slots=True)
class FutureCostItem:
    name: str

    category: FutureCostCategory

    amount: float

    year: float = 1.0

    timing: CostTiming = CostTiming.ONE_TIME

    annual_growth_rate: float | None = None

    probability: float = 1.0

    confidence: float = 1.0

    include_in_total: bool = True

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.amount = non_negative(
            self.amount
        )

        self.year = non_negative(
            self.year
        )

        self.probability = clamp(
            self.probability
        )

        self.confidence = clamp(
            self.confidence
        )

        if self.annual_growth_rate is not None:
            self.annual_growth_rate = finite(
                self.annual_growth_rate
            )


# =========================================================
# Forecast assumptions
# =========================================================

@dataclass(slots=True)
class FutureCostAssumptions:
    horizon_years: int = 5

    discount_rate: float = 0.08

    inflation_rate: float = 0.03

    maintenance_escalation_rate: float | None = None
    labor_escalation_rate: float | None = None
    energy_escalation_rate: float | None = None
    spare_parts_escalation_rate: float | None = None

    failure_cost_escalation_rate: float | None = None

    currency: str = "INR"

    def __post_init__(self) -> None:
        self.horizon_years = max(
            1,
            int(self.horizon_years),
        )

        self.discount_rate = finite(
            self.discount_rate
        )

        self.inflation_rate = finite(
            self.inflation_rate
        )

        for name in (
            "maintenance_escalation_rate",
            "labor_escalation_rate",
            "energy_escalation_rate",
            "spare_parts_escalation_rate",
            "failure_cost_escalation_rate",
        ):
            value = getattr(self, name)

            if value is not None:
                setattr(
                    self,
                    name,
                    finite(value),
                )


# =========================================================
# Asset context
# =========================================================

@dataclass(slots=True)
class FutureAssetContext:
    asset_id: str | None = None
    component_id: str | None = None
    site_id: str | None = None

    current_asset_value: float | None = None
    replacement_cost: float | None = None

    remaining_useful_life_years: float | None = None

    current_health_score: float | None = None

    failure_probability: float | None = None

    annual_failure_probability_growth: float = 0.0

    degradation_rate: float = 0.0

    safety_risk_score: float = 0.0

    critical_defect: bool = False

    obsolete: bool = False

    parts_available: bool = True

    confidence: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "current_asset_value",
            "replacement_cost",
            "remaining_useful_life_years",
        ):
            value = getattr(self, name)

            if value is not None:
                setattr(
                    self,
                    name,
                    non_negative(value),
                )

        if self.current_health_score is not None:
            self.current_health_score = clamp(
                self.current_health_score
            )

        if self.failure_probability is not None:
            self.failure_probability = clamp(
                self.failure_probability
            )

        self.annual_failure_probability_growth = finite(
            self.annual_failure_probability_growth
        )

        self.degradation_rate = non_negative(
            self.degradation_rate
        )

        self.safety_risk_score = clamp(
            self.safety_risk_score
        )

        self.confidence = clamp(
            self.confidence
        )


# =========================================================
# Yearly forecast
# =========================================================

@dataclass(slots=True)
class YearlyFutureCost:
    year: int

    maintenance_cost: float = 0.0
    repair_cost: float = 0.0
    overhaul_cost: float = 0.0
    replacement_cost: float = 0.0

    failure_cost: float = 0.0
    downtime_cost: float = 0.0
    production_loss: float = 0.0

    energy_cost: float = 0.0
    spares_cost: float = 0.0
    labor_cost: float = 0.0

    inspection_cost: float = 0.0
    contractor_cost: float = 0.0

    safety_cost: float = 0.0
    emergency_cost: float = 0.0
    environmental_cost: float = 0.0

    regulatory_cost: float = 0.0
    software_cost: float = 0.0
    licensing_cost: float = 0.0

    other_cost: float = 0.0

    undiscounted_total: float = 0.0
    discounted_total: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# =========================================================
# Category totals
# =========================================================

@dataclass(slots=True)
class FutureCostBreakdown:
    maintenance: float = 0.0
    repair: float = 0.0
    overhaul: float = 0.0
    replacement: float = 0.0

    failure: float = 0.0
    downtime: float = 0.0
    production_loss: float = 0.0

    energy: float = 0.0
    spares: float = 0.0
    labor: float = 0.0

    inspection: float = 0.0
    contractor: float = 0.0

    safety: float = 0.0
    emergency: float = 0.0
    environmental: float = 0.0

    regulatory: float = 0.0
    software: float = 0.0
    licensing: float = 0.0

    other: float = 0.0

    @property
    def total(self) -> float:
        return sum(
            (
                self.maintenance,
                self.repair,
                self.overhaul,
                self.replacement,
                self.failure,
                self.downtime,
                self.production_loss,
                self.energy,
                self.spares,
                self.labor,
                self.inspection,
                self.contractor,
                self.safety,
                self.emergency,
                self.environmental,
                self.regulatory,
                self.software,
                self.licensing,
                self.other,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        result = json_safe(self)
        result["total"] = round(
            self.total,
            2,
        )
        return result


# =========================================================
# Result
# =========================================================

@dataclass(slots=True)
class FutureCostResult:
    calculation_id: str

    asset_id: str | None
    component_id: str | None
    site_id: str | None

    scenario: ForecastScenario

    horizon_years: int

    undiscounted_future_cost: float

    present_value_future_cost: float

    average_annual_cost: float

    equivalent_annual_cost: float

    cost_to_asset_value_ratio: float | None

    cost_to_replacement_ratio: float | None

    breakdown: FutureCostBreakdown

    yearly_forecast: list[YearlyFutureCost]

    economic_risk: FutureEconomicRisk

    recommendation: FutureCostDecision

    confidence: float
    confidence_level: ForecastConfidence

    currency: str

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
# Scenario multipliers
# =========================================================

@dataclass(slots=True)
class ScenarioMultipliers:
    optimistic: float = 0.85
    base: float = 1.00
    pessimistic: float = 1.20
    stress: float = 1.50

    def get(
        self,
        scenario: ForecastScenario,
    ) -> float:
        mapping = {
            ForecastScenario.OPTIMISTIC:
                self.optimistic,

            ForecastScenario.BASE:
                self.base,

            ForecastScenario.PESSIMISTIC:
                self.pessimistic,

            ForecastScenario.STRESS:
                self.stress,
        }

        return non_negative(
            mapping[scenario],
            1.0,
        )


# =========================================================
# Config
# =========================================================

@dataclass(slots=True)
class FutureCostConfig:
    moderate_risk_ratio: float = 0.25
    high_risk_ratio: float = 0.50
    critical_risk_ratio: float = 0.80
    extreme_risk_ratio: float = 1.20

    minimum_confidence: float = 0.50

    replacement_ratio_threshold: float = 0.80

    critical_safety_threshold: float = 0.80

    default_currency: str = "INR"

    scenario_multipliers: ScenarioMultipliers = field(
        default_factory=ScenarioMultipliers
    )


# =========================================================
# Calculator
# =========================================================

class FutureCostCalculator:

    def __init__(
        self,
        *,
        config: FutureCostConfig | None = None,
    ) -> None:
        self.config = (
            config
            or FutureCostConfig()
        )

    # =====================================================
    # Growth rate by category
    # =====================================================

    @staticmethod
    def growth_rate_for_category(
        *,
        category: FutureCostCategory,
        item: FutureCostItem,
        assumptions: FutureCostAssumptions,
    ) -> float:

        if item.annual_growth_rate is not None:
            return item.annual_growth_rate

        if (
            category
            == FutureCostCategory.MAINTENANCE
            and assumptions
            .maintenance_escalation_rate
            is not None
        ):
            return (
                assumptions
                .maintenance_escalation_rate
            )

        if (
            category
            == FutureCostCategory.LABOR
            and assumptions
            .labor_escalation_rate
            is not None
        ):
            return (
                assumptions
                .labor_escalation_rate
            )

        if (
            category
            == FutureCostCategory.ENERGY
            and assumptions
            .energy_escalation_rate
            is not None
        ):
            return (
                assumptions
                .energy_escalation_rate
            )

        if (
            category
            == FutureCostCategory.SPARES
            and assumptions
            .spare_parts_escalation_rate
            is not None
        ):
            return (
                assumptions
                .spare_parts_escalation_rate
            )

        if (
            category
            == FutureCostCategory.FAILURE
            and assumptions
            .failure_cost_escalation_rate
            is not None
        ):
            return (
                assumptions
                .failure_cost_escalation_rate
            )

        return assumptions.inflation_rate

    # =====================================================
    # Future value
    # =====================================================

    @staticmethod
    def future_value(
        *,
        current_cost: float,
        annual_growth_rate: float,
        years: float,
    ) -> float:

        cost = non_negative(current_cost)

        years = non_negative(years)

        rate = finite(
            annual_growth_rate
        )

        base = 1.0 + rate

        if base <= 0:
            return 0.0

        return cost * (base ** years)

    # =====================================================
    # Present value
    # =====================================================

    @staticmethod
    def present_value(
        *,
        future_cost: float,
        discount_rate: float,
        years: float,
    ) -> float:

        cost = non_negative(
            future_cost
        )

        years = non_negative(
            years
        )

        rate = finite(
            discount_rate
        )

        base = 1.0 + rate

        if base <= 0:
            return cost

        return cost / (base ** years)

    # =====================================================
    # Equivalent annual cost
    # =====================================================

    @staticmethod
    def equivalent_annual_cost(
        *,
        present_value: float,
        discount_rate: float,
        years: int,
    ) -> float:

        pv = non_negative(
            present_value
        )

        n = max(
            1,
            int(years),
        )

        rate = finite(
            discount_rate
        )

        if abs(rate) <= 1e-12:
            return pv / n

        if 1.0 + rate <= 0:
            return pv / n

        factor = (
            rate
            * (1.0 + rate) ** n
            / (
                (1.0 + rate) ** n
                - 1.0
            )
        )

        return pv * factor

    # =====================================================
    # Failure probability projection
    # =====================================================

    @staticmethod
    def project_failure_probability(
        *,
        current_probability: float,
        annual_growth: float,
        degradation_rate: float,
        year: int,
    ) -> float:

        probability = clamp(
            current_probability
        )

        growth = finite(
            annual_growth
        )

        degradation = non_negative(
            degradation_rate
        )

        years = max(
            0,
            int(year) - 1,
        )

        projected = probability

        for _ in range(years):
            projected = (
                projected
                + projected * growth
                + degradation
                * (1.0 - projected)
            )

            projected = clamp(
                projected
            )

        return projected

    # =====================================================
    # Timing occurrences
    # =====================================================

    @staticmethod
    def occurrences_per_year(
        timing: CostTiming,
    ) -> float:

        return {
            CostTiming.ONE_TIME: 1.0,
            CostTiming.ANNUAL: 1.0,
            CostTiming.MONTHLY: 12.0,
            CostTiming.QUARTERLY: 4.0,
            CostTiming.EVENT_BASED: 1.0,
        }[timing]

    # =====================================================
    # Does item occur in year?
    # =====================================================

    @staticmethod
    def item_occurs(
        *,
        item: FutureCostItem,
        year: int,
    ) -> bool:

        start_year = max(
            1,
            int(math.ceil(item.year)),
        )

        if item.timing == CostTiming.ONE_TIME:
            return year == start_year

        return year >= start_year

    # =====================================================
    # Item cost in year
    # =====================================================

    def item_cost_for_year(
        self,
        *,
        item: FutureCostItem,
        year: int,
        assumptions: FutureCostAssumptions,
        scenario: ForecastScenario,
        asset: FutureAssetContext | None,
    ) -> float:

        if not item.include_in_total:
            return 0.0

        if not self.item_occurs(
            item=item,
            year=year,
        ):
            return 0.0

        start_year = max(
            1,
            int(math.ceil(item.year)),
        )

        years_since_start = max(
            0,
            year - start_year,
        )

        growth = self.growth_rate_for_category(
            category=item.category,
            item=item,
            assumptions=assumptions,
        )

        amount = self.future_value(
            current_cost=item.amount,
            annual_growth_rate=growth,
            years=years_since_start,
        )

        amount *= self.occurrences_per_year(
            item.timing
        )

        probability = item.probability

        # Failure events can use projected asset
        # failure probability if available.
        if (
            item.category
            == FutureCostCategory.FAILURE
            and asset is not None
            and asset.failure_probability
            is not None
        ):
            projected = (
                self.project_failure_probability(
                    current_probability=(
                        asset.failure_probability
                    ),
                    annual_growth=(
                        asset
                        .annual_failure_probability_growth
                    ),
                    degradation_rate=(
                        asset.degradation_rate
                    ),
                    year=year,
                )
            )

            probability = clamp(
                probability * projected
            )

        amount *= probability

        amount *= (
            self.config
            .scenario_multipliers
            .get(scenario)
        )

        return max(
            0.0,
            amount,
        )

    # =====================================================
    # Add cost to yearly bucket
    # =====================================================

    @staticmethod
    def add_category_cost(
        *,
        yearly: YearlyFutureCost,
        category: FutureCostCategory,
        amount: float,
    ) -> None:

        mapping = {
            FutureCostCategory.MAINTENANCE:
                "maintenance_cost",

            FutureCostCategory.REPAIR:
                "repair_cost",

            FutureCostCategory.OVERHAUL:
                "overhaul_cost",

            FutureCostCategory.REPLACEMENT:
                "replacement_cost",

            FutureCostCategory.FAILURE:
                "failure_cost",

            FutureCostCategory.DOWNTIME:
                "downtime_cost",

            FutureCostCategory.PRODUCTION_LOSS:
                "production_loss",

            FutureCostCategory.ENERGY:
                "energy_cost",

            FutureCostCategory.SPARES:
                "spares_cost",

            FutureCostCategory.LABOR:
                "labor_cost",

            FutureCostCategory.INSPECTION:
                "inspection_cost",

            FutureCostCategory.CONTRACTOR:
                "contractor_cost",

            FutureCostCategory.SAFETY:
                "safety_cost",

            FutureCostCategory.EMERGENCY:
                "emergency_cost",

            FutureCostCategory.ENVIRONMENTAL:
                "environmental_cost",

            FutureCostCategory.REGULATORY:
                "regulatory_cost",

            FutureCostCategory.SOFTWARE:
                "software_cost",

            FutureCostCategory.LICENSING:
                "licensing_cost",

            FutureCostCategory.OTHER:
                "other_cost",
        }

        attribute = mapping[category]

        setattr(
            yearly,
            attribute,
            getattr(yearly, attribute)
            + amount,
        )

    # =====================================================
    # Build yearly forecast
    # =====================================================

    def build_yearly_forecast(
        self,
        *,
        items: Sequence[FutureCostItem],
        assumptions: FutureCostAssumptions,
        scenario: ForecastScenario,
        asset: FutureAssetContext | None,
    ) -> list[YearlyFutureCost]:

        forecast: list[
            YearlyFutureCost
        ] = []

        for year in range(
            1,
            assumptions.horizon_years + 1,
        ):
            yearly = YearlyFutureCost(
                year=year
            )

            for item in items:
                amount = (
                    self.item_cost_for_year(
                        item=item,
                        year=year,
                        assumptions=assumptions,
                        scenario=scenario,
                        asset=asset,
                    )
                )

                self.add_category_cost(
                    yearly=yearly,
                    category=item.category,
                    amount=amount,
                )

            yearly.undiscounted_total = sum(
                (
                    yearly.maintenance_cost,
                    yearly.repair_cost,
                    yearly.overhaul_cost,
                    yearly.replacement_cost,
                    yearly.failure_cost,
                    yearly.downtime_cost,
                    yearly.production_loss,
                    yearly.energy_cost,
                    yearly.spares_cost,
                    yearly.labor_cost,
                    yearly.inspection_cost,
                    yearly.contractor_cost,
                    yearly.safety_cost,
                    yearly.emergency_cost,
                    yearly.environmental_cost,
                    yearly.regulatory_cost,
                    yearly.software_cost,
                    yearly.licensing_cost,
                    yearly.other_cost,
                )
            )

            yearly.discounted_total = (
                self.present_value(
                    future_cost=(
                        yearly.undiscounted_total
                    ),
                    discount_rate=(
                        assumptions.discount_rate
                    ),
                    years=year,
                )
            )

            forecast.append(yearly)

        return forecast

    # =====================================================
    # Breakdown
    # =====================================================

    @staticmethod
    def build_breakdown(
        forecast: Sequence[
            YearlyFutureCost
        ],
    ) -> FutureCostBreakdown:

        return FutureCostBreakdown(
            maintenance=sum(
                x.maintenance_cost
                for x in forecast
            ),

            repair=sum(
                x.repair_cost
                for x in forecast
            ),

            overhaul=sum(
                x.overhaul_cost
                for x in forecast
            ),

            replacement=sum(
                x.replacement_cost
                for x in forecast
            ),

            failure=sum(
                x.failure_cost
                for x in forecast
            ),

            downtime=sum(
                x.downtime_cost
                for x in forecast
            ),

            production_loss=sum(
                x.production_loss
                for x in forecast
            ),

            energy=sum(
                x.energy_cost
                for x in forecast
            ),

            spares=sum(
                x.spares_cost
                for x in forecast
            ),

            labor=sum(
                x.labor_cost
                for x in forecast
            ),

            inspection=sum(
                x.inspection_cost
                for x in forecast
            ),

            contractor=sum(
                x.contractor_cost
                for x in forecast
            ),

            safety=sum(
                x.safety_cost
                for x in forecast
            ),

            emergency=sum(
                x.emergency_cost
                for x in forecast
            ),

            environmental=sum(
                x.environmental_cost
                for x in forecast
            ),

            regulatory=sum(
                x.regulatory_cost
                for x in forecast
            ),

            software=sum(
                x.software_cost
                for x in forecast
            ),

            licensing=sum(
                x.licensing_cost
                for x in forecast
            ),

            other=sum(
                x.other_cost
                for x in forecast
            ),
        )

    # =====================================================
    # Confidence
    # =====================================================

    @staticmethod
    def calculate_confidence(
        *,
        items: Sequence[
            FutureCostItem
        ],
        asset: FutureAssetContext | None,
    ) -> tuple[
        float,
        ForecastConfidence,
    ]:

        active_items = [
            item
            for item in items
            if item.include_in_total
        ]

        if active_items:
            item_confidence = sum(
                item.confidence
                for item in active_items
            ) / len(active_items)
        else:
            item_confidence = 0.0

        asset_confidence = (
            asset.confidence
            if asset is not None
            else 0.60
        )

        score = clamp(
            0.75 * item_confidence
            + 0.25 * asset_confidence
        )

        if score >= 0.90:
            level = (
                ForecastConfidence.VERY_HIGH
            )

        elif score >= 0.75:
            level = (
                ForecastConfidence.HIGH
            )

        elif score >= 0.50:
            level = (
                ForecastConfidence.MODERATE
            )

        else:
            level = (
                ForecastConfidence.LOW
            )

        return round(score, 4), level

    # =====================================================
    # Economic risk
    # =====================================================

    def economic_risk(
        self,
        *,
        present_value_cost: float,
        asset: FutureAssetContext | None,
    ) -> FutureEconomicRisk:

        reference_value = None

        if asset is not None:
            reference_value = (
                asset.replacement_cost
                or asset.current_asset_value
            )

        if (
            reference_value is None
            or reference_value <= 0
        ):
            return FutureEconomicRisk.MODERATE

        ratio = (
            present_value_cost
            / reference_value
        )

        if (
            ratio
            >= self.config.extreme_risk_ratio
        ):
            return FutureEconomicRisk.EXTREME

        if (
            ratio
            >= self.config.critical_risk_ratio
        ):
            return FutureEconomicRisk.CRITICAL

        if (
            ratio
            >= self.config.high_risk_ratio
        ):
            return FutureEconomicRisk.HIGH

        if (
            ratio
            >= self.config.moderate_risk_ratio
        ):
            return FutureEconomicRisk.MODERATE

        return FutureEconomicRisk.LOW

    # =====================================================
    # Recommendation
    # =====================================================

    def recommend(
        self,
        *,
        present_value_cost: float,
        asset: FutureAssetContext | None,
        confidence: float,
    ) -> FutureCostDecision:

        if confidence < self.config.minimum_confidence:
            return FutureCostDecision.HUMAN_REVIEW

        if asset is None:
            return FutureCostDecision.MONITOR

        # Safety is never overridden by financial logic.
        if (
            asset.safety_risk_score
            >= self.config
            .critical_safety_threshold
        ):
            return FutureCostDecision.HUMAN_REVIEW

        if asset.critical_defect:
            return FutureCostDecision.HUMAN_REVIEW

        replacement_cost = (
            asset.replacement_cost
        )

        if (
            replacement_cost is not None
            and replacement_cost > 0
        ):
            ratio = (
                present_value_cost
                / replacement_cost
            )

            if (
                ratio
                >= self.config
                .replacement_ratio_threshold
            ):
                return FutureCostDecision.REPLACE

        if asset.obsolete:
            return FutureCostDecision.REPLACE

        if not asset.parts_available:
            return FutureCostDecision.REPLACE

        if (
            asset.remaining_useful_life_years
            is not None
            and asset.remaining_useful_life_years
            <= 1.0
        ):
            return FutureCostDecision.REPLACE

        if (
            asset.current_health_score
            is not None
        ):
            if asset.current_health_score < 0.35:
                return FutureCostDecision.OVERHAUL

            if asset.current_health_score < 0.60:
                return FutureCostDecision.REPAIR

            if asset.current_health_score < 0.80:
                return FutureCostDecision.MAINTAIN

        return FutureCostDecision.RETAIN

    # =====================================================
    # Main forecast
    # =====================================================

    def forecast(
        self,
        *,
        items: Sequence[
            FutureCostItem
        ],
        assumptions: FutureCostAssumptions,
        asset: FutureAssetContext | None = None,
        scenario: ForecastScenario = (
            ForecastScenario.BASE
        ),
        metadata: Mapping[str, Any] | None = None,
    ) -> FutureCostResult:

        warnings: list[str] = []

        yearly = self.build_yearly_forecast(
            items=items,
            assumptions=assumptions,
            scenario=scenario,
            asset=asset,
        )

        breakdown = self.build_breakdown(
            yearly
        )

        undiscounted = sum(
            year.undiscounted_total
            for year in yearly
        )

        present_value_cost = sum(
            year.discounted_total
            for year in yearly
        )

        average_annual = (
            undiscounted
            / assumptions.horizon_years
        )

        eac = self.equivalent_annual_cost(
            present_value=present_value_cost,
            discount_rate=(
                assumptions.discount_rate
            ),
            years=assumptions.horizon_years,
        )

        confidence, confidence_level = (
            self.calculate_confidence(
                items=items,
                asset=asset,
            )
        )

        risk = self.economic_risk(
            present_value_cost=(
                present_value_cost
            ),
            asset=asset,
        )

        recommendation = self.recommend(
            present_value_cost=(
                present_value_cost
            ),
            asset=asset,
            confidence=confidence,
        )

        asset_ratio = None
        replacement_ratio = None

        if (
            asset is not None
            and asset.current_asset_value
            is not None
            and asset.current_asset_value > 0
        ):
            asset_ratio = (
                present_value_cost
                / asset.current_asset_value
            )

        if (
            asset is not None
            and asset.replacement_cost
            is not None
            and asset.replacement_cost > 0
        ):
            replacement_ratio = (
                present_value_cost
                / asset.replacement_cost
            )

        # ---------------------------------------------
        # Validation warnings
        # ---------------------------------------------

        if not items:
            warnings.append(
                "No future cost items were supplied."
            )

        if confidence < (
            self.config.minimum_confidence
        ):
            warnings.append(
                "Future cost forecast has low "
                "confidence."
            )

        if assumptions.horizon_years > 10:
            warnings.append(
                "Long forecast horizons increase "
                "uncertainty. Consider scenario or "
                "Monte Carlo analysis."
            )

        if (
            assumptions.discount_rate
            < -0.50
        ):
            warnings.append(
                "Unusually negative discount rate "
                "detected."
            )

        if (
            assumptions.inflation_rate
            > 0.25
        ):
            warnings.append(
                "High inflation assumption detected."
            )

        if asset is not None:
            if (
                asset.safety_risk_score
                >= self.config
                .critical_safety_threshold
            ):
                warnings.append(
                    "Critical safety risk exists. "
                    "Financial optimization must not "
                    "override safety controls."
                )

            if asset.critical_defect:
                warnings.append(
                    "Critical defect detected. "
                    "Engineering/human review is "
                    "required."
                )

            if asset.obsolete:
                warnings.append(
                    "Asset is marked obsolete."
                )

            if not asset.parts_available:
                warnings.append(
                    "Required spare parts are marked "
                    "unavailable."
                )

        currencies = {
            assumptions.currency
        }

        return FutureCostResult(
            calculation_id=new_id(),

            asset_id=(
                asset.asset_id
                if asset is not None
                else None
            ),

            component_id=(
                asset.component_id
                if asset is not None
                else None
            ),

            site_id=(
                asset.site_id
                if asset is not None
                else None
            ),

            scenario=scenario,

            horizon_years=(
                assumptions.horizon_years
            ),

            undiscounted_future_cost=round(
                undiscounted,
                2,
            ),

            present_value_future_cost=round(
                present_value_cost,
                2,
            ),

            average_annual_cost=round(
                average_annual,
                2,
            ),

            equivalent_annual_cost=round(
                eac,
                2,
            ),

            cost_to_asset_value_ratio=(
                round(asset_ratio, 6)
                if asset_ratio is not None
                else None
            ),

            cost_to_replacement_ratio=(
                round(replacement_ratio, 6)
                if replacement_ratio is not None
                else None
            ),

            breakdown=breakdown,

            yearly_forecast=yearly,

            economic_risk=risk,

            recommendation=recommendation,

            confidence=confidence,

            confidence_level=(
                confidence_level
            ),

            currency=(
                next(iter(currencies))
                if currencies
                else self.config.default_currency
            ),

            warnings=warnings,

            metadata=dict(
                metadata or {}
            ),
        )

    # =====================================================
    # Multi-scenario forecast
    # =====================================================

    def forecast_scenarios(
        self,
        *,
        items: Sequence[
            FutureCostItem
        ],
        assumptions: FutureCostAssumptions,
        asset: FutureAssetContext | None = None,
    ) -> dict[
        ForecastScenario,
        FutureCostResult,
    ]:

        return {
            scenario: self.forecast(
                items=items,
                assumptions=assumptions,
                asset=asset,
                scenario=scenario,
            )
            for scenario in ForecastScenario
        }

    # =====================================================
    # Compare strategies
    # =====================================================

    def compare_strategies(
        self,
        *,
        strategies: Mapping[
            str,
            Sequence[FutureCostItem]
        ],
        assumptions: FutureCostAssumptions,
        asset: FutureAssetContext | None = None,
        scenario: ForecastScenario = (
            ForecastScenario.BASE
        ),
    ) -> dict[str, Any]:

        results: dict[
            str,
            FutureCostResult
        ] = {}

        for name, items in strategies.items():
            results[name] = self.forecast(
                items=items,
                assumptions=assumptions,
                asset=asset,
                scenario=scenario,
            )

        if not results:
            return {
                "strategies": {},
                "recommended_strategy": None,
            }

        best_name = min(
            results,
            key=lambda name: (
                results[name]
                .present_value_future_cost
            ),
        )

        best_cost = (
            results[best_name]
            .present_value_future_cost
        )

        return {
            "recommended_strategy": best_name,

            "recommended_present_value_cost":
                round(best_cost, 2),

            "strategies": {
                name: {
                    "present_value_future_cost":
                        result
                        .present_value_future_cost,

                    "equivalent_annual_cost":
                        result
                        .equivalent_annual_cost,

                    "economic_risk":
                        result
                        .economic_risk.value,

                    "recommendation":
                        result
                        .recommendation.value,

                    "difference_from_best":
                        round(
                            result
                            .present_value_future_cost
                            - best_cost,
                            2,
                        ),
                }
                for name, result
                in results.items()
            },
        }

    # =====================================================
    # Cost at specific future year
    # =====================================================

    def cost_at_year(
        self,
        *,
        current_cost: float,
        year: float,
        annual_growth_rate: float,
        discount_rate: float | None = None,
    ) -> dict[str, float]:

        future = self.future_value(
            current_cost=current_cost,
            annual_growth_rate=(
                annual_growth_rate
            ),
            years=year,
        )

        result = {
            "future_nominal_cost": round(
                future,
                2,
            )
        }

        if discount_rate is not None:
            result["present_value"] = round(
                self.present_value(
                    future_cost=future,
                    discount_rate=(
                        discount_rate
                    ),
                    years=year,
                ),
                2,
            )

        return result

    # =====================================================
    # Aggregate asset forecasts
    # =====================================================

    @staticmethod
    def aggregate(
        results: Sequence[
            FutureCostResult
        ],
    ) -> dict[str, Any]:

        if not results:
            return {
                "asset_count": 0,
                "present_value_future_cost": 0.0,
                "undiscounted_future_cost": 0.0,
                "equivalent_annual_cost": 0.0,
                "currency": None,
            }

        currencies = {
            result.currency
            for result in results
        }

        total_pv = sum(
            result.present_value_future_cost
            for result in results
        )

        total_nominal = sum(
            result.undiscounted_future_cost
            for result in results
        )

        total_eac = sum(
            result.equivalent_annual_cost
            for result in results
        )

        risk_rank = {
            FutureEconomicRisk.LOW: 1,
            FutureEconomicRisk.MODERATE: 2,
            FutureEconomicRisk.HIGH: 3,
            FutureEconomicRisk.CRITICAL: 4,
            FutureEconomicRisk.EXTREME: 5,
        }

        highest = max(
            results,
            key=lambda result: risk_rank[
                result.economic_risk
            ],
        )

        return {
            "asset_count": len(results),

            "present_value_future_cost":
                round(total_pv, 2),

            "undiscounted_future_cost":
                round(total_nominal, 2),

            "equivalent_annual_cost":
                round(total_eac, 2),

            "highest_economic_risk":
                highest.economic_risk.value,

            "highest_risk_asset_id":
                highest.asset_id,

            "currency": (
                next(iter(currencies))
                if len(currencies) == 1
                else "MIXED"
            ),

            "mixed_currency":
                len(currencies) > 1,
        }


# =========================================================
# Default calculator
# =========================================================

future_cost_calculator = (
    FutureCostCalculator()
)


# =========================================================
# Agent-facing helpers
# =========================================================

def calculate_future_cost(
    *,
    items: Sequence[FutureCostItem],
    assumptions: FutureCostAssumptions,
    asset: FutureAssetContext | None = None,
    scenario: ForecastScenario = (
        ForecastScenario.BASE
    ),
) -> FutureCostResult:

    return future_cost_calculator.forecast(
        items=items,
        assumptions=assumptions,
        asset=asset,
        scenario=scenario,
    )


def calculate_future_cost_scenarios(
    *,
    items: Sequence[FutureCostItem],
    assumptions: FutureCostAssumptions,
    asset: FutureAssetContext | None = None,
) -> dict[
    ForecastScenario,
    FutureCostResult,
]:

    return (
        future_cost_calculator
        .forecast_scenarios(
            items=items,
            assumptions=assumptions,
            asset=asset,
        )
    )


def calculate_future_value(
    *,
    current_cost: float,
    annual_growth_rate: float,
    years: float,
) -> float:

    return round(
        future_cost_calculator.future_value(
            current_cost=current_cost,
            annual_growth_rate=(
                annual_growth_rate
            ),
            years=years,
        ),
        2,
    )


def calculate_present_value(
    *,
    future_cost: float,
    discount_rate: float,
    years: float,
) -> float:

    return round(
        future_cost_calculator.present_value(
            future_cost=future_cost,
            discount_rate=discount_rate,
            years=years,
        ),
        2,
    )


# =========================================================
# Exports
# =========================================================

__all__ = [
    "FutureCostCategory",
    "CostTiming",
    "ForecastScenario",
    "FutureEconomicRisk",
    "FutureCostDecision",
    "ForecastConfidence",

    "FutureCostItem",
    "FutureCostAssumptions",
    "FutureAssetContext",

    "YearlyFutureCost",
    "FutureCostBreakdown",
    "FutureCostResult",

    "ScenarioMultipliers",
    "FutureCostConfig",
    "FutureCostCalculator",

    "future_cost_calculator",

    "calculate_future_cost",
    "calculate_future_cost_scenarios",
    "calculate_future_value",
    "calculate_present_value",
]