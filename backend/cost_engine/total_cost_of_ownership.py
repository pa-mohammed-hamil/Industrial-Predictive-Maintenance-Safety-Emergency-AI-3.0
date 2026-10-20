# app/agents/tools/total_cost_of_ownership.py

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
    default: float | None = None,
) -> float | None:
    if abs(denominator) <= 1e-12:
        return default
    return numerator / denominator


def ensure_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None

    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)

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

class TCOCategory(str, Enum):
    ACQUISITION = "acquisition"
    INSTALLATION = "installation"
    ENGINEERING = "engineering"
    COMMISSIONING = "commissioning"
    TRAINING = "training"

    MAINTENANCE = "maintenance"
    REPAIR = "repair"
    OVERHAUL = "overhaul"

    SPARE_PARTS = "spare_parts"
    LABOR = "labor"

    ENERGY = "energy"
    UTILITIES = "utilities"

    DOWNTIME = "downtime"
    PRODUCTION_LOSS = "production_loss"

    SCRAP = "scrap"
    QUALITY = "quality"

    EXPECTED_FAILURE = "expected_failure"

    SAFETY = "safety"
    EMERGENCY = "emergency"
    ENVIRONMENTAL = "environmental"

    SOFTWARE = "software"
    LICENSING = "licensing"
    SUPPORT = "support"

    INSPECTION = "inspection"
    CALIBRATION = "calibration"

    CONTRACTOR = "contractor"
    LOGISTICS = "logistics"

    TAX = "tax"
    INSURANCE = "insurance"

    REPLACEMENT = "replacement"

    DECOMMISSIONING = "decommissioning"
    DISPOSAL = "disposal"

    OTHER = "other"


class TCOCostTiming(str, Enum):
    ONE_TIME = "one_time"
    ANNUAL = "annual"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    EVENT_BASED = "event_based"


class TCOScenario(str, Enum):
    OPTIMISTIC = "optimistic"
    BASE = "base"
    PESSIMISTIC = "pessimistic"
    STRESS = "stress"


class TCOEconomicRisk(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    EXTREME = "extreme"


class TCOConfidence(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class TCORecommendation(str, Enum):
    RETAIN = "retain"
    MONITOR = "monitor"
    MAINTAIN = "maintain"
    REPAIR = "repair"
    OVERHAUL = "overhaul"
    REPLACE = "replace"
    HUMAN_REVIEW = "human_review"
    INSUFFICIENT_DATA = "insufficient_data"


# =========================================================
# Cost item
# =========================================================

@dataclass(slots=True)
class TCOCostItem:
    name: str
    category: TCOCategory

    amount: float

    start_year: int = 0

    timing: TCOCostTiming = (
        TCOCostTiming.ONE_TIME
    )

    probability: float = 1.0

    annual_growth_rate: float | None = None

    confidence: float = 1.0

    include_in_tco: bool = True

    source: str | None = None

    source_calculation_id: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.amount = non_negative(self.amount)

        self.start_year = max(
            0,
            int(self.start_year),
        )

        self.probability = clamp(
            self.probability
        )

        if self.annual_growth_rate is not None:
            self.annual_growth_rate = finite(
                self.annual_growth_rate
            )

        self.confidence = clamp(
            self.confidence
        )


# =========================================================
# Credits / residual value
# =========================================================

@dataclass(slots=True)
class TCOCredits:
    salvage_value: float = 0.0
    residual_value: float = 0.0

    warranty_recovery: float = 0.0
    insurance_recovery: float = 0.0
    supplier_credit: float = 0.0

    tax_credit: float = 0.0
    grant_credit: float = 0.0

    recycling_credit: float = 0.0

    other_credit: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "salvage_value",
            "residual_value",
            "warranty_recovery",
            "insurance_recovery",
            "supplier_credit",
            "tax_credit",
            "grant_credit",
            "recycling_credit",
            "other_credit",
        ):
            setattr(
                self,
                name,
                non_negative(getattr(self, name)),
            )

    @property
    def terminal_credit(self) -> float:
        return (
            self.salvage_value
            + self.residual_value
            + self.recycling_credit
        )

    @property
    def operating_recovery(self) -> float:
        return (
            self.warranty_recovery
            + self.insurance_recovery
            + self.supplier_credit
            + self.tax_credit
            + self.grant_credit
            + self.other_credit
        )

    @property
    def total(self) -> float:
        return (
            self.terminal_credit
            + self.operating_recovery
        )


# =========================================================
# Asset context
# =========================================================

@dataclass(slots=True)
class TCOAssetContext:
    asset_id: str

    component_id: str | None = None
    site_id: str | None = None

    asset_name: str | None = None

    current_asset_value: float = 0.0
    replacement_cost: float = 0.0

    age_years: float = 0.0
    remaining_useful_life_years: float | None = None

    health_score: float | None = None

    annual_failure_probability: float | None = None

    safety_risk_score: float = 0.0

    critical_defect: bool = False
    obsolete: bool = False

    parts_available: bool = True

    confidence: float = 1.0

    def __post_init__(self) -> None:
        self.current_asset_value = non_negative(
            self.current_asset_value
        )

        self.replacement_cost = non_negative(
            self.replacement_cost
        )

        self.age_years = non_negative(
            self.age_years
        )

        if self.remaining_useful_life_years is not None:
            self.remaining_useful_life_years = non_negative(
                self.remaining_useful_life_years
            )

        if self.health_score is not None:
            self.health_score = clamp(
                self.health_score
            )

        if self.annual_failure_probability is not None:
            self.annual_failure_probability = clamp(
                self.annual_failure_probability
            )

        self.safety_risk_score = clamp(
            self.safety_risk_score
        )

        self.confidence = clamp(
            self.confidence
        )


# =========================================================
# Assumptions
# =========================================================

@dataclass(slots=True)
class TCOAssumptions:
    horizon_years: int = 10

    discount_rate: float = 0.08
    inflation_rate: float = 0.03

    maintenance_escalation_rate: float | None = None
    labor_escalation_rate: float | None = None
    spare_parts_escalation_rate: float | None = None
    energy_escalation_rate: float | None = None
    repair_escalation_rate: float | None = None
    failure_escalation_rate: float | None = None

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
            "spare_parts_escalation_rate",
            "energy_escalation_rate",
            "repair_escalation_rate",
            "failure_escalation_rate",
        ):
            value = getattr(self, name)

            if value is not None:
                setattr(
                    self,
                    name,
                    finite(value),
                )


# =========================================================
# Input
# =========================================================

@dataclass(slots=True)
class TCOInput:
    asset: TCOAssetContext

    costs: list[TCOCostItem]

    assumptions: TCOAssumptions = field(
        default_factory=TCOAssumptions
    )

    credits: TCOCredits = field(
        default_factory=TCOCredits
    )

    scenario: TCOScenario = (
        TCOScenario.BASE
    )

    include_initial_asset_value: bool = True

    calculation_name: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# =========================================================
# Year result
# =========================================================

@dataclass(slots=True)
class TCOYear:
    year: int

    acquisition: float = 0.0
    installation: float = 0.0
    engineering: float = 0.0
    commissioning: float = 0.0
    training: float = 0.0

    maintenance: float = 0.0
    repair: float = 0.0
    overhaul: float = 0.0

    spare_parts: float = 0.0
    labor: float = 0.0

    energy: float = 0.0
    utilities: float = 0.0

    downtime: float = 0.0
    production_loss: float = 0.0

    scrap: float = 0.0
    quality: float = 0.0

    expected_failure: float = 0.0

    safety: float = 0.0
    emergency: float = 0.0
    environmental: float = 0.0

    software: float = 0.0
    licensing: float = 0.0
    support: float = 0.0

    inspection: float = 0.0
    calibration: float = 0.0

    contractor: float = 0.0
    logistics: float = 0.0

    tax: float = 0.0
    insurance: float = 0.0

    replacement: float = 0.0

    decommissioning: float = 0.0
    disposal: float = 0.0

    other: float = 0.0

    credits: float = 0.0

    gross_cost: float = 0.0
    net_cost: float = 0.0

    discount_factor: float = 1.0

    discounted_gross_cost: float = 0.0
    discounted_credits: float = 0.0
    discounted_net_cost: float = 0.0

    cumulative_net_cost: float = 0.0
    cumulative_discounted_cost: float = 0.0


# =========================================================
# Breakdown
# =========================================================

@dataclass(slots=True)
class TCOBreakdown:
    category_totals: dict[str, float]

    capital_cost: float
    operating_cost: float
    maintenance_repair_cost: float

    reliability_cost: float
    production_quality_cost: float

    safety_environmental_cost: float

    end_of_life_cost: float

    gross_lifecycle_cost: float

    recovery_credits: float

    net_lifecycle_cost: float

    discounted_lifecycle_cost: float

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# =========================================================
# Result
# =========================================================

@dataclass(slots=True)
class TCOResult:
    calculation_id: str

    calculation_name: str | None

    asset_id: str
    component_id: str | None
    site_id: str | None

    scenario: TCOScenario

    horizon_years: int

    gross_tco: float
    total_credits: float
    net_tco: float

    present_value_tco: float

    equivalent_annual_cost: float

    average_annual_cost: float

    tco_to_current_asset_value_ratio: float | None
    tco_to_replacement_cost_ratio: float | None

    cost_per_remaining_life_year: float | None

    yearly_costs: list[TCOYear]

    breakdown: TCOBreakdown

    economic_risk: TCOEconomicRisk

    confidence: float
    confidence_level: TCOConfidence

    recommendation: TCORecommendation

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
# Comparison
# =========================================================

@dataclass(slots=True)
class TCOComparisonResult:
    comparison_id: str

    options: dict[str, TCOResult]

    lowest_pv_option: str | None

    lowest_eac_option: str | None

    pv_difference: float | None

    recommended_option: str | None

    warnings: list[str]

    generated_at: datetime = field(
        default_factory=utc_now
    )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# =========================================================
# Config
# =========================================================

@dataclass(slots=True)
class TCOConfig:
    moderate_replacement_ratio: float = 0.50
    high_replacement_ratio: float = 0.80
    critical_replacement_ratio: float = 1.20

    critical_safety_risk: float = 0.80

    minimum_confidence: float = 0.50

    default_currency: str = "INR"


# =========================================================
# Calculator
# =========================================================

class TotalCostOfOwnershipCalculator:

    def __init__(
        self,
        *,
        config: TCOConfig | None = None,
    ) -> None:
        self.config = config or TCOConfig()

    # =====================================================
    # Financial helpers
    # =====================================================

    @staticmethod
    def discount_factor(
        *,
        rate: float,
        year: float,
    ) -> float:

        rate = finite(rate)
        year = non_negative(year)

        if rate <= -1.0:
            return 1.0

        return 1.0 / (
            (1.0 + rate) ** year
        )

    @classmethod
    def present_value(
        cls,
        *,
        amount: float,
        rate: float,
        year: float,
    ) -> float:

        return (
            finite(amount)
            * cls.discount_factor(
                rate=rate,
                year=year,
            )
        )

    @staticmethod
    def future_value(
        *,
        amount: float,
        growth_rate: float,
        years: float,
    ) -> float:

        amount = non_negative(amount)
        years = non_negative(years)

        return amount * (
            (1.0 + finite(growth_rate))
            ** years
        )

    @staticmethod
    def equivalent_annual_cost(
        *,
        present_value: float,
        discount_rate: float,
        years: int,
    ) -> float:

        pv = non_negative(present_value)

        if years <= 0:
            return 0.0

        rate = finite(discount_rate)

        if abs(rate) <= 1e-12:
            return pv / years

        if rate <= -1.0:
            return pv / years

        denominator = (
            1.0
            - (1.0 + rate) ** (-years)
        )

        if abs(denominator) <= 1e-12:
            return pv / years

        return (
            pv
            * rate
            / denominator
        )

    # =====================================================
    # Growth rates
    # =====================================================

    @staticmethod
    def growth_rate_for_category(
        *,
        category: TCOCategory,
        assumptions: TCOAssumptions,
    ) -> float:

        mapping: dict[
            TCOCategory,
            float | None
        ] = {
            TCOCategory.MAINTENANCE:
                assumptions.maintenance_escalation_rate,

            TCOCategory.REPAIR:
                assumptions.repair_escalation_rate,

            TCOCategory.LABOR:
                assumptions.labor_escalation_rate,

            TCOCategory.CONTRACTOR:
                assumptions.labor_escalation_rate,

            TCOCategory.SPARE_PARTS:
                assumptions.spare_parts_escalation_rate,

            TCOCategory.ENERGY:
                assumptions.energy_escalation_rate,

            TCOCategory.UTILITIES:
                assumptions.energy_escalation_rate,

            TCOCategory.EXPECTED_FAILURE:
                assumptions.failure_escalation_rate,
        }

        selected = mapping.get(category)

        if selected is not None:
            return selected

        return assumptions.inflation_rate

    # =====================================================
    # Occurrences
    # =====================================================

    @staticmethod
    def occurrences_per_year(
        timing: TCOCostTiming,
    ) -> float:

        if timing == TCOCostTiming.MONTHLY:
            return 12.0

        if timing == TCOCostTiming.QUARTERLY:
            return 4.0

        return 1.0

    # =====================================================
    # Cost occurrence
    # =====================================================

    @staticmethod
    def item_occurs(
        *,
        item: TCOCostItem,
        year: int,
    ) -> bool:

        if not item.include_in_tco:
            return False

        if item.timing == TCOCostTiming.ONE_TIME:
            return year == item.start_year

        return year >= max(
            1,
            item.start_year,
        )

    # =====================================================
    # Item cost
    # =====================================================

    def item_cost_for_year(
        self,
        *,
        item: TCOCostItem,
        year: int,
        assumptions: TCOAssumptions,
        scenario_multiplier: float,
    ) -> float:

        if not self.item_occurs(
            item=item,
            year=year,
        ):
            return 0.0

        base_amount = (
            item.amount
            * item.probability
        )

        if item.timing == TCOCostTiming.ONE_TIME:
            growth_years = max(
                0,
                year - item.start_year,
            )
        else:
            growth_years = max(
                0,
                year - max(1, item.start_year),
            )

        growth_rate = (
            item.annual_growth_rate
            if item.annual_growth_rate is not None
            else self.growth_rate_for_category(
                category=item.category,
                assumptions=assumptions,
            )
        )

        amount = self.future_value(
            amount=base_amount,
            growth_rate=growth_rate,
            years=growth_years,
        )

        amount *= self.occurrences_per_year(
            item.timing
        )

        return (
            amount
            * max(0.0, scenario_multiplier)
        )

    # =====================================================
    # Scenario multiplier
    # =====================================================

    @staticmethod
    def scenario_multiplier(
        scenario: TCOScenario,
    ) -> float:

        return {
            TCOScenario.OPTIMISTIC: 0.85,
            TCOScenario.BASE: 1.00,
            TCOScenario.PESSIMISTIC: 1.20,
            TCOScenario.STRESS: 1.50,
        }[scenario]

    # =====================================================
    # Add category
    # =====================================================

    @staticmethod
    def add_category_cost(
        year: TCOYear,
        *,
        category: TCOCategory,
        amount: float,
    ) -> None:

        attribute = category.value

        if hasattr(year, attribute):
            setattr(
                year,
                attribute,
                getattr(year, attribute)
                + amount,
            )
        else:
            year.other += amount

    # =====================================================
    # Yearly forecast
    # =====================================================

    def build_yearly_costs(
        self,
        *,
        request: TCOInput,
    ) -> list[TCOYear]:

        assumptions = request.assumptions

        multiplier = self.scenario_multiplier(
            request.scenario
        )

        years = [
            TCOYear(year=year)
            for year in range(
                0,
                assumptions.horizon_years + 1,
            )
        ]

        # Current acquisition/current asset basis.
        if request.include_initial_asset_value:
            years[0].acquisition += (
                request.asset.current_asset_value
            )

        for item in request.costs:
            for year in range(
                0,
                assumptions.horizon_years + 1,
            ):
                amount = self.item_cost_for_year(
                    item=item,
                    year=year,
                    assumptions=assumptions,
                    scenario_multiplier=multiplier,
                )

                if amount <= 0:
                    continue

                self.add_category_cost(
                    years[year],
                    category=item.category,
                    amount=amount,
                )

        # Operating recoveries at year zero.
        years[0].credits += (
            request.credits.operating_recovery
        )

        # Terminal residual/salvage value.
        years[
            assumptions.horizon_years
        ].credits += (
            request.credits.terminal_credit
        )

        cumulative = 0.0
        cumulative_discounted = 0.0

        for year in years:
            categories = [
                field_name
                for field_name in (
                    category.value
                    for category in TCOCategory
                )
            ]

            gross = sum(
                non_negative(
                    getattr(year, name, 0.0)
                )
                for name in categories
            )

            net = gross - year.credits

            factor = self.discount_factor(
                rate=assumptions.discount_rate,
                year=year.year,
            )

            discounted_gross = (
                gross * factor
            )

            discounted_credit = (
                year.credits * factor
            )

            discounted_net = (
                net * factor
            )

            cumulative += net
            cumulative_discounted += (
                discounted_net
            )

            year.gross_cost = round(
                gross, 2
            )

            year.net_cost = round(
                net, 2
            )

            year.discount_factor = round(
                factor, 8
            )

            year.discounted_gross_cost = round(
                discounted_gross, 2
            )

            year.discounted_credits = round(
                discounted_credit, 2
            )

            year.discounted_net_cost = round(
                discounted_net, 2
            )

            year.cumulative_net_cost = round(
                cumulative, 2
            )

            year.cumulative_discounted_cost = round(
                cumulative_discounted, 2
            )

        return years

    # =====================================================
    # Category totals
    # =====================================================

    @staticmethod
    def category_totals(
        yearly_costs: Sequence[TCOYear],
    ) -> dict[str, float]:

        totals = {
            category.value: 0.0
            for category in TCOCategory
        }

        for year in yearly_costs:
            for category in TCOCategory:
                totals[category.value] += (
                    non_negative(
                        getattr(
                            year,
                            category.value,
                            0.0,
                        )
                    )
                )

        return {
            key: round(value, 2)
            for key, value in totals.items()
        }

    # =====================================================
    # Breakdown
    # =====================================================

    def build_breakdown(
        self,
        *,
        yearly_costs: Sequence[TCOYear],
    ) -> TCOBreakdown:

        totals = self.category_totals(
            yearly_costs
        )

        def total(
            *categories: TCOCategory,
        ) -> float:
            return sum(
                totals[category.value]
                for category in categories
            )

        capital = total(
            TCOCategory.ACQUISITION,
            TCOCategory.INSTALLATION,
            TCOCategory.ENGINEERING,
            TCOCategory.COMMISSIONING,
            TCOCategory.TRAINING,
            TCOCategory.REPLACEMENT,
        )

        maintenance_repair = total(
            TCOCategory.MAINTENANCE,
            TCOCategory.REPAIR,
            TCOCategory.OVERHAUL,
            TCOCategory.SPARE_PARTS,
            TCOCategory.LABOR,
            TCOCategory.INSPECTION,
            TCOCategory.CALIBRATION,
            TCOCategory.CONTRACTOR,
        )

        operating = total(
            TCOCategory.ENERGY,
            TCOCategory.UTILITIES,
            TCOCategory.SOFTWARE,
            TCOCategory.LICENSING,
            TCOCategory.SUPPORT,
            TCOCategory.LOGISTICS,
            TCOCategory.TAX,
            TCOCategory.INSURANCE,
        )

        reliability = total(
            TCOCategory.DOWNTIME,
            TCOCategory.EXPECTED_FAILURE,
        )

        production_quality = total(
            TCOCategory.PRODUCTION_LOSS,
            TCOCategory.SCRAP,
            TCOCategory.QUALITY,
        )

        safety_environmental = total(
            TCOCategory.SAFETY,
            TCOCategory.EMERGENCY,
            TCOCategory.ENVIRONMENTAL,
        )

        end_of_life = total(
            TCOCategory.DECOMMISSIONING,
            TCOCategory.DISPOSAL,
        )

        gross = sum(
            year.gross_cost
            for year in yearly_costs
        )

        credits = sum(
            year.credits
            for year in yearly_costs
        )

        net = sum(
            year.net_cost
            for year in yearly_costs
        )

        discounted = sum(
            year.discounted_net_cost
            for year in yearly_costs
        )

        return TCOBreakdown(
            category_totals=totals,

            capital_cost=round(
                capital, 2
            ),

            operating_cost=round(
                operating, 2
            ),

            maintenance_repair_cost=round(
                maintenance_repair, 2
            ),

            reliability_cost=round(
                reliability, 2
            ),

            production_quality_cost=round(
                production_quality, 2
            ),

            safety_environmental_cost=round(
                safety_environmental, 2
            ),

            end_of_life_cost=round(
                end_of_life, 2
            ),

            gross_lifecycle_cost=round(
                gross, 2
            ),

            recovery_credits=round(
                credits, 2
            ),

            net_lifecycle_cost=round(
                net, 2
            ),

            discounted_lifecycle_cost=round(
                discounted, 2
            ),
        )

    # =====================================================
    # Confidence
    # =====================================================

    @staticmethod
    def calculate_confidence(
        request: TCOInput,
    ) -> tuple[float, TCOConfidence]:

        scores: list[float] = [
            request.asset.confidence
        ]

        scores.extend(
            item.confidence
            for item in request.costs
            if item.include_in_tco
        )

        score = clamp(
            sum(scores) / len(scores)
            if scores
            else 0.0
        )

        if score >= 0.90:
            level = TCOConfidence.VERY_HIGH
        elif score >= 0.75:
            level = TCOConfidence.HIGH
        elif score >= 0.50:
            level = TCOConfidence.MODERATE
        else:
            level = TCOConfidence.LOW

        return round(score, 4), level

    # =====================================================
    # Economic risk
    # =====================================================

    def economic_risk(
        self,
        *,
        pv_tco: float,
        replacement_cost: float,
    ) -> TCOEconomicRisk:

        if replacement_cost <= 0:
            return TCOEconomicRisk.MODERATE

        ratio = (
            pv_tco
            / replacement_cost
        )

        if (
            ratio
            >= self.config.critical_replacement_ratio
        ):
            return TCOEconomicRisk.EXTREME

        if (
            ratio
            >= self.config.high_replacement_ratio
        ):
            return TCOEconomicRisk.CRITICAL

        if (
            ratio
            >= self.config.moderate_replacement_ratio
        ):
            return TCOEconomicRisk.HIGH

        if ratio >= 0.25:
            return TCOEconomicRisk.MODERATE

        return TCOEconomicRisk.LOW

    # =====================================================
    # Recommendation
    # =====================================================

    def recommend(
        self,
        *,
        request: TCOInput,
        pv_tco: float,
        confidence: float,
    ) -> TCORecommendation:

        asset = request.asset

        # Economics must never override safety.
        if (
            asset.critical_defect
            or asset.safety_risk_score
            >= self.config.critical_safety_risk
        ):
            return TCORecommendation.HUMAN_REVIEW

        if confidence < self.config.minimum_confidence:
            return TCORecommendation.INSUFFICIENT_DATA

        if asset.obsolete:
            return TCORecommendation.REPLACE

        if not asset.parts_available:
            return TCORecommendation.HUMAN_REVIEW

        replacement_cost = asset.replacement_cost

        if replacement_cost <= 0:
            return TCORecommendation.MONITOR

        ratio = pv_tco / replacement_cost

        if ratio >= 1.20:
            return TCORecommendation.REPLACE

        if ratio >= 0.80:
            return TCORecommendation.OVERHAUL

        if ratio >= 0.50:
            return TCORecommendation.REPAIR

        if ratio >= 0.25:
            return TCORecommendation.MAINTAIN

        return TCORecommendation.RETAIN

    # =====================================================
    # Main TCO
    # =====================================================

    def calculate(
        self,
        *,
        request: TCOInput,
        metadata: Mapping[str, Any] | None = None,
    ) -> TCOResult:

        warnings: list[str] = []

        yearly = self.build_yearly_costs(
            request=request
        )

        breakdown = self.build_breakdown(
            yearly_costs=yearly
        )

        gross_tco = (
            breakdown.gross_lifecycle_cost
        )

        credits = (
            breakdown.recovery_credits
        )

        net_tco = (
            breakdown.net_lifecycle_cost
        )

        pv_tco = (
            breakdown.discounted_lifecycle_cost
        )

        horizon = (
            request.assumptions.horizon_years
        )

        eac = self.equivalent_annual_cost(
            present_value=max(0.0, pv_tco),
            discount_rate=(
                request.assumptions.discount_rate
            ),
            years=horizon,
        )

        average_annual = (
            net_tco / horizon
            if horizon > 0
            else 0.0
        )

        asset_value_ratio = safe_divide(
            pv_tco,
            request.asset.current_asset_value,
        )

        replacement_ratio = safe_divide(
            pv_tco,
            request.asset.replacement_cost,
        )

        remaining_life_cost = None

        if (
            request.asset.remaining_useful_life_years
            is not None
            and request.asset.remaining_useful_life_years > 0
        ):
            remaining_life_cost = (
                pv_tco
                / request.asset.remaining_useful_life_years
            )

        confidence, confidence_level = (
            self.calculate_confidence(request)
        )

        risk = self.economic_risk(
            pv_tco=pv_tco,
            replacement_cost=(
                request.asset.replacement_cost
            ),
        )

        recommendation = self.recommend(
            request=request,
            pv_tco=pv_tco,
            confidence=confidence,
        )

        # -------------------------------------------------
        # Validation
        # -------------------------------------------------

        if not request.costs:
            warnings.append(
                "No lifecycle cost items were supplied."
            )

        if request.asset.replacement_cost <= 0:
            warnings.append(
                "Replacement cost is unavailable; "
                "replacement-relative TCO analysis is limited."
            )

        if (
            confidence
            < self.config.minimum_confidence
        ):
            warnings.append(
                "TCO estimate has low confidence."
            )

        categories = {
            item.category
            for item in request.costs
            if item.include_in_tco
        }

        if (
            TCOCategory.DOWNTIME in categories
            and TCOCategory.PRODUCTION_LOSS in categories
        ):
            warnings.append(
                "Downtime and production-loss costs are both "
                "included. Confirm downtime cost excludes "
                "embedded production loss."
            )

        if (
            TCOCategory.REPAIR in categories
            and TCOCategory.SPARE_PARTS in categories
        ):
            warnings.append(
                "Repair and spare-parts costs are both included. "
                "Confirm the repair amount does not already "
                "include parts."
            )

        if (
            TCOCategory.REPAIR in categories
            and TCOCategory.LABOR in categories
        ):
            warnings.append(
                "Repair and labor costs are both included. "
                "Confirm repair cost excludes technician labor."
            )

        if (
            TCOCategory.EXPECTED_FAILURE in categories
            and (
                TCOCategory.DOWNTIME in categories
                or TCOCategory.PRODUCTION_LOSS in categories
                or TCOCategory.SCRAP in categories
                or TCOCategory.EMERGENCY in categories
            )
        ):
            warnings.append(
                "Expected failure cost is combined with one or "
                "more failure-consequence categories. Confirm "
                "those consequences are not already embedded "
                "inside expected failure cost."
            )

        if (
            TCOCategory.SCRAP in categories
            and TCOCategory.QUALITY in categories
        ):
            warnings.append(
                "Scrap and quality costs are both included. "
                "Confirm quality cost does not already contain "
                "scrap/rework losses."
            )

        if (
            TCOCategory.ENERGY in categories
            and TCOCategory.UTILITIES in categories
        ):
            warnings.append(
                "Energy and utilities are both included. "
                "Confirm electricity or other utilities are "
                "not duplicated."
            )

        if (
            request.asset.critical_defect
            or request.asset.safety_risk_score
            >= self.config.critical_safety_risk
        ):
            warnings.append(
                "Safety-critical condition detected. TCO is "
                "informational only and must not override "
                "required safety actions or engineering controls."
            )

        if request.asset.obsolete:
            warnings.append(
                "Asset is marked obsolete; future support, "
                "parts availability, and escalation assumptions "
                "should be reviewed."
            )

        if not request.asset.parts_available:
            warnings.append(
                "Required parts are reported unavailable."
            )

        return TCOResult(
            calculation_id=new_id(),

            calculation_name=(
                request.calculation_name
            ),

            asset_id=request.asset.asset_id,
            component_id=(
                request.asset.component_id
            ),
            site_id=request.asset.site_id,

            scenario=request.scenario,

            horizon_years=horizon,

            gross_tco=round(
                gross_tco, 2
            ),

            total_credits=round(
                credits, 2
            ),

            net_tco=round(
                net_tco, 2
            ),

            present_value_tco=round(
                pv_tco, 2
            ),

            equivalent_annual_cost=round(
                eac, 2
            ),

            average_annual_cost=round(
                average_annual, 2
            ),

            tco_to_current_asset_value_ratio=(
                round(asset_value_ratio, 6)
                if asset_value_ratio is not None
                else None
            ),

            tco_to_replacement_cost_ratio=(
                round(replacement_ratio, 6)
                if replacement_ratio is not None
                else None
            ),

            cost_per_remaining_life_year=(
                round(remaining_life_cost, 2)
                if remaining_life_cost is not None
                else None
            ),

            yearly_costs=yearly,

            breakdown=breakdown,

            economic_risk=risk,

            confidence=confidence,
            confidence_level=confidence_level,

            recommendation=recommendation,

            currency=(
                request.assumptions.currency
                or self.config.default_currency
            ),

            warnings=list(
                dict.fromkeys(warnings)
            ),

            metadata={
                **request.metadata,
                **dict(metadata or {}),
            },
        )

    # =====================================================
    # Scenario analysis
    # =====================================================

    def scenario_analysis(
        self,
        *,
        request: TCOInput,
    ) -> dict[str, TCOResult]:

        results: dict[str, TCOResult] = {}

        for scenario in TCOScenario:
            scenario_request = TCOInput(
                asset=request.asset,
                costs=list(request.costs),
                assumptions=request.assumptions,
                credits=request.credits,
                scenario=scenario,
                include_initial_asset_value=(
                    request.include_initial_asset_value
                ),
                calculation_name=(
                    request.calculation_name
                ),
                metadata={
                    **request.metadata,
                    "scenario": scenario.value,
                },
            )

            results[scenario.value] = (
                self.calculate(
                    request=scenario_request
                )
            )

        return results

    # =====================================================
    # Compare lifecycle options
    # =====================================================

    def compare_options(
        self,
        *,
        options: Mapping[str, TCOInput],
    ) -> TCOComparisonResult:

        warnings: list[str] = []

        if not options:
            return TCOComparisonResult(
                comparison_id=new_id(),
                options={},
                lowest_pv_option=None,
                lowest_eac_option=None,
                pv_difference=None,
                recommended_option=None,
                warnings=[
                    "No TCO options were supplied."
                ],
            )

        results = {
            name: self.calculate(
                request=request
            )
            for name, request in options.items()
        }

        currencies = {
            result.currency
            for result in results.values()
        }

        if len(currencies) > 1:
            return TCOComparisonResult(
                comparison_id=new_id(),
                options=results,
                lowest_pv_option=None,
                lowest_eac_option=None,
                pv_difference=None,
                recommended_option=None,
                warnings=[
                    "Lifecycle options use multiple currencies. "
                    "Normalize currencies before comparing TCO."
                ],
            )

        # Safety-feasible options only.
        feasible = {
            name: result
            for name, result in results.items()
            if result.recommendation
            != TCORecommendation.HUMAN_REVIEW
        }

        if not feasible:
            return TCOComparisonResult(
                comparison_id=new_id(),
                options=results,
                lowest_pv_option=None,
                lowest_eac_option=None,
                pv_difference=None,
                recommended_option=None,
                warnings=[
                    "No option is eligible for automatic "
                    "economic selection because safety or "
                    "engineering review is required."
                ],
            )

        lowest_pv = min(
            feasible,
            key=lambda name: (
                feasible[name].present_value_tco
            ),
        )

        lowest_eac = min(
            feasible,
            key=lambda name: (
                feasible[name].equivalent_annual_cost
            ),
        )

        pv_values = sorted(
            result.present_value_tco
            for result in feasible.values()
        )

        difference = (
            pv_values[1] - pv_values[0]
            if len(pv_values) >= 2
            else None
        )

        recommended = lowest_eac

        if lowest_pv != lowest_eac:
            warnings.append(
                "Lowest present-value option differs from "
                "lowest equivalent-annual-cost option. "
                "Asset lives/horizons should be reviewed."
            )

        return TCOComparisonResult(
            comparison_id=new_id(),

            options=results,

            lowest_pv_option=lowest_pv,
            lowest_eac_option=lowest_eac,

            pv_difference=(
                round(difference, 2)
                if difference is not None
                else None
            ),

            recommended_option=recommended,

            warnings=warnings,
        )

    # =====================================================
    # Cost per operating hour
    # =====================================================

    @staticmethod
    def cost_per_operating_hour(
        *,
        total_cost: float,
        operating_hours: float,
    ) -> float | None:

        value = safe_divide(
            non_negative(total_cost),
            non_negative(operating_hours),
        )

        return (
            round(value, 4)
            if value is not None
            else None
        )

    # =====================================================
    # Cost per production unit
    # =====================================================

    @staticmethod
    def cost_per_production_unit(
        *,
        total_cost: float,
        production_units: float,
    ) -> float | None:

        value = safe_divide(
            non_negative(total_cost),
            non_negative(production_units),
        )

        return (
            round(value, 4)
            if value is not None
            else None
        )


# =========================================================
# Default calculator
# =========================================================

total_cost_of_ownership_calculator = (
    TotalCostOfOwnershipCalculator()
)


# =========================================================
# Agent-facing helpers
# =========================================================

def calculate_total_cost_of_ownership(
    *,
    request: TCOInput,
) -> TCOResult:

    return (
        total_cost_of_ownership_calculator
        .calculate(
            request=request
        )
    )


def calculate_tco_scenarios(
    *,
    request: TCOInput,
) -> dict[str, TCOResult]:

    return (
        total_cost_of_ownership_calculator
        .scenario_analysis(
            request=request
        )
    )


def compare_tco_options(
    *,
    options: Mapping[str, TCOInput],
) -> TCOComparisonResult:

    return (
        total_cost_of_ownership_calculator
        .compare_options(
            options=options
        )
    )


def calculate_tco_per_operating_hour(
    *,
    total_cost: float,
    operating_hours: float,
) -> float | None:

    return (
        total_cost_of_ownership_calculator
        .cost_per_operating_hour(
            total_cost=total_cost,
            operating_hours=operating_hours,
        )
    )


def calculate_tco_per_production_unit(
    *,
    total_cost: float,
    production_units: float,
) -> float | None:

    return (
        total_cost_of_ownership_calculator
        .cost_per_production_unit(
            total_cost=total_cost,
            production_units=production_units,
        )
    )


# =========================================================
# Exports
# =========================================================

__all__ = [
    "TCOCategory",
    "TCOCostTiming",
    "TCOScenario",
    "TCOEconomicRisk",
    "TCOConfidence",
    "TCORecommendation",

    "TCOCostItem",
    "TCOCredits",
    "TCOAssetContext",
    "TCOAssumptions",
    "TCOInput",

    "TCOYear",
    "TCOBreakdown",
    "TCOResult",
    "TCOComparisonResult",

    "TCOConfig",
    "TotalCostOfOwnershipCalculator",

    "total_cost_of_ownership_calculator",

    "calculate_total_cost_of_ownership",
    "calculate_tco_scenarios",
    "compare_tco_options",
    "calculate_tco_per_operating_hour",
    "calculate_tco_per_production_unit",
]