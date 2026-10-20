# app/agents/tools/roi_calculator.py

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

class ROIType(str, Enum):
    MAINTENANCE = "maintenance"
    REPAIR = "repair"
    REPLACEMENT = "replacement"
    ENERGY = "energy"
    SAFETY = "safety"
    PREDICTIVE_MAINTENANCE = "predictive_maintenance"
    FAILURE_PREVENTION = "failure_prevention"
    DOWNTIME_REDUCTION = "downtime_reduction"
    AUTOMATION = "automation"
    MODERNIZATION = "modernization"
    OTHER = "other"


class BenefitCategory(str, Enum):
    AVOIDED_FAILURE = "avoided_failure"
    AVOIDED_DOWNTIME = "avoided_downtime"
    PRODUCTION = "production"
    MAINTENANCE = "maintenance"
    ENERGY = "energy"
    QUALITY = "quality"
    SCRAP = "scrap"
    LABOR = "labor"
    INVENTORY = "inventory"
    SAFETY = "safety"
    EMERGENCY = "emergency"
    ENVIRONMENTAL = "environmental"
    PRODUCTIVITY = "productivity"
    REVENUE = "revenue"
    OTHER = "other"


class CostCategory(str, Enum):
    CAPITAL = "capital"
    IMPLEMENTATION = "implementation"
    INSTALLATION = "installation"
    INTEGRATION = "integration"
    LABOR = "labor"
    TRAINING = "training"
    SOFTWARE = "software"
    HARDWARE = "hardware"
    MAINTENANCE = "maintenance"
    ENERGY = "energy"
    DOWNTIME = "downtime"
    OPERATING = "operating"
    OTHER = "other"


class ROIRisk(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class ROIConfidence(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ROIRecommendation(str, Enum):
    STRONGLY_FAVORABLE = "strongly_favorable"
    FAVORABLE = "favorable"
    MARGINAL = "marginal"
    UNFAVORABLE = "unfavorable"
    INSUFFICIENT_DATA = "insufficient_data"
    MANDATORY = "mandatory"


# =========================================================
# Benefits
# =========================================================

@dataclass(slots=True)
class Benefit:
    category: BenefitCategory

    amount: float

    year: float = 1.0

    probability: float = 1.0

    confidence: float = 1.0

    recurring: bool = False

    annual_growth_rate: float = 0.0

    description: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.amount = non_negative(self.amount)
        self.year = non_negative(self.year)
        self.probability = clamp(self.probability)
        self.confidence = clamp(self.confidence)
        self.annual_growth_rate = finite(
            self.annual_growth_rate
        )

    @property
    def expected_amount(self) -> float:
        return (
            self.amount
            * self.probability
        )


# =========================================================
# Costs
# =========================================================

@dataclass(slots=True)
class Cost:
    category: CostCategory

    amount: float

    year: float = 0.0

    probability: float = 1.0

    confidence: float = 1.0

    recurring: bool = False

    annual_growth_rate: float = 0.0

    description: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.amount = non_negative(self.amount)
        self.year = non_negative(self.year)
        self.probability = clamp(self.probability)
        self.confidence = clamp(self.confidence)
        self.annual_growth_rate = finite(
            self.annual_growth_rate
        )

    @property
    def expected_amount(self) -> float:
        return (
            self.amount
            * self.probability
        )


# =========================================================
# ROI input
# =========================================================

@dataclass(slots=True)
class ROIInput:
    roi_type: ROIType

    initial_investment: float

    benefits: list[Benefit] = field(
        default_factory=list
    )

    costs: list[Cost] = field(
        default_factory=list
    )

    analysis_horizon_years: int = 5

    discount_rate: float = 0.08

    inflation_rate: float = 0.0

    residual_value: float = 0.0

    tax_credit: float = 0.0
    grant_credit: float = 0.0

    asset_id: str | None = None
    component_id: str | None = None
    site_id: str | None = None

    intervention_id: str | None = None

    currency: str = "INR"

    mandatory: bool = False
    safety_critical: bool = False

    confidence: float = 1.0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.initial_investment = non_negative(
            self.initial_investment
        )

        self.analysis_horizon_years = max(
            1,
            int(self.analysis_horizon_years),
        )

        self.discount_rate = finite(
            self.discount_rate
        )

        self.inflation_rate = finite(
            self.inflation_rate
        )

        self.residual_value = non_negative(
            self.residual_value
        )

        self.tax_credit = non_negative(
            self.tax_credit
        )

        self.grant_credit = non_negative(
            self.grant_credit
        )

        self.confidence = clamp(
            self.confidence
        )


# =========================================================
# Yearly cash flow
# =========================================================

@dataclass(slots=True)
class AnnualCashFlow:
    year: int

    benefits: float
    costs: float

    net_cash_flow: float

    discount_factor: float

    discounted_benefits: float
    discounted_costs: float
    discounted_net_cash_flow: float

    cumulative_cash_flow: float
    cumulative_discounted_cash_flow: float


# =========================================================
# Breakdown
# =========================================================

@dataclass(slots=True)
class ROIBreakdown:
    initial_investment: float

    benefit_by_category: dict[str, float]
    cost_by_category: dict[str, float]

    gross_benefits: float
    operating_costs: float

    residual_value: float
    credits: float

    net_benefits: float

    discounted_benefits: float
    discounted_costs: float

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# =========================================================
# Result
# =========================================================

@dataclass(slots=True)
class ROIResult:
    calculation_id: str

    asset_id: str | None
    component_id: str | None
    site_id: str | None

    intervention_id: str | None

    roi_type: ROIType

    initial_investment: float

    total_benefits: float
    total_costs: float

    net_benefit: float

    simple_roi_percent: float | None

    npv: float

    discounted_roi_percent: float | None

    benefit_cost_ratio: float | None

    payback_period_years: float | None
    discounted_payback_period_years: float | None

    annualized_roi_percent: float | None

    profitability_index: float | None

    annual_cash_flows: list[AnnualCashFlow]

    breakdown: ROIBreakdown

    economic_risk: ROIRisk

    confidence: float
    confidence_level: ROIConfidence

    recommendation: ROIRecommendation

    mandatory: bool
    safety_critical: bool

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
# Config
# =========================================================

@dataclass(slots=True)
class ROICalculatorConfig:
    favorable_roi_percent: float = 20.0
    strong_roi_percent: float = 50.0

    marginal_roi_percent: float = 0.0

    high_risk_payback_years: float = 5.0

    minimum_confidence: float = 0.50

    default_discount_rate: float = 0.08

    default_currency: str = "INR"


# =========================================================
# Calculator
# =========================================================

class ROICalculator:

    def __init__(
        self,
        *,
        config: ROICalculatorConfig | None = None,
    ) -> None:
        self.config = (
            config
            or ROICalculatorConfig()
        )

    # =====================================================
    # Discounting
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

    # =====================================================
    # Expand recurring item
    # =====================================================

    @staticmethod
    def recurring_values(
        *,
        amount: float,
        start_year: int,
        horizon: int,
        growth_rate: float,
    ) -> dict[int, float]:

        result: dict[int, float] = {}

        if start_year > horizon:
            return result

        current = non_negative(amount)

        for year in range(
            max(1, start_year),
            horizon + 1,
        ):
            result[year] = current

            current *= (
                1.0 + growth_rate
            )

            current = max(
                0.0,
                current,
            )

        return result

    # =====================================================
    # Benefit schedule
    # =====================================================

    def benefit_schedule(
        self,
        *,
        benefits: Sequence[Benefit],
        horizon: int,
    ) -> dict[int, float]:

        schedule = {
            year: 0.0
            for year in range(
                0,
                horizon + 1,
            )
        }

        for benefit in benefits:
            amount = benefit.expected_amount

            start = max(
                0,
                int(math.floor(benefit.year)),
            )

            if benefit.recurring:
                values = self.recurring_values(
                    amount=amount,
                    start_year=max(1, start),
                    horizon=horizon,
                    growth_rate=(
                        benefit.annual_growth_rate
                    ),
                )

                for year, value in values.items():
                    schedule[year] += value

            elif start <= horizon:
                schedule[start] += amount

        return schedule

    # =====================================================
    # Cost schedule
    # =====================================================

    def cost_schedule(
        self,
        *,
        costs: Sequence[Cost],
        horizon: int,
    ) -> dict[int, float]:

        schedule = {
            year: 0.0
            for year in range(
                0,
                horizon + 1,
            )
        }

        for cost in costs:
            amount = cost.expected_amount

            start = max(
                0,
                int(math.floor(cost.year)),
            )

            if cost.recurring:
                values = self.recurring_values(
                    amount=amount,
                    start_year=max(1, start),
                    horizon=horizon,
                    growth_rate=(
                        cost.annual_growth_rate
                    ),
                )

                for year, value in values.items():
                    schedule[year] += value

            elif start <= horizon:
                schedule[start] += amount

        return schedule

    # =====================================================
    # Cash flows
    # =====================================================

    def build_cash_flows(
        self,
        *,
        roi: ROIInput,
    ) -> list[AnnualCashFlow]:

        horizon = (
            roi.analysis_horizon_years
        )

        benefits = self.benefit_schedule(
            benefits=roi.benefits,
            horizon=horizon,
        )

        costs = self.cost_schedule(
            costs=roi.costs,
            horizon=horizon,
        )

        # Investment occurs at t=0.
        costs[0] += roi.initial_investment

        # Credits reduce initial cost.
        credits = (
            roi.tax_credit
            + roi.grant_credit
        )

        costs[0] = max(
            0.0,
            costs[0] - credits,
        )

        # Residual value is a terminal benefit.
        benefits[horizon] += (
            roi.residual_value
        )

        flows: list[AnnualCashFlow] = []

        cumulative = 0.0
        cumulative_discounted = 0.0

        for year in range(
            0,
            horizon + 1,
        ):
            benefit = benefits[year]
            cost = costs[year]

            net = benefit - cost

            factor = self.discount_factor(
                rate=roi.discount_rate,
                year=year,
            )

            pv_benefit = (
                benefit * factor
            )

            pv_cost = (
                cost * factor
            )

            pv_net = (
                net * factor
            )

            cumulative += net
            cumulative_discounted += pv_net

            flows.append(
                AnnualCashFlow(
                    year=year,

                    benefits=round(
                        benefit,
                        2,
                    ),

                    costs=round(
                        cost,
                        2,
                    ),

                    net_cash_flow=round(
                        net,
                        2,
                    ),

                    discount_factor=round(
                        factor,
                        8,
                    ),

                    discounted_benefits=round(
                        pv_benefit,
                        2,
                    ),

                    discounted_costs=round(
                        pv_cost,
                        2,
                    ),

                    discounted_net_cash_flow=round(
                        pv_net,
                        2,
                    ),

                    cumulative_cash_flow=round(
                        cumulative,
                        2,
                    ),

                    cumulative_discounted_cash_flow=round(
                        cumulative_discounted,
                        2,
                    ),
                )
            )

        return flows

    # =====================================================
    # Payback
    # =====================================================

    @staticmethod
    def payback_period(
        flows: Sequence[AnnualCashFlow],
        *,
        discounted: bool = False,
    ) -> float | None:

        cumulative = 0.0

        for index, flow in enumerate(flows):
            value = (
                flow.discounted_net_cash_flow
                if discounted
                else flow.net_cash_flow
            )

            previous = cumulative
            cumulative += value

            if cumulative >= 0:
                if index == 0:
                    return 0.0

                if value <= 0:
                    return float(flow.year)

                remaining = abs(previous)

                fraction = min(
                    1.0,
                    remaining / value,
                )

                return (
                    float(flow.year - 1)
                    + fraction
                )

        return None

    # =====================================================
    # Confidence
    # =====================================================

    @staticmethod
    def confidence(
        roi: ROIInput,
    ) -> tuple[float, ROIConfidence]:

        scores: list[float] = [
            roi.confidence
        ]

        scores.extend(
            benefit.confidence
            for benefit in roi.benefits
        )

        scores.extend(
            cost.confidence
            for cost in roi.costs
        )

        score = clamp(
            sum(scores) / len(scores)
            if scores
            else 0.0
        )

        if score >= 0.90:
            level = ROIConfidence.VERY_HIGH
        elif score >= 0.75:
            level = ROIConfidence.HIGH
        elif score >= 0.50:
            level = ROIConfidence.MODERATE
        else:
            level = ROIConfidence.LOW

        return round(score, 4), level

    # =====================================================
    # Risk
    # =====================================================

    def economic_risk(
        self,
        *,
        npv: float,
        roi_percent: float | None,
        payback_years: float | None,
    ) -> ROIRisk:

        if (
            npv < 0
            and (
                roi_percent is None
                or roi_percent < 0
            )
        ):
            return ROIRisk.CRITICAL

        if payback_years is None:
            return ROIRisk.HIGH

        if (
            payback_years
            > self.config.high_risk_payback_years
        ):
            return ROIRisk.HIGH

        if (
            roi_percent is not None
            and roi_percent
            < self.config.favorable_roi_percent
        ):
            return ROIRisk.MODERATE

        return ROIRisk.LOW

    # =====================================================
    # Recommendation
    # =====================================================

    def recommend(
        self,
        *,
        roi: ROIInput,
        roi_percent: float | None,
        npv: float,
        confidence: float,
    ) -> ROIRecommendation:

        # Mandatory safety/regulatory interventions
        # are not optional economic decisions.
        if roi.mandatory:
            return ROIRecommendation.MANDATORY

        if confidence < (
            self.config.minimum_confidence
        ):
            return (
                ROIRecommendation
                .INSUFFICIENT_DATA
            )

        if roi_percent is None:
            return (
                ROIRecommendation
                .INSUFFICIENT_DATA
            )

        if (
            roi_percent
            >= self.config.strong_roi_percent
            and npv > 0
        ):
            return (
                ROIRecommendation
                .STRONGLY_FAVORABLE
            )

        if (
            roi_percent
            >= self.config.favorable_roi_percent
            and npv > 0
        ):
            return (
                ROIRecommendation
                .FAVORABLE
            )

        if (
            roi_percent
            >= self.config.marginal_roi_percent
            and npv >= 0
        ):
            return (
                ROIRecommendation
                .MARGINAL
            )

        return ROIRecommendation.UNFAVORABLE

    # =====================================================
    # Breakdown
    # =====================================================

    @staticmethod
    def build_breakdown(
        *,
        roi: ROIInput,
        flows: Sequence[AnnualCashFlow],
    ) -> ROIBreakdown:

        benefit_categories: dict[str, float] = {}

        for item in roi.benefits:
            key = item.category.value

            benefit_categories[key] = (
                benefit_categories.get(
                    key,
                    0.0,
                )
                + item.expected_amount
            )

        cost_categories: dict[str, float] = {}

        for item in roi.costs:
            key = item.category.value

            cost_categories[key] = (
                cost_categories.get(
                    key,
                    0.0,
                )
                + item.expected_amount
            )

        gross_benefits = sum(
            flow.benefits
            for flow in flows
        )

        total_costs = sum(
            flow.costs
            for flow in flows
        )

        discounted_benefits = sum(
            flow.discounted_benefits
            for flow in flows
        )

        discounted_costs = sum(
            flow.discounted_costs
            for flow in flows
        )

        return ROIBreakdown(
            initial_investment=round(
                roi.initial_investment,
                2,
            ),

            benefit_by_category={
                key: round(value, 2)
                for key, value
                in benefit_categories.items()
            },

            cost_by_category={
                key: round(value, 2)
                for key, value
                in cost_categories.items()
            },

            gross_benefits=round(
                gross_benefits,
                2,
            ),

            operating_costs=round(
                max(
                    0.0,
                    total_costs
                    - roi.initial_investment,
                ),
                2,
            ),

            residual_value=round(
                roi.residual_value,
                2,
            ),

            credits=round(
                roi.tax_credit
                + roi.grant_credit,
                2,
            ),

            net_benefits=round(
                gross_benefits
                - total_costs,
                2,
            ),

            discounted_benefits=round(
                discounted_benefits,
                2,
            ),

            discounted_costs=round(
                discounted_costs,
                2,
            ),
        )

    # =====================================================
    # Main calculation
    # =====================================================

    def calculate(
        self,
        *,
        roi: ROIInput,
        metadata: Mapping[str, Any] | None = None,
    ) -> ROIResult:

        warnings: list[str] = []

        flows = self.build_cash_flows(
            roi=roi
        )

        total_benefits = sum(
            flow.benefits
            for flow in flows
        )

        total_costs = sum(
            flow.costs
            for flow in flows
        )

        net_benefit = (
            total_benefits
            - total_costs
        )

        simple_roi = safe_divide(
            net_benefit,
            total_costs,
        )

        if simple_roi is not None:
            simple_roi *= 100.0

        discounted_benefits = sum(
            flow.discounted_benefits
            for flow in flows
        )

        discounted_costs = sum(
            flow.discounted_costs
            for flow in flows
        )

        npv = (
            discounted_benefits
            - discounted_costs
        )

        discounted_roi = safe_divide(
            npv,
            discounted_costs,
        )

        if discounted_roi is not None:
            discounted_roi *= 100.0

        bcr = safe_divide(
            discounted_benefits,
            discounted_costs,
        )

        profitability_index = bcr

        payback = self.payback_period(
            flows
        )

        discounted_payback = (
            self.payback_period(
                flows,
                discounted=True,
            )
        )

        annualized_roi = None

        if (
            simple_roi is not None
            and roi.analysis_horizon_years > 0
        ):
            total_return_factor = (
                1.0
                + simple_roi / 100.0
            )

            if total_return_factor > 0:
                annualized_roi = (
                    (
                        total_return_factor
                        ** (
                            1.0
                            / roi.analysis_horizon_years
                        )
                    )
                    - 1.0
                ) * 100.0

        confidence, confidence_level = (
            self.confidence(roi)
        )

        risk = self.economic_risk(
            npv=npv,
            roi_percent=simple_roi,
            payback_years=payback,
        )

        recommendation = self.recommend(
            roi=roi,
            roi_percent=simple_roi,
            npv=npv,
            confidence=confidence,
        )

        breakdown = self.build_breakdown(
            roi=roi,
            flows=flows,
        )

        # ---------------------------------------------
        # Validation / warnings
        # ---------------------------------------------

        if roi.initial_investment <= 0:
            warnings.append(
                "Initial investment is zero. ROI percentage "
                "may not be meaningful for this case."
            )

        if not roi.benefits:
            warnings.append(
                "No benefits were supplied."
            )

        if confidence < (
            self.config.minimum_confidence
        ):
            warnings.append(
                "ROI estimate has low confidence."
            )

        if npv < 0:
            warnings.append(
                "Net present value is negative over the "
                "selected analysis horizon."
            )

        if payback is None:
            warnings.append(
                "Investment does not achieve simple payback "
                "within the analysis horizon."
            )

        if discounted_payback is None:
            warnings.append(
                "Investment does not achieve discounted "
                "payback within the analysis horizon."
            )

        if roi.safety_critical:
            warnings.append(
                "Safety-critical intervention ROI is "
                "informational only. Financial return must "
                "not override required safety controls."
            )

        if roi.mandatory:
            warnings.append(
                "This intervention is mandatory. Negative "
                "ROI, NPV, or BCR must not be used to reject "
                "a safety, regulatory, or engineering requirement."
            )

        # Check common double-counting risks.
        benefit_types = {
            item.category
            for item in roi.benefits
        }

        if (
            BenefitCategory.AVOIDED_DOWNTIME
            in benefit_types
            and BenefitCategory.PRODUCTION
            in benefit_types
        ):
            warnings.append(
                "Avoided downtime and production benefits "
                "are both included. Confirm production loss "
                "is not already embedded in downtime savings."
            )

        if (
            BenefitCategory.AVOIDED_FAILURE
            in benefit_types
            and BenefitCategory.EMERGENCY
            in benefit_types
        ):
            warnings.append(
                "Avoided failure and emergency benefits are "
                "both included. Confirm emergency consequence "
                "is not already included in avoided failure cost."
            )

        return ROIResult(
            calculation_id=new_id(),

            asset_id=roi.asset_id,
            component_id=roi.component_id,
            site_id=roi.site_id,

            intervention_id=(
                roi.intervention_id
            ),

            roi_type=roi.roi_type,

            initial_investment=round(
                roi.initial_investment,
                2,
            ),

            total_benefits=round(
                total_benefits,
                2,
            ),

            total_costs=round(
                total_costs,
                2,
            ),

            net_benefit=round(
                net_benefit,
                2,
            ),

            simple_roi_percent=(
                round(simple_roi, 2)
                if simple_roi is not None
                else None
            ),

            npv=round(
                npv,
                2,
            ),

            discounted_roi_percent=(
                round(discounted_roi, 2)
                if discounted_roi is not None
                else None
            ),

            benefit_cost_ratio=(
                round(bcr, 4)
                if bcr is not None
                else None
            ),

            payback_period_years=(
                round(payback, 4)
                if payback is not None
                else None
            ),

            discounted_payback_period_years=(
                round(
                    discounted_payback,
                    4,
                )
                if discounted_payback is not None
                else None
            ),

            annualized_roi_percent=(
                round(
                    annualized_roi,
                    2,
                )
                if annualized_roi is not None
                else None
            ),

            profitability_index=(
                round(
                    profitability_index,
                    4,
                )
                if profitability_index
                is not None
                else None
            ),

            annual_cash_flows=flows,

            breakdown=breakdown,

            economic_risk=risk,

            confidence=confidence,
            confidence_level=(
                confidence_level
            ),

            recommendation=(
                recommendation
            ),

            mandatory=roi.mandatory,
            safety_critical=(
                roi.safety_critical
            ),

            currency=(
                roi.currency
                or self.config.default_currency
            ),

            warnings=list(
                dict.fromkeys(warnings)
            ),

            metadata={
                **roi.metadata,
                **dict(metadata or {}),
            },
        )

    # =====================================================
    # Simple ROI
    # =====================================================

    @staticmethod
    def simple_roi(
        *,
        investment: float,
        total_benefit: float,
    ) -> float | None:

        investment = non_negative(
            investment
        )

        benefit = non_negative(
            total_benefit
        )

        if investment <= 0:
            return None

        return (
            (benefit - investment)
            / investment
            * 100.0
        )

    # =====================================================
    # Benefit-cost ratio
    # =====================================================

    @staticmethod
    def benefit_cost_ratio(
        *,
        benefits: float,
        costs: float,
    ) -> float | None:

        return safe_divide(
            non_negative(benefits),
            non_negative(costs),
        )

    # =====================================================
    # Break-even benefit
    # =====================================================

    @staticmethod
    def break_even_benefit(
        *,
        investment: float,
        additional_costs: float = 0.0,
    ) -> float:

        return (
            non_negative(investment)
            + non_negative(
                additional_costs
            )
        )

    # =====================================================
    # Break-even risk reduction
    # =====================================================

    @staticmethod
    def required_risk_reduction(
        *,
        intervention_cost: float,
        failure_consequence_cost: float,
    ) -> float | None:

        consequence = non_negative(
            failure_consequence_cost
        )

        if consequence <= 0:
            return None

        return clamp(
            non_negative(
                intervention_cost
            )
            / consequence
        )

    # =====================================================
    # Scenario ROI
    # =====================================================

    def scenario_analysis(
        self,
        *,
        base: ROIInput,
        benefit_multipliers: Sequence[
            float
        ] = (0.75, 1.0, 1.25),
        cost_multipliers: Sequence[
            float
        ] = (1.25, 1.0, 0.90),
    ) -> list[dict[str, Any]]:

        results: list[
            dict[str, Any]
        ] = []

        count = min(
            len(benefit_multipliers),
            len(cost_multipliers),
        )

        for index in range(count):
            benefit_multiplier = max(
                0.0,
                finite(
                    benefit_multipliers[
                        index
                    ]
                ),
            )

            cost_multiplier = max(
                0.0,
                finite(
                    cost_multipliers[
                        index
                    ]
                ),
            )

            benefits = [
                Benefit(
                    category=item.category,
                    amount=(
                        item.amount
                        * benefit_multiplier
                    ),
                    year=item.year,
                    probability=item.probability,
                    confidence=item.confidence,
                    recurring=item.recurring,
                    annual_growth_rate=(
                        item.annual_growth_rate
                    ),
                    description=item.description,
                    metadata=dict(
                        item.metadata
                    ),
                )
                for item in base.benefits
            ]

            costs = [
                Cost(
                    category=item.category,
                    amount=(
                        item.amount
                        * cost_multiplier
                    ),
                    year=item.year,
                    probability=item.probability,
                    confidence=item.confidence,
                    recurring=item.recurring,
                    annual_growth_rate=(
                        item.annual_growth_rate
                    ),
                    description=item.description,
                    metadata=dict(
                        item.metadata
                    ),
                )
                for item in base.costs
            ]

            scenario = ROIInput(
                roi_type=base.roi_type,

                initial_investment=(
                    base.initial_investment
                    * cost_multiplier
                ),

                benefits=benefits,
                costs=costs,

                analysis_horizon_years=(
                    base.analysis_horizon_years
                ),

                discount_rate=(
                    base.discount_rate
                ),

                inflation_rate=(
                    base.inflation_rate
                ),

                residual_value=(
                    base.residual_value
                ),

                tax_credit=base.tax_credit,
                grant_credit=base.grant_credit,

                asset_id=base.asset_id,
                component_id=(
                    base.component_id
                ),
                site_id=base.site_id,

                intervention_id=(
                    base.intervention_id
                ),

                currency=base.currency,

                mandatory=base.mandatory,
                safety_critical=(
                    base.safety_critical
                ),

                confidence=base.confidence,

                metadata={
                    **base.metadata,
                    "scenario_index": index,
                },
            )

            result = self.calculate(
                roi=scenario
            )

            results.append(
                {
                    "scenario_index": index,
                    "benefit_multiplier": (
                        benefit_multiplier
                    ),
                    "cost_multiplier": (
                        cost_multiplier
                    ),
                    "roi_percent": (
                        result
                        .simple_roi_percent
                    ),
                    "npv": result.npv,
                    "bcr": (
                        result
                        .benefit_cost_ratio
                    ),
                    "payback_years": (
                        result
                        .payback_period_years
                    ),
                    "recommendation": (
                        result
                        .recommendation.value
                    ),
                }
            )

        return results


# =========================================================
# Default calculator
# =========================================================

roi_calculator = ROICalculator()


# =========================================================
# Agent-facing helpers
# =========================================================

def calculate_roi(
    *,
    roi: ROIInput,
) -> ROIResult:
    return roi_calculator.calculate(
        roi=roi
    )


def calculate_simple_roi(
    *,
    investment: float,
    total_benefit: float,
) -> float | None:
    return roi_calculator.simple_roi(
        investment=investment,
        total_benefit=total_benefit,
    )


def calculate_benefit_cost_ratio(
    *,
    benefits: float,
    costs: float,
) -> float | None:
    return (
        roi_calculator
        .benefit_cost_ratio(
            benefits=benefits,
            costs=costs,
        )
    )


def calculate_required_risk_reduction(
    *,
    intervention_cost: float,
    failure_consequence_cost: float,
) -> float | None:
    return (
        roi_calculator
        .required_risk_reduction(
            intervention_cost=(
                intervention_cost
            ),
            failure_consequence_cost=(
                failure_consequence_cost
            ),
        )
    )


# =========================================================
# Exports
# =========================================================

__all__ = [
    "ROIType",
    "BenefitCategory",
    "CostCategory",
    "ROIRisk",
    "ROIConfidence",
    "ROIRecommendation",

    "Benefit",
    "Cost",
    "ROIInput",

    "AnnualCashFlow",
    "ROIBreakdown",
    "ROIResult",

    "ROICalculatorConfig",
    "ROICalculator",

    "roi_calculator",

    "calculate_roi",
    "calculate_simple_roi",
    "calculate_benefit_cost_ratio",
    "calculate_required_risk_reduction",
]