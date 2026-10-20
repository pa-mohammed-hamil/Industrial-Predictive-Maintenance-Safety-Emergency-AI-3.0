
"""
app/services/roi_service.py

Industrial Predictive Maintenance, Safety & Emergency AI
========================================================

Return on Investment (ROI) Analysis Service

Features:
- Predictive maintenance ROI
- Preventive maintenance ROI
- Industrial AI deployment ROI
- Equipment repair ROI
- Equipment replacement ROI
- Downtime reduction savings
- Avoided failure cost
- Avoided production losses
- Maintenance efficiency savings
- Energy efficiency savings
- Safety cost avoidance
- Annual net benefits
- Payback period
- Discounted cash flow
- Net Present Value (NPV)
- Internal Rate of Return (IRR)
- Benefit-Cost Ratio (BCR)
- Multi-year ROI forecasting
- Optimistic/base/pessimistic scenarios
- Investment comparison
- Optional async repository persistence
- Optional audit integration

Financial conventions:
- ROI = (total benefits - total costs) / total costs * 100
- NPV discounts future cash flows to present value.
- IRR is calculated for conventional cash-flow patterns.
- Payback is based on cumulative net cash flows.
- Savings must be incremental and non-overlapping.
- Avoided safety costs are financial estimates, not
  evidence of a reduction in actual safety risk.
- Recommendations are advisory, not financial guarantees.

Expected optional repository:
    async create_roi_analysis(data)
    async get_roi_analysis(analysis_id)
    async list_roi_analyses(filters)

Optional audit service:
    async record_event(data)
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import logging

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from enum import Enum
from typing import Any, Mapping, Optional, Sequence
from uuid import uuid4


logger = logging.getLogger(__name__)

ZERO = Decimal("0")
ONE = Decimal("1")
HUNDRED = Decimal("100")
MONEY_QUANT = Decimal("0.01")
PERCENT_QUANT = Decimal("0.01")


# ============================================================
# EXCEPTIONS
# ============================================================

class ROIServiceError(Exception):
    """Base ROI service exception."""


class ROIValidationError(ROIServiceError):
    """Invalid ROI calculation input."""


class ROIAnalysisNotFoundError(ROIServiceError):
    """ROI analysis was not found."""


class ROIRepositoryError(ROIServiceError):
    """ROI repository operation failed."""


# ============================================================
# ENUMS
# ============================================================

class InvestmentType(str, Enum):
    PREDICTIVE_MAINTENANCE = "predictive_maintenance"
    PREVENTIVE_MAINTENANCE = "preventive_maintenance"
    AI_DEPLOYMENT = "ai_deployment"
    EQUIPMENT_REPAIR = "equipment_repair"
    EQUIPMENT_REPLACEMENT = "equipment_replacement"
    SAFETY_UPGRADE = "safety_upgrade"
    ENERGY_OPTIMIZATION = "energy_optimization"
    AUTOMATION = "automation"
    DIGITAL_TRANSFORMATION = "digital_transformation"
    OTHER = "other"


class ROIScenario(str, Enum):
    OPTIMISTIC = "optimistic"
    BASE = "base"
    PESSIMISTIC = "pessimistic"


class InvestmentDecision(str, Enum):
    FINANCIALLY_ATTRACTIVE = "financially_attractive"
    MARGINAL = "marginal"
    NOT_FINANCIALLY_ATTRACTIVE = "not_financially_attractive"
    REVIEW_REQUIRED = "review_required"


# ============================================================
# HELPERS
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def decimal_value(
    value: Any,
    name: str,
    minimum: Optional[Decimal] = None,
) -> Decimal:

    if isinstance(value, bool):
        raise ROIValidationError(
            f"{name} must be numeric"
        )

    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ROIValidationError(
            f"Invalid numeric value for {name}"
        ) from exc

    if not result.is_finite():
        raise ROIValidationError(
            f"{name} must be finite"
        )

    if minimum is not None and result < minimum:
        raise ROIValidationError(
            f"{name} must be >= {minimum}"
        )

    return result


def money(value: Any) -> Decimal:
    return decimal_value(
        value,
        "money",
    ).quantize(
        MONEY_QUANT,
        rounding=ROUND_HALF_UP,
    )


def percent(
    value: Any,
    name: str,
    maximum: Optional[Decimal] = HUNDRED,
) -> Decimal:

    result = decimal_value(
        value,
        name,
        ZERO,
    )

    if maximum is not None and result > maximum:
        raise ROIValidationError(
            f"{name} cannot exceed {maximum}"
        )

    return result


def required_text(
    value: Any,
    name: str,
    max_length: int = 255,
) -> str:

    if not isinstance(value, str):
        raise ROIValidationError(
            f"{name} must be a string"
        )

    result = value.strip()

    if not result:
        raise ROIValidationError(
            f"{name} cannot be empty"
        )

    if len(result) > max_length:
        raise ROIValidationError(
            f"{name} exceeds {max_length} characters"
        )

    return result


def optional_text(
    value: Optional[str],
    name: str,
) -> Optional[str]:

    if value is None:
        return None

    return required_text(value, name)


def currency_code(value: str) -> str:

    code = required_text(
        value,
        "currency",
        3,
    ).upper()

    if (
        len(code) != 3
        or not code.isascii()
        or not code.isalpha()
    ):
        raise ROIValidationError(
            "Currency must be a three-letter ASCII code"
        )

    return code


def parse_enum(
    enum_class: type[Enum],
    value: Any,
    name: str,
) -> Enum:

    try:
        return enum_class(value)
    except (ValueError, TypeError) as exc:
        raise ROIValidationError(
            f"Invalid {name}: {value}"
        ) from exc


def positive_years(
    value: Any,
    maximum: int = 50,
) -> int:

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= maximum
    ):
        raise ROIValidationError(
            f"analysis_years must be between 1 and {maximum}"
        )

    return value


def serialize(value: Any) -> Any:

    if isinstance(value, Decimal):
        return str(value)

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if hasattr(value, "__dataclass_fields__"):
        return serialize(asdict(value))

    if isinstance(value, Mapping):
        return {
            str(key): serialize(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [serialize(item) for item in value]

    return value


async def invoke(
    method: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:

    result = method(*args, **kwargs)

    if inspect.isawaitable(result):
        return await result

    return result


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class ROIBenefits:
    avoided_failure_cost: Decimal = ZERO
    downtime_savings: Decimal = ZERO
    production_loss_savings: Decimal = ZERO
    maintenance_savings: Decimal = ZERO
    energy_savings: Decimal = ZERO
    labor_savings: Decimal = ZERO
    safety_cost_avoidance: Decimal = ZERO
    quality_savings: Decimal = ZERO
    other_savings: Decimal = ZERO

    @property
    def total(self) -> Decimal:
        return money(
            sum(
                (
                    value
                    for value in asdict(self).values()
                ),
                ZERO,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **serialize(self),
            "total": str(self.total),
        }


@dataclass(frozen=True)
class ROIExpenses:
    initial_investment: Decimal
    annual_operating_cost: Decimal = ZERO
    annual_maintenance_cost: Decimal = ZERO
    annual_subscription_cost: Decimal = ZERO
    annual_training_cost: Decimal = ZERO
    annual_other_cost: Decimal = ZERO

    @property
    def annual_recurring_cost(self) -> Decimal:
        return money(
            self.annual_operating_cost
            + self.annual_maintenance_cost
            + self.annual_subscription_cost
            + self.annual_training_cost
            + self.annual_other_cost
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **serialize(self),
            "annual_recurring_cost": str(
                self.annual_recurring_cost
            ),
        }


@dataclass(frozen=True)
class AnnualROICashFlow:
    year: int
    benefits: Decimal
    expenses: Decimal
    net_cash_flow: Decimal
    discounted_cash_flow: Decimal
    cumulative_cash_flow: Decimal

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


@dataclass
class ROIAnalysisResult:
    analysis_id: str
    company_id: str
    site_id: str
    asset_id: Optional[str]

    investment_name: str
    investment_type: str
    scenario: str
    currency: str

    analysis_years: int
    discount_rate_percent: Decimal
    benefit_growth_percent: Decimal
    expense_growth_percent: Decimal

    benefits: ROIBenefits
    expenses: ROIExpenses

    annual_cash_flows: list[AnnualROICashFlow]

    total_benefits: Decimal
    total_costs: Decimal
    total_net_benefit: Decimal

    roi_percent: Optional[Decimal]
    npv: Decimal
    irr_percent: Optional[Decimal]
    payback_years: Optional[Decimal]
    discounted_payback_years: Optional[Decimal]
    benefit_cost_ratio: Optional[Decimal]

    recommendation: str
    explanation: str

    created_by: str
    created_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


@dataclass(frozen=True)
class ROIComparisonResult:
    currency: str
    best_npv_analysis_id: str
    best_roi_analysis_id: Optional[str]
    comparisons: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


# ============================================================
# ROI SERVICE
# ============================================================

class ROIService:
    """
    Industrial investment ROI analysis.

    Calculations are deterministic and do not require
    an external AI model.

    ROI projections must be validated against actual
    operating data before investment decisions.
    """

    SCENARIO_BENEFIT_MULTIPLIERS = {
        ROIScenario.OPTIMISTIC: Decimal("1.20"),
        ROIScenario.BASE: Decimal("1.00"),
        ROIScenario.PESSIMISTIC: Decimal("0.75"),
    }

    SCENARIO_EXPENSE_MULTIPLIERS = {
        ROIScenario.OPTIMISTIC: Decimal("0.90"),
        ROIScenario.BASE: Decimal("1.00"),
        ROIScenario.PESSIMISTIC: Decimal("1.20"),
    }

    def __init__(
        self,
        roi_repository: Any = None,
        audit_service: Any = None,
        default_currency: str = "INR",
    ):

        self.repository = roi_repository
        self.audit = audit_service
        self.default_currency = currency_code(
            default_currency
        )

    # ========================================================
    # BENEFITS
    # ========================================================

    @staticmethod
    def calculate_benefits(
        *,
        avoided_failure_cost: Any = 0,
        downtime_savings: Any = 0,
        production_loss_savings: Any = 0,
        maintenance_savings: Any = 0,
        energy_savings: Any = 0,
        labor_savings: Any = 0,
        safety_cost_avoidance: Any = 0,
        quality_savings: Any = 0,
        other_savings: Any = 0,
    ) -> ROIBenefits:

        values = {
            "avoided_failure_cost": avoided_failure_cost,
            "downtime_savings": downtime_savings,
            "production_loss_savings": (
                production_loss_savings
            ),
            "maintenance_savings": maintenance_savings,
            "energy_savings": energy_savings,
            "labor_savings": labor_savings,
            "safety_cost_avoidance": (
                safety_cost_avoidance
            ),
            "quality_savings": quality_savings,
            "other_savings": other_savings,
        }

        normalized = {
            key: money(
                decimal_value(
                    value,
                    key,
                    ZERO,
                )
            )
            for key, value in values.items()
        }

        return ROIBenefits(**normalized)

    # ========================================================
    # EXPENSES
    # ========================================================

    @staticmethod
    def calculate_expenses(
        *,
        initial_investment: Any,
        annual_operating_cost: Any = 0,
        annual_maintenance_cost: Any = 0,
        annual_subscription_cost: Any = 0,
        annual_training_cost: Any = 0,
        annual_other_cost: Any = 0,
    ) -> ROIExpenses:

        values = {
            "initial_investment": initial_investment,
            "annual_operating_cost": annual_operating_cost,
            "annual_maintenance_cost": (
                annual_maintenance_cost
            ),
            "annual_subscription_cost": (
                annual_subscription_cost
            ),
            "annual_training_cost": annual_training_cost,
            "annual_other_cost": annual_other_cost,
        }

        normalized = {
            key: money(
                decimal_value(
                    value,
                    key,
                    ZERO,
                )
            )
            for key, value in values.items()
        }

        return ROIExpenses(**normalized)

    # ========================================================
    # SIMPLE ROI
    # ========================================================

    @staticmethod
    def calculate_roi(
        *,
        total_benefits: Any,
        total_costs: Any,
    ) -> Optional[Decimal]:

        benefits = decimal_value(
            total_benefits,
            "total_benefits",
            ZERO,
        )

        costs = decimal_value(
            total_costs,
            "total_costs",
            ZERO,
        )

        if costs == ZERO:
            return None

        return (
            (benefits - costs)
            / costs
            * HUNDRED
        ).quantize(
            PERCENT_QUANT,
            rounding=ROUND_HALF_UP,
        )

    # ========================================================
    # NPV
    # ========================================================

    @staticmethod
    def calculate_npv(
        *,
        initial_investment: Any,
        annual_cash_flows: Sequence[Any],
        discount_rate_percent: Any = 8,
    ) -> Decimal:

        initial = decimal_value(
            initial_investment,
            "initial_investment",
            ZERO,
        )

        discount_rate = percent(
            discount_rate_percent,
            "discount_rate_percent",
        ) / HUNDRED

        npv = -initial

        for year, cash_flow in enumerate(
            annual_cash_flows,
            start=1,
        ):
            amount = decimal_value(
                cash_flow,
                f"cash_flow_{year}",
            )

            npv += (
                amount
                / (ONE + discount_rate) ** year
            )

        return money(npv)

    # ========================================================
    # INTERNAL RATE OF RETURN
    # ========================================================

    @staticmethod
    def calculate_irr(
        *,
        initial_investment: Any,
        annual_cash_flows: Sequence[Any],
        tolerance: Decimal = Decimal("0.00000001"),
        max_iterations: int = 250,
    ) -> Optional[Decimal]:

        initial = decimal_value(
            initial_investment,
            "initial_investment",
            ZERO,
        )

        if initial <= ZERO:
            return None

        flows = [
            decimal_value(
                flow,
                "cash_flow",
            )
            for flow in annual_cash_flows
        ]

        if not flows:
            return None

        # Restrict to conventional investment cash flows:
        # initial outflow followed by non-negative inflows.
        # Non-conventional flows can have multiple IRRs.

        if any(flow < ZERO for flow in flows):
            return None

        if sum(flows, ZERO) <= initial:
            return None

        def npv_at(rate: Decimal) -> Decimal:
            value = -initial

            for year, cash_flow in enumerate(
                flows,
                start=1,
            ):
                value += (
                    cash_flow
                    / (ONE + rate) ** year
                )

            return value

        low = ZERO
        high = ONE

        # Expand the upper bound when necessary.
        for _ in range(32):
            if npv_at(high) <= ZERO:
                break

            high *= 2
        else:
            return None

        for _ in range(max_iterations):
            midpoint = (low + high) / 2
            current_npv = npv_at(midpoint)

            if abs(current_npv) <= tolerance:
                return (
                    midpoint * HUNDRED
                ).quantize(
                    PERCENT_QUANT,
                    rounding=ROUND_HALF_UP,
                )

            if current_npv > ZERO:
                low = midpoint
            else:
                high = midpoint

        return (
            ((low + high) / 2) * HUNDRED
        ).quantize(
            PERCENT_QUANT,
            rounding=ROUND_HALF_UP,
        )

    # ========================================================
    # PAYBACK PERIOD
    # ========================================================

    @staticmethod
    def calculate_payback_period(
        *,
        initial_investment: Any,
        annual_cash_flows: Sequence[Any],
        discount_rate_percent: Any = 0,
    ) -> Optional[Decimal]:

        initial = decimal_value(
            initial_investment,
            "initial_investment",
            ZERO,
        )

        if initial == ZERO:
            return ZERO

        discount_rate = percent(
            discount_rate_percent,
            "discount_rate_percent",
        ) / HUNDRED

        cumulative = -initial

        for year, flow in enumerate(
            annual_cash_flows,
            start=1,
        ):
            amount = decimal_value(
                flow,
                f"cash_flow_{year}",
            )

            discounted_amount = (
                amount
                / (ONE + discount_rate) ** year
            )

            previous_cumulative = cumulative
            cumulative += discounted_amount

            if cumulative >= ZERO:
                if discounted_amount <= ZERO:
                    return Decimal(year)

                remaining = -previous_cumulative

                fraction = (
                    remaining / discounted_amount
                )

                return (
                    Decimal(year - 1) + fraction
                ).quantize(
                    Decimal("0.01"),
                    rounding=ROUND_HALF_UP,
                )

        return None

    # ========================================================
    # BENEFIT-COST RATIO
    # ========================================================

    @staticmethod
    def calculate_benefit_cost_ratio(
        *,
        present_value_benefits: Any,
        present_value_costs: Any,
    ) -> Optional[Decimal]:

        benefits = decimal_value(
            present_value_benefits,
            "present_value_benefits",
            ZERO,
        )

        costs = decimal_value(
            present_value_costs,
            "present_value_costs",
            ZERO,
        )

        if costs == ZERO:
            return None

        return (
            benefits / costs
        ).quantize(
            Decimal("0.0001"),
            rounding=ROUND_HALF_UP,
        )

    # ========================================================
    # INVESTMENT RECOMMENDATION
    # ========================================================

    @staticmethod
    def evaluate_investment(
        *,
        npv: Decimal,
        roi_percent: Optional[Decimal],
        benefit_cost_ratio: Optional[Decimal],
    ) -> tuple[str, str]:

        if (
            roi_percent is None
            or benefit_cost_ratio is None
        ):
            return (
                InvestmentDecision.REVIEW_REQUIRED.value,
                "Insufficient cost information for "
                "a complete financial recommendation.",
            )

        if (
            npv > ZERO
            and roi_percent > ZERO
            and benefit_cost_ratio > ONE
        ):
            return (
                InvestmentDecision.FINANCIALLY_ATTRACTIVE.value,
                "Projected discounted benefits exceed "
                "projected costs under the supplied "
                "financial assumptions.",
            )

        if npv < ZERO and roi_percent < ZERO:
            return (
                InvestmentDecision.NOT_FINANCIALLY_ATTRACTIVE.value,
                "Projected investment costs exceed "
                "projected benefits under the supplied "
                "financial assumptions.",
            )

        return (
            InvestmentDecision.MARGINAL.value,
            "Financial indicators are mixed or near "
            "break-even. Review assumptions, risks "
            "and alternative investments.",
        )

    # ========================================================
    # COMPLETE ROI ANALYSIS
    # ========================================================

    def analyze_roi(
        self,
        *,
        company_id: str,
        site_id: str,
        investment_name: str,
        investment_type: str | InvestmentType,
        initial_investment: Any,

        avoided_failure_cost: Any = 0,
        downtime_savings: Any = 0,
        production_loss_savings: Any = 0,
        maintenance_savings: Any = 0,
        energy_savings: Any = 0,
        labor_savings: Any = 0,
        safety_cost_avoidance: Any = 0,
        quality_savings: Any = 0,
        other_savings: Any = 0,

        annual_operating_cost: Any = 0,
        annual_maintenance_cost: Any = 0,
        annual_subscription_cost: Any = 0,
        annual_training_cost: Any = 0,
        annual_other_cost: Any = 0,

        analysis_years: int = 5,
        discount_rate_percent: Any = 8,
        benefit_growth_percent: Any = 0,
        expense_growth_percent: Any = 0,

        scenario: str | ROIScenario = ROIScenario.BASE,
        currency: Optional[str] = None,
        asset_id: Optional[str] = None,
        created_by: str = "system",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ROIAnalysisResult:

        selected_type = parse_enum(
            InvestmentType,
            investment_type,
            "investment_type",
        )

        selected_scenario = parse_enum(
            ROIScenario,
            scenario,
            "scenario",
        )

        selected_currency = currency_code(
            currency or self.default_currency
        )

        years = positive_years(
            analysis_years
        )

        discount_rate = percent(
            discount_rate_percent,
            "discount_rate_percent",
        )

        benefit_growth = percent(
            benefit_growth_percent,
            "benefit_growth_percent",
        )

        expense_growth = percent(
            expense_growth_percent,
            "expense_growth_percent",
        )

        benefits = self.calculate_benefits(
            avoided_failure_cost=avoided_failure_cost,
            downtime_savings=downtime_savings,
            production_loss_savings=(
                production_loss_savings
            ),
            maintenance_savings=maintenance_savings,
            energy_savings=energy_savings,
            labor_savings=labor_savings,
            safety_cost_avoidance=(
                safety_cost_avoidance
            ),
            quality_savings=quality_savings,
            other_savings=other_savings,
        )

        expenses = self.calculate_expenses(
            initial_investment=initial_investment,
            annual_operating_cost=annual_operating_cost,
            annual_maintenance_cost=(
                annual_maintenance_cost
            ),
            annual_subscription_cost=(
                annual_subscription_cost
            ),
            annual_training_cost=annual_training_cost,
            annual_other_cost=annual_other_cost,
        )

        benefit_multiplier = (
            self.SCENARIO_BENEFIT_MULTIPLIERS[
                selected_scenario
            ]
        )

        expense_multiplier = (
            self.SCENARIO_EXPENSE_MULTIPLIERS[
                selected_scenario
            ]
        )

        initial_cost = (
            expenses.initial_investment
            * expense_multiplier
        )

        annual_benefits_base = (
            benefits.total * benefit_multiplier
        )

        annual_expenses_base = (
            expenses.annual_recurring_cost
            * expense_multiplier
        )

        benefit_growth_factor = (
            ONE + benefit_growth / HUNDRED
        )

        expense_growth_factor = (
            ONE + expense_growth / HUNDRED
        )

        discount_factor = (
            ONE + discount_rate / HUNDRED
        )

        annual_cash_flows: list[
            AnnualROICashFlow
        ] = []

        total_benefits = ZERO
        total_recurring_costs = ZERO

        present_value_benefits = ZERO
        present_value_recurring_costs = ZERO

        cumulative_cash_flow = -initial_cost

        net_cash_flows: list[Decimal] = []

        for year in range(1, years + 1):

            year_benefits = (
                annual_benefits_base
                * benefit_growth_factor ** (year - 1)
            )

            year_expenses = (
                annual_expenses_base
                * expense_growth_factor ** (year - 1)
            )

            year_net = (
                year_benefits - year_expenses
            )

            discounted_net = (
                year_net
                / discount_factor ** year
            )

            cumulative_cash_flow += year_net

            total_benefits += year_benefits
            total_recurring_costs += year_expenses

            present_value_benefits += (
                year_benefits
                / discount_factor ** year
            )

            present_value_recurring_costs += (
                year_expenses
                / discount_factor ** year
            )

            net_cash_flows.append(year_net)

            annual_cash_flows.append(
                AnnualROICashFlow(
                    year=year,
                    benefits=money(year_benefits),
                    expenses=money(year_expenses),
                    net_cash_flow=money(year_net),
                    discounted_cash_flow=money(
                        discounted_net
                    ),
                    cumulative_cash_flow=money(
                        cumulative_cash_flow
                    ),
                )
            )

        total_costs = (
            initial_cost
            + total_recurring_costs
        )

        total_net_benefit = (
            total_benefits - total_costs
        )

        roi_percent = self.calculate_roi(
            total_benefits=total_benefits,
            total_costs=total_costs,
        )

        npv = self.calculate_npv(
            initial_investment=initial_cost,
            annual_cash_flows=net_cash_flows,
            discount_rate_percent=discount_rate,
        )

        irr = self.calculate_irr(
            initial_investment=initial_cost,
            annual_cash_flows=net_cash_flows,
        )

        payback = self.calculate_payback_period(
            initial_investment=initial_cost,
            annual_cash_flows=net_cash_flows,
        )

        discounted_payback = (
            self.calculate_payback_period(
                initial_investment=initial_cost,
                annual_cash_flows=net_cash_flows,
                discount_rate_percent=discount_rate,
            )
        )

        benefit_cost_ratio = (
            self.calculate_benefit_cost_ratio(
                present_value_benefits=(
                    present_value_benefits
                ),
                present_value_costs=(
                    initial_cost
                    + present_value_recurring_costs
                ),
            )
        )

        recommendation, explanation = (
            self.evaluate_investment(
                npv=npv,
                roi_percent=roi_percent,
                benefit_cost_ratio=benefit_cost_ratio,
            )
        )

        return ROIAnalysisResult(
            analysis_id=str(uuid4()),
            company_id=required_text(
                company_id,
                "company_id",
            ),
            site_id=required_text(
                site_id,
                "site_id",
            ),
            asset_id=optional_text(
                asset_id,
                "asset_id",
            ),
            investment_name=required_text(
                investment_name,
                "investment_name",
            ),
            investment_type=selected_type.value,
            scenario=selected_scenario.value,
            currency=selected_currency,
            analysis_years=years,
            discount_rate_percent=discount_rate,
            benefit_growth_percent=benefit_growth,
            expense_growth_percent=expense_growth,
            benefits=benefits,
            expenses=expenses,
            annual_cash_flows=annual_cash_flows,
            total_benefits=money(total_benefits),
            total_costs=money(total_costs),
            total_net_benefit=money(
                total_net_benefit
            ),
            roi_percent=roi_percent,
            npv=npv,
            irr_percent=irr,
            payback_years=payback,
            discounted_payback_years=(
                discounted_payback
            ),
            benefit_cost_ratio=benefit_cost_ratio,
            recommendation=recommendation,
            explanation=explanation,
            created_by=required_text(
                created_by,
                "created_by",
            ),
            metadata={
                **dict(metadata or {}),
                "benefit_multiplier": str(
                    benefit_multiplier
                ),
                "expense_multiplier": str(
                    expense_multiplier
                ),
                "savings_assumption": (
                    "incremental_non_overlapping_"
                    "annual_benefits"
                ),
                "financial_model": (
                    "nominal_multi_year_discounted_cash_flow"
                ),
            },
        )

    # ========================================================
    # SCENARIO ANALYSIS
    # ========================================================

    def compare_scenarios(
        self,
        **analysis_kwargs: Any,
    ) -> dict[str, Any]:

        results: dict[str, Any] = {}

        for scenario in ROIScenario:

            analysis = self.analyze_roi(
                **{
                    **analysis_kwargs,
                    "scenario": scenario,
                }
            )

            results[scenario.value] = {
                "analysis_id": analysis.analysis_id,
                "currency": analysis.currency,
                "roi_percent": (
                    str(analysis.roi_percent)
                    if analysis.roi_percent is not None
                    else None
                ),
                "npv": str(analysis.npv),
                "irr_percent": (
                    str(analysis.irr_percent)
                    if analysis.irr_percent is not None
                    else None
                ),
                "payback_years": (
                    str(analysis.payback_years)
                    if analysis.payback_years is not None
                    else None
                ),
                "recommendation": (
                    analysis.recommendation
                ),
            }

        return results

    # ========================================================
    # INVESTMENT COMPARISON
    # ========================================================

    @staticmethod
    def compare_investments(
        analyses: Sequence[ROIAnalysisResult],
    ) -> ROIComparisonResult:

        if not analyses:
            raise ROIValidationError(
                "At least one ROI analysis is required"
            )

        currencies = {
            analysis.currency
            for analysis in analyses
        }

        if len(currencies) != 1:
            raise ROIValidationError(
                "Cannot compare different currencies"
            )

        identifiers = [
            analysis.analysis_id
            for analysis in analyses
        ]

        if len(identifiers) != len(set(identifiers)):
            raise ROIValidationError(
                "Duplicate ROI analyses detected"
            )

        best_npv = max(
            analyses,
            key=lambda analysis: analysis.npv,
        )

        roi_candidates = [
            analysis
            for analysis in analyses
            if analysis.roi_percent is not None
        ]

        best_roi = (
            max(
                roi_candidates,
                key=lambda analysis: analysis.roi_percent,
            )
            if roi_candidates
            else None
        )

        comparisons = [
            {
                "analysis_id": analysis.analysis_id,
                "investment_name": (
                    analysis.investment_name
                ),
                "npv": str(analysis.npv),
                "roi_percent": (
                    str(analysis.roi_percent)
                    if analysis.roi_percent is not None
                    else None
                ),
                "payback_years": (
                    str(analysis.payback_years)
                    if analysis.payback_years is not None
                    else None
                ),
                "recommendation": (
                    analysis.recommendation
                ),
            }
            for analysis in analyses
        ]

        return ROIComparisonResult(
            currency=next(iter(currencies)),
            best_npv_analysis_id=(
                best_npv.analysis_id
            ),
            best_roi_analysis_id=(
                best_roi.analysis_id
                if best_roi is not None
                else None
            ),
            comparisons=comparisons,
        )

    # ========================================================
    # AUDIT
    # ========================================================

    async def _audit(
        self,
        *,
        analysis_id: str,
        actor_id: str,
        action: str,
        details: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> None:

        if self.audit is None:
            return

        method = getattr(
            self.audit,
            "record_event",
            None,
        )

        if method is None:
            return

        try:
            await invoke(
                method,
                {
                    "event_id": str(uuid4()),
                    "entity_type": "roi_analysis",
                    "entity_id": analysis_id,
                    "actor_id": actor_id,
                    "action": action,
                    "details": dict(details or {}),
                    "timestamp": utc_now().isoformat(),
                },
            )

        except Exception:
            logger.exception(
                "ROI audit failed: %s",
                analysis_id,
            )

    # ========================================================
    # SAVE ANALYSIS
    # ========================================================

    async def save_roi_analysis(
        self,
        analysis: ROIAnalysisResult,
    ) -> Any:

        if self.repository is None:
            raise ROIRepositoryError(
                "ROI repository is not configured"
            )

        if not isinstance(
            analysis,
            ROIAnalysisResult,
        ):
            raise ROIValidationError(
                "Expected ROIAnalysisResult"
            )

        saved = await invoke(
            self.repository.create_roi_analysis,
            analysis.to_dict(),
        )

        await self._audit(
            analysis_id=analysis.analysis_id,
            actor_id=analysis.created_by,
            action="roi_analysis_created",
            details={
                "company_id": analysis.company_id,
                "site_id": analysis.site_id,
                "investment_type": (
                    analysis.investment_type
                ),
                "npv": str(analysis.npv),
                "roi_percent": (
                    str(analysis.roi_percent)
                    if analysis.roi_percent is not None
                    else None
                ),
            },
        )

        return saved

    # ========================================================
    # GET ANALYSIS
    # ========================================================

    async def get_roi_analysis(
        self,
        analysis_id: str,
    ) -> Any:

        if self.repository is None:
            raise ROIRepositoryError(
                "ROI repository is not configured"
            )

        identifier = required_text(
            analysis_id,
            "analysis_id",
        )

        record = await invoke(
            self.repository.get_roi_analysis,
            identifier,
        )

        if record is None:
            raise ROIAnalysisNotFoundError(
                f"ROI analysis not found: {identifier}"
            )

        return record

    # ========================================================
    # LIST ANALYSES
    # ========================================================

    async def list_roi_analyses(
        self,
        *,
        company_id: Optional[str] = None,
        site_id: Optional[str] = None,
        asset_id: Optional[str] = None,
        investment_type: Optional[str] = None,
    ) -> Any:

        if self.repository is None:
            raise ROIRepositoryError(
                "ROI repository is not configured"
            )

        filters: dict[str, Any] = {}

        for name, value in {
            "company_id": company_id,
            "site_id": site_id,
            "asset_id": asset_id,
        }.items():
            if value is not None:
                filters[name] = required_text(
                    value,
                    name,
                )

        if investment_type is not None:
            filters["investment_type"] = parse_enum(
                InvestmentType,
                investment_type,
                "investment_type",
            ).value

        return await invoke(
            self.repository.list_roi_analyses,
            filters,
        )


# ============================================================
# IN-MEMORY REPOSITORY
# ============================================================

class InMemoryROIRepository:
    """
    Development-only ROI repository.

    Not durable or concurrency-safe.
    """

    def __init__(self):
        self.records: dict[
            str,
            dict[str, Any],
        ] = {}

    async def create_roi_analysis(
        self,
        data: dict[str, Any],
    ) -> dict[str, Any]:

        record = copy.deepcopy(data)

        analysis_id = record["analysis_id"]

        if analysis_id in self.records:
            raise ROIRepositoryError(
                "Duplicate ROI analysis ID"
            )

        self.records[analysis_id] = record

        return copy.deepcopy(record)

    async def get_roi_analysis(
        self,
        analysis_id: str,
    ) -> Optional[dict[str, Any]]:

        record = self.records.get(
            analysis_id
        )

        return (
            copy.deepcopy(record)
            if record is not None
            else None
        )

    async def list_roi_analyses(
        self,
        filters: dict[str, Any],
    ) -> list[dict[str, Any]]:

        return [
            copy.deepcopy(record)
            for record in self.records.values()
            if all(
                record.get(key) == value
                for key, value in filters.items()
            )
        ]


# ============================================================
# DEMONSTRATION
# ============================================================

async def main() -> None:

    logging.basicConfig(
        level=logging.INFO
    )

    repository = InMemoryROIRepository()

    service = ROIService(
        roi_repository=repository,
        default_currency="INR",
    )

    # --------------------------------------------------------
    # PREDICTIVE MAINTENANCE ROI
    # --------------------------------------------------------

    predictive_analysis = service.analyze_roi(
        company_id="COMPANY-001",
        site_id="SITE-001",
        asset_id="MACHINE-001",
        investment_name=(
            "AI Predictive Maintenance Deployment"
        ),
        investment_type="predictive_maintenance",
        initial_investment=500000,
        avoided_failure_cost=180000,
        downtime_savings=120000,
        production_loss_savings=80000,
        maintenance_savings=60000,
        energy_savings=25000,
        labor_savings=15000,
        annual_operating_cost=35000,
        annual_maintenance_cost=20000,
        annual_subscription_cost=25000,
        annual_training_cost=10000,
        analysis_years=5,
        discount_rate_percent=8,
        benefit_growth_percent=3,
        expense_growth_percent=4,
        scenario="base",
        currency="INR",
        created_by="ENGINEER-001",
    )

    print("\nPREDICTIVE MAINTENANCE ROI")

    print(
        json.dumps(
            predictive_analysis.to_dict(),
            indent=2,
        )
    )

    await service.save_roi_analysis(
        predictive_analysis
    )

    # --------------------------------------------------------
    # EQUIPMENT REPLACEMENT ROI
    # --------------------------------------------------------

    replacement_analysis = service.analyze_roi(
        company_id="COMPANY-001",
        site_id="SITE-001",
        asset_id="MACHINE-002",
        investment_name=(
            "High-Efficiency Motor Replacement"
        ),
        investment_type="equipment_replacement",
        initial_investment=300000,
        avoided_failure_cost=75000,
        downtime_savings=40000,
        maintenance_savings=45000,
        energy_savings=90000,
        annual_operating_cost=10000,
        annual_maintenance_cost=15000,
        analysis_years=5,
        discount_rate_percent=8,
        scenario="base",
        currency="INR",
    )

    await service.save_roi_analysis(
        replacement_analysis
    )

    # --------------------------------------------------------
    # INVESTMENT COMPARISON
    # --------------------------------------------------------

    comparison = service.compare_investments(
        [
            predictive_analysis,
            replacement_analysis,
        ]
    )

    print("\nINVESTMENT COMPARISON")

    print(
        json.dumps(
            comparison.to_dict(),
            indent=2,
        )
    )

    # --------------------------------------------------------
    # SCENARIO ANALYSIS
    # --------------------------------------------------------

    scenarios = service.compare_scenarios(
        company_id="COMPANY-001",
        site_id="SITE-001",
        investment_name="Industrial AI Deployment",
        investment_type="ai_deployment",
        initial_investment=400000,
        avoided_failure_cost=120000,
        downtime_savings=90000,
        maintenance_savings=50000,
        energy_savings=30000,
        annual_operating_cost=25000,
        annual_subscription_cost=20000,
        analysis_years=5,
        discount_rate_percent=8,
    )

    print("\nSCENARIO ANALYSIS")

    print(
        json.dumps(
            scenarios,
            indent=2,
        )
    )

    # --------------------------------------------------------
    # SAVED ANALYSES
    # --------------------------------------------------------

    records = await service.list_roi_analyses(
        company_id="COMPANY-001"
    )

    print("\nSAVED ROI ANALYSES")

    print(
        json.dumps(
            {
                "total_analyses": len(records),
                "investment_names": [
                    record["investment_name"]
                    for record in records
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
