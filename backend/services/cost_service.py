
"""
app/services/cost_service.py

Industrial Predictive Maintenance, Safety & Emergency AI
Cost Intelligence Service

Capabilities:
    - Repair and replacement cost estimates
    - Spare parts and labor cost calculation
    - Downtime and production loss calculation
    - Emergency and energy costs
    - Failure cost and expected failure loss
    - Repair versus replacement comparison
    - Preventive maintenance ROI
    - Future cost forecasting
    - Financial risk classification
    - Optional async repository persistence

All amounts must use the same currency.
Financial calculations are estimates, not guarantees.
"""

from __future__ import annotations

import inspect
import logging
import math

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Mapping, Optional, Sequence
from uuid import uuid4


logger = logging.getLogger(__name__)


class CostServiceError(Exception):
    """Base cost service exception."""


class CostValidationError(CostServiceError):
    """Invalid cost or financial input."""


class FinancialRisk(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Decision(str, Enum):
    REPAIR = "repair"
    REPLACE = "replace"
    REVIEW = "manual_review"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def money(value: Any) -> Decimal:
    """
    Convert a numeric value into Decimal currency.

    Reject NaN, infinity, boolean and negative amounts.
    """

    if isinstance(value, bool):
        raise CostValidationError(
            "Boolean is not a valid monetary value"
        )

    try:
        result = Decimal(str(value))
    except Exception as exc:
        raise CostValidationError(
            f"Invalid monetary amount: {value}"
        ) from exc

    if not result.is_finite():
        raise CostValidationError(
            "Monetary value must be finite"
        )

    if result < 0:
        raise CostValidationError(
            "Monetary value cannot be negative"
        )

    return result.quantize(
        Decimal("0.01"),
        rounding=ROUND_HALF_UP,
    )


def number(
    value: Any,
    *,
    minimum: float = 0.0,
    maximum: Optional[float] = None,
) -> float:

    if isinstance(value, bool):
        raise CostValidationError(
            "Boolean is not a valid number"
        )

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CostValidationError(
            f"Invalid numeric value: {value}"
        ) from exc

    if not math.isfinite(result):
        raise CostValidationError(
            "Number must be finite"
        )

    if result < minimum:
        raise CostValidationError(
            f"Value must be >= {minimum}"
        )

    if maximum is not None and result > maximum:
        raise CostValidationError(
            f"Value must be <= {maximum}"
        )

    return result


def decimal_string(value: Decimal) -> str:
    return format(money(value), ".2f")


@dataclass
class CostItem:
    name: str
    quantity: float
    unit_cost: Decimal

    def total(self) -> Decimal:
        return money(
            Decimal(str(self.quantity))
            * self.unit_cost
        )


@dataclass
class CostBreakdown:
    asset_id: str
    currency: str

    spare_parts: Decimal = Decimal("0")
    labor: Decimal = Decimal("0")
    downtime: Decimal = Decimal("0")
    production_loss: Decimal = Decimal("0")
    emergency: Decimal = Decimal("0")
    energy: Decimal = Decimal("0")
    scrap: Decimal = Decimal("0")
    other: Decimal = Decimal("0")

    timestamp: str = field(
        default_factory=utc_now
    )

    def total(self) -> Decimal:
        return money(
            self.spare_parts
            + self.labor
            + self.downtime
            + self.production_loss
            + self.emergency
            + self.energy
            + self.scrap
            + self.other
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)

        for key, value in data.items():
            if isinstance(value, Decimal):
                data[key] = decimal_string(value)

        data["total_cost"] = decimal_string(
            self.total()
        )

        return data


class CostService:
    """
    Industrial financial intelligence service.

    Repository contract (optional):

        async create_cost_record(data: dict)
        async get_cost_records(asset_id: str)

    Financial calculations can be used without
    configuring a database repository.
    """

    def __init__(
        self,
        cost_repository: Any = None,
        default_currency: str = "INR",
    ):

        self.repository = cost_repository

        if (
            not isinstance(default_currency, str)
            or len(default_currency) != 3
            or not default_currency.isalpha()
        ):
            raise CostValidationError(
                "Currency must be a 3-letter ISO code"
            )

        self.currency = default_currency.upper()

    def calculate_spare_parts_cost(
        self,
        parts: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:

        items = []
        total = Decimal("0")

        for part in parts:
            name = str(
                part.get("name", "")
            ).strip()

            if not name:
                raise CostValidationError(
                    "Spare part name is required"
                )

            quantity = number(
                part.get("quantity", 1)
            )

            unit_cost = money(
                part.get("unit_cost", 0)
            )

            item = CostItem(
                name=name,
                quantity=quantity,
                unit_cost=unit_cost,
            )

            line_total = item.total()
            total += line_total

            items.append({
                "name": name,
                "quantity": quantity,
                "unit_cost": decimal_string(
                    unit_cost
                ),
                "total": decimal_string(
                    line_total
                ),
            })

        return {
            "currency": self.currency,
            "items": items,
            "total": decimal_string(total),
        }

    def calculate_labor_cost(
        self,
        hours: float,
        hourly_rate: float,
        technician_count: int = 1,
    ) -> Decimal:

        hours = number(hours)
        rate = money(hourly_rate)

        if (
            isinstance(technician_count, bool)
            or not isinstance(technician_count, int)
            or technician_count < 1
        ):
            raise CostValidationError(
                "technician_count must be "
                "a positive integer"
            )

        return money(
            Decimal(str(hours))
            * rate
            * technician_count
        )

    def calculate_downtime_cost(
        self,
        downtime_hours: float,
        cost_per_hour: float,
    ) -> Decimal:

        hours = number(downtime_hours)
        rate = money(cost_per_hour)

        return money(
            Decimal(str(hours)) * rate
        )

    def calculate_production_loss(
        self,
        lost_units: float,
        contribution_margin_per_unit: float,
    ) -> Decimal:
        """
        Contribution margin is preferred over revenue
        for estimating lost production profit.
        """

        units = number(lost_units)
        margin = money(
            contribution_margin_per_unit
        )

        return money(
            Decimal(str(units)) * margin
        )

    def calculate_repair_cost(
        self,
        *,
        parts: Sequence[Mapping[str, Any]],
        labor_hours: float,
        hourly_rate: float,
        technician_count: int = 1,
        additional_cost: float = 0,
    ) -> dict[str, Any]:

        parts_result = (
            self.calculate_spare_parts_cost(parts)
        )

        parts_cost = money(
            parts_result["total"]
        )

        labor_cost = self.calculate_labor_cost(
            hours=labor_hours,
            hourly_rate=hourly_rate,
            technician_count=technician_count,
        )

        additional = money(additional_cost)

        total = money(
            parts_cost + labor_cost + additional
        )

        return {
            "currency": self.currency,
            "spare_parts_cost": decimal_string(
                parts_cost
            ),
            "labor_cost": decimal_string(
                labor_cost
            ),
            "additional_cost": decimal_string(
                additional
            ),
            "total_repair_cost": decimal_string(
                total
            ),
            "parts": parts_result["items"],
        }

    def calculate_replacement_cost(
        self,
        *,
        purchase_price: float,
        installation_cost: float = 0,
        transport_cost: float = 0,
        commissioning_cost: float = 0,
        disposal_cost: float = 0,
        salvage_value: float = 0,
    ) -> dict[str, Any]:

        purchase = money(purchase_price)
        installation = money(installation_cost)
        transport = money(transport_cost)
        commissioning = money(
            commissioning_cost
        )
        disposal = money(disposal_cost)
        salvage = money(salvage_value)

        gross = money(
            purchase
            + installation
            + transport
            + commissioning
            + disposal
        )

        if salvage > gross:
            raise CostValidationError(
                "Salvage value exceeds gross "
                "replacement cost"
            )

        net = money(gross - salvage)

        return {
            "currency": self.currency,
            "purchase_price": decimal_string(
                purchase
            ),
            "installation_cost": decimal_string(
                installation
            ),
            "transport_cost": decimal_string(
                transport
            ),
            "commissioning_cost": decimal_string(
                commissioning
            ),
            "disposal_cost": decimal_string(
                disposal
            ),
            "salvage_value": decimal_string(
                salvage
            ),
            "gross_replacement_cost": decimal_string(
                gross
            ),
            "net_replacement_cost": decimal_string(
                net
            ),
        }

    def calculate_total_failure_cost(
        self,
        *,
        asset_id: str,
        spare_parts: float = 0,
        labor: float = 0,
        downtime: float = 0,
        production_loss: float = 0,
        emergency: float = 0,
        energy: float = 0,
        scrap: float = 0,
        other: float = 0,
    ) -> dict[str, Any]:

        if not str(asset_id).strip():
            raise CostValidationError(
                "asset_id is required"
            )

        breakdown = CostBreakdown(
            asset_id=str(asset_id),
            currency=self.currency,
            spare_parts=money(spare_parts),
            labor=money(labor),
            downtime=money(downtime),
            production_loss=money(
                production_loss
            ),
            emergency=money(emergency),
            energy=money(energy),
            scrap=money(scrap),
            other=money(other),
        )

        return breakdown.to_dict()

    def calculate_expected_failure_cost(
        self,
        *,
        failure_probability: float,
        failure_cost: float,
        horizon_days: int = 30,
    ) -> dict[str, Any]:
        """
        Expected loss for one defined prediction horizon.

        Assumes failure_probability refers to the
        probability of the modeled loss event during
        horizon_days.
        """

        probability = number(
            failure_probability,
            maximum=1.0,
        )

        if (
            isinstance(horizon_days, bool)
            or not isinstance(horizon_days, int)
            or horizon_days < 1
        ):
            raise CostValidationError(
                "horizon_days must be positive"
            )

        cost = money(failure_cost)

        expected = money(
            Decimal(str(probability)) * cost
        )

        return {
            "currency": self.currency,
            "failure_probability": probability,
            "horizon_days": horizon_days,
            "failure_cost": decimal_string(cost),
            "expected_failure_cost": decimal_string(
                expected
            ),
        }

    def compare_repair_vs_replacement(
        self,
        *,
        repair_cost: float,
        replacement_cost: float,
        expected_repair_life_years: float,
        expected_replacement_life_years: float,
        annual_repair_operating_cost: float = 0,
        annual_replacement_operating_cost: float = 0,
        analysis_years: float = 5,
        discount_rate: float = 0.08,
    ) -> dict[str, Any]:
        """
        Simplified discounted cost comparison.

        Assumes both alternatives remain operational
        throughout the analysis period. Future major
        capital replacements and residual values are
        not modeled.
        """

        repair = money(repair_cost)
        replacement = money(replacement_cost)

        repair_life = number(
            expected_repair_life_years,
            minimum=0.000001,
        )

        replacement_life = number(
            expected_replacement_life_years,
            minimum=0.000001,
        )

        years = number(
            analysis_years,
            minimum=0.000001,
        )

        rate = number(discount_rate)

        repair_operating = money(
            annual_repair_operating_cost
        )

        replacement_operating = money(
            annual_replacement_operating_cost
        )

        if (
            repair_life < years
            or replacement_life < years
        ):
            return {
                "currency": self.currency,
                "decision": Decision.REVIEW.value,
                "reason": (
                    "Expected service life is shorter "
                    "than the analysis horizon. "
                    "A replacement-cycle cash-flow "
                    "model is required."
                ),
            }

        # Annual end-of-year cash flows.
        # Use whole-year horizon for this model.
        if not years.is_integer():
            raise CostValidationError(
                "analysis_years must be a whole number"
            )

        discount_factor = sum(
            1 / ((1 + rate) ** year)
            for year in range(1, int(years) + 1)
        )

        repair_npv = money(
            repair
            + repair_operating
            * Decimal(str(discount_factor))
        )

        replacement_npv = money(
            replacement
            + replacement_operating
            * Decimal(str(discount_factor))
        )

        if repair_npv < replacement_npv:
            decision = Decision.REPAIR

        elif replacement_npv < repair_npv:
            decision = Decision.REPLACE

        else:
            decision = Decision.REVIEW

        return {
            "currency": self.currency,
            "analysis_years": int(years),
            "discount_rate": rate,
            "repair_npv_cost": decimal_string(
                repair_npv
            ),
            "replacement_npv_cost": decimal_string(
                replacement_npv
            ),
            "estimated_cost_difference": decimal_string(
                abs(repair_npv - replacement_npv)
            ),
            "decision": decision.value,
            "requires_engineering_approval": True,
        }

    def calculate_maintenance_roi(
        self,
        *,
        maintenance_investment: float,
        expected_avoided_failure_cost: float,
        additional_savings: float = 0,
    ) -> dict[str, Any]:

        investment = money(
            maintenance_investment
        )

        avoided = money(
            expected_avoided_failure_cost
        )

        additional = money(
            additional_savings
        )

        benefits = money(avoided + additional)

        net_benefit = benefits - investment

        if investment == 0:
            roi_percent = None
        else:
            roi_percent = round(
                float(
                    net_benefit / investment * 100
                ),
                2,
            )

        return {
            "currency": self.currency,
            "maintenance_investment": decimal_string(
                investment
            ),
            "total_expected_benefits": decimal_string(
                benefits
            ),
            "net_benefit": decimal_string(
                net_benefit
            ),
            "roi_percent": roi_percent,
        }

    def forecast_future_cost(
        self,
        *,
        current_annual_cost: float,
        annual_growth_rate: float,
        years: int = 5,
    ) -> dict[str, Any]:

        current = money(current_annual_cost)

        growth = number(
            annual_growth_rate,
            minimum=-1.0,
        )

        if (
            isinstance(years, bool)
            or not isinstance(years, int)
            or not 1 <= years <= 50
        ):
            raise CostValidationError(
                "years must be between 1 and 50"
            )

        forecasts = []
        cumulative = Decimal("0")

        for year in range(1, years + 1):
            projected = money(
                current
                * Decimal(
                    str((1 + growth) ** year)
                )
            )

            cumulative += projected

            forecasts.append({
                "year": year,
                "projected_cost": decimal_string(
                    projected
                ),
            })

        return {
            "currency": self.currency,
            "annual_growth_rate": growth,
            "forecast": forecasts,
            "cumulative_projected_cost": decimal_string(
                cumulative
            ),
            "forecast_method": (
                "constant_annual_growth"
            ),
        }

    def classify_financial_risk(
        self,
        *,
        expected_loss: float,
        asset_value: float,
    ) -> dict[str, Any]:

        loss = money(expected_loss)
        value = money(asset_value)

        if value == 0:
            raise CostValidationError(
                "Asset value must be greater than zero"
            )

        ratio = float(loss / value)

        if ratio >= 0.50:
            risk = FinancialRisk.CRITICAL
        elif ratio >= 0.25:
            risk = FinancialRisk.HIGH
        elif ratio >= 0.10:
            risk = FinancialRisk.MEDIUM
        else:
            risk = FinancialRisk.LOW

        return {
            "financial_risk": risk.value,
            "loss_to_asset_value_ratio": round(
                ratio, 4
            ),
            "expected_loss": decimal_string(loss),
            "asset_value": decimal_string(value),
            "currency": self.currency,
        }

    async def save_cost_analysis(
        self,
        *,
        asset_id: str,
        analysis_type: str,
        result: Mapping[str, Any],
    ) -> Any:

        if self.repository is None:
            raise CostServiceError(
                "Cost repository is not configured"
            )

        if not str(asset_id).strip():
            raise CostValidationError(
                "asset_id is required"
            )

        if not str(analysis_type).strip():
            raise CostValidationError(
                "analysis_type is required"
            )

        record = {
            "id": str(uuid4()),
            "asset_id": str(asset_id),
            "analysis_type": analysis_type,
            "currency": self.currency,
            "result": dict(result),
            "created_at": utc_now(),
        }

        create_method = getattr(
            self.repository,
            "create_cost_record",
            None,
        )

        if create_method is None:
            raise CostServiceError(
                "Repository must implement "
                "create_cost_record"
            )

        saved = create_method(record)

        if inspect.isawaitable(saved):
            saved = await saved

        logger.info(
            "Saved cost analysis asset=%s type=%s",
            asset_id,
            analysis_type,
        )

        return saved

    async def get_asset_cost_history(
        self,
        asset_id: str,
    ) -> Any:

        if self.repository is None:
            raise CostServiceError(
                "Cost repository is not configured"
            )

        method = getattr(
            self.repository,
            "get_cost_records",
            None,
        )

        if method is None:
            raise CostServiceError(
                "Repository must implement "
                "get_cost_records"
            )

        result = method(str(asset_id))

        if inspect.isawaitable(result):
            result = await result

        return result


if __name__ == "__main__":

    import json

    service = CostService(
        default_currency="INR"
    )

    repair = service.calculate_repair_cost(
        parts=[
            {
                "name": "Bearing",
                "quantity": 2,
                "unit_cost": 3500,
            },
            {
                "name": "Motor Seal",
                "quantity": 1,
                "unit_cost": 1800,
            },
        ],
        labor_hours=6,
        hourly_rate=800,
        technician_count=2,
    )

    failure = service.calculate_total_failure_cost(
        asset_id="MACHINE-001",
        spare_parts=8800,
        labor=9600,
        downtime=30000,
        production_loss=45000,
        emergency=5000,
    )

    roi = service.calculate_maintenance_roi(
        maintenance_investment=15000,
        expected_avoided_failure_cost=90000,
    )

    print(
        json.dumps(
            {
                "repair": repair,
                "failure": failure,
                "roi": roi,
            },
            indent=2,
        )
    )
