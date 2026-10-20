
"""
app/services/downtime_cost_service.py

Industrial Predictive Maintenance, Safety & Emergency AI

Downtime Cost Intelligence Service

Features:
- Planned and unplanned downtime
- Downtime duration calculation
- Lost contribution margin
- Idle labor costs
- Equipment recovery costs
- Restart and emergency expenses
- Expected downtime loss
- Avoided downtime savings
- Downtime financial forecasting
- Asset-level cost summaries

Financial estimates only.
Safety decisions must remain with authorized systems.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from enum import Enum
from typing import Any, Optional, Sequence
from uuid import uuid4


logger = logging.getLogger(__name__)

CENT = Decimal("0.01")


class DowntimeCostError(Exception):
    """Base downtime service exception."""


class DowntimeValidationError(DowntimeCostError):
    """Invalid downtime calculation input."""


class DowntimeType(str, Enum):
    PLANNED = "planned"
    UNPLANNED = "unplanned"
    EMERGENCY = "emergency"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def to_decimal(
    value: Any,
    *,
    allow_negative: bool = False,
) -> Decimal:

    if isinstance(value, bool):
        raise DowntimeValidationError(
            "Boolean is not a valid numeric value"
        )

    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise DowntimeValidationError(
            f"Invalid numeric value: {value}"
        ) from exc

    if not result.is_finite():
        raise DowntimeValidationError(
            "Numeric value must be finite"
        )

    if not allow_negative and result < 0:
        raise DowntimeValidationError(
            "Negative values are not allowed"
        )

    return result


def currency_amount(value: Any) -> Decimal:
    return to_decimal(value).quantize(
        CENT,
        rounding=ROUND_HALF_UP,
    )


def format_money(value: Decimal) -> str:
    return format(
        value.quantize(
            CENT,
            rounding=ROUND_HALF_UP,
        ),
        ".2f",
    )


@dataclass
class DowntimeCostResult:
    analysis_id: str
    asset_id: str
    downtime_type: str
    duration_hours: Decimal
    currency: str

    production_loss: Decimal
    idle_labor_cost: Decimal
    recovery_cost: Decimal
    restart_cost: Decimal
    emergency_cost: Decimal
    other_cost: Decimal

    total_cost: Decimal
    created_at: datetime = field(
        default_factory=utc_now
    )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)

        for key, value in data.items():
            if isinstance(value, Decimal):
                if key == "duration_hours":
                    data[key] = str(value)
                else:
                    data[key] = format_money(value)

            elif isinstance(value, datetime):
                data[key] = value.isoformat()

        return data


class DowntimeCostService:
    """
    Financial downtime analysis service.

    Optional repository contract:

    async create_downtime_record(data: dict)
    async get_downtime_records(asset_id: str)

    Costs must be in a consistent currency.
    """

    def __init__(
        self,
        downtime_repository: Any = None,
        currency: str = "INR",
    ):

        if (
            not isinstance(currency, str)
            or len(currency) != 3
            or not currency.isalpha()
        ):
            raise DowntimeValidationError(
                "Currency must be a three-letter code"
            )

        self.repository = downtime_repository
        self.currency = currency.upper()

    @staticmethod
    def calculate_duration(
        start_time: datetime,
        end_time: datetime,
    ) -> Decimal:

        if not isinstance(start_time, datetime):
            raise DowntimeValidationError(
                "start_time must be datetime"
            )

        if not isinstance(end_time, datetime):
            raise DowntimeValidationError(
                "end_time must be datetime"
            )

        for name, value in (
            ("start_time", start_time),
            ("end_time", end_time),
        ):
            if (
                value.tzinfo is None
                or value.utcoffset() is None
            ):
                raise DowntimeValidationError(
                    f"{name} must be timezone-aware"
                )

        if end_time < start_time:
            raise DowntimeValidationError(
                "end_time cannot precede start_time"
            )

        seconds = Decimal(
            str(
                (end_time - start_time).total_seconds()
            )
        )

        return seconds / Decimal("3600")

    @staticmethod
    def calculate_production_loss(
        duration_hours: Any,
        units_per_hour: Any,
        contribution_margin_per_unit: Any,
        production_loss_fraction: Any = 1,
    ) -> Decimal:
        """
        Estimate lost contribution margin.

        production_loss_fraction:
            0.0 = no production lost
            1.0 = complete production loss
        """

        hours = to_decimal(duration_hours)
        rate = to_decimal(units_per_hour)
        margin = to_decimal(
            contribution_margin_per_unit
        )
        fraction = to_decimal(
            production_loss_fraction
        )

        if fraction > 1:
            raise DowntimeValidationError(
                "production_loss_fraction must be <= 1"
            )

        return currency_amount(
            hours * rate * margin * fraction
        )

    @staticmethod
    def calculate_idle_labor_cost(
        duration_hours: Any,
        employee_count: int,
        hourly_cost_per_employee: Any,
        idle_fraction: Any = 1,
    ) -> Decimal:

        hours = to_decimal(duration_hours)
        hourly_cost = to_decimal(
            hourly_cost_per_employee
        )
        fraction = to_decimal(idle_fraction)

        if (
            isinstance(employee_count, bool)
            or not isinstance(employee_count, int)
            or employee_count < 0
        ):
            raise DowntimeValidationError(
                "employee_count must be "
                "a non-negative integer"
            )

        if fraction > 1:
            raise DowntimeValidationError(
                "idle_fraction must be <= 1"
            )

        return currency_amount(
            hours
            * employee_count
            * hourly_cost
            * fraction
        )

    def analyze_downtime(
        self,
        *,
        asset_id: str,
        downtime_type: str | DowntimeType,
        duration_hours: Any,
        units_per_hour: Any = 0,
        contribution_margin_per_unit: Any = 0,
        production_loss_fraction: Any = 1,
        employee_count: int = 0,
        hourly_cost_per_employee: Any = 0,
        idle_fraction: Any = 1,
        recovery_cost: Any = 0,
        restart_cost: Any = 0,
        emergency_cost: Any = 0,
        other_cost: Any = 0,
    ) -> DowntimeCostResult:

        if not str(asset_id).strip():
            raise DowntimeValidationError(
                "asset_id is required"
            )

        try:
            kind = DowntimeType(downtime_type)
        except ValueError as exc:
            raise DowntimeValidationError(
                "Invalid downtime_type"
            ) from exc

        hours = to_decimal(duration_hours)

        production = self.calculate_production_loss(
            duration_hours=hours,
            units_per_hour=units_per_hour,
            contribution_margin_per_unit=(
                contribution_margin_per_unit
            ),
            production_loss_fraction=(
                production_loss_fraction
            ),
        )

        labor = self.calculate_idle_labor_cost(
            duration_hours=hours,
            employee_count=employee_count,
            hourly_cost_per_employee=(
                hourly_cost_per_employee
            ),
            idle_fraction=idle_fraction,
        )

        recovery = currency_amount(recovery_cost)
        restart = currency_amount(restart_cost)
        emergency = currency_amount(emergency_cost)
        other = currency_amount(other_cost)

        total = currency_amount(
            production
            + labor
            + recovery
            + restart
            + emergency
            + other
        )

        result = DowntimeCostResult(
            analysis_id=str(uuid4()),
            asset_id=str(asset_id),
            downtime_type=kind.value,
            duration_hours=hours,
            currency=self.currency,
            production_loss=production,
            idle_labor_cost=labor,
            recovery_cost=recovery,
            restart_cost=restart,
            emergency_cost=emergency,
            other_cost=other,
            total_cost=total,
        )

        logger.info(
            "Downtime analyzed asset=%s "
            "type=%s hours=%s total=%s",
            asset_id,
            kind.value,
            hours,
            total,
        )

        return result

    def analyze_downtime_period(
        self,
        *,
        asset_id: str,
        downtime_type: str | DowntimeType,
        start_time: datetime,
        end_time: datetime,
        **kwargs: Any,
    ) -> DowntimeCostResult:

        duration = self.calculate_duration(
            start_time,
            end_time,
        )

        return self.analyze_downtime(
            asset_id=asset_id,
            downtime_type=downtime_type,
            duration_hours=duration,
            **kwargs,
        )

    def calculate_expected_downtime_loss(
        self,
        *,
        failure_probability: Any,
        expected_downtime_hours: Any,
        hourly_loss: Any,
    ) -> dict[str, Any]:
        """
        Expected loss for one defined failure horizon.
        """

        probability = to_decimal(
            failure_probability
        )

        if probability > 1:
            raise DowntimeValidationError(
                "failure_probability must be <= 1"
            )

        hours = to_decimal(
            expected_downtime_hours
        )

        hourly = to_decimal(hourly_loss)

        loss_if_failure = currency_amount(
            hours * hourly
        )

        expected_loss = currency_amount(
            probability * loss_if_failure
        )

        return {
            "currency": self.currency,
            "failure_probability": str(probability),
            "expected_downtime_hours": str(hours),
            "loss_if_failure": format_money(
                loss_if_failure
            ),
            "expected_loss": format_money(
                expected_loss
            ),
        }

    def calculate_avoided_downtime_savings(
        self,
        *,
        baseline_downtime_hours: Any,
        actual_downtime_hours: Any,
        hourly_loss: Any,
        intervention_cost: Any = 0,
    ) -> dict[str, Any]:

        baseline = to_decimal(
            baseline_downtime_hours
        )

        actual = to_decimal(
            actual_downtime_hours
        )

        hourly = to_decimal(hourly_loss)
        intervention = currency_amount(
            intervention_cost
        )

        avoided_hours = baseline - actual

        gross_savings = currency_amount(
            avoided_hours * hourly
        )

        net_savings = (
            gross_savings - intervention
        ).quantize(
            CENT,
            rounding=ROUND_HALF_UP,
        )

        return {
            "currency": self.currency,
            "baseline_hours": str(baseline),
            "actual_hours": str(actual),
            "avoided_hours": str(avoided_hours),
            "gross_savings": format_money(
                gross_savings
            ),
            "intervention_cost": format_money(
                intervention
            ),
            "net_savings": format_money(
                net_savings
            ),
        }

    def forecast_downtime_cost(
        self,
        *,
        current_annual_downtime_hours: Any,
        hourly_loss: Any,
        annual_change_rate: Any = 0,
        years: int = 5,
    ) -> dict[str, Any]:

        hours = to_decimal(
            current_annual_downtime_hours
        )

        hourly = to_decimal(hourly_loss)

        rate = to_decimal(
            annual_change_rate,
            allow_negative=True,
        )

        if rate < -1:
            raise DowntimeValidationError(
                "annual_change_rate cannot be < -1"
            )

        if (
            isinstance(years, bool)
            or not isinstance(years, int)
            or not 1 <= years <= 50
        ):
            raise DowntimeValidationError(
                "years must be between 1 and 50"
            )

        forecast = []
        cumulative = Decimal("0")

        for year in range(1, years + 1):

            projected_hours = (
                hours * ((Decimal("1") + rate) ** year)
            )

            projected_cost = currency_amount(
                projected_hours * hourly
            )

            cumulative += projected_cost

            forecast.append({
                "year": year,
                "downtime_hours": str(
                    projected_hours.quantize(
                        Decimal("0.001")
                    )
                ),
                "cost": format_money(projected_cost),
            })

        return {
            "currency": self.currency,
            "forecast": forecast,
            "cumulative_cost": format_money(
                cumulative
            ),
            "forecast_method": (
                "constant_annual_downtime_change"
            ),
        }

    def summarize_asset_downtime(
        self,
        records: Sequence[DowntimeCostResult],
    ) -> dict[str, Any]:

        total_hours = Decimal("0")
        total_cost = Decimal("0")

        by_type = {
            kind.value: {
                "events": 0,
                "hours": Decimal("0"),
                "cost": Decimal("0"),
            }
            for kind in DowntimeType
        }

        for record in records:
            if record.currency != self.currency:
                raise DowntimeValidationError(
                    "Mixed currencies are not supported"
                )

            total_hours += record.duration_hours
            total_cost += record.total_cost

            entry = by_type[record.downtime_type]
            entry["events"] += 1
            entry["hours"] += record.duration_hours
            entry["cost"] += record.total_cost

        formatted = {}

        for kind, entry in by_type.items():
            formatted[kind] = {
                "events": entry["events"],
                "hours": str(entry["hours"]),
                "cost": format_money(entry["cost"]),
            }

        return {
            "currency": self.currency,
            "total_events": len(records),
            "total_hours": str(total_hours),
            "total_cost": format_money(total_cost),
            "by_type": formatted,
        }

    async def save_downtime_analysis(
        self,
        result: DowntimeCostResult,
    ) -> Any:

        if self.repository is None:
            raise DowntimeCostError(
                "Downtime repository not configured"
            )

        return await (
            self.repository.create_downtime_record(
                result.to_dict()
            )
        )

    async def get_asset_downtime_history(
        self,
        asset_id: str,
    ) -> Any:

        if self.repository is None:
            raise DowntimeCostError(
                "Downtime repository not configured"
            )

        return await (
            self.repository.get_downtime_records(
                asset_id
            )
        )


if __name__ == "__main__":

    import json

    service = DowntimeCostService(
        currency="INR"
    )

    result = service.analyze_downtime(
        asset_id="MACHINE-001",
        downtime_type="unplanned",
        duration_hours=6,
        units_per_hour=100,
        contribution_margin_per_unit=250,
        production_loss_fraction=0.8,
        employee_count=5,
        hourly_cost_per_employee=300,
        recovery_cost=5000,
        restart_cost=2000,
    )

    print(
        json.dumps(
            result.to_dict(),
            indent=2,
        )
    )
