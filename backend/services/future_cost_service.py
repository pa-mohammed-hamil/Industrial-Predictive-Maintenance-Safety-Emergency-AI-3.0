
"""
app/services/future_cost_service.py

Industrial Predictive Maintenance, Safety & Emergency AI

Future Cost Forecasting Service

Features:
- Multi-year equipment cost forecasting
- Preventive maintenance projections
- Repair cost forecasts
- Energy and operating cost forecasts
- Expected failure cost estimates
- Inflation-adjusted costs
- Discounted present value
- Best/base/worst financial scenarios
- Asset-level forecast summaries
- Optional async repository persistence

All financial values are estimates.
Failure probabilities must be calibrated to
their prediction horizons before financial use.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from enum import Enum
from typing import Any, Mapping, Optional, Sequence
from uuid import uuid4


logger = logging.getLogger(__name__)

CENT = Decimal("0.01")


class FutureCostError(Exception):
    pass


class FutureCostValidationError(FutureCostError):
    pass


class ForecastScenario(str, Enum):
    OPTIMISTIC = "optimistic"
    BASE = "base"
    PESSIMISTIC = "pessimistic"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def decimal_value(
    value: Any,
    *,
    allow_negative: bool = False,
) -> Decimal:

    if isinstance(value, bool):
        raise FutureCostValidationError(
            "Boolean is not a valid number"
        )

    try:
        result = Decimal(str(value))
    except (
        InvalidOperation,
        TypeError,
        ValueError,
    ) as exc:
        raise FutureCostValidationError(
            f"Invalid numeric value: {value}"
        ) from exc

    if not result.is_finite():
        raise FutureCostValidationError(
            "Numeric values must be finite"
        )

    if not allow_negative and result < 0:
        raise FutureCostValidationError(
            "Negative values are not permitted"
        )

    return result


def money(value: Any) -> Decimal:
    return decimal_value(value).quantize(
        CENT,
        rounding=ROUND_HALF_UP,
    )


def money_str(value: Decimal) -> str:
    return format(
        value.quantize(
            CENT,
            rounding=ROUND_HALF_UP,
        ),
        ".2f",
    )


def validate_rate(
    value: Any,
    *,
    allow_negative: bool = False,
) -> Decimal:

    rate = decimal_value(
        value,
        allow_negative=allow_negative,
    )

    if rate < Decimal("-1"):
        raise FutureCostValidationError(
            "Rate cannot be less than -100%"
        )

    return rate


@dataclass
class AnnualCostForecast:
    year: int
    maintenance_cost: Decimal
    repair_cost: Decimal
    energy_cost: Decimal
    operating_cost: Decimal
    expected_failure_cost: Decimal
    total_cost: Decimal
    discounted_cost: Decimal

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)

        for key, value in data.items():
            if isinstance(value, Decimal):
                data[key] = money_str(value)

        return data


@dataclass
class FutureCostForecastResult:
    forecast_id: str
    asset_id: str
    currency: str
    scenario: str
    forecast_years: int
    inflation_rate: str
    discount_rate: str
    annual_forecasts: list[AnnualCostForecast]
    cumulative_nominal_cost: Decimal
    net_present_cost: Decimal
    created_at: str = field(
        default_factory=utc_now
    )
    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "forecast_id": self.forecast_id,
            "asset_id": self.asset_id,
            "currency": self.currency,
            "scenario": self.scenario,
            "forecast_years": self.forecast_years,
            "inflation_rate": self.inflation_rate,
            "discount_rate": self.discount_rate,
            "annual_forecasts": [
                item.to_dict()
                for item in self.annual_forecasts
            ],
            "cumulative_nominal_cost": money_str(
                self.cumulative_nominal_cost
            ),
            "net_present_cost": money_str(
                self.net_present_cost
            ),
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


class FutureCostService:
    """
    Asset lifecycle financial forecasting.

    Expected optional repository contract:

    async create_forecast(data: dict)
    async get_forecasts(asset_id: str)

    Costs are assumed to be current annual baseline
    amounts in one consistent currency.

    Scenario assumptions are configurable.
    """

    SCENARIO_MULTIPLIERS = {
        ForecastScenario.OPTIMISTIC: Decimal("0.85"),
        ForecastScenario.BASE: Decimal("1.00"),
        ForecastScenario.PESSIMISTIC: Decimal("1.25"),
    }

    def __init__(
        self,
        forecast_repository: Any = None,
        currency: str = "INR",
    ):

        if (
            not isinstance(currency, str)
            or len(currency) != 3
            or not currency.isalpha()
        ):
            raise FutureCostValidationError(
                "Currency must be a 3-letter code"
            )

        self.repository = forecast_repository
        self.currency = currency.upper()

    def forecast_asset_costs(
        self,
        *,
        asset_id: str,
        annual_maintenance_cost: Any,
        annual_repair_cost: Any,
        annual_energy_cost: Any,
        annual_operating_cost: Any,
        failure_cost: Any = 0,
        annual_failure_probability: Any = 0,
        years: int = 5,
        inflation_rate: Any = 0.04,
        discount_rate: Any = 0.08,
        maintenance_growth_rate: Any = 0,
        repair_growth_rate: Any = 0,
        energy_growth_rate: Any = 0,
        operating_growth_rate: Any = 0,
        scenario: str | ForecastScenario = (
            ForecastScenario.BASE
        ),
        metadata: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> FutureCostForecastResult:

        if not isinstance(asset_id, str) or not (
            asset_id.strip()
        ):
            raise FutureCostValidationError(
                "asset_id is required"
            )

        if (
            isinstance(years, bool)
            or not isinstance(years, int)
            or not 1 <= years <= 50
        ):
            raise FutureCostValidationError(
                "years must be between 1 and 50"
            )

        try:
            selected_scenario = (
                ForecastScenario(scenario)
            )
        except ValueError as exc:
            raise FutureCostValidationError(
                "Invalid forecast scenario"
            ) from exc

        maintenance = money(
            annual_maintenance_cost
        )
        repair = money(annual_repair_cost)
        energy = money(annual_energy_cost)
        operating = money(
            annual_operating_cost
        )
        failure = money(failure_cost)

        probability = decimal_value(
            annual_failure_probability
        )

        if probability > 1:
            raise FutureCostValidationError(
                "Annual failure probability "
                "must be between 0 and 1"
            )

        inflation = validate_rate(
            inflation_rate,
            allow_negative=True,
        )

        discount = validate_rate(
            discount_rate
        )

        growth_rates = {
            "maintenance": validate_rate(
                maintenance_growth_rate,
                allow_negative=True,
            ),
            "repair": validate_rate(
                repair_growth_rate,
                allow_negative=True,
            ),
            "energy": validate_rate(
                energy_growth_rate,
                allow_negative=True,
            ),
            "operating": validate_rate(
                operating_growth_rate,
                allow_negative=True,
            ),
        }

        multiplier = self.SCENARIO_MULTIPLIERS[
            selected_scenario
        ]

        annual_forecasts = []
        cumulative = Decimal("0")
        net_present = Decimal("0")

        for year in range(1, years + 1):

            inflation_factor = (
                Decimal("1") + inflation
            ) ** year

            discount_factor = (
                Decimal("1") + discount
            ) ** year

            maintenance_year = money(
                maintenance
                * (
                    Decimal("1")
                    + growth_rates["maintenance"]
                ) ** year
                * inflation_factor
                * multiplier
            )

            repair_year = money(
                repair
                * (
                    Decimal("1")
                    + growth_rates["repair"]
                ) ** year
                * inflation_factor
                * multiplier
            )

            energy_year = money(
                energy
                * (
                    Decimal("1")
                    + growth_rates["energy"]
                ) ** year
                * inflation_factor
                * multiplier
            )

            operating_year = money(
                operating
                * (
                    Decimal("1")
                    + growth_rates["operating"]
                ) ** year
                * inflation_factor
                * multiplier
            )

            expected_failure_year = money(
                failure
                * probability
                * inflation_factor
                * multiplier
            )

            total_year = money(
                maintenance_year
                + repair_year
                + energy_year
                + operating_year
                + expected_failure_year
            )

            discounted_year = money(
                total_year / discount_factor
            )

            cumulative += total_year
            net_present += discounted_year

            annual_forecasts.append(
                AnnualCostForecast(
                    year=year,
                    maintenance_cost=maintenance_year,
                    repair_cost=repair_year,
                    energy_cost=energy_year,
                    operating_cost=operating_year,
                    expected_failure_cost=(
                        expected_failure_year
                    ),
                    total_cost=total_year,
                    discounted_cost=discounted_year,
                )
            )

        logger.info(
            "Future cost forecast asset=%s "
            "scenario=%s years=%s",
            asset_id,
            selected_scenario.value,
            years,
        )

        return FutureCostForecastResult(
            forecast_id=str(uuid4()),
            asset_id=asset_id,
            currency=self.currency,
            scenario=selected_scenario.value,
            forecast_years=years,
            inflation_rate=str(inflation),
            discount_rate=str(discount),
            annual_forecasts=annual_forecasts,
            cumulative_nominal_cost=money(
                cumulative
            ),
            net_present_cost=money(net_present),
            metadata=dict(metadata or {}),
        )

    def compare_scenarios(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:

        kwargs.pop("scenario", None)

        forecasts = {}

        for scenario in ForecastScenario:
            result = self.forecast_asset_costs(
                **kwargs,
                scenario=scenario,
            )

            forecasts[scenario.value] = (
                result.to_dict()
            )

        optimistic = decimal_value(
            forecasts["optimistic"][
                "net_present_cost"
            ]
        )

        pessimistic = decimal_value(
            forecasts["pessimistic"][
                "net_present_cost"
            ]
        )

        return {
            "asset_id": kwargs.get("asset_id"),
            "currency": self.currency,
            "scenarios": forecasts,
            "npv_cost_range": {
                "minimum": money_str(optimistic),
                "maximum": money_str(pessimistic),
            },
        }

    def calculate_lifecycle_cost(
        self,
        *,
        acquisition_cost: Any,
        installation_cost: Any,
        forecast: FutureCostForecastResult,
        end_of_life_cost: Any = 0,
        residual_value: Any = 0,
    ) -> dict[str, Any]:
        """
        Present-value lifecycle cost.

        End-of-life costs and residual value are
        assumed to occur at the end of the forecast.
        """

        acquisition = money(acquisition_cost)
        installation = money(installation_cost)
        disposal = money(end_of_life_cost)
        residual = money(residual_value)

        if forecast.currency != self.currency:
            raise FutureCostValidationError(
                "Forecast currency mismatch"
            )

        discount = decimal_value(
            forecast.discount_rate
        )

        terminal_factor = (
            Decimal("1") + discount
        ) ** forecast.forecast_years

        terminal_net_cost = (
            disposal - residual
        )

        discounted_terminal = (
            terminal_net_cost / terminal_factor
        )

        total_npv = (
            acquisition
            + installation
            + forecast.net_present_cost
            + discounted_terminal
        )

        return {
            "asset_id": forecast.asset_id,
            "currency": self.currency,
            "acquisition_cost": money_str(
                acquisition
            ),
            "installation_cost": money_str(
                installation
            ),
            "operating_cost_npv": money_str(
                forecast.net_present_cost
            ),
            "terminal_cost_npv": money_str(
                discounted_terminal
            ),
            "total_lifecycle_cost_npv": money_str(
                total_npv
            ),
        }

    async def save_forecast(
        self,
        forecast: FutureCostForecastResult,
    ) -> Any:

        if self.repository is None:
            raise FutureCostError(
                "Forecast repository not configured"
            )

        method = getattr(
            self.repository,
            "create_forecast",
            None,
        )

        if method is None:
            raise FutureCostError(
                "Repository must implement "
                "create_forecast"
            )

        result = method(
            forecast.to_dict()
        )

        if inspect.isawaitable(result):
            result = await result

        return result

    async def get_asset_forecasts(
        self,
        asset_id: str,
    ) -> Any:

        if self.repository is None:
            raise FutureCostError(
                "Forecast repository not configured"
            )

        method = getattr(
            self.repository,
            "get_forecasts",
            None,
        )

        if method is None:
            raise FutureCostError(
                "Repository must implement "
                "get_forecasts"
            )

        result = method(asset_id)

        if inspect.isawaitable(result):
            result = await result

        return result


if __name__ == "__main__":

    import json

    service = FutureCostService(
        currency="INR"
    )

    forecast = service.forecast_asset_costs(
        asset_id="MACHINE-001",
        annual_maintenance_cost=120000,
        annual_repair_cost=80000,
        annual_energy_cost=150000,
        annual_operating_cost=100000,
        failure_cost=500000,
        annual_failure_probability=0.15,
        years=5,
        inflation_rate=0.04,
        discount_rate=0.08,
        scenario="base",
    )

    print(
        json.dumps(
            forecast.to_dict(),
            indent=2,
        )
    )
