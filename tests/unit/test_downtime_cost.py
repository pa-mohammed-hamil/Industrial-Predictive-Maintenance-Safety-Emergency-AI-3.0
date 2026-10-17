
"""
tests/unit/test_downtime_cost.py

Industrial Predictive Maintenance, Safety &
Emergency Intelligence AI

Downtime Cost Unit Tests

Run:
    pytest tests/unit/test_downtime_cost.py -v

Coverage:
    - Basic downtime cost
    - Production loss
    - Idle labor cost
    - Fixed recovery cost
    - Planned downtime
    - Unplanned downtime
    - Emergency downtime
    - Multi-asset downtime
    - Downtime forecasting
    - Cost per hour
    - Partial-hour downtime
    - Zero downtime
    - Decimal precision
    - Invalid inputs
    - Deterministic results
    - Financial integrity

IMPORTANT:
This suite defines proposed financial formulas.

Connect DowntimeCostAdapter to your actual
production downtime cost engine.

All values are synthetic and denominated in INR.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol

import pytest


# ==================================================
# Financial helpers
# ==================================================

def D(value: Any) -> Decimal:
    return Decimal(str(value))


def assert_money(
    actual: Any,
    expected: Any,
) -> None:
    assert D(actual) == D(expected)


# ==================================================
# Input models
# ==================================================

@dataclass(frozen=True)
class DowntimeInput:
    downtime_hours: Decimal
    production_loss_per_hour: Decimal
    idle_labor_cost_per_hour: Decimal = D("0")
    fixed_recovery_cost: Decimal = D("0")


@dataclass(frozen=True)
class EmergencyDowntimeInput:
    downtime_hours: Decimal
    production_loss_per_hour: Decimal
    idle_labor_cost_per_hour: Decimal
    emergency_response_cost: Decimal
    restart_cost: Decimal = D("0")


@dataclass(frozen=True)
class DowntimeForecastInput:
    expected_downtime_hours: Decimal
    probability_of_failure: Decimal
    production_loss_per_hour: Decimal
    idle_labor_cost_per_hour: Decimal = D("0")


@dataclass(frozen=True)
class AssetDowntimeInput:
    asset_id: str
    downtime_hours: Decimal
    production_loss_per_hour: Decimal
    idle_labor_cost_per_hour: Decimal = D("0")


# ==================================================
# Service interface
# ==================================================

class DowntimeCostProtocol(Protocol):

    def calculate_downtime_cost(
        self,
        data: DowntimeInput,
    ) -> Decimal:
        ...

    def calculate_production_loss(
        self,
        downtime_hours: Decimal,
        production_loss_per_hour: Decimal,
    ) -> Decimal:
        ...

    def calculate_idle_labor_cost(
        self,
        downtime_hours: Decimal,
        labor_cost_per_hour: Decimal,
    ) -> Decimal:
        ...

    def calculate_emergency_downtime_cost(
        self,
        data: EmergencyDowntimeInput,
    ) -> Decimal:
        ...

    def calculate_expected_downtime_cost(
        self,
        data: DowntimeForecastInput,
    ) -> Decimal:
        ...

    def calculate_multi_asset_downtime(
        self,
        assets: list[AssetDowntimeInput],
    ) -> Decimal:
        ...

    def compare_planned_unplanned(
        self,
        planned_cost: Decimal,
        unplanned_cost: Decimal,
    ) -> dict[str, Decimal]:
        ...


# ==================================================
# Production adapter
# ==================================================

class DowntimeCostAdapter:
    """
    Connect to your actual downtime cost module.

    Example:

        from app.cost.downtime_cost import (
            DowntimeCostEngine
        )

        self.engine = DowntimeCostEngine()

    Update import paths and mappings to match
    your implementation.

    Do not implement a second calculator here
    when testing production business logic.
    """

    def __init__(self) -> None:
        self.engine: Any = None

        # TODO: Initialize production engine.

    def calculate_downtime_cost(
        self,
        data: DowntimeInput,
    ) -> Decimal:
        raise NotImplementedError

    def calculate_production_loss(
        self,
        downtime_hours: Decimal,
        production_loss_per_hour: Decimal,
    ) -> Decimal:
        raise NotImplementedError

    def calculate_idle_labor_cost(
        self,
        downtime_hours: Decimal,
        labor_cost_per_hour: Decimal,
    ) -> Decimal:
        raise NotImplementedError

    def calculate_emergency_downtime_cost(
        self,
        data: EmergencyDowntimeInput,
    ) -> Decimal:
        raise NotImplementedError

    def calculate_expected_downtime_cost(
        self,
        data: DowntimeForecastInput,
    ) -> Decimal:
        raise NotImplementedError

    def calculate_multi_asset_downtime(
        self,
        assets: list[AssetDowntimeInput],
    ) -> Decimal:
        raise NotImplementedError

    def compare_planned_unplanned(
        self,
        planned_cost: Decimal,
        unplanned_cost: Decimal,
    ) -> dict[str, Decimal]:
        raise NotImplementedError


# ==================================================
# Fixtures
# ==================================================

@pytest.fixture
def engine() -> DowntimeCostProtocol:
    adapter = DowntimeCostAdapter()

    if adapter.engine is None:
        pytest.skip(
            "Connect DowntimeCostAdapter to the "
            "production downtime cost engine."
        )

    return adapter


@pytest.fixture
def downtime_input() -> DowntimeInput:
    return DowntimeInput(
        downtime_hours=D("10"),
        production_loss_per_hour=D("8000"),
        idle_labor_cost_per_hour=D("1000"),
        fixed_recovery_cost=D("5000"),
    )


@pytest.fixture
def emergency_input() -> EmergencyDowntimeInput:
    return EmergencyDowntimeInput(
        downtime_hours=D("8"),
        production_loss_per_hour=D("10000"),
        idle_labor_cost_per_hour=D("1500"),
        emergency_response_cost=D("20000"),
        restart_cost=D("5000"),
    )


@pytest.fixture
def forecast_input() -> DowntimeForecastInput:
    return DowntimeForecastInput(
        expected_downtime_hours=D("12"),
        probability_of_failure=D("0.25"),
        production_loss_per_hour=D("10000"),
        idle_labor_cost_per_hour=D("2000"),
    )


# ==================================================
# Basic downtime cost
# ==================================================

class TestBasicDowntimeCost:

    def test_total_downtime_cost(
        self,
        engine: DowntimeCostProtocol,
        downtime_input: DowntimeInput,
    ) -> None:

        result = engine.calculate_downtime_cost(
            downtime_input
        )

        # 10 * (8000 + 1000) + 5000
        assert_money(result, D("95000"))

    def test_zero_downtime(
        self,
        engine: DowntimeCostProtocol,
    ) -> None:

        data = DowntimeInput(
            downtime_hours=D("0"),
            production_loss_per_hour=D("8000"),
        )

        result = engine.calculate_downtime_cost(data)

        assert_money(result, D("0"))

    def test_one_hour_downtime(
        self,
        engine: DowntimeCostProtocol,
    ) -> None:

        data = DowntimeInput(
            downtime_hours=D("1"),
            production_loss_per_hour=D("5000"),
            idle_labor_cost_per_hour=D("1000"),
        )

        result = engine.calculate_downtime_cost(data)

        assert_money(result, D("6000"))

    def test_fractional_downtime(
        self,
        engine: DowntimeCostProtocol,
    ) -> None:

        data = DowntimeInput(
            downtime_hours=D("1.5"),
            production_loss_per_hour=D("8000"),
        )

        result = engine.calculate_downtime_cost(data)

        assert_money(result, D("12000"))

    def test_fixed_recovery_cost(
        self,
        engine: DowntimeCostProtocol,
    ) -> None:

        data = DowntimeInput(
            downtime_hours=D("2"),
            production_loss_per_hour=D("5000"),
            fixed_recovery_cost=D("3000"),
        )

        result = engine.calculate_downtime_cost(data)

        assert_money(result, D("13000"))


# ==================================================
# Production loss
# ==================================================

class TestProductionLoss:

    def test_production_loss(
        self,
        engine: DowntimeCostProtocol,
    ) -> None:

        result = engine.calculate_production_loss(
            D("10"),
            D("8000"),
        )

        assert_money(result, D("80000"))

    def test_zero_production_loss(
        self,
        engine: DowntimeCostProtocol,
    ) -> None:

        result = engine.calculate_production_loss(
            D("0"),
            D("8000"),
        )

        assert_money(result, D("0"))

    def test_production_loss_scales(
        self,
        engine: DowntimeCostProtocol,
    ) -> None:

        first = engine.calculate_production_loss(
            D("2"),
            D("5000"),
        )

        second = engine.calculate_production_loss(
            D("4"),
            D("5000"),
        )

        assert_money(second, first * 2)

    def test_fractional_production_loss(
        self,
        engine: DowntimeCostProtocol,
    ) -> None:

        result = engine.calculate_production_loss(
            D("0.5"),
            D("7500"),
        )

        assert_money(result, D("3750"))


# ==================================================
# Idle labor
# ==================================================

class TestIdleLaborCost:

    def test_idle_labor_cost(
        self,
        engine: DowntimeCostProtocol,
    ) -> None:

        result = engine.calculate_idle_labor_cost(
            D("8"),
            D("1200"),
        )

        assert_money(result, D("9600"))

    def test_zero_idle_labor(
        self,
        engine: DowntimeCostProtocol,
    ) -> None:

        result = engine.calculate_idle_labor_cost(
            D("8"),
            D("0"),
        )

        assert_money(result, D("0"))

    def test_idle_labor_scales(
        self,
        engine: DowntimeCostProtocol,
    ) -> None:

        first = engine.calculate_idle_labor_cost(
            D("2"),
            D("1000"),
        )

        second = engine.calculate_idle_labor_cost(
            D("6"),
            D("1000"),
        )

        assert_money(second, first * 3)


# ==================================================
# Emergency downtime
# ==================================================

class TestEmergencyDowntime:

    def test_emergency_downtime_total(
        self,
        engine: DowntimeCostProtocol,
        emergency_input: EmergencyDowntimeInput,
    ) -> None:

        result = (
            engine.calculate_emergency_downtime_cost(
                emergency_input
            )
        )

        # 8 * (10000 + 1500) + 20000 + 5000
        assert_money(result, D("117000"))

    def test_emergency_cost_increases_total(
        self,
        engine: DowntimeCostProtocol,
    ) -> None:

        normal = DowntimeInput(
            downtime_hours=D("5"),
            production_loss_per_hour=D("10000"),
            idle_labor_cost_per_hour=D("1000"),
        )

        emergency = EmergencyDowntimeInput(
            downtime_hours=D("5"),
            production_loss_per_hour=D("10000"),
            idle_labor_cost_per_hour=D("1000"),
            emergency_response_cost=D("15000"),
        )

        normal_cost = engine.calculate_downtime_cost(
            normal
        )

        emergency_cost = (
            engine.calculate_emergency_downtime_cost(
                emergency
            )
        )

        assert emergency_cost > normal_cost


# ==================================================
# Forecasting
# ==================================================

class TestDowntimeForecast:

    def test_expected_downtime_cost(
        self,
        engine: DowntimeCostProtocol,
        forecast_input: DowntimeForecastInput,
    ) -> None:

        result = engine.calculate_expected_downtime_cost(
            forecast_input
        )

        # 0.25 * 12 * (10000 + 2000)
        assert_money(result, D("36000"))

    def test_zero_failure_probability(
        self,
        engine: DowntimeCostProtocol,
    ) -> None:

        data = DowntimeForecastInput(
            expected_downtime_hours=D("10"),
            probability_of_failure=D("0"),
            production_loss_per_hour=D("8000"),
        )

        result = engine.calculate_expected_downtime_cost(
            data
        )

        assert_money(result, D("0"))

    def test_certain_failure(
        self,
        engine: DowntimeCostProtocol,
    ) -> None:

        data = DowntimeForecastInput(
            expected_downtime_hours=D("10"),
            probability_of_failure=D("1"),
            production_loss_per_hour=D("8000"),
        )

        result = engine.calculate_expected_downtime_cost(
            data
        )

        assert_money(result, D("80000"))

    def test_expected_cost_increases_with_probability(
        self,
        engine: DowntimeCostProtocol,
    ) -> None:

        low = DowntimeForecastInput(
            expected_downtime_hours=D("10"),
            probability_of_failure=D("0.1"),
            production_loss_per_hour=D("8000"),
        )

        high = DowntimeForecastInput(
            expected_downtime_hours=D("10"),
            probability_of_failure=D("0.8"),
            production_loss_per_hour=D("8000"),
        )

        low_cost = engine.calculate_expected_downtime_cost(
            low
        )

        high_cost = engine.calculate_expected_downtime_cost(
            high
        )

        assert high_cost > low_cost


# ==================================================
# Multi-asset downtime
# ==================================================

class TestMultiAssetDowntime:

    def test_multi_asset_total(
        self,
        engine: DowntimeCostProtocol,
    ) -> None:

        assets = [
            AssetDowntimeInput(
                asset_id="PUMP-001",
                downtime_hours=D("5"),
                production_loss_per_hour=D("8000"),
            ),
            AssetDowntimeInput(
                asset_id="MOTOR-001",
                downtime_hours=D("3"),
                production_loss_per_hour=D("6000"),
            ),
        ]

        result = engine.calculate_multi_asset_downtime(
            assets
        )

        # (5 * 8000) + (3 * 6000)
        assert_money(result, D("58000"))

    def test_empty_asset_list(
        self,
        engine: DowntimeCostProtocol,
    ) -> None:

        result = engine.calculate_multi_asset_downtime(
            []
        )

        assert_money(result, D("0"))

    def test_multi_asset_with_idle_labor(
        self,
        engine: DowntimeCostProtocol,
    ) -> None:

        assets = [
            AssetDowntimeInput(
                asset_id="PUMP-001",
                downtime_hours=D("2"),
                production_loss_per_hour=D("5000"),
                idle_labor_cost_per_hour=D("1000"),
            ),
            AssetDowntimeInput(
                asset_id="MOTOR-001",
                downtime_hours=D("4"),
                production_loss_per_hour=D("3000"),
                idle_labor_cost_per_hour=D("500"),
            ),
        ]

        result = engine.calculate_multi_asset_downtime(
            assets
        )

        # 2*(5000+1000) + 4*(3000+500)
        assert_money(result, D("26000"))


# ==================================================
# Planned vs unplanned downtime
# ==================================================

class TestPlannedUnplannedDowntime:

    def test_unplanned_downtime_difference(
        self,
        engine: DowntimeCostProtocol,
    ) -> None:

        result = engine.compare_planned_unplanned(
            planned_cost=D("20000"),
            unplanned_cost=D("80000"),
        )

        assert_money(
            result["cost_difference"],
            D("60000"),
        )

    def test_unplanned_downtime_ratio(
        self,
        engine: DowntimeCostProtocol,
    ) -> None:

        result = engine.compare_planned_unplanned(
            planned_cost=D("20000"),
            unplanned_cost=D("80000"),
        )

        assert_money(
            result["cost_ratio"],
            D("4"),
        )

    def test_equal_planned_unplanned_cost(
        self,
        engine: DowntimeCostProtocol,
    ) -> None:

        result = engine.compare_planned_unplanned(
            planned_cost=D("50000"),
            unplanned_cost=D("50000"),
        )

        assert_money(
            result["cost_difference"],
            D("0"),
        )


# ==================================================
# Input validation
# ==================================================

class TestDowntimeValidation:

    def test_negative_downtime_rejected(
        self,
        engine: DowntimeCostProtocol,
    ) -> None:

        data = DowntimeInput(
            downtime_hours=D("-5"),
            production_loss_per_hour=D("8000"),
        )

        with pytest.raises(ValueError):
            engine.calculate_downtime_cost(data)

    def test_negative_production_rate_rejected(
        self,
        engine: DowntimeCostProtocol,
    ) -> None:

        data = DowntimeInput(
            downtime_hours=D("5"),
            production_loss_per_hour=D("-8000"),
        )

        with pytest.raises(ValueError):
            engine.calculate_downtime_cost(data)

    def test_negative_idle_labor_rejected(
        self,
        engine: DowntimeCostProtocol,
    ) -> None:

        data = DowntimeInput(
            downtime_hours=D("5"),
            production_loss_per_hour=D("8000"),
            idle_labor_cost_per_hour=D("-1000"),
        )

        with pytest.raises(ValueError):
            engine.calculate_downtime_cost(data)

    @pytest.mark.parametrize(
        "probability",
        [
            D("-0.1"),
            D("1.1"),
            D("2"),
        ],
    )
    def test_invalid_failure_probability(
        self,
        engine: DowntimeCostProtocol,
        probability: Decimal,
    ) -> None:

        data = DowntimeForecastInput(
            expected_downtime_hours=D("10"),
            probability_of_failure=probability,
            production_loss_per_hour=D("8000"),
        )

        with pytest.raises(ValueError):
            engine.calculate_expected_downtime_cost(
                data
            )

    def test_nan_rejected(
        self,
        engine: DowntimeCostProtocol,
    ) -> None:

        data = DowntimeInput(
            downtime_hours=D("NaN"),
            production_loss_per_hour=D("8000"),
        )

        with pytest.raises(
            (ValueError, ArithmeticError)
        ):
            engine.calculate_downtime_cost(data)

    def test_infinity_rejected(
        self,
        engine: DowntimeCostProtocol,
    ) -> None:

        data = DowntimeInput(
            downtime_hours=D("Infinity"),
            production_loss_per_hour=D("8000"),
        )

        with pytest.raises(
            (ValueError, ArithmeticError)
        ):
            engine.calculate_downtime_cost(data)


# ==================================================
# Financial integrity
# ==================================================

class TestFinancialIntegrity:

    def test_deterministic_downtime_cost(
        self,
        engine: DowntimeCostProtocol,
        downtime_input: DowntimeInput,
    ) -> None:

        first = engine.calculate_downtime_cost(
            downtime_input
        )

        second = engine.calculate_downtime_cost(
            downtime_input
        )

        assert_money(first, second)

    def test_decimal_precision(
        self,
        engine: DowntimeCostProtocol,
    ) -> None:

        data = DowntimeInput(
            downtime_hours=D("0.1"),
            production_loss_per_hour=D("0.2"),
        )

        result = engine.calculate_downtime_cost(data)

        assert_money(result, D("0.02"))

    def test_large_downtime_cost(
        self,
        engine: DowntimeCostProtocol,
    ) -> None:

        data = DowntimeInput(
            downtime_hours=D("1000"),
            production_loss_per_hour=D("1000000"),
        )

        result = engine.calculate_downtime_cost(data)

        assert_money(result, D("1000000000"))

    def test_nonnegative_downtime_cost(
        self,
        engine: DowntimeCostProtocol,
        downtime_input: DowntimeInput,
    ) -> None:

        result = engine.calculate_downtime_cost(
            downtime_input
        )

        assert D(result) >= 0

    def test_cost_increases_with_downtime(
        self,
        engine: DowntimeCostProtocol,
    ) -> None:

        short = DowntimeInput(
            downtime_hours=D("2"),
            production_loss_per_hour=D("5000"),
        )

        long = DowntimeInput(
            downtime_hours=D("10"),
            production_loss_per_hour=D("5000"),
        )

        short_cost = engine.calculate_downtime_cost(
            short
        )

        long_cost = engine.calculate_downtime_cost(
            long
        )

        assert long_cost > short_cost
