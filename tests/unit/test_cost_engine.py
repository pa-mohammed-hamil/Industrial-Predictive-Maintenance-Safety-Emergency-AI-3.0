
"""
tests/unit/test_cost_engine.py

Industrial Predictive Maintenance, Safety &
Emergency Intelligence AI

Cost Intelligence Engine Unit Tests

Run:
    pytest tests/unit/test_cost_engine.py -v

Coverage:
    - Repair cost
    - Replacement cost
    - Spare parts cost
    - Labor cost
    - Downtime cost
    - Production loss
    - Energy cost
    - Emergency cost
    - Scrap cost
    - Expected failure cost
    - Future cost projection
    - Repair vs replacement
    - Maintenance ROI
    - Cost aggregation
    - Input validation
    - Precision and deterministic behavior

IMPORTANT:
These are proposed business rules, not verified
production implementation details.

Connect CostEngineAdapter to the actual
app cost engine before running the tests.

All amounts are synthetic test values in INR.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol

import pytest


# ==================================================
# Financial helpers
# ==================================================

def D(value: str | int | float | Decimal) -> Decimal:
    """
    Convert numeric values to Decimal without
    introducing unnecessary binary float artifacts.
    """
    return Decimal(str(value))


def assert_money_equal(
    actual: Decimal | int | float,
    expected: Decimal | int | float,
) -> None:
    assert D(actual) == D(expected)


# ==================================================
# Input models
# ==================================================

@dataclass(frozen=True)
class RepairCostInput:
    spare_parts_cost: Decimal
    labor_hours: Decimal
    labor_rate_per_hour: Decimal
    service_cost: Decimal = Decimal("0")
    transport_cost: Decimal = Decimal("0")


@dataclass(frozen=True)
class ReplacementCostInput:
    equipment_price: Decimal
    installation_cost: Decimal
    transport_cost: Decimal
    disposal_cost: Decimal = Decimal("0")
    salvage_value: Decimal = Decimal("0")


@dataclass(frozen=True)
class DowntimeCostInput:
    downtime_hours: Decimal
    production_loss_per_hour: Decimal
    idle_labor_cost_per_hour: Decimal = Decimal("0")


@dataclass(frozen=True)
class EnergyCostInput:
    energy_kwh: Decimal
    tariff_per_kwh: Decimal


@dataclass(frozen=True)
class FailureCostInput:
    failure_probability: Decimal
    repair_cost: Decimal
    downtime_cost: Decimal
    emergency_cost: Decimal = Decimal("0")
    scrap_cost: Decimal = Decimal("0")


@dataclass(frozen=True)
class ROICalculationInput:
    investment_cost: Decimal
    avoided_failure_cost: Decimal
    operational_savings: Decimal = Decimal("0")


# ==================================================
# Cost engine interface
# ==================================================

class CostEngineProtocol(Protocol):

    def calculate_repair_cost(
        self,
        data: RepairCostInput,
    ) -> Decimal:
        ...

    def calculate_replacement_cost(
        self,
        data: ReplacementCostInput,
    ) -> Decimal:
        ...

    def calculate_downtime_cost(
        self,
        data: DowntimeCostInput,
    ) -> Decimal:
        ...

    def calculate_energy_cost(
        self,
        data: EnergyCostInput,
    ) -> Decimal:
        ...

    def calculate_expected_failure_cost(
        self,
        data: FailureCostInput,
    ) -> Decimal:
        ...

    def calculate_roi(
        self,
        data: ROICalculationInput,
    ) -> Decimal:
        ...

    def aggregate_costs(
        self,
        costs: list[Decimal],
    ) -> Decimal:
        ...

    def recommend_repair_or_replace(
        self,
        repair_cost: Decimal,
        replacement_cost: Decimal,
    ) -> str:
        ...


# ==================================================
# Production adapter
# ==================================================

class CostEngineAdapter:
    """
    Connect this adapter to your actual production
    cost engine.

    Example:

        from app.cost.cost_engine import CostEngine

        self.engine = CostEngine()

    Update the import path and method mappings
    to match the actual implementation.

    Do not implement a separate calculator here
    when the goal is to test production code.
    """

    def __init__(self) -> None:
        self.engine: Any = None

        # TODO: Initialize the actual cost engine.

    def calculate_repair_cost(
        self,
        data: RepairCostInput,
    ) -> Decimal:
        raise NotImplementedError

    def calculate_replacement_cost(
        self,
        data: ReplacementCostInput,
    ) -> Decimal:
        raise NotImplementedError

    def calculate_downtime_cost(
        self,
        data: DowntimeCostInput,
    ) -> Decimal:
        raise NotImplementedError

    def calculate_energy_cost(
        self,
        data: EnergyCostInput,
    ) -> Decimal:
        raise NotImplementedError

    def calculate_expected_failure_cost(
        self,
        data: FailureCostInput,
    ) -> Decimal:
        raise NotImplementedError

    def calculate_roi(
        self,
        data: ROICalculationInput,
    ) -> Decimal:
        raise NotImplementedError

    def aggregate_costs(
        self,
        costs: list[Decimal],
    ) -> Decimal:
        raise NotImplementedError

    def recommend_repair_or_replace(
        self,
        repair_cost: Decimal,
        replacement_cost: Decimal,
    ) -> str:
        raise NotImplementedError


# ==================================================
# Fixtures
# ==================================================

@pytest.fixture
def engine() -> CostEngineProtocol:
    adapter = CostEngineAdapter()

    if adapter.engine is None:
        pytest.skip(
            "Connect CostEngineAdapter to your "
            "production cost engine."
        )

    return adapter


@pytest.fixture
def repair_input() -> RepairCostInput:
    return RepairCostInput(
        spare_parts_cost=D("12000"),
        labor_hours=D("8"),
        labor_rate_per_hour=D("500"),
        service_cost=D("2000"),
        transport_cost=D("1000"),
    )


@pytest.fixture
def replacement_input() -> ReplacementCostInput:
    return ReplacementCostInput(
        equipment_price=D("150000"),
        installation_cost=D("12000"),
        transport_cost=D("5000"),
        disposal_cost=D("3000"),
        salvage_value=D("10000"),
    )


@pytest.fixture
def downtime_input() -> DowntimeCostInput:
    return DowntimeCostInput(
        downtime_hours=D("10"),
        production_loss_per_hour=D("8000"),
        idle_labor_cost_per_hour=D("1000"),
    )


@pytest.fixture
def failure_input() -> FailureCostInput:
    return FailureCostInput(
        failure_probability=D("0.25"),
        repair_cost=D("20000"),
        downtime_cost=D("80000"),
        emergency_cost=D("10000"),
        scrap_cost=D("5000"),
    )


# ==================================================
# Repair cost tests
# ==================================================

class TestRepairCost:

    def test_repair_cost_calculation(
        self,
        engine: CostEngineProtocol,
        repair_input: RepairCostInput,
    ) -> None:

        result = engine.calculate_repair_cost(
            repair_input
        )

        # 12000 + (8 * 500) + 2000 + 1000
        assert_money_equal(result, D("19000"))

    def test_repair_cost_zero_labor(
        self,
        engine: CostEngineProtocol,
    ) -> None:

        data = RepairCostInput(
            spare_parts_cost=D("5000"),
            labor_hours=D("0"),
            labor_rate_per_hour=D("500"),
        )

        result = engine.calculate_repair_cost(data)

        assert_money_equal(result, D("5000"))

    def test_repair_cost_only_labor(
        self,
        engine: CostEngineProtocol,
    ) -> None:

        data = RepairCostInput(
            spare_parts_cost=D("0"),
            labor_hours=D("5"),
            labor_rate_per_hour=D("800"),
        )

        result = engine.calculate_repair_cost(data)

        assert_money_equal(result, D("4000"))

    def test_repair_cost_deterministic(
        self,
        engine: CostEngineProtocol,
        repair_input: RepairCostInput,
    ) -> None:

        first = engine.calculate_repair_cost(
            repair_input
        )

        second = engine.calculate_repair_cost(
            repair_input
        )

        assert_money_equal(first, second)


# ==================================================
# Replacement cost tests
# ==================================================

class TestReplacementCost:

    def test_replacement_cost_calculation(
        self,
        engine: CostEngineProtocol,
        replacement_input: ReplacementCostInput,
    ) -> None:

        result = engine.calculate_replacement_cost(
            replacement_input
        )

        # 150000 + 12000 + 5000 + 3000 - 10000
        assert_money_equal(result, D("160000"))

    def test_replacement_without_salvage(
        self,
        engine: CostEngineProtocol,
    ) -> None:

        data = ReplacementCostInput(
            equipment_price=D("100000"),
            installation_cost=D("10000"),
            transport_cost=D("5000"),
        )

        result = engine.calculate_replacement_cost(data)

        assert_money_equal(result, D("115000"))

    def test_salvage_reduces_replacement_cost(
        self,
        engine: CostEngineProtocol,
    ) -> None:

        without_salvage = ReplacementCostInput(
            equipment_price=D("100000"),
            installation_cost=D("10000"),
            transport_cost=D("5000"),
            salvage_value=D("0"),
        )

        with_salvage = ReplacementCostInput(
            equipment_price=D("100000"),
            installation_cost=D("10000"),
            transport_cost=D("5000"),
            salvage_value=D("15000"),
        )

        first = engine.calculate_replacement_cost(
            without_salvage
        )

        second = engine.calculate_replacement_cost(
            with_salvage
        )

        assert second < first
        assert_money_equal(first - second, D("15000"))


# ==================================================
# Downtime cost tests
# ==================================================

class TestDowntimeCost:

    def test_downtime_cost(
        self,
        engine: CostEngineProtocol,
        downtime_input: DowntimeCostInput,
    ) -> None:

        result = engine.calculate_downtime_cost(
            downtime_input
        )

        # 10 * (8000 + 1000)
        assert_money_equal(result, D("90000"))

    def test_zero_downtime(
        self,
        engine: CostEngineProtocol,
    ) -> None:

        data = DowntimeCostInput(
            downtime_hours=D("0"),
            production_loss_per_hour=D("8000"),
        )

        result = engine.calculate_downtime_cost(data)

        assert_money_equal(result, D("0"))

    def test_downtime_scales_linearly(
        self,
        engine: CostEngineProtocol,
    ) -> None:

        one_hour = DowntimeCostInput(
            downtime_hours=D("1"),
            production_loss_per_hour=D("5000"),
        )

        five_hours = DowntimeCostInput(
            downtime_hours=D("5"),
            production_loss_per_hour=D("5000"),
        )

        first = engine.calculate_downtime_cost(
            one_hour
        )

        second = engine.calculate_downtime_cost(
            five_hours
        )

        assert_money_equal(second, first * 5)


# ==================================================
# Energy cost tests
# ==================================================

class TestEnergyCost:

    def test_energy_cost(
        self,
        engine: CostEngineProtocol,
    ) -> None:

        data = EnergyCostInput(
            energy_kwh=D("500"),
            tariff_per_kwh=D("8"),
        )

        result = engine.calculate_energy_cost(data)

        assert_money_equal(result, D("4000"))

    def test_zero_energy(
        self,
        engine: CostEngineProtocol,
    ) -> None:

        data = EnergyCostInput(
            energy_kwh=D("0"),
            tariff_per_kwh=D("8"),
        )

        result = engine.calculate_energy_cost(data)

        assert_money_equal(result, D("0"))

    def test_fractional_energy_cost(
        self,
        engine: CostEngineProtocol,
    ) -> None:

        data = EnergyCostInput(
            energy_kwh=D("12.5"),
            tariff_per_kwh=D("7.5"),
        )

        result = engine.calculate_energy_cost(data)

        assert_money_equal(result, D("93.75"))


# ==================================================
# Expected failure cost tests
# ==================================================

class TestExpectedFailureCost:

    def test_expected_failure_cost(
        self,
        engine: CostEngineProtocol,
        failure_input: FailureCostInput,
    ) -> None:

        result = engine.calculate_expected_failure_cost(
            failure_input
        )

        # 0.25 * (20000 + 80000 + 10000 + 5000)
        assert_money_equal(result, D("28750"))

    def test_zero_failure_probability(
        self,
        engine: CostEngineProtocol,
    ) -> None:

        data = FailureCostInput(
            failure_probability=D("0"),
            repair_cost=D("20000"),
            downtime_cost=D("80000"),
        )

        result = engine.calculate_expected_failure_cost(
            data
        )

        assert_money_equal(result, D("0"))

    def test_certain_failure(
        self,
        engine: CostEngineProtocol,
    ) -> None:

        data = FailureCostInput(
            failure_probability=D("1"),
            repair_cost=D("20000"),
            downtime_cost=D("80000"),
        )

        result = engine.calculate_expected_failure_cost(
            data
        )

        assert_money_equal(result, D("100000"))

    def test_failure_cost_increases_with_probability(
        self,
        engine: CostEngineProtocol,
    ) -> None:

        low = FailureCostInput(
            failure_probability=D("0.1"),
            repair_cost=D("20000"),
            downtime_cost=D("80000"),
        )

        high = FailureCostInput(
            failure_probability=D("0.8"),
            repair_cost=D("20000"),
            downtime_cost=D("80000"),
        )

        low_cost = engine.calculate_expected_failure_cost(
            low
        )

        high_cost = engine.calculate_expected_failure_cost(
            high
        )

        assert high_cost > low_cost


# ==================================================
# ROI tests
# ==================================================

class TestMaintenanceROI:

    def test_roi_calculation(
        self,
        engine: CostEngineProtocol,
    ) -> None:

        data = ROICalculationInput(
            investment_cost=D("20000"),
            avoided_failure_cost=D("50000"),
            operational_savings=D("10000"),
        )

        result = engine.calculate_roi(data)

        # ((50000 + 10000 - 20000) / 20000) * 100
        assert_money_equal(result, D("200"))

    def test_negative_roi(
        self,
        engine: CostEngineProtocol,
    ) -> None:

        data = ROICalculationInput(
            investment_cost=D("50000"),
            avoided_failure_cost=D("10000"),
        )

        result = engine.calculate_roi(data)

        assert_money_equal(result, D("-80"))

    def test_zero_investment_rejected(
        self,
        engine: CostEngineProtocol,
    ) -> None:

        data = ROICalculationInput(
            investment_cost=D("0"),
            avoided_failure_cost=D("10000"),
        )

        with pytest.raises(
            (ValueError, ZeroDivisionError)
        ):
            engine.calculate_roi(data)


# ==================================================
# Repair vs replacement
# ==================================================

class TestRepairReplacementDecision:

    def test_repair_recommended_when_cheaper(
        self,
        engine: CostEngineProtocol,
    ) -> None:

        result = engine.recommend_repair_or_replace(
            repair_cost=D("20000"),
            replacement_cost=D("150000"),
        )

        assert result.lower() == "repair"

    def test_replace_recommended_when_cheaper(
        self,
        engine: CostEngineProtocol,
    ) -> None:

        result = engine.recommend_repair_or_replace(
            repair_cost=D("200000"),
            replacement_cost=D("100000"),
        )

        assert result.lower() == "replace"

    def test_equal_cost_decision_is_valid(
        self,
        engine: CostEngineProtocol,
    ) -> None:

        result = engine.recommend_repair_or_replace(
            repair_cost=D("50000"),
            replacement_cost=D("50000"),
        )

        assert result.lower() in {
            "repair",
            "replace",
            "equal",
            "review",
        }


# ==================================================
# Cost aggregation
# ==================================================

class TestCostAggregation:

    def test_aggregate_costs(
        self,
        engine: CostEngineProtocol,
    ) -> None:

        result = engine.aggregate_costs([
            D("10000"),
            D("20000"),
            D("5000"),
            D("2500"),
        ])

        assert_money_equal(result, D("37500"))

    def test_empty_costs(
        self,
        engine: CostEngineProtocol,
    ) -> None:

        result = engine.aggregate_costs([])

        assert_money_equal(result, D("0"))

    def test_fractional_costs(
        self,
        engine: CostEngineProtocol,
    ) -> None:

        result = engine.aggregate_costs([
            D("10.25"),
            D("20.50"),
            D("5.75"),
        ])

        assert_money_equal(result, D("36.50"))

    def test_large_cost_values(
        self,
        engine: CostEngineProtocol,
    ) -> None:

        result = engine.aggregate_costs([
            D("1000000000"),
            D("2000000000"),
        ])

        assert_money_equal(result, D("3000000000"))


# ==================================================
# Input validation
# ==================================================

class TestCostValidation:

    def test_negative_spare_parts_cost_rejected(
        self,
        engine: CostEngineProtocol,
    ) -> None:

        data = RepairCostInput(
            spare_parts_cost=D("-100"),
            labor_hours=D("5"),
            labor_rate_per_hour=D("500"),
        )

        with pytest.raises(ValueError):
            engine.calculate_repair_cost(data)

    def test_negative_labor_hours_rejected(
        self,
        engine: CostEngineProtocol,
    ) -> None:

        data = RepairCostInput(
            spare_parts_cost=D("1000"),
            labor_hours=D("-5"),
            labor_rate_per_hour=D("500"),
        )

        with pytest.raises(ValueError):
            engine.calculate_repair_cost(data)

    def test_negative_downtime_rejected(
        self,
        engine: CostEngineProtocol,
    ) -> None:

        data = DowntimeCostInput(
            downtime_hours=D("-10"),
            production_loss_per_hour=D("5000"),
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
        engine: CostEngineProtocol,
        probability: Decimal,
    ) -> None:

        data = FailureCostInput(
            failure_probability=probability,
            repair_cost=D("10000"),
            downtime_cost=D("20000"),
        )

        with pytest.raises(ValueError):
            engine.calculate_expected_failure_cost(
                data
            )

    def test_nan_cost_rejected(
        self,
        engine: CostEngineProtocol,
    ) -> None:

        data = RepairCostInput(
            spare_parts_cost=D("NaN"),
            labor_hours=D("5"),
            labor_rate_per_hour=D("500"),
        )

        with pytest.raises(
            (ValueError, ArithmeticError)
        ):
            engine.calculate_repair_cost(data)

    def test_infinite_cost_rejected(
        self,
        engine: CostEngineProtocol,
    ) -> None:

        data = RepairCostInput(
            spare_parts_cost=D("Infinity"),
            labor_hours=D("5"),
            labor_rate_per_hour=D("500"),
        )

        with pytest.raises(
            (ValueError, ArithmeticError)
        ):
            engine.calculate_repair_cost(data)


# ==================================================
# Financial integrity
# ==================================================

class TestFinancialIntegrity:

    def test_repair_cost_is_nonnegative(
        self,
        engine: CostEngineProtocol,
        repair_input: RepairCostInput,
    ) -> None:

        result = engine.calculate_repair_cost(
            repair_input
        )

        assert D(result) >= 0

    def test_replacement_cost_is_nonnegative(
        self,
        engine: CostEngineProtocol,
        replacement_input: ReplacementCostInput,
    ) -> None:

        result = engine.calculate_replacement_cost(
            replacement_input
        )

        assert D(result) >= 0

    def test_cost_engine_is_deterministic(
        self,
        engine: CostEngineProtocol,
        repair_input: RepairCostInput,
    ) -> None:

        results = [
            engine.calculate_repair_cost(repair_input)
            for _ in range(5)
        ]

        assert all(
            D(result) == D(results[0])
            for result in results
        )

    def test_decimal_precision(
        self,
        engine: CostEngineProtocol,
    ) -> None:

        data = EnergyCostInput(
            energy_kwh=D("0.1"),
            tariff_per_kwh=D("0.2"),
        )

        result = engine.calculate_energy_cost(data)

        assert_money_equal(result, D("0.02"))
