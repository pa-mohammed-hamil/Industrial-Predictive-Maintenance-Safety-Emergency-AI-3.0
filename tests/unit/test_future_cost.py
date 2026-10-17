
"""
tests/unit/test_future_cost.py

Industrial Predictive Maintenance, Safety &
Emergency Intelligence AI

Future Cost Forecasting Unit Tests

Run:
    pytest tests/unit/test_future_cost.py -v

Coverage:
    - Future value calculations
    - Inflation-adjusted costs
    - Maintenance cost forecasting
    - Repair cost forecasting
    - Replacement cost forecasting
    - Expected failure costs
    - Downtime loss forecasting
    - Discounted cash flows
    - Net present value
    - Multi-year projections
    - Scenario analysis
    - Cost escalation
    - Financial precision
    - Input validation
    - Deterministic behavior

IMPORTANT:
This file specifies proposed financial rules.
Connect the adapter to the production forecasting
engine before running engine-dependent tests.

Synthetic currency: INR.
No live industrial integrations are required.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol

import pytest


# ==================================================
# Decimal utilities
# ==================================================

def D(value: Any) -> Decimal:
    return Decimal(str(value))


def assert_decimal_equal(
    actual: Any,
    expected: Any,
    tolerance: Decimal = D("0.000001"),
) -> None:
    assert abs(D(actual) - D(expected)) <= tolerance


# ==================================================
# Forecast input models
# ==================================================

@dataclass(frozen=True)
class FutureCostInput:
    current_cost: Decimal
    annual_growth_rate: Decimal
    years: int


@dataclass(frozen=True)
class MaintenanceForecastInput:
    annual_maintenance_cost: Decimal
    annual_increase_rate: Decimal
    years: int


@dataclass(frozen=True)
class FailureForecastInput:
    failure_probability: Decimal
    repair_cost: Decimal
    downtime_cost: Decimal
    emergency_cost: Decimal = D("0")


@dataclass(frozen=True)
class DiscountedCashFlowInput:
    cash_flows: tuple[Decimal, ...]
    discount_rate: Decimal
    initial_investment: Decimal = D("0")


@dataclass(frozen=True)
class ScenarioInput:
    base_cost: Decimal
    optimistic_growth: Decimal
    expected_growth: Decimal
    pessimistic_growth: Decimal
    years: int


# ==================================================
# Forecast engine protocol
# ==================================================

class FutureCostEngineProtocol(Protocol):

    def future_value(
        self,
        data: FutureCostInput,
    ) -> Decimal:
        ...

    def forecast_maintenance(
        self,
        data: MaintenanceForecastInput,
    ) -> list[Decimal]:
        ...

    def forecast_repair_cost(
        self,
        data: FutureCostInput,
    ) -> Decimal:
        ...

    def forecast_replacement_cost(
        self,
        data: FutureCostInput,
    ) -> Decimal:
        ...

    def expected_failure_cost(
        self,
        data: FailureForecastInput,
    ) -> Decimal:
        ...

    def discounted_cash_flow(
        self,
        data: DiscountedCashFlowInput,
    ) -> Decimal:
        ...

    def scenario_analysis(
        self,
        data: ScenarioInput,
    ) -> dict[str, Decimal]:
        ...


# ==================================================
# Production adapter
# ==================================================

class FutureCostAdapter:
    """
    Connect this adapter to the actual forecasting
    implementation in your backend.

    Possible module location:

        app/cost/future_cost.py

    Adjust the import path to match your repository.

    Example:

        from app.cost.future_cost import FutureCostEngine

        self.engine = FutureCostEngine()

    Map the production method signatures and
    outputs to the protocol above.

    Do not implement forecasting formulas inside
    this adapter if your goal is to verify the
    production engine.
    """

    def __init__(self) -> None:
        self.engine: Any = None

        # TODO: Initialize production engine.

    def future_value(
        self,
        data: FutureCostInput,
    ) -> Decimal:
        raise NotImplementedError

    def forecast_maintenance(
        self,
        data: MaintenanceForecastInput,
    ) -> list[Decimal]:
        raise NotImplementedError

    def forecast_repair_cost(
        self,
        data: FutureCostInput,
    ) -> Decimal:
        raise NotImplementedError

    def forecast_replacement_cost(
        self,
        data: FutureCostInput,
    ) -> Decimal:
        raise NotImplementedError

    def expected_failure_cost(
        self,
        data: FailureForecastInput,
    ) -> Decimal:
        raise NotImplementedError

    def discounted_cash_flow(
        self,
        data: DiscountedCashFlowInput,
    ) -> Decimal:
        raise NotImplementedError

    def scenario_analysis(
        self,
        data: ScenarioInput,
    ) -> dict[str, Decimal]:
        raise NotImplementedError


# ==================================================
# Fixtures
# ==================================================

@pytest.fixture
def engine() -> FutureCostEngineProtocol:

    adapter = FutureCostAdapter()

    if adapter.engine is None:
        pytest.skip(
            "Connect FutureCostAdapter to your "
            "production future cost engine."
        )

    return adapter


@pytest.fixture
def future_cost_input() -> FutureCostInput:
    return FutureCostInput(
        current_cost=D("100000"),
        annual_growth_rate=D("0.05"),
        years=3,
    )


@pytest.fixture
def maintenance_input() -> MaintenanceForecastInput:
    return MaintenanceForecastInput(
        annual_maintenance_cost=D("20000"),
        annual_increase_rate=D("0.10"),
        years=3,
    )


@pytest.fixture
def failure_input() -> FailureForecastInput:
    return FailureForecastInput(
        failure_probability=D("0.25"),
        repair_cost=D("20000"),
        downtime_cost=D("80000"),
        emergency_cost=D("10000"),
    )


@pytest.fixture
def scenario_input() -> ScenarioInput:
    return ScenarioInput(
        base_cost=D("100000"),
        optimistic_growth=D("0.02"),
        expected_growth=D("0.05"),
        pessimistic_growth=D("0.10"),
        years=3,
    )


# ==================================================
# Future value tests
# ==================================================

class TestFutureValue:

    def test_future_value_three_years(
        self,
        engine: FutureCostEngineProtocol,
        future_cost_input: FutureCostInput,
    ) -> None:

        result = engine.future_value(
            future_cost_input
        )

        # 100000 * (1.05 ** 3)
        assert_decimal_equal(
            result,
            D("115762.5"),
        )

    def test_zero_growth(
        self,
        engine: FutureCostEngineProtocol,
    ) -> None:

        data = FutureCostInput(
            current_cost=D("100000"),
            annual_growth_rate=D("0"),
            years=5,
        )

        result = engine.future_value(data)

        assert_decimal_equal(
            result,
            D("100000"),
        )

    def test_zero_years(
        self,
        engine: FutureCostEngineProtocol,
    ) -> None:

        data = FutureCostInput(
            current_cost=D("50000"),
            annual_growth_rate=D("0.10"),
            years=0,
        )

        result = engine.future_value(data)

        assert_decimal_equal(
            result,
            D("50000"),
        )

    def test_negative_growth(
        self,
        engine: FutureCostEngineProtocol,
    ) -> None:

        data = FutureCostInput(
            current_cost=D("100000"),
            annual_growth_rate=D("-0.10"),
            years=2,
        )

        result = engine.future_value(data)

        # 100000 * 0.9 ** 2
        assert_decimal_equal(
            result,
            D("81000"),
        )

    def test_future_cost_increases_with_growth(
        self,
        engine: FutureCostEngineProtocol,
    ) -> None:

        low = FutureCostInput(
            current_cost=D("100000"),
            annual_growth_rate=D("0.03"),
            years=5,
        )

        high = FutureCostInput(
            current_cost=D("100000"),
            annual_growth_rate=D("0.10"),
            years=5,
        )

        assert (
            engine.future_value(high)
            > engine.future_value(low)
        )


# ==================================================
# Maintenance forecasting
# ==================================================

class TestMaintenanceForecast:

    def test_three_year_maintenance_forecast(
        self,
        engine: FutureCostEngineProtocol,
        maintenance_input: MaintenanceForecastInput,
    ) -> None:

        result = engine.forecast_maintenance(
            maintenance_input
        )

        # Year 1 = 20000
        # Year 2 = 22000
        # Year 3 = 24200

        expected = [
            D("20000"),
            D("22000"),
            D("24200"),
        ]

        assert len(result) == len(expected)

        for actual, target in zip(
            result,
            expected,
        ):
            assert_decimal_equal(
                actual,
                target,
            )

    def test_zero_maintenance_growth(
        self,
        engine: FutureCostEngineProtocol,
    ) -> None:

        data = MaintenanceForecastInput(
            annual_maintenance_cost=D("15000"),
            annual_increase_rate=D("0"),
            years=3,
        )

        result = engine.forecast_maintenance(data)

        assert result == [
            D("15000"),
            D("15000"),
            D("15000"),
        ]

    def test_zero_forecast_years(
        self,
        engine: FutureCostEngineProtocol,
    ) -> None:

        data = MaintenanceForecastInput(
            annual_maintenance_cost=D("15000"),
            annual_increase_rate=D("0.05"),
            years=0,
        )

        result = engine.forecast_maintenance(data)

        assert result == []

    def test_maintenance_cost_growth(
        self,
        engine: FutureCostEngineProtocol,
        maintenance_input: MaintenanceForecastInput,
    ) -> None:

        result = engine.forecast_maintenance(
            maintenance_input
        )

        assert result[1] > result[0]
        assert result[2] > result[1]

    def test_maintenance_forecast_total(
        self,
        engine: FutureCostEngineProtocol,
        maintenance_input: MaintenanceForecastInput,
    ) -> None:

        result = engine.forecast_maintenance(
            maintenance_input
        )

        assert_decimal_equal(
            sum(result),
            D("66200"),
        )


# ==================================================
# Repair cost forecasting
# ==================================================

class TestRepairCostForecast:

    def test_future_repair_cost(
        self,
        engine: FutureCostEngineProtocol,
    ) -> None:

        data = FutureCostInput(
            current_cost=D("20000"),
            annual_growth_rate=D("0.05"),
            years=2,
        )

        result = engine.forecast_repair_cost(data)

        assert_decimal_equal(
            result,
            D("22050"),
        )

    def test_repair_cost_no_growth(
        self,
        engine: FutureCostEngineProtocol,
    ) -> None:

        data = FutureCostInput(
            current_cost=D("20000"),
            annual_growth_rate=D("0"),
            years=5,
        )

        result = engine.forecast_repair_cost(data)

        assert_decimal_equal(
            result,
            D("20000"),
        )

    def test_repair_cost_increases_over_time(
        self,
        engine: FutureCostEngineProtocol,
    ) -> None:

        short = FutureCostInput(
            current_cost=D("20000"),
            annual_growth_rate=D("0.08"),
            years=1,
        )

        long = FutureCostInput(
            current_cost=D("20000"),
            annual_growth_rate=D("0.08"),
            years=5,
        )

        assert (
            engine.forecast_repair_cost(long)
            > engine.forecast_repair_cost(short)
        )


# ==================================================
# Replacement cost forecasting
# ==================================================

class TestReplacementCostForecast:

    def test_future_replacement_cost(
        self,
        engine: FutureCostEngineProtocol,
    ) -> None:

        data = FutureCostInput(
            current_cost=D("150000"),
            annual_growth_rate=D("0.05"),
            years=2,
        )

        result = engine.forecast_replacement_cost(
            data
        )

        assert_decimal_equal(
            result,
            D("165375"),
        )

    def test_zero_replacement_growth(
        self,
        engine: FutureCostEngineProtocol,
    ) -> None:

        data = FutureCostInput(
            current_cost=D("150000"),
            annual_growth_rate=D("0"),
            years=5,
        )

        result = engine.forecast_replacement_cost(
            data
        )

        assert_decimal_equal(
            result,
            D("150000"),
        )

    def test_replacement_cost_growth(
        self,
        engine: FutureCostEngineProtocol,
    ) -> None:

        low = FutureCostInput(
            current_cost=D("150000"),
            annual_growth_rate=D("0.03"),
            years=5,
        )

        high = FutureCostInput(
            current_cost=D("150000"),
            annual_growth_rate=D("0.10"),
            years=5,
        )

        assert (
            engine.forecast_replacement_cost(high)
            > engine.forecast_replacement_cost(low)
        )


# ==================================================
# Expected failure costs
# ==================================================

class TestExpectedFailureCost:

    def test_expected_failure_cost(
        self,
        engine: FutureCostEngineProtocol,
        failure_input: FailureForecastInput,
    ) -> None:

        result = engine.expected_failure_cost(
            failure_input
        )

        # 0.25 * (20000 + 80000 + 10000)
        assert_decimal_equal(
            result,
            D("27500"),
        )

    def test_zero_failure_probability(
        self,
        engine: FutureCostEngineProtocol,
    ) -> None:

        data = FailureForecastInput(
            failure_probability=D("0"),
            repair_cost=D("20000"),
            downtime_cost=D("80000"),
        )

        result = engine.expected_failure_cost(data)

        assert_decimal_equal(
            result,
            D("0"),
        )

    def test_certain_failure(
        self,
        engine: FutureCostEngineProtocol,
    ) -> None:

        data = FailureForecastInput(
            failure_probability=D("1"),
            repair_cost=D("20000"),
            downtime_cost=D("80000"),
        )

        result = engine.expected_failure_cost(data)

        assert_decimal_equal(
            result,
            D("100000"),
        )

    def test_expected_cost_scales_with_probability(
        self,
        engine: FutureCostEngineProtocol,
    ) -> None:

        low = FailureForecastInput(
            failure_probability=D("0.1"),
            repair_cost=D("20000"),
            downtime_cost=D("80000"),
        )

        high = FailureForecastInput(
            failure_probability=D("0.8"),
            repair_cost=D("20000"),
            downtime_cost=D("80000"),
        )

        assert (
            engine.expected_failure_cost(high)
            > engine.expected_failure_cost(low)
        )


# ==================================================
# Discounted cash flow and NPV
# ==================================================

class TestDiscountedCashFlow:

    def test_discounted_cash_flow(
        self,
        engine: FutureCostEngineProtocol,
    ) -> None:

        data = DiscountedCashFlowInput(
            cash_flows=(
                D("10000"),
                D("10000"),
                D("10000"),
            ),
            discount_rate=D("0.10"),
        )

        result = engine.discounted_cash_flow(
            data
        )

        expected = (
            D("10000") / D("1.1")
            + D("10000") / (D("1.1") ** 2)
            + D("10000") / (D("1.1") ** 3)
        )

        assert_decimal_equal(
            result,
            expected,
        )

    def test_zero_discount_rate(
        self,
        engine: FutureCostEngineProtocol,
    ) -> None:

        data = DiscountedCashFlowInput(
            cash_flows=(
                D("10000"),
                D("20000"),
                D("30000"),
            ),
            discount_rate=D("0"),
        )

        result = engine.discounted_cash_flow(
            data
        )

        assert_decimal_equal(
            result,
            D("60000"),
        )

    def test_npv_with_initial_investment(
        self,
        engine: FutureCostEngineProtocol,
    ) -> None:

        data = DiscountedCashFlowInput(
            cash_flows=(
                D("30000"),
                D("30000"),
            ),
            discount_rate=D("0.10"),
            initial_investment=D("50000"),
        )

        result = engine.discounted_cash_flow(
            data
        )

        expected = (
            D("30000") / D("1.1")
            + D("30000") / (D("1.1") ** 2)
            - D("50000")
        )

        assert_decimal_equal(
            result,
            expected,
        )

    def test_empty_cash_flows(
        self,
        engine: FutureCostEngineProtocol,
    ) -> None:

        data = DiscountedCashFlowInput(
            cash_flows=(),
            discount_rate=D("0.10"),
        )

        result = engine.discounted_cash_flow(
            data
        )

        assert_decimal_equal(
            result,
            D("0"),
        )

    def test_discount_rate_reduces_present_value(
        self,
        engine: FutureCostEngineProtocol,
    ) -> None:

        low = DiscountedCashFlowInput(
            cash_flows=(
                D("10000"),
                D("10000"),
            ),
            discount_rate=D("0.05"),
        )

        high = DiscountedCashFlowInput(
            cash_flows=(
                D("10000"),
                D("10000"),
            ),
            discount_rate=D("0.15"),
        )

        assert (
            engine.discounted_cash_flow(high)
            < engine.discounted_cash_flow(low)
        )


# ==================================================
# Scenario analysis
# ==================================================

class TestScenarioAnalysis:

    def test_scenario_analysis(
        self,
        engine: FutureCostEngineProtocol,
        scenario_input: ScenarioInput,
    ) -> None:

        result = engine.scenario_analysis(
            scenario_input
        )

        assert set(result) >= {
            "optimistic",
            "expected",
            "pessimistic",
        }

    def test_scenario_order(
        self,
        engine: FutureCostEngineProtocol,
        scenario_input: ScenarioInput,
    ) -> None:

        result = engine.scenario_analysis(
            scenario_input
        )

        assert (
            result["optimistic"]
            <= result["expected"]
            <= result["pessimistic"]
        )

    def test_scenario_values(
        self,
        engine: FutureCostEngineProtocol,
        scenario_input: ScenarioInput,
    ) -> None:

        result = engine.scenario_analysis(
            scenario_input
        )

        assert_decimal_equal(
            result["optimistic"],
            D("106120.8"),
        )

        assert_decimal_equal(
            result["expected"],
            D("115762.5"),
        )

        assert_decimal_equal(
            result["pessimistic"],
            D("133100"),
        )

    def test_zero_year_scenarios(
        self,
        engine: FutureCostEngineProtocol,
    ) -> None:

        data = ScenarioInput(
            base_cost=D("100000"),
            optimistic_growth=D("0.02"),
            expected_growth=D("0.05"),
            pessimistic_growth=D("0.10"),
            years=0,
        )

        result = engine.scenario_analysis(data)

        for value in result.values():
            assert_decimal_equal(
                value,
                D("100000"),
            )


# ==================================================
# Validation
# ==================================================

class TestFutureCostValidation:

    def test_negative_current_cost_rejected(
        self,
        engine: FutureCostEngineProtocol,
    ) -> None:

        data = FutureCostInput(
            current_cost=D("-10000"),
            annual_growth_rate=D("0.05"),
            years=3,
        )

        with pytest.raises(ValueError):
            engine.future_value(data)

    def test_negative_years_rejected(
        self,
        engine: FutureCostEngineProtocol,
    ) -> None:

        data = FutureCostInput(
            current_cost=D("10000"),
            annual_growth_rate=D("0.05"),
            years=-1,
        )

        with pytest.raises(ValueError):
            engine.future_value(data)

    def test_growth_below_negative_one_rejected(
        self,
        engine: FutureCostEngineProtocol,
    ) -> None:

        data = FutureCostInput(
            current_cost=D("10000"),
            annual_growth_rate=D("-1.5"),
            years=3,
        )

        with pytest.raises(ValueError):
            engine.future_value(data)

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
        engine: FutureCostEngineProtocol,
        probability: Decimal,
    ) -> None:

        data = FailureForecastInput(
            failure_probability=probability,
            repair_cost=D("20000"),
            downtime_cost=D("80000"),
        )

        with pytest.raises(ValueError):
            engine.expected_failure_cost(data)

    def test_nan_cost_rejected(
        self,
        engine: FutureCostEngineProtocol,
    ) -> None:

        data = FutureCostInput(
            current_cost=D("NaN"),
            annual_growth_rate=D("0.05"),
            years=3,
        )

        with pytest.raises(
            (ValueError, ArithmeticError)
        ):
            engine.future_value(data)

    def test_infinite_cost_rejected(
        self,
        engine: FutureCostEngineProtocol,
    ) -> None:

        data = FutureCostInput(
            current_cost=D("Infinity"),
            annual_growth_rate=D("0.05"),
            years=3,
        )

        with pytest.raises(
            (ValueError, ArithmeticError)
        ):
            engine.future_value(data)

    def test_invalid_discount_rate_rejected(
        self,
        engine: FutureCostEngineProtocol,
    ) -> None:

        data = DiscountedCashFlowInput(
            cash_flows=(D("10000"),),
            discount_rate=D("-1"),
        )

        with pytest.raises(ValueError):
            engine.discounted_cash_flow(data)


# ==================================================
# Financial integrity
# ==================================================

class TestFinancialIntegrity:

    def test_forecast_is_deterministic(
        self,
        engine: FutureCostEngineProtocol,
        future_cost_input: FutureCostInput,
    ) -> None:

        results = [
            engine.future_value(future_cost_input)
            for _ in range(5)
        ]

        for result in results:
            assert_decimal_equal(
                result,
                results[0],
            )

    def test_decimal_precision(
        self,
        engine: FutureCostEngineProtocol,
    ) -> None:

        data = FutureCostInput(
            current_cost=D("0.1"),
            annual_growth_rate=D("0.2"),
            years=1,
        )

        result = engine.future_value(data)

        assert_decimal_equal(
            result,
            D("0.12"),
        )

    def test_large_future_cost(
        self,
        engine: FutureCostEngineProtocol,
    ) -> None:

        data = FutureCostInput(
            current_cost=D("1000000000"),
            annual_growth_rate=D("0.10"),
            years=2,
        )

        result = engine.future_value(data)

        assert_decimal_equal(
            result,
            D("1210000000"),
        )

    def test_zero_current_cost(
        self,
        engine: FutureCostEngineProtocol,
    ) -> None:

        data = FutureCostInput(
            current_cost=D("0"),
            annual_growth_rate=D("0.10"),
            years=5,
        )

        result = engine.future_value(data)

        assert_decimal_equal(
            result,
            D("0"),
        )

    def test_maintenance_forecast_repeatability(
        self,
        engine: FutureCostEngineProtocol,
        maintenance_input: MaintenanceForecastInput,
    ) -> None:

        first = engine.forecast_maintenance(
            maintenance_input
        )

        second = engine.forecast_maintenance(
            maintenance_input
        )

        assert first == second
