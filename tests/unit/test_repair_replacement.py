
"""
tests/unit/test_repair_replacement.py

Industrial Predictive Maintenance, Safety &
Emergency Intelligence AI

Repair vs Replacement Decision Engine Tests

Run:
    pytest tests/unit/test_repair_replacement.py -v

Coverage:
    - Repair cost calculation
    - Replacement cost calculation
    - Labor and spare parts
    - Downtime and installation costs
    - Salvage value
    - Repair vs replacement comparison
    - Future failure cost
    - Remaining useful life (RUL)
    - Asset age and condition
    - Cost savings
    - Decision confidence
    - Invalid financial inputs
    - Batch recommendations
    - Deterministic results

IMPORTANT:
Connect RepairReplacementAdapter to the actual
production repair/replacement engine.

All amounts are synthetic INR values.
These are financial decision-support tests,
not authorization for physical asset replacement.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol

import pytest


# ==================================================
# Decimal helpers
# ==================================================

def D(value: Any) -> Decimal:
    return Decimal(str(value))


def assert_money_equal(
    actual: Any,
    expected: Any,
    tolerance: Decimal = D("0.000001"),
) -> None:
    assert abs(D(actual) - D(expected)) <= tolerance


# ==================================================
# Input models
# ==================================================

@dataclass(frozen=True)
class RepairCostInput:
    parts_cost: Decimal
    labor_hours: Decimal
    labor_rate: Decimal
    downtime_cost: Decimal = D("0")
    additional_cost: Decimal = D("0")


@dataclass(frozen=True)
class ReplacementCostInput:
    new_asset_cost: Decimal
    installation_cost: Decimal
    downtime_cost: Decimal = D("0")
    disposal_cost: Decimal = D("0")
    salvage_value: Decimal = D("0")


@dataclass(frozen=True)
class RepairReplacementInput:
    asset_id: str
    repair_cost: Decimal
    replacement_cost: Decimal
    expected_future_repair_cost: Decimal = D("0")
    expected_future_failure_cost: Decimal = D("0")
    asset_age_years: Decimal = D("0")
    expected_life_years: Decimal = D("10")
    remaining_useful_life_hours: Decimal | None = None
    failure_probability: Decimal = D("0")
    repair_restores_reliability: bool = True


@dataclass(frozen=True)
class RepairReplacementResult:
    asset_id: str
    recommendation: str
    repair_total: Decimal
    replacement_total: Decimal
    estimated_savings: Decimal
    reason: str
    confidence: float | None = None


# ==================================================
# Production protocol
# ==================================================

class RepairReplacementProtocol(Protocol):

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

    def recommend(
        self,
        data: RepairReplacementInput,
    ) -> RepairReplacementResult:
        ...

    def recommend_batch(
        self,
        assets: list[RepairReplacementInput],
    ) -> list[RepairReplacementResult]:
        ...


# ==================================================
# Production adapter
# ==================================================

class RepairReplacementAdapter:
    """
    Connect to the production implementation.

    Possible module locations:

        app/cost/repair_replacement.py
        app/services/repair_replacement_service.py

    Example:

        from app.cost.repair_replacement import (
            RepairReplacementEngine
        )

        self.engine = RepairReplacementEngine()

    Replace the example with your actual module,
    class, constructor, and method signatures.

    IMPORTANT:
    The adapter must call the production engine.
    Do not duplicate its formulas here, because
    that would test a second implementation.
    """

    def __init__(self) -> None:
        self.engine: Any = None

        # TODO:
        # self.engine = RepairReplacementEngine()

    def calculate_repair_cost(
        self,
        data: RepairCostInput,
    ) -> Decimal:

        if self.engine is None:
            raise RuntimeError(
                "Repair/replacement engine not configured"
            )

        # TODO: Map to production method.
        raise NotImplementedError

    def calculate_replacement_cost(
        self,
        data: ReplacementCostInput,
    ) -> Decimal:

        if self.engine is None:
            raise RuntimeError(
                "Repair/replacement engine not configured"
            )

        # TODO: Map to production method.
        raise NotImplementedError

    def recommend(
        self,
        data: RepairReplacementInput,
    ) -> RepairReplacementResult:

        if self.engine is None:
            raise RuntimeError(
                "Repair/replacement engine not configured"
            )

        # TODO:
        # Convert production output to:
        #
        # RepairReplacementResult(
        #     asset_id=...,
        #     recommendation=...,
        #     repair_total=...,
        #     replacement_total=...,
        #     estimated_savings=...,
        #     reason=...,
        #     confidence=...,
        # )

        raise NotImplementedError

    def recommend_batch(
        self,
        assets: list[RepairReplacementInput],
    ) -> list[RepairReplacementResult]:

        return [
            self.recommend(asset)
            for asset in assets
        ]


# ==================================================
# Fixtures
# ==================================================

@pytest.fixture
def engine() -> RepairReplacementProtocol:

    adapter = RepairReplacementAdapter()

    if adapter.engine is None:
        pytest.skip(
            "Connect RepairReplacementAdapter "
            "to the production engine."
        )

    return adapter


@pytest.fixture
def repair_input() -> RepairCostInput:

    return RepairCostInput(
        parts_cost=D("12000"),
        labor_hours=D("8"),
        labor_rate=D("500"),
        downtime_cost=D("2000"),
        additional_cost=D("1000"),
    )


@pytest.fixture
def replacement_input() -> ReplacementCostInput:

    return ReplacementCostInput(
        new_asset_cost=D("150000"),
        installation_cost=D("12000"),
        downtime_cost=D("5000"),
        disposal_cost=D("3000"),
        salvage_value=D("10000"),
    )


@pytest.fixture
def repair_favorable() -> RepairReplacementInput:

    return RepairReplacementInput(
        asset_id="MOTOR-001",
        repair_cost=D("20000"),
        replacement_cost=D("150000"),
        expected_future_repair_cost=D("5000"),
        expected_future_failure_cost=D("5000"),
        asset_age_years=D("3"),
        expected_life_years=D("12"),
        remaining_useful_life_hours=D("20000"),
        failure_probability=D("0.05"),
        repair_restores_reliability=True,
    )


@pytest.fixture
def replacement_favorable() -> RepairReplacementInput:

    return RepairReplacementInput(
        asset_id="PUMP-002",
        repair_cost=D("100000"),
        replacement_cost=D("120000"),
        expected_future_repair_cost=D("50000"),
        expected_future_failure_cost=D("60000"),
        asset_age_years=D("14"),
        expected_life_years=D("15"),
        remaining_useful_life_hours=D("200"),
        failure_probability=D("0.85"),
        repair_restores_reliability=False,
    )


# ==================================================
# Repair cost calculations
# ==================================================

class TestRepairCost:

    def test_total_repair_cost(
        self,
        engine: RepairReplacementProtocol,
        repair_input: RepairCostInput,
    ) -> None:

        result = engine.calculate_repair_cost(
            repair_input
        )

        # Parts:      12000
        # Labor:      8 * 500 = 4000
        # Downtime:   2000
        # Additional: 1000
        # Total:      19000

        assert_money_equal(
            result,
            D("19000"),
        )

    def test_parts_only_repair(
        self,
        engine: RepairReplacementProtocol,
    ) -> None:

        data = RepairCostInput(
            parts_cost=D("10000"),
            labor_hours=D("0"),
            labor_rate=D("500"),
        )

        result = engine.calculate_repair_cost(data)

        assert_money_equal(
            result,
            D("10000"),
        )

    def test_labor_only_repair(
        self,
        engine: RepairReplacementProtocol,
    ) -> None:

        data = RepairCostInput(
            parts_cost=D("0"),
            labor_hours=D("10"),
            labor_rate=D("800"),
        )

        result = engine.calculate_repair_cost(data)

        assert_money_equal(
            result,
            D("8000"),
        )

    def test_downtime_included(
        self,
        engine: RepairReplacementProtocol,
    ) -> None:

        data = RepairCostInput(
            parts_cost=D("5000"),
            labor_hours=D("5"),
            labor_rate=D("500"),
            downtime_cost=D("10000"),
        )

        result = engine.calculate_repair_cost(data)

        assert_money_equal(
            result,
            D("17500"),
        )

    def test_zero_repair_cost(
        self,
        engine: RepairReplacementProtocol,
    ) -> None:

        data = RepairCostInput(
            parts_cost=D("0"),
            labor_hours=D("0"),
            labor_rate=D("0"),
        )

        result = engine.calculate_repair_cost(data)

        assert_money_equal(result, D("0"))

    def test_repair_cost_increases_with_labor(
        self,
        engine: RepairReplacementProtocol,
    ) -> None:

        low = RepairCostInput(
            parts_cost=D("5000"),
            labor_hours=D("2"),
            labor_rate=D("500"),
        )

        high = RepairCostInput(
            parts_cost=D("5000"),
            labor_hours=D("10"),
            labor_rate=D("500"),
        )

        assert (
            engine.calculate_repair_cost(high)
            > engine.calculate_repair_cost(low)
        )


# ==================================================
# Replacement cost calculations
# ==================================================

class TestReplacementCost:

    def test_total_replacement_cost(
        self,
        engine: RepairReplacementProtocol,
        replacement_input: ReplacementCostInput,
    ) -> None:

        result = engine.calculate_replacement_cost(
            replacement_input
        )

        # New asset:    150000
        # Installation: 12000
        # Downtime:     5000
        # Disposal:     3000
        # Salvage:     -10000
        # Total:        160000

        assert_money_equal(
            result,
            D("160000"),
        )

    def test_asset_price_only(
        self,
        engine: RepairReplacementProtocol,
    ) -> None:

        data = ReplacementCostInput(
            new_asset_cost=D("100000"),
            installation_cost=D("0"),
        )

        result = engine.calculate_replacement_cost(
            data
        )

        assert_money_equal(
            result,
            D("100000"),
        )

    def test_installation_cost_included(
        self,
        engine: RepairReplacementProtocol,
    ) -> None:

        data = ReplacementCostInput(
            new_asset_cost=D("100000"),
            installation_cost=D("15000"),
        )

        result = engine.calculate_replacement_cost(
            data
        )

        assert_money_equal(
            result,
            D("115000"),
        )

    def test_salvage_reduces_replacement_cost(
        self,
        engine: RepairReplacementProtocol,
    ) -> None:

        without_salvage = ReplacementCostInput(
            new_asset_cost=D("100000"),
            installation_cost=D("10000"),
            salvage_value=D("0"),
        )

        with_salvage = ReplacementCostInput(
            new_asset_cost=D("100000"),
            installation_cost=D("10000"),
            salvage_value=D("15000"),
        )

        assert (
            engine.calculate_replacement_cost(
                with_salvage
            )
            < engine.calculate_replacement_cost(
                without_salvage
            )
        )

    def test_downtime_increases_replacement_cost(
        self,
        engine: RepairReplacementProtocol,
    ) -> None:

        low = ReplacementCostInput(
            new_asset_cost=D("100000"),
            installation_cost=D("10000"),
            downtime_cost=D("1000"),
        )

        high = ReplacementCostInput(
            new_asset_cost=D("100000"),
            installation_cost=D("10000"),
            downtime_cost=D("20000"),
        )

        assert (
            engine.calculate_replacement_cost(high)
            > engine.calculate_replacement_cost(low)
        )


# ==================================================
# Decision recommendations
# ==================================================

class TestRepairReplacementDecision:

    def test_repair_recommended(
        self,
        engine: RepairReplacementProtocol,
        repair_favorable: RepairReplacementInput,
    ) -> None:

        result = engine.recommend(
            repair_favorable
        )

        assert result.recommendation == "repair"

    def test_replacement_recommended(
        self,
        engine: RepairReplacementProtocol,
        replacement_favorable: RepairReplacementInput,
    ) -> None:

        result = engine.recommend(
            replacement_favorable
        )

        assert result.recommendation == "replace"

    def test_recommendation_is_valid(
        self,
        engine: RepairReplacementProtocol,
        repair_favorable: RepairReplacementInput,
    ) -> None:

        result = engine.recommend(
            repair_favorable
        )

        assert result.recommendation in {
            "repair",
            "replace",
            "inspect",
            "manual_review",
        }

    def test_result_asset_id(
        self,
        engine: RepairReplacementProtocol,
        repair_favorable: RepairReplacementInput,
    ) -> None:

        result = engine.recommend(
            repair_favorable
        )

        assert result.asset_id == "MOTOR-001"

    def test_result_has_reason(
        self,
        engine: RepairReplacementProtocol,
        repair_favorable: RepairReplacementInput,
    ) -> None:

        result = engine.recommend(
            repair_favorable
        )

        assert isinstance(result.reason, str)
        assert result.reason.strip()

    def test_result_costs_nonnegative(
        self,
        engine: RepairReplacementProtocol,
        repair_favorable: RepairReplacementInput,
    ) -> None:

        result = engine.recommend(
            repair_favorable
        )

        assert result.repair_total >= 0
        assert result.replacement_total >= 0

    def test_confidence_range(
        self,
        engine: RepairReplacementProtocol,
        repair_favorable: RepairReplacementInput,
    ) -> None:

        result = engine.recommend(
            repair_favorable
        )

        if result.confidence is not None:
            assert 0 <= result.confidence <= 1


# ==================================================
# Financial comparison
# ==================================================

class TestFinancialComparison:

    def test_repair_favorable_total_cost(
        self,
        engine: RepairReplacementProtocol,
        repair_favorable: RepairReplacementInput,
    ) -> None:

        result = engine.recommend(
            repair_favorable
        )

        # Proposed policy:
        # Repair lifecycle cost =
        # repair + future repairs + future failures

        assert_money_equal(
            result.repair_total,
            D("30000"),
        )

        assert_money_equal(
            result.replacement_total,
            D("150000"),
        )

    def test_repair_favorable_savings(
        self,
        engine: RepairReplacementProtocol,
        repair_favorable: RepairReplacementInput,
    ) -> None:

        result = engine.recommend(
            repair_favorable
        )

        # 150000 - 30000 = 120000
        assert_money_equal(
            result.estimated_savings,
            D("120000"),
        )

    def test_replacement_favorable_total_cost(
        self,
        engine: RepairReplacementProtocol,
        replacement_favorable: RepairReplacementInput,
    ) -> None:

        result = engine.recommend(
            replacement_favorable
        )

        # Repair lifecycle cost:
        # 100000 + 50000 + 60000 = 210000

        assert_money_equal(
            result.repair_total,
            D("210000"),
        )

        assert_money_equal(
            result.replacement_total,
            D("120000"),
        )

    def test_replacement_favorable_savings(
        self,
        engine: RepairReplacementProtocol,
        replacement_favorable: RepairReplacementInput,
    ) -> None:

        result = engine.recommend(
            replacement_favorable
        )

        # 210000 - 120000 = 90000
        assert_money_equal(
            result.estimated_savings,
            D("90000"),
        )

    def test_savings_are_nonnegative(
        self,
        engine: RepairReplacementProtocol,
        repair_favorable: RepairReplacementInput,
        replacement_favorable: RepairReplacementInput,
    ) -> None:

        for data in [
            repair_favorable,
            replacement_favorable,
        ]:
            result = engine.recommend(data)

            assert result.estimated_savings >= 0


# ==================================================
# Failure risk and RUL
# ==================================================

class TestReliabilityFactors:

    def test_high_failure_risk_supported(
        self,
        engine: RepairReplacementProtocol,
        replacement_favorable: RepairReplacementInput,
    ) -> None:

        result = engine.recommend(
            replacement_favorable
        )

        assert isinstance(
            result,
            RepairReplacementResult,
        )

    def test_low_remaining_useful_life(
        self,
        engine: RepairReplacementProtocol,
        replacement_favorable: RepairReplacementInput,
    ) -> None:

        result = engine.recommend(
            replacement_favorable
        )

        assert result.recommendation == "replace"

    def test_missing_rul_supported(
        self,
        engine: RepairReplacementProtocol,
    ) -> None:

        data = RepairReplacementInput(
            asset_id="FAN-003",
            repair_cost=D("10000"),
            replacement_cost=D("100000"),
            remaining_useful_life_hours=None,
        )

        result = engine.recommend(data)

        assert isinstance(
            result,
            RepairReplacementResult,
        )

    def test_unreliable_repair_scenario(
        self,
        engine: RepairReplacementProtocol,
        replacement_favorable: RepairReplacementInput,
    ) -> None:

        result = engine.recommend(
            replacement_favorable
        )

        assert result.recommendation == "replace"


# ==================================================
# Batch processing
# ==================================================

class TestBatchRecommendations:

    def test_batch_result_count(
        self,
        engine: RepairReplacementProtocol,
        repair_favorable: RepairReplacementInput,
        replacement_favorable: RepairReplacementInput,
    ) -> None:

        assets = [
            repair_favorable,
            replacement_favorable,
        ]

        results = engine.recommend_batch(assets)

        assert len(results) == 2

    def test_batch_preserves_asset_ids(
        self,
        engine: RepairReplacementProtocol,
        repair_favorable: RepairReplacementInput,
        replacement_favorable: RepairReplacementInput,
    ) -> None:

        results = engine.recommend_batch([
            repair_favorable,
            replacement_favorable,
        ])

        assert [r.asset_id for r in results] == [
            "MOTOR-001",
            "PUMP-002",
        ]

    def test_empty_batch(
        self,
        engine: RepairReplacementProtocol,
    ) -> None:

        assert engine.recommend_batch([]) == []

    def test_batch_matches_individual(
        self,
        engine: RepairReplacementProtocol,
        repair_favorable: RepairReplacementInput,
        replacement_favorable: RepairReplacementInput,
    ) -> None:

        assets = [
            repair_favorable,
            replacement_favorable,
        ]

        batch = engine.recommend_batch(assets)

        individual = [
            engine.recommend(asset)
            for asset in assets
        ]

        assert batch == individual


# ==================================================
# Input validation
# ==================================================

class TestRepairReplacementValidation:

    def test_negative_parts_cost_rejected(
        self,
        engine: RepairReplacementProtocol,
    ) -> None:

        data = RepairCostInput(
            parts_cost=D("-1000"),
            labor_hours=D("5"),
            labor_rate=D("500"),
        )

        with pytest.raises(ValueError):
            engine.calculate_repair_cost(data)

    def test_negative_labor_hours_rejected(
        self,
        engine: RepairReplacementProtocol,
    ) -> None:

        data = RepairCostInput(
            parts_cost=D("1000"),
            labor_hours=D("-5"),
            labor_rate=D("500"),
        )

        with pytest.raises(ValueError):
            engine.calculate_repair_cost(data)

    def test_negative_labor_rate_rejected(
        self,
        engine: RepairReplacementProtocol,
    ) -> None:

        data = RepairCostInput(
            parts_cost=D("1000"),
            labor_hours=D("5"),
            labor_rate=D("-500"),
        )

        with pytest.raises(ValueError):
            engine.calculate_repair_cost(data)

    def test_negative_asset_price_rejected(
        self,
        engine: RepairReplacementProtocol,
    ) -> None:

        data = ReplacementCostInput(
            new_asset_cost=D("-100000"),
            installation_cost=D("10000"),
        )

        with pytest.raises(ValueError):
            engine.calculate_replacement_cost(data)

    def test_negative_salvage_rejected(
        self,
        engine: RepairReplacementProtocol,
    ) -> None:

        data = ReplacementCostInput(
            new_asset_cost=D("100000"),
            installation_cost=D("10000"),
            salvage_value=D("-5000"),
        )

        with pytest.raises(ValueError):
            engine.calculate_replacement_cost(data)

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
        engine: RepairReplacementProtocol,
        probability: Decimal,
    ) -> None:

        data = RepairReplacementInput(
            asset_id="MOTOR-001",
            repair_cost=D("20000"),
            replacement_cost=D("100000"),
            failure_probability=probability,
        )

        with pytest.raises(ValueError):
            engine.recommend(data)

    def test_negative_asset_age_rejected(
        self,
        engine: RepairReplacementProtocol,
    ) -> None:

        data = RepairReplacementInput(
            asset_id="MOTOR-001",
            repair_cost=D("20000"),
            replacement_cost=D("100000"),
            asset_age_years=D("-1"),
        )

        with pytest.raises(ValueError):
            engine.recommend(data)

    def test_zero_expected_life_rejected(
        self,
        engine: RepairReplacementProtocol,
    ) -> None:

        data = RepairReplacementInput(
            asset_id="MOTOR-001",
            repair_cost=D("20000"),
            replacement_cost=D("100000"),
            expected_life_years=D("0"),
        )

        with pytest.raises(ValueError):
            engine.recommend(data)

    def test_negative_rul_rejected(
        self,
        engine: RepairReplacementProtocol,
    ) -> None:

        data = RepairReplacementInput(
            asset_id="MOTOR-001",
            repair_cost=D("20000"),
            replacement_cost=D("100000"),
            remaining_useful_life_hours=D("-100"),
        )

        with pytest.raises(ValueError):
            engine.recommend(data)

    def test_empty_asset_id_rejected(
        self,
        engine: RepairReplacementProtocol,
    ) -> None:

        data = RepairReplacementInput(
            asset_id="",
            repair_cost=D("20000"),
            replacement_cost=D("100000"),
        )

        with pytest.raises(ValueError):
            engine.recommend(data)

    def test_nan_repair_cost_rejected(
        self,
        engine: RepairReplacementProtocol,
    ) -> None:

        data = RepairCostInput(
            parts_cost=D("NaN"),
            labor_hours=D("5"),
            labor_rate=D("500"),
        )

        with pytest.raises(
            (ValueError, ArithmeticError)
        ):
            engine.calculate_repair_cost(data)

    def test_infinite_replacement_cost_rejected(
        self,
        engine: RepairReplacementProtocol,
    ) -> None:

        data = ReplacementCostInput(
            new_asset_cost=D("Infinity"),
            installation_cost=D("10000"),
        )

        with pytest.raises(
            (ValueError, ArithmeticError)
        ):
            engine.calculate_replacement_cost(data)


# ==================================================
# Financial precision and integrity
# ==================================================

class TestFinancialIntegrity:

    def test_decimal_repair_precision(
        self,
        engine: RepairReplacementProtocol,
    ) -> None:

        data = RepairCostInput(
            parts_cost=D("0.10"),
            labor_hours=D("1"),
            labor_rate=D("0.20"),
        )

        result = engine.calculate_repair_cost(data)

        assert_money_equal(
            result,
            D("0.30"),
        )

    def test_large_replacement_cost(
        self,
        engine: RepairReplacementProtocol,
    ) -> None:

        data = ReplacementCostInput(
            new_asset_cost=D("1000000000"),
            installation_cost=D("50000000"),
            downtime_cost=D("10000000"),
            salvage_value=D("5000000"),
        )

        result = engine.calculate_replacement_cost(
            data
        )

        assert_money_equal(
            result,
            D("1055000000"),
        )

    def test_deterministic_recommendation(
        self,
        engine: RepairReplacementProtocol,
        repair_favorable: RepairReplacementInput,
    ) -> None:

        results = [
            engine.recommend(repair_favorable)
            for _ in range(5)
        ]

        assert all(
            result == results[0]
            for result in results
        )

    def test_input_not_modified(
        self,
        engine: RepairReplacementProtocol,
        repair_favorable: RepairReplacementInput,
    ) -> None:

        original = repair_favorable

        engine.recommend(repair_favorable)

        assert repair_favorable == original

    def test_recommendation_result_type(
        self,
        engine: RepairReplacementProtocol,
        repair_favorable: RepairReplacementInput,
    ) -> None:

        result = engine.recommend(
            repair_favorable
        )

        assert isinstance(
            result,
            RepairReplacementResult,
        )
