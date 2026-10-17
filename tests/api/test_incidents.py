
"""
INDUSTRA AI
File: backend/tests/test_cost_analysis.py

Industrial Cost Intelligence — Reference Calculation Tests

Run:
    pip install pytest
    pytest backend/tests/test_cost_analysis.py -v

These tests validate standalone reference formulas.
They do NOT verify the application's production cost engine
until the reference functions are replaced with calls to
actual production implementations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest


# ============================================================
# REFERENCE FINANCIAL MODELS
# ============================================================

@dataclass(frozen=True)
class RepairCost:
    labor: float
    parts: float
    contractor: float
    logistics: float
    downtime_hours: float
    contribution_margin_per_hour: float


@dataclass(frozen=True)
class ReplacementCost:
    purchase: float
    installation: float
    commissioning: float
    logistics: float
    decommissioning: float
    disposal: float
    salvage: float
    downtime_hours: float
    contribution_margin_per_hour: float


@dataclass(frozen=True)
class SparePart:
    unit_price: float
    quantity: int
    annual_carrying_rate: float = 0.20


@dataclass(frozen=True)
class OwnershipCost:
    acquisition: float
    annual_maintenance: float
    annual_energy: float
    annual_labor: float
    annual_downtime: float
    annual_expected_failure: float
    disposal: float
    salvage: float


# ============================================================
# VALIDATION
# ============================================================

def validate_nonnegative(**values: float) -> None:
    for name, value in values.items():
        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")

        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")

        if value < 0:
            raise ValueError(f"{name} cannot be negative")


def validate_probability(value: float) -> None:
    validate_nonnegative(probability=value)

    if value > 1:
        raise ValueError("Probability cannot exceed 1")


def validate_positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


# ============================================================
# COST CALCULATIONS
# ============================================================

def calculate_repair_cost(model: RepairCost) -> dict:
    validate_nonnegative(**vars(model))

    direct = (
        model.labor
        + model.parts
        + model.contractor
        + model.logistics
    )

    downtime = (
        model.downtime_hours
        * model.contribution_margin_per_hour
    )

    return {
        "direct": direct,
        "downtime": downtime,
        "total": direct + downtime,
    }


def calculate_replacement_cost(model: ReplacementCost) -> dict:
    validate_nonnegative(**vars(model))

    gross_direct = (
        model.purchase
        + model.installation
        + model.commissioning
        + model.logistics
        + model.decommissioning
        + model.disposal
    )

    net_direct = gross_direct - model.salvage

    downtime = (
        model.downtime_hours
        * model.contribution_margin_per_hour
    )

    return {
        "gross_direct": gross_direct,
        "net_direct": net_direct,
        "downtime": downtime,
        "total": net_direct + downtime,
    }


def calculate_spare_parts_cost(
    parts: list[SparePart],
) -> dict:
    inventory_value = 0.0
    annual_carrying_cost = 0.0

    for part in parts:
        validate_nonnegative(
            unit_price=part.unit_price,
            annual_carrying_rate=part.annual_carrying_rate,
        )

        if type(part.quantity) is not int or part.quantity < 0:
            raise ValueError("Quantity must be a nonnegative integer")

        value = part.unit_price * part.quantity
        inventory_value += value

        annual_carrying_cost += (
            value * part.annual_carrying_rate
        )

    return {
        "inventory_value": inventory_value,
        "annual_carrying_cost": annual_carrying_cost,
    }


def calculate_downtime_loss(
    hours: float,
    margin_per_hour: float,
) -> float:
    validate_nonnegative(
        hours=hours,
        margin_per_hour=margin_per_hour,
    )

    return hours * margin_per_hour


def calculate_production_loss(
    lost_units: float,
    contribution_margin_per_unit: float,
) -> float:
    validate_nonnegative(
        lost_units=lost_units,
        contribution_margin_per_unit=contribution_margin_per_unit,
    )

    return lost_units * contribution_margin_per_unit


def calculate_expected_failure_cost(
    annual_probability: float,
    repair_cost: float,
    downtime_loss: float,
    emergency_cost: float = 0,
) -> float:
    validate_probability(annual_probability)

    validate_nonnegative(
        repair_cost=repair_cost,
        downtime_loss=downtime_loss,
        emergency_cost=emergency_cost,
    )

    consequence = (
        repair_cost
        + downtime_loss
        + emergency_cost
    )

    return annual_probability * consequence


def calculate_tco(
    model: OwnershipCost,
    years: int,
) -> float:
    validate_positive_integer(years, "years")
    validate_nonnegative(**vars(model))

    annual_cost = (
        model.annual_maintenance
        + model.annual_energy
        + model.annual_labor
        + model.annual_downtime
        + model.annual_expected_failure
    )

    return (
        model.acquisition
        + annual_cost * years
        + model.disposal
        - model.salvage
    )


def calculate_npv(
    initial_investment: float,
    annual_cashflows: list[float],
    discount_rate: float,
) -> float:
    validate_nonnegative(
        initial_investment=initial_investment,
        discount_rate=discount_rate,
    )

    if discount_rate <= -1:
        raise ValueError("Invalid discount rate")

    if not all(math.isfinite(x) for x in annual_cashflows):
        raise ValueError("Cashflows must be finite")

    return -initial_investment + sum(
        cashflow / ((1 + discount_rate) ** year)
        for year, cashflow in enumerate(
            annual_cashflows,
            start=1,
        )
    )


def calculate_roi(
    total_benefits: float,
    total_investment: float,
) -> float:
    validate_nonnegative(
        total_benefits=total_benefits,
        total_investment=total_investment,
    )

    if total_investment == 0:
        raise ValueError("Investment cannot be zero")

    return (
        (total_benefits - total_investment)
        / total_investment
        * 100
    )


def calculate_payback_period(
    investment: float,
    annual_cashflows: list[float],
) -> float | None:
    validate_nonnegative(investment=investment)

    if investment == 0:
        return 0.0

    cumulative = 0.0

    for year, cashflow in enumerate(annual_cashflows, start=1):
        if not math.isfinite(cashflow):
            raise ValueError("Cashflow must be finite")

        previous = cumulative
        cumulative += cashflow

        if cumulative >= investment:
            if cashflow <= 0:
                return float(year)

            fraction = (
                investment - previous
            ) / cashflow

            return (year - 1) + fraction

    return None


# ============================================================
# PYTEST FIXTURES
# ============================================================

@pytest.fixture
def repair_sample() -> RepairCost:
    return RepairCost(
        labor=65000,
        parts=145000,
        contractor=38000,
        logistics=12000,
        downtime_hours=36,
        contribution_margin_per_hour=18000,
    )


@pytest.fixture
def replacement_sample() -> ReplacementCost:
    return ReplacementCost(
        purchase=680000,
        installation=95000,
        commissioning=28000,
        logistics=22000,
        decommissioning=18000,
        disposal=8000,
        salvage=45000,
        downtime_hours=58,
        contribution_margin_per_hour=18000,
    )


@pytest.fixture
def ownership_sample() -> OwnershipCost:
    return OwnershipCost(
        acquisition=520000,
        annual_maintenance=222000,
        annual_energy=936000,
        annual_labor=120000,
        annual_downtime=264000,
        annual_expected_failure=56000,
        disposal=25000,
        salvage=45000,
    )


# ============================================================
# 1. REPAIR COST TESTS
# ============================================================

def test_repair_direct_cost(repair_sample):
    result = calculate_repair_cost(repair_sample)
    assert result["direct"] == 260000


def test_repair_downtime_cost(repair_sample):
    result = calculate_repair_cost(repair_sample)
    assert result["downtime"] == 648000


def test_repair_total_cost(repair_sample):
    result = calculate_repair_cost(repair_sample)
    assert result["total"] == 908000


def test_repair_zero_cost():
    model = RepairCost(0, 0, 0, 0, 0, 0)
    assert calculate_repair_cost(model)["total"] == 0


@pytest.mark.parametrize("invalid", [-1, -100, -5000])
def test_repair_negative_labor_rejected(invalid):
    model = RepairCost(invalid, 100, 50, 20, 2, 100)

    with pytest.raises(ValueError):
        calculate_repair_cost(model)


# ============================================================
# 2. REPLACEMENT COST TESTS
# ============================================================

def test_replacement_gross_direct_cost(replacement_sample):
    result = calculate_replacement_cost(replacement_sample)
    assert result["gross_direct"] == 851000


def test_replacement_net_direct_cost(replacement_sample):
    result = calculate_replacement_cost(replacement_sample)
    assert result["net_direct"] == 806000


def test_replacement_downtime_cost(replacement_sample):
    result = calculate_replacement_cost(replacement_sample)
    assert result["downtime"] == 1044000


def test_replacement_total_cost(replacement_sample):
    result = calculate_replacement_cost(replacement_sample)
    assert result["total"] == 1850000


def test_replacement_salvage_reduces_cost():
    baseline = ReplacementCost(
        100000, 10000, 5000, 2000,
        3000, 1000, 0, 10, 1000
    )

    salvaged = ReplacementCost(
        100000, 10000, 5000, 2000,
        3000, 1000, 20000, 10, 1000
    )

    difference = (
        calculate_replacement_cost(baseline)["total"]
        - calculate_replacement_cost(salvaged)["total"]
    )

    assert difference == 20000


def test_replacement_negative_purchase_rejected():
    model = ReplacementCost(
        -1, 100, 100, 100, 100, 100, 0, 1, 100
    )

    with pytest.raises(ValueError):
        calculate_replacement_cost(model)


# ============================================================
# 3. SPARE PARTS COST TESTS
# ============================================================

def test_spare_parts_inventory_value():
    parts = [
        SparePart(8500, 18),
        SparePart(4200, 35),
    ]

    result = calculate_spare_parts_cost(parts)

    assert result["inventory_value"] == 300000


def test_spare_parts_carrying_cost():
    parts = [
        SparePart(10000, 10, 0.20),
    ]

    result = calculate_spare_parts_cost(parts)

    assert result["annual_carrying_cost"] == 20000


def test_spare_parts_empty_inventory():
    result = calculate_spare_parts_cost([])

    assert result["inventory_value"] == 0
    assert result["annual_carrying_cost"] == 0


def test_spare_parts_zero_quantity():
    result = calculate_spare_parts_cost([
        SparePart(25000, 0)
    ])

    assert result["inventory_value"] == 0


def test_spare_parts_negative_quantity_rejected():
    with pytest.raises(ValueError):
        calculate_spare_parts_cost([
            SparePart(1000, -3)
        ])


# ============================================================
# 4. DOWNTIME COST TESTS
# ============================================================

@pytest.mark.parametrize(
    "hours,margin,expected",
    [
        (0, 1000, 0),
        (10, 1000, 10000),
        (24, 15000, 360000),
        (36, 18000, 648000),
        (48, 24000, 1152000),
    ],
)
def test_downtime_loss(hours, margin, expected):
    assert calculate_downtime_loss(hours, margin) == expected


def test_negative_downtime_rejected():
    with pytest.raises(ValueError):
        calculate_downtime_loss(-5, 10000)


# ============================================================
# 5. PRODUCTION LOSS TESTS
# ============================================================

def test_production_loss_calculation():
    assert calculate_production_loss(1200, 180) == 216000


def test_production_loss_zero_units():
    assert calculate_production_loss(0, 180) == 0


def test_production_loss_negative_units_rejected():
    with pytest.raises(ValueError):
        calculate_production_loss(-100, 180)


# ============================================================
# 6. EXPECTED FAILURE COST TESTS
# ============================================================

def test_expected_failure_cost():
    result = calculate_expected_failure_cost(
        annual_probability=0.25,
        repair_cost=200000,
        downtime_loss=300000,
        emergency_cost=50000,
    )

    assert result == 137500


def test_zero_failure_probability():
    assert calculate_expected_failure_cost(
        0, 100000, 200000
    ) == 0


def test_certain_failure_probability():
    assert calculate_expected_failure_cost(
        1, 100000, 200000
    ) == 300000


@pytest.mark.parametrize("probability", [-0.1, 1.1, 2.0])
def test_invalid_failure_probability(probability):
    with pytest.raises(ValueError):
        calculate_expected_failure_cost(
            probability, 100000, 200000
        )


# ============================================================
# 7. TOTAL COST OF OWNERSHIP TESTS
# ============================================================

def test_tco_five_years(ownership_sample):
    result = calculate_tco(ownership_sample, 5)

    annual = 222000 + 936000 + 120000 + 264000 + 56000
    expected = 520000 + annual * 5 + 25000 - 45000

    assert result == expected


def test_tco_one_year(ownership_sample):
    result = calculate_tco(ownership_sample, 1)

    assert result > ownership_sample.acquisition


def test_tco_salvage_reduces_total(ownership_sample):
    baseline = OwnershipCost(
        **{
            **vars(ownership_sample),
            "salvage": 0,
        }
    )

    difference = (
        calculate_tco(baseline, 5)
        - calculate_tco(ownership_sample, 5)
    )

    assert difference == 45000


@pytest.mark.parametrize("years", [0, -1, -5, 1.5])
def test_invalid_tco_horizon(ownership_sample, years):
    with pytest.raises(ValueError):
        calculate_tco(ownership_sample, years)


# ============================================================
# 8. NET PRESENT VALUE TESTS
# ============================================================

def test_npv_positive_project():
    result = calculate_npv(
        initial_investment=100000,
        annual_cashflows=[50000, 50000, 50000],
        discount_rate=0.10,
    )

    assert result > 0


def test_npv_negative_project():
    result = calculate_npv(
        initial_investment=500000,
        annual_cashflows=[20000, 20000, 20000],
        discount_rate=0.10,
    )

    assert result < 0


def test_npv_zero_discount_rate():
    result = calculate_npv(
        100000,
        [40000, 40000, 40000],
        0,
    )

    assert result == 20000


def test_npv_discounting():
    result = calculate_npv(
        100000,
        [110000],
        0.10,
    )

    assert result == pytest.approx(0)


def test_npv_invalid_discount_rate():
    with pytest.raises(ValueError):
        calculate_npv(
            100000,
            [20000, 30000],
            -1.5,
        )


# ============================================================
# 9. ROI TESTS
# ============================================================

def test_roi_positive():
    assert calculate_roi(150000, 100000) == 50


def test_roi_negative():
    assert calculate_roi(80000, 100000) == -20


def test_roi_break_even():
    assert calculate_roi(100000, 100000) == 0


def test_roi_zero_investment_rejected():
    with pytest.raises(ValueError):
        calculate_roi(100000, 0)


# ============================================================
# 10. PAYBACK PERIOD TESTS
# ============================================================

def test_payback_two_years():
    result = calculate_payback_period(
        100000,
        [50000, 50000, 50000],
    )

    assert result == pytest.approx(2.0)


def test_payback_fractional_year():
    result = calculate_payback_period(
        100000,
        [40000, 40000, 40000],
    )

    assert result == pytest.approx(2.5)


def test_payback_not_reached():
    result = calculate_payback_period(
        500000,
        [20000, 30000, 40000],
    )

    assert result is None


def test_payback_zero_investment():
    assert calculate_payback_period(0, [1000]) == 0


# ============================================================
# 11. SCENARIO ANALYSIS TESTS
# ============================================================

def test_repair_labor_increase(repair_sample):
    baseline = calculate_repair_cost(repair_sample)["total"]

    increased = RepairCost(
        labor=repair_sample.labor * 1.25,
        parts=repair_sample.parts,
        contractor=repair_sample.contractor,
        logistics=repair_sample.logistics,
        downtime_hours=repair_sample.downtime_hours,
        contribution_margin_per_hour=(
            repair_sample.contribution_margin_per_hour
        ),
    )

    new_total = calculate_repair_cost(increased)["total"]

    assert new_total > baseline
    assert new_total - baseline == pytest.approx(16250)


def test_downtime_reduction_scenario():
    baseline = calculate_downtime_loss(40, 15000)
    improved = calculate_downtime_loss(40 * 0.70, 15000)

    assert baseline == 600000
    assert improved == pytest.approx(420000)
    assert baseline - improved == pytest.approx(180000)


def test_failure_probability_reduction():
    baseline = calculate_expected_failure_cost(
        0.40, 200000, 300000
    )

    improved = calculate_expected_failure_cost(
        0.20, 200000, 300000
    )

    assert baseline == 200000
    assert improved == 100000


def test_energy_cost_increase():
    baseline = 100000 * 12
    increased = 100000 * 1.15 * 12

    assert increased > baseline
    assert increased == pytest.approx(1380000)


# ============================================================
# 12. FINANCIAL INVARIANTS
# ============================================================

def test_repair_cost_components_sum(repair_sample):
    result = calculate_repair_cost(repair_sample)

    assert result["total"] == (
        result["direct"] + result["downtime"]
    )


def test_replacement_cost_components_sum(replacement_sample):
    result = calculate_replacement_cost(replacement_sample)

    assert result["total"] == (
        result["net_direct"] + result["downtime"]
    )


def test_expected_cost_monotonic_in_probability():
    low = calculate_expected_failure_cost(
        0.10, 100000, 200000
    )

    high = calculate_expected_failure_cost(
        0.40, 100000, 200000
    )

    assert high > low


def test_tco_increases_with_horizon(ownership_sample):
    one_year = calculate_tco(ownership_sample, 1)
    five_year = calculate_tco(ownership_sample, 5)

    assert five_year > one_year


def test_discounted_npv_decreases_with_higher_rate():
    low_discount = calculate_npv(
        100000, [50000] * 5, 0.05
    )

    high_discount = calculate_npv(
        100000, [50000] * 5, 0.20
    )

    assert high_discount < low_discount


# ============================================================
# 13. NONFINITE INPUT VALIDATION
# ============================================================

@pytest.mark.parametrize("invalid", [
    float("nan"),
    float("inf"),
    float("-inf"),
])
def test_nonfinite_repair_cost_rejected(invalid):
    model = RepairCost(
        invalid, 100, 100, 100, 1, 100
    )

    with pytest.raises(ValueError):
        calculate_repair_cost(model)


@pytest.mark.parametrize("invalid", [
    float("nan"),
    float("inf"),
])
def test_nonfinite_probability_rejected(invalid):
    with pytest.raises(ValueError):
        calculate_expected_failure_cost(
            invalid, 1000, 2000
        )


def test_nonfinite_cashflow_rejected():
    with pytest.raises(ValueError):
        calculate_npv(
            100000,
            [10000, float("nan")],
            0.10,
        )


# ============================================================
# END OF TEST SUITE
# ============================================================
