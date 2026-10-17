# app/cost_engine/avoided_cost.py

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# =========================================================
# Enums
# =========================================================

class AvoidedCostType(str, Enum):
    FAILURE = "failure"
    DOWNTIME = "downtime"
    PRODUCTION_LOSS = "production_loss"
    REPAIR = "repair"
    REPLACEMENT = "replacement"
    LABOR = "labor"
    SPARE_PARTS = "spare_parts"
    SCRAP = "scrap"
    ENERGY = "energy"
    SAFETY = "safety"
    EMERGENCY = "emergency"
    QUALITY = "quality"
    TOTAL = "total"


class InterventionType(str, Enum):
    PREVENTIVE_MAINTENANCE = "preventive_maintenance"
    PREDICTIVE_MAINTENANCE = "predictive_maintenance"
    REPAIR = "repair"
    COMPONENT_REPLACEMENT = "component_replacement"
    INSPECTION = "inspection"
    SAFETY_ACTION = "safety_action"
    EMERGENCY_ACTION = "emergency_action"
    PROCESS_OPTIMIZATION = "process_optimization"
    ENERGY_OPTIMIZATION = "energy_optimization"


# =========================================================
# Models
# =========================================================

@dataclass(slots=True)
class CostBreakdown:
    repair_cost: float = 0.0
    replacement_cost: float = 0.0
    labor_cost: float = 0.0
    spare_parts_cost: float = 0.0

    downtime_cost: float = 0.0
    production_loss: float = 0.0

    scrap_cost: float = 0.0
    quality_loss: float = 0.0

    energy_loss: float = 0.0

    safety_cost: float = 0.0
    emergency_cost: float = 0.0

    other_cost: float = 0.0

    @property
    def total(self) -> float:
        return round(
            max(0.0, self.repair_cost)
            + max(0.0, self.replacement_cost)
            + max(0.0, self.labor_cost)
            + max(0.0, self.spare_parts_cost)
            + max(0.0, self.downtime_cost)
            + max(0.0, self.production_loss)
            + max(0.0, self.scrap_cost)
            + max(0.0, self.quality_loss)
            + max(0.0, self.energy_loss)
            + max(0.0, self.safety_cost)
            + max(0.0, self.emergency_cost)
            + max(0.0, self.other_cost),
            2,
        )


@dataclass(slots=True)
class AvoidedCostInput:
    asset_id: str

    failure_probability_before: float
    failure_probability_after: float

    failure_cost: CostBreakdown

    intervention_cost: float = 0.0

    intervention_type: InterventionType = (
        InterventionType.PREDICTIVE_MAINTENANCE
    )

    site_id: str | None = None

    prediction_id: str | None = None
    work_order_id: str | None = None

    currency: str = "INR"

    horizon_hours: float | None = None


@dataclass(slots=True)
class AvoidedCostResult:
    asset_id: str

    site_id: str | None

    intervention_type: str

    failure_probability_before: float
    failure_probability_after: float

    probability_reduction: float
    relative_risk_reduction_percent: float

    failure_consequence_cost: float

    expected_loss_before: float
    expected_loss_after: float

    gross_avoided_cost: float

    intervention_cost: float

    net_avoided_cost: float

    roi_percent: float | None
    benefit_cost_ratio: float | None

    currency: str

    horizon_hours: float | None

    breakdown: dict[str, float]

    calculated_at: datetime

    prediction_id: str | None = None
    work_order_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)

        data["calculated_at"] = (
            self.calculated_at.isoformat()
        )

        return data


# =========================================================
# Validation
# =========================================================

def _validate_probability(
    value: float,
    name: str,
) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{name} must be between 0 and 1."
        )


def _safe_cost(
    value: float,
) -> float:
    return max(0.0, float(value))


# =========================================================
# Expected loss
# =========================================================

def calculate_expected_loss(
    failure_probability: float,
    consequence_cost: float,
) -> float:
    """
    Expected Loss =
        Probability of Failure × Consequence Cost
    """

    _validate_probability(
        failure_probability,
        "failure_probability",
    )

    return round(
        failure_probability
        * _safe_cost(consequence_cost),
        2,
    )


# =========================================================
# Avoided cost
# =========================================================

def calculate_avoided_cost(
    data: AvoidedCostInput,
) -> AvoidedCostResult:
    """
    Calculate the economic value created by an intervention.

    Example:

        failure probability before = 0.80
        failure probability after  = 0.20
        consequence cost           = ₹1,000,000

        expected loss before = ₹800,000
        expected loss after  = ₹200,000

        gross avoided cost = ₹600,000

        if intervention cost = ₹100,000:

        net avoided cost = ₹500,000
    """

    _validate_probability(
        data.failure_probability_before,
        "failure_probability_before",
    )

    _validate_probability(
        data.failure_probability_after,
        "failure_probability_after",
    )

    consequence_cost = (
        data.failure_cost.total
    )

    probability_reduction = max(
        0.0,
        data.failure_probability_before
        - data.failure_probability_after,
    )

    if data.failure_probability_before > 0:
        relative_reduction = (
            probability_reduction
            / data.failure_probability_before
        ) * 100
    else:
        relative_reduction = 0.0

    expected_before = (
        calculate_expected_loss(
            data.failure_probability_before,
            consequence_cost,
        )
    )

    expected_after = (
        calculate_expected_loss(
            data.failure_probability_after,
            consequence_cost,
        )
    )

    gross_avoided = max(
        0.0,
        expected_before - expected_after,
    )

    intervention_cost = _safe_cost(
        data.intervention_cost
    )

    net_avoided = (
        gross_avoided
        - intervention_cost
    )

    if intervention_cost > 0:
        roi = (
            net_avoided
            / intervention_cost
        ) * 100

        benefit_cost_ratio = (
            gross_avoided
            / intervention_cost
        )

    else:
        roi = None
        benefit_cost_ratio = None

    breakdown = (
        calculate_avoided_cost_breakdown(
            failure_probability_before=(
                data.failure_probability_before
            ),
            failure_probability_after=(
                data.failure_probability_after
            ),
            costs=data.failure_cost,
        )
    )

    return AvoidedCostResult(
        asset_id=data.asset_id,
        site_id=data.site_id,

        intervention_type=(
            data.intervention_type.value
        ),

        failure_probability_before=(
            data.failure_probability_before
        ),

        failure_probability_after=(
            data.failure_probability_after
        ),

        probability_reduction=round(
            probability_reduction,
            4,
        ),

        relative_risk_reduction_percent=round(
            relative_reduction,
            2,
        ),

        failure_consequence_cost=(
            consequence_cost
        ),

        expected_loss_before=(
            expected_before
        ),

        expected_loss_after=(
            expected_after
        ),

        gross_avoided_cost=round(
            gross_avoided,
            2,
        ),

        intervention_cost=round(
            intervention_cost,
            2,
        ),

        net_avoided_cost=round(
            net_avoided,
            2,
        ),

        roi_percent=(
            round(roi, 2)
            if roi is not None
            else None
        ),

        benefit_cost_ratio=(
            round(
                benefit_cost_ratio,
                3,
            )
            if benefit_cost_ratio
            is not None
            else None
        ),

        currency=data.currency,

        horizon_hours=data.horizon_hours,

        breakdown=breakdown,

        calculated_at=datetime.now(
            timezone.utc
        ),

        prediction_id=data.prediction_id,
        work_order_id=data.work_order_id,
    )


# =========================================================
# Breakdown
# =========================================================

def calculate_avoided_cost_breakdown(
    *,
    failure_probability_before: float,
    failure_probability_after: float,
    costs: CostBreakdown,
) -> dict[str, float]:

    _validate_probability(
        failure_probability_before,
        "failure_probability_before",
    )

    _validate_probability(
        failure_probability_after,
        "failure_probability_after",
    )

    probability_delta = max(
        0.0,
        failure_probability_before
        - failure_probability_after,
    )

    fields = {
        AvoidedCostType.REPAIR.value:
            costs.repair_cost,

        AvoidedCostType.REPLACEMENT.value:
            costs.replacement_cost,

        AvoidedCostType.LABOR.value:
            costs.labor_cost,

        AvoidedCostType.SPARE_PARTS.value:
            costs.spare_parts_cost,

        AvoidedCostType.DOWNTIME.value:
            costs.downtime_cost,

        AvoidedCostType.PRODUCTION_LOSS.value:
            costs.production_loss,

        AvoidedCostType.SCRAP.value:
            costs.scrap_cost,

        AvoidedCostType.QUALITY.value:
            costs.quality_loss,

        AvoidedCostType.ENERGY.value:
            costs.energy_loss,

        AvoidedCostType.SAFETY.value:
            costs.safety_cost,

        AvoidedCostType.EMERGENCY.value:
            costs.emergency_cost,

        "other":
            costs.other_cost,
    }

    result = {
        name: round(
            probability_delta
            * _safe_cost(value),
            2,
        )
        for name, value
        in fields.items()
    }

    result[
        AvoidedCostType.TOTAL.value
    ] = round(
        sum(result.values()),
        2,
    )

    return result


# =========================================================
# Downtime avoided cost
# =========================================================

def calculate_downtime_avoided_cost(
    *,
    downtime_hours_avoided: float,
    downtime_cost_per_hour: float,
) -> float:

    return round(
        _safe_cost(
            downtime_hours_avoided
        )
        * _safe_cost(
            downtime_cost_per_hour
        ),
        2,
    )


# =========================================================
# Production loss avoided
# =========================================================

def calculate_production_avoided_cost(
    *,
    production_units_per_hour: float,
    downtime_hours_avoided: float,
    profit_per_unit: float,
) -> float:

    units_saved = (
        _safe_cost(
            production_units_per_hour
        )
        * _safe_cost(
            downtime_hours_avoided
        )
    )

    return round(
        units_saved
        * _safe_cost(
            profit_per_unit
        ),
        2,
    )


# =========================================================
# Scrap avoided cost
# =========================================================

def calculate_scrap_avoided_cost(
    *,
    units_saved: float,
    cost_per_unit: float,
) -> float:

    return round(
        _safe_cost(units_saved)
        * _safe_cost(cost_per_unit),
        2,
    )


# =========================================================
# Energy avoided cost
# =========================================================

def calculate_energy_avoided_cost(
    *,
    energy_saved_kwh: float,
    electricity_cost_per_kwh: float,
) -> float:

    return round(
        _safe_cost(energy_saved_kwh)
        * _safe_cost(
            electricity_cost_per_kwh
        ),
        2,
    )


# =========================================================
# Emergency avoided cost
# =========================================================

def calculate_emergency_avoided_cost(
    *,
    emergency_probability_before: float,
    emergency_probability_after: float,
    emergency_consequence_cost: float,
) -> float:

    _validate_probability(
        emergency_probability_before,
        "emergency_probability_before",
    )

    _validate_probability(
        emergency_probability_after,
        "emergency_probability_after",
    )

    before = calculate_expected_loss(
        emergency_probability_before,
        emergency_consequence_cost,
    )

    after = calculate_expected_loss(
        emergency_probability_after,
        emergency_consequence_cost,
    )

    return round(
        max(0.0, before - after),
        2,
    )


# =========================================================
# Portfolio / fleet aggregation
# =========================================================

def aggregate_avoided_costs(
    results: list[AvoidedCostResult],
) -> dict[str, Any]:

    if not results:
        return {
            "assets": 0,
            "gross_avoided_cost": 0.0,
            "intervention_cost": 0.0,
            "net_avoided_cost": 0.0,
            "roi_percent": None,
        }

    gross = sum(
        result.gross_avoided_cost
        for result in results
    )

    intervention = sum(
        result.intervention_cost
        for result in results
    )

    net = sum(
        result.net_avoided_cost
        for result in results
    )

    roi = (
        (net / intervention) * 100
        if intervention > 0
        else None
    )

    return {
        "assets": len(
            {
                result.asset_id
                for result in results
            }
        ),
        "calculations": len(results),
        "gross_avoided_cost": round(
            gross,
            2,
        ),
        "intervention_cost": round(
            intervention,
            2,
        ),
        "net_avoided_cost": round(
            net,
            2,
        ),
        "roi_percent": (
            round(roi, 2)
            if roi is not None
            else None
        ),
    }


# =========================================================
# Service
# =========================================================

class AvoidedCostEngine:

    def calculate(
        self,
        data: AvoidedCostInput,
    ) -> AvoidedCostResult:
        return calculate_avoided_cost(
            data
        )

    def calculate_many(
        self,
        items: list[AvoidedCostInput],
    ) -> list[AvoidedCostResult]:

        return [
            self.calculate(item)
            for item in items
        ]

    def aggregate(
        self,
        results: list[
            AvoidedCostResult
        ],
    ) -> dict[str, Any]:

        return aggregate_avoided_costs(
            results
        )


avoided_cost_engine = AvoidedCostEngine()