"""
Celery tasks for industrial cost intelligence.

Responsibilities:
- Calculate downtime costs
- Calculate production losses
- Estimate repair and replacement costs
- Calculate expected failure costs
- Forecast future maintenance costs
- Calculate avoided costs and ROI
- Support asynchronous cost-analysis workloads
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from celery import Task

from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now() -> datetime:
    """Return the current UTC timestamp."""

    return datetime.now(timezone.utc)


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """Safely convert a value to float."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _positive(
    value: Any,
    default: float = 0.0,
) -> float:
    """Convert a value to a non-negative float."""

    return max(
        0.0,
        _safe_float(value, default),
    )


def _get_cost_service() -> Any:
    """
    Lazily create the cost service.

    Lazy imports prevent worker startup from unnecessarily
    initializing database/model dependencies.
    """

    from app.services.cost_service import CostService

    return CostService()


def _get_downtime_service() -> Any:
    """Return the downtime cost service."""

    from app.services.downtime_cost_service import (
        DowntimeCostService,
    )

    return DowntimeCostService()


def _get_production_loss_service() -> Any:
    """Return the production loss service."""

    from app.services.production_loss_service import (
        ProductionLossService,
    )

    return ProductionLossService()


def _get_repair_cost_service() -> Any:
    """Return the repair cost service."""

    from app.services.repair_cost_service import (
        RepairCostService,
    )

    return RepairCostService()


def _get_replacement_cost_service() -> Any:
    """Return the replacement cost service."""

    from app.services.replacement_cost_service import (
        ReplacementCostService,
    )

    return ReplacementCostService()


def _get_future_cost_service() -> Any:
    """Return the future cost service."""

    from app.services.future_cost_service import (
        FutureCostService,
    )

    return FutureCostService()


def _get_roi_service() -> Any:
    """Return the ROI service."""

    from app.services.roi_service import ROIService

    return ROIService()


# ---------------------------------------------------------------------------
# Base task
# ---------------------------------------------------------------------------

class CostTask(Task):
    """Base Celery task for cost-intelligence jobs."""

    autoretry_for = (
        ConnectionError,
        TimeoutError,
    )

    retry_backoff = True
    retry_backoff_max = 300
    retry_jitter = True
    max_retries = 3

    acks_late = True
    reject_on_worker_lost = True

    def on_failure(
        self,
        exc: Exception,
        task_id: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        einfo: Any,
    ) -> None:
        logger.error(
            "Cost task failed",
            extra={
                "task_id": task_id,
                "task_name": self.name,
                "error": str(exc),
            },
            exc_info=True,
        )

        super().on_failure(
            exc,
            task_id,
            args,
            kwargs,
            einfo,
        )


# ---------------------------------------------------------------------------
# Downtime cost
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    base=CostTask,
    name="app.workers.cost_tasks.calculate_downtime_cost",
)
def calculate_downtime_cost(
    self: CostTask,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """
    Calculate financial impact caused by equipment downtime.

    Expected payload:

        {
            "asset_id": "...",
            "downtime_hours": 4,
            "hourly_cost": 2500
        }
    """

    asset_id = payload.get("asset_id")

    downtime_hours = _positive(
        payload.get("downtime_hours")
    )

    hourly_cost = _positive(
        payload.get("hourly_cost")
    )

    downtime_cost = (
        downtime_hours * hourly_cost
    )

    logger.info(
        "Downtime cost calculated",
        extra={
            "asset_id": asset_id,
            "downtime_hours": downtime_hours,
            "downtime_cost": downtime_cost,
        },
    )

    return {
        "status": "completed",
        "task": "calculate_downtime_cost",
        "asset_id": asset_id,
        "downtime_hours": downtime_hours,
        "hourly_cost": hourly_cost,
        "downtime_cost": downtime_cost,
        "calculated_at": _utc_now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Production loss
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    base=CostTask,
    name="app.workers.cost_tasks.calculate_production_loss",
)
def calculate_production_loss(
    self: CostTask,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """
    Calculate production loss caused by downtime.

    Expected payload:

        {
            "asset_id": "...",
            "downtime_hours": 4,
            "units_per_hour": 120,
            "value_per_unit": 15
        }
    """

    asset_id = payload.get("asset_id")

    downtime_hours = _positive(
        payload.get("downtime_hours")
    )

    units_per_hour = _positive(
        payload.get("units_per_hour")
    )

    value_per_unit = _positive(
        payload.get("value_per_unit")
    )

    lost_units = (
        downtime_hours * units_per_hour
    )

    production_loss = (
        lost_units * value_per_unit
    )

    return {
        "status": "completed",
        "task": "calculate_production_loss",
        "asset_id": asset_id,
        "downtime_hours": downtime_hours,
        "lost_units": lost_units,
        "value_per_unit": value_per_unit,
        "production_loss": production_loss,
        "calculated_at": _utc_now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Repair cost
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    base=CostTask,
    name="app.workers.cost_tasks.calculate_repair_cost",
)
def calculate_repair_cost(
    self: CostTask,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """
    Calculate estimated repair cost.

    The worker supports a direct calculation fallback while allowing
    the production RepairCostService to be integrated later.
    """

    asset_id = payload.get("asset_id")

    parts_cost = _positive(
        payload.get("parts_cost")
    )

    labor_cost = _positive(
        payload.get("labor_cost")
    )

    emergency_cost = _positive(
        payload.get("emergency_cost")
    )

    other_cost = _positive(
        payload.get("other_cost")
    )

    total_cost = (
        parts_cost
        + labor_cost
        + emergency_cost
        + other_cost
    )

    return {
        "status": "completed",
        "task": "calculate_repair_cost",
        "asset_id": asset_id,
        "parts_cost": parts_cost,
        "labor_cost": labor_cost,
        "emergency_cost": emergency_cost,
        "other_cost": other_cost,
        "repair_cost": total_cost,
        "calculated_at": _utc_now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Replacement cost
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    base=CostTask,
    name="app.workers.cost_tasks.calculate_replacement_cost",
)
def calculate_replacement_cost(
    self: CostTask,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Calculate total asset replacement cost."""

    asset_id = payload.get("asset_id")

    asset_price = _positive(
        payload.get("asset_price")
    )

    installation_cost = _positive(
        payload.get("installation_cost")
    )

    commissioning_cost = _positive(
        payload.get("commissioning_cost")
    )

    disposal_cost = _positive(
        payload.get("disposal_cost")
    )

    downtime_cost = _positive(
        payload.get("downtime_cost")
    )

    total_cost = (
        asset_price
        + installation_cost
        + commissioning_cost
        + disposal_cost
        + downtime_cost
    )

    return {
        "status": "completed",
        "task": "calculate_replacement_cost",
        "asset_id": asset_id,
        "asset_price": asset_price,
        "installation_cost": installation_cost,
        "commissioning_cost": commissioning_cost,
        "disposal_cost": disposal_cost,
        "downtime_cost": downtime_cost,
        "replacement_cost": total_cost,
        "calculated_at": _utc_now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Expected failure cost
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    base=CostTask,
    name="app.workers.cost_tasks.calculate_expected_failure_cost",
)
def calculate_expected_failure_cost(
    self: CostTask,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """
    Calculate expected failure cost.

    Formula:

        expected_cost =
            failure_probability * failure_impact
    """

    asset_id = payload.get("asset_id")

    failure_probability = min(
        1.0,
        max(
            0.0,
            _safe_float(
                payload.get(
                    "failure_probability"
                )
            ),
        ),
    )

    failure_impact = _positive(
        payload.get("failure_impact")
    )

    expected_cost = (
        failure_probability
        * failure_impact
    )

    return {
        "status": "completed",
        "task": "calculate_expected_failure_cost",
        "asset_id": asset_id,
        "failure_probability": failure_probability,
        "failure_impact": failure_impact,
        "expected_failure_cost": expected_cost,
        "calculated_at": _utc_now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Future cost forecast
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    base=CostTask,
    name="app.workers.cost_tasks.forecast_future_cost",
)
def forecast_future_cost(
    self: CostTask,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """
    Generate a basic future maintenance-cost forecast.

    For production ML forecasting, this task should delegate to
    app.services.future_cost_service.FutureCostService.
    """

    asset_id = payload.get("asset_id")

    monthly_cost = _positive(
        payload.get("monthly_cost")
    )

    months = max(
        1,
        int(
            _positive(
                payload.get("months", 12)
            )
        ),
    )

    growth_rate = _safe_float(
        payload.get("annual_growth_rate", 0.0)
    )

    growth_rate = max(
        -0.99,
        growth_rate,
    )

    monthly_forecast: list[dict[str, Any]] = []

    current_monthly_cost = monthly_cost

    for month in range(1, months + 1):
        if month > 1:
            current_monthly_cost *= (
                1.0
                + growth_rate / 12.0
            )

        monthly_forecast.append(
            {
                "month": month,
                "estimated_cost": round(
                    current_monthly_cost,
                    2,
                ),
            }
        )

    total_forecast = sum(
        item["estimated_cost"]
        for item in monthly_forecast
    )

    return {
        "status": "completed",
        "task": "forecast_future_cost",
        "asset_id": asset_id,
        "months": months,
        "monthly_forecast": monthly_forecast,
        "total_forecast_cost": round(
            total_forecast,
            2,
        ),
        "calculated_at": _utc_now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Repair vs replacement
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    base=CostTask,
    name="app.workers.cost_tasks.repair_vs_replacement",
)
def repair_vs_replacement(
    self: CostTask,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """
    Compare repair and replacement economics.
    """

    asset_id = payload.get("asset_id")

    repair_cost = _positive(
        payload.get("repair_cost")
    )

    replacement_cost = _positive(
        payload.get("replacement_cost")
    )

    repair_life_months = _positive(
        payload.get("repair_life_months", 0)
    )

    replacement_life_months = _positive(
        payload.get("replacement_life_months", 0)
    )

    if repair_cost < replacement_cost:
        recommendation = "repair"
    elif replacement_cost < repair_cost:
        recommendation = "replace"
    else:
        recommendation = "evaluate"

    savings = abs(
        replacement_cost - repair_cost
    )

    return {
        "status": "completed",
        "task": "repair_vs_replacement",
        "asset_id": asset_id,
        "repair_cost": repair_cost,
        "replacement_cost": replacement_cost,
        "repair_life_months": repair_life_months,
        "replacement_life_months": replacement_life_months,
        "recommended_action": recommendation,
        "cost_difference": savings,
        "calculated_at": _utc_now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Avoided cost
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    base=CostTask,
    name="app.workers.cost_tasks.calculate_avoided_cost",
)
def calculate_avoided_cost(
    self: CostTask,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """
    Calculate cost avoided through predictive maintenance.

    Formula:

        avoided_cost =
            predicted_failure_cost
            - preventive_maintenance_cost
    """

    asset_id = payload.get("asset_id")

    predicted_failure_cost = _positive(
        payload.get(
            "predicted_failure_cost"
        )
    )

    preventive_cost = _positive(
        payload.get(
            "preventive_maintenance_cost"
        )
    )

    avoided_cost = max(
        0.0,
        predicted_failure_cost
        - preventive_cost,
    )

    return {
        "status": "completed",
        "task": "calculate_avoided_cost",
        "asset_id": asset_id,
        "predicted_failure_cost": (
            predicted_failure_cost
        ),
        "preventive_maintenance_cost": (
            preventive_cost
        ),
        "avoided_cost": avoided_cost,
        "calculated_at": _utc_now().isoformat(),
    }


# ---------------------------------------------------------------------------
# ROI
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    base=CostTask,
    name="app.workers.cost_tasks.calculate_roi",
)
def calculate_roi(
    self: CostTask,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """
    Calculate predictive-maintenance ROI.

    Formula:

        ROI (%) =
            ((benefit - investment) / investment) * 100
    """

    asset_id = payload.get("asset_id")

    investment = _positive(
        payload.get("investment")
    )

    benefit = _positive(
        payload.get("benefit")
    )

    if investment <= 0:
        roi_percentage = 0.0
    else:
        roi_percentage = (
            (benefit - investment)
            / investment
            * 100.0
        )

    return {
        "status": "completed",
        "task": "calculate_roi",
        "asset_id": asset_id,
        "investment": investment,
        "benefit": benefit,
        "net_benefit": benefit - investment,
        "roi_percentage": round(
            roi_percentage,
            2,
        ),
        "calculated_at": _utc_now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Complete cost analysis
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    base=CostTask,
    name="app.workers.cost_tasks.run_cost_analysis",
)
def run_cost_analysis(
    self: CostTask,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """
    Run the major cost calculations for an asset.

    This task intentionally keeps calculations deterministic and
    composable so individual calculations can later be delegated
    to the dedicated cost-engine services.
    """

    asset_id = payload.get("asset_id")

    downtime_result = calculate_downtime_cost.run(
        payload
    )

    production_result = calculate_production_loss.run(
        payload
    )

    repair_result = calculate_repair_cost.run(
        payload
    )

    replacement_result = (
        calculate_replacement_cost.run(payload)
    )

    expected_failure_result = (
        calculate_expected_failure_cost.run(
            payload
        )
    )

    return {
        "status": "completed",
        "task": "run_cost_analysis",
        "asset_id": asset_id,
        "downtime": downtime_result,
        "production_loss": production_result,
        "repair": repair_result,
        "replacement": replacement_result,
        "expected_failure": expected_failure_result,
        "completed_at": _utc_now().isoformat(),
    }


__all__ = [
    "calculate_downtime_cost",
    "calculate_production_loss",
    "calculate_repair_cost",
    "calculate_replacement_cost",
    "calculate_expected_failure_cost",
    "forecast_future_cost",
    "repair_vs_replacement",
    "calculate_avoided_cost",
    "calculate_roi",
    "run_cost_analysis",
]