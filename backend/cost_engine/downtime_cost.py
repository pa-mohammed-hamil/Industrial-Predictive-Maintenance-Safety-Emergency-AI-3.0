# app/agents/tools/downtime_cost.py

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence
from uuid import UUID, uuid4


# =========================================================
# Utilities
# =========================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid4())


def finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    return result if math.isfinite(result) else default


def non_negative(value: Any, default: float = 0.0) -> float:
    return max(0.0, finite(value, default))


def clamp(
    value: float,
    low: float = 0.0,
    high: float = 1.0,
) -> float:
    return max(low, min(high, finite(value)))


def safe_divide(
    numerator: float,
    denominator: float,
    default: float = 0.0,
) -> float:
    if abs(denominator) <= 1e-12:
        return default

    return numerator / denominator


def json_safe(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, UUID):
        return str(value)

    if is_dataclass(value):
        return json_safe(asdict(value))

    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


# =========================================================
# Enums
# =========================================================

class DowntimeType(str, Enum):
    PLANNED = "planned"
    UNPLANNED = "unplanned"
    FAILURE = "failure"
    MAINTENANCE = "maintenance"
    EMERGENCY = "emergency"
    SAFETY = "safety"
    CHANGEOVER = "changeover"
    QUALITY = "quality"
    POWER_OUTAGE = "power_outage"
    MATERIAL_SHORTAGE = "material_shortage"
    OTHER = "other"


class DowntimeCostCategory(str, Enum):
    DIRECT_DOWNTIME = "direct_downtime"
    PRODUCTION_LOSS = "production_loss"
    LABOR = "labor"
    IDLE_LABOR = "idle_labor"
    OVERTIME = "overtime"
    SCRAP = "scrap"
    RESTART = "restart"
    QUALITY = "quality"
    ENERGY = "energy"
    LOGISTICS = "logistics"
    CONTRACTOR = "contractor"
    PENALTY = "penalty"
    SLA = "sla"
    EXPEDITED_SHIPPING = "expedited_shipping"
    RENTAL = "rental"
    OTHER = "other"


class DowntimeSeverity(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class DowntimeCostBasis(str, Enum):
    COMPONENTIZED = "componentized"
    ALL_IN = "all_in"


class ProductionLossMethod(str, Enum):
    CONTRIBUTION_MARGIN = "contribution_margin"
    LOST_REVENUE = "lost_revenue"
    UNITS = "units"


# =========================================================
# Production profile
# =========================================================

@dataclass(slots=True)
class ProductionProfile:
    units_per_hour: float = 0.0

    selling_price_per_unit: float = 0.0

    variable_cost_per_unit: float = 0.0

    contribution_margin_per_unit: float | None = None

    revenue_per_hour: float | None = None

    contribution_margin_per_hour: float | None = None

    utilization: float = 1.0

    recovery_rate: float = 0.0

    production_criticality: float = 1.0

    def __post_init__(self) -> None:
        self.units_per_hour = non_negative(
            self.units_per_hour
        )

        self.selling_price_per_unit = non_negative(
            self.selling_price_per_unit
        )

        self.variable_cost_per_unit = non_negative(
            self.variable_cost_per_unit
        )

        if self.contribution_margin_per_unit is not None:
            self.contribution_margin_per_unit = non_negative(
                self.contribution_margin_per_unit
            )

        if self.revenue_per_hour is not None:
            self.revenue_per_hour = non_negative(
                self.revenue_per_hour
            )

        if self.contribution_margin_per_hour is not None:
            self.contribution_margin_per_hour = non_negative(
                self.contribution_margin_per_hour
            )

        self.utilization = clamp(self.utilization)
        self.recovery_rate = clamp(self.recovery_rate)
        self.production_criticality = clamp(
            self.production_criticality
        )


# =========================================================
# Labor profile
# =========================================================

@dataclass(slots=True)
class LaborProfile:
    affected_workers: int = 0

    hourly_cost_per_worker: float = 0.0

    idle_fraction: float = 1.0

    overtime_hours: float = 0.0

    overtime_hourly_cost_per_worker: float = 0.0

    contractor_cost: float = 0.0

    def __post_init__(self) -> None:
        self.affected_workers = max(
            0,
            int(self.affected_workers),
        )

        self.hourly_cost_per_worker = non_negative(
            self.hourly_cost_per_worker
        )

        self.idle_fraction = clamp(
            self.idle_fraction
        )

        self.overtime_hours = non_negative(
            self.overtime_hours
        )

        self.overtime_hourly_cost_per_worker = non_negative(
            self.overtime_hourly_cost_per_worker
        )

        self.contractor_cost = non_negative(
            self.contractor_cost
        )


# =========================================================
# Restart / quality / indirect costs
# =========================================================

@dataclass(slots=True)
class DowntimeAdditionalCosts:
    restart_cost: float = 0.0
    startup_energy_cost: float = 0.0

    scrap_cost: float = 0.0
    quality_loss: float = 0.0

    logistics_cost: float = 0.0
    expedited_shipping_cost: float = 0.0

    penalty_cost: float = 0.0
    sla_penalty: float = 0.0

    rental_cost: float = 0.0

    other_cost: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "restart_cost",
            "startup_energy_cost",
            "scrap_cost",
            "quality_loss",
            "logistics_cost",
            "expedited_shipping_cost",
            "penalty_cost",
            "sla_penalty",
            "rental_cost",
            "other_cost",
        ):
            setattr(
                self,
                name,
                non_negative(getattr(self, name)),
            )


# =========================================================
# Downtime input
# =========================================================

@dataclass(slots=True)
class DowntimeCostInput:
    downtime_hours: float

    downtime_type: DowntimeType = (
        DowntimeType.UNPLANNED
    )

    direct_downtime_cost_per_hour: float = 0.0

    production: ProductionProfile | None = None
    labor: LaborProfile | None = None

    additional_costs: DowntimeAdditionalCosts = field(
        default_factory=DowntimeAdditionalCosts
    )

    production_loss_method: ProductionLossMethod = (
        ProductionLossMethod.CONTRIBUTION_MARGIN
    )

    cost_basis: DowntimeCostBasis = (
        DowntimeCostBasis.COMPONENTIZED
    )

    all_in_cost_per_hour: float | None = None

    currency: str = "INR"

    asset_id: str | None = None
    site_id: str | None = None

    confidence: float = 1.0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.downtime_hours = non_negative(
            self.downtime_hours
        )

        self.direct_downtime_cost_per_hour = non_negative(
            self.direct_downtime_cost_per_hour
        )

        if self.all_in_cost_per_hour is not None:
            self.all_in_cost_per_hour = non_negative(
                self.all_in_cost_per_hour
            )

        self.confidence = clamp(self.confidence)


# =========================================================
# Cost breakdown
# =========================================================

@dataclass(slots=True)
class DowntimeCostBreakdown:
    direct_downtime: float = 0.0

    production_loss: float = 0.0

    idle_labor: float = 0.0
    overtime: float = 0.0
    contractor: float = 0.0

    scrap: float = 0.0
    quality: float = 0.0

    restart: float = 0.0
    energy: float = 0.0

    logistics: float = 0.0
    expedited_shipping: float = 0.0

    penalties: float = 0.0
    sla: float = 0.0

    rental: float = 0.0
    other: float = 0.0

    @property
    def total(self) -> float:
        return sum(
            (
                self.direct_downtime,
                self.production_loss,
                self.idle_labor,
                self.overtime,
                self.contractor,
                self.scrap,
                self.quality,
                self.restart,
                self.energy,
                self.logistics,
                self.expedited_shipping,
                self.penalties,
                self.sla,
                self.rental,
                self.other,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        result = json_safe(self)
        result["total"] = round(
            self.total,
            2,
        )
        return result


# =========================================================
# Result
# =========================================================

@dataclass(slots=True)
class DowntimeCostResult:
    calculation_id: str

    asset_id: str | None
    site_id: str | None

    downtime_type: DowntimeType

    downtime_hours: float

    cost_basis: DowntimeCostBasis

    production_loss_method: ProductionLossMethod

    breakdown: DowntimeCostBreakdown

    total_cost: float

    cost_per_downtime_hour: float

    lost_units: float

    recoverable_units: float

    net_lost_units: float

    production_loss_before_recovery: float

    production_recovery_credit: float

    confidence: float

    severity: DowntimeSeverity

    currency: str

    warnings: list[str]

    generated_at: datetime = field(
        default_factory=utc_now
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# =========================================================
# Downtime savings
# =========================================================

@dataclass(slots=True)
class DowntimeReductionInput:
    baseline_hours: float
    improved_hours: float

    baseline_cost_per_hour: float

    intervention_cost: float = 0.0

    probability_of_avoiding_event: float = 1.0

    currency: str = "INR"

    def __post_init__(self) -> None:
        self.baseline_hours = non_negative(
            self.baseline_hours
        )

        self.improved_hours = non_negative(
            self.improved_hours
        )

        self.baseline_cost_per_hour = non_negative(
            self.baseline_cost_per_hour
        )

        self.intervention_cost = non_negative(
            self.intervention_cost
        )

        self.probability_of_avoiding_event = clamp(
            self.probability_of_avoiding_event
        )


@dataclass(slots=True)
class DowntimeReductionResult:
    baseline_hours: float

    improved_hours: float

    hours_saved: float

    gross_savings: float

    expected_savings: float

    intervention_cost: float

    net_savings: float

    roi_percent: float | None

    currency: str


# =========================================================
# Config
# =========================================================

@dataclass(slots=True)
class DowntimeCostConfig:
    moderate_cost_ratio: float = 0.10
    high_cost_ratio: float = 0.25
    critical_cost_ratio: float = 0.50

    default_currency: str = "INR"

    warn_all_in_components: bool = True


# =========================================================
# Calculator
# =========================================================

class DowntimeCostCalculator:

    def __init__(
        self,
        *,
        config: DowntimeCostConfig | None = None,
    ) -> None:
        self.config = (
            config or DowntimeCostConfig()
        )

    # =====================================================
    # Production economics
    # =====================================================

    @staticmethod
    def contribution_margin_per_unit(
        production: ProductionProfile,
    ) -> float:

        if (
            production.contribution_margin_per_unit
            is not None
        ):
            return (
                production.contribution_margin_per_unit
            )

        return max(
            0.0,
            production.selling_price_per_unit
            - production.variable_cost_per_unit,
        )

    def contribution_margin_per_hour(
        self,
        production: ProductionProfile,
    ) -> float:

        if (
            production.contribution_margin_per_hour
            is not None
        ):
            return (
                production.contribution_margin_per_hour
            )

        return (
            production.units_per_hour
            * self.contribution_margin_per_unit(
                production
            )
        )

    def revenue_per_hour(
        self,
        production: ProductionProfile,
    ) -> float:

        if production.revenue_per_hour is not None:
            return production.revenue_per_hour

        return (
            production.units_per_hour
            * production.selling_price_per_unit
        )

    # =====================================================
    # Lost production
    # =====================================================

    @staticmethod
    def lost_units(
        *,
        downtime_hours: float,
        production: ProductionProfile,
    ) -> tuple[float, float, float]:

        gross = (
            non_negative(downtime_hours)
            * production.units_per_hour
            * production.utilization
        )

        recoverable = (
            gross
            * production.recovery_rate
        )

        net = max(
            0.0,
            gross - recoverable,
        )

        return gross, recoverable, net

    # =====================================================
    # Production loss
    # =====================================================

    def production_loss(
        self,
        *,
        downtime_hours: float,
        production: ProductionProfile,
        method: ProductionLossMethod,
    ) -> tuple[float, float, float]:

        gross_units, _, net_units = (
            self.lost_units(
                downtime_hours=downtime_hours,
                production=production,
            )
        )

        if method == (
            ProductionLossMethod.LOST_REVENUE
        ):
            rate = self.revenue_per_hour(
                production
            )

            gross_loss = (
                downtime_hours
                * production.utilization
                * rate
            )

        elif method == (
            ProductionLossMethod.UNITS
        ):
            margin = (
                self.contribution_margin_per_unit(
                    production
                )
            )

            gross_loss = (
                gross_units * margin
            )

        else:
            rate = (
                self.contribution_margin_per_hour(
                    production
                )
            )

            gross_loss = (
                downtime_hours
                * production.utilization
                * rate
            )

        recovery_credit = (
            gross_loss
            * production.recovery_rate
        )

        net_loss = max(
            0.0,
            gross_loss - recovery_credit,
        )

        return (
            gross_loss,
            recovery_credit,
            net_loss,
        )

    # =====================================================
    # Labor
    # =====================================================

    @staticmethod
    def labor_cost(
        *,
        downtime_hours: float,
        labor: LaborProfile,
    ) -> tuple[float, float, float]:

        idle_labor = (
            downtime_hours
            * labor.affected_workers
            * labor.hourly_cost_per_worker
            * labor.idle_fraction
        )

        overtime = (
            labor.overtime_hours
            * labor.affected_workers
            * labor.overtime_hourly_cost_per_worker
        )

        contractor = labor.contractor_cost

        return (
            idle_labor,
            overtime,
            contractor,
        )

    # =====================================================
    # Severity
    # =====================================================

    def classify_severity(
        self,
        *,
        total_cost: float,
        reference_asset_value: float | None = None,
    ) -> DowntimeSeverity:

        if (
            reference_asset_value is None
            or reference_asset_value <= 0
        ):
            if total_cost <= 0:
                return DowntimeSeverity.LOW

            # No asset-value baseline available.
            return DowntimeSeverity.MODERATE

        ratio = (
            total_cost
            / reference_asset_value
        )

        if (
            ratio
            >= self.config.critical_cost_ratio
        ):
            return DowntimeSeverity.CRITICAL

        if ratio >= self.config.high_cost_ratio:
            return DowntimeSeverity.HIGH

        if (
            ratio
            >= self.config.moderate_cost_ratio
        ):
            return DowntimeSeverity.MODERATE

        return DowntimeSeverity.LOW

    # =====================================================
    # Main calculation
    # =====================================================

    def calculate(
        self,
        data: DowntimeCostInput,
        *,
        reference_asset_value: float | None = None,
    ) -> DowntimeCostResult:

        warnings: list[str] = []

        lost_units = 0.0
        recoverable_units = 0.0
        net_lost_units = 0.0

        production_loss_before_recovery = 0.0
        production_recovery_credit = 0.0

        if (
            data.cost_basis
            == DowntimeCostBasis.ALL_IN
        ):
            all_in_rate = (
                data.all_in_cost_per_hour
                if data.all_in_cost_per_hour
                is not None
                else data
                .direct_downtime_cost_per_hour
            )

            total = (
                data.downtime_hours
                * all_in_rate
            )

            breakdown = DowntimeCostBreakdown(
                direct_downtime=total
            )

            if (
                self.config.warn_all_in_components
                and (
                    data.production is not None
                    or data.labor is not None
                    or data.additional_costs
                    != DowntimeAdditionalCosts()
                )
            ):
                warnings.append(
                    "ALL_IN cost basis is active. "
                    "Production, labor, restart, scrap, "
                    "quality, energy and other component "
                    "costs were not added separately to "
                    "avoid double counting."
                )

        else:
            direct_cost = (
                data.downtime_hours
                * data
                .direct_downtime_cost_per_hour
            )

            production_loss = 0.0

            if data.production is not None:
                (
                    lost_units,
                    recoverable_units,
                    net_lost_units,
                ) = self.lost_units(
                    downtime_hours=(
                        data.downtime_hours
                    ),
                    production=data.production,
                )

                (
                    production_loss_before_recovery,
                    production_recovery_credit,
                    production_loss,
                ) = self.production_loss(
                    downtime_hours=(
                        data.downtime_hours
                    ),
                    production=data.production,
                    method=(
                        data.production_loss_method
                    ),
                )

            idle_labor = 0.0
            overtime = 0.0
            contractor = 0.0

            if data.labor is not None:
                (
                    idle_labor,
                    overtime,
                    contractor,
                ) = self.labor_cost(
                    downtime_hours=(
                        data.downtime_hours
                    ),
                    labor=data.labor,
                )

            additional = data.additional_costs

            breakdown = DowntimeCostBreakdown(
                direct_downtime=direct_cost,

                production_loss=production_loss,

                idle_labor=idle_labor,
                overtime=overtime,
                contractor=contractor,

                scrap=additional.scrap_cost,
                quality=additional.quality_loss,

                restart=additional.restart_cost,
                energy=(
                    additional.startup_energy_cost
                ),

                logistics=(
                    additional.logistics_cost
                ),

                expedited_shipping=(
                    additional
                    .expedited_shipping_cost
                ),

                penalties=(
                    additional.penalty_cost
                ),

                sla=additional.sla_penalty,

                rental=additional.rental_cost,

                other=additional.other_cost,
            )

            total = breakdown.total

            if (
                data.direct_downtime_cost_per_hour > 0
                and data.production is not None
            ):
                warnings.append(
                    "Direct downtime cost and production "
                    "loss are both included. Confirm "
                    "direct_downtime_cost_per_hour does "
                    "not already contain lost production."
                )

            if (
                data.direct_downtime_cost_per_hour > 0
                and data.labor is not None
            ):
                warnings.append(
                    "Verify direct downtime rate does "
                    "not already include idle labor."
                )

        cost_per_hour = safe_divide(
            total,
            data.downtime_hours,
            default=0.0,
        )

        severity = self.classify_severity(
            total_cost=total,
            reference_asset_value=(
                reference_asset_value
            ),
        )

        if data.downtime_hours <= 0:
            warnings.append(
                "Downtime duration is zero."
            )

        if (
            data.production is None
            and data.cost_basis
            == DowntimeCostBasis.COMPONENTIZED
        ):
            warnings.append(
                "Production profile is unavailable; "
                "production-loss cost is excluded."
            )

        if data.confidence < 0.50:
            warnings.append(
                "Downtime cost estimate has low "
                "input confidence."
            )

        return DowntimeCostResult(
            calculation_id=new_id(),

            asset_id=data.asset_id,
            site_id=data.site_id,

            downtime_type=data.downtime_type,

            downtime_hours=round(
                data.downtime_hours,
                4,
            ),

            cost_basis=data.cost_basis,

            production_loss_method=(
                data.production_loss_method
            ),

            breakdown=breakdown,

            total_cost=round(total, 2),

            cost_per_downtime_hour=round(
                cost_per_hour,
                2,
            ),

            lost_units=round(
                lost_units,
                4,
            ),

            recoverable_units=round(
                recoverable_units,
                4,
            ),

            net_lost_units=round(
                net_lost_units,
                4,
            ),

            production_loss_before_recovery=round(
                production_loss_before_recovery,
                2,
            ),

            production_recovery_credit=round(
                production_recovery_credit,
                2,
            ),

            confidence=round(
                data.confidence,
                4,
            ),

            severity=severity,

            currency=(
                data.currency
                or self.config.default_currency
            ),

            warnings=warnings,

            metadata=dict(data.metadata),
        )

    # =====================================================
    # Downtime reduction savings
    # =====================================================

    @staticmethod
    def calculate_reduction(
        data: DowntimeReductionInput,
    ) -> DowntimeReductionResult:

        hours_saved = max(
            0.0,
            data.baseline_hours
            - data.improved_hours,
        )

        gross_savings = (
            hours_saved
            * data.baseline_cost_per_hour
        )

        expected_savings = (
            gross_savings
            * data.probability_of_avoiding_event
        )

        net_savings = (
            expected_savings
            - data.intervention_cost
        )

        roi = None

        if data.intervention_cost > 0:
            roi = (
                net_savings
                / data.intervention_cost
                * 100.0
            )

        return DowntimeReductionResult(
            baseline_hours=round(
                data.baseline_hours,
                4,
            ),

            improved_hours=round(
                data.improved_hours,
                4,
            ),

            hours_saved=round(
                hours_saved,
                4,
            ),

            gross_savings=round(
                gross_savings,
                2,
            ),

            expected_savings=round(
                expected_savings,
                2,
            ),

            intervention_cost=round(
                data.intervention_cost,
                2,
            ),

            net_savings=round(
                net_savings,
                2,
            ),

            roi_percent=(
                round(roi, 2)
                if roi is not None
                else None
            ),

            currency=data.currency,
        )

    # =====================================================
    # Expected downtime cost
    # =====================================================

    @staticmethod
    def expected_downtime_cost(
        *,
        downtime_probability: float,
        downtime_hours: float,
        cost_per_hour: float,
    ) -> float:

        return round(
            clamp(downtime_probability)
            * non_negative(downtime_hours)
            * non_negative(cost_per_hour),
            2,
        )

    # =====================================================
    # Annual expected downtime cost
    # =====================================================

    @staticmethod
    def annual_expected_cost(
        *,
        expected_events_per_year: float,
        average_downtime_hours: float,
        average_cost_per_hour: float,
    ) -> float:

        return round(
            non_negative(
                expected_events_per_year
            )
            * non_negative(
                average_downtime_hours
            )
            * non_negative(
                average_cost_per_hour
            ),
            2,
        )

    # =====================================================
    # MTTR savings
    # =====================================================

    @staticmethod
    def mttr_savings(
        *,
        baseline_mttr_hours: float,
        improved_mttr_hours: float,
        failures_per_year: float,
        downtime_cost_per_hour: float,
    ) -> float:

        saved_hours_per_failure = max(
            0.0,
            non_negative(
                baseline_mttr_hours
            )
            - non_negative(
                improved_mttr_hours
            ),
        )

        return round(
            saved_hours_per_failure
            * non_negative(failures_per_year)
            * non_negative(
                downtime_cost_per_hour
            ),
            2,
        )

    # =====================================================
    # Availability impact
    # =====================================================

    @staticmethod
    def availability(
        *,
        scheduled_hours: float,
        downtime_hours: float,
    ) -> float:

        scheduled = non_negative(
            scheduled_hours
        )

        if scheduled <= 0:
            return 0.0

        operating = max(
            0.0,
            scheduled
            - non_negative(downtime_hours),
        )

        return clamp(
            operating / scheduled
        )

    @classmethod
    def availability_improvement(
        cls,
        *,
        scheduled_hours: float,
        baseline_downtime_hours: float,
        improved_downtime_hours: float,
    ) -> dict[str, float]:

        baseline = cls.availability(
            scheduled_hours=scheduled_hours,
            downtime_hours=(
                baseline_downtime_hours
            ),
        )

        improved = cls.availability(
            scheduled_hours=scheduled_hours,
            downtime_hours=(
                improved_downtime_hours
            ),
        )

        return {
            "baseline_availability":
                round(baseline, 4),

            "improved_availability":
                round(improved, 4),

            "absolute_improvement":
                round(
                    max(0.0, improved - baseline),
                    4,
                ),

            "percentage_point_improvement":
                round(
                    max(
                        0.0,
                        improved - baseline,
                    )
                    * 100.0,
                    2,
                ),
        }

    # =====================================================
    # Batch calculation
    # =====================================================

    def calculate_many(
        self,
        items: Sequence[DowntimeCostInput],
    ) -> list[DowntimeCostResult]:

        return [
            self.calculate(item)
            for item in items
        ]

    # =====================================================
    # Aggregate
    # =====================================================

    @staticmethod
    def aggregate(
        results: Sequence[
            DowntimeCostResult
        ],
    ) -> dict[str, Any]:

        if not results:
            return {
                "event_count": 0,
                "total_downtime_hours": 0.0,
                "total_cost": 0.0,
                "average_cost_per_event": 0.0,
                "average_cost_per_hour": 0.0,
            }

        total_hours = sum(
            result.downtime_hours
            for result in results
        )

        total_cost = sum(
            result.total_cost
            for result in results
        )

        currencies = {
            result.currency
            for result in results
        }

        return {
            "event_count": len(results),

            "total_downtime_hours":
                round(total_hours, 4),

            "total_cost":
                round(total_cost, 2),

            "average_cost_per_event":
                round(
                    total_cost / len(results),
                    2,
                ),

            "average_cost_per_hour":
                round(
                    safe_divide(
                        total_cost,
                        total_hours,
                    ),
                    2,
                ),

            "currency": (
                next(iter(currencies))
                if len(currencies) == 1
                else "MIXED"
            ),

            "mixed_currency":
                len(currencies) > 1,
        }


# =========================================================
# Default calculator
# =========================================================

downtime_cost_calculator = (
    DowntimeCostCalculator()
)


# =========================================================
# Agent-facing helpers
# =========================================================

def calculate_downtime_cost(
    data: DowntimeCostInput,
    *,
    reference_asset_value: float | None = None,
) -> DowntimeCostResult:

    return downtime_cost_calculator.calculate(
        data,
        reference_asset_value=(
            reference_asset_value
        ),
    )


def calculate_production_loss(
    *,
    downtime_hours: float,
    production: ProductionProfile,
    method: ProductionLossMethod = (
        ProductionLossMethod
        .CONTRIBUTION_MARGIN
    ),
) -> float:

    _, _, net_loss = (
        downtime_cost_calculator
        .production_loss(
            downtime_hours=downtime_hours,
            production=production,
            method=method,
        )
    )

    return round(net_loss, 2)


def calculate_expected_downtime_cost(
    *,
    downtime_probability: float,
    downtime_hours: float,
    cost_per_hour: float,
) -> float:

    return (
        downtime_cost_calculator
        .expected_downtime_cost(
            downtime_probability=(
                downtime_probability
            ),
            downtime_hours=downtime_hours,
            cost_per_hour=cost_per_hour,
        )
    )


def calculate_mttr_savings(
    *,
    baseline_mttr_hours: float,
    improved_mttr_hours: float,
    failures_per_year: float,
    downtime_cost_per_hour: float,
) -> float:

    return (
        downtime_cost_calculator
        .mttr_savings(
            baseline_mttr_hours=(
                baseline_mttr_hours
            ),
            improved_mttr_hours=(
                improved_mttr_hours
            ),
            failures_per_year=(
                failures_per_year
            ),
            downtime_cost_per_hour=(
                downtime_cost_per_hour
            ),
        )
    )


# =========================================================
# Exports
# =========================================================

__all__ = [
    "DowntimeType",
    "DowntimeCostCategory",
    "DowntimeSeverity",
    "DowntimeCostBasis",
    "ProductionLossMethod",

    "ProductionProfile",
    "LaborProfile",
    "DowntimeAdditionalCosts",
    "DowntimeCostInput",

    "DowntimeCostBreakdown",
    "DowntimeCostResult",

    "DowntimeReductionInput",
    "DowntimeReductionResult",

    "DowntimeCostConfig",
    "DowntimeCostCalculator",

    "downtime_cost_calculator",

    "calculate_downtime_cost",
    "calculate_production_loss",
    "calculate_expected_downtime_cost",
    "calculate_mttr_savings",
]