# app/agents/tools/production_loss.py

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


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
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
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


# =========================================================
# Enums
# =========================================================

class ProductionLossType(str, Enum):
    COMPLETE_STOPPAGE = "complete_stoppage"
    PARTIAL_CAPACITY = "partial_capacity"
    SPEED_LOSS = "speed_loss"
    QUALITY_LOSS = "quality_loss"
    STARTUP_LOSS = "startup_loss"
    CHANGEOVER_LOSS = "changeover_loss"
    MICRO_STOPPAGE = "micro_stoppage"
    EQUIPMENT_FAILURE = "equipment_failure"
    MAINTENANCE = "maintenance"
    SAFETY_SHUTDOWN = "safety_shutdown"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"
    POWER_OUTAGE = "power_outage"
    MATERIAL_SHORTAGE = "material_shortage"
    OTHER = "other"


class ProductionLossMethod(str, Enum):
    CONTRIBUTION_MARGIN = "contribution_margin"
    REVENUE = "revenue"
    UNIT_COST = "unit_cost"
    FIXED_RATE = "fixed_rate"


class ProductionLossSeverity(str, Enum):
    NEGLIGIBLE = "negligible"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class ProductionLossConfidence(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


# =========================================================
# Production profile
# =========================================================

@dataclass(slots=True)
class ProductionProfile:
    rated_units_per_hour: float

    actual_units_per_hour: float | None = None

    selling_price_per_unit: float = 0.0
    variable_cost_per_unit: float = 0.0

    fixed_loss_rate_per_hour: float = 0.0

    quality_yield: float = 1.0

    normal_utilization: float = 1.0

    currency: str = "INR"

    def __post_init__(self) -> None:
        self.rated_units_per_hour = non_negative(
            self.rated_units_per_hour
        )

        if self.actual_units_per_hour is not None:
            self.actual_units_per_hour = non_negative(
                self.actual_units_per_hour
            )

        self.selling_price_per_unit = non_negative(
            self.selling_price_per_unit
        )

        self.variable_cost_per_unit = non_negative(
            self.variable_cost_per_unit
        )

        self.fixed_loss_rate_per_hour = non_negative(
            self.fixed_loss_rate_per_hour
        )

        self.quality_yield = clamp(self.quality_yield)
        self.normal_utilization = clamp(self.normal_utilization)

    @property
    def contribution_margin_per_unit(self) -> float:
        return max(
            0.0,
            self.selling_price_per_unit
            - self.variable_cost_per_unit,
        )

    @property
    def normal_output_per_hour(self) -> float:
        return (
            self.rated_units_per_hour
            * self.normal_utilization
            * self.quality_yield
        )


# =========================================================
# Loss input
# =========================================================

@dataclass(slots=True)
class ProductionLossInput:
    loss_type: ProductionLossType

    duration_hours: float

    production: ProductionProfile

    capacity_loss_fraction: float = 1.0

    recoverable_fraction: float = 0.0

    probability: float = 1.0

    recovery_overtime_cost: float = 0.0
    expedited_production_cost: float = 0.0
    subcontracting_cost: float = 0.0

    customer_penalty_cost: float = 0.0

    startup_scrap_units: float = 0.0
    scrap_cost_per_unit: float = 0.0

    additional_loss: float = 0.0

    method: ProductionLossMethod = (
        ProductionLossMethod.CONTRIBUTION_MARGIN
    )

    asset_id: str | None = None
    line_id: str | None = None
    site_id: str | None = None

    incident_id: str | None = None
    work_order_id: str | None = None

    confidence: float = 1.0

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.duration_hours = non_negative(self.duration_hours)

        self.capacity_loss_fraction = clamp(
            self.capacity_loss_fraction
        )

        self.recoverable_fraction = clamp(
            self.recoverable_fraction
        )

        self.probability = clamp(self.probability)

        for name in (
            "recovery_overtime_cost",
            "expedited_production_cost",
            "subcontracting_cost",
            "customer_penalty_cost",
            "startup_scrap_units",
            "scrap_cost_per_unit",
            "additional_loss",
        ):
            setattr(
                self,
                name,
                non_negative(getattr(self, name)),
            )

        self.confidence = clamp(self.confidence)


# =========================================================
# Breakdown
# =========================================================

@dataclass(slots=True)
class ProductionLossBreakdown:
    gross_lost_units: float = 0.0
    recoverable_units: float = 0.0
    net_lost_units: float = 0.0

    gross_margin_loss: float = 0.0
    recovery_credit: float = 0.0
    net_margin_loss: float = 0.0

    fixed_production_loss: float = 0.0

    startup_scrap_loss: float = 0.0

    recovery_overtime_cost: float = 0.0
    expedited_production_cost: float = 0.0
    subcontracting_cost: float = 0.0

    customer_penalty_cost: float = 0.0

    additional_loss: float = 0.0

    @property
    def recovery_cost(self) -> float:
        return sum(
            (
                self.recovery_overtime_cost,
                self.expedited_production_cost,
                self.subcontracting_cost,
            )
        )

    @property
    def total(self) -> float:
        return sum(
            (
                self.net_margin_loss,
                self.fixed_production_loss,
                self.startup_scrap_loss,
                self.recovery_cost,
                self.customer_penalty_cost,
                self.additional_loss,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        result = json_safe(self)
        result["recovery_cost"] = round(self.recovery_cost, 2)
        result["total"] = round(self.total, 2)
        return result


# =========================================================
# Result
# =========================================================

@dataclass(slots=True)
class ProductionLossResult:
    calculation_id: str

    asset_id: str | None
    line_id: str | None
    site_id: str | None

    incident_id: str | None
    work_order_id: str | None

    loss_type: ProductionLossType
    method: ProductionLossMethod

    duration_hours: float
    capacity_loss_fraction: float

    normal_units_per_hour: float

    gross_lost_units: float
    recovered_units: float
    net_lost_units: float

    production_loss: float
    expected_production_loss: float

    loss_per_hour: float | None

    loss_per_lost_unit: float | None

    breakdown: ProductionLossBreakdown

    severity: ProductionLossSeverity

    confidence: float
    confidence_level: ProductionLossConfidence

    currency: str

    warnings: list[str]

    generated_at: datetime = field(default_factory=utc_now)

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# =========================================================
# Savings / avoided production loss
# =========================================================

@dataclass(slots=True)
class ProductionRecoveryResult:
    baseline_loss: float
    improved_loss: float

    avoided_production_loss: float
    reduction_percent: float

    intervention_cost: float

    net_savings: float

    benefit_cost_ratio: float | None
    roi_percent: float | None

    economically_positive: bool

    currency: str

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# =========================================================
# Config
# =========================================================

@dataclass(slots=True)
class ProductionLossConfig:
    moderate_loss_threshold: float = 50_000.0
    high_loss_threshold: float = 250_000.0
    critical_loss_threshold: float = 1_000_000.0

    minimum_confidence: float = 0.50

    default_currency: str = "INR"

    prefer_contribution_margin: bool = True


# =========================================================
# Calculator
# =========================================================

class ProductionLossCalculator:

    def __init__(
        self,
        *,
        config: ProductionLossConfig | None = None,
    ) -> None:
        self.config = config or ProductionLossConfig()

    # =====================================================
    # Normal production
    # =====================================================

    @staticmethod
    def normal_units_per_hour(
        profile: ProductionProfile,
    ) -> float:
        return profile.normal_output_per_hour

    # =====================================================
    # Lost units
    # =====================================================

    def lost_units(
        self,
        *,
        loss: ProductionLossInput,
    ) -> tuple[float, float, float]:

        normal_rate = self.normal_units_per_hour(
            loss.production
        )

        gross_lost = (
            normal_rate
            * loss.duration_hours
            * loss.capacity_loss_fraction
        )

        recovered = (
            gross_lost
            * loss.recoverable_fraction
        )

        net_lost = max(
            0.0,
            gross_lost - recovered,
        )

        return (
            gross_lost,
            recovered,
            net_lost,
        )

    # =====================================================
    # Unit economic value
    # =====================================================

    @staticmethod
    def value_per_lost_unit(
        profile: ProductionProfile,
        method: ProductionLossMethod,
    ) -> float:

        if (
            method
            == ProductionLossMethod.CONTRIBUTION_MARGIN
        ):
            return (
                profile.contribution_margin_per_unit
            )

        if method == ProductionLossMethod.REVENUE:
            return profile.selling_price_per_unit

        if method == ProductionLossMethod.UNIT_COST:
            return profile.variable_cost_per_unit

        return 0.0

    # =====================================================
    # Main loss
    # =====================================================

    def calculate(
        self,
        *,
        loss: ProductionLossInput,
        metadata: Mapping[str, Any] | None = None,
    ) -> ProductionLossResult:

        warnings: list[str] = []

        gross_lost, recovered, net_lost = (
            self.lost_units(loss=loss)
        )

        profile = loss.production

        unit_value = self.value_per_lost_unit(
            profile,
            loss.method,
        )

        if (
            loss.method
            == ProductionLossMethod.FIXED_RATE
        ):
            gross_margin_loss = 0.0
            recovery_credit = 0.0
            net_margin_loss = 0.0

            fixed_loss = (
                loss.duration_hours
                * profile.fixed_loss_rate_per_hour
                * loss.capacity_loss_fraction
            )

        else:
            gross_margin_loss = (
                gross_lost * unit_value
            )

            recovery_credit = (
                recovered * unit_value
            )

            net_margin_loss = max(
                0.0,
                gross_margin_loss
                - recovery_credit,
            )

            fixed_loss = 0.0

        startup_scrap_loss = (
            loss.startup_scrap_units
            * loss.scrap_cost_per_unit
        )

        breakdown = ProductionLossBreakdown(
            gross_lost_units=gross_lost,

            recoverable_units=recovered,

            net_lost_units=net_lost,

            gross_margin_loss=gross_margin_loss,

            recovery_credit=recovery_credit,

            net_margin_loss=net_margin_loss,

            fixed_production_loss=fixed_loss,

            startup_scrap_loss=startup_scrap_loss,

            recovery_overtime_cost=(
                loss.recovery_overtime_cost
            ),

            expedited_production_cost=(
                loss.expedited_production_cost
            ),

            subcontracting_cost=(
                loss.subcontracting_cost
            ),

            customer_penalty_cost=(
                loss.customer_penalty_cost
            ),

            additional_loss=loss.additional_loss,
        )

        total_loss = breakdown.total

        expected_loss = (
            total_loss
            * loss.probability
        )

        loss_per_hour = None

        if loss.duration_hours > 0:
            loss_per_hour = (
                total_loss
                / loss.duration_hours
            )

        loss_per_unit = None

        if net_lost > 0:
            loss_per_unit = (
                total_loss / net_lost
            )

        severity = self.classify_severity(
            total_loss
        )

        confidence_level = (
            self.confidence_level(
                loss.confidence
            )
        )

        # ---------------------------------------------
        # Validation
        # ---------------------------------------------

        if loss.duration_hours <= 0:
            warnings.append(
                "Production-loss duration is zero."
            )

        if profile.rated_units_per_hour <= 0:
            warnings.append(
                "Rated production capacity is zero "
                "or unavailable."
            )

        if (
            loss.method
            == ProductionLossMethod.CONTRIBUTION_MARGIN
            and profile.contribution_margin_per_unit <= 0
        ):
            warnings.append(
                "Contribution margin per unit is zero. "
                "Check selling price and variable cost."
            )

        if (
            loss.method
            == ProductionLossMethod.REVENUE
        ):
            warnings.append(
                "Revenue method can overstate true "
                "economic production loss. "
                "Contribution margin is preferred "
                "when variable costs are avoidable."
            )

        if (
            loss.method
            == ProductionLossMethod.FIXED_RATE
            and profile.fixed_loss_rate_per_hour <= 0
        ):
            warnings.append(
                "FIXED_RATE method selected without "
                "a positive fixed_loss_rate_per_hour."
            )

        if (
            loss.recoverable_fraction > 0
            and (
                loss.recovery_overtime_cost > 0
                or loss.expedited_production_cost > 0
                or loss.subcontracting_cost > 0
            )
        ):
            warnings.append(
                "Recovered production receives an "
                "economic credit while recovery costs "
                "are added separately. Verify the "
                "recovery assumptions are consistent."
            )

        if (
            loss.confidence
            < self.config.minimum_confidence
        ):
            warnings.append(
                "Production-loss estimate has low "
                "confidence."
            )

        if (
            loss.loss_type
            in {
                ProductionLossType.SAFETY_SHUTDOWN,
                ProductionLossType.EMERGENCY_SHUTDOWN,
            }
        ):
            warnings.append(
                "Production economics must never "
                "override required safety shutdowns "
                "or emergency controls."
            )

        return ProductionLossResult(
            calculation_id=new_id(),

            asset_id=loss.asset_id,
            line_id=loss.line_id,
            site_id=loss.site_id,

            incident_id=loss.incident_id,
            work_order_id=loss.work_order_id,

            loss_type=loss.loss_type,
            method=loss.method,

            duration_hours=round(
                loss.duration_hours,
                4,
            ),

            capacity_loss_fraction=round(
                loss.capacity_loss_fraction,
                4,
            ),

            normal_units_per_hour=round(
                profile.normal_output_per_hour,
                4,
            ),

            gross_lost_units=round(
                gross_lost,
                4,
            ),

            recovered_units=round(
                recovered,
                4,
            ),

            net_lost_units=round(
                net_lost,
                4,
            ),

            production_loss=round(
                total_loss,
                2,
            ),

            expected_production_loss=round(
                expected_loss,
                2,
            ),

            loss_per_hour=(
                round(loss_per_hour, 2)
                if loss_per_hour is not None
                else None
            ),

            loss_per_lost_unit=(
                round(loss_per_unit, 4)
                if loss_per_unit is not None
                else None
            ),

            breakdown=breakdown,

            severity=severity,

            confidence=round(
                loss.confidence,
                4,
            ),

            confidence_level=confidence_level,

            currency=(
                profile.currency
                or self.config.default_currency
            ),

            warnings=warnings,

            metadata={
                **loss.metadata,
                **dict(metadata or {}),
            },
        )

    # =====================================================
    # Severity
    # =====================================================

    def classify_severity(
        self,
        total_loss: float,
    ) -> ProductionLossSeverity:

        value = non_negative(total_loss)

        if (
            value
            >= self.config.critical_loss_threshold
        ):
            return ProductionLossSeverity.CRITICAL

        if (
            value
            >= self.config.high_loss_threshold
        ):
            return ProductionLossSeverity.HIGH

        if (
            value
            >= self.config.moderate_loss_threshold
        ):
            return ProductionLossSeverity.MODERATE

        if value > 0:
            return ProductionLossSeverity.LOW

        return ProductionLossSeverity.NEGLIGIBLE

    # =====================================================
    # Confidence
    # =====================================================

    @staticmethod
    def confidence_level(
        confidence: float,
    ) -> ProductionLossConfidence:

        value = clamp(confidence)

        if value >= 0.90:
            return ProductionLossConfidence.VERY_HIGH

        if value >= 0.75:
            return ProductionLossConfidence.HIGH

        if value >= 0.50:
            return ProductionLossConfidence.MODERATE

        return ProductionLossConfidence.LOW

    # =====================================================
    # Production loss from downtime
    # =====================================================

    def downtime_production_loss(
        self,
        *,
        downtime_hours: float,
        units_per_hour: float,
        contribution_margin_per_unit: float,
        capacity_loss_fraction: float = 1.0,
        recoverable_fraction: float = 0.0,
    ) -> float:

        gross_units = (
            non_negative(downtime_hours)
            * non_negative(units_per_hour)
            * clamp(capacity_loss_fraction)
        )

        net_units = (
            gross_units
            * (
                1.0
                - clamp(recoverable_fraction)
            )
        )

        return round(
            net_units
            * non_negative(
                contribution_margin_per_unit
            ),
            2,
        )

    # =====================================================
    # Partial capacity loss
    # =====================================================

    @staticmethod
    def capacity_loss_fraction(
        *,
        normal_units_per_hour: float,
        actual_units_per_hour: float,
    ) -> float:

        normal = non_negative(
            normal_units_per_hour
        )

        actual = non_negative(
            actual_units_per_hour
        )

        if normal <= 0:
            return 0.0

        return clamp(
            (normal - actual) / normal
        )

    # =====================================================
    # Quality production loss
    # =====================================================

    @staticmethod
    def quality_loss(
        *,
        produced_units: float,
        defect_rate: float,
        contribution_margin_per_unit: float,
        recoverable_defect_fraction: float = 0.0,
        rework_cost_per_unit: float = 0.0,
    ) -> float:

        units = non_negative(
            produced_units
        )

        defects = (
            units
            * clamp(defect_rate)
        )

        recoverable = (
            defects
            * clamp(
                recoverable_defect_fraction
            )
        )

        unrecoverable = max(
            0.0,
            defects - recoverable,
        )

        margin_loss = (
            unrecoverable
            * non_negative(
                contribution_margin_per_unit
            )
        )

        rework_cost = (
            recoverable
            * non_negative(
                rework_cost_per_unit
            )
        )

        return round(
            margin_loss + rework_cost,
            2,
        )

    # =====================================================
    # Expected loss
    # =====================================================

    @staticmethod
    def expected_loss(
        *,
        production_loss: float,
        probability: float,
    ) -> float:

        return round(
            non_negative(production_loss)
            * clamp(probability),
            2,
        )

    # =====================================================
    # Avoided production loss
    # =====================================================

    @staticmethod
    def avoided_loss(
        *,
        baseline_loss: float,
        improved_loss: float,
        intervention_cost: float = 0.0,
        currency: str = "INR",
    ) -> ProductionRecoveryResult:

        baseline = non_negative(
            baseline_loss
        )

        improved = non_negative(
            improved_loss
        )

        intervention = non_negative(
            intervention_cost
        )

        avoided = max(
            0.0,
            baseline - improved,
        )

        reduction_percent = (
            safe_divide(
                avoided,
                baseline,
            )
            * 100.0
        )

        net_savings = (
            avoided - intervention
        )

        bcr = None
        roi = None

        if intervention > 0:
            bcr = avoided / intervention

            roi = (
                net_savings
                / intervention
                * 100.0
            )

        return ProductionRecoveryResult(
            baseline_loss=round(
                baseline,
                2,
            ),

            improved_loss=round(
                improved,
                2,
            ),

            avoided_production_loss=round(
                avoided,
                2,
            ),

            reduction_percent=round(
                reduction_percent,
                2,
            ),

            intervention_cost=round(
                intervention,
                2,
            ),

            net_savings=round(
                net_savings,
                2,
            ),

            benefit_cost_ratio=(
                round(bcr, 4)
                if bcr is not None
                else None
            ),

            roi_percent=(
                round(roi, 2)
                if roi is not None
                else None
            ),

            economically_positive=(
                net_savings > 0
            ),

            currency=currency,
        )

    # =====================================================
    # OEE-related loss
    # =====================================================

    @staticmethod
    def oee_production_loss(
        *,
        planned_production_hours: float,
        ideal_units_per_hour: float,
        availability: float,
        performance: float,
        quality: float,
        contribution_margin_per_unit: float,
    ) -> dict[str, float]:

        hours = non_negative(
            planned_production_hours
        )

        ideal_rate = non_negative(
            ideal_units_per_hour
        )

        availability = clamp(availability)
        performance = clamp(performance)
        quality = clamp(quality)

        ideal_units = (
            hours * ideal_rate
        )

        availability_units = (
            ideal_units * availability
        )

        performance_units = (
            availability_units * performance
        )

        good_units = (
            performance_units * quality
        )

        availability_loss_units = max(
            0.0,
            ideal_units - availability_units,
        )

        performance_loss_units = max(
            0.0,
            availability_units
            - performance_units,
        )

        quality_loss_units = max(
            0.0,
            performance_units - good_units,
        )

        margin = non_negative(
            contribution_margin_per_unit
        )

        availability_loss = (
            availability_loss_units * margin
        )

        performance_loss = (
            performance_loss_units * margin
        )

        quality_loss = (
            quality_loss_units * margin
        )

        total_loss = (
            availability_loss
            + performance_loss
            + quality_loss
        )

        return {
            "ideal_units": round(
                ideal_units,
                4,
            ),

            "good_units": round(
                good_units,
                4,
            ),

            "oee": round(
                availability
                * performance
                * quality,
                6,
            ),

            "availability_loss_units": round(
                availability_loss_units,
                4,
            ),

            "performance_loss_units": round(
                performance_loss_units,
                4,
            ),

            "quality_loss_units": round(
                quality_loss_units,
                4,
            ),

            "availability_loss": round(
                availability_loss,
                2,
            ),

            "performance_loss": round(
                performance_loss,
                2,
            ),

            "quality_loss": round(
                quality_loss,
                2,
            ),

            "total_production_loss": round(
                total_loss,
                2,
            ),
        }

    # =====================================================
    # Batch
    # =====================================================

    def calculate_many(
        self,
        losses: Sequence[
            ProductionLossInput
        ],
    ) -> list[ProductionLossResult]:

        return [
            self.calculate(loss=loss)
            for loss in losses
        ]

    # =====================================================
    # Aggregate
    # =====================================================

    @staticmethod
    def aggregate(
        results: Sequence[
            ProductionLossResult
        ],
    ) -> dict[str, Any]:

        if not results:
            return {
                "event_count": 0,
                "total_duration_hours": 0.0,
                "gross_lost_units": 0.0,
                "recovered_units": 0.0,
                "net_lost_units": 0.0,
                "production_loss": 0.0,
                "expected_production_loss": 0.0,
                "currency": None,
            }

        currencies = {
            result.currency
            for result in results
        }

        total_duration = sum(
            result.duration_hours
            for result in results
        )

        gross_units = sum(
            result.gross_lost_units
            for result in results
        )

        recovered_units = sum(
            result.recovered_units
            for result in results
        )

        net_units = sum(
            result.net_lost_units
            for result in results
        )

        total_loss = sum(
            result.production_loss
            for result in results
        )

        expected_loss = sum(
            result.expected_production_loss
            for result in results
        )

        return {
            "event_count": len(results),

            "total_duration_hours": round(
                total_duration,
                4,
            ),

            "gross_lost_units": round(
                gross_units,
                4,
            ),

            "recovered_units": round(
                recovered_units,
                4,
            ),

            "net_lost_units": round(
                net_units,
                4,
            ),

            "production_loss": round(
                total_loss,
                2,
            ),

            "expected_production_loss": round(
                expected_loss,
                2,
            ),

            "average_loss_per_event": round(
                total_loss / len(results),
                2,
            ),

            "average_loss_per_hour": (
                round(
                    total_loss
                    / total_duration,
                    2,
                )
                if total_duration > 0
                else None
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

production_loss_calculator = (
    ProductionLossCalculator()
)


# =========================================================
# Agent-facing helpers
# =========================================================

def calculate_production_loss(
    *,
    loss: ProductionLossInput,
) -> ProductionLossResult:

    return (
        production_loss_calculator
        .calculate(loss=loss)
    )


def calculate_downtime_production_loss(
    *,
    downtime_hours: float,
    units_per_hour: float,
    contribution_margin_per_unit: float,
    capacity_loss_fraction: float = 1.0,
    recoverable_fraction: float = 0.0,
) -> float:

    return (
        production_loss_calculator
        .downtime_production_loss(
            downtime_hours=downtime_hours,
            units_per_hour=units_per_hour,
            contribution_margin_per_unit=(
                contribution_margin_per_unit
            ),
            capacity_loss_fraction=(
                capacity_loss_fraction
            ),
            recoverable_fraction=(
                recoverable_fraction
            ),
        )
    )


def calculate_quality_production_loss(
    *,
    produced_units: float,
    defect_rate: float,
    contribution_margin_per_unit: float,
    recoverable_defect_fraction: float = 0.0,
    rework_cost_per_unit: float = 0.0,
) -> float:

    return (
        production_loss_calculator
        .quality_loss(
            produced_units=produced_units,
            defect_rate=defect_rate,
            contribution_margin_per_unit=(
                contribution_margin_per_unit
            ),
            recoverable_defect_fraction=(
                recoverable_defect_fraction
            ),
            rework_cost_per_unit=(
                rework_cost_per_unit
            ),
        )
    )


def calculate_avoided_production_loss(
    *,
    baseline_loss: float,
    improved_loss: float,
    intervention_cost: float = 0.0,
    currency: str = "INR",
) -> ProductionRecoveryResult:

    return (
        production_loss_calculator
        .avoided_loss(
            baseline_loss=baseline_loss,
            improved_loss=improved_loss,
            intervention_cost=(
                intervention_cost
            ),
            currency=currency,
        )
    )


# =========================================================
# Exports
# =========================================================

__all__ = [
    "ProductionLossType",
    "ProductionLossMethod",
    "ProductionLossSeverity",
    "ProductionLossConfidence",

    "ProductionProfile",
    "ProductionLossInput",
    "ProductionLossBreakdown",
    "ProductionLossResult",

    "ProductionRecoveryResult",

    "ProductionLossConfig",
    "ProductionLossCalculator",

    "production_loss_calculator",

    "calculate_production_loss",
    "calculate_downtime_production_loss",
    "calculate_quality_production_loss",
    "calculate_avoided_production_loss",
]