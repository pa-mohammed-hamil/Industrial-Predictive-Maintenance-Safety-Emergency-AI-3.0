"""
app/services/production_loss_service.py

Industrial Predictive Maintenance, Safety & Emergency AI
========================================================

Production Loss Analysis Service

Features:
    - Production loss calculation
    - Planned vs actual production analysis
    - Machine downtime production loss
    - Reduced production speed loss
    - Defective product loss
    - Rework cost calculation
    - Scrap cost calculation
    - Lost contribution margin
    - OEE availability, performance and quality analysis
    - Production efficiency analysis
    - Production loss forecasting
    - Avoided production loss estimation
    - Multi-asset production loss analysis
    - Production loss reporting
    - Optional async repository persistence
    - Optional audit integration

Financial assumptions:
    - Monetary amounts use Decimal.
    - Currency must be supplied explicitly.
    - Lost production value and lost contribution margin
      are alternative valuation approaches.
    - Lost revenue must not automatically be treated
      as profit or added to contribution margin.
    - Scrap, rework and downtime costs must not be
      counted twice across financial modules.

Expected repository methods:
    async create_production_loss(data: dict)
    async get_production_loss(loss_id: str)
    async list_production_losses(filters: dict)

Optional audit method:
    async record_event(data: dict)
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


# ============================================================
# EXCEPTIONS
# ============================================================

class ProductionLossServiceError(Exception):
    """Base production loss exception."""


class ProductionLossValidationError(ProductionLossServiceError):
    """Invalid production loss input."""


class ProductionLossNotFoundError(ProductionLossServiceError):
    """Production loss record not found."""


class ProductionLossRepositoryError(ProductionLossServiceError):
    """Repository operation failed."""


# ============================================================
# ENUMS
# ============================================================

class ProductionLossType(str, Enum):
    DOWNTIME = "downtime"
    REDUCED_SPEED = "reduced_speed"
    QUALITY_DEFECT = "quality_defect"
    REWORK = "rework"
    SCRAP = "scrap"
    MATERIAL_SHORTAGE = "material_shortage"
    EQUIPMENT_FAILURE = "equipment_failure"
    MAINTENANCE = "maintenance"
    ENERGY_INTERRUPTION = "energy_interruption"
    OTHER = "other"


class ProductionLossSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ProductionLossSource(str, Enum):
    MANUAL = "manual"
    TELEMETRY = "telemetry"
    MES = "mes"
    ERP = "erp"
    PREDICTION = "prediction"
    SIMULATION = "simulation"


class ValuationMethod(str, Enum):
    CONTRIBUTION_MARGIN = "contribution_margin"
    REVENUE = "revenue"


# ============================================================
# DECIMAL HELPERS
# ============================================================

MONEY_QUANT = Decimal("0.01")
PERCENT_QUANT = Decimal("0.01")
QUANTITY_QUANT = Decimal("0.0001")


def to_decimal(
    value: Any,
    name: str,
    minimum: Optional[Decimal] = Decimal("0"),
) -> Decimal:

    if isinstance(value, bool):
        raise ProductionLossValidationError(
            f"{name} must be numeric"
        )

    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ProductionLossValidationError(
            f"Invalid numeric value for {name}"
        ) from exc

    if not result.is_finite():
        raise ProductionLossValidationError(
            f"{name} must be finite"
        )

    if minimum is not None and result < minimum:
        raise ProductionLossValidationError(
            f"{name} must be >= {minimum}"
        )

    return result


def money(value: Any) -> Decimal:
    return to_decimal(
        value,
        "money",
    ).quantize(
        MONEY_QUANT,
        rounding=ROUND_HALF_UP,
    )


def quantity(value: Any) -> Decimal:
    return to_decimal(
        value,
        "quantity",
    ).quantize(
        QUANTITY_QUANT,
        rounding=ROUND_HALF_UP,
    )


def percentage(
    numerator: Decimal,
    denominator: Decimal,
) -> Decimal:

    if denominator <= 0:
        return Decimal("0.00")

    return (
        numerator / denominator * Decimal("100")
    ).quantize(
        PERCENT_QUANT,
        rounding=ROUND_HALF_UP,
    )


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def validate_text(
    value: Any,
    name: str,
    max_length: int = 255,
) -> str:

    if not isinstance(value, str):
        raise ProductionLossValidationError(
            f"{name} must be a string"
        )

    cleaned = value.strip()

    if not cleaned:
        raise ProductionLossValidationError(
            f"{name} cannot be empty"
        )

    if len(cleaned) > max_length:
        raise ProductionLossValidationError(
            f"{name} exceeds {max_length} characters"
        )

    return cleaned


def validate_datetime(
    value: datetime,
    name: str,
) -> datetime:

    if not isinstance(value, datetime):
        raise ProductionLossValidationError(
            f"{name} must be datetime"
        )

    if value.tzinfo is None or value.utcoffset() is None:
        raise ProductionLossValidationError(
            f"{name} must be timezone-aware"
        )

    return value


def parse_enum(
    enum_class: type[Enum],
    value: Any,
    name: str,
) -> Enum:

    try:
        return enum_class(value)
    except (ValueError, TypeError) as exc:
        raise ProductionLossValidationError(
            f"Invalid {name}: {value}"
        ) from exc


def serialize(value: Any) -> Any:

    if isinstance(value, Decimal):
        return str(value)

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if hasattr(value, "__dataclass_fields__"):
        return serialize(asdict(value))

    if isinstance(value, Mapping):
        return {
            str(key): serialize(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            serialize(item)
            for item in value
        ]

    return value


async def invoke(
    method: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:

    result = method(*args, **kwargs)

    if inspect.isawaitable(result):
        return await result

    return result


# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class ProductionMetrics:
    planned_units: Decimal
    actual_units: Decimal
    good_units: Decimal
    defective_units: Decimal
    lost_units: Decimal
    production_achievement_percent: Decimal
    quality_rate_percent: Decimal


@dataclass
class ProductionLossBreakdown:
    lost_revenue: Decimal = Decimal("0")
    lost_contribution_margin: Decimal = Decimal("0")
    scrap_cost: Decimal = Decimal("0")
    rework_cost: Decimal = Decimal("0")
    additional_operating_cost: Decimal = Decimal("0")
    selected_output_loss: Decimal = Decimal("0")
    total_economic_loss: Decimal = Decimal("0")


@dataclass
class ProductionLossResult:
    loss_id: str
    company_id: str
    site_id: str
    asset_id: str
    currency: str
    loss_type: str
    source: str
    valuation_method: str
    severity: str

    metrics: ProductionMetrics
    breakdown: ProductionLossBreakdown

    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    downtime_hours: Decimal = Decimal("0")

    notes: Optional[str] = None

    calculated_at: datetime = field(
        default_factory=utc_now
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


@dataclass
class OEEResult:
    availability_percent: Decimal
    performance_percent: Decimal
    quality_percent: Decimal
    oee_percent: Decimal
    planned_production_hours: Decimal
    operating_hours: Decimal
    total_units: Decimal
    good_units: Decimal

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


@dataclass
class ProductionLossForecast:
    asset_id: str
    forecast_days: int
    expected_downtime_hours: Decimal
    expected_lost_units: Decimal
    expected_economic_loss: Decimal
    currency: str
    assumptions: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


# ============================================================
# PRODUCTION LOSS SERVICE
# ============================================================

class ProductionLossService:
    """
    Production loss financial analysis.

    Calculations are independent of persistence.

    Repository and audit integrations are optional.
    """

    def __init__(
        self,
        production_loss_repository: Any = None,
        audit_service: Any = None,
        default_currency: str = "INR",
    ):

        self.repository = production_loss_repository
        self.audit = audit_service

        self.default_currency = self._currency(
            default_currency
        )

    # --------------------------------------------------------
    # CURRENCY VALIDATION
    # --------------------------------------------------------

    @staticmethod
    def _currency(value: str) -> str:

        value = validate_text(
            value,
            "currency",
            3,
        ).upper()

        if len(value) != 3 or not value.isalpha():
            raise ProductionLossValidationError(
                "Currency must be a 3-letter code"
            )

        return value

    # --------------------------------------------------------
    # SEVERITY CLASSIFICATION
    # --------------------------------------------------------

    @staticmethod
    def classify_loss_severity(
        loss_percentage: Any,
    ) -> ProductionLossSeverity:

        percent = to_decimal(
            loss_percentage,
            "loss_percentage",
        )

        if percent > 100:
            raise ProductionLossValidationError(
                "loss_percentage cannot exceed 100"
            )

        if percent >= 50:
            return ProductionLossSeverity.CRITICAL

        if percent >= 25:
            return ProductionLossSeverity.HIGH

        if percent >= 10:
            return ProductionLossSeverity.MEDIUM

        return ProductionLossSeverity.LOW

    # ========================================================
    # PRODUCTION METRICS
    # ========================================================

    @staticmethod
    def calculate_production_metrics(
        *,
        planned_units: Any,
        actual_units: Any,
        defective_units: Any = 0,
    ) -> ProductionMetrics:

        planned = to_decimal(
            planned_units,
            "planned_units",
        )

        actual = to_decimal(
            actual_units,
            "actual_units",
        )

        defective = to_decimal(
            defective_units,
            "defective_units",
        )

        if defective > actual:
            raise ProductionLossValidationError(
                "Defective units cannot exceed actual units"
            )

        good = actual - defective

        lost = max(
            Decimal("0"),
            planned - good,
        )

        achievement = percentage(
            good,
            planned,
        )

        quality_rate = percentage(
            good,
            actual,
        )

        return ProductionMetrics(
            planned_units=quantity(planned),
            actual_units=quantity(actual),
            good_units=quantity(good),
            defective_units=quantity(defective),
            lost_units=quantity(lost),
            production_achievement_percent=achievement,
            quality_rate_percent=quality_rate,
        )

    # ========================================================
    # DOWNTIME PRODUCTION LOSS
    # ========================================================

    @staticmethod
    def calculate_downtime_production_loss(
        *,
        downtime_hours: Any,
        production_rate_per_hour: Any,
        contribution_margin_per_unit: Any,
        selling_price_per_unit: Any = 0,
    ) -> dict[str, Decimal]:

        hours = to_decimal(
            downtime_hours,
            "downtime_hours",
        )

        rate = to_decimal(
            production_rate_per_hour,
            "production_rate_per_hour",
        )

        margin = to_decimal(
            contribution_margin_per_unit,
            "contribution_margin_per_unit",
        )

        price = to_decimal(
            selling_price_per_unit,
            "selling_price_per_unit",
        )

        lost_units = hours * rate

        return {
            "downtime_hours": hours,
            "lost_units": quantity(lost_units),
            "lost_revenue": money(
                lost_units * price
            ),
            "lost_contribution_margin": money(
                lost_units * margin
            ),
        }

    # ========================================================
    # REDUCED SPEED LOSS
    # ========================================================

    @staticmethod
    def calculate_reduced_speed_loss(
        *,
        ideal_rate_per_hour: Any,
        actual_rate_per_hour: Any,
        operating_hours: Any,
        contribution_margin_per_unit: Any,
    ) -> dict[str, Decimal]:

        ideal_rate = to_decimal(
            ideal_rate_per_hour,
            "ideal_rate_per_hour",
        )

        actual_rate = to_decimal(
            actual_rate_per_hour,
            "actual_rate_per_hour",
        )

        hours = to_decimal(
            operating_hours,
            "operating_hours",
        )

        margin = to_decimal(
            contribution_margin_per_unit,
            "contribution_margin_per_unit",
        )

        lost_rate = max(
            Decimal("0"),
            ideal_rate - actual_rate,
        )

        lost_units = lost_rate * hours

        return {
            "ideal_rate_per_hour": ideal_rate,
            "actual_rate_per_hour": actual_rate,
            "lost_units": quantity(lost_units),
            "lost_contribution_margin": money(
                lost_units * margin
            ),
        }

    # ========================================================
    # QUALITY DEFECT LOSS
    # ========================================================

    @staticmethod
    def calculate_quality_loss(
        *,
        defective_units: Any,
        scrap_units: Any,
        rework_units: Any,
        scrap_cost_per_unit: Any,
        rework_cost_per_unit: Any,
    ) -> dict[str, Decimal]:

        defective = to_decimal(
            defective_units,
            "defective_units",
        )

        scrap = to_decimal(
            scrap_units,
            "scrap_units",
        )

        rework = to_decimal(
            rework_units,
            "rework_units",
        )

        scrap_unit_cost = to_decimal(
            scrap_cost_per_unit,
            "scrap_cost_per_unit",
        )

        rework_unit_cost = to_decimal(
            rework_cost_per_unit,
            "rework_cost_per_unit",
        )

        if scrap + rework > defective:
            raise ProductionLossValidationError(
                "Scrap plus rework units cannot exceed "
                "defective units"
            )

        scrap_total = scrap * scrap_unit_cost
        rework_total = rework * rework_unit_cost

        return {
            "defective_units": quantity(defective),
            "scrap_units": quantity(scrap),
            "rework_units": quantity(rework),
            "scrap_cost": money(scrap_total),
            "rework_cost": money(rework_total),
            "total_quality_cost": money(
                scrap_total + rework_total
            ),
        }

    # ========================================================
    # OEE CALCULATION
    # ========================================================

    @staticmethod
    def calculate_oee(
        *,
        planned_production_hours: Any,
        downtime_hours: Any,
        ideal_cycle_time_hours: Any,
        total_units: Any,
        good_units: Any,
    ) -> OEEResult:

        planned = to_decimal(
            planned_production_hours,
            "planned_production_hours",
        )

        downtime = to_decimal(
            downtime_hours,
            "downtime_hours",
        )

        cycle_time = to_decimal(
            ideal_cycle_time_hours,
            "ideal_cycle_time_hours",
        )

        total = to_decimal(
            total_units,
            "total_units",
        )

        good = to_decimal(
            good_units,
            "good_units",
        )

        if downtime > planned:
            raise ProductionLossValidationError(
                "Downtime cannot exceed planned hours"
            )

        if good > total:
            raise ProductionLossValidationError(
                "Good units cannot exceed total units"
            )

        operating = planned - downtime

        if operating == 0 and total > 0:
            raise ProductionLossValidationError(
                "Production cannot occur with zero "
                "operating hours"
            )

        availability = (
            operating / planned
            if planned > 0
            else Decimal("0")
        )

        performance = (
            cycle_time * total / operating
            if operating > 0
            else Decimal("0")
        )

        if performance > 1:
            raise ProductionLossValidationError(
                "Calculated performance exceeds 100%; "
                "verify ideal cycle time and units"
            )

        quality_rate = (
            good / total
            if total > 0
            else Decimal("0")
        )

        oee = (
            availability
            * performance
            * quality_rate
        )

        return OEEResult(
            availability_percent=percentage(
                operating, planned
            ),
            performance_percent=(
                performance * 100
            ).quantize(PERCENT_QUANT),
            quality_percent=percentage(
                good, total
            ),
            oee_percent=(
                oee * 100
            ).quantize(PERCENT_QUANT),
            planned_production_hours=planned,
            operating_hours=operating,
            total_units=quantity(total),
            good_units=quantity(good),
        )

    # ========================================================
    # COMPREHENSIVE PRODUCTION LOSS ANALYSIS
    # ========================================================

    def analyze_production_loss(
        self,
        *,
        company_id: str,
        site_id: str,
        asset_id: str,
        planned_units: Any,
        actual_units: Any,
        defective_units: Any = 0,
        selling_price_per_unit: Any = 0,
        contribution_margin_per_unit: Any = 0,
        scrap_cost: Any = 0,
        rework_cost: Any = 0,
        additional_operating_cost: Any = 0,
        downtime_hours: Any = 0,
        loss_type: str | ProductionLossType = (
            ProductionLossType.OTHER
        ),
        source: str | ProductionLossSource = (
            ProductionLossSource.MANUAL
        ),
        valuation_method: str | ValuationMethod = (
            ValuationMethod.CONTRIBUTION_MARGIN
        ),
        currency: Optional[str] = None,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
        notes: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ProductionLossResult:

        kind = parse_enum(
            ProductionLossType,
            loss_type,
            "loss_type",
        )

        source_enum = parse_enum(
            ProductionLossSource,
            source,
            "source",
        )

        method = parse_enum(
            ValuationMethod,
            valuation_method,
            "valuation_method",
        )

        selected_currency = self._currency(
            currency or self.default_currency
        )

        if period_start is not None:
            validate_datetime(
                period_start,
                "period_start",
            )

        if period_end is not None:
            validate_datetime(
                period_end,
                "period_end",
            )

        if (
            period_start is not None
            and period_end is not None
            and period_end <= period_start
        ):
            raise ProductionLossValidationError(
                "period_end must be after period_start"
            )

        metrics = self.calculate_production_metrics(
            planned_units=planned_units,
            actual_units=actual_units,
            defective_units=defective_units,
        )

        price = to_decimal(
            selling_price_per_unit,
            "selling_price_per_unit",
        )

        margin = to_decimal(
            contribution_margin_per_unit,
            "contribution_margin_per_unit",
        )

        if margin > price and price > 0:
            raise ProductionLossValidationError(
                "Contribution margin per unit cannot "
                "exceed selling price per unit"
            )

        scrap = to_decimal(
            scrap_cost,
            "scrap_cost",
        )

        rework = to_decimal(
            rework_cost,
            "rework_cost",
        )

        additional = to_decimal(
            additional_operating_cost,
            "additional_operating_cost",
        )

        hours = to_decimal(
            downtime_hours,
            "downtime_hours",
        )

        lost_revenue = money(
            metrics.lost_units * price
        )

        lost_margin = money(
            metrics.lost_units * margin
        )

        selected_output_loss = (
            lost_margin
            if method == ValuationMethod.CONTRIBUTION_MARGIN
            else lost_revenue
        )

        total_loss = money(
            selected_output_loss
            + scrap
            + rework
            + additional
        )

        breakdown = ProductionLossBreakdown(
            lost_revenue=lost_revenue,
            lost_contribution_margin=lost_margin,
            scrap_cost=money(scrap),
            rework_cost=money(rework),
            additional_operating_cost=money(additional),
            selected_output_loss=selected_output_loss,
            total_economic_loss=total_loss,
        )

        if metrics.planned_units > 0:
            loss_percentage = percentage(
                metrics.lost_units,
                metrics.planned_units,
            )
            loss_percentage = min(
                loss_percentage,
                Decimal("100"),
            )
        else:
            loss_percentage = Decimal("0")

        severity = self.classify_loss_severity(
            loss_percentage
        )

        return ProductionLossResult(
            loss_id=str(uuid4()),
            company_id=validate_text(
                company_id,
                "company_id",
            ),
            site_id=validate_text(
                site_id,
                "site_id",
            ),
            asset_id=validate_text(
                asset_id,
                "asset_id",
            ),
            currency=selected_currency,
            loss_type=kind.value,
            source=source_enum.value,
            valuation_method=method.value,
            severity=severity.value,
            metrics=metrics,
            breakdown=breakdown,
            downtime_hours=hours,
            period_start=period_start,
            period_end=period_end,
            notes=notes,
            metadata=dict(metadata or {}),
        )

    # ========================================================
    # PRODUCTION LOSS FORECAST
    # ========================================================

    def forecast_production_loss(
        self,
        *,
        asset_id: str,
        forecast_days: int,
        expected_downtime_hours_per_day: Any,
        production_rate_per_hour: Any,
        contribution_margin_per_unit: Any,
        currency: Optional[str] = None,
    ) -> ProductionLossForecast:

        if (
            isinstance(forecast_days, bool)
            or not isinstance(forecast_days, int)
            or forecast_days <= 0
        ):
            raise ProductionLossValidationError(
                "forecast_days must be positive integer"
            )

        daily_downtime = to_decimal(
            expected_downtime_hours_per_day,
            "expected_downtime_hours_per_day",
        )

        rate = to_decimal(
            production_rate_per_hour,
            "production_rate_per_hour",
        )

        margin = to_decimal(
            contribution_margin_per_unit,
            "contribution_margin_per_unit",
        )

        total_hours = (
            daily_downtime * forecast_days
        )

        expected_units = total_hours * rate
        expected_loss = expected_units * margin

        return ProductionLossForecast(
            asset_id=validate_text(
                asset_id,
                "asset_id",
            ),
            forecast_days=forecast_days,
            expected_downtime_hours=total_hours,
            expected_lost_units=quantity(
                expected_units
            ),
            expected_economic_loss=money(
                expected_loss
            ),
            currency=self._currency(
                currency or self.default_currency
            ),
            assumptions={
                "daily_downtime_hours": str(
                    daily_downtime
                ),
                "production_rate_per_hour": str(
                    rate
                ),
                "contribution_margin_per_unit": str(
                    margin
                ),
                "forecast_method": (
                    "constant_rate_scenario"
                ),
            },
        )

    # ========================================================
    # AVOIDED PRODUCTION LOSS
    # ========================================================

    @staticmethod
    def calculate_avoided_production_loss(
        *,
        baseline_downtime_hours: Any,
        improved_downtime_hours: Any,
        production_rate_per_hour: Any,
        contribution_margin_per_unit: Any,
    ) -> dict[str, Decimal]:

        baseline = to_decimal(
            baseline_downtime_hours,
            "baseline_downtime_hours",
        )

        improved = to_decimal(
            improved_downtime_hours,
            "improved_downtime_hours",
        )

        rate = to_decimal(
            production_rate_per_hour,
            "production_rate_per_hour",
        )

        margin = to_decimal(
            contribution_margin_per_unit,
            "contribution_margin_per_unit",
        )

        avoided_hours = max(
            Decimal("0"),
            baseline - improved,
        )

        avoided_units = avoided_hours * rate
        avoided_loss = avoided_units * margin

        return {
            "avoided_downtime_hours": avoided_hours,
            "avoided_lost_units": quantity(
                avoided_units
            ),
            "estimated_avoided_loss": money(
                avoided_loss
            ),
        }

    # ========================================================
    # MULTI-ASSET ANALYSIS
    # ========================================================

    @staticmethod
    def aggregate_production_losses(
        results: Sequence[ProductionLossResult],
    ) -> dict[str, Any]:

        if not results:
            return {
                "total_records": 0,
                "currency": None,
                "total_lost_units": "0",
                "total_economic_loss": "0.00",
                "total_lost_revenue": "0.00",
                "total_lost_contribution_margin": "0.00",
            }

        currencies = {
            result.currency
            for result in results
        }

        if len(currencies) != 1:
            raise ProductionLossValidationError(
                "Cannot aggregate different currencies "
                "without exchange-rate conversion"
            )

        methods = {
            result.valuation_method
            for result in results
        }

        if len(methods) != 1:
            raise ProductionLossValidationError(
                "Cannot aggregate different valuation "
                "methods without normalization"
            )

        ids = [
            result.loss_id
            for result in results
        ]

        if len(ids) != len(set(ids)):
            raise ProductionLossValidationError(
                "Duplicate production loss records detected"
            )

        total_units = sum(
            (
                result.metrics.lost_units
                for result in results
            ),
            Decimal("0"),
        )

        total_loss = sum(
            (
                result.breakdown.total_economic_loss
                for result in results
            ),
            Decimal("0"),
        )

        total_revenue = sum(
            (
                result.breakdown.lost_revenue
                for result in results
            ),
            Decimal("0"),
        )

        total_margin = sum(
            (
                result.breakdown.lost_contribution_margin
                for result in results
            ),
            Decimal("0"),
        )

        return {
            "total_records": len(results),
            "currency": next(iter(currencies)),
            "valuation_method": next(iter(methods)),
            "total_lost_units": str(
                quantity(total_units)
            ),
            "total_economic_loss": str(
                money(total_loss)
            ),
            "total_lost_revenue": str(
                money(total_revenue)
            ),
            "total_lost_contribution_margin": str(
                money(total_margin)
            ),
        }

    # ========================================================
    # SAVE PRODUCTION LOSS
    # ========================================================

    async def save_production_loss(
        self,
        result: ProductionLossResult,
        actor_id: str = "system",
    ) -> Any:

        if self.repository is None:
            raise ProductionLossRepositoryError(
                "Production loss repository not configured"
            )

        if not isinstance(
            result,
            ProductionLossResult,
        ):
            raise ProductionLossValidationError(
                "result must be ProductionLossResult"
            )

        saved = await invoke(
            self.repository.create_production_loss,
            result.to_dict(),
        )

        if self.audit is not None:
            method = getattr(
                self.audit,
                "record_event",
                None,
            )

            if method is not None:
                try:
                    await invoke(
                        method,
                        {
                            "event_id": str(uuid4()),
                            "entity_type": "production_loss",
                            "entity_id": result.loss_id,
                            "action": "production_loss_created",
                            "actor_id": actor_id,
                            "details": {
                                "asset_id": result.asset_id,
                                "total_loss": str(
                                    result.breakdown.total_economic_loss
                                ),
                                "currency": result.currency,
                            },
                            "timestamp": utc_now().isoformat(),
                        },
                    )
                except Exception:
                    logger.exception(
                        "Production loss audit failed: %s",
                        result.loss_id,
                    )

        return saved

    # ========================================================
    # GET PRODUCTION LOSS
    # ========================================================

    async def get_production_loss(
        self,
        loss_id: str,
    ) -> Any:

        if self.repository is None:
            raise ProductionLossRepositoryError(
                "Production loss repository not configured"
            )

        loss_id = validate_text(
            loss_id,
            "loss_id",
        )

        record = await invoke(
            self.repository.get_production_loss,
            loss_id,
        )

        if record is None:
            raise ProductionLossNotFoundError(
                f"Production loss not found: {loss_id}"
            )

        return record

    # ========================================================
    # LIST PRODUCTION LOSSES
    # ========================================================

    async def list_production_losses(
        self,
        *,
        company_id: Optional[str] = None,
        site_id: Optional[str] = None,
        asset_id: Optional[str] = None,
        loss_type: Optional[str] = None,
    ) -> Any:

        if self.repository is None:
            raise ProductionLossRepositoryError(
                "Production loss repository not configured"
            )

        filters = {}

        if company_id is not None:
            filters["company_id"] = validate_text(
                company_id,
                "company_id",
            )

        if site_id is not None:
            filters["site_id"] = validate_text(
                site_id,
                "site_id",
            )

        if asset_id is not None:
            filters["asset_id"] = validate_text(
                asset_id,
                "asset_id",
            )

        if loss_type is not None:
            filters["loss_type"] = parse_enum(
                ProductionLossType,
                loss_type,
                "loss_type",
            ).value

        return await invoke(
            self.repository.list_production_losses,
            filters,
        )

    # ========================================================
    # PRODUCTION LOSS SUMMARY
    # ========================================================

    async def get_production_loss_summary(
        self,
        *,
        company_id: Optional[str] = None,
        site_id: Optional[str] = None,
        asset_id: Optional[str] = None,
    ) -> dict[str, Any]:

        records = await self.list_production_losses(
            company_id=company_id,
            site_id=site_id,
            asset_id=asset_id,
        )

        if not records:
            return {
                "total_records": 0,
                "currency": None,
                "total_economic_loss": "0.00",
                "total_lost_units": "0.0000",
                "generated_at": utc_now().isoformat(),
            }

        currencies = {
            record["currency"]
            for record in records
        }

        if len(currencies) != 1:
            raise ProductionLossValidationError(
                "Mixed currencies in production loss summary"
            )

        methods = {
            record["valuation_method"]
            for record in records
        }

        if len(methods) != 1:
            raise ProductionLossValidationError(
                "Mixed valuation methods in summary"
            )

        total_loss = sum(
            (
                to_decimal(
                    record["breakdown"]["total_economic_loss"],
                    "total_economic_loss",
                )
                for record in records
            ),
            Decimal("0"),
        )

        total_units = sum(
            (
                to_decimal(
                    record["metrics"]["lost_units"],
                    "lost_units",
                )
                for record in records
            ),
            Decimal("0"),
        )

        return {
            "total_records": len(records),
            "currency": next(iter(currencies)),
            "valuation_method": next(iter(methods)),
            "total_economic_loss": str(
                money(total_loss)
            ),
            "total_lost_units": str(
                quantity(total_units)
            ),
            "generated_at": utc_now().isoformat(),
        }


# ============================================================
# IN-MEMORY REPOSITORY
# ============================================================

class InMemoryProductionLossRepository:
    """
    Development-only repository.

    Not durable or concurrency-safe.
    """

    def __init__(self):
        self.records: dict[str, dict] = {}

    async def create_production_loss(
        self,
        data: dict,
    ) -> dict:

        import copy

        record = copy.deepcopy(data)

        self.records[
            record["loss_id"]
        ] = record

        return copy.deepcopy(record)

    async def get_production_loss(
        self,
        loss_id: str,
    ) -> Optional[dict]:

        import copy

        record = self.records.get(loss_id)

        return (
            copy.deepcopy(record)
            if record is not None
            else None
        )

    async def list_production_losses(
        self,
        filters: dict,
    ) -> list[dict]:

        import copy

        return [
            copy.deepcopy(record)
            for record in self.records.values()
            if all(
                record.get(key) == value
                for key, value in filters.items()
            )
        ]


# ============================================================
# DEMONSTRATION
# ============================================================

if __name__ == "__main__":

    import asyncio
    import json

    async def main():

        repository = (
            InMemoryProductionLossRepository()
        )

        service = ProductionLossService(
            production_loss_repository=repository,
            default_currency="INR",
        )

        result = service.analyze_production_loss(
            company_id="COMPANY-001",
            site_id="SITE-001",
            asset_id="MACHINE-001",
            planned_units=1000,
            actual_units=850,
            defective_units=20,
            selling_price_per_unit=500,
            contribution_margin_per_unit=200,
            scrap_cost=3000,
            rework_cost=1500,
            additional_operating_cost=1000,
            downtime_hours=2,
            loss_type="equipment_failure",
            source="telemetry",
            valuation_method="contribution_margin",
            notes="Motor failure reduced production output.",
        )

        print("\nPRODUCTION LOSS ANALYSIS")

        print(
            json.dumps(
                result.to_dict(),
                indent=2,
            )
        )

        await service.save_production_loss(
            result,
            actor_id="ENGINEER-001",
        )

        oee = service.calculate_oee(
            planned_production_hours=8,
            downtime_hours=1,
            ideal_cycle_time_hours=0.007,
            total_units=850,
            good_units=830,
        )

        print("\nOEE ANALYSIS")

        print(
            json.dumps(
                oee.to_dict(),
                indent=2,
            )
        )

        forecast = service.forecast_production_loss(
            asset_id="MACHINE-001",
            forecast_days=30,
            expected_downtime_hours_per_day=0.5,
            production_rate_per_hour=100,
            contribution_margin_per_unit=200,
        )

        print("\nPRODUCTION LOSS FORECAST")

        print(
            json.dumps(
                forecast.to_dict(),
                indent=2,
            )
        )

        summary = (
            await service.get_production_loss_summary(
                asset_id="MACHINE-001"
            )
        )

        print("\nPRODUCTION LOSS SUMMARY")

        print(
            json.dumps(
                summary,
                indent=2,
            )
        )

    asyncio.run(main())