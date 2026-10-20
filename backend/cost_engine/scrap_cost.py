# app/agents/tools/scrap_cost.py

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
    default: float | None = None,
) -> float | None:
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

class ScrapType(str, Enum):
    RAW_MATERIAL = "raw_material"
    WIP = "work_in_progress"
    FINISHED_GOOD = "finished_good"
    PACKAGING = "packaging"
    COMPONENT = "component"
    TOOLING = "tooling"
    STARTUP = "startup"
    QUALITY_REJECT = "quality_reject"
    PROCESS_FAILURE = "process_failure"
    MACHINE_FAILURE = "machine_failure"
    CONTAMINATION = "contamination"
    DAMAGE = "damage"
    OTHER = "other"


class ScrapCause(str, Enum):
    EQUIPMENT_FAILURE = "equipment_failure"
    PROCESS_DEVIATION = "process_deviation"
    QUALITY_DEFECT = "quality_defect"
    STARTUP_LOSS = "startup_loss"
    SHUTDOWN_LOSS = "shutdown_loss"
    SENSOR_FAILURE = "sensor_failure"
    CONTROL_FAILURE = "control_failure"
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    VIBRATION = "vibration"
    OPERATOR_ERROR = "operator_error"
    MATERIAL_DEFECT = "material_defect"
    CONTAMINATION = "contamination"
    POWER_FAILURE = "power_failure"
    EMERGENCY_EVENT = "emergency_event"
    UNKNOWN = "unknown"


class ScrapCostCategory(str, Enum):
    MATERIAL = "material"
    PROCESSING = "processing"
    LABOR = "labor"
    ENERGY = "energy"
    MACHINE = "machine"
    DISPOSAL = "disposal"
    HANDLING = "handling"
    INSPECTION = "inspection"
    QUALITY = "quality"
    REPLACEMENT_PRODUCTION = "replacement_production"
    LOGISTICS = "logistics"
    ENVIRONMENTAL = "environmental"
    OTHER = "other"


class ScrapSeverity(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class ScrapEconomicRisk(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class ScrapConfidence(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


# =========================================================
# Material
# =========================================================

@dataclass(slots=True)
class MaterialScrap:
    quantity: float = 0.0
    unit_material_cost: float = 0.0

    additional_material_cost: float = 0.0

    recoverable_fraction: float = 0.0
    recovery_value_per_unit: float = 0.0

    def __post_init__(self) -> None:
        self.quantity = non_negative(self.quantity)
        self.unit_material_cost = non_negative(
            self.unit_material_cost
        )
        self.additional_material_cost = non_negative(
            self.additional_material_cost
        )
        self.recoverable_fraction = clamp(
            self.recoverable_fraction
        )
        self.recovery_value_per_unit = non_negative(
            self.recovery_value_per_unit
        )

    @property
    def gross_material_cost(self) -> float:
        return (
            self.quantity
            * self.unit_material_cost
            + self.additional_material_cost
        )

    @property
    def recoverable_quantity(self) -> float:
        return (
            self.quantity
            * self.recoverable_fraction
        )

    @property
    def recovery_credit(self) -> float:
        return (
            self.recoverable_quantity
            * self.recovery_value_per_unit
        )

    @property
    def net_material_cost(self) -> float:
        return max(
            0.0,
            self.gross_material_cost
            - self.recovery_credit,
        )


# =========================================================
# Processing loss
# =========================================================

@dataclass(slots=True)
class ProcessingScrap:
    processed_units: float = 0.0

    processing_cost_per_unit: float = 0.0

    machine_hours: float = 0.0
    machine_cost_per_hour: float = 0.0

    setup_cost: float = 0.0
    tooling_cost: float = 0.0

    other_processing_cost: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "processed_units",
            "processing_cost_per_unit",
            "machine_hours",
            "machine_cost_per_hour",
            "setup_cost",
            "tooling_cost",
            "other_processing_cost",
        ):
            setattr(
                self,
                name,
                non_negative(getattr(self, name)),
            )

    @property
    def unit_processing_cost(self) -> float:
        return (
            self.processed_units
            * self.processing_cost_per_unit
        )

    @property
    def machine_cost(self) -> float:
        return (
            self.machine_hours
            * self.machine_cost_per_hour
        )

    @property
    def total(self) -> float:
        return sum(
            (
                self.unit_processing_cost,
                self.machine_cost,
                self.setup_cost,
                self.tooling_cost,
                self.other_processing_cost,
            )
        )


# =========================================================
# Labor
# =========================================================

@dataclass(slots=True)
class ScrapLaborCost:
    production_labor_hours: float = 0.0
    production_labor_rate: float = 0.0

    sorting_hours: float = 0.0
    sorting_rate: float = 0.0

    inspection_hours: float = 0.0
    inspection_rate: float = 0.0

    handling_hours: float = 0.0
    handling_rate: float = 0.0

    overtime_cost: float = 0.0
    contractor_cost: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "production_labor_hours",
            "production_labor_rate",
            "sorting_hours",
            "sorting_rate",
            "inspection_hours",
            "inspection_rate",
            "handling_hours",
            "handling_rate",
            "overtime_cost",
            "contractor_cost",
        ):
            setattr(
                self,
                name,
                non_negative(getattr(self, name)),
            )

    @property
    def production_cost(self) -> float:
        return (
            self.production_labor_hours
            * self.production_labor_rate
        )

    @property
    def sorting_cost(self) -> float:
        return (
            self.sorting_hours
            * self.sorting_rate
        )

    @property
    def inspection_cost(self) -> float:
        return (
            self.inspection_hours
            * self.inspection_rate
        )

    @property
    def handling_cost(self) -> float:
        return (
            self.handling_hours
            * self.handling_rate
        )

    @property
    def total(self) -> float:
        return sum(
            (
                self.production_cost,
                self.sorting_cost,
                self.inspection_cost,
                self.handling_cost,
                self.overtime_cost,
                self.contractor_cost,
            )
        )


# =========================================================
# Energy
# =========================================================

@dataclass(slots=True)
class ScrapEnergyCost:
    wasted_energy_kwh: float = 0.0
    energy_rate_per_kwh: float = 0.0

    compressed_air_cost: float = 0.0
    steam_cost: float = 0.0
    water_cost: float = 0.0
    fuel_cost: float = 0.0

    other_utility_cost: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "wasted_energy_kwh",
            "energy_rate_per_kwh",
            "compressed_air_cost",
            "steam_cost",
            "water_cost",
            "fuel_cost",
            "other_utility_cost",
        ):
            setattr(
                self,
                name,
                non_negative(getattr(self, name)),
            )

    @property
    def electricity_cost(self) -> float:
        return (
            self.wasted_energy_kwh
            * self.energy_rate_per_kwh
        )

    @property
    def total(self) -> float:
        return sum(
            (
                self.electricity_cost,
                self.compressed_air_cost,
                self.steam_cost,
                self.water_cost,
                self.fuel_cost,
                self.other_utility_cost,
            )
        )


# =========================================================
# Disposal / environmental
# =========================================================

@dataclass(slots=True)
class ScrapDisposalCost:
    disposal_quantity: float = 0.0
    disposal_cost_per_unit: float = 0.0

    transportation_cost: float = 0.0
    waste_handling_cost: float = 0.0

    hazardous_waste_cost: float = 0.0
    environmental_fee: float = 0.0

    recycling_cost: float = 0.0
    recycling_credit: float = 0.0

    other_cost: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "disposal_quantity",
            "disposal_cost_per_unit",
            "transportation_cost",
            "waste_handling_cost",
            "hazardous_waste_cost",
            "environmental_fee",
            "recycling_cost",
            "recycling_credit",
            "other_cost",
        ):
            setattr(
                self,
                name,
                non_negative(getattr(self, name)),
            )

    @property
    def base_disposal_cost(self) -> float:
        return (
            self.disposal_quantity
            * self.disposal_cost_per_unit
        )

    @property
    def gross_cost(self) -> float:
        return sum(
            (
                self.base_disposal_cost,
                self.transportation_cost,
                self.waste_handling_cost,
                self.hazardous_waste_cost,
                self.environmental_fee,
                self.recycling_cost,
                self.other_cost,
            )
        )

    @property
    def net_cost(self) -> float:
        return max(
            0.0,
            self.gross_cost
            - self.recycling_credit,
        )


# =========================================================
# Replacement production
# =========================================================

@dataclass(slots=True)
class ReplacementProductionCost:
    replacement_units: float = 0.0

    material_cost_per_unit: float = 0.0
    conversion_cost_per_unit: float = 0.0

    overtime_cost: float = 0.0
    expedited_cost: float = 0.0
    subcontract_cost: float = 0.0

    additional_energy_cost: float = 0.0
    additional_quality_cost: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "replacement_units",
            "material_cost_per_unit",
            "conversion_cost_per_unit",
            "overtime_cost",
            "expedited_cost",
            "subcontract_cost",
            "additional_energy_cost",
            "additional_quality_cost",
        ):
            setattr(
                self,
                name,
                non_negative(getattr(self, name)),
            )

    @property
    def unit_replacement_cost(self) -> float:
        return (
            self.replacement_units
            * (
                self.material_cost_per_unit
                + self.conversion_cost_per_unit
            )
        )

    @property
    def total(self) -> float:
        return sum(
            (
                self.unit_replacement_cost,
                self.overtime_cost,
                self.expedited_cost,
                self.subcontract_cost,
                self.additional_energy_cost,
                self.additional_quality_cost,
            )
        )


# =========================================================
# Input
# =========================================================

@dataclass(slots=True)
class ScrapCostInput:
    scrap_type: ScrapType

    material: MaterialScrap = field(
        default_factory=MaterialScrap
    )

    processing: ProcessingScrap = field(
        default_factory=ProcessingScrap
    )

    labor: ScrapLaborCost = field(
        default_factory=ScrapLaborCost
    )

    energy: ScrapEnergyCost = field(
        default_factory=ScrapEnergyCost
    )

    disposal: ScrapDisposalCost = field(
        default_factory=ScrapDisposalCost
    )

    replacement_production: ReplacementProductionCost = field(
        default_factory=ReplacementProductionCost
    )

    causes: list[ScrapCause] = field(
        default_factory=list
    )

    total_production_units: float | None = None
    good_units: float | None = None

    expected_scrap_rate: float | None = None

    quality_cost: float = 0.0
    logistics_cost: float = 0.0
    penalty_cost: float = 0.0
    other_cost: float = 0.0

    insurance_recovery: float = 0.0
    supplier_recovery: float = 0.0
    other_recovery: float = 0.0

    asset_id: str | None = None
    component_id: str | None = None
    site_id: str | None = None

    batch_id: str | None = None
    product_id: str | None = None
    incident_id: str | None = None

    currency: str = "INR"

    confidence: float = 1.0

    timestamp: datetime = field(
        default_factory=utc_now
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.total_production_units is not None:
            self.total_production_units = non_negative(
                self.total_production_units
            )

        if self.good_units is not None:
            self.good_units = non_negative(
                self.good_units
            )

        if self.expected_scrap_rate is not None:
            self.expected_scrap_rate = clamp(
                self.expected_scrap_rate
            )

        for name in (
            "quality_cost",
            "logistics_cost",
            "penalty_cost",
            "other_cost",
            "insurance_recovery",
            "supplier_recovery",
            "other_recovery",
        ):
            setattr(
                self,
                name,
                non_negative(getattr(self, name)),
            )

        self.confidence = clamp(
            self.confidence
        )

        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(
                tzinfo=timezone.utc
            )


# =========================================================
# Breakdown
# =========================================================

@dataclass(slots=True)
class ScrapCostBreakdown:
    material: float = 0.0
    processing: float = 0.0
    labor: float = 0.0
    energy: float = 0.0
    disposal: float = 0.0
    replacement_production: float = 0.0

    quality: float = 0.0
    logistics: float = 0.0
    penalty: float = 0.0
    other: float = 0.0

    material_recovery: float = 0.0
    recycling_recovery: float = 0.0
    insurance_recovery: float = 0.0
    supplier_recovery: float = 0.0
    other_recovery: float = 0.0

    @property
    def gross_cost(self) -> float:
        return sum(
            (
                self.material,
                self.processing,
                self.labor,
                self.energy,
                self.disposal,
                self.replacement_production,
                self.quality,
                self.logistics,
                self.penalty,
                self.other,
            )
        )

    @property
    def recovery_credits(self) -> float:
        return sum(
            (
                self.material_recovery,
                self.recycling_recovery,
                self.insurance_recovery,
                self.supplier_recovery,
                self.other_recovery,
            )
        )

    @property
    def net_cost(self) -> float:
        return max(
            0.0,
            self.gross_cost
            - self.recovery_credits,
        )

    def to_dict(self) -> dict[str, Any]:
        result = json_safe(self)

        result["gross_cost"] = round(
            self.gross_cost,
            2,
        )

        result["recovery_credits"] = round(
            self.recovery_credits,
            2,
        )

        result["net_cost"] = round(
            self.net_cost,
            2,
        )

        return result


# =========================================================
# Result
# =========================================================

@dataclass(slots=True)
class ScrapCostResult:
    calculation_id: str

    scrap_type: ScrapType
    causes: list[ScrapCause]

    asset_id: str | None
    component_id: str | None
    site_id: str | None

    batch_id: str | None
    product_id: str | None
    incident_id: str | None

    scrap_quantity: float

    total_production_units: float | None

    scrap_rate: float | None
    expected_scrap_rate: float | None
    excess_scrap_rate: float | None

    gross_scrap_cost: float
    recovery_credits: float
    net_scrap_cost: float

    cost_per_scrap_unit: float | None
    cost_per_production_unit: float | None

    excess_scrap_cost: float | None

    severity: ScrapSeverity
    economic_risk: ScrapEconomicRisk

    confidence: float
    confidence_level: ScrapConfidence

    breakdown: ScrapCostBreakdown

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
# Config
# =========================================================

@dataclass(slots=True)
class ScrapCostConfig:
    moderate_scrap_rate: float = 0.03
    high_scrap_rate: float = 0.07
    critical_scrap_rate: float = 0.15

    moderate_cost_ratio: float = 0.03
    high_cost_ratio: float = 0.08
    critical_cost_ratio: float = 0.15

    minimum_confidence: float = 0.50

    default_currency: str = "INR"


# =========================================================
# Calculator
# =========================================================

class ScrapCostCalculator:

    def __init__(
        self,
        *,
        config: ScrapCostConfig | None = None,
    ) -> None:
        self.config = (
            config
            or ScrapCostConfig()
        )

    # =====================================================
    # Scrap rate
    # =====================================================

    @staticmethod
    def scrap_rate(
        *,
        scrap_quantity: float,
        total_production_units: float | None,
    ) -> float | None:

        if (
            total_production_units is None
            or total_production_units <= 0
        ):
            return None

        return clamp(
            scrap_quantity
            / total_production_units
        )

    # =====================================================
    # Breakdown
    # =====================================================

    @staticmethod
    def calculate_breakdown(
        scrap: ScrapCostInput,
    ) -> ScrapCostBreakdown:

        # Material and disposal are recorded gross here,
        # with their credits separated below. This makes
        # gross/net economics auditable.
        material_gross = (
            scrap.material.gross_material_cost
        )

        disposal_gross = (
            scrap.disposal.gross_cost
        )

        return ScrapCostBreakdown(
            material=material_gross,

            processing=(
                scrap.processing.total
            ),

            labor=(
                scrap.labor.total
            ),

            energy=(
                scrap.energy.total
            ),

            disposal=disposal_gross,

            replacement_production=(
                scrap.replacement_production.total
            ),

            quality=scrap.quality_cost,
            logistics=scrap.logistics_cost,
            penalty=scrap.penalty_cost,
            other=scrap.other_cost,

            material_recovery=(
                scrap.material.recovery_credit
            ),

            recycling_recovery=(
                scrap.disposal.recycling_credit
            ),

            insurance_recovery=(
                scrap.insurance_recovery
            ),

            supplier_recovery=(
                scrap.supplier_recovery
            ),

            other_recovery=(
                scrap.other_recovery
            ),
        )

    # =====================================================
    # Confidence
    # =====================================================

    @staticmethod
    def confidence_level(
        confidence: float,
    ) -> ScrapConfidence:

        value = clamp(confidence)

        if value >= 0.90:
            return ScrapConfidence.VERY_HIGH

        if value >= 0.75:
            return ScrapConfidence.HIGH

        if value >= 0.50:
            return ScrapConfidence.MODERATE

        return ScrapConfidence.LOW

    # =====================================================
    # Severity
    # =====================================================

    def severity(
        self,
        scrap_rate: float | None,
    ) -> ScrapSeverity:

        if scrap_rate is None:
            return ScrapSeverity.MODERATE

        if (
            scrap_rate
            >= self.config.critical_scrap_rate
        ):
            return ScrapSeverity.CRITICAL

        if (
            scrap_rate
            >= self.config.high_scrap_rate
        ):
            return ScrapSeverity.HIGH

        if (
            scrap_rate
            >= self.config.moderate_scrap_rate
        ):
            return ScrapSeverity.MODERATE

        return ScrapSeverity.LOW

    # =====================================================
    # Economic risk
    # =====================================================

    def economic_risk(
        self,
        *,
        net_scrap_cost: float,
        production_value: float | None,
        scrap_rate: float | None,
    ) -> ScrapEconomicRisk:

        rate_risk = ScrapEconomicRisk.LOW

        if scrap_rate is not None:
            if (
                scrap_rate
                >= self.config.critical_scrap_rate
            ):
                rate_risk = ScrapEconomicRisk.CRITICAL

            elif (
                scrap_rate
                >= self.config.high_scrap_rate
            ):
                rate_risk = ScrapEconomicRisk.HIGH

            elif (
                scrap_rate
                >= self.config.moderate_scrap_rate
            ):
                rate_risk = ScrapEconomicRisk.MODERATE

        cost_risk = ScrapEconomicRisk.LOW

        if (
            production_value is not None
            and production_value > 0
        ):
            ratio = (
                net_scrap_cost
                / production_value
            )

            if (
                ratio
                >= self.config.critical_cost_ratio
            ):
                cost_risk = ScrapEconomicRisk.CRITICAL

            elif (
                ratio
                >= self.config.high_cost_ratio
            ):
                cost_risk = ScrapEconomicRisk.HIGH

            elif (
                ratio
                >= self.config.moderate_cost_ratio
            ):
                cost_risk = ScrapEconomicRisk.MODERATE

        rank = {
            ScrapEconomicRisk.LOW: 0,
            ScrapEconomicRisk.MODERATE: 1,
            ScrapEconomicRisk.HIGH: 2,
            ScrapEconomicRisk.CRITICAL: 3,
        }

        return max(
            (rate_risk, cost_risk),
            key=lambda item: rank[item],
        )

    # =====================================================
    # Main calculation
    # =====================================================

    def calculate(
        self,
        *,
        scrap: ScrapCostInput,
        production_value: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ScrapCostResult:

        warnings: list[str] = []

        breakdown = (
            self.calculate_breakdown(scrap)
        )

        scrap_quantity = (
            scrap.material.quantity
        )

        rate = self.scrap_rate(
            scrap_quantity=scrap_quantity,
            total_production_units=(
                scrap.total_production_units
            ),
        )

        excess_rate = None

        if (
            rate is not None
            and scrap.expected_scrap_rate
            is not None
        ):
            excess_rate = max(
                0.0,
                rate - scrap.expected_scrap_rate,
            )

        cost_per_scrap_unit = safe_divide(
            breakdown.net_cost,
            scrap_quantity,
        )

        cost_per_production_unit = (
            safe_divide(
                breakdown.net_cost,
                scrap.total_production_units,
            )
            if scrap.total_production_units
            is not None
            else None
        )

        # Estimate excess cost using excess scrap units
        # and observed average scrap cost per unit.
        excess_scrap_cost = None

        if (
            excess_rate is not None
            and scrap.total_production_units
            is not None
            and cost_per_scrap_unit
            is not None
        ):
            excess_units = (
                excess_rate
                * scrap.total_production_units
            )

            excess_scrap_cost = (
                excess_units
                * cost_per_scrap_unit
            )

        severity = self.severity(
            rate
        )

        risk = self.economic_risk(
            net_scrap_cost=(
                breakdown.net_cost
            ),
            production_value=(
                production_value
            ),
            scrap_rate=rate,
        )

        confidence_level = (
            self.confidence_level(
                scrap.confidence
            )
        )

        # ---------------------------------------------
        # Warnings
        # ---------------------------------------------

        if scrap_quantity <= 0:
            warnings.append(
                "Scrap quantity is zero or unavailable."
            )

        if (
            scrap.total_production_units
            is None
        ):
            warnings.append(
                "Total production quantity is unavailable; "
                "scrap rate cannot be calculated."
            )

        elif (
            scrap_quantity
            > scrap.total_production_units
        ):
            warnings.append(
                "Scrap quantity exceeds total production "
                "quantity. Verify unit consistency."
            )

        if (
            scrap.good_units is not None
            and scrap.total_production_units
            is not None
            and (
                scrap.good_units
                + scrap_quantity
                > scrap.total_production_units
                + 1e-9
            )
        ):
            warnings.append(
                "Good units plus scrap units exceed total "
                "production. Verify production accounting."
            )

        if (
            scrap.material.additional_material_cost > 0
            and scrap.replacement_production
            .material_cost_per_unit > 0
        ):
            warnings.append(
                "Additional material and replacement "
                "production material costs are both included. "
                "Verify they do not represent the same cost."
            )

        if (
            scrap.processing.machine_cost > 0
            and scrap.energy.total > 0
        ):
            warnings.append(
                "Machine processing cost and energy cost are "
                "both included. Confirm machine-hour rate does "
                "not already include utilities."
            )

        if (
            scrap.labor.production_cost > 0
            and scrap.processing
            .processing_cost_per_unit > 0
        ):
            warnings.append(
                "Production labor and processing cost are both "
                "included. Confirm unit processing cost does "
                "not already include direct labor."
            )

        if (
            scrap.replacement_production.total > 0
            and scrap.quality_cost > 0
        ):
            warnings.append(
                "Replacement production and quality costs are "
                "both included. Verify rework/replacement cost "
                "is not duplicated."
            )

        if (
            scrap.confidence
            < self.config.minimum_confidence
        ):
            warnings.append(
                "Scrap cost estimate has low confidence."
            )

        return ScrapCostResult(
            calculation_id=new_id(),

            scrap_type=scrap.scrap_type,
            causes=list(scrap.causes),

            asset_id=scrap.asset_id,
            component_id=(
                scrap.component_id
            ),
            site_id=scrap.site_id,

            batch_id=scrap.batch_id,
            product_id=scrap.product_id,
            incident_id=scrap.incident_id,

            scrap_quantity=round(
                scrap_quantity,
                4,
            ),

            total_production_units=(
                round(
                    scrap.total_production_units,
                    4,
                )
                if scrap.total_production_units
                is not None
                else None
            ),

            scrap_rate=(
                round(rate, 6)
                if rate is not None
                else None
            ),

            expected_scrap_rate=(
                round(
                    scrap.expected_scrap_rate,
                    6,
                )
                if scrap.expected_scrap_rate
                is not None
                else None
            ),

            excess_scrap_rate=(
                round(excess_rate, 6)
                if excess_rate is not None
                else None
            ),

            gross_scrap_cost=round(
                breakdown.gross_cost,
                2,
            ),

            recovery_credits=round(
                breakdown.recovery_credits,
                2,
            ),

            net_scrap_cost=round(
                breakdown.net_cost,
                2,
            ),

            cost_per_scrap_unit=(
                round(
                    cost_per_scrap_unit,
                    4,
                )
                if cost_per_scrap_unit
                is not None
                else None
            ),

            cost_per_production_unit=(
                round(
                    cost_per_production_unit,
                    4,
                )
                if cost_per_production_unit
                is not None
                else None
            ),

            excess_scrap_cost=(
                round(
                    excess_scrap_cost,
                    2,
                )
                if excess_scrap_cost
                is not None
                else None
            ),

            severity=severity,
            economic_risk=risk,

            confidence=round(
                scrap.confidence,
                4,
            ),

            confidence_level=(
                confidence_level
            ),

            breakdown=breakdown,

            currency=(
                scrap.currency
                or self.config.default_currency
            ),

            warnings=list(
                dict.fromkeys(warnings)
            ),

            metadata={
                **scrap.metadata,
                **dict(metadata or {}),
            },
        )

    # =====================================================
    # Simple scrap cost
    # =====================================================

    @staticmethod
    def simple_scrap_cost(
        *,
        scrap_units: float,
        material_cost_per_unit: float,
        processing_cost_per_unit: float = 0.0,
        disposal_cost_per_unit: float = 0.0,
        recovery_value_per_unit: float = 0.0,
    ) -> float:

        units = non_negative(
            scrap_units
        )

        gross_per_unit = sum(
            (
                non_negative(
                    material_cost_per_unit
                ),
                non_negative(
                    processing_cost_per_unit
                ),
                non_negative(
                    disposal_cost_per_unit
                ),
            )
        )

        recovery = non_negative(
            recovery_value_per_unit
        )

        return round(
            units
            * max(
                0.0,
                gross_per_unit - recovery,
            ),
            2,
        )

    # =====================================================
    # Scrap savings
    # =====================================================

    @staticmethod
    def scrap_reduction_savings(
        *,
        production_units: float,
        current_scrap_rate: float,
        target_scrap_rate: float,
        cost_per_scrap_unit: float,
    ) -> float:

        units = non_negative(
            production_units
        )

        current = clamp(
            current_scrap_rate
        )

        target = clamp(
            target_scrap_rate
        )

        reduction = max(
            0.0,
            current - target,
        )

        avoided_scrap_units = (
            units * reduction
        )

        return round(
            avoided_scrap_units
            * non_negative(
                cost_per_scrap_unit
            ),
            2,
        )

    # =====================================================
    # Expected scrap cost
    # =====================================================

    @staticmethod
    def expected_scrap_cost(
        *,
        event_probability: float,
        scrap_cost_if_event_occurs: float,
    ) -> float:

        return round(
            clamp(event_probability)
            * non_negative(
                scrap_cost_if_event_occurs
            ),
            2,
        )

    # =====================================================
    # Batch
    # =====================================================

    def calculate_many(
        self,
        scraps: Sequence[ScrapCostInput],
    ) -> list[ScrapCostResult]:

        return [
            self.calculate(scrap=item)
            for item in scraps
        ]

    # =====================================================
    # Aggregate
    # =====================================================

    @staticmethod
    def aggregate(
        results: Sequence[ScrapCostResult],
    ) -> dict[str, Any]:

        if not results:
            return {
                "record_count": 0,
                "scrap_quantity": 0.0,
                "net_scrap_cost": 0.0,
                "currency": None,
            }

        currencies = {
            result.currency
            for result in results
        }

        if len(currencies) > 1:
            return {
                "record_count": len(results),
                "scrap_quantity": round(
                    sum(
                        result.scrap_quantity
                        for result in results
                    ),
                    4,
                ),
                "net_scrap_cost": None,
                "currency": "MIXED",
                "mixed_currency": True,
                "warning": (
                    "Scrap costs were not aggregated because "
                    "multiple currencies are present."
                ),
            }

        quantity = sum(
            result.scrap_quantity
            for result in results
        )

        gross = sum(
            result.gross_scrap_cost
            for result in results
        )

        recovery = sum(
            result.recovery_credits
            for result in results
        )

        net = sum(
            result.net_scrap_cost
            for result in results
        )

        return {
            "record_count": len(results),

            "scrap_quantity": round(
                quantity,
                4,
            ),

            "gross_scrap_cost": round(
                gross,
                2,
            ),

            "recovery_credits": round(
                recovery,
                2,
            ),

            "net_scrap_cost": round(
                net,
                2,
            ),

            "average_cost_per_scrap_unit": (
                round(
                    net / quantity,
                    4,
                )
                if quantity > 0
                else None
            ),

            "currency": next(
                iter(currencies)
            ),

            "mixed_currency": False,
        }


# =========================================================
# Default calculator
# =========================================================

scrap_cost_calculator = ScrapCostCalculator()


# =========================================================
# Agent-facing helpers
# =========================================================

def calculate_scrap_cost(
    *,
    scrap: ScrapCostInput,
    production_value: float | None = None,
) -> ScrapCostResult:

    return scrap_cost_calculator.calculate(
        scrap=scrap,
        production_value=production_value,
    )


def calculate_simple_scrap_cost(
    *,
    scrap_units: float,
    material_cost_per_unit: float,
    processing_cost_per_unit: float = 0.0,
    disposal_cost_per_unit: float = 0.0,
    recovery_value_per_unit: float = 0.0,
) -> float:

    return (
        scrap_cost_calculator
        .simple_scrap_cost(
            scrap_units=scrap_units,
            material_cost_per_unit=(
                material_cost_per_unit
            ),
            processing_cost_per_unit=(
                processing_cost_per_unit
            ),
            disposal_cost_per_unit=(
                disposal_cost_per_unit
            ),
            recovery_value_per_unit=(
                recovery_value_per_unit
            ),
        )
    )


def calculate_scrap_reduction_savings(
    *,
    production_units: float,
    current_scrap_rate: float,
    target_scrap_rate: float,
    cost_per_scrap_unit: float,
) -> float:

    return (
        scrap_cost_calculator
        .scrap_reduction_savings(
            production_units=production_units,
            current_scrap_rate=current_scrap_rate,
            target_scrap_rate=target_scrap_rate,
            cost_per_scrap_unit=cost_per_scrap_unit,
        )
    )


def calculate_expected_scrap_cost(
    *,
    event_probability: float,
    scrap_cost_if_event_occurs: float,
) -> float:

    return (
        scrap_cost_calculator
        .expected_scrap_cost(
            event_probability=event_probability,
            scrap_cost_if_event_occurs=(
                scrap_cost_if_event_occurs
            ),
        )
    )


# =========================================================
# Exports
# =========================================================

__all__ = [
    "ScrapType",
    "ScrapCause",
    "ScrapCostCategory",
    "ScrapSeverity",
    "ScrapEconomicRisk",
    "ScrapConfidence",

    "MaterialScrap",
    "ProcessingScrap",
    "ScrapLaborCost",
    "ScrapEnergyCost",
    "ScrapDisposalCost",
    "ReplacementProductionCost",

    "ScrapCostInput",
    "ScrapCostBreakdown",
    "ScrapCostResult",

    "ScrapCostConfig",
    "ScrapCostCalculator",

    "scrap_cost_calculator",

    "calculate_scrap_cost",
    "calculate_simple_scrap_cost",
    "calculate_scrap_reduction_savings",
    "calculate_expected_scrap_cost",
]