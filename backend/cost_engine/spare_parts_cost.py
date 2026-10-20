# app/agents/tools/spare_parts_cost.py

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


def ensure_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None

    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc)


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

class SparePartCategory(str, Enum):
    BEARING = "bearing"
    SEAL = "seal"
    BELT = "belt"
    FILTER = "filter"
    LUBRICANT = "lubricant"

    MOTOR = "motor"
    GEARBOX = "gearbox"
    PUMP = "pump"
    COMPRESSOR = "compressor"
    VALVE = "valve"

    SENSOR = "sensor"
    ACTUATOR = "actuator"
    PLC = "plc"
    CONTROLLER = "controller"

    ELECTRICAL = "electrical"
    ELECTRONIC = "electronic"
    MECHANICAL = "mechanical"
    HYDRAULIC = "hydraulic"
    PNEUMATIC = "pneumatic"

    SAFETY = "safety"
    CONSUMABLE = "consumable"
    TOOLING = "tooling"

    OTHER = "other"


class PartCriticality(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ProcurementType(str, Enum):
    STOCK = "stock"
    STANDARD = "standard"
    EXPEDITED = "expedited"
    EMERGENCY = "emergency"
    IMPORT = "import"
    LOCAL = "local"


class InventoryPolicy(str, Enum):
    NONE = "none"
    MIN_MAX = "min_max"
    REORDER_POINT = "reorder_point"
    SAFETY_STOCK = "safety_stock"
    CRITICAL_SPARE = "critical_spare"


class StockStatus(str, Enum):
    AVAILABLE = "available"
    LOW_STOCK = "low_stock"
    OUT_OF_STOCK = "out_of_stock"
    OVERSTOCK = "overstock"
    OBSOLETE = "obsolete"
    UNKNOWN = "unknown"


class SparePartEconomicRisk(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class SparePartConfidence(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


# =========================================================
# Part
# =========================================================

@dataclass(slots=True)
class SparePart:
    part_id: str
    name: str

    category: SparePartCategory = (
        SparePartCategory.OTHER
    )

    quantity_required: float = 1.0

    unit_cost: float = 0.0

    criticality: PartCriticality = (
        PartCriticality.MEDIUM
    )

    supplier: str | None = None
    manufacturer: str | None = None
    manufacturer_part_number: str | None = None

    lead_time_days: float = 0.0

    stock_quantity: float = 0.0
    reserved_quantity: float = 0.0

    reorder_point: float = 0.0
    safety_stock: float = 0.0

    annual_usage: float = 0.0

    shelf_life_days: float | None = None

    obsolete: bool = False
    repairable: bool = False

    core_return_credit: float = 0.0

    confidence: float = 1.0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.part_id = str(self.part_id).strip()
        self.name = str(self.name).strip()

        for name in (
            "quantity_required",
            "unit_cost",
            "lead_time_days",
            "stock_quantity",
            "reserved_quantity",
            "reorder_point",
            "safety_stock",
            "annual_usage",
            "core_return_credit",
        ):
            setattr(
                self,
                name,
                non_negative(getattr(self, name)),
            )

        if self.shelf_life_days is not None:
            self.shelf_life_days = non_negative(
                self.shelf_life_days
            )

        self.confidence = clamp(self.confidence)

    @property
    def available_quantity(self) -> float:
        return max(
            0.0,
            self.stock_quantity
            - self.reserved_quantity,
        )

    @property
    def shortage_quantity(self) -> float:
        return max(
            0.0,
            self.quantity_required
            - self.available_quantity,
        )

    @property
    def stock_usage_quantity(self) -> float:
        return min(
            self.quantity_required,
            self.available_quantity,
        )

    @property
    def required_parts_value(self) -> float:
        return (
            self.quantity_required
            * self.unit_cost
        )

    @property
    def shortage_purchase_value(self) -> float:
        return (
            self.shortage_quantity
            * self.unit_cost
        )


# =========================================================
# Procurement
# =========================================================

@dataclass(slots=True)
class ProcurementCost:
    procurement_type: ProcurementType = (
        ProcurementType.STANDARD
    )

    freight_cost: float = 0.0
    shipping_cost: float = 0.0

    customs_duty: float = 0.0
    import_tax: float = 0.0

    handling_cost: float = 0.0
    brokerage_cost: float = 0.0

    emergency_premium: float = 0.0
    expedited_premium: float = 0.0

    inspection_cost: float = 0.0

    other_cost: float = 0.0

    supplier_discount: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "freight_cost",
            "shipping_cost",
            "customs_duty",
            "import_tax",
            "handling_cost",
            "brokerage_cost",
            "emergency_premium",
            "expedited_premium",
            "inspection_cost",
            "other_cost",
            "supplier_discount",
        ):
            setattr(
                self,
                name,
                non_negative(getattr(self, name)),
            )

    @property
    def gross_cost(self) -> float:
        return sum(
            (
                self.freight_cost,
                self.shipping_cost,
                self.customs_duty,
                self.import_tax,
                self.handling_cost,
                self.brokerage_cost,
                self.emergency_premium,
                self.expedited_premium,
                self.inspection_cost,
                self.other_cost,
            )
        )

    @property
    def net_cost(self) -> float:
        return max(
            0.0,
            self.gross_cost
            - self.supplier_discount,
        )


# =========================================================
# Inventory carrying cost
# =========================================================

@dataclass(slots=True)
class InventoryCarryingCost:
    average_inventory_value: float = 0.0

    annual_capital_rate: float = 0.0
    annual_storage_rate: float = 0.0
    annual_insurance_rate: float = 0.0
    annual_obsolescence_rate: float = 0.0

    warehouse_cost: float = 0.0
    inventory_management_cost: float = 0.0

    horizon_years: float = 1.0

    def __post_init__(self) -> None:
        self.average_inventory_value = non_negative(
            self.average_inventory_value
        )

        self.annual_capital_rate = clamp(
            self.annual_capital_rate
        )
        self.annual_storage_rate = clamp(
            self.annual_storage_rate
        )
        self.annual_insurance_rate = clamp(
            self.annual_insurance_rate
        )
        self.annual_obsolescence_rate = clamp(
            self.annual_obsolescence_rate
        )

        self.warehouse_cost = non_negative(
            self.warehouse_cost
        )
        self.inventory_management_cost = non_negative(
            self.inventory_management_cost
        )
        self.horizon_years = non_negative(
            self.horizon_years
        )

    @property
    def carrying_rate(self) -> float:
        return sum(
            (
                self.annual_capital_rate,
                self.annual_storage_rate,
                self.annual_insurance_rate,
                self.annual_obsolescence_rate,
            )
        )

    @property
    def variable_carrying_cost(self) -> float:
        return (
            self.average_inventory_value
            * self.carrying_rate
            * self.horizon_years
        )

    @property
    def total(self) -> float:
        return (
            self.variable_carrying_cost
            + (
                self.warehouse_cost
                + self.inventory_management_cost
            )
            * self.horizon_years
        )


# =========================================================
# Stockout consequence
# =========================================================

@dataclass(slots=True)
class StockoutCost:
    stockout_probability: float = 0.0

    expected_delay_hours: float = 0.0

    downtime_cost_per_hour: float = 0.0
    production_loss_per_hour: float = 0.0

    emergency_procurement_cost: float = 0.0
    contractor_delay_cost: float = 0.0
    logistics_delay_cost: float = 0.0

    penalty_cost: float = 0.0
    other_cost: float = 0.0

    def __post_init__(self) -> None:
        self.stockout_probability = clamp(
            self.stockout_probability
        )

        for name in (
            "expected_delay_hours",
            "downtime_cost_per_hour",
            "production_loss_per_hour",
            "emergency_procurement_cost",
            "contractor_delay_cost",
            "logistics_delay_cost",
            "penalty_cost",
            "other_cost",
        ):
            setattr(
                self,
                name,
                non_negative(getattr(self, name)),
            )

    @property
    def downtime_cost(self) -> float:
        return (
            self.expected_delay_hours
            * self.downtime_cost_per_hour
        )

    @property
    def production_loss(self) -> float:
        return (
            self.expected_delay_hours
            * self.production_loss_per_hour
        )

    @property
    def consequence_cost(self) -> float:
        return sum(
            (
                self.downtime_cost,
                self.production_loss,
                self.emergency_procurement_cost,
                self.contractor_delay_cost,
                self.logistics_delay_cost,
                self.penalty_cost,
                self.other_cost,
            )
        )

    @property
    def expected_cost(self) -> float:
        return (
            self.stockout_probability
            * self.consequence_cost
        )


# =========================================================
# Input
# =========================================================

@dataclass(slots=True)
class SparePartsCostInput:
    parts: list[SparePart]

    procurement: ProcurementCost = field(
        default_factory=ProcurementCost
    )

    carrying_cost: InventoryCarryingCost = field(
        default_factory=InventoryCarryingCost
    )

    stockout: StockoutCost = field(
        default_factory=StockoutCost
    )

    inventory_policy: InventoryPolicy = (
        InventoryPolicy.REORDER_POINT
    )

    installation_consumables_cost: float = 0.0
    packaging_cost: float = 0.0
    internal_handling_cost: float = 0.0
    quality_inspection_cost: float = 0.0
    other_cost: float = 0.0

    warranty_credit: float = 0.0
    supplier_credit: float = 0.0
    insurance_recovery: float = 0.0
    other_recovery: float = 0.0

    asset_id: str | None = None
    component_id: str | None = None
    site_id: str | None = None

    work_order_id: str | None = None
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
        for name in (
            "installation_consumables_cost",
            "packaging_cost",
            "internal_handling_cost",
            "quality_inspection_cost",
            "other_cost",
            "warranty_credit",
            "supplier_credit",
            "insurance_recovery",
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

        self.timestamp = (
            ensure_utc(self.timestamp)
            or utc_now()
        )


# =========================================================
# Per-part result
# =========================================================

@dataclass(slots=True)
class PartCostResult:
    part_id: str
    name: str

    category: SparePartCategory
    criticality: PartCriticality

    quantity_required: float
    available_quantity: float
    stock_usage_quantity: float
    shortage_quantity: float

    unit_cost: float

    required_parts_value: float
    stock_value_used: float
    purchase_value_required: float

    core_return_credit: float
    net_part_value: float

    lead_time_days: float

    stock_status: StockStatus

    supplier: str | None
    manufacturer_part_number: str | None

    warnings: list[str] = field(
        default_factory=list
    )


# =========================================================
# Breakdown
# =========================================================

@dataclass(slots=True)
class SparePartsCostBreakdown:
    parts_value: float = 0.0
    stock_value_used: float = 0.0
    new_purchase_value: float = 0.0

    procurement_cost: float = 0.0
    inventory_carrying_cost: float = 0.0
    expected_stockout_cost: float = 0.0

    installation_consumables: float = 0.0
    packaging: float = 0.0
    internal_handling: float = 0.0
    quality_inspection: float = 0.0
    other: float = 0.0

    core_return_credit: float = 0.0
    supplier_discount: float = 0.0
    warranty_credit: float = 0.0
    supplier_credit: float = 0.0
    insurance_recovery: float = 0.0
    other_recovery: float = 0.0

    @property
    def gross_cost(self) -> float:
        return sum(
            (
                self.parts_value,
                self.procurement_cost,
                self.inventory_carrying_cost,
                self.expected_stockout_cost,
                self.installation_consumables,
                self.packaging,
                self.internal_handling,
                self.quality_inspection,
                self.other,
            )
        )

    @property
    def recovery_credits(self) -> float:
        return sum(
            (
                self.core_return_credit,
                self.supplier_discount,
                self.warranty_credit,
                self.supplier_credit,
                self.insurance_recovery,
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
            self.gross_cost, 2
        )
        result["recovery_credits"] = round(
            self.recovery_credits, 2
        )
        result["net_cost"] = round(
            self.net_cost, 2
        )
        return result


# =========================================================
# Result
# =========================================================

@dataclass(slots=True)
class SparePartsCostResult:
    calculation_id: str

    asset_id: str | None
    component_id: str | None
    site_id: str | None

    work_order_id: str | None
    incident_id: str | None

    part_count: int

    total_quantity_required: float
    total_available_quantity: float
    total_shortage_quantity: float

    gross_parts_cost: float
    recovery_credits: float
    net_spare_parts_cost: float

    inventory_value_used: float
    procurement_purchase_value: float

    expected_stockout_cost: float
    inventory_carrying_cost: float

    shortage_ratio: float
    critical_parts_shortage_count: int

    economic_risk: SparePartEconomicRisk

    confidence: float
    confidence_level: SparePartConfidence

    parts: list[PartCostResult]

    breakdown: SparePartsCostBreakdown

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
class SparePartsCostConfig:
    moderate_shortage_ratio: float = 0.10
    high_shortage_ratio: float = 0.30
    critical_shortage_ratio: float = 0.60

    long_lead_time_days: float = 30.0
    very_long_lead_time_days: float = 90.0

    minimum_confidence: float = 0.50

    default_currency: str = "INR"


# =========================================================
# Calculator
# =========================================================

class SparePartsCostCalculator:

    def __init__(
        self,
        *,
        config: SparePartsCostConfig | None = None,
    ) -> None:
        self.config = (
            config
            or SparePartsCostConfig()
        )

    # =====================================================
    # Stock status
    # =====================================================

    @staticmethod
    def stock_status(
        part: SparePart,
    ) -> StockStatus:

        if part.obsolete:
            return StockStatus.OBSOLETE

        available = part.available_quantity

        if available <= 0:
            return StockStatus.OUT_OF_STOCK

        if available < part.quantity_required:
            return StockStatus.LOW_STOCK

        if (
            part.reorder_point > 0
            and available <= part.reorder_point
        ):
            return StockStatus.LOW_STOCK

        overstock_threshold = max(
            part.reorder_point * 3.0,
            part.safety_stock * 4.0,
        )

        if (
            overstock_threshold > 0
            and available > overstock_threshold
        ):
            return StockStatus.OVERSTOCK

        return StockStatus.AVAILABLE

    # =====================================================
    # Per-part cost
    # =====================================================

    def calculate_part(
        self,
        part: SparePart,
    ) -> PartCostResult:

        warnings: list[str] = []

        stock_value = (
            part.stock_usage_quantity
            * part.unit_cost
        )

        purchase_value = (
            part.shortage_quantity
            * part.unit_cost
        )

        gross_required_value = (
            part.required_parts_value
        )

        core_credit = min(
            part.core_return_credit,
            gross_required_value,
        )

        net_value = max(
            0.0,
            gross_required_value
            - core_credit,
        )

        status = self.stock_status(part)

        if not part.part_id:
            warnings.append(
                "Part ID is missing."
            )

        if part.unit_cost <= 0:
            warnings.append(
                "Part unit cost is zero or unavailable."
            )

        if part.shortage_quantity > 0:
            warnings.append(
                "Required spare quantity exceeds "
                "currently available inventory."
            )

        if (
            part.lead_time_days
            >= self.config.long_lead_time_days
        ):
            warnings.append(
                "Part has a long procurement lead time."
            )

        if (
            part.lead_time_days
            >= self.config.very_long_lead_time_days
        ):
            warnings.append(
                "Part has a very long procurement lead time."
            )

        if part.obsolete:
            warnings.append(
                "Part is marked obsolete."
            )

        if (
            part.criticality
            == PartCriticality.CRITICAL
            and part.shortage_quantity > 0
        ):
            warnings.append(
                "Critical spare has insufficient stock."
            )

        return PartCostResult(
            part_id=part.part_id,
            name=part.name,

            category=part.category,
            criticality=part.criticality,

            quantity_required=round(
                part.quantity_required, 4
            ),
            available_quantity=round(
                part.available_quantity, 4
            ),
            stock_usage_quantity=round(
                part.stock_usage_quantity, 4
            ),
            shortage_quantity=round(
                part.shortage_quantity, 4
            ),

            unit_cost=round(
                part.unit_cost, 2
            ),

            required_parts_value=round(
                gross_required_value, 2
            ),
            stock_value_used=round(
                stock_value, 2
            ),
            purchase_value_required=round(
                purchase_value, 2
            ),

            core_return_credit=round(
                core_credit, 2
            ),
            net_part_value=round(
                net_value, 2
            ),

            lead_time_days=round(
                part.lead_time_days, 2
            ),

            stock_status=status,

            supplier=part.supplier,
            manufacturer_part_number=(
                part.manufacturer_part_number
            ),

            warnings=warnings,
        )

    # =====================================================
    # Confidence
    # =====================================================

    @staticmethod
    def confidence(
        request: SparePartsCostInput,
    ) -> tuple[float, SparePartConfidence]:

        scores = [request.confidence]

        scores.extend(
            part.confidence
            for part in request.parts
        )

        score = clamp(
            sum(scores) / len(scores)
            if scores
            else 0.0
        )

        if score >= 0.90:
            level = SparePartConfidence.VERY_HIGH
        elif score >= 0.75:
            level = SparePartConfidence.HIGH
        elif score >= 0.50:
            level = SparePartConfidence.MODERATE
        else:
            level = SparePartConfidence.LOW

        return round(score, 4), level

    # =====================================================
    # Risk
    # =====================================================

    def economic_risk(
        self,
        *,
        shortage_ratio: float,
        critical_shortage_count: int,
        expected_stockout_cost: float,
    ) -> SparePartEconomicRisk:

        if critical_shortage_count > 0:
            return SparePartEconomicRisk.CRITICAL

        if (
            shortage_ratio
            >= self.config.critical_shortage_ratio
        ):
            return SparePartEconomicRisk.CRITICAL

        if (
            shortage_ratio
            >= self.config.high_shortage_ratio
        ):
            return SparePartEconomicRisk.HIGH

        if (
            shortage_ratio
            >= self.config.moderate_shortage_ratio
            or expected_stockout_cost > 0
        ):
            return SparePartEconomicRisk.MODERATE

        return SparePartEconomicRisk.LOW

    # =====================================================
    # Main calculation
    # =====================================================

    def calculate(
        self,
        *,
        request: SparePartsCostInput,
        metadata: Mapping[str, Any] | None = None,
    ) -> SparePartsCostResult:

        warnings: list[str] = []

        part_results = [
            self.calculate_part(part)
            for part in request.parts
        ]

        total_required = sum(
            part.quantity_required
            for part in request.parts
        )

        total_available = sum(
            min(
                part.available_quantity,
                part.quantity_required,
            )
            for part in request.parts
        )

        total_shortage = sum(
            part.shortage_quantity
            for part in request.parts
        )

        required_parts_value = sum(
            part.required_parts_value
            for part in request.parts
        )

        stock_value_used = sum(
            part.stock_usage_quantity
            * part.unit_cost
            for part in request.parts
        )

        purchase_value = sum(
            part.shortage_purchase_value
            for part in request.parts
        )

        core_return_credit = sum(
            min(
                part.core_return_credit,
                part.required_parts_value,
            )
            for part in request.parts
        )

        procurement_cost = (
            request.procurement.net_cost
        )

        carrying_cost = (
            request.carrying_cost.total
        )

        expected_stockout_cost = (
            request.stockout.expected_cost
        )

        breakdown = SparePartsCostBreakdown(
            parts_value=required_parts_value,

            stock_value_used=stock_value_used,
            new_purchase_value=purchase_value,

            procurement_cost=procurement_cost,

            inventory_carrying_cost=(
                carrying_cost
            ),

            expected_stockout_cost=(
                expected_stockout_cost
            ),

            installation_consumables=(
                request.installation_consumables_cost
            ),

            packaging=request.packaging_cost,

            internal_handling=(
                request.internal_handling_cost
            ),

            quality_inspection=(
                request.quality_inspection_cost
            ),

            other=request.other_cost,

            core_return_credit=(
                core_return_credit
            ),

            supplier_discount=(
                request.procurement.supplier_discount
            ),

            warranty_credit=(
                request.warranty_credit
            ),

            supplier_credit=(
                request.supplier_credit
            ),

            insurance_recovery=(
                request.insurance_recovery
            ),

            other_recovery=(
                request.other_recovery
            ),
        )

        shortage_ratio = (
            safe_divide(
                total_shortage,
                total_required,
                0.0,
            )
            or 0.0
        )

        critical_shortages = sum(
            1
            for part in request.parts
            if (
                part.criticality
                == PartCriticality.CRITICAL
                and part.shortage_quantity > 0
            )
        )

        confidence, confidence_level = (
            self.confidence(request)
        )

        risk = self.economic_risk(
            shortage_ratio=shortage_ratio,
            critical_shortage_count=(
                critical_shortages
            ),
            expected_stockout_cost=(
                expected_stockout_cost
            ),
        )

        # -------------------------------------------------
        # Validation warnings
        # -------------------------------------------------

        if not request.parts:
            warnings.append(
                "No spare parts were supplied."
            )

        if total_shortage > 0:
            warnings.append(
                "One or more required spare parts "
                "have insufficient available inventory."
            )

        if critical_shortages > 0:
            warnings.append(
                f"{critical_shortages} critical spare "
                "part(s) have insufficient inventory."
            )

        if (
            request.stockout.downtime_cost > 0
            and request.stockout.production_loss > 0
        ):
            warnings.append(
                "Stockout downtime cost and production "
                "loss are both included. Confirm downtime "
                "cost does not already embed production loss."
            )

        if (
            request.procurement.freight_cost > 0
            and request.procurement.shipping_cost > 0
        ):
            warnings.append(
                "Freight and shipping are both included. "
                "Verify they represent separate costs."
            )

        if (
            request.quality_inspection_cost > 0
            and request.procurement.inspection_cost > 0
        ):
            warnings.append(
                "Procurement inspection and internal quality "
                "inspection are both included. Verify they "
                "represent separate activities."
            )

        if (
            request.carrying_cost.average_inventory_value > 0
            and request.carrying_cost.warehouse_cost > 0
        ):
            warnings.append(
                "Inventory carrying-rate costs and explicit "
                "warehouse costs are both included. Verify "
                "storage cost is not duplicated."
            )

        if (
            confidence
            < self.config.minimum_confidence
        ):
            warnings.append(
                "Spare-parts cost estimate has low confidence."
            )

        for part_result in part_results:
            for warning in part_result.warnings:
                warnings.append(
                    f"{part_result.part_id}: {warning}"
                )

        return SparePartsCostResult(
            calculation_id=new_id(),

            asset_id=request.asset_id,
            component_id=request.component_id,
            site_id=request.site_id,

            work_order_id=request.work_order_id,
            incident_id=request.incident_id,

            part_count=len(request.parts),

            total_quantity_required=round(
                total_required, 4
            ),

            total_available_quantity=round(
                total_available, 4
            ),

            total_shortage_quantity=round(
                total_shortage, 4
            ),

            gross_parts_cost=round(
                breakdown.gross_cost, 2
            ),

            recovery_credits=round(
                breakdown.recovery_credits, 2
            ),

            net_spare_parts_cost=round(
                breakdown.net_cost, 2
            ),

            inventory_value_used=round(
                stock_value_used, 2
            ),

            procurement_purchase_value=round(
                purchase_value, 2
            ),

            expected_stockout_cost=round(
                expected_stockout_cost, 2
            ),

            inventory_carrying_cost=round(
                carrying_cost, 2
            ),

            shortage_ratio=round(
                shortage_ratio, 6
            ),

            critical_parts_shortage_count=(
                critical_shortages
            ),

            economic_risk=risk,

            confidence=confidence,
            confidence_level=confidence_level,

            parts=part_results,

            breakdown=breakdown,

            currency=(
                request.currency
                or self.config.default_currency
            ),

            warnings=list(
                dict.fromkeys(warnings)
            ),

            metadata={
                **request.metadata,
                **dict(metadata or {}),
            },
        )

    # =====================================================
    # Simple part cost
    # =====================================================

    @staticmethod
    def simple_parts_cost(
        *,
        quantity: float,
        unit_cost: float,
        procurement_cost: float = 0.0,
        recovery_credit: float = 0.0,
    ) -> float:

        gross = (
            non_negative(quantity)
            * non_negative(unit_cost)
            + non_negative(procurement_cost)
        )

        return round(
            max(
                0.0,
                gross
                - non_negative(
                    recovery_credit
                ),
            ),
            2,
        )

    # =====================================================
    # Reorder point
    # =====================================================

    @staticmethod
    def reorder_point(
        *,
        average_daily_usage: float,
        lead_time_days: float,
        safety_stock: float = 0.0,
    ) -> float:

        return round(
            (
                non_negative(
                    average_daily_usage
                )
                * non_negative(
                    lead_time_days
                )
            )
            + non_negative(
                safety_stock
            ),
            4,
        )

    # =====================================================
    # Safety stock
    # =====================================================

    @staticmethod
    def safety_stock_simple(
        *,
        max_daily_usage: float,
        max_lead_time_days: float,
        average_daily_usage: float,
        average_lead_time_days: float,
    ) -> float:

        maximum_usage = (
            non_negative(max_daily_usage)
            * non_negative(
                max_lead_time_days
            )
        )

        normal_usage = (
            non_negative(
                average_daily_usage
            )
            * non_negative(
                average_lead_time_days
            )
        )

        return round(
            max(
                0.0,
                maximum_usage - normal_usage,
            ),
            4,
        )

    # =====================================================
    # Annual carrying cost
    # =====================================================

    @staticmethod
    def annual_carrying_cost(
        *,
        average_inventory_value: float,
        carrying_rate: float,
    ) -> float:

        return round(
            non_negative(
                average_inventory_value
            )
            * non_negative(
                carrying_rate
            ),
            2,
        )

    # =====================================================
    # Expected stockout loss
    # =====================================================

    @staticmethod
    def expected_stockout_loss(
        *,
        stockout_probability: float,
        downtime_hours: float,
        downtime_cost_per_hour: float,
        production_loss_per_hour: float = 0.0,
        emergency_procurement_cost: float = 0.0,
    ) -> float:

        consequence = (
            non_negative(
                downtime_hours
            )
            * (
                non_negative(
                    downtime_cost_per_hour
                )
                + non_negative(
                    production_loss_per_hour
                )
            )
            + non_negative(
                emergency_procurement_cost
            )
        )

        return round(
            clamp(
                stockout_probability
            )
            * consequence,
            2,
        )

    # =====================================================
    # Batch
    # =====================================================

    def calculate_many(
        self,
        requests: Sequence[
            SparePartsCostInput
        ],
    ) -> list[SparePartsCostResult]:

        return [
            self.calculate(request=item)
            for item in requests
        ]

    # =====================================================
    # Aggregate
    # =====================================================

    @staticmethod
    def aggregate(
        results: Sequence[
            SparePartsCostResult
        ],
    ) -> dict[str, Any]:

        if not results:
            return {
                "record_count": 0,
                "net_spare_parts_cost": 0.0,
                "currency": None,
            }

        currencies = {
            result.currency
            for result in results
        }

        # Never silently sum different currencies.
        if len(currencies) > 1:
            return {
                "record_count": len(results),
                "net_spare_parts_cost": None,
                "currency": "MIXED",
                "mixed_currency": True,
                "warning": (
                    "Spare-parts costs were not aggregated "
                    "because multiple currencies are present."
                ),
            }

        gross = sum(
            result.gross_parts_cost
            for result in results
        )

        recoveries = sum(
            result.recovery_credits
            for result in results
        )

        net = sum(
            result.net_spare_parts_cost
            for result in results
        )

        shortage = sum(
            result.total_shortage_quantity
            for result in results
        )

        critical_shortages = sum(
            result.critical_parts_shortage_count
            for result in results
        )

        return {
            "record_count": len(results),

            "gross_spare_parts_cost": round(
                gross, 2
            ),

            "recovery_credits": round(
                recoveries, 2
            ),

            "net_spare_parts_cost": round(
                net, 2
            ),

            "total_shortage_quantity": round(
                shortage, 4
            ),

            "critical_parts_shortage_count": (
                critical_shortages
            ),

            "currency": next(
                iter(currencies)
            ),

            "mixed_currency": False,
        }


# =========================================================
# Default calculator
# =========================================================

spare_parts_cost_calculator = (
    SparePartsCostCalculator()
)


# =========================================================
# Agent-facing helpers
# =========================================================

def calculate_spare_parts_cost(
    *,
    request: SparePartsCostInput,
) -> SparePartsCostResult:

    return (
        spare_parts_cost_calculator
        .calculate(
            request=request
        )
    )


def calculate_simple_spare_parts_cost(
    *,
    quantity: float,
    unit_cost: float,
    procurement_cost: float = 0.0,
    recovery_credit: float = 0.0,
) -> float:

    return (
        spare_parts_cost_calculator
        .simple_parts_cost(
            quantity=quantity,
            unit_cost=unit_cost,
            procurement_cost=(
                procurement_cost
            ),
            recovery_credit=(
                recovery_credit
            ),
        )
    )


def calculate_reorder_point(
    *,
    average_daily_usage: float,
    lead_time_days: float,
    safety_stock: float = 0.0,
) -> float:

    return (
        spare_parts_cost_calculator
        .reorder_point(
            average_daily_usage=(
                average_daily_usage
            ),
            lead_time_days=(
                lead_time_days
            ),
            safety_stock=safety_stock,
        )
    )


def calculate_expected_stockout_loss(
    *,
    stockout_probability: float,
    downtime_hours: float,
    downtime_cost_per_hour: float,
    production_loss_per_hour: float = 0.0,
    emergency_procurement_cost: float = 0.0,
) -> float:

    return (
        spare_parts_cost_calculator
        .expected_stockout_loss(
            stockout_probability=(
                stockout_probability
            ),
            downtime_hours=(
                downtime_hours
            ),
            downtime_cost_per_hour=(
                downtime_cost_per_hour
            ),
            production_loss_per_hour=(
                production_loss_per_hour
            ),
            emergency_procurement_cost=(
                emergency_procurement_cost
            ),
        )
    )


# =========================================================
# Exports
# =========================================================

__all__ = [
    "SparePartCategory",
    "PartCriticality",
    "ProcurementType",
    "InventoryPolicy",
    "StockStatus",
    "SparePartEconomicRisk",
    "SparePartConfidence",

    "SparePart",
    "ProcurementCost",
    "InventoryCarryingCost",
    "StockoutCost",

    "SparePartsCostInput",
    "PartCostResult",
    "SparePartsCostBreakdown",
    "SparePartsCostResult",

    "SparePartsCostConfig",
    "SparePartsCostCalculator",

    "spare_parts_cost_calculator",

    "calculate_spare_parts_cost",
    "calculate_simple_spare_parts_cost",
    "calculate_reorder_point",
    "calculate_expected_stockout_loss",
]