# app/agents/tools/repair_cost.py

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

class RepairType(str, Enum):
    MINOR = "minor"
    CORRECTIVE = "corrective"
    COMPONENT = "component"
    MAJOR = "major"
    OVERHAUL = "overhaul"
    EMERGENCY = "emergency"
    TEMPORARY = "temporary"
    WARRANTY = "warranty"
    OTHER = "other"


class RepairCostCategory(str, Enum):
    PARTS = "parts"
    MATERIALS = "materials"
    LABOR = "labor"
    CONTRACTOR = "contractor"
    OEM = "oem"
    DIAGNOSTICS = "diagnostics"
    INSPECTION = "inspection"
    TOOLING = "tooling"
    EQUIPMENT_RENTAL = "equipment_rental"
    LOGISTICS = "logistics"
    SHIPPING = "shipping"
    TRAVEL = "travel"
    DOWNTIME = "downtime"
    PRODUCTION_LOSS = "production_loss"
    TESTING = "testing"
    COMMISSIONING = "commissioning"
    SAFETY = "safety"
    CLEANUP = "cleanup"
    WARRANTY = "warranty"
    OTHER = "other"


class RepairUrgency(str, Enum):
    ROUTINE = "routine"
    PRIORITY = "priority"
    URGENT = "urgent"
    EMERGENCY = "emergency"


class RepairEconomicRisk(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class RepairConfidence(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class RepairDecision(str, Enum):
    REPAIR = "repair"
    REPAIR_AND_MONITOR = "repair_and_monitor"
    OVERHAUL = "overhaul"
    REPLACE = "replace"
    HUMAN_REVIEW = "human_review"


# =========================================================
# Parts
# =========================================================

@dataclass(slots=True)
class RepairPart:
    part_id: str | None
    name: str

    quantity: float = 1.0
    unit_cost: float = 0.0

    shipping_cost: float = 0.0
    tax_cost: float = 0.0

    installation_consumables_cost: float = 0.0

    salvage_credit: float = 0.0
    warranty_credit: float = 0.0

    available: bool = True
    lead_time_hours: float | None = None

    confidence: float = 1.0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.quantity = non_negative(
            self.quantity
        )

        for name in (
            "unit_cost",
            "shipping_cost",
            "tax_cost",
            "installation_consumables_cost",
            "salvage_credit",
            "warranty_credit",
        ):
            setattr(
                self,
                name,
                non_negative(getattr(self, name)),
            )

        if self.lead_time_hours is not None:
            self.lead_time_hours = non_negative(
                self.lead_time_hours
            )

        self.confidence = clamp(
            self.confidence
        )

    @property
    def purchase_cost(self) -> float:
        return (
            self.quantity
            * self.unit_cost
        )

    @property
    def gross_cost(self) -> float:
        return sum(
            (
                self.purchase_cost,
                self.shipping_cost,
                self.tax_cost,
                self.installation_consumables_cost,
            )
        )

    @property
    def credits(self) -> float:
        return (
            self.salvage_credit
            + self.warranty_credit
        )

    @property
    def net_cost(self) -> float:
        return max(
            0.0,
            self.gross_cost - self.credits,
        )


# =========================================================
# Labor
# =========================================================

@dataclass(slots=True)
class RepairLabor:
    role: str

    worker_count: int = 1
    hours_per_worker: float = 0.0
    hourly_rate: float = 0.0

    overtime_hours_per_worker: float = 0.0
    overtime_multiplier: float = 1.5

    contractor_callout_cost: float = 0.0

    confidence: float = 1.0

    def __post_init__(self) -> None:
        self.worker_count = max(
            0,
            int(self.worker_count),
        )

        self.hours_per_worker = non_negative(
            self.hours_per_worker
        )

        self.hourly_rate = non_negative(
            self.hourly_rate
        )

        self.overtime_hours_per_worker = non_negative(
            self.overtime_hours_per_worker
        )

        self.overtime_multiplier = non_negative(
            self.overtime_multiplier,
            1.5,
        )

        self.contractor_callout_cost = non_negative(
            self.contractor_callout_cost
        )

        self.confidence = clamp(
            self.confidence
        )

    @property
    def regular_hours(self) -> float:
        return (
            self.worker_count
            * self.hours_per_worker
        )

    @property
    def overtime_hours(self) -> float:
        return (
            self.worker_count
            * self.overtime_hours_per_worker
        )

    @property
    def regular_cost(self) -> float:
        return (
            self.regular_hours
            * self.hourly_rate
        )

    @property
    def overtime_cost(self) -> float:
        return (
            self.overtime_hours
            * self.hourly_rate
            * self.overtime_multiplier
        )

    @property
    def total(self) -> float:
        return sum(
            (
                self.regular_cost,
                self.overtime_cost,
                self.contractor_callout_cost,
            )
        )


# =========================================================
# Additional costs
# =========================================================

@dataclass(slots=True)
class RepairAdditionalCosts:
    diagnostics: float = 0.0
    inspection: float = 0.0

    materials: float = 0.0
    consumables: float = 0.0

    tooling: float = 0.0
    equipment_rental: float = 0.0

    contractor: float = 0.0
    oem_service: float = 0.0

    logistics: float = 0.0
    travel: float = 0.0
    accommodation: float = 0.0

    testing: float = 0.0
    commissioning: float = 0.0

    safety_controls: float = 0.0
    cleanup: float = 0.0

    permits: float = 0.0

    other: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "diagnostics",
            "inspection",
            "materials",
            "consumables",
            "tooling",
            "equipment_rental",
            "contractor",
            "oem_service",
            "logistics",
            "travel",
            "accommodation",
            "testing",
            "commissioning",
            "safety_controls",
            "cleanup",
            "permits",
            "other",
        ):
            setattr(
                self,
                name,
                non_negative(getattr(self, name)),
            )

    @property
    def total(self) -> float:
        return sum(
            (
                self.diagnostics,
                self.inspection,
                self.materials,
                self.consumables,
                self.tooling,
                self.equipment_rental,
                self.contractor,
                self.oem_service,
                self.logistics,
                self.travel,
                self.accommodation,
                self.testing,
                self.commissioning,
                self.safety_controls,
                self.cleanup,
                self.permits,
                self.other,
            )
        )


# =========================================================
# Repair economics
# =========================================================

@dataclass(slots=True)
class RepairEconomics:
    downtime_hours: float = 0.0

    direct_downtime_cost_per_hour: float = 0.0
    production_loss_per_hour: float = 0.0

    repair_success_probability: float = 1.0

    repeat_failure_probability: float = 0.0

    repeat_failure_cost: float = 0.0

    life_extension_years: float = 0.0

    post_repair_failure_probability: float | None = None

    def __post_init__(self) -> None:
        self.downtime_hours = non_negative(
            self.downtime_hours
        )

        self.direct_downtime_cost_per_hour = non_negative(
            self.direct_downtime_cost_per_hour
        )

        self.production_loss_per_hour = non_negative(
            self.production_loss_per_hour
        )

        self.repair_success_probability = clamp(
            self.repair_success_probability
        )

        self.repeat_failure_probability = clamp(
            self.repeat_failure_probability
        )

        self.repeat_failure_cost = non_negative(
            self.repeat_failure_cost
        )

        self.life_extension_years = non_negative(
            self.life_extension_years
        )

        if self.post_repair_failure_probability is not None:
            self.post_repair_failure_probability = clamp(
                self.post_repair_failure_probability
            )

    @property
    def direct_downtime_cost(self) -> float:
        return (
            self.downtime_hours
            * self.direct_downtime_cost_per_hour
        )

    @property
    def production_loss(self) -> float:
        return (
            self.downtime_hours
            * self.production_loss_per_hour
        )

    @property
    def expected_repeat_failure_cost(self) -> float:
        return (
            self.repeat_failure_probability
            * self.repeat_failure_cost
        )


# =========================================================
# Input
# =========================================================

@dataclass(slots=True)
class RepairCostInput:
    repair_type: RepairType

    parts: list[RepairPart] = field(
        default_factory=list
    )

    labor: list[RepairLabor] = field(
        default_factory=list
    )

    additional_costs: RepairAdditionalCosts = field(
        default_factory=RepairAdditionalCosts
    )

    economics: RepairEconomics = field(
        default_factory=RepairEconomics
    )

    urgency: RepairUrgency = RepairUrgency.ROUTINE

    contingency_rate: float = 0.0

    probability: float = 1.0

    asset_value: float | None = None
    replacement_cost: float | None = None

    asset_id: str | None = None
    component_id: str | None = None
    site_id: str | None = None

    incident_id: str | None = None
    work_order_id: str | None = None

    safety_critical: bool = False

    currency: str = "INR"

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.contingency_rate = non_negative(
            self.contingency_rate
        )

        self.probability = clamp(
            self.probability
        )

        if self.asset_value is not None:
            self.asset_value = non_negative(
                self.asset_value
            )

        if self.replacement_cost is not None:
            self.replacement_cost = non_negative(
                self.replacement_cost
            )


# =========================================================
# Breakdown
# =========================================================

@dataclass(slots=True)
class RepairCostBreakdown:
    parts: float = 0.0
    parts_credits: float = 0.0

    labor: float = 0.0

    diagnostics: float = 0.0
    inspection: float = 0.0

    materials: float = 0.0
    consumables: float = 0.0

    tooling: float = 0.0
    equipment_rental: float = 0.0

    contractor: float = 0.0
    oem_service: float = 0.0

    logistics: float = 0.0
    travel: float = 0.0
    accommodation: float = 0.0

    testing: float = 0.0
    commissioning: float = 0.0

    safety_controls: float = 0.0
    cleanup: float = 0.0
    permits: float = 0.0

    direct_downtime: float = 0.0
    production_loss: float = 0.0

    expected_repeat_failure: float = 0.0

    other: float = 0.0

    contingency: float = 0.0

    @property
    def total(self) -> float:
        return sum(
            (
                self.parts,
                self.labor,
                self.diagnostics,
                self.inspection,
                self.materials,
                self.consumables,
                self.tooling,
                self.equipment_rental,
                self.contractor,
                self.oem_service,
                self.logistics,
                self.travel,
                self.accommodation,
                self.testing,
                self.commissioning,
                self.safety_controls,
                self.cleanup,
                self.permits,
                self.direct_downtime,
                self.production_loss,
                self.expected_repeat_failure,
                self.other,
                self.contingency,
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
class RepairCostResult:
    calculation_id: str

    asset_id: str | None
    component_id: str | None
    site_id: str | None

    incident_id: str | None
    work_order_id: str | None

    repair_type: RepairType
    urgency: RepairUrgency

    parts_count: int
    labor_roles: int

    labor_hours: float

    direct_repair_cost: float
    downtime_cost: float
    production_loss: float

    expected_repeat_failure_cost: float

    contingency_cost: float

    total_repair_cost: float
    expected_repair_cost: float

    repair_to_asset_value_ratio: float | None
    repair_to_replacement_ratio: float | None

    repair_success_probability: float

    life_extension_years: float

    economic_risk: RepairEconomicRisk
    recommendation: RepairDecision

    confidence: float
    confidence_level: RepairConfidence

    breakdown: RepairCostBreakdown

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
class RepairCostConfig:
    moderate_replacement_ratio: float = 0.30
    high_replacement_ratio: float = 0.50
    critical_replacement_ratio: float = 0.75

    replace_ratio_threshold: float = 0.70

    minimum_success_probability: float = 0.60

    minimum_confidence: float = 0.50

    emergency_cost_multiplier: float = 1.15
    urgent_cost_multiplier: float = 1.08

    default_currency: str = "INR"


# =========================================================
# Calculator
# =========================================================

class RepairCostCalculator:

    def __init__(
        self,
        *,
        config: RepairCostConfig | None = None,
    ) -> None:
        self.config = (
            config
            or RepairCostConfig()
        )

    # =====================================================
    # Parts
    # =====================================================

    @staticmethod
    def calculate_parts(
        parts: Sequence[RepairPart],
    ) -> tuple[float, float]:

        gross = sum(
            part.gross_cost
            for part in parts
        )

        credits = sum(
            part.credits
            for part in parts
        )

        net = max(
            0.0,
            gross - credits,
        )

        return net, credits

    # =====================================================
    # Labor
    # =====================================================

    @staticmethod
    def calculate_labor(
        labor: Sequence[RepairLabor],
    ) -> tuple[float, float]:

        cost = sum(
            item.total
            for item in labor
        )

        hours = sum(
            item.regular_hours
            + item.overtime_hours
            for item in labor
        )

        return cost, hours

    # =====================================================
    # Urgency multiplier
    # =====================================================

    def urgency_multiplier(
        self,
        urgency: RepairUrgency,
    ) -> float:

        if urgency == RepairUrgency.EMERGENCY:
            return (
                self.config
                .emergency_cost_multiplier
            )

        if urgency == RepairUrgency.URGENT:
            return (
                self.config
                .urgent_cost_multiplier
            )

        return 1.0

    # =====================================================
    # Confidence
    # =====================================================

    @staticmethod
    def calculate_confidence(
        *,
        parts: Sequence[RepairPart],
        labor: Sequence[RepairLabor],
    ) -> tuple[float, RepairConfidence]:

        values: list[float] = []

        values.extend(
            part.confidence
            for part in parts
        )

        values.extend(
            item.confidence
            for item in labor
        )

        if not values:
            score = 0.50
        else:
            score = clamp(
                sum(values) / len(values)
            )

        if score >= 0.90:
            level = RepairConfidence.VERY_HIGH
        elif score >= 0.75:
            level = RepairConfidence.HIGH
        elif score >= 0.50:
            level = RepairConfidence.MODERATE
        else:
            level = RepairConfidence.LOW

        return round(score, 4), level

    # =====================================================
    # Economic risk
    # =====================================================

    def classify_economic_risk(
        self,
        *,
        repair_cost: float,
        replacement_cost: float | None,
    ) -> RepairEconomicRisk:

        if (
            replacement_cost is None
            or replacement_cost <= 0
        ):
            return RepairEconomicRisk.MODERATE

        ratio = (
            repair_cost
            / replacement_cost
        )

        if (
            ratio
            >= self.config
            .critical_replacement_ratio
        ):
            return RepairEconomicRisk.CRITICAL

        if (
            ratio
            >= self.config
            .high_replacement_ratio
        ):
            return RepairEconomicRisk.HIGH

        if (
            ratio
            >= self.config
            .moderate_replacement_ratio
        ):
            return RepairEconomicRisk.MODERATE

        return RepairEconomicRisk.LOW

    # =====================================================
    # Recommendation
    # =====================================================

    def recommend(
        self,
        *,
        repair: RepairCostInput,
        total_cost: float,
        confidence: float,
    ) -> RepairDecision:

        economics = repair.economics

        if confidence < self.config.minimum_confidence:
            return RepairDecision.HUMAN_REVIEW

        # Financial logic must not decide whether
        # a safety-critical asset can continue operating.
        if repair.safety_critical:
            return RepairDecision.HUMAN_REVIEW

        if (
            economics.repair_success_probability
            < self.config.minimum_success_probability
        ):
            if (
                repair.replacement_cost is not None
                and repair.replacement_cost > 0
            ):
                return RepairDecision.REPLACE

            return RepairDecision.HUMAN_REVIEW

        if (
            repair.replacement_cost is not None
            and repair.replacement_cost > 0
        ):
            ratio = (
                total_cost
                / repair.replacement_cost
            )

            if (
                ratio
                >= self.config
                .replace_ratio_threshold
            ):
                return RepairDecision.REPLACE

        if (
            repair.repair_type
            == RepairType.OVERHAUL
        ):
            return RepairDecision.OVERHAUL

        if (
            economics.repeat_failure_probability
            >= 0.40
        ):
            return RepairDecision.REPAIR_AND_MONITOR

        return RepairDecision.REPAIR

    # =====================================================
    # Main calculation
    # =====================================================

    def calculate(
        self,
        *,
        repair: RepairCostInput,
        metadata: Mapping[str, Any] | None = None,
    ) -> RepairCostResult:

        warnings: list[str] = []

        parts_cost, parts_credits = (
            self.calculate_parts(
                repair.parts
            )
        )

        labor_cost, labor_hours = (
            self.calculate_labor(
                repair.labor
            )
        )

        additional = repair.additional_costs
        economics = repair.economics

        direct_base = sum(
            (
                parts_cost,
                labor_cost,
                additional.total,
            )
        )

        urgency_multiplier = (
            self.urgency_multiplier(
                repair.urgency
            )
        )

        urgency_adjustment = max(
            0.0,
            direct_base
            * (urgency_multiplier - 1.0),
        )

        direct_repair_cost = (
            direct_base
            + urgency_adjustment
        )

        downtime_cost = (
            economics.direct_downtime_cost
        )

        production_loss = (
            economics.production_loss
        )

        repeat_failure_cost = (
            economics
            .expected_repeat_failure_cost
        )

        subtotal = sum(
            (
                direct_repair_cost,
                downtime_cost,
                production_loss,
                repeat_failure_cost,
            )
        )

        contingency = (
            subtotal
            * repair.contingency_rate
        )

        total = (
            subtotal
            + contingency
        )

        expected_cost = (
            total
            * repair.probability
        )

        asset_ratio = None
        replacement_ratio = None

        if (
            repair.asset_value is not None
            and repair.asset_value > 0
        ):
            asset_ratio = (
                total
                / repair.asset_value
            )

        if (
            repair.replacement_cost is not None
            and repair.replacement_cost > 0
        ):
            replacement_ratio = (
                total
                / repair.replacement_cost
            )

        confidence, confidence_level = (
            self.calculate_confidence(
                parts=repair.parts,
                labor=repair.labor,
            )
        )

        economic_risk = (
            self.classify_economic_risk(
                repair_cost=total,
                replacement_cost=(
                    repair.replacement_cost
                ),
            )
        )

        recommendation = self.recommend(
            repair=repair,
            total_cost=total,
            confidence=confidence,
        )

        breakdown = RepairCostBreakdown(
            parts=parts_cost,
            parts_credits=parts_credits,

            labor=labor_cost,

            diagnostics=additional.diagnostics,
            inspection=additional.inspection,

            materials=additional.materials,
            consumables=additional.consumables,

            tooling=additional.tooling,

            equipment_rental=(
                additional.equipment_rental
            ),

            contractor=additional.contractor,

            oem_service=(
                additional.oem_service
            ),

            logistics=additional.logistics,
            travel=additional.travel,
            accommodation=(
                additional.accommodation
            ),

            testing=additional.testing,

            commissioning=(
                additional.commissioning
            ),

            safety_controls=(
                additional.safety_controls
            ),

            cleanup=additional.cleanup,
            permits=additional.permits,

            direct_downtime=downtime_cost,

            production_loss=production_loss,

            expected_repeat_failure=(
                repeat_failure_cost
            ),

            other=(
                additional.other
                + urgency_adjustment
            ),

            contingency=contingency,
        )

        # ---------------------------------------------
        # Warnings
        # ---------------------------------------------

        if not repair.parts:
            warnings.append(
                "No repair parts were supplied."
            )

        if not repair.labor:
            warnings.append(
                "No repair labor resources were supplied."
            )

        unavailable_parts = [
            part.name
            for part in repair.parts
            if not part.available
        ]

        if unavailable_parts:
            warnings.append(
                "Unavailable repair parts: "
                + ", ".join(unavailable_parts)
            )

        if (
            repair.replacement_cost is not None
            and repair.replacement_cost > 0
            and replacement_ratio is not None
            and replacement_ratio >= (
                self.config
                .replace_ratio_threshold
            )
        ):
            warnings.append(
                "Repair lifecycle cost is high relative "
                "to replacement cost."
            )

        if (
            economics.repair_success_probability
            < self.config.minimum_success_probability
        ):
            warnings.append(
                "Repair success probability is below "
                "the configured threshold."
            )

        if (
            economics.repeat_failure_probability
            >= 0.40
        ):
            warnings.append(
                "High repeat-failure probability remains "
                "after the proposed repair."
            )

        if confidence < self.config.minimum_confidence:
            warnings.append(
                "Repair-cost estimate has low confidence."
            )

        if repair.safety_critical:
            warnings.append(
                "Safety-critical repair requires engineering/"
                "human authorization. Economic recommendation "
                "must not determine safe operation."
            )

        if (
            downtime_cost > 0
            and production_loss > 0
        ):
            warnings.append(
                "Direct downtime cost and production loss are "
                "both included. Confirm they represent separate "
                "economic components to avoid double counting."
            )

        if (
            labor_cost > 0
            and additional.contractor > 0
        ):
            warnings.append(
                "Labor and contractor costs are both present. "
                "Verify contractor labor is not duplicated."
            )

        if (
            parts_cost > 0
            and additional.materials > 0
        ):
            warnings.append(
                "Parts and additional material costs are both "
                "included; verify they do not overlap."
            )

        return RepairCostResult(
            calculation_id=new_id(),

            asset_id=repair.asset_id,
            component_id=repair.component_id,
            site_id=repair.site_id,

            incident_id=repair.incident_id,
            work_order_id=repair.work_order_id,

            repair_type=repair.repair_type,
            urgency=repair.urgency,

            parts_count=len(repair.parts),
            labor_roles=len(repair.labor),

            labor_hours=round(
                labor_hours,
                4,
            ),

            direct_repair_cost=round(
                direct_repair_cost,
                2,
            ),

            downtime_cost=round(
                downtime_cost,
                2,
            ),

            production_loss=round(
                production_loss,
                2,
            ),

            expected_repeat_failure_cost=round(
                repeat_failure_cost,
                2,
            ),

            contingency_cost=round(
                contingency,
                2,
            ),

            total_repair_cost=round(
                total,
                2,
            ),

            expected_repair_cost=round(
                expected_cost,
                2,
            ),

            repair_to_asset_value_ratio=(
                round(asset_ratio, 6)
                if asset_ratio is not None
                else None
            ),

            repair_to_replacement_ratio=(
                round(replacement_ratio, 6)
                if replacement_ratio is not None
                else None
            ),

            repair_success_probability=(
                economics
                .repair_success_probability
            ),

            life_extension_years=(
                economics.life_extension_years
            ),

            economic_risk=economic_risk,

            recommendation=recommendation,

            confidence=confidence,
            confidence_level=(
                confidence_level
            ),

            breakdown=breakdown,

            currency=(
                repair.currency
                or self.config.default_currency
            ),

            warnings=warnings,

            metadata={
                **repair.metadata,
                **dict(metadata or {}),
            },
        )

    # =====================================================
    # Repair vs replacement
    # =====================================================

    @staticmethod
    def compare_repair_replacement(
        *,
        repair_cost: float,
        replacement_cost: float,
        repair_success_probability: float = 1.0,
        residual_failure_probability: float = 0.0,
        residual_failure_cost: float = 0.0,
        replacement_residual_risk_cost: float = 0.0,
    ) -> dict[str, Any]:

        repair = non_negative(
            repair_cost
        )

        replacement = non_negative(
            replacement_cost
        )

        success = clamp(
            repair_success_probability
        )

        residual_probability = clamp(
            residual_failure_probability
        )

        residual_cost = non_negative(
            residual_failure_cost
        )

        replacement_risk = non_negative(
            replacement_residual_risk_cost
        )

        # Failure of repair is explicitly represented.
        failed_repair_exposure = (
            (1.0 - success)
            * residual_cost
        )

        residual_repair_exposure = (
            success
            * residual_probability
            * residual_cost
        )

        risk_adjusted_repair_cost = sum(
            (
                repair,
                failed_repair_exposure,
                residual_repair_exposure,
            )
        )

        risk_adjusted_replacement_cost = (
            replacement
            + replacement_risk
        )

        difference = (
            risk_adjusted_repair_cost
            - risk_adjusted_replacement_cost
        )

        if (
            risk_adjusted_repair_cost
            < risk_adjusted_replacement_cost
        ):
            decision = RepairDecision.REPAIR
        elif (
            risk_adjusted_repair_cost
            > risk_adjusted_replacement_cost
        ):
            decision = RepairDecision.REPLACE
        else:
            decision = RepairDecision.HUMAN_REVIEW

        return {
            "repair_cost": round(
                repair,
                2,
            ),

            "replacement_cost": round(
                replacement,
                2,
            ),

            "repair_success_probability": round(
                success,
                6,
            ),

            "failed_repair_exposure": round(
                failed_repair_exposure,
                2,
            ),

            "residual_repair_exposure": round(
                residual_repair_exposure,
                2,
            ),

            "risk_adjusted_repair_cost": round(
                risk_adjusted_repair_cost,
                2,
            ),

            "risk_adjusted_replacement_cost": round(
                risk_adjusted_replacement_cost,
                2,
            ),

            "repair_minus_replacement": round(
                difference,
                2,
            ),

            "economic_decision": decision.value,
        }

    # =====================================================
    # Future repair cost
    # =====================================================

    @staticmethod
    def future_repair_cost(
        *,
        current_repair_cost: float,
        annual_escalation_rate: float,
        years: float,
    ) -> float:

        current = non_negative(
            current_repair_cost
        )

        rate = finite(
            annual_escalation_rate
        )

        years = non_negative(
            years
        )

        base = 1.0 + rate

        if base <= 0:
            return 0.0

        return round(
            current
            * (base ** years),
            2,
        )

    # =====================================================
    # Expected repair cost
    # =====================================================

    @staticmethod
    def expected_repair_cost(
        *,
        repair_cost: float,
        repair_probability: float,
    ) -> float:

        return round(
            non_negative(repair_cost)
            * clamp(repair_probability),
            2,
        )

    # =====================================================
    # Batch
    # =====================================================

    def calculate_many(
        self,
        repairs: Sequence[
            RepairCostInput
        ],
    ) -> list[RepairCostResult]:

        return [
            self.calculate(repair=repair)
            for repair in repairs
        ]

    # =====================================================
    # Aggregate
    # =====================================================

    @staticmethod
    def aggregate(
        results: Sequence[
            RepairCostResult
        ],
    ) -> dict[str, Any]:

        if not results:
            return {
                "repair_count": 0,
                "total_repair_cost": 0.0,
                "expected_repair_cost": 0.0,
                "labor_hours": 0.0,
                "currency": None,
            }

        currencies = {
            result.currency
            for result in results
        }

        total = sum(
            result.total_repair_cost
            for result in results
        )

        expected = sum(
            result.expected_repair_cost
            for result in results
        )

        labor_hours = sum(
            result.labor_hours
            for result in results
        )

        return {
            "repair_count": len(results),

            "total_repair_cost": round(
                total,
                2,
            ),

            "expected_repair_cost": round(
                expected,
                2,
            ),

            "labor_hours": round(
                labor_hours,
                4,
            ),

            "average_repair_cost": round(
                total / len(results),
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

repair_cost_calculator = (
    RepairCostCalculator()
)


# =========================================================
# Agent-facing helpers
# =========================================================

def calculate_repair_cost(
    *,
    repair: RepairCostInput,
) -> RepairCostResult:

    return (
        repair_cost_calculator
        .calculate(repair=repair)
    )


def calculate_expected_repair_cost(
    *,
    repair_cost: float,
    repair_probability: float,
) -> float:

    return (
        repair_cost_calculator
        .expected_repair_cost(
            repair_cost=repair_cost,
            repair_probability=(
                repair_probability
            ),
        )
    )


def calculate_future_repair_cost(
    *,
    current_repair_cost: float,
    annual_escalation_rate: float,
    years: float,
) -> float:

    return (
        repair_cost_calculator
        .future_repair_cost(
            current_repair_cost=(
                current_repair_cost
            ),
            annual_escalation_rate=(
                annual_escalation_rate
            ),
            years=years,
        )
    )


def compare_repair_vs_replacement(
    *,
    repair_cost: float,
    replacement_cost: float,
    repair_success_probability: float = 1.0,
    residual_failure_probability: float = 0.0,
    residual_failure_cost: float = 0.0,
    replacement_residual_risk_cost: float = 0.0,
) -> dict[str, Any]:

    return (
        repair_cost_calculator
        .compare_repair_replacement(
            repair_cost=repair_cost,
            replacement_cost=replacement_cost,
            repair_success_probability=(
                repair_success_probability
            ),
            residual_failure_probability=(
                residual_failure_probability
            ),
            residual_failure_cost=(
                residual_failure_cost
            ),
            replacement_residual_risk_cost=(
                replacement_residual_risk_cost
            ),
        )
    )


# =========================================================
# Exports
# =========================================================

__all__ = [
    "RepairType",
    "RepairCostCategory",
    "RepairUrgency",
    "RepairEconomicRisk",
    "RepairConfidence",
    "RepairDecision",

    "RepairPart",
    "RepairLabor",
    "RepairAdditionalCosts",
    "RepairEconomics",
    "RepairCostInput",

    "RepairCostBreakdown",
    "RepairCostResult",

    "RepairCostConfig",
    "RepairCostCalculator",

    "repair_cost_calculator",

    "calculate_repair_cost",
    "calculate_expected_repair_cost",
    "calculate_future_repair_cost",
    "compare_repair_vs_replacement",
]