# app/agents/tools/failure_cost.py

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

class FailureType(str, Enum):
    MECHANICAL = "mechanical"
    ELECTRICAL = "electrical"
    HYDRAULIC = "hydraulic"
    PNEUMATIC = "pneumatic"
    THERMAL = "thermal"
    BEARING = "bearing"
    MOTOR = "motor"
    GEARBOX = "gearbox"
    PUMP = "pump"
    COMPRESSOR = "compressor"
    VALVE = "valve"
    SENSOR = "sensor"
    CONTROL_SYSTEM = "control_system"
    STRUCTURAL = "structural"
    LUBRICATION = "lubrication"
    COOLING = "cooling"
    POWER_SUPPLY = "power_supply"
    PROCESS = "process"
    SAFETY_SYSTEM = "safety_system"
    UNKNOWN = "unknown"


class FailureSeverity(str, Enum):
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CRITICAL = "critical"
    CATASTROPHIC = "catastrophic"


class FailureCostCategory(str, Enum):
    DIAGNOSTIC = "diagnostic"
    INSPECTION = "inspection"

    PARTS = "parts"
    LABOR = "labor"
    REPAIR = "repair"
    REPLACEMENT = "replacement"

    CONTRACTOR = "contractor"
    LOGISTICS = "logistics"

    DOWNTIME = "downtime"
    PRODUCTION_LOSS = "production_loss"

    SCRAP = "scrap"
    REWORK = "rework"
    QUALITY = "quality"

    ENERGY = "energy"

    SAFETY = "safety"
    EMERGENCY = "emergency"
    ENVIRONMENTAL = "environmental"

    PENALTY = "penalty"
    REGULATORY = "regulatory"

    RECOVERY = "recovery"
    RESTART = "restart"

    OTHER = "other"


class FailureCostBasis(str, Enum):
    COMPONENTIZED = "componentized"
    ALL_IN = "all_in"


class FailureEconomicImpact(str, Enum):
    NEGLIGIBLE = "negligible"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    CATASTROPHIC = "catastrophic"


class CostConfidence(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


# =========================================================
# Repair / replacement
# =========================================================

@dataclass(slots=True)
class FailureRepairCost:
    diagnostic_cost: float = 0.0
    inspection_cost: float = 0.0

    parts_cost: float = 0.0
    labor_cost: float = 0.0

    contractor_cost: float = 0.0
    tooling_cost: float = 0.0

    logistics_cost: float = 0.0
    expedited_shipping_cost: float = 0.0

    consumables_cost: float = 0.0

    testing_cost: float = 0.0
    recommissioning_cost: float = 0.0

    other_cost: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "diagnostic_cost",
            "inspection_cost",
            "parts_cost",
            "labor_cost",
            "contractor_cost",
            "tooling_cost",
            "logistics_cost",
            "expedited_shipping_cost",
            "consumables_cost",
            "testing_cost",
            "recommissioning_cost",
            "other_cost",
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
                self.diagnostic_cost,
                self.inspection_cost,
                self.parts_cost,
                self.labor_cost,
                self.contractor_cost,
                self.tooling_cost,
                self.logistics_cost,
                self.expedited_shipping_cost,
                self.consumables_cost,
                self.testing_cost,
                self.recommissioning_cost,
                self.other_cost,
            )
        )


@dataclass(slots=True)
class FailureReplacementCost:
    replacement_asset_cost: float = 0.0

    removal_cost: float = 0.0
    installation_cost: float = 0.0

    commissioning_cost: float = 0.0
    engineering_cost: float = 0.0

    logistics_cost: float = 0.0

    temporary_equipment_cost: float = 0.0

    disposal_cost: float = 0.0

    training_cost: float = 0.0

    salvage_value: float = 0.0

    warranty_recovery: float = 0.0

    other_cost: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "replacement_asset_cost",
            "removal_cost",
            "installation_cost",
            "commissioning_cost",
            "engineering_cost",
            "logistics_cost",
            "temporary_equipment_cost",
            "disposal_cost",
            "training_cost",
            "salvage_value",
            "warranty_recovery",
            "other_cost",
        ):
            setattr(
                self,
                name,
                non_negative(getattr(self, name)),
            )

    @property
    def gross_total(self) -> float:
        return sum(
            (
                self.replacement_asset_cost,
                self.removal_cost,
                self.installation_cost,
                self.commissioning_cost,
                self.engineering_cost,
                self.logistics_cost,
                self.temporary_equipment_cost,
                self.disposal_cost,
                self.training_cost,
                self.other_cost,
            )
        )

    @property
    def recovery_credit(self) -> float:
        return (
            self.salvage_value
            + self.warranty_recovery
        )

    @property
    def net_total(self) -> float:
        return max(
            0.0,
            self.gross_total
            - self.recovery_credit,
        )


# =========================================================
# Downtime / production
# =========================================================

@dataclass(slots=True)
class FailureDowntimeCost:
    downtime_hours: float = 0.0

    downtime_overhead_per_hour: float = 0.0

    production_units_per_hour: float = 0.0

    contribution_margin_per_unit: float = 0.0

    recoverable_production_fraction: float = 0.0

    restart_cost: float = 0.0

    startup_energy_cost: float = 0.0

    startup_scrap_cost: float = 0.0

    overtime_recovery_cost: float = 0.0

    other_cost: float = 0.0

    def __post_init__(self) -> None:
        self.downtime_hours = non_negative(
            self.downtime_hours
        )

        self.downtime_overhead_per_hour = non_negative(
            self.downtime_overhead_per_hour
        )

        self.production_units_per_hour = non_negative(
            self.production_units_per_hour
        )

        self.contribution_margin_per_unit = non_negative(
            self.contribution_margin_per_unit
        )

        self.recoverable_production_fraction = clamp(
            self.recoverable_production_fraction
        )

        for name in (
            "restart_cost",
            "startup_energy_cost",
            "startup_scrap_cost",
            "overtime_recovery_cost",
            "other_cost",
        ):
            setattr(
                self,
                name,
                non_negative(getattr(self, name)),
            )

    @property
    def direct_downtime_cost(self) -> float:
        return (
            self.downtime_hours
            * self.downtime_overhead_per_hour
        )

    @property
    def lost_units(self) -> float:
        return (
            self.downtime_hours
            * self.production_units_per_hour
        )

    @property
    def gross_production_loss(self) -> float:
        return (
            self.lost_units
            * self.contribution_margin_per_unit
        )

    @property
    def production_recovery_credit(self) -> float:
        return (
            self.gross_production_loss
            * self.recoverable_production_fraction
        )

    @property
    def net_production_loss(self) -> float:
        return max(
            0.0,
            self.gross_production_loss
            - self.production_recovery_credit,
        )

    @property
    def restart_total(self) -> float:
        return sum(
            (
                self.restart_cost,
                self.startup_energy_cost,
                self.startup_scrap_cost,
                self.overtime_recovery_cost,
                self.other_cost,
            )
        )

    @property
    def total(self) -> float:
        return sum(
            (
                self.direct_downtime_cost,
                self.net_production_loss,
                self.restart_total,
            )
        )


# =========================================================
# Quality loss
# =========================================================

@dataclass(slots=True)
class FailureQualityCost:
    scrap_units: float = 0.0
    scrap_cost_per_unit: float = 0.0

    rework_units: float = 0.0
    rework_cost_per_unit: float = 0.0

    inspection_cost: float = 0.0

    customer_return_cost: float = 0.0
    warranty_claim_cost: float = 0.0

    quality_penalty_cost: float = 0.0

    other_cost: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "scrap_units",
            "scrap_cost_per_unit",
            "rework_units",
            "rework_cost_per_unit",
            "inspection_cost",
            "customer_return_cost",
            "warranty_claim_cost",
            "quality_penalty_cost",
            "other_cost",
        ):
            setattr(
                self,
                name,
                non_negative(getattr(self, name)),
            )

    @property
    def scrap_cost(self) -> float:
        return (
            self.scrap_units
            * self.scrap_cost_per_unit
        )

    @property
    def rework_cost(self) -> float:
        return (
            self.rework_units
            * self.rework_cost_per_unit
        )

    @property
    def total(self) -> float:
        return sum(
            (
                self.scrap_cost,
                self.rework_cost,
                self.inspection_cost,
                self.customer_return_cost,
                self.warranty_claim_cost,
                self.quality_penalty_cost,
                self.other_cost,
            )
        )


# =========================================================
# Energy impact
# =========================================================

@dataclass(slots=True)
class FailureEnergyCost:
    wasted_energy_kwh: float = 0.0

    energy_rate_per_kwh: float = 0.0

    restart_energy_kwh: float = 0.0

    temporary_equipment_kwh: float = 0.0

    demand_penalty_cost: float = 0.0

    other_cost: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "wasted_energy_kwh",
            "energy_rate_per_kwh",
            "restart_energy_kwh",
            "temporary_equipment_kwh",
            "demand_penalty_cost",
            "other_cost",
        ):
            setattr(
                self,
                name,
                non_negative(getattr(self, name)),
            )

    @property
    def energy_cost(self) -> float:
        return (
            (
                self.wasted_energy_kwh
                + self.restart_energy_kwh
                + self.temporary_equipment_kwh
            )
            * self.energy_rate_per_kwh
        )

    @property
    def total(self) -> float:
        return (
            self.energy_cost
            + self.demand_penalty_cost
            + self.other_cost
        )


# =========================================================
# Safety / emergency impact
# =========================================================

@dataclass(slots=True)
class FailureSafetyCost:
    emergency_response_cost: float = 0.0

    evacuation_cost: float = 0.0

    medical_response_cost: float = 0.0

    safety_investigation_cost: float = 0.0

    incident_management_cost: float = 0.0

    containment_cost: float = 0.0

    cleanup_cost: float = 0.0

    environmental_cost: float = 0.0

    regulatory_cost: float = 0.0

    legal_cost: float = 0.0

    other_cost: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "emergency_response_cost",
            "evacuation_cost",
            "medical_response_cost",
            "safety_investigation_cost",
            "incident_management_cost",
            "containment_cost",
            "cleanup_cost",
            "environmental_cost",
            "regulatory_cost",
            "legal_cost",
            "other_cost",
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
                self.emergency_response_cost,
                self.evacuation_cost,
                self.medical_response_cost,
                self.safety_investigation_cost,
                self.incident_management_cost,
                self.containment_cost,
                self.cleanup_cost,
                self.environmental_cost,
                self.regulatory_cost,
                self.legal_cost,
                self.other_cost,
            )
        )


# =========================================================
# Other business impact
# =========================================================

@dataclass(slots=True)
class FailureBusinessCost:
    customer_penalty_cost: float = 0.0

    contract_penalty_cost: float = 0.0

    expedited_fulfillment_cost: float = 0.0

    alternate_production_cost: float = 0.0

    supply_chain_cost: float = 0.0

    business_interruption_cost: float = 0.0

    reputation_proxy_cost: float = 0.0

    other_cost: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "customer_penalty_cost",
            "contract_penalty_cost",
            "expedited_fulfillment_cost",
            "alternate_production_cost",
            "supply_chain_cost",
            "business_interruption_cost",
            "reputation_proxy_cost",
            "other_cost",
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
                self.customer_penalty_cost,
                self.contract_penalty_cost,
                self.expedited_fulfillment_cost,
                self.alternate_production_cost,
                self.supply_chain_cost,
                self.business_interruption_cost,
                self.reputation_proxy_cost,
                self.other_cost,
            )
        )


# =========================================================
# Recovery credits
# =========================================================

@dataclass(slots=True)
class FailureRecoveryCredits:
    insurance_recovery: float = 0.0

    supplier_recovery: float = 0.0

    warranty_recovery: float = 0.0

    salvage_recovery: float = 0.0

    production_recovery: float = 0.0

    other_recovery: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "insurance_recovery",
            "supplier_recovery",
            "warranty_recovery",
            "salvage_recovery",
            "production_recovery",
            "other_recovery",
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
                self.insurance_recovery,
                self.supplier_recovery,
                self.warranty_recovery,
                self.salvage_recovery,
                self.production_recovery,
                self.other_recovery,
            )
        )


# =========================================================
# Failure cost input
# =========================================================

@dataclass(slots=True)
class FailureCostInput:
    failure_type: FailureType

    severity: FailureSeverity = FailureSeverity.MODERATE

    repair: FailureRepairCost = field(
        default_factory=FailureRepairCost
    )

    replacement: FailureReplacementCost = field(
        default_factory=FailureReplacementCost
    )

    downtime: FailureDowntimeCost = field(
        default_factory=FailureDowntimeCost
    )

    quality: FailureQualityCost = field(
        default_factory=FailureQualityCost
    )

    energy: FailureEnergyCost = field(
        default_factory=FailureEnergyCost
    )

    safety: FailureSafetyCost = field(
        default_factory=FailureSafetyCost
    )

    business: FailureBusinessCost = field(
        default_factory=FailureBusinessCost
    )

    recoveries: FailureRecoveryCredits = field(
        default_factory=FailureRecoveryCredits
    )

    use_repair_cost: bool = True
    use_replacement_cost: bool = False

    cost_basis: FailureCostBasis = (
        FailureCostBasis.COMPONENTIZED
    )

    all_in_failure_cost: float | None = None

    confidence: float = 1.0

    currency: str = "INR"

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.confidence = clamp(
            self.confidence
        )

        if self.all_in_failure_cost is not None:
            self.all_in_failure_cost = non_negative(
                self.all_in_failure_cost
            )


# =========================================================
# Cost breakdown
# =========================================================

@dataclass(slots=True)
class FailureCostBreakdown:
    repair: float = 0.0
    replacement: float = 0.0

    downtime: float = 0.0
    production_loss: float = 0.0
    restart: float = 0.0

    quality: float = 0.0
    energy: float = 0.0

    safety_emergency: float = 0.0

    business: float = 0.0

    recovery_credits: float = 0.0

    other: float = 0.0

    @property
    def gross_total(self) -> float:
        return sum(
            (
                self.repair,
                self.replacement,
                self.downtime,
                self.production_loss,
                self.restart,
                self.quality,
                self.energy,
                self.safety_emergency,
                self.business,
                self.other,
            )
        )

    @property
    def net_total(self) -> float:
        return max(
            0.0,
            self.gross_total
            - self.recovery_credits,
        )

    def to_dict(self) -> dict[str, Any]:
        result = json_safe(self)

        result["gross_total"] = round(
            self.gross_total,
            2,
        )

        result["net_total"] = round(
            self.net_total,
            2,
        )

        return result


# =========================================================
# Result
# =========================================================

@dataclass(slots=True)
class FailureCostResult:
    calculation_id: str

    asset_id: str | None
    component_id: str | None
    site_id: str | None

    failure_type: FailureType
    severity: FailureSeverity

    gross_failure_cost: float

    recovery_credit: float

    net_failure_cost: float

    breakdown: FailureCostBreakdown

    asset_value: float | None

    failure_cost_to_asset_value_ratio: float | None

    economic_impact: FailureEconomicImpact

    confidence: float
    confidence_level: CostConfidence

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
# Scenario
# =========================================================

@dataclass(slots=True)
class FailureCostScenario:
    name: str

    probability: float

    failure: FailureCostInput

    scenario_weight: float = 1.0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.probability = clamp(
            self.probability
        )

        self.scenario_weight = non_negative(
            self.scenario_weight
        )


@dataclass(slots=True)
class FailureCostScenarioResult:
    name: str

    probability: float

    failure_cost: float

    expected_failure_cost: float

    scenario_weight: float

    weighted_expected_cost: float

    currency: str

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# =========================================================
# Config
# =========================================================

@dataclass(slots=True)
class FailureCostConfig:
    low_impact_ratio: float = 0.05

    moderate_impact_ratio: float = 0.15

    high_impact_ratio: float = 0.30

    critical_impact_ratio: float = 0.60

    catastrophic_impact_ratio: float = 1.00

    minimum_confidence: float = 0.50

    default_currency: str = "INR"

    warn_double_count: bool = True


# =========================================================
# Calculator
# =========================================================

class FailureCostCalculator:

    def __init__(
        self,
        *,
        config: FailureCostConfig | None = None,
    ) -> None:
        self.config = (
            config
            or FailureCostConfig()
        )

    # =====================================================
    # Breakdown
    # =====================================================

    @staticmethod
    def calculate_breakdown(
        failure: FailureCostInput,
    ) -> FailureCostBreakdown:

        if (
            failure.cost_basis
            == FailureCostBasis.ALL_IN
            and failure.all_in_failure_cost
            is not None
        ):
            return FailureCostBreakdown(
                other=(
                    failure.all_in_failure_cost
                ),
                recovery_credits=(
                    failure.recoveries.total
                ),
            )

        repair_cost = (
            failure.repair.total
            if failure.use_repair_cost
            else 0.0
        )

        replacement_cost = (
            failure.replacement.gross_total
            if failure.use_replacement_cost
            else 0.0
        )

        replacement_credit = (
            failure.replacement.recovery_credit
            if failure.use_replacement_cost
            else 0.0
        )

        total_recoveries = (
            failure.recoveries.total
            + replacement_credit
        )

        return FailureCostBreakdown(
            repair=repair_cost,

            replacement=replacement_cost,

            downtime=(
                failure.downtime
                .direct_downtime_cost
            ),

            production_loss=(
                failure.downtime
                .net_production_loss
            ),

            restart=(
                failure.downtime.restart_total
            ),

            quality=failure.quality.total,

            energy=failure.energy.total,

            safety_emergency=(
                failure.safety.total
            ),

            business=failure.business.total,

            recovery_credits=(
                total_recoveries
            ),
        )

    # =====================================================
    # Confidence
    # =====================================================

    @staticmethod
    def confidence_level(
        confidence: float,
    ) -> CostConfidence:

        confidence = clamp(confidence)

        if confidence >= 0.90:
            return CostConfidence.VERY_HIGH

        if confidence >= 0.75:
            return CostConfidence.HIGH

        if confidence >= 0.50:
            return CostConfidence.MODERATE

        return CostConfidence.LOW

    # =====================================================
    # Economic impact
    # =====================================================

    def classify_impact(
        self,
        *,
        net_failure_cost: float,
        asset_value: float | None,
        severity: FailureSeverity,
    ) -> FailureEconomicImpact:

        severity_floor = {
            FailureSeverity.MINOR:
                FailureEconomicImpact.NEGLIGIBLE,

            FailureSeverity.MODERATE:
                FailureEconomicImpact.LOW,

            FailureSeverity.MAJOR:
                FailureEconomicImpact.MODERATE,

            FailureSeverity.CRITICAL:
                FailureEconomicImpact.HIGH,

            FailureSeverity.CATASTROPHIC:
                FailureEconomicImpact.CRITICAL,
        }[severity]

        rank = {
            FailureEconomicImpact.NEGLIGIBLE: 0,
            FailureEconomicImpact.LOW: 1,
            FailureEconomicImpact.MODERATE: 2,
            FailureEconomicImpact.HIGH: 3,
            FailureEconomicImpact.CRITICAL: 4,
            FailureEconomicImpact.CATASTROPHIC: 5,
        }

        ratio_impact = (
            FailureEconomicImpact.NEGLIGIBLE
        )

        if (
            asset_value is not None
            and asset_value > 0
        ):
            ratio = (
                net_failure_cost
                / asset_value
            )

            if (
                ratio
                >= self.config
                .catastrophic_impact_ratio
            ):
                ratio_impact = (
                    FailureEconomicImpact
                    .CATASTROPHIC
                )

            elif (
                ratio
                >= self.config
                .critical_impact_ratio
            ):
                ratio_impact = (
                    FailureEconomicImpact.CRITICAL
                )

            elif (
                ratio
                >= self.config
                .high_impact_ratio
            ):
                ratio_impact = (
                    FailureEconomicImpact.HIGH
                )

            elif (
                ratio
                >= self.config
                .moderate_impact_ratio
            ):
                ratio_impact = (
                    FailureEconomicImpact.MODERATE
                )

            elif (
                ratio
                >= self.config.low_impact_ratio
            ):
                ratio_impact = (
                    FailureEconomicImpact.LOW
                )

        return max(
            (
                severity_floor,
                ratio_impact,
            ),
            key=lambda item: rank[item],
        )

    # =====================================================
    # Main calculation
    # =====================================================

    def calculate(
        self,
        *,
        failure: FailureCostInput,
        asset_id: str | None = None,
        component_id: str | None = None,
        site_id: str | None = None,
        asset_value: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> FailureCostResult:

        warnings: list[str] = []

        breakdown = self.calculate_breakdown(
            failure
        )

        gross_cost = breakdown.gross_total

        recovery_credit = (
            breakdown.recovery_credits
        )

        net_cost = breakdown.net_total

        normalized_asset_value = (
            non_negative(asset_value)
            if asset_value is not None
            else None
        )

        ratio = None

        if (
            normalized_asset_value is not None
            and normalized_asset_value > 0
        ):
            ratio = (
                net_cost
                / normalized_asset_value
            )

        impact = self.classify_impact(
            net_failure_cost=net_cost,
            asset_value=normalized_asset_value,
            severity=failure.severity,
        )

        confidence_level = (
            self.confidence_level(
                failure.confidence
            )
        )

        # ---------------------------------------------
        # Validation / warnings
        # ---------------------------------------------

        if gross_cost <= 0:
            warnings.append(
                "Failure consequence cost is zero "
                "or unavailable."
            )

        if (
            failure.confidence
            < self.config.minimum_confidence
        ):
            warnings.append(
                "Failure-cost estimate has low "
                "confidence."
            )

        if recovery_credit > gross_cost:
            warnings.append(
                "Recovery credits exceed gross "
                "failure cost; net cost was floored "
                "at zero."
            )

        if (
            failure.use_repair_cost
            and failure.use_replacement_cost
        ):
            warnings.append(
                "Repair and replacement costs are "
                "both enabled. Confirm this models "
                "a repair-then-replace sequence and "
                "is not double counting alternatives."
            )

        if (
            failure.cost_basis
            == FailureCostBasis.ALL_IN
            and failure.all_in_failure_cost is None
        ):
            warnings.append(
                "ALL_IN cost basis selected without "
                "an all_in_failure_cost value."
            )

        if (
            failure.cost_basis
            == FailureCostBasis.COMPONENTIZED
            and self.config.warn_double_count
        ):
            if (
                failure.downtime
                .downtime_overhead_per_hour > 0
                and failure.downtime
                .contribution_margin_per_unit > 0
            ):
                warnings.append(
                    "Downtime overhead and production "
                    "loss are both included. Ensure "
                    "the downtime rate excludes lost "
                    "production margin."
                )

            if (
                failure.safety
                .emergency_response_cost > 0
                and failure.safety
                .incident_management_cost > 0
            ):
                warnings.append(
                    "Verify emergency-response and "
                    "incident-management costs are "
                    "separate cost categories."
                )

            if (
                failure.business
                .business_interruption_cost > 0
                and (
                    failure.downtime
                    .direct_downtime_cost > 0
                    or failure.downtime
                    .net_production_loss > 0
                )
            ):
                warnings.append(
                    "Business interruption is included "
                    "with downtime/production losses. "
                    "Confirm these values do not "
                    "overlap."
                )

            if (
                failure.downtime
                .startup_energy_cost > 0
                and failure.energy
                .restart_energy_kwh > 0
            ):
                warnings.append(
                    "Restart energy appears in both "
                    "downtime and energy components. "
                    "Use only one source for the same "
                    "cost."
                )

        if (
            failure.severity
            in {
                FailureSeverity.CRITICAL,
                FailureSeverity.CATASTROPHIC,
            }
            and (
                failure.safety.total > 0
            )
        ):
            warnings.append(
                "Safety-critical failure detected. "
                "Financial cost must not override "
                "required safety controls."
            )

        return FailureCostResult(
            calculation_id=new_id(),

            asset_id=asset_id,
            component_id=component_id,
            site_id=site_id,

            failure_type=failure.failure_type,

            severity=failure.severity,

            gross_failure_cost=round(
                gross_cost,
                2,
            ),

            recovery_credit=round(
                recovery_credit,
                2,
            ),

            net_failure_cost=round(
                net_cost,
                2,
            ),

            breakdown=breakdown,

            asset_value=(
                round(
                    normalized_asset_value,
                    2,
                )
                if normalized_asset_value
                is not None
                else None
            ),

            failure_cost_to_asset_value_ratio=(
                round(ratio, 6)
                if ratio is not None
                else None
            ),

            economic_impact=impact,

            confidence=round(
                failure.confidence,
                4,
            ),

            confidence_level=(
                confidence_level
            ),

            currency=(
                failure.currency
                or self.config.default_currency
            ),

            warnings=warnings,

            metadata={
                **failure.metadata,
                **dict(metadata or {}),
            },
        )

    # =====================================================
    # Expected failure cost
    # =====================================================

    def expected_failure_cost(
        self,
        *,
        probability: float,
        failure_cost: float,
    ) -> float:

        return round(
            clamp(probability)
            * non_negative(failure_cost),
            2,
        )

    # =====================================================
    # Expected cost from result
    # =====================================================

    def expected_from_result(
        self,
        *,
        probability: float,
        result: FailureCostResult,
    ) -> float:

        return self.expected_failure_cost(
            probability=probability,
            failure_cost=(
                result.net_failure_cost
            ),
        )

    # =====================================================
    # Cost per downtime hour
    # =====================================================

    @staticmethod
    def cost_per_downtime_hour(
        *,
        result: FailureCostResult,
        downtime_hours: float,
    ) -> float | None:

        hours = non_negative(
            downtime_hours
        )

        if hours <= 0:
            return None

        return round(
            result.net_failure_cost / hours,
            2,
        )

    # =====================================================
    # Repair vs replacement
    # =====================================================

    @staticmethod
    def repair_replacement_comparison(
        *,
        repair: FailureRepairCost,
        replacement: FailureReplacementCost,
        repair_success_probability: float = 1.0,
        post_repair_failure_cost: float = 0.0,
    ) -> dict[str, Any]:

        success = clamp(
            repair_success_probability
        )

        repair_cost = repair.total

        replacement_cost = (
            replacement.net_total
        )

        # Expected residual failure exposure if repair
        # does not succeed.
        residual_failure_cost = (
            (1.0 - success)
            * non_negative(
                post_repair_failure_cost
            )
        )

        risk_adjusted_repair_cost = (
            repair_cost
            + residual_failure_cost
        )

        if (
            risk_adjusted_repair_cost
            < replacement_cost
        ):
            recommendation = "repair"

        elif (
            replacement_cost
            < risk_adjusted_repair_cost
        ):
            recommendation = "replace"

        else:
            recommendation = "equivalent"

        return {
            "repair_cost": round(
                repair_cost,
                2,
            ),

            "repair_success_probability": round(
                success,
                4,
            ),

            "residual_failure_cost": round(
                residual_failure_cost,
                2,
            ),

            "risk_adjusted_repair_cost": round(
                risk_adjusted_repair_cost,
                2,
            ),

            "replacement_cost": round(
                replacement_cost,
                2,
            ),

            "difference": round(
                abs(
                    risk_adjusted_repair_cost
                    - replacement_cost
                ),
                2,
            ),

            "recommendation": recommendation,
        }

    # =====================================================
    # Scenario analysis
    # =====================================================

    def scenario_analysis(
        self,
        *,
        scenarios: Sequence[
            FailureCostScenario
        ],
        asset_id: str | None = None,
        asset_value: float | None = None,
    ) -> list[FailureCostScenarioResult]:

        results: list[
            FailureCostScenarioResult
        ] = []

        for scenario in scenarios:
            calculated = self.calculate(
                failure=scenario.failure,
                asset_id=asset_id,
                asset_value=asset_value,
            )

            expected = (
                scenario.probability
                * calculated.net_failure_cost
            )

            weighted = (
                expected
                * scenario.scenario_weight
            )

            results.append(
                FailureCostScenarioResult(
                    name=scenario.name,

                    probability=round(
                        scenario.probability,
                        6,
                    ),

                    failure_cost=round(
                        calculated.net_failure_cost,
                        2,
                    ),

                    expected_failure_cost=round(
                        expected,
                        2,
                    ),

                    scenario_weight=round(
                        scenario.scenario_weight,
                        4,
                    ),

                    weighted_expected_cost=round(
                        weighted,
                        2,
                    ),

                    currency=(
                        calculated.currency
                    ),
                )
            )

        return results

    # =====================================================
    # Weighted scenario exposure
    # =====================================================

    def weighted_scenario_exposure(
        self,
        *,
        scenarios: Sequence[
            FailureCostScenario
        ],
        asset_id: str | None = None,
        asset_value: float | None = None,
    ) -> float:

        results = self.scenario_analysis(
            scenarios=scenarios,
            asset_id=asset_id,
            asset_value=asset_value,
        )

        if not results:
            return 0.0

        total_weight = sum(
            item.scenario_weight
            for item in results
        )

        if total_weight <= 0:
            return 0.0

        return round(
            sum(
                item.weighted_expected_cost
                for item in results
            )
            / total_weight,
            2,
        )

    # =====================================================
    # Failure cost ratio
    # =====================================================

    @staticmethod
    def failure_cost_ratio(
        *,
        failure_cost: float,
        asset_value: float,
    ) -> float | None:

        value = non_negative(
            asset_value
        )

        if value <= 0:
            return None

        return round(
            non_negative(failure_cost)
            / value,
            6,
        )

    # =====================================================
    # Batch
    # =====================================================

    def calculate_many(
        self,
        failures: Sequence[
            FailureCostInput
        ],
    ) -> list[FailureCostResult]:

        return [
            self.calculate(
                failure=failure
            )
            for failure in failures
        ]

    # =====================================================
    # Portfolio aggregate
    # =====================================================

    @staticmethod
    def aggregate(
        results: Sequence[
            FailureCostResult
        ],
    ) -> dict[str, Any]:

        if not results:
            return {
                "failure_count": 0,
                "gross_failure_cost": 0.0,
                "recovery_credit": 0.0,
                "net_failure_cost": 0.0,
                "maximum_failure_cost": 0.0,
                "average_failure_cost": 0.0,
                "currency": None,
            }

        gross = sum(
            item.gross_failure_cost
            for item in results
        )

        recoveries = sum(
            item.recovery_credit
            for item in results
        )

        net = sum(
            item.net_failure_cost
            for item in results
        )

        maximum = max(
            item.net_failure_cost
            for item in results
        )

        currencies = {
            item.currency
            for item in results
        }

        impact_rank = {
            FailureEconomicImpact.NEGLIGIBLE: 0,
            FailureEconomicImpact.LOW: 1,
            FailureEconomicImpact.MODERATE: 2,
            FailureEconomicImpact.HIGH: 3,
            FailureEconomicImpact.CRITICAL: 4,
            FailureEconomicImpact.CATASTROPHIC: 5,
        }

        highest = max(
            results,
            key=lambda item: impact_rank[
                item.economic_impact
            ],
        )

        return {
            "failure_count": len(results),

            "gross_failure_cost": round(
                gross,
                2,
            ),

            "recovery_credit": round(
                recoveries,
                2,
            ),

            "net_failure_cost": round(
                net,
                2,
            ),

            "maximum_failure_cost": round(
                maximum,
                2,
            ),

            "average_failure_cost": round(
                net / len(results),
                2,
            ),

            "highest_impact":
                highest.economic_impact.value,

            "highest_impact_asset_id":
                highest.asset_id,

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

failure_cost_calculator = (
    FailureCostCalculator()
)


# =========================================================
# Agent-facing helpers
# =========================================================

def calculate_failure_cost(
    *,
    failure: FailureCostInput,
    asset_id: str | None = None,
    component_id: str | None = None,
    site_id: str | None = None,
    asset_value: float | None = None,
) -> FailureCostResult:

    return failure_cost_calculator.calculate(
        failure=failure,
        asset_id=asset_id,
        component_id=component_id,
        site_id=site_id,
        asset_value=asset_value,
    )


def calculate_expected_failure_cost(
    *,
    probability: float,
    failure_cost: float,
) -> float:

    return (
        failure_cost_calculator
        .expected_failure_cost(
            probability=probability,
            failure_cost=failure_cost,
        )
    )


def compare_failure_repair_replacement(
    *,
    repair: FailureRepairCost,
    replacement: FailureReplacementCost,
    repair_success_probability: float = 1.0,
    post_repair_failure_cost: float = 0.0,
) -> dict[str, Any]:

    return (
        failure_cost_calculator
        .repair_replacement_comparison(
            repair=repair,
            replacement=replacement,
            repair_success_probability=(
                repair_success_probability
            ),
            post_repair_failure_cost=(
                post_repair_failure_cost
            ),
        )
    )


# =========================================================
# Exports
# =========================================================

__all__ = [
    "FailureType",
    "FailureSeverity",
    "FailureCostCategory",
    "FailureCostBasis",
    "FailureEconomicImpact",
    "CostConfidence",

    "FailureRepairCost",
    "FailureReplacementCost",
    "FailureDowntimeCost",
    "FailureQualityCost",
    "FailureEnergyCost",
    "FailureSafetyCost",
    "FailureBusinessCost",
    "FailureRecoveryCredits",

    "FailureCostInput",
    "FailureCostBreakdown",
    "FailureCostResult",

    "FailureCostScenario",
    "FailureCostScenarioResult",

    "FailureCostConfig",
    "FailureCostCalculator",

    "failure_cost_calculator",

    "calculate_failure_cost",
    "calculate_expected_failure_cost",
    "compare_failure_repair_replacement",
]