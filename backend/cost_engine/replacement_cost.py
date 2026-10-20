model_run# app/agents/tools/replacement_cost.py

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

class ReplacementType(str, Enum):
    LIKE_FOR_LIKE = "like_for_like"
    UPGRADE = "upgrade"
    MODERNIZATION = "modernization"
    CAPACITY_EXPANSION = "capacity_expansion"
    SAFETY_REPLACEMENT = "safety_replacement"
    OBSOLESCENCE = "obsolescence"
    EMERGENCY = "emergency"
    PLANNED = "planned"
    OTHER = "other"


class ReplacementReason(str, Enum):
    END_OF_LIFE = "end_of_life"
    LOW_RUL = "low_rul"
    REPEATED_FAILURE = "repeated_failure"
    HIGH_REPAIR_COST = "high_repair_cost"
    OBSOLESCENCE = "obsolescence"
    PARTS_UNAVAILABLE = "parts_unavailable"
    SAFETY = "safety"
    REGULATORY = "regulatory"
    EFFICIENCY = "efficiency"
    CAPACITY = "capacity"
    TECHNOLOGY_UPGRADE = "technology_upgrade"
    OTHER = "other"


class ReplacementCostCategory(str, Enum):
    ASSET_PURCHASE = "asset_purchase"
    ENGINEERING = "engineering"
    PROCUREMENT = "procurement"
    SHIPPING = "shipping"
    TAX = "tax"
    DUTY = "duty"
    INSURANCE = "insurance"

    REMOVAL = "removal"
    DISMANTLING = "dismantling"
    DISPOSAL = "disposal"

    INSTALLATION = "installation"
    CIVIL_WORK = "civil_work"
    ELECTRICAL_WORK = "electrical_work"
    MECHANICAL_WORK = "mechanical_work"
    INSTRUMENTATION = "instrumentation"

    SOFTWARE = "software"
    INTEGRATION = "integration"

    TESTING = "testing"
    COMMISSIONING = "commissioning"
    VALIDATION = "validation"

    TRAINING = "training"

    LABOR = "labor"
    CONTRACTOR = "contractor"

    DOWNTIME = "downtime"
    PRODUCTION_LOSS = "production_loss"

    TEMPORARY_EQUIPMENT = "temporary_equipment"

    SAFETY = "safety"

    CONTINGENCY = "contingency"

    OTHER = "other"


class ReplacementUrgency(str, Enum):
    ROUTINE = "routine"
    PLANNED = "planned"
    PRIORITY = "priority"
    URGENT = "urgent"
    EMERGENCY = "emergency"


class ReplacementEconomicRisk(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class ReplacementConfidence(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


# =========================================================
# New asset purchase
# =========================================================

@dataclass(slots=True)
class AssetPurchaseCost:
    base_asset_cost: float = 0.0

    accessories_cost: float = 0.0
    spare_parts_package_cost: float = 0.0

    software_license_cost: float = 0.0
    controls_package_cost: float = 0.0

    shipping_cost: float = 0.0
    freight_cost: float = 0.0

    import_duty: float = 0.0
    tax_cost: float = 0.0
    insurance_cost: float = 0.0

    procurement_cost: float = 0.0

    discount_credit: float = 0.0
    vendor_credit: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "base_asset_cost",
            "accessories_cost",
            "spare_parts_package_cost",
            "software_license_cost",
            "controls_package_cost",
            "shipping_cost",
            "freight_cost",
            "import_duty",
            "tax_cost",
            "insurance_cost",
            "procurement_cost",
            "discount_credit",
            "vendor_credit",
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
                self.base_asset_cost,
                self.accessories_cost,
                self.spare_parts_package_cost,
                self.software_license_cost,
                self.controls_package_cost,
                self.shipping_cost,
                self.freight_cost,
                self.import_duty,
                self.tax_cost,
                self.insurance_cost,
                self.procurement_cost,
            )
        )

    @property
    def credits(self) -> float:
        return (
            self.discount_credit
            + self.vendor_credit
        )

    @property
    def net_cost(self) -> float:
        return max(
            0.0,
            self.gross_cost - self.credits,
        )


# =========================================================
# Removal / disposal
# =========================================================

@dataclass(slots=True)
class RemovalDisposalCost:
    shutdown_preparation_cost: float = 0.0

    dismantling_cost: float = 0.0
    removal_labor_cost: float = 0.0

    lifting_rigging_cost: float = 0.0
    crane_cost: float = 0.0

    transportation_cost: float = 0.0

    disposal_cost: float = 0.0
    hazardous_material_cost: float = 0.0

    site_cleanup_cost: float = 0.0

    salvage_credit: float = 0.0
    resale_credit: float = 0.0
    recycling_credit: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "shutdown_preparation_cost",
            "dismantling_cost",
            "removal_labor_cost",
            "lifting_rigging_cost",
            "crane_cost",
            "transportation_cost",
            "disposal_cost",
            "hazardous_material_cost",
            "site_cleanup_cost",
            "salvage_credit",
            "resale_credit",
            "recycling_credit",
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
                self.shutdown_preparation_cost,
                self.dismantling_cost,
                self.removal_labor_cost,
                self.lifting_rigging_cost,
                self.crane_cost,
                self.transportation_cost,
                self.disposal_cost,
                self.hazardous_material_cost,
                self.site_cleanup_cost,
            )
        )

    @property
    def credits(self) -> float:
        return sum(
            (
                self.salvage_credit,
                self.resale_credit,
                self.recycling_credit,
            )
        )

    @property
    def net_cost(self) -> float:
        return max(
            0.0,
            self.gross_cost - self.credits,
        )


# =========================================================
# Installation
# =========================================================

@dataclass(slots=True)
class InstallationCost:
    engineering_cost: float = 0.0
    project_management_cost: float = 0.0

    civil_work_cost: float = 0.0
    foundation_cost: float = 0.0

    mechanical_installation_cost: float = 0.0
    electrical_installation_cost: float = 0.0
    instrumentation_cost: float = 0.0

    controls_integration_cost: float = 0.0
    software_integration_cost: float = 0.0

    network_integration_cost: float = 0.0

    contractor_cost: float = 0.0
    installation_labor_cost: float = 0.0

    tooling_cost: float = 0.0

    temporary_equipment_cost: float = 0.0

    permits_cost: float = 0.0

    safety_controls_cost: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "engineering_cost",
            "project_management_cost",
            "civil_work_cost",
            "foundation_cost",
            "mechanical_installation_cost",
            "electrical_installation_cost",
            "instrumentation_cost",
            "controls_integration_cost",
            "software_integration_cost",
            "network_integration_cost",
            "contractor_cost",
            "installation_labor_cost",
            "tooling_cost",
            "temporary_equipment_cost",
            "permits_cost",
            "safety_controls_cost",
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
                self.engineering_cost,
                self.project_management_cost,
                self.civil_work_cost,
                self.foundation_cost,
                self.mechanical_installation_cost,
                self.electrical_installation_cost,
                self.instrumentation_cost,
                self.controls_integration_cost,
                self.software_integration_cost,
                self.network_integration_cost,
                self.contractor_cost,
                self.installation_labor_cost,
                self.tooling_cost,
                self.temporary_equipment_cost,
                self.permits_cost,
                self.safety_controls_cost,
            )
        )


# =========================================================
# Commissioning
# =========================================================

@dataclass(slots=True)
class CommissioningCost:
    testing_cost: float = 0.0
    calibration_cost: float = 0.0

    commissioning_labor_cost: float = 0.0

    validation_cost: float = 0.0
    certification_cost: float = 0.0

    training_cost: float = 0.0

    documentation_cost: float = 0.0

    startup_material_cost: float = 0.0
    startup_energy_cost: float = 0.0
    startup_scrap_cost: float = 0.0

    other_cost: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "testing_cost",
            "calibration_cost",
            "commissioning_labor_cost",
            "validation_cost",
            "certification_cost",
            "training_cost",
            "documentation_cost",
            "startup_material_cost",
            "startup_energy_cost",
            "startup_scrap_cost",
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
                self.testing_cost,
                self.calibration_cost,
                self.commissioning_labor_cost,
                self.validation_cost,
                self.certification_cost,
                self.training_cost,
                self.documentation_cost,
                self.startup_material_cost,
                self.startup_energy_cost,
                self.startup_scrap_cost,
                self.other_cost,
            )
        )


# =========================================================
# Downtime
# =========================================================

@dataclass(slots=True)
class ReplacementDowntimeCost:
    shutdown_hours: float = 0.0

    direct_downtime_cost_per_hour: float = 0.0
    production_loss_per_hour: float = 0.0

    recoverable_production_fraction: float = 0.0

    overtime_recovery_cost: float = 0.0
    subcontracting_cost: float = 0.0
    expedited_production_cost: float = 0.0

    def __post_init__(self) -> None:
        self.shutdown_hours = non_negative(
            self.shutdown_hours
        )

        self.direct_downtime_cost_per_hour = non_negative(
            self.direct_downtime_cost_per_hour
        )

        self.production_loss_per_hour = non_negative(
            self.production_loss_per_hour
        )

        self.recoverable_production_fraction = clamp(
            self.recoverable_production_fraction
        )

        self.overtime_recovery_cost = non_negative(
            self.overtime_recovery_cost
        )

        self.subcontracting_cost = non_negative(
            self.subcontracting_cost
        )

        self.expedited_production_cost = non_negative(
            self.expedited_production_cost
        )

    @property
    def direct_downtime_cost(self) -> float:
        return (
            self.shutdown_hours
            * self.direct_downtime_cost_per_hour
        )

    @property
    def gross_production_loss(self) -> float:
        return (
            self.shutdown_hours
            * self.production_loss_per_hour
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
    def recovery_cost(self) -> float:
        return sum(
            (
                self.overtime_recovery_cost,
                self.subcontracting_cost,
                self.expedited_production_cost,
            )
        )

    @property
    def total(self) -> float:
        return (
            self.direct_downtime_cost
            + self.net_production_loss
            + self.recovery_cost
        )


# =========================================================
# Lifecycle economics
# =========================================================

@dataclass(slots=True)
class ReplacementLifecycle:
    useful_life_years: float = 10.0

    annual_maintenance_cost: float = 0.0
    annual_energy_cost: float = 0.0
    annual_other_operating_cost: float = 0.0

    annual_energy_savings: float = 0.0
    annual_maintenance_savings: float = 0.0
    annual_productivity_savings: float = 0.0
    annual_quality_savings: float = 0.0

    residual_failure_probability: float = 0.0
    residual_failure_cost: float = 0.0

    end_of_life_salvage_value: float = 0.0

    discount_rate: float = 0.08

    def __post_init__(self) -> None:
        self.useful_life_years = non_negative(
            self.useful_life_years
        )

        for name in (
            "annual_maintenance_cost",
            "annual_energy_cost",
            "annual_other_operating_cost",
            "annual_energy_savings",
            "annual_maintenance_savings",
            "annual_productivity_savings",
            "annual_quality_savings",
            "residual_failure_cost",
            "end_of_life_salvage_value",
        ):
            setattr(
                self,
                name,
                non_negative(getattr(self, name)),
            )

        self.residual_failure_probability = clamp(
            self.residual_failure_probability
        )

        self.discount_rate = finite(
            self.discount_rate
        )

    @property
    def annual_operating_cost(self) -> float:
        return sum(
            (
                self.annual_maintenance_cost,
                self.annual_energy_cost,
                self.annual_other_operating_cost,
            )
        )

    @property
    def annual_savings(self) -> float:
        return sum(
            (
                self.annual_energy_savings,
                self.annual_maintenance_savings,
                self.annual_productivity_savings,
                self.annual_quality_savings,
            )
        )

    @property
    def expected_residual_failure_cost(self) -> float:
        return (
            self.residual_failure_probability
            * self.residual_failure_cost
        )


# =========================================================
# Input
# =========================================================

@dataclass(slots=True)
class ReplacementCostInput:
    replacement_type: ReplacementType

    purchase: AssetPurchaseCost

    removal: RemovalDisposalCost = field(
        default_factory=RemovalDisposalCost
    )

    installation: InstallationCost = field(
        default_factory=InstallationCost
    )

    commissioning: CommissioningCost = field(
        default_factory=CommissioningCost
    )

    downtime: ReplacementDowntimeCost = field(
        default_factory=ReplacementDowntimeCost
    )

    lifecycle: ReplacementLifecycle = field(
        default_factory=ReplacementLifecycle
    )

    reasons: list[ReplacementReason] = field(
        default_factory=list
    )

    urgency: ReplacementUrgency = (
        ReplacementUrgency.PLANNED
    )

    contingency_rate: float = 0.0

    replacement_probability: float = 1.0

    current_asset_value: float | None = None

    safety_critical: bool = False
    mandatory_replacement: bool = False

    asset_id: str | None = None
    component_id: str | None = None
    site_id: str | None = None

    incident_id: str | None = None
    work_order_id: str | None = None

    currency: str = "INR"

    confidence: float = 1.0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.contingency_rate = non_negative(
            self.contingency_rate
        )

        self.replacement_probability = clamp(
            self.replacement_probability
        )

        if self.current_asset_value is not None:
            self.current_asset_value = non_negative(
                self.current_asset_value
            )

        self.confidence = clamp(
            self.confidence
        )


# =========================================================
# Breakdown
# =========================================================

@dataclass(slots=True)
class ReplacementCostBreakdown:
    purchase: float = 0.0
    purchase_credits: float = 0.0

    removal_disposal: float = 0.0
    old_asset_recovery_credits: float = 0.0

    installation: float = 0.0
    commissioning: float = 0.0

    direct_downtime: float = 0.0
    production_loss: float = 0.0
    production_recovery_cost: float = 0.0

    contingency: float = 0.0

    @property
    def capital_and_transition_cost(self) -> float:
        return sum(
            (
                self.purchase,
                self.removal_disposal,
                self.installation,
                self.commissioning,
                self.direct_downtime,
                self.production_loss,
                self.production_recovery_cost,
                self.contingency,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        result = json_safe(self)

        result["capital_and_transition_cost"] = round(
            self.capital_and_transition_cost,
            2,
        )

        return result


# =========================================================
# Result
# =========================================================

@dataclass(slots=True)
class ReplacementCostResult:
    calculation_id: str

    asset_id: str | None
    component_id: str | None
    site_id: str | None

    incident_id: str | None
    work_order_id: str | None

    replacement_type: ReplacementType
    urgency: ReplacementUrgency

    reasons: list[ReplacementReason]

    purchase_cost: float
    removal_disposal_cost: float

    installation_cost: float
    commissioning_cost: float

    downtime_cost: float
    production_loss: float

    contingency_cost: float

    initial_replacement_cost: float

    annual_operating_cost: float
    annual_savings: float

    lifecycle_operating_cost_pv: float
    lifecycle_savings_pv: float

    expected_residual_failure_cost: float

    salvage_value_pv: float

    lifecycle_present_value_cost: float

    equivalent_annual_cost: float | None

    expected_replacement_cost: float

    replacement_to_current_asset_value_ratio: float | None

    economic_risk: ReplacementEconomicRisk

    confidence: float
    confidence_level: ReplacementConfidence

    breakdown: ReplacementCostBreakdown

    safety_critical: bool
    mandatory_replacement: bool

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
class ReplacementCostConfig:
    moderate_asset_value_ratio: float = 0.50
    high_asset_value_ratio: float = 1.00
    critical_asset_value_ratio: float = 1.50

    minimum_confidence: float = 0.50

    urgent_multiplier: float = 1.08
    emergency_multiplier: float = 1.15

    default_analysis_horizon_years: float = 10.0

    default_currency: str = "INR"


# =========================================================
# Calculator
# =========================================================

class ReplacementCostCalculator:

    def __init__(
        self,
        *,
        config: ReplacementCostConfig | None = None,
    ) -> None:
        self.config = (
            config
            or ReplacementCostConfig()
        )

    # =====================================================
    # Financial helpers
    # =====================================================

    @staticmethod
    def present_value(
        *,
        amount: float,
        discount_rate: float,
        years: float,
    ) -> float:

        amount = non_negative(amount)
        years = non_negative(years)
        rate = finite(discount_rate)

        if years <= 0:
            return amount

        if abs(rate) <= 1e-12:
            return amount

        if rate <= -1.0:
            return amount

        return (
            amount
            / ((1.0 + rate) ** years)
        )

    @staticmethod
    def annuity_present_value(
        *,
        annual_amount: float,
        discount_rate: float,
        years: float,
    ) -> float:

        amount = non_negative(
            annual_amount
        )

        years = non_negative(
            years
        )

        rate = finite(
            discount_rate
        )

        if amount <= 0 or years <= 0:
            return 0.0

        if abs(rate) <= 1e-12:
            return amount * years

        if rate <= -1.0:
            return amount * years

        factor = (
            1.0
            - (1.0 + rate) ** (-years)
        ) / rate

        return amount * factor

    @staticmethod
    def equivalent_annual_cost(
        *,
        present_value_cost: float,
        discount_rate: float,
        years: float,
    ) -> float | None:

        pv = non_negative(
            present_value_cost
        )

        years = non_negative(
            years
        )

        rate = finite(
            discount_rate
        )

        if years <= 0:
            return None

        if abs(rate) <= 1e-12:
            return pv / years

        if rate <= -1.0:
            return pv / years

        denominator = (
            1.0
            - (1.0 + rate) ** (-years)
        )

        if abs(denominator) <= 1e-12:
            return None

        return (
            pv
            * rate
            / denominator
        )

    # =====================================================
    # Urgency
    # =====================================================

    def urgency_multiplier(
        self,
        urgency: ReplacementUrgency,
    ) -> float:

        if urgency == ReplacementUrgency.EMERGENCY:
            return (
                self.config
                .emergency_multiplier
            )

        if urgency == ReplacementUrgency.URGENT:
            return (
                self.config
                .urgent_multiplier
            )

        return 1.0

    # =====================================================
    # Confidence
    # =====================================================

    @staticmethod
    def confidence_level(
        confidence: float,
    ) -> ReplacementConfidence:

        value = clamp(confidence)

        if value >= 0.90:
            return ReplacementConfidence.VERY_HIGH

        if value >= 0.75:
            return ReplacementConfidence.HIGH

        if value >= 0.50:
            return ReplacementConfidence.MODERATE

        return ReplacementConfidence.LOW

    # =====================================================
    # Risk
    # =====================================================

    def economic_risk(
        self,
        *,
        lifecycle_cost: float,
        current_asset_value: float | None,
    ) -> ReplacementEconomicRisk:

        if (
            current_asset_value is None
            or current_asset_value <= 0
        ):
            return ReplacementEconomicRisk.MODERATE

        ratio = (
            lifecycle_cost
            / current_asset_value
        )

        if (
            ratio
            >= self.config
            .critical_asset_value_ratio
        ):
            return ReplacementEconomicRisk.CRITICAL

        if (
            ratio
            >= self.config
            .high_asset_value_ratio
        ):
            return ReplacementEconomicRisk.HIGH

        if (
            ratio
            >= self.config
            .moderate_asset_value_ratio
        ):
            return ReplacementEconomicRisk.MODERATE

        return ReplacementEconomicRisk.LOW

    # =====================================================
    # Main calculation
    # =====================================================

    def calculate(
        self,
        *,
        replacement: ReplacementCostInput,
        metadata: Mapping[str, Any] | None = None,
    ) -> ReplacementCostResult:

        warnings: list[str] = []

        purchase = replacement.purchase
        removal = replacement.removal
        installation = replacement.installation
        commissioning = replacement.commissioning
        downtime = replacement.downtime
        lifecycle = replacement.lifecycle

        # ---------------------------------------------
        # Initial capital / transition cost
        # ---------------------------------------------

        purchase_cost = purchase.net_cost

        removal_cost = removal.net_cost

        installation_cost = (
            installation.total
        )

        commissioning_cost = (
            commissioning.total
        )

        direct_downtime = (
            downtime.direct_downtime_cost
        )

        production_loss = (
            downtime.net_production_loss
        )

        recovery_cost = (
            downtime.recovery_cost
        )

        base_initial_cost = sum(
            (
                purchase_cost,
                removal_cost,
                installation_cost,
                commissioning_cost,
                direct_downtime,
                production_loss,
                recovery_cost,
            )
        )

        urgency_adjustment = max(
            0.0,
            base_initial_cost
            * (
                self.urgency_multiplier(
                    replacement.urgency
                )
                - 1.0
            ),
        )

        pre_contingency = (
            base_initial_cost
            + urgency_adjustment
        )

        contingency = (
            pre_contingency
            * replacement.contingency_rate
        )

        initial_cost = (
            pre_contingency
            + contingency
        )

        # ---------------------------------------------
        # Lifecycle
        # ---------------------------------------------

        horizon = (
            lifecycle.useful_life_years
            or self.config
            .default_analysis_horizon_years
        )

        operating_pv = (
            self.annuity_present_value(
                annual_amount=(
                    lifecycle.annual_operating_cost
                ),
                discount_rate=(
                    lifecycle.discount_rate
                ),
                years=horizon,
            )
        )

        savings_pv = (
            self.annuity_present_value(
                annual_amount=(
                    lifecycle.annual_savings
                ),
                discount_rate=(
                    lifecycle.discount_rate
                ),
                years=horizon,
            )
        )

        residual_failure_cost = (
            lifecycle
            .expected_residual_failure_cost
        )

        salvage_pv = self.present_value(
            amount=(
                lifecycle
                .end_of_life_salvage_value
            ),
            discount_rate=(
                lifecycle.discount_rate
            ),
            years=horizon,
        )

        lifecycle_pv = max(
            0.0,
            initial_cost
            + operating_pv
            + residual_failure_cost
            - savings_pv
            - salvage_pv,
        )

        eac = self.equivalent_annual_cost(
            present_value_cost=lifecycle_pv,
            discount_rate=(
                lifecycle.discount_rate
            ),
            years=horizon,
        )

        expected_replacement_cost = (
            lifecycle_pv
            * replacement.replacement_probability
        )

        # ---------------------------------------------
        # Ratios
        # ---------------------------------------------

        asset_ratio = None

        if (
            replacement.current_asset_value
            is not None
            and replacement.current_asset_value > 0
        ):
            asset_ratio = (
                lifecycle_pv
                / replacement.current_asset_value
            )

        # ---------------------------------------------
        # Risk / confidence
        # ---------------------------------------------

        risk = self.economic_risk(
            lifecycle_cost=lifecycle_pv,
            current_asset_value=(
                replacement.current_asset_value
            ),
        )

        confidence_level = (
            self.confidence_level(
                replacement.confidence
            )
        )

        # ---------------------------------------------
        # Breakdown
        # ---------------------------------------------

        breakdown = ReplacementCostBreakdown(
            purchase=purchase_cost,

            purchase_credits=(
                purchase.credits
            ),

            removal_disposal=(
                removal_cost
            ),

            old_asset_recovery_credits=(
                removal.credits
            ),

            installation=(
                installation_cost
            ),

            commissioning=(
                commissioning_cost
            ),

            direct_downtime=(
                direct_downtime
            ),

            production_loss=(
                production_loss
            ),

            production_recovery_cost=(
                recovery_cost
            ),

            # urgency premium is an explicit transition
            # cost and is included with contingency here.
            contingency=(
                contingency
                + urgency_adjustment
            ),
        )

        # ---------------------------------------------
        # Warnings
        # ---------------------------------------------

        if purchase.base_asset_cost <= 0:
            warnings.append(
                "Replacement asset purchase cost is "
                "zero or unavailable."
            )

        if lifecycle.useful_life_years <= 0:
            warnings.append(
                "Replacement useful life is unavailable; "
                "the configured default horizon is used."
            )

        if (
            replacement.confidence
            < self.config.minimum_confidence
        ):
            warnings.append(
                "Replacement cost estimate has low "
                "confidence."
            )

        if (
            direct_downtime > 0
            and production_loss > 0
        ):
            warnings.append(
                "Direct downtime cost and production "
                "loss are both included. Confirm they "
                "represent separate economic effects."
            )

        if (
            commissioning.startup_energy_cost > 0
            and lifecycle.annual_energy_cost > 0
        ):
            warnings.append(
                "Startup energy and annual energy are "
                "both included; ensure they refer to "
                "different periods."
            )

        if (
            installation.installation_labor_cost > 0
            and installation.contractor_cost > 0
        ):
            warnings.append(
                "Installation labor and contractor cost "
                "are both included. Verify contractor "
                "labor is not duplicated."
            )

        if (
            purchase.shipping_cost > 0
            and purchase.freight_cost > 0
        ):
            warnings.append(
                "Shipping and freight costs are both "
                "included. Verify they are distinct."
            )

        if (
            replacement.safety_critical
            or ReplacementReason.SAFETY
            in replacement.reasons
        ):
            warnings.append(
                "Safety-critical replacement economics "
                "must not override safety requirements."
            )

        if replacement.mandatory_replacement:
            warnings.append(
                "Replacement is mandatory. Financial "
                "metrics are informational and must not "
                "be used to reject the replacement."
            )

        return ReplacementCostResult(
            calculation_id=new_id(),

            asset_id=replacement.asset_id,
            component_id=(
                replacement.component_id
            ),
            site_id=replacement.site_id,

            incident_id=(
                replacement.incident_id
            ),
            work_order_id=(
                replacement.work_order_id
            ),

            replacement_type=(
                replacement.replacement_type
            ),

            urgency=replacement.urgency,

            reasons=list(
                replacement.reasons
            ),

            purchase_cost=round(
                purchase_cost,
                2,
            ),

            removal_disposal_cost=round(
                removal_cost,
                2,
            ),

            installation_cost=round(
                installation_cost,
                2,
            ),

            commissioning_cost=round(
                commissioning_cost,
                2,
            ),

            downtime_cost=round(
                direct_downtime,
                2,
            ),

            production_loss=round(
                production_loss,
                2,
            ),

            contingency_cost=round(
                contingency
                + urgency_adjustment,
                2,
            ),

            initial_replacement_cost=round(
                initial_cost,
                2,
            ),

            annual_operating_cost=round(
                lifecycle.annual_operating_cost,
                2,
            ),

            annual_savings=round(
                lifecycle.annual_savings,
                2,
            ),

            lifecycle_operating_cost_pv=round(
                operating_pv,
                2,
            ),

            lifecycle_savings_pv=round(
                savings_pv,
                2,
            ),

            expected_residual_failure_cost=round(
                residual_failure_cost,
                2,
            ),

            salvage_value_pv=round(
                salvage_pv,
                2,
            ),

            lifecycle_present_value_cost=round(
                lifecycle_pv,
                2,
            ),

            equivalent_annual_cost=(
                round(eac, 2)
                if eac is not None
                else None
            ),

            expected_replacement_cost=round(
                expected_replacement_cost,
                2,
            ),

            replacement_to_current_asset_value_ratio=(
                round(asset_ratio, 6)
                if asset_ratio is not None
                else None
            ),

            economic_risk=risk,

            confidence=round(
                replacement.confidence,
                4,
            ),

            confidence_level=(
                confidence_level
            ),

            breakdown=breakdown,

            safety_critical=(
                replacement.safety_critical
            ),

            mandatory_replacement=(
                replacement.mandatory_replacement
            ),

            currency=(
                replacement.currency
                or self.config.default_currency
            ),

            warnings=warnings,

            metadata={
                **replacement.metadata,
                **dict(metadata or {}),
            },
        )

    # =====================================================
    # Simple replacement cost
    # =====================================================

    @staticmethod
    def simple_replacement_cost(
        *,
        asset_purchase_cost: float,
        removal_cost: float = 0.0,
        installation_cost: float = 0.0,
        commissioning_cost: float = 0.0,
        downtime_cost: float = 0.0,
        production_loss: float = 0.0,
        salvage_credit: float = 0.0,
    ) -> float:

        gross = sum(
            (
                non_negative(
                    asset_purchase_cost
                ),
                non_negative(
                    removal_cost
                ),
                non_negative(
                    installation_cost
                ),
                non_negative(
                    commissioning_cost
                ),
                non_negative(
                    downtime_cost
                ),
                non_negative(
                    production_loss
                ),
            )
        )

        return round(
            max(
                0.0,
                gross
                - non_negative(
                    salvage_credit
                ),
            ),
            2,
        )

    # =====================================================
    # Future replacement cost
    # =====================================================

    @staticmethod
    def future_replacement_cost(
        *,
        current_replacement_cost: float,
        annual_escalation_rate: float,
        years: float,
    ) -> float:

        current = non_negative(
            current_replacement_cost
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
    # Replacement savings
    # =====================================================

    @staticmethod
    def annual_replacement_savings(
        *,
        old_maintenance_cost: float,
        new_maintenance_cost: float,
        old_energy_cost: float,
        new_energy_cost: float,
        productivity_savings: float = 0.0,
        quality_savings: float = 0.0,
    ) -> dict[str, float]:

        maintenance_savings = max(
            0.0,
            non_negative(
                old_maintenance_cost
            )
            - non_negative(
                new_maintenance_cost
            ),
        )

        energy_savings = max(
            0.0,
            non_negative(
                old_energy_cost
            )
            - non_negative(
                new_energy_cost
            ),
        )

        productivity = non_negative(
            productivity_savings
        )

        quality = non_negative(
            quality_savings
        )

        total = sum(
            (
                maintenance_savings,
                energy_savings,
                productivity,
                quality,
            )
        )

        return {
            "maintenance_savings": round(
                maintenance_savings,
                2,
            ),
            "energy_savings": round(
                energy_savings,
                2,
            ),
            "productivity_savings": round(
                productivity,
                2,
            ),
            "quality_savings": round(
                quality,
                2,
            ),
            "annual_total_savings": round(
                total,
                2,
            ),
        }

    # =====================================================
    # Payback
    # =====================================================

    @staticmethod
    def payback_period_years(
        *,
        replacement_investment: float,
        annual_savings: float,
    ) -> float | None:

        investment = non_negative(
            replacement_investment
        )

        savings = non_negative(
            annual_savings
        )

        if savings <= 0:
            return None

        return investment / savings

    # =====================================================
    # Replacement ROI
    # =====================================================

    @staticmethod
    def replacement_roi(
        *,
        replacement_investment: float,
        lifecycle_savings: float,
    ) -> float | None:

        investment = non_negative(
            replacement_investment
        )

        if investment <= 0:
            return None

        savings = non_negative(
            lifecycle_savings
        )

        return (
            (savings - investment)
            / investment
            * 100.0
        )

    # =====================================================
    # Batch
    # =====================================================

    def calculate_many(
        self,
        replacements: Sequence[
            ReplacementCostInput
        ],
    ) -> list[ReplacementCostResult]:

        return [
            self.calculate(
                replacement=item
            )
            for item in replacements
        ]

    # =====================================================
    # Aggregate
    # =====================================================

    @staticmethod
    def aggregate(
        results: Sequence[
            ReplacementCostResult
        ],
    ) -> dict[str, Any]:

        if not results:
            return {
                "replacement_count": 0,
                "initial_replacement_cost": 0.0,
                "lifecycle_present_value_cost": 0.0,
                "expected_replacement_cost": 0.0,
                "currency": None,
            }

        currencies = {
            result.currency
            for result in results
        }

        # Do not silently sum unlike currencies.
        if len(currencies) > 1:
            return {
                "replacement_count": len(
                    results
                ),
                "initial_replacement_cost": None,
                "lifecycle_present_value_cost": None,
                "expected_replacement_cost": None,
                "currency": "MIXED",
                "mixed_currency": True,
                "warning": (
                    "Replacement costs were not "
                    "aggregated because multiple "
                    "currencies are present."
                ),
            }

        initial = sum(
            result.initial_replacement_cost
            for result in results
        )

        lifecycle = sum(
            result.lifecycle_present_value_cost
            for result in results
        )

        expected = sum(
            result.expected_replacement_cost
            for result in results
        )

        savings = sum(
            result.lifecycle_savings_pv
            for result in results
        )

        return {
            "replacement_count": len(
                results
            ),

            "initial_replacement_cost": round(
                initial,
                2,
            ),

            "lifecycle_present_value_cost": round(
                lifecycle,
                2,
            ),

            "expected_replacement_cost": round(
                expected,
                2,
            ),

            "lifecycle_savings_pv": round(
                savings,
                2,
            ),

            "average_lifecycle_cost": round(
                lifecycle / len(results),
                2,
            ),

            "currency": next(
                iter(currencies)
            ),

            "mixed_currency": False,
        }


# =========================================================
# Default calculator
# =========================================================

replacement_cost_calculator = (
    ReplacementCostCalculator()
)


# =========================================================
# Agent-facing helpers
# =========================================================

def calculate_replacement_cost(
    *,
    replacement: ReplacementCostInput,
) -> ReplacementCostResult:

    return (
        replacement_cost_calculator
        .calculate(
            replacement=replacement
        )
    )


def calculate_simple_replacement_cost(
    *,
    asset_purchase_cost: float,
    removal_cost: float = 0.0,
    installation_cost: float = 0.0,
    commissioning_cost: float = 0.0,
    downtime_cost: float = 0.0,
    production_loss: float = 0.0,
    salvage_credit: float = 0.0,
) -> float:

    return (
        replacement_cost_calculator
        .simple_replacement_cost(
            asset_purchase_cost=(
                asset_purchase_cost
            ),
            removal_cost=removal_cost,
            installation_cost=(
                installation_cost
            ),
            commissioning_cost=(
                commissioning_cost
            ),
            downtime_cost=downtime_cost,
            production_loss=(
                production_loss
            ),
            salvage_credit=(
                salvage_credit
            ),
        )
    )


def calculate_future_replacement_cost(
    *,
    current_replacement_cost: float,
    annual_escalation_rate: float,
    years: float,
) -> float:

    return (
        replacement_cost_calculator
        .future_replacement_cost(
            current_replacement_cost=(
                current_replacement_cost
            ),
            annual_escalation_rate=(
                annual_escalation_rate
            ),
            years=years,
        )
    )


def calculate_replacement_payback(
    *,
    replacement_investment: float,
    annual_savings: float,
) -> float | None:

    return (
        replacement_cost_calculator
        .payback_period_years(
            replacement_investment=(
                replacement_investment
            ),
            annual_savings=annual_savings,
        )
    )


# =========================================================
# Exports
# =========================================================

__all__ = [
    "ReplacementType",
    "ReplacementReason",
    "ReplacementCostCategory",
    "ReplacementUrgency",
    "ReplacementEconomicRisk",
    "ReplacementConfidence",

    "AssetPurchaseCost",
    "RemovalDisposalCost",
    "InstallationCost",
    "CommissioningCost",
    "ReplacementDowntimeCost",
    "ReplacementLifecycle",
    "ReplacementCostInput",

    "ReplacementCostBreakdown",
    "ReplacementCostResult",

    "ReplacementCostConfig",
    "ReplacementCostCalculator",

    "replacement_cost_calculator",

    "calculate_replacement_cost",
    "calculate_simple_replacement_cost",
    "calculate_future_replacement_cost",
    "calculate_replacement_payback",
]