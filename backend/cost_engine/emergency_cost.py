# app/agents/tools/emergency_cost.py

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


def finite(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    if not math.isfinite(result):
        return default

    return result


def non_negative(
    value: Any,
    default: float = 0.0,
) -> float:
    return max(
        0.0,
        finite(value, default),
    )


def clamp(
    value: float,
    low: float = 0.0,
    high: float = 1.0,
) -> float:
    return max(
        low,
        min(
            high,
            finite(value),
        ),
    )


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
        return json_safe(
            asdict(value)
        )

    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (list, tuple, set),
    ):
        return [
            json_safe(item)
            for item in value
        ]

    if (
        isinstance(value, float)
        and not math.isfinite(value)
    ):
        return None

    return value


# =========================================================
# Enums
# =========================================================

class EmergencyType(str, Enum):
    FIRE = "fire"
    SMOKE = "smoke"

    EXPLOSION = "explosion"

    ELECTRICAL = "electrical"
    ARC_FLASH = "arc_flash"

    GAS_LEAK = "gas_leak"
    CHEMICAL_LEAK = "chemical_leak"
    FLUID_LEAK = "fluid_leak"

    OVERHEATING = "overheating"

    HIGH_PRESSURE = "high_pressure"
    PRESSURE_RELEASE = "pressure_release"

    MACHINE_FAILURE = "machine_failure"
    STRUCTURAL_FAILURE = "structural_failure"

    WORKER_INJURY = "worker_injury"
    WORKER_FALL = "worker_fall"

    ENVIRONMENTAL = "environmental"

    POWER_FAILURE = "power_failure"

    EVACUATION = "evacuation"

    MULTI_HAZARD = "multi_hazard"

    OTHER = "other"


class EmergencySeverity(str, Enum):
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CRITICAL = "critical"
    CATASTROPHIC = "catastrophic"


class EmergencyCostCategory(str, Enum):
    RESPONSE = "response"

    FIRE_RESPONSE = "fire_response"
    MEDICAL_RESPONSE = "medical_response"

    EVACUATION = "evacuation"

    SHUTDOWN = "shutdown"
    DOWNTIME = "downtime"
    PRODUCTION_LOSS = "production_loss"

    ASSET_DAMAGE = "asset_damage"
    REPAIR = "repair"
    REPLACEMENT = "replacement"

    LABOR = "labor"
    CONTRACTOR = "contractor"

    CLEANUP = "cleanup"
    ENVIRONMENTAL = "environmental"

    SAFETY = "safety"

    PENALTY = "penalty"
    REGULATORY = "regulatory"

    LOGISTICS = "logistics"

    TEMPORARY_EQUIPMENT = (
        "temporary_equipment"
    )

    RECOVERY = "recovery"
    RESTART = "restart"

    OTHER = "other"


class EmergencyCostBasis(str, Enum):
    COMPONENTIZED = "componentized"
    ALL_IN = "all_in"


class EmergencyEconomicRisk(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    CATASTROPHIC = "catastrophic"


class EmergencyCostConfidence(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


# =========================================================
# Emergency probability
# =========================================================

@dataclass(slots=True)
class EmergencyRisk:
    emergency_type: EmergencyType

    probability: float

    severity: EmergencySeverity

    confidence: float = 1.0

    horizon_hours: float | None = None

    safety_risk_score: float | None = None

    source: str | None = None

    def __post_init__(self) -> None:
        self.probability = clamp(
            self.probability
        )

        self.confidence = clamp(
            self.confidence
        )

        if self.horizon_hours is not None:
            self.horizon_hours = (
                non_negative(
                    self.horizon_hours
                )
            )

        if (
            self.safety_risk_score
            is not None
        ):
            self.safety_risk_score = clamp(
                self.safety_risk_score
            )


# =========================================================
# Response cost
# =========================================================

@dataclass(slots=True)
class EmergencyResponseCost:
    emergency_team_cost: float = 0.0

    fire_response_cost: float = 0.0

    medical_response_cost: float = 0.0

    security_cost: float = 0.0

    specialist_cost: float = 0.0

    contractor_cost: float = 0.0

    emergency_equipment_cost: float = 0.0

    consumables_cost: float = 0.0

    transport_cost: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "emergency_team_cost",
            "fire_response_cost",
            "medical_response_cost",
            "security_cost",
            "specialist_cost",
            "contractor_cost",
            "emergency_equipment_cost",
            "consumables_cost",
            "transport_cost",
        ):
            setattr(
                self,
                name,
                non_negative(
                    getattr(self, name)
                ),
            )

    @property
    def total(self) -> float:
        return sum(
            (
                self.emergency_team_cost,
                self.fire_response_cost,
                self.medical_response_cost,
                self.security_cost,
                self.specialist_cost,
                self.contractor_cost,
                self.emergency_equipment_cost,
                self.consumables_cost,
                self.transport_cost,
            )
        )


# =========================================================
# Evacuation cost
# =========================================================

@dataclass(slots=True)
class EvacuationCost:
    workers_evacuated: int = 0

    evacuation_hours: float = 0.0

    labor_cost_per_worker_hour: float = 0.0

    evacuation_transport_cost: float = 0.0

    temporary_facility_cost: float = 0.0

    accommodation_cost: float = 0.0

    other_cost: float = 0.0

    def __post_init__(self) -> None:
        self.workers_evacuated = max(
            0,
            int(self.workers_evacuated),
        )

        self.evacuation_hours = non_negative(
            self.evacuation_hours
        )

        self.labor_cost_per_worker_hour = (
            non_negative(
                self.labor_cost_per_worker_hour
            )
        )

        self.evacuation_transport_cost = (
            non_negative(
                self.evacuation_transport_cost
            )
        )

        self.temporary_facility_cost = (
            non_negative(
                self.temporary_facility_cost
            )
        )

        self.accommodation_cost = (
            non_negative(
                self.accommodation_cost
            )
        )

        self.other_cost = non_negative(
            self.other_cost
        )

    @property
    def lost_labor_cost(self) -> float:
        return (
            self.workers_evacuated
            * self.evacuation_hours
            * self.labor_cost_per_worker_hour
        )

    @property
    def total(self) -> float:
        return sum(
            (
                self.lost_labor_cost,
                self.evacuation_transport_cost,
                self.temporary_facility_cost,
                self.accommodation_cost,
                self.other_cost,
            )
        )


# =========================================================
# Shutdown / production cost
# =========================================================

@dataclass(slots=True)
class EmergencyShutdownCost:
    shutdown_hours: float = 0.0

    direct_downtime_cost_per_hour: float = 0.0

    production_loss_per_hour: float = 0.0

    production_recovery_fraction: float = 0.0

    restart_cost: float = 0.0

    startup_energy_cost: float = 0.0

    startup_scrap_cost: float = 0.0

    startup_quality_loss: float = 0.0

    def __post_init__(self) -> None:
        self.shutdown_hours = non_negative(
            self.shutdown_hours
        )

        self.direct_downtime_cost_per_hour = (
            non_negative(
                self.direct_downtime_cost_per_hour
            )
        )

        self.production_loss_per_hour = (
            non_negative(
                self.production_loss_per_hour
            )
        )

        self.production_recovery_fraction = (
            clamp(
                self.production_recovery_fraction
            )
        )

        self.restart_cost = non_negative(
            self.restart_cost
        )

        self.startup_energy_cost = (
            non_negative(
                self.startup_energy_cost
            )
        )

        self.startup_scrap_cost = (
            non_negative(
                self.startup_scrap_cost
            )
        )

        self.startup_quality_loss = (
            non_negative(
                self.startup_quality_loss
            )
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
            * self.production_recovery_fraction
        )

    @property
    def net_production_loss(self) -> float:
        return max(
            0.0,
            self.gross_production_loss
            - self.production_recovery_credit,
        )

    @property
    def total(self) -> float:
        return sum(
            (
                self.direct_downtime_cost,
                self.net_production_loss,
                self.restart_cost,
                self.startup_energy_cost,
                self.startup_scrap_cost,
                self.startup_quality_loss,
            )
        )


# =========================================================
# Asset damage
# =========================================================

@dataclass(slots=True)
class EmergencyAssetDamage:
    repair_cost: float = 0.0

    replacement_cost: float = 0.0

    parts_cost: float = 0.0

    labor_cost: float = 0.0

    inspection_cost: float = 0.0

    engineering_cost: float = 0.0

    temporary_equipment_cost: float = 0.0

    logistics_cost: float = 0.0

    salvage_recovery: float = 0.0

    insurance_recovery: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "repair_cost",
            "replacement_cost",
            "parts_cost",
            "labor_cost",
            "inspection_cost",
            "engineering_cost",
            "temporary_equipment_cost",
            "logistics_cost",
            "salvage_recovery",
            "insurance_recovery",
        ):
            setattr(
                self,
                name,
                non_negative(
                    getattr(self, name)
                ),
            )

    @property
    def gross_cost(self) -> float:
        return sum(
            (
                self.repair_cost,
                self.replacement_cost,
                self.parts_cost,
                self.labor_cost,
                self.inspection_cost,
                self.engineering_cost,
                self.temporary_equipment_cost,
                self.logistics_cost,
            )
        )

    @property
    def recovery_credit(self) -> float:
        return (
            self.salvage_recovery
            + self.insurance_recovery
        )

    @property
    def net_cost(self) -> float:
        return max(
            0.0,
            self.gross_cost
            - self.recovery_credit,
        )


# =========================================================
# Environmental / regulatory
# =========================================================

@dataclass(slots=True)
class EnvironmentalEmergencyCost:
    containment_cost: float = 0.0

    cleanup_cost: float = 0.0

    disposal_cost: float = 0.0

    remediation_cost: float = 0.0

    monitoring_cost: float = 0.0

    regulatory_penalty: float = 0.0

    legal_cost: float = 0.0

    reporting_cost: float = 0.0

    other_cost: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "containment_cost",
            "cleanup_cost",
            "disposal_cost",
            "remediation_cost",
            "monitoring_cost",
            "regulatory_penalty",
            "legal_cost",
            "reporting_cost",
            "other_cost",
        ):
            setattr(
                self,
                name,
                non_negative(
                    getattr(self, name)
                ),
            )

    @property
    def total(self) -> float:
        return sum(
            (
                self.containment_cost,
                self.cleanup_cost,
                self.disposal_cost,
                self.remediation_cost,
                self.monitoring_cost,
                self.regulatory_penalty,
                self.legal_cost,
                self.reporting_cost,
                self.other_cost,
            )
        )


# =========================================================
# Recovery
# =========================================================

@dataclass(slots=True)
class EmergencyRecoveryCost:
    recovery_labor_cost: float = 0.0

    recommissioning_cost: float = 0.0

    validation_cost: float = 0.0

    safety_review_cost: float = 0.0

    root_cause_analysis_cost: float = 0.0

    investigation_cost: float = 0.0

    retraining_cost: float = 0.0

    corrective_action_cost: float = 0.0

    other_cost: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "recovery_labor_cost",
            "recommissioning_cost",
            "validation_cost",
            "safety_review_cost",
            "root_cause_analysis_cost",
            "investigation_cost",
            "retraining_cost",
            "corrective_action_cost",
            "other_cost",
        ):
            setattr(
                self,
                name,
                non_negative(
                    getattr(self, name)
                ),
            )

    @property
    def total(self) -> float:
        return sum(
            (
                self.recovery_labor_cost,
                self.recommissioning_cost,
                self.validation_cost,
                self.safety_review_cost,
                self.root_cause_analysis_cost,
                self.investigation_cost,
                self.retraining_cost,
                self.corrective_action_cost,
                self.other_cost,
            )
        )


# =========================================================
# Emergency consequence
# =========================================================

@dataclass(slots=True)
class EmergencyConsequence:
    response: EmergencyResponseCost = field(
        default_factory=EmergencyResponseCost
    )

    evacuation: EvacuationCost = field(
        default_factory=EvacuationCost
    )

    shutdown: EmergencyShutdownCost = field(
        default_factory=EmergencyShutdownCost
    )

    asset_damage: EmergencyAssetDamage = field(
        default_factory=EmergencyAssetDamage
    )

    environmental: EnvironmentalEmergencyCost = field(
        default_factory=EnvironmentalEmergencyCost
    )

    recovery: EmergencyRecoveryCost = field(
        default_factory=EmergencyRecoveryCost
    )

    worker_safety_cost: float = 0.0

    business_interruption_cost: float = 0.0

    customer_penalty_cost: float = 0.0

    reputation_proxy_cost: float = 0.0

    other_cost: float = 0.0

    cost_basis: EmergencyCostBasis = (
        EmergencyCostBasis.COMPONENTIZED
    )

    all_in_emergency_cost: float | None = None

    currency: str = "INR"

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        for name in (
            "worker_safety_cost",
            "business_interruption_cost",
            "customer_penalty_cost",
            "reputation_proxy_cost",
            "other_cost",
        ):
            setattr(
                self,
                name,
                non_negative(
                    getattr(self, name)
                ),
            )

        if self.all_in_emergency_cost is not None:
            self.all_in_emergency_cost = (
                non_negative(
                    self.all_in_emergency_cost
                )
            )

    @property
    def component_total(self) -> float:
        return sum(
            (
                self.response.total,
                self.evacuation.total,
                self.shutdown.total,
                self.asset_damage.net_cost,
                self.environmental.total,
                self.recovery.total,
                self.worker_safety_cost,
                self.business_interruption_cost,
                self.customer_penalty_cost,
                self.reputation_proxy_cost,
                self.other_cost,
            )
        )

    @property
    def total(self) -> float:
        if (
            self.cost_basis
            == EmergencyCostBasis.ALL_IN
            and self.all_in_emergency_cost
            is not None
        ):
            return self.all_in_emergency_cost

        return self.component_total


# =========================================================
# Breakdown
# =========================================================

@dataclass(slots=True)
class EmergencyCostBreakdown:
    response: float = 0.0
    evacuation: float = 0.0

    direct_downtime: float = 0.0
    production_loss: float = 0.0

    restart: float = 0.0

    asset_damage: float = 0.0

    worker_safety: float = 0.0

    environmental: float = 0.0

    recovery: float = 0.0

    business_interruption: float = 0.0

    customer_penalties: float = 0.0

    reputation_proxy: float = 0.0

    other: float = 0.0

    recovery_credits: float = 0.0

    @property
    def gross_total(self) -> float:
        return sum(
            (
                self.response,
                self.evacuation,
                self.direct_downtime,
                self.production_loss,
                self.restart,
                self.asset_damage,
                self.worker_safety,
                self.environmental,
                self.recovery,
                self.business_interruption,
                self.customer_penalties,
                self.reputation_proxy,
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
class EmergencyCostResult:
    calculation_id: str

    asset_id: str | None
    site_id: str | None
    zone_id: str | None

    emergency_type: EmergencyType
    severity: EmergencySeverity

    probability: float

    consequence_cost: float

    expected_emergency_loss: float

    breakdown: EmergencyCostBreakdown

    economic_risk: EmergencyEconomicRisk

    confidence: float

    confidence_level: EmergencyCostConfidence

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
# Avoided emergency cost
# =========================================================

@dataclass(slots=True)
class EmergencyIntervention:
    intervention_cost: float

    post_intervention_probability: float

    success_probability: float = 1.0

    intervention_downtime_cost: float = 0.0

    intervention_production_loss: float = 0.0

    additional_cost: float = 0.0

    confidence: float = 1.0

    description: str | None = None

    def __post_init__(self) -> None:
        self.intervention_cost = (
            non_negative(
                self.intervention_cost
            )
        )

        self.post_intervention_probability = (
            clamp(
                self.post_intervention_probability
            )
        )

        self.success_probability = clamp(
            self.success_probability
        )

        self.intervention_downtime_cost = (
            non_negative(
                self.intervention_downtime_cost
            )
        )

        self.intervention_production_loss = (
            non_negative(
                self.intervention_production_loss
            )
        )

        self.additional_cost = (
            non_negative(
                self.additional_cost
            )
        )

        self.confidence = clamp(
            self.confidence
        )


@dataclass(slots=True)
class AvoidedEmergencyCostResult:
    baseline_probability: float

    target_post_probability: float

    effective_post_probability: float

    baseline_expected_loss: float

    post_expected_loss: float

    gross_avoided_emergency_cost: float

    total_intervention_cost: float

    net_avoided_emergency_cost: float

    risk_reduction_absolute: float

    risk_reduction_percent: float

    benefit_cost_ratio: float | None

    roi_percent: float | None

    economically_positive: bool

    currency: str

    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# =========================================================
# Config
# =========================================================

@dataclass(slots=True)
class EmergencyCostConfig:
    moderate_risk_threshold: float = 0.20

    high_risk_threshold: float = 0.40

    critical_risk_threshold: float = 0.65

    catastrophic_risk_threshold: float = 0.85

    minimum_confidence: float = 0.50

    default_currency: str = "INR"

    warn_component_double_count: bool = True


# =========================================================
# Calculator
# =========================================================

class EmergencyCostCalculator:

    def __init__(
        self,
        *,
        config: EmergencyCostConfig | None = None,
    ) -> None:
        self.config = (
            config
            or EmergencyCostConfig()
        )

    # =====================================================
    # Breakdown
    # =====================================================

    @staticmethod
    def breakdown(
        consequence: EmergencyConsequence,
    ) -> EmergencyCostBreakdown:

        if (
            consequence.cost_basis
            == EmergencyCostBasis.ALL_IN
        ):
            return EmergencyCostBreakdown(
                other=consequence.total
            )

        shutdown = consequence.shutdown
        damage = consequence.asset_damage

        restart = sum(
            (
                shutdown.restart_cost,
                shutdown.startup_energy_cost,
                shutdown.startup_scrap_cost,
                shutdown.startup_quality_loss,
            )
        )

        return EmergencyCostBreakdown(
            response=(
                consequence.response.total
            ),

            evacuation=(
                consequence.evacuation.total
            ),

            direct_downtime=(
                shutdown.direct_downtime_cost
            ),

            production_loss=(
                shutdown.net_production_loss
            ),

            restart=restart,

            asset_damage=(
                damage.gross_cost
            ),

            worker_safety=(
                consequence.worker_safety_cost
            ),

            environmental=(
                consequence.environmental.total
            ),

            recovery=(
                consequence.recovery.total
            ),

            business_interruption=(
                consequence
                .business_interruption_cost
            ),

            customer_penalties=(
                consequence
                .customer_penalty_cost
            ),

            reputation_proxy=(
                consequence
                .reputation_proxy_cost
            ),

            other=consequence.other_cost,

            recovery_credits=(
                damage.recovery_credit
            ),
        )

    # =====================================================
    # Expected emergency loss
    # =====================================================

    @staticmethod
    def expected_loss(
        *,
        probability: float,
        consequence_cost: float,
    ) -> float:

        return (
            clamp(probability)
            * non_negative(
                consequence_cost
            )
        )

    # =====================================================
    # Severity multiplier
    # =====================================================

    @staticmethod
    def severity_weight(
        severity: EmergencySeverity,
    ) -> float:

        return {
            EmergencySeverity.MINOR: 0.20,
            EmergencySeverity.MODERATE: 0.40,
            EmergencySeverity.MAJOR: 0.60,
            EmergencySeverity.CRITICAL: 0.85,
            EmergencySeverity.CATASTROPHIC: 1.00,
        }[severity]

    # =====================================================
    # Economic risk
    # =====================================================

    def classify_economic_risk(
        self,
        *,
        risk: EmergencyRisk,
        expected_loss: float,
        reference_asset_value: float | None,
    ) -> EmergencyEconomicRisk:

        severity_score = (
            self.severity_weight(
                risk.severity
            )
        )

        probability_score = (
            risk.probability
        )

        exposure_ratio = 0.0

        if (
            reference_asset_value is not None
            and reference_asset_value > 0
        ):
            exposure_ratio = clamp(
                expected_loss
                / reference_asset_value
            )

        score = max(
            severity_score
            * probability_score,
            exposure_ratio,
            (
                risk.safety_risk_score
                if risk.safety_risk_score
                is not None
                else 0.0
            ),
        )

        if (
            score
            >= self.config
            .catastrophic_risk_threshold
        ):
            return (
                EmergencyEconomicRisk
                .CATASTROPHIC
            )

        if (
            score
            >= self.config
            .critical_risk_threshold
        ):
            return (
                EmergencyEconomicRisk.CRITICAL
            )

        if (
            score
            >= self.config
            .high_risk_threshold
        ):
            return (
                EmergencyEconomicRisk.HIGH
            )

        if (
            score
            >= self.config
            .moderate_risk_threshold
        ):
            return (
                EmergencyEconomicRisk.MODERATE
            )

        return EmergencyEconomicRisk.LOW

    # =====================================================
    # Confidence
    # =====================================================

    @staticmethod
    def calculate_confidence(
        *,
        risk: EmergencyRisk,
        consequence: EmergencyConsequence,
    ) -> tuple[
        float,
        EmergencyCostConfidence,
    ]:

        coverage = 0.0

        if consequence.response.total > 0:
            coverage += 0.15

        if consequence.shutdown.total > 0:
            coverage += 0.20

        if consequence.asset_damage.gross_cost > 0:
            coverage += 0.20

        if consequence.evacuation.total > 0:
            coverage += 0.10

        if consequence.environmental.total > 0:
            coverage += 0.10

        if consequence.recovery.total > 0:
            coverage += 0.10

        if (
            consequence.worker_safety_cost > 0
        ):
            coverage += 0.15

        if (
            consequence.cost_basis
            == EmergencyCostBasis.ALL_IN
            and consequence.total > 0
        ):
            coverage = max(
                coverage,
                0.75,
            )

        score = clamp(
            0.70 * risk.confidence
            + 0.30 * clamp(coverage)
        )

        if score >= 0.90:
            level = (
                EmergencyCostConfidence
                .VERY_HIGH
            )

        elif score >= 0.75:
            level = (
                EmergencyCostConfidence.HIGH
            )

        elif score >= 0.50:
            level = (
                EmergencyCostConfidence
                .MODERATE
            )

        else:
            level = (
                EmergencyCostConfidence.LOW
            )

        return (
            round(score, 4),
            level,
        )

    # =====================================================
    # Main calculation
    # =====================================================

    def calculate(
        self,
        *,
        risk: EmergencyRisk,
        consequence: EmergencyConsequence,
        asset_id: str | None = None,
        site_id: str | None = None,
        zone_id: str | None = None,
        reference_asset_value: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> EmergencyCostResult:

        warnings: list[str] = []

        breakdown = self.breakdown(
            consequence
        )

        consequence_cost = (
            consequence.total
        )

        expected_loss = self.expected_loss(
            probability=risk.probability,
            consequence_cost=(
                consequence_cost
            ),
        )

        economic_risk = (
            self.classify_economic_risk(
                risk=risk,
                expected_loss=expected_loss,
                reference_asset_value=(
                    reference_asset_value
                ),
            )
        )

        confidence, confidence_level = (
            self.calculate_confidence(
                risk=risk,
                consequence=consequence,
            )
        )

        if consequence_cost <= 0:
            warnings.append(
                "Emergency consequence cost is "
                "unavailable or zero."
            )

        if (
            confidence
            < self.config.minimum_confidence
        ):
            warnings.append(
                "Emergency cost estimate has low "
                "confidence."
            )

        if (
            consequence.cost_basis
            == EmergencyCostBasis.COMPONENTIZED
            and self.config
            .warn_component_double_count
        ):
            shutdown = (
                consequence.shutdown
            )

            if (
                shutdown
                .direct_downtime_cost_per_hour > 0
                and shutdown
                .production_loss_per_hour > 0
            ):
                warnings.append(
                    "Direct downtime and production "
                    "loss are both included. Confirm "
                    "the direct downtime rate excludes "
                    "lost production."
                )

            damage = (
                consequence.asset_damage
            )

            if (
                damage.repair_cost > 0
                and (
                    damage.parts_cost > 0
                    or damage.labor_cost > 0
                )
            ):
                warnings.append(
                    "Verify repair_cost does not "
                    "already include parts and labor."
                )

            if (
                consequence
                .business_interruption_cost > 0
                and (
                    shutdown.direct_downtime_cost > 0
                    or shutdown.net_production_loss > 0
                )
            ):
                warnings.append(
                    "Business interruption cost is "
                    "included together with downtime "
                    "and production loss. Verify these "
                    "cost definitions do not overlap."
                )

        if (
            risk.safety_risk_score
            is not None
            and risk.safety_risk_score >= 0.80
        ):
            warnings.append(
                "High safety risk detected. Emergency "
                "response decisions must not be "
                "optimized solely on financial cost."
            )

        return EmergencyCostResult(
            calculation_id=new_id(),

            asset_id=asset_id,
            site_id=site_id,
            zone_id=zone_id,

            emergency_type=(
                risk.emergency_type
            ),

            severity=risk.severity,

            probability=round(
                risk.probability,
                4,
            ),

            consequence_cost=round(
                consequence_cost,
                2,
            ),

            expected_emergency_loss=round(
                expected_loss,
                2,
            ),

            breakdown=breakdown,

            economic_risk=economic_risk,

            confidence=confidence,

            confidence_level=(
                confidence_level
            ),

            currency=(
                consequence.currency
                or self.config
                .default_currency
            ),

            warnings=warnings,

            metadata=dict(
                metadata or {}
            ),
        )

    # =====================================================
    # Effective post-intervention probability
    # =====================================================

    @staticmethod
    def effective_post_probability(
        *,
        baseline_probability: float,
        target_probability: float,
        success_probability: float,
    ) -> float:

        baseline = clamp(
            baseline_probability
        )

        target = clamp(
            target_probability
        )

        success = clamp(
            success_probability
        )

        return clamp(
            success * target
            + (1.0 - success) * baseline
        )

    # =====================================================
    # Avoided emergency cost
    # =====================================================

    def avoided_emergency_cost(
        self,
        *,
        baseline_risk: EmergencyRisk,
        consequence: EmergencyConsequence,
        intervention: EmergencyIntervention,
    ) -> AvoidedEmergencyCostResult:

        warnings: list[str] = []

        effective_post = (
            self.effective_post_probability(
                baseline_probability=(
                    baseline_risk.probability
                ),

                target_probability=(
                    intervention
                    .post_intervention_probability
                ),

                success_probability=(
                    intervention
                    .success_probability
                ),
            )
        )

        consequence_cost = (
            consequence.total
        )

        baseline_loss = self.expected_loss(
            probability=(
                baseline_risk.probability
            ),
            consequence_cost=(
                consequence_cost
            ),
        )

        post_loss = self.expected_loss(
            probability=effective_post,
            consequence_cost=(
                consequence_cost
            ),
        )

        gross_avoided = max(
            0.0,
            baseline_loss - post_loss,
        )

        total_intervention_cost = sum(
            (
                intervention.intervention_cost,

                intervention
                .intervention_downtime_cost,

                intervention
                .intervention_production_loss,

                intervention.additional_cost,
            )
        )

        net_avoided = (
            gross_avoided
            - total_intervention_cost
        )

        absolute_reduction = max(
            0.0,
            baseline_risk.probability
            - effective_post,
        )

        reduction_percent = (
            safe_divide(
                absolute_reduction,
                baseline_risk.probability,
                default=0.0,
            )
            * 100.0
        )

        benefit_cost_ratio = None
        roi_percent = None

        if total_intervention_cost > 0:
            benefit_cost_ratio = (
                gross_avoided
                / total_intervention_cost
            )

            roi_percent = (
                net_avoided
                / total_intervention_cost
                * 100.0
            )

        if (
            effective_post
            >= baseline_risk.probability
        ):
            warnings.append(
                "The intervention does not reduce "
                "effective emergency probability."
            )

        if net_avoided < 0:
            warnings.append(
                "Expected intervention cost exceeds "
                "the estimated avoided emergency "
                "loss."
            )

        if (
            baseline_risk.safety_risk_score
            is not None
            and baseline_risk
            .safety_risk_score >= 0.80
        ):
            warnings.append(
                "A negative financial ROI must not "
                "be used to reject a safety-required "
                "intervention."
            )

        return AvoidedEmergencyCostResult(
            baseline_probability=round(
                baseline_risk.probability,
                4,
            ),

            target_post_probability=round(
                intervention
                .post_intervention_probability,
                4,
            ),

            effective_post_probability=round(
                effective_post,
                4,
            ),

            baseline_expected_loss=round(
                baseline_loss,
                2,
            ),

            post_expected_loss=round(
                post_loss,
                2,
            ),

            gross_avoided_emergency_cost=round(
                gross_avoided,
                2,
            ),

            total_intervention_cost=round(
                total_intervention_cost,
                2,
            ),

            net_avoided_emergency_cost=round(
                net_avoided,
                2,
            ),

            risk_reduction_absolute=round(
                absolute_reduction,
                4,
            ),

            risk_reduction_percent=round(
                reduction_percent,
                2,
            ),

            benefit_cost_ratio=(
                round(
                    benefit_cost_ratio,
                    4,
                )
                if benefit_cost_ratio
                is not None
                else None
            ),

            roi_percent=(
                round(
                    roi_percent,
                    2,
                )
                if roi_percent is not None
                else None
            ),

            economically_positive=(
                net_avoided > 0
            ),

            currency=consequence.currency,

            warnings=warnings,
        )

    # =====================================================
    # Annual expected emergency cost
    # =====================================================

    @staticmethod
    def annual_expected_cost(
        *,
        expected_events_per_year: float,
        average_cost_per_event: float,
    ) -> float:

        return round(
            non_negative(
                expected_events_per_year
            )
            * non_negative(
                average_cost_per_event
            ),
            2,
        )

    # =====================================================
    # Scenario aggregation
    # =====================================================

    def aggregate_expected_exposure(
        self,
        results: Sequence[
            EmergencyCostResult
        ],
    ) -> dict[str, Any]:

        if not results:
            return {
                "event_count": 0,
                "total_expected_loss": 0.0,
                "maximum_expected_loss": 0.0,
                "highest_risk": None,
                "currency": None,
            }

        total_expected = sum(
            result.expected_emergency_loss
            for result in results
        )

        maximum = max(
            result.expected_emergency_loss
            for result in results
        )

        risk_rank = {
            EmergencyEconomicRisk.LOW: 1,
            EmergencyEconomicRisk.MODERATE: 2,
            EmergencyEconomicRisk.HIGH: 3,
            EmergencyEconomicRisk.CRITICAL: 4,
            EmergencyEconomicRisk.CATASTROPHIC: 5,
        }

        highest = max(
            results,
            key=lambda result: risk_rank[
                result.economic_risk
            ],
        )

        currencies = {
            result.currency
            for result in results
        }

        return {
            "event_count": len(results),

            "total_expected_loss": round(
                total_expected,
                2,
            ),

            "maximum_expected_loss": round(
                maximum,
                2,
            ),

            "highest_risk":
                highest.economic_risk.value,

            "currency": (
                next(iter(currencies))
                if len(currencies) == 1
                else "MIXED"
            ),

            "mixed_currency":
                len(currencies) > 1,
        }


# =========================================================
# Default instance
# =========================================================

emergency_cost_calculator = (
    EmergencyCostCalculator()
)


# =========================================================
# Agent-facing helpers
# =========================================================

def calculate_emergency_cost(
    *,
    risk: EmergencyRisk,
    consequence: EmergencyConsequence,
    asset_id: str | None = None,
    site_id: str | None = None,
    zone_id: str | None = None,
    reference_asset_value: float | None = None,
) -> EmergencyCostResult:

    return (
        emergency_cost_calculator.calculate(
            risk=risk,
            consequence=consequence,

            asset_id=asset_id,
            site_id=site_id,
            zone_id=zone_id,

            reference_asset_value=(
                reference_asset_value
            ),
        )
    )


def calculate_expected_emergency_loss(
    *,
    probability: float,
    consequence_cost: float,
) -> float:

    return round(
        emergency_cost_calculator
        .expected_loss(
            probability=probability,
            consequence_cost=(
                consequence_cost
            ),
        ),
        2,
    )


def calculate_avoided_emergency_cost(
    *,
    baseline_risk: EmergencyRisk,
    consequence: EmergencyConsequence,
    intervention: EmergencyIntervention,
) -> AvoidedEmergencyCostResult:

    return (
        emergency_cost_calculator
        .avoided_emergency_cost(
            baseline_risk=baseline_risk,
            consequence=consequence,
            intervention=intervention,
        )
    )


# =========================================================
# Exports
# =========================================================

__all__ = [
    "EmergencyType",
    "EmergencySeverity",
    "EmergencyCostCategory",
    "EmergencyCostBasis",
    "EmergencyEconomicRisk",
    "EmergencyCostConfidence",

    "EmergencyRisk",

    "EmergencyResponseCost",
    "EvacuationCost",
    "EmergencyShutdownCost",
    "EmergencyAssetDamage",
    "EnvironmentalEmergencyCost",
    "EmergencyRecoveryCost",

    "EmergencyConsequence",
    "EmergencyCostBreakdown",
    "EmergencyCostResult",

    "EmergencyIntervention",
    "AvoidedEmergencyCostResult",

    "EmergencyCostConfig",
    "EmergencyCostCalculator",

    "emergency_cost_calculator",

    "calculate_emergency_cost",
    "calculate_expected_emergency_loss",
    "calculate_avoided_emergency_cost",
]