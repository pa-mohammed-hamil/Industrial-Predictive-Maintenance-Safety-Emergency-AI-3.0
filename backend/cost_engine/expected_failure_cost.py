# app/agents/tools/expected_failure_cost.py

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

class FailureCostCategory(str, Enum):
    REPAIR = "repair"
    REPLACEMENT = "replacement"

    PARTS = "parts"
    LABOR = "labor"
    SPARES = "spares"

    DOWNTIME = "downtime"
    PRODUCTION_LOSS = "production_loss"

    SCRAP = "scrap"
    QUALITY = "quality"

    ENERGY = "energy"

    SAFETY = "safety"
    EMERGENCY = "emergency"

    ENVIRONMENTAL = "environmental"

    LOGISTICS = "logistics"
    CONTRACTOR = "contractor"

    PENALTY = "penalty"
    REGULATORY = "regulatory"

    RECOVERY = "recovery"

    OTHER = "other"


class FailureCostBasis(str, Enum):
    COMPONENTIZED = "componentized"
    ALL_IN = "all_in"


class FailureSeverity(str, Enum):
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CRITICAL = "critical"
    CATASTROPHIC = "catastrophic"


class FailureEconomicRisk(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    EXTREME = "extreme"


class FailureCostConfidence(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


# =========================================================
# Failure probability
# =========================================================

@dataclass(slots=True)
class FailureProbability:
    probability: float

    horizon_hours: float | None = None

    model_confidence: float = 1.0

    data_quality_score: float = 1.0

    model_drift_detected: bool = False

    source: str | None = None

    model_version: str | None = None

    timestamp: datetime = field(
        default_factory=utc_now
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.probability = clamp(
            self.probability
        )

        self.model_confidence = clamp(
            self.model_confidence
        )

        self.data_quality_score = clamp(
            self.data_quality_score
        )

        if self.horizon_hours is not None:
            self.horizon_hours = non_negative(
                self.horizon_hours
            )


# =========================================================
# Failure consequence
# =========================================================

@dataclass(slots=True)
class FailureConsequenceCost:
    repair_cost: float = 0.0
    replacement_cost: float = 0.0

    parts_cost: float = 0.0
    labor_cost: float = 0.0
    spares_cost: float = 0.0

    downtime_cost: float = 0.0
    production_loss: float = 0.0

    scrap_cost: float = 0.0
    quality_loss: float = 0.0

    energy_loss: float = 0.0

    safety_cost: float = 0.0
    emergency_cost: float = 0.0

    environmental_cost: float = 0.0

    logistics_cost: float = 0.0
    contractor_cost: float = 0.0

    penalty_cost: float = 0.0
    regulatory_cost: float = 0.0

    recovery_cost: float = 0.0

    other_cost: float = 0.0

    salvage_recovery: float = 0.0
    insurance_recovery: float = 0.0
    warranty_recovery: float = 0.0

    cost_basis: FailureCostBasis = (
        FailureCostBasis.COMPONENTIZED
    )

    all_in_failure_cost: float | None = None

    currency: str = "INR"

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        for name in (
            "repair_cost",
            "replacement_cost",
            "parts_cost",
            "labor_cost",
            "spares_cost",
            "downtime_cost",
            "production_loss",
            "scrap_cost",
            "quality_loss",
            "energy_loss",
            "safety_cost",
            "emergency_cost",
            "environmental_cost",
            "logistics_cost",
            "contractor_cost",
            "penalty_cost",
            "regulatory_cost",
            "recovery_cost",
            "other_cost",
            "salvage_recovery",
            "insurance_recovery",
            "warranty_recovery",
        ):
            setattr(
                self,
                name,
                non_negative(getattr(self, name)),
            )

        if self.all_in_failure_cost is not None:
            self.all_in_failure_cost = non_negative(
                self.all_in_failure_cost
            )

    @property
    def gross_component_cost(self) -> float:
        return sum(
            (
                self.repair_cost,
                self.replacement_cost,
                self.parts_cost,
                self.labor_cost,
                self.spares_cost,
                self.downtime_cost,
                self.production_loss,
                self.scrap_cost,
                self.quality_loss,
                self.energy_loss,
                self.safety_cost,
                self.emergency_cost,
                self.environmental_cost,
                self.logistics_cost,
                self.contractor_cost,
                self.penalty_cost,
                self.regulatory_cost,
                self.recovery_cost,
                self.other_cost,
            )
        )

    @property
    def recovery_credit(self) -> float:
        return sum(
            (
                self.salvage_recovery,
                self.insurance_recovery,
                self.warranty_recovery,
            )
        )

    @property
    def gross_cost(self) -> float:
        if (
            self.cost_basis == FailureCostBasis.ALL_IN
            and self.all_in_failure_cost is not None
        ):
            return self.all_in_failure_cost

        return self.gross_component_cost

    @property
    def net_cost(self) -> float:
        return max(
            0.0,
            self.gross_cost
            - self.recovery_credit,
        )


# =========================================================
# Breakdown
# =========================================================

@dataclass(slots=True)
class ExpectedFailureCostBreakdown:
    repair: float = 0.0
    replacement: float = 0.0

    parts: float = 0.0
    labor: float = 0.0
    spares: float = 0.0

    downtime: float = 0.0
    production_loss: float = 0.0

    scrap: float = 0.0
    quality: float = 0.0

    energy: float = 0.0

    safety: float = 0.0
    emergency: float = 0.0

    environmental: float = 0.0

    logistics: float = 0.0
    contractor: float = 0.0

    penalties: float = 0.0
    regulatory: float = 0.0

    recovery: float = 0.0
    other: float = 0.0

    recovery_credit: float = 0.0

    @property
    def gross_expected_cost(self) -> float:
        return sum(
            (
                self.repair,
                self.replacement,
                self.parts,
                self.labor,
                self.spares,
                self.downtime,
                self.production_loss,
                self.scrap,
                self.quality,
                self.energy,
                self.safety,
                self.emergency,
                self.environmental,
                self.logistics,
                self.contractor,
                self.penalties,
                self.regulatory,
                self.recovery,
                self.other,
            )
        )

    @property
    def net_expected_cost(self) -> float:
        return max(
            0.0,
            self.gross_expected_cost
            - self.recovery_credit,
        )

    def to_dict(self) -> dict[str, Any]:
        result = json_safe(self)

        result["gross_expected_cost"] = round(
            self.gross_expected_cost,
            2,
        )

        result["net_expected_cost"] = round(
            self.net_expected_cost,
            2,
        )

        return result


# =========================================================
# Result
# =========================================================

@dataclass(slots=True)
class ExpectedFailureCostResult:
    calculation_id: str

    asset_id: str | None
    component_id: str | None
    site_id: str | None

    failure_probability: float
    horizon_hours: float | None

    severity: FailureSeverity

    gross_failure_consequence: float

    recovery_credit: float

    net_failure_consequence: float

    gross_expected_failure_cost: float

    expected_recovery_credit: float

    expected_failure_cost: float

    expected_cost_per_hour: float | None

    expected_annualized_cost: float | None

    breakdown: ExpectedFailureCostBreakdown

    economic_risk: FailureEconomicRisk

    confidence: float
    confidence_level: FailureCostConfidence

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
# Failure scenario
# =========================================================

@dataclass(slots=True)
class FailureScenario:
    name: str

    probability: float

    consequence: FailureConsequenceCost

    severity: FailureSeverity = (
        FailureSeverity.MODERATE
    )

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
class FailureScenarioResult:
    name: str

    probability: float

    consequence_cost: float

    expected_cost: float

    scenario_weight: float

    weighted_expected_cost: float

    severity: FailureSeverity

    currency: str

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# =========================================================
# Intervention comparison
# =========================================================

@dataclass(slots=True)
class FailureIntervention:
    intervention_cost: float

    post_failure_probability: float

    success_probability: float = 1.0

    intervention_downtime_cost: float = 0.0

    additional_operating_cost: float = 0.0

    confidence: float = 1.0

    description: str | None = None

    def __post_init__(self) -> None:
        self.intervention_cost = non_negative(
            self.intervention_cost
        )

        self.post_failure_probability = clamp(
            self.post_failure_probability
        )

        self.success_probability = clamp(
            self.success_probability
        )

        self.intervention_downtime_cost = (
            non_negative(
                self.intervention_downtime_cost
            )
        )

        self.additional_operating_cost = (
            non_negative(
                self.additional_operating_cost
            )
        )

        self.confidence = clamp(
            self.confidence
        )


@dataclass(slots=True)
class FailureInterventionResult:
    baseline_expected_failure_cost: float

    target_post_probability: float

    effective_post_probability: float

    post_expected_failure_cost: float

    gross_avoided_failure_cost: float

    intervention_cost: float

    total_intervention_cost: float

    net_avoided_cost: float

    benefit_cost_ratio: float | None

    roi_percent: float | None

    risk_reduction_percent: float

    economically_positive: bool

    currency: str

    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# =========================================================
# Config
# =========================================================

@dataclass(slots=True)
class ExpectedFailureCostConfig:
    moderate_risk_ratio: float = 0.10
    high_risk_ratio: float = 0.25
    critical_risk_ratio: float = 0.50
    extreme_risk_ratio: float = 0.80

    minimum_confidence: float = 0.50

    annual_hours: float = 8760.0

    warn_component_double_count: bool = True

    default_currency: str = "INR"


# =========================================================
# Calculator
# =========================================================

class ExpectedFailureCostCalculator:

    def __init__(
        self,
        *,
        config: ExpectedFailureCostConfig | None = None,
    ) -> None:
        self.config = (
            config
            or ExpectedFailureCostConfig()
        )

    # =====================================================
    # Expected value
    # =====================================================

    @staticmethod
    def expected_value(
        *,
        probability: float,
        consequence_cost: float,
    ) -> float:
        return (
            clamp(probability)
            * non_negative(consequence_cost)
        )

    # =====================================================
    # Expected breakdown
    # =====================================================

    def calculate_breakdown(
        self,
        *,
        probability: float,
        consequence: FailureConsequenceCost,
    ) -> ExpectedFailureCostBreakdown:

        probability = clamp(probability)

        if (
            consequence.cost_basis
            == FailureCostBasis.ALL_IN
        ):
            return ExpectedFailureCostBreakdown(
                other=(
                    probability
                    * consequence.gross_cost
                ),
                recovery_credit=(
                    probability
                    * consequence.recovery_credit
                ),
            )

        def expected(value: float) -> float:
            return probability * value

        return ExpectedFailureCostBreakdown(
            repair=expected(
                consequence.repair_cost
            ),

            replacement=expected(
                consequence.replacement_cost
            ),

            parts=expected(
                consequence.parts_cost
            ),

            labor=expected(
                consequence.labor_cost
            ),

            spares=expected(
                consequence.spares_cost
            ),

            downtime=expected(
                consequence.downtime_cost
            ),

            production_loss=expected(
                consequence.production_loss
            ),

            scrap=expected(
                consequence.scrap_cost
            ),

            quality=expected(
                consequence.quality_loss
            ),

            energy=expected(
                consequence.energy_loss
            ),

            safety=expected(
                consequence.safety_cost
            ),

            emergency=expected(
                consequence.emergency_cost
            ),

            environmental=expected(
                consequence.environmental_cost
            ),

            logistics=expected(
                consequence.logistics_cost
            ),

            contractor=expected(
                consequence.contractor_cost
            ),

            penalties=expected(
                consequence.penalty_cost
            ),

            regulatory=expected(
                consequence.regulatory_cost
            ),

            recovery=expected(
                consequence.recovery_cost
            ),

            other=expected(
                consequence.other_cost
            ),

            recovery_credit=expected(
                consequence.recovery_credit
            ),
        )

    # =====================================================
    # Confidence
    # =====================================================

    @staticmethod
    def calculate_confidence(
        *,
        failure: FailureProbability,
        consequence: FailureConsequenceCost,
    ) -> tuple[
        float,
        FailureCostConfidence,
    ]:

        consequence_coverage = 0.0

        if consequence.gross_cost > 0:
            consequence_coverage += 0.30

        if (
            consequence.repair_cost > 0
            or consequence.replacement_cost > 0
        ):
            consequence_coverage += 0.15

        if (
            consequence.downtime_cost > 0
            or consequence.production_loss > 0
        ):
            consequence_coverage += 0.20

        if (
            consequence.safety_cost > 0
            or consequence.emergency_cost > 0
        ):
            consequence_coverage += 0.15

        if (
            consequence.parts_cost > 0
            or consequence.labor_cost > 0
        ):
            consequence_coverage += 0.10

        if (
            consequence.environmental_cost > 0
            or consequence.regulatory_cost > 0
        ):
            consequence_coverage += 0.10

        if (
            consequence.cost_basis
            == FailureCostBasis.ALL_IN
            and consequence.gross_cost > 0
        ):
            consequence_coverage = max(
                consequence_coverage,
                0.80,
            )

        score = clamp(
            0.45 * failure.model_confidence
            + 0.30 * failure.data_quality_score
            + 0.25 * clamp(
                consequence_coverage
            )
        )

        if failure.model_drift_detected:
            score *= 0.80

        score = clamp(score)

        if score >= 0.90:
            level = (
                FailureCostConfidence.VERY_HIGH
            )

        elif score >= 0.75:
            level = (
                FailureCostConfidence.HIGH
            )

        elif score >= 0.50:
            level = (
                FailureCostConfidence.MODERATE
            )

        else:
            level = (
                FailureCostConfidence.LOW
            )

        return round(score, 4), level

    # =====================================================
    # Economic risk
    # =====================================================

    def economic_risk(
        self,
        *,
        expected_failure_cost: float,
        reference_asset_value: float | None,
        probability: float,
        severity: FailureSeverity,
    ) -> FailureEconomicRisk:

        severity_weight = {
            FailureSeverity.MINOR: 0.15,
            FailureSeverity.MODERATE: 0.30,
            FailureSeverity.MAJOR: 0.55,
            FailureSeverity.CRITICAL: 0.80,
            FailureSeverity.CATASTROPHIC: 1.00,
        }[severity]

        exposure_ratio = 0.0

        if (
            reference_asset_value is not None
            and reference_asset_value > 0
        ):
            exposure_ratio = clamp(
                expected_failure_cost
                / reference_asset_value
            )

        score = max(
            probability * severity_weight,
            exposure_ratio,
        )

        if (
            score
            >= self.config.extreme_risk_ratio
        ):
            return FailureEconomicRisk.EXTREME

        if (
            score
            >= self.config.critical_risk_ratio
        ):
            return FailureEconomicRisk.CRITICAL

        if (
            score
            >= self.config.high_risk_ratio
        ):
            return FailureEconomicRisk.HIGH

        if (
            score
            >= self.config.moderate_risk_ratio
        ):
            return FailureEconomicRisk.MODERATE

        return FailureEconomicRisk.LOW

    # =====================================================
    # Annualize expected cost
    # =====================================================

    def annualize(
        self,
        *,
        expected_cost: float,
        horizon_hours: float | None,
    ) -> float | None:

        if (
            horizon_hours is None
            or horizon_hours <= 0
        ):
            return None

        return (
            expected_cost
            * self.config.annual_hours
            / horizon_hours
        )

    # =====================================================
    # Main calculation
    # =====================================================

    def calculate(
        self,
        *,
        failure: FailureProbability,
        consequence: FailureConsequenceCost,
        severity: FailureSeverity = (
            FailureSeverity.MODERATE
        ),
        asset_id: str | None = None,
        component_id: str | None = None,
        site_id: str | None = None,
        reference_asset_value: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ExpectedFailureCostResult:

        warnings: list[str] = []

        gross_consequence = (
            consequence.gross_cost
        )

        recovery_credit = (
            consequence.recovery_credit
        )

        net_consequence = (
            consequence.net_cost
        )

        gross_expected = (
            self.expected_value(
                probability=failure.probability,
                consequence_cost=(
                    gross_consequence
                ),
            )
        )

        expected_recovery = (
            self.expected_value(
                probability=failure.probability,
                consequence_cost=(
                    recovery_credit
                ),
            )
        )

        expected_cost = max(
            0.0,
            gross_expected
            - expected_recovery,
        )

        breakdown = (
            self.calculate_breakdown(
                probability=(
                    failure.probability
                ),
                consequence=consequence,
            )
        )

        expected_per_hour = None

        if (
            failure.horizon_hours is not None
            and failure.horizon_hours > 0
        ):
            expected_per_hour = (
                expected_cost
                / failure.horizon_hours
            )

        annualized = self.annualize(
            expected_cost=expected_cost,
            horizon_hours=(
                failure.horizon_hours
            ),
        )

        confidence, confidence_level = (
            self.calculate_confidence(
                failure=failure,
                consequence=consequence,
            )
        )

        economic_risk = (
            self.economic_risk(
                expected_failure_cost=(
                    expected_cost
                ),
                reference_asset_value=(
                    reference_asset_value
                ),
                probability=(
                    failure.probability
                ),
                severity=severity,
            )
        )

        if gross_consequence <= 0:
            warnings.append(
                "Failure consequence cost is zero "
                "or unavailable."
            )

        if (
            confidence
            < self.config.minimum_confidence
        ):
            warnings.append(
                "Expected failure cost has low "
                "confidence."
            )

        if failure.model_drift_detected:
            warnings.append(
                "Failure-model drift detected. "
                "Expected cost should be reviewed "
                "before high-value decisions."
            )

        if (
            failure.horizon_hours is None
        ):
            warnings.append(
                "Failure probability horizon is "
                "unknown; annualized expected cost "
                "cannot be calculated reliably."
            )

        if (
            consequence.cost_basis
            == FailureCostBasis.COMPONENTIZED
            and self.config
            .warn_component_double_count
        ):
            if (
                consequence.repair_cost > 0
                and (
                    consequence.parts_cost > 0
                    or consequence.labor_cost > 0
                )
            ):
                warnings.append(
                    "Verify repair_cost excludes "
                    "parts and labor before adding "
                    "those components separately."
                )

            if (
                consequence.downtime_cost > 0
                and consequence.production_loss > 0
            ):
                warnings.append(
                    "Downtime cost and production "
                    "loss are both included. Confirm "
                    "downtime cost does not already "
                    "contain lost production."
                )

            if (
                consequence.emergency_cost > 0
                and consequence.safety_cost > 0
            ):
                warnings.append(
                    "Verify emergency_cost and "
                    "safety_cost represent separate "
                    "economic consequences."
                )

        if (
            consequence.insurance_recovery
            + consequence.warranty_recovery
            > gross_consequence
        ):
            warnings.append(
                "Recovery credits exceed gross "
                "failure consequence; net cost was "
                "floored at zero."
            )

        return ExpectedFailureCostResult(
            calculation_id=new_id(),

            asset_id=asset_id,
            component_id=component_id,
            site_id=site_id,

            failure_probability=round(
                failure.probability,
                6,
            ),

            horizon_hours=(
                round(
                    failure.horizon_hours,
                    4,
                )
                if failure.horizon_hours
                is not None
                else None
            ),

            severity=severity,

            gross_failure_consequence=round(
                gross_consequence,
                2,
            ),

            recovery_credit=round(
                recovery_credit,
                2,
            ),

            net_failure_consequence=round(
                net_consequence,
                2,
            ),

            gross_expected_failure_cost=round(
                gross_expected,
                2,
            ),

            expected_recovery_credit=round(
                expected_recovery,
                2,
            ),

            expected_failure_cost=round(
                expected_cost,
                2,
            ),

            expected_cost_per_hour=(
                round(
                    expected_per_hour,
                    4,
                )
                if expected_per_hour
                is not None
                else None
            ),

            expected_annualized_cost=(
                round(annualized, 2)
                if annualized is not None
                else None
            ),

            breakdown=breakdown,

            economic_risk=economic_risk,

            confidence=confidence,

            confidence_level=(
                confidence_level
            ),

            currency=(
                consequence.currency
                or self.config.default_currency
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
    # Intervention economics
    # =====================================================

    def compare_intervention(
        self,
        *,
        failure: FailureProbability,
        consequence: FailureConsequenceCost,
        intervention: FailureIntervention,
    ) -> FailureInterventionResult:

        warnings: list[str] = []

        effective_post = (
            self.effective_post_probability(
                baseline_probability=(
                    failure.probability
                ),
                target_probability=(
                    intervention
                    .post_failure_probability
                ),
                success_probability=(
                    intervention
                    .success_probability
                ),
            )
        )

        baseline_expected = (
            failure.probability
            * consequence.net_cost
        )

        post_expected = (
            effective_post
            * consequence.net_cost
        )

        gross_avoided = max(
            0.0,
            baseline_expected
            - post_expected,
        )

        total_intervention_cost = sum(
            (
                intervention.intervention_cost,
                intervention
                .intervention_downtime_cost,
                intervention
                .additional_operating_cost,
            )
        )

        net_avoided = (
            gross_avoided
            - total_intervention_cost
        )

        bcr = None
        roi = None

        if total_intervention_cost > 0:
            bcr = (
                gross_avoided
                / total_intervention_cost
            )

            roi = (
                net_avoided
                / total_intervention_cost
                * 100.0
            )

        absolute_reduction = max(
            0.0,
            failure.probability
            - effective_post,
        )

        reduction_percent = (
            safe_divide(
                absolute_reduction,
                failure.probability,
            )
            * 100.0
        )

        if (
            effective_post
            >= failure.probability
        ):
            warnings.append(
                "Intervention does not reduce "
                "effective failure probability."
            )

        if net_avoided < 0:
            warnings.append(
                "Intervention cost exceeds expected "
                "avoided failure cost."
            )

        if intervention.confidence < 0.50:
            warnings.append(
                "Intervention-effect estimate has "
                "low confidence."
            )

        return FailureInterventionResult(
            baseline_expected_failure_cost=round(
                baseline_expected,
                2,
            ),

            target_post_probability=round(
                intervention
                .post_failure_probability,
                6,
            ),

            effective_post_probability=round(
                effective_post,
                6,
            ),

            post_expected_failure_cost=round(
                post_expected,
                2,
            ),

            gross_avoided_failure_cost=round(
                gross_avoided,
                2,
            ),

            intervention_cost=round(
                intervention.intervention_cost,
                2,
            ),

            total_intervention_cost=round(
                total_intervention_cost,
                2,
            ),

            net_avoided_cost=round(
                net_avoided,
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

            risk_reduction_percent=round(
                reduction_percent,
                2,
            ),

            economically_positive=(
                net_avoided > 0
            ),

            currency=consequence.currency,

            warnings=warnings,
        )

    # =====================================================
    # Scenario analysis
    # =====================================================

    def scenario_analysis(
        self,
        scenarios: Sequence[
            FailureScenario
        ],
    ) -> list[FailureScenarioResult]:

        results: list[
            FailureScenarioResult
        ] = []

        for scenario in scenarios:
            expected = (
                scenario.probability
                * scenario.consequence.net_cost
            )

            weighted = (
                expected
                * scenario.scenario_weight
            )

            results.append(
                FailureScenarioResult(
                    name=scenario.name,

                    probability=round(
                        scenario.probability,
                        6,
                    ),

                    consequence_cost=round(
                        scenario
                        .consequence.net_cost,
                        2,
                    ),

                    expected_cost=round(
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

                    severity=scenario.severity,

                    currency=(
                        scenario
                        .consequence.currency
                    ),
                )
            )

        return results

    # =====================================================
    # Weighted scenario cost
    # =====================================================

    def weighted_scenario_cost(
        self,
        scenarios: Sequence[
            FailureScenario
        ],
    ) -> float:

        results = self.scenario_analysis(
            scenarios
        )

        total_weight = sum(
            result.scenario_weight
            for result in results
        )

        if total_weight <= 0:
            return 0.0

        return round(
            sum(
                result.weighted_expected_cost
                for result in results
            )
            / total_weight,
            2,
        )

    # =====================================================
    # Break-even intervention cost
    # =====================================================

    def break_even_intervention_cost(
        self,
        *,
        failure: FailureProbability,
        consequence: FailureConsequenceCost,
        target_probability: float,
        success_probability: float = 1.0,
    ) -> float:

        effective_post = (
            self.effective_post_probability(
                baseline_probability=(
                    failure.probability
                ),
                target_probability=(
                    target_probability
                ),
                success_probability=(
                    success_probability
                ),
            )
        )

        baseline = (
            failure.probability
            * consequence.net_cost
        )

        post = (
            effective_post
            * consequence.net_cost
        )

        return round(
            max(
                0.0,
                baseline - post,
            ),
            2,
        )

    # =====================================================
    # Required probability reduction
    # =====================================================

    @staticmethod
    def required_probability_reduction(
        *,
        intervention_cost: float,
        failure_consequence_cost: float,
    ) -> float | None:

        consequence = non_negative(
            failure_consequence_cost
        )

        if consequence <= 0:
            return None

        return clamp(
            non_negative(intervention_cost)
            / consequence
        )

    # =====================================================
    # Portfolio aggregation
    # =====================================================

    @staticmethod
    def aggregate(
        results: Sequence[
            ExpectedFailureCostResult
        ],
    ) -> dict[str, Any]:

        if not results:
            return {
                "asset_count": 0,
                "total_expected_failure_cost": 0.0,
                "maximum_expected_failure_cost": 0.0,
                "average_expected_failure_cost": 0.0,
                "currency": None,
            }

        total = sum(
            result.expected_failure_cost
            for result in results
        )

        maximum = max(
            result.expected_failure_cost
            for result in results
        )

        currencies = {
            result.currency
            for result in results
        }

        risk_rank = {
            FailureEconomicRisk.LOW: 1,
            FailureEconomicRisk.MODERATE: 2,
            FailureEconomicRisk.HIGH: 3,
            FailureEconomicRisk.CRITICAL: 4,
            FailureEconomicRisk.EXTREME: 5,
        }

        highest = max(
            results,
            key=lambda item: risk_rank[
                item.economic_risk
            ],
        )

        return {
            "asset_count": len(results),

            "total_expected_failure_cost": round(
                total,
                2,
            ),

            "maximum_expected_failure_cost": round(
                maximum,
                2,
            ),

            "average_expected_failure_cost": round(
                total / len(results),
                2,
            ),

            "highest_economic_risk":
                highest.economic_risk.value,

            "highest_risk_asset_id":
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

expected_failure_cost_calculator = (
    ExpectedFailureCostCalculator()
)


# =========================================================
# Agent-facing helpers
# =========================================================

def calculate_expected_failure_cost(
    *,
    failure: FailureProbability,
    consequence: FailureConsequenceCost,
    severity: FailureSeverity = (
        FailureSeverity.MODERATE
    ),
    asset_id: str | None = None,
    component_id: str | None = None,
    site_id: str | None = None,
    reference_asset_value: float | None = None,
) -> ExpectedFailureCostResult:

    return (
        expected_failure_cost_calculator
        .calculate(
            failure=failure,
            consequence=consequence,
            severity=severity,
            asset_id=asset_id,
            component_id=component_id,
            site_id=site_id,
            reference_asset_value=(
                reference_asset_value
            ),
        )
    )


def compare_failure_intervention(
    *,
    failure: FailureProbability,
    consequence: FailureConsequenceCost,
    intervention: FailureIntervention,
) -> FailureInterventionResult:

    return (
        expected_failure_cost_calculator
        .compare_intervention(
            failure=failure,
            consequence=consequence,
            intervention=intervention,
        )
    )


def calculate_failure_break_even_cost(
    *,
    failure: FailureProbability,
    consequence: FailureConsequenceCost,
    target_probability: float,
    success_probability: float = 1.0,
) -> float:

    return (
        expected_failure_cost_calculator
        .break_even_intervention_cost(
            failure=failure,
            consequence=consequence,
            target_probability=(
                target_probability
            ),
            success_probability=(
                success_probability
            ),
        )
    )


# =========================================================
# Exports
# =========================================================

__all__ = [
    "FailureCostCategory",
    "FailureCostBasis",
    "FailureSeverity",
    "FailureEconomicRisk",
    "FailureCostConfidence",

    "FailureProbability",
    "FailureConsequenceCost",

    "ExpectedFailureCostBreakdown",
    "ExpectedFailureCostResult",

    "FailureScenario",
    "FailureScenarioResult",

    "FailureIntervention",
    "FailureInterventionResult",

    "ExpectedFailureCostConfig",
    "ExpectedFailureCostCalculator",

    "expected_failure_cost_calculator",

    "calculate_expected_failure_cost",
    "compare_failure_intervention",
    "calculate_failure_break_even_cost",
]