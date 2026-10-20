# app/agents/tools/repair_vs_replacement.py

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


def ensure_utc(value: datetime | None) -> datetime:
    if value is None:
        return utc_now()

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

class Decision(str, Enum):
    REPAIR = "repair"
    REPAIR_AND_MONITOR = "repair_and_monitor"
    OVERHAUL = "overhaul"
    REPLACE = "replace"
    DEFER = "defer"
    HUMAN_REVIEW = "human_review"


class DecisionStrength(str, Enum):
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


class EconomicRisk(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class SafetyCriticality(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ConfidenceLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class CostBasis(str, Enum):
    DIRECT = "direct"
    RISK_ADJUSTED = "risk_adjusted"
    LIFECYCLE = "lifecycle"


class OptionType(str, Enum):
    REPAIR = "repair"
    OVERHAUL = "overhaul"
    REPLACEMENT = "replacement"


# =========================================================
# Common cost model
# =========================================================

@dataclass(slots=True)
class CostProfile:
    direct_cost: float = 0.0

    labor_cost: float = 0.0
    parts_cost: float = 0.0

    downtime_cost: float = 0.0
    production_loss: float = 0.0

    logistics_cost: float = 0.0
    installation_cost: float = 0.0

    testing_cost: float = 0.0
    commissioning_cost: float = 0.0

    disposal_cost: float = 0.0

    other_cost: float = 0.0

    credits: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "direct_cost",
            "labor_cost",
            "parts_cost",
            "downtime_cost",
            "production_loss",
            "logistics_cost",
            "installation_cost",
            "testing_cost",
            "commissioning_cost",
            "disposal_cost",
            "other_cost",
            "credits",
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
                self.direct_cost,
                self.labor_cost,
                self.parts_cost,
                self.downtime_cost,
                self.production_loss,
                self.logistics_cost,
                self.installation_cost,
                self.testing_cost,
                self.commissioning_cost,
                self.disposal_cost,
                self.other_cost,
            )
        )

    @property
    def net_cost(self) -> float:
        return max(
            0.0,
            self.gross_cost - self.credits,
        )


# =========================================================
# Repair option
# =========================================================

@dataclass(slots=True)
class RepairOption:
    costs: CostProfile

    success_probability: float = 1.0

    post_repair_failure_probability: float = 0.0
    future_failure_cost: float = 0.0

    life_extension_years: float = 1.0

    annual_maintenance_cost: float = 0.0
    annual_energy_cost: float = 0.0

    repeated_failure_probability: float = 0.0
    repeated_failure_cost: float = 0.0

    salvage_value_after_horizon: float = 0.0

    confidence: float = 1.0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.success_probability = clamp(
            self.success_probability
        )

        self.post_repair_failure_probability = clamp(
            self.post_repair_failure_probability
        )

        self.repeated_failure_probability = clamp(
            self.repeated_failure_probability
        )

        self.future_failure_cost = non_negative(
            self.future_failure_cost
        )

        self.life_extension_years = non_negative(
            self.life_extension_years
        )

        self.annual_maintenance_cost = non_negative(
            self.annual_maintenance_cost
        )

        self.annual_energy_cost = non_negative(
            self.annual_energy_cost
        )

        self.repeated_failure_cost = non_negative(
            self.repeated_failure_cost
        )

        self.salvage_value_after_horizon = non_negative(
            self.salvage_value_after_horizon
        )

        self.confidence = clamp(
            self.confidence
        )


# =========================================================
# Overhaul option
# =========================================================

@dataclass(slots=True)
class OverhaulOption:
    costs: CostProfile

    success_probability: float = 1.0

    post_overhaul_failure_probability: float = 0.0
    future_failure_cost: float = 0.0

    life_extension_years: float = 3.0

    annual_maintenance_cost: float = 0.0
    annual_energy_cost: float = 0.0

    salvage_value_after_horizon: float = 0.0

    confidence: float = 1.0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.success_probability = clamp(
            self.success_probability
        )

        self.post_overhaul_failure_probability = clamp(
            self.post_overhaul_failure_probability
        )

        self.future_failure_cost = non_negative(
            self.future_failure_cost
        )

        self.life_extension_years = non_negative(
            self.life_extension_years
        )

        self.annual_maintenance_cost = non_negative(
            self.annual_maintenance_cost
        )

        self.annual_energy_cost = non_negative(
            self.annual_energy_cost
        )

        self.salvage_value_after_horizon = non_negative(
            self.salvage_value_after_horizon
        )

        self.confidence = clamp(
            self.confidence
        )


# =========================================================
# Replacement option
# =========================================================

@dataclass(slots=True)
class ReplacementOption:
    costs: CostProfile

    useful_life_years: float = 10.0

    annual_maintenance_cost: float = 0.0
    annual_energy_cost: float = 0.0

    residual_failure_probability: float = 0.0
    residual_failure_cost: float = 0.0

    salvage_value: float = 0.0

    old_asset_disposal_credit: float = 0.0

    efficiency_savings_per_year: float = 0.0
    productivity_savings_per_year: float = 0.0

    confidence: float = 1.0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.useful_life_years = non_negative(
            self.useful_life_years
        )

        self.annual_maintenance_cost = non_negative(
            self.annual_maintenance_cost
        )

        self.annual_energy_cost = non_negative(
            self.annual_energy_cost
        )

        self.residual_failure_probability = clamp(
            self.residual_failure_probability
        )

        self.residual_failure_cost = non_negative(
            self.residual_failure_cost
        )

        self.salvage_value = non_negative(
            self.salvage_value
        )

        self.old_asset_disposal_credit = non_negative(
            self.old_asset_disposal_credit
        )

        self.efficiency_savings_per_year = non_negative(
            self.efficiency_savings_per_year
        )

        self.productivity_savings_per_year = non_negative(
            self.productivity_savings_per_year
        )

        self.confidence = clamp(
            self.confidence
        )


# =========================================================
# Asset context
# =========================================================

@dataclass(slots=True)
class AssetDecisionContext:
    asset_id: str

    component_id: str | None = None
    site_id: str | None = None

    asset_age_years: float = 0.0
    expected_life_years: float = 0.0

    current_asset_value: float = 0.0
    replacement_asset_value: float = 0.0

    health_score: float | None = None
    rul_years: float | None = None

    failure_probability: float | None = None

    safety_criticality: SafetyCriticality = (
        SafetyCriticality.LOW
    )

    critical_defect_present: bool = False

    mandatory_replacement: bool = False

    repair_prohibited: bool = False

    repeated_failure_count: int = 0

    currency: str = "INR"

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.asset_age_years = non_negative(
            self.asset_age_years
        )

        self.expected_life_years = non_negative(
            self.expected_life_years
        )

        self.current_asset_value = non_negative(
            self.current_asset_value
        )

        self.replacement_asset_value = non_negative(
            self.replacement_asset_value
        )

        if self.health_score is not None:
            self.health_score = clamp(
                self.health_score
            )

        if self.rul_years is not None:
            self.rul_years = non_negative(
                self.rul_years
            )

        if self.failure_probability is not None:
            self.failure_probability = clamp(
                self.failure_probability
            )

        self.repeated_failure_count = max(
            0,
            int(self.repeated_failure_count),
        )

    @property
    def life_consumed_ratio(self) -> float | None:
        if self.expected_life_years <= 0:
            return None

        return self.asset_age_years / self.expected_life_years


# =========================================================
# Evaluation result
# =========================================================

@dataclass(slots=True)
class OptionEvaluation:
    option: OptionType

    direct_cost: float

    expected_failure_cost: float

    expected_repeat_failure_cost: float

    lifecycle_operating_cost: float

    lifecycle_savings: float

    salvage_credit: float

    present_value_cost: float

    equivalent_annual_cost: float | None

    effective_life_years: float

    confidence: float

    risk_adjusted_cost: float

    warnings: list[str] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# =========================================================
# Decision result
# =========================================================

@dataclass(slots=True)
class RepairReplacementResult:
    decision_id: str

    asset_id: str
    component_id: str | None
    site_id: str | None

    decision: Decision
    decision_strength: DecisionStrength

    cost_basis: CostBasis

    repair: OptionEvaluation
    replacement: OptionEvaluation

    overhaul: OptionEvaluation | None

    preferred_option_cost: float

    alternative_cost: float

    economic_advantage: float
    economic_advantage_percent: float

    repair_to_replacement_ratio: float

    asset_life_consumed_ratio: float | None

    economic_risk: EconomicRisk

    confidence: float
    confidence_level: ConfidenceLevel

    safety_override: bool
    human_review_required: bool

    reasons: list[str]
    warnings: list[str]

    currency: str

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
class RepairReplacementConfig:
    discount_rate: float = 0.08

    analysis_horizon_years: float = 5.0

    repair_ratio_warning: float = 0.50
    replacement_ratio_threshold: float = 0.70

    minimum_repair_success_probability: float = 0.60

    high_repeat_failure_probability: float = 0.35

    high_life_consumed_ratio: float = 0.80

    low_rul_years: float = 1.0

    minimum_confidence: float = 0.50

    strong_advantage_percent: float = 20.0
    very_strong_advantage_percent: float = 40.0

    default_currency: str = "INR"


# =========================================================
# Decision engine
# =========================================================

class RepairVsReplacementCalculator:

    def __init__(
        self,
        *,
        config: RepairReplacementConfig | None = None,
    ) -> None:
        self.config = (
            config
            or RepairReplacementConfig()
        )

    # =====================================================
    # Finance
    # =====================================================

    def present_value_annuity(
        self,
        *,
        annual_cost: float,
        years: float,
    ) -> float:

        annual = non_negative(annual_cost)
        years = non_negative(years)

        if annual <= 0 or years <= 0:
            return 0.0

        rate = finite(
            self.config.discount_rate
        )

        if abs(rate) <= 1e-12:
            return annual * years

        if rate <= -1.0:
            return annual * years

        factor = (
            1.0
            - (1.0 + rate) ** (-years)
        ) / rate

        return annual * factor

    def present_value_single(
        self,
        *,
        amount: float,
        years: float,
    ) -> float:

        amount = non_negative(amount)
        years = non_negative(years)

        rate = finite(
            self.config.discount_rate
        )

        if abs(rate) <= 1e-12:
            return amount

        if rate <= -1.0:
            return amount

        return amount / (
            (1.0 + rate) ** years
        )

    def equivalent_annual_cost(
        self,
        *,
        present_value_cost: float,
        years: float,
    ) -> float | None:

        pv = non_negative(
            present_value_cost
        )

        years = non_negative(years)

        if years <= 0:
            return None

        rate = finite(
            self.config.discount_rate
        )

        if abs(rate) <= 1e-12:
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
    # Repair
    # =====================================================

    def evaluate_repair(
        self,
        option: RepairOption,
    ) -> OptionEvaluation:

        warnings: list[str] = []

        horizon = min(
            self.config.analysis_horizon_years,
            option.life_extension_years,
        )

        direct = option.costs.net_cost

        expected_failure = (
            option.post_repair_failure_probability
            * option.future_failure_cost
        )

        failed_repair_cost = (
            (1.0 - option.success_probability)
            * option.future_failure_cost
        )

        expected_repeat = (
            option.repeated_failure_probability
            * option.repeated_failure_cost
        )

        annual_operating = (
            option.annual_maintenance_cost
            + option.annual_energy_cost
        )

        lifecycle_operating = (
            self.present_value_annuity(
                annual_cost=annual_operating,
                years=horizon,
            )
        )

        salvage = self.present_value_single(
            amount=option.salvage_value_after_horizon,
            years=horizon,
        )

        pv = max(
            0.0,
            direct
            + expected_failure
            + failed_repair_cost
            + expected_repeat
            + lifecycle_operating
            - salvage,
        )

        eac = self.equivalent_annual_cost(
            present_value_cost=pv,
            years=horizon,
        )

        if option.success_probability < (
            self.config
            .minimum_repair_success_probability
        ):
            warnings.append(
                "Repair success probability is below "
                "the configured minimum."
            )

        if (
            option.repeated_failure_probability
            >= self.config
            .high_repeat_failure_probability
        ):
            warnings.append(
                "Repair retains high repeat-failure risk."
            )

        if horizon <= 0:
            warnings.append(
                "Repair provides no usable life extension."
            )

        return OptionEvaluation(
            option=OptionType.REPAIR,

            direct_cost=round(direct, 2),

            expected_failure_cost=round(
                expected_failure
                + failed_repair_cost,
                2,
            ),

            expected_repeat_failure_cost=round(
                expected_repeat,
                2,
            ),

            lifecycle_operating_cost=round(
                lifecycle_operating,
                2,
            ),

            lifecycle_savings=0.0,

            salvage_credit=round(
                salvage,
                2,
            ),

            present_value_cost=round(
                pv,
                2,
            ),

            equivalent_annual_cost=(
                round(eac, 2)
                if eac is not None
                else None
            ),

            effective_life_years=round(
                horizon,
                4,
            ),

            confidence=round(
                option.confidence,
                4,
            ),

            risk_adjusted_cost=round(
                pv,
                2,
            ),

            warnings=warnings,
        )

    # =====================================================
    # Overhaul
    # =====================================================

    def evaluate_overhaul(
        self,
        option: OverhaulOption,
    ) -> OptionEvaluation:

        warnings: list[str] = []

        horizon = min(
            self.config.analysis_horizon_years,
            option.life_extension_years,
        )

        direct = option.costs.net_cost

        expected_failure = (
            option.post_overhaul_failure_probability
            * option.future_failure_cost
        )

        failed_overhaul_cost = (
            (1.0 - option.success_probability)
            * option.future_failure_cost
        )

        annual_operating = (
            option.annual_maintenance_cost
            + option.annual_energy_cost
        )

        lifecycle_operating = (
            self.present_value_annuity(
                annual_cost=annual_operating,
                years=horizon,
            )
        )

        salvage = self.present_value_single(
            amount=option.salvage_value_after_horizon,
            years=horizon,
        )

        pv = max(
            0.0,
            direct
            + expected_failure
            + failed_overhaul_cost
            + lifecycle_operating
            - salvage,
        )

        eac = self.equivalent_annual_cost(
            present_value_cost=pv,
            years=horizon,
        )

        if option.success_probability < 0.70:
            warnings.append(
                "Overhaul success probability is relatively low."
            )

        return OptionEvaluation(
            option=OptionType.OVERHAUL,

            direct_cost=round(direct, 2),

            expected_failure_cost=round(
                expected_failure
                + failed_overhaul_cost,
                2,
            ),

            expected_repeat_failure_cost=0.0,

            lifecycle_operating_cost=round(
                lifecycle_operating,
                2,
            ),

            lifecycle_savings=0.0,

            salvage_credit=round(
                salvage,
                2,
            ),

            present_value_cost=round(
                pv,
                2,
            ),

            equivalent_annual_cost=(
                round(eac, 2)
                if eac is not None
                else None
            ),

            effective_life_years=round(
                horizon,
                4,
            ),

            confidence=round(
                option.confidence,
                4,
            ),

            risk_adjusted_cost=round(
                pv,
                2,
            ),

            warnings=warnings,
        )

    # =====================================================
    # Replacement
    # =====================================================

    def evaluate_replacement(
        self,
        option: ReplacementOption,
    ) -> OptionEvaluation:

        warnings: list[str] = []

        horizon = min(
            self.config.analysis_horizon_years,
            option.useful_life_years,
        )

        direct = max(
            0.0,
            option.costs.net_cost
            - option.old_asset_disposal_credit,
        )

        expected_failure = (
            option.residual_failure_probability
            * option.residual_failure_cost
        )

        annual_operating = (
            option.annual_maintenance_cost
            + option.annual_energy_cost
        )

        lifecycle_operating = (
            self.present_value_annuity(
                annual_cost=annual_operating,
                years=horizon,
            )
        )

        annual_savings = (
            option.efficiency_savings_per_year
            + option.productivity_savings_per_year
        )

        lifecycle_savings = (
            self.present_value_annuity(
                annual_cost=annual_savings,
                years=horizon,
            )
        )

        salvage = self.present_value_single(
            amount=option.salvage_value,
            years=horizon,
        )

        pv = max(
            0.0,
            direct
            + expected_failure
            + lifecycle_operating
            - lifecycle_savings
            - salvage,
        )

        eac = self.equivalent_annual_cost(
            present_value_cost=pv,
            years=horizon,
        )

        if option.useful_life_years <= 0:
            warnings.append(
                "Replacement useful life is unavailable."
            )

        return OptionEvaluation(
            option=OptionType.REPLACEMENT,

            direct_cost=round(direct, 2),

            expected_failure_cost=round(
                expected_failure,
                2,
            ),

            expected_repeat_failure_cost=0.0,

            lifecycle_operating_cost=round(
                lifecycle_operating,
                2,
            ),

            lifecycle_savings=round(
                lifecycle_savings,
                2,
            ),

            salvage_credit=round(
                salvage,
                2,
            ),

            present_value_cost=round(
                pv,
                2,
            ),

            equivalent_annual_cost=(
                round(eac, 2)
                if eac is not None
                else None
            ),

            effective_life_years=round(
                horizon,
                4,
            ),

            confidence=round(
                option.confidence,
                4,
            ),

            risk_adjusted_cost=round(
                pv,
                2,
            ),

            warnings=warnings,
        )

    # =====================================================
    # Confidence
    # =====================================================

    @staticmethod
    def confidence(
        evaluations: Sequence[
            OptionEvaluation
        ],
    ) -> tuple[float, ConfidenceLevel]:

        if not evaluations:
            return 0.0, ConfidenceLevel.LOW

        score = clamp(
            sum(
                item.confidence
                for item in evaluations
            )
            / len(evaluations)
        )

        if score >= 0.90:
            level = ConfidenceLevel.VERY_HIGH
        elif score >= 0.75:
            level = ConfidenceLevel.HIGH
        elif score >= 0.50:
            level = ConfidenceLevel.MODERATE
        else:
            level = ConfidenceLevel.LOW

        return round(score, 4), level

    # =====================================================
    # Decision strength
    # =====================================================

    def decision_strength(
        self,
        advantage_percent: float,
    ) -> DecisionStrength:

        value = abs(
            finite(advantage_percent)
        )

        if value >= (
            self.config
            .very_strong_advantage_percent
        ):
            return DecisionStrength.VERY_STRONG

        if value >= (
            self.config
            .strong_advantage_percent
        ):
            return DecisionStrength.STRONG

        if value >= 10.0:
            return DecisionStrength.MODERATE

        return DecisionStrength.WEAK

    # =====================================================
    # Economic risk
    # =====================================================

    def economic_risk(
        self,
        *,
        repair: OptionEvaluation,
        replacement: OptionEvaluation,
    ) -> EconomicRisk:

        denominator = (
            replacement.present_value_cost
        )

        if denominator <= 0:
            return EconomicRisk.MODERATE

        ratio = (
            repair.present_value_cost
            / denominator
        )

        if ratio >= 0.90:
            return EconomicRisk.CRITICAL

        if ratio >= 0.70:
            return EconomicRisk.HIGH

        if ratio >= 0.50:
            return EconomicRisk.MODERATE

        return EconomicRisk.LOW

    # =====================================================
    # Main decision
    # =====================================================

    def calculate(
        self,
        *,
        asset: AssetDecisionContext,
        repair: RepairOption,
        replacement: ReplacementOption,
        overhaul: OverhaulOption | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RepairReplacementResult:

        reasons: list[str] = []
        warnings: list[str] = []

        repair_eval = self.evaluate_repair(
            repair
        )

        replacement_eval = (
            self.evaluate_replacement(
                replacement
            )
        )

        overhaul_eval = (
            self.evaluate_overhaul(overhaul)
            if overhaul is not None
            else None
        )

        evaluations = [
            repair_eval,
            replacement_eval,
        ]

        if overhaul_eval is not None:
            evaluations.append(
                overhaul_eval
            )

        confidence, confidence_level = (
            self.confidence(evaluations)
        )

        life_ratio = (
            asset.life_consumed_ratio
        )

        # ---------------------------------------------
        # Safety override
        # ---------------------------------------------

        safety_override = False
        human_review = False

        if asset.mandatory_replacement:
            decision = Decision.REPLACE
            safety_override = True

            reasons.append(
                "Replacement is mandatory under the "
                "asset safety/engineering context."
            )

        elif asset.repair_prohibited:
            decision = Decision.REPLACE
            safety_override = True

            reasons.append(
                "Repair is prohibited for this asset "
                "or failure condition."
            )

        elif (
            asset.critical_defect_present
            and asset.safety_criticality
            == SafetyCriticality.CRITICAL
        ):
            # Economics must not autonomously resolve
            # a critical safety case.
            decision = Decision.HUMAN_REVIEW
            safety_override = True
            human_review = True

            reasons.append(
                "Critical safety defect requires "
                "engineering/human authorization."
            )

        elif (
            confidence
            < self.config.minimum_confidence
        ):
            decision = Decision.HUMAN_REVIEW
            human_review = True

            reasons.append(
                "Decision confidence is below the "
                "configured minimum."
            )

        else:
            # -----------------------------------------
            # Compare equivalent annual cost when
            # option lives differ.
            # -----------------------------------------

            candidates: list[
                tuple[Decision, float]
            ] = []

            repair_metric = (
                repair_eval.equivalent_annual_cost
                if repair_eval.equivalent_annual_cost
                is not None
                else repair_eval.present_value_cost
            )

            replacement_metric = (
                replacement_eval.equivalent_annual_cost
                if replacement_eval.equivalent_annual_cost
                is not None
                else replacement_eval.present_value_cost
            )

            candidates.append(
                (
                    Decision.REPAIR,
                    repair_metric,
                )
            )

            candidates.append(
                (
                    Decision.REPLACE,
                    replacement_metric,
                )
            )

            if overhaul_eval is not None:
                overhaul_metric = (
                    overhaul_eval
                    .equivalent_annual_cost
                    if (
                        overhaul_eval
                        .equivalent_annual_cost
                        is not None
                    )
                    else overhaul_eval
                    .present_value_cost
                )

                candidates.append(
                    (
                        Decision.OVERHAUL,
                        overhaul_metric,
                    )
                )

            candidates.sort(
                key=lambda item: item[1]
            )

            decision = candidates[0][0]

            # -----------------------------------------
            # Reliability overrides
            # -----------------------------------------

            if (
                repair.success_probability
                < self.config
                .minimum_repair_success_probability
                and decision == Decision.REPAIR
            ):
                decision = (
                    Decision.REPAIR_AND_MONITOR
                )

                reasons.append(
                    "Repair is economically attractive "
                    "but has low success probability."
                )

            if (
                repair.repeated_failure_probability
                >= self.config
                .high_repeat_failure_probability
                and decision == Decision.REPAIR
            ):
                decision = (
                    Decision.REPAIR_AND_MONITOR
                )

                reasons.append(
                    "Repair retains significant "
                    "repeat-failure risk."
                )

            if (
                life_ratio is not None
                and life_ratio
                >= self.config
                .high_life_consumed_ratio
            ):
                reasons.append(
                    "Asset has consumed a high "
                    "percentage of expected useful life."
                )

            if (
                asset.rul_years is not None
                and asset.rul_years
                <= self.config.low_rul_years
            ):
                reasons.append(
                    "Remaining useful life is low."
                )

            if (
                asset.repeated_failure_count >= 3
            ):
                reasons.append(
                    "Asset has repeated historical failures."
                )

                if decision == Decision.REPAIR:
                    decision = (
                        Decision.REPAIR_AND_MONITOR
                    )

        # ---------------------------------------------
        # Economic comparison
        # ---------------------------------------------

        repair_cost = (
            repair_eval.present_value_cost
        )

        replacement_cost = (
            replacement_eval.present_value_cost
        )

        ratio = safe_divide(
            repair_cost,
            replacement_cost,
            default=0.0,
        )

        if decision in {
            Decision.REPAIR,
            Decision.REPAIR_AND_MONITOR,
        }:
            preferred = repair_cost
            alternative = replacement_cost

        elif decision == Decision.OVERHAUL:
            preferred = (
                overhaul_eval.present_value_cost
                if overhaul_eval is not None
                else repair_cost
            )
            alternative = min(
                repair_cost,
                replacement_cost,
            )

        elif decision == Decision.REPLACE:
            preferred = replacement_cost
            alternative = repair_cost

        else:
            preferred = min(
                repair_cost,
                replacement_cost,
            )
            alternative = max(
                repair_cost,
                replacement_cost,
            )

        advantage = (
            alternative - preferred
        )

        advantage_percent = (
            safe_divide(
                advantage,
                alternative,
            )
            * 100.0
        )

        strength = self.decision_strength(
            advantage_percent
        )

        risk = self.economic_risk(
            repair=repair_eval,
            replacement=replacement_eval,
        )

        # ---------------------------------------------
        # Reasons
        # ---------------------------------------------

        if (
            repair_cost
            < replacement_cost
        ):
            reasons.append(
                "Repair has lower lifecycle present-value "
                "cost than replacement."
            )

        elif (
            replacement_cost
            < repair_cost
        ):
            reasons.append(
                "Replacement has lower lifecycle present-value "
                "cost than repair."
            )

        else:
            reasons.append(
                "Repair and replacement lifecycle costs "
                "are approximately equal."
            )

        if ratio >= (
            self.config
            .replacement_ratio_threshold
        ):
            reasons.append(
                "Repair cost is high relative to "
                "replacement cost."
            )

        # ---------------------------------------------
        # Warnings
        # ---------------------------------------------

        warnings.extend(
            repair_eval.warnings
        )

        warnings.extend(
            replacement_eval.warnings
        )

        if overhaul_eval is not None:
            warnings.extend(
                overhaul_eval.warnings
            )

        if asset.safety_criticality in {
            SafetyCriticality.HIGH,
            SafetyCriticality.CRITICAL,
        }:
            warnings.append(
                "Economic ranking must not authorize "
                "continued operation of a safety-critical asset."
            )

        if safety_override:
            warnings.append(
                "Safety/engineering constraints override "
                "normal economic optimization."
            )

        if (
            asset.currency
            != self.config.default_currency
        ):
            # Not an error. It makes the absence of
            # automatic FX conversion explicit.
            warnings.append(
                "No currency conversion is performed by "
                "this decision engine."
            )

        return RepairReplacementResult(
            decision_id=new_id(),

            asset_id=asset.asset_id,
            component_id=asset.component_id,
            site_id=asset.site_id,

            decision=decision,

            decision_strength=strength,

            cost_basis=CostBasis.LIFECYCLE,

            repair=repair_eval,
            replacement=replacement_eval,
            overhaul=overhaul_eval,

            preferred_option_cost=round(
                preferred,
                2,
            ),

            alternative_cost=round(
                alternative,
                2,
            ),

            economic_advantage=round(
                advantage,
                2,
            ),

            economic_advantage_percent=round(
                advantage_percent,
                2,
            ),

            repair_to_replacement_ratio=round(
                ratio,
                6,
            ),

            asset_life_consumed_ratio=(
                round(life_ratio, 6)
                if life_ratio is not None
                else None
            ),

            economic_risk=risk,

            confidence=confidence,
            confidence_level=confidence_level,

            safety_override=safety_override,

            human_review_required=(
                human_review
                or decision
                == Decision.HUMAN_REVIEW
            ),

            reasons=reasons,
            warnings=list(dict.fromkeys(warnings)),

            currency=(
                asset.currency
                or self.config.default_currency
            ),

            metadata={
                **asset.metadata,
                **dict(metadata or {}),
            },
        )

    # =====================================================
    # Batch
    # =====================================================

    def calculate_many(
        self,
        cases: Sequence[
            tuple[
                AssetDecisionContext,
                RepairOption,
                ReplacementOption,
                OverhaulOption | None,
            ]
        ],
    ) -> list[RepairReplacementResult]:

        return [
            self.calculate(
                asset=asset,
                repair=repair,
                replacement=replacement,
                overhaul=overhaul,
            )
            for (
                asset,
                repair,
                replacement,
                overhaul,
            ) in cases
        ]


# =========================================================
# Default engine
# =========================================================

repair_vs_replacement_calculator = (
    RepairVsReplacementCalculator()
)


# =========================================================
# Agent-facing helpers
# =========================================================

def compare_repair_vs_replacement(
    *,
    asset: AssetDecisionContext,
    repair: RepairOption,
    replacement: ReplacementOption,
    overhaul: OverhaulOption | None = None,
) -> RepairReplacementResult:

    return (
        repair_vs_replacement_calculator
        .calculate(
            asset=asset,
            repair=repair,
            replacement=replacement,
            overhaul=overhaul,
        )
    )


def calculate_repair_replacement_ratio(
    *,
    repair_cost: float,
    replacement_cost: float,
) -> float:

    return round(
        safe_divide(
            non_negative(repair_cost),
            non_negative(replacement_cost),
        ),
        6,
    )


def should_replace_economically(
    *,
    repair_cost: float,
    replacement_cost: float,
    threshold: float = 0.70,
) -> bool:

    replacement = non_negative(
        replacement_cost
    )

    if replacement <= 0:
        return False

    ratio = (
        non_negative(repair_cost)
        / replacement
    )

    return ratio >= clamp(
        threshold,
        0.0,
        1.0,
    )


# =========================================================
# Exports
# =========================================================

__all__ = [
    "Decision",
    "DecisionStrength",
    "EconomicRisk",
    "SafetyCriticality",
    "ConfidenceLevel",
    "CostBasis",
    "OptionType",

    "CostProfile",

    "RepairOption",
    "OverhaulOption",
    "ReplacementOption",

    "AssetDecisionContext",

    "OptionEvaluation",
    "RepairReplacementResult",

    "RepairReplacementConfig",
    "RepairVsReplacementCalculator",

    "repair_vs_replacement_calculator",

    "compare_repair_vs_replacement",
    "calculate_repair_replacement_ratio",
    "should_replace_economically",
]