# app/agents/tools/asset_valuation.py

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Mapping
from uuid import UUID, uuid4


# =========================================================
# Utilities
# =========================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid4())


def clamp(
    value: float,
    low: float = 0.0,
    high: float = 1.0,
) -> float:
    return max(low, min(high, float(value)))


def finite(
    value: Any,
    *,
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
    *,
    default: float = 0.0,
) -> float:
    return max(
        0.0,
        finite(value, default=default),
    )


def safe_divide(
    numerator: float,
    denominator: float,
    *,
    default: float = 0.0,
) -> float:
    if abs(denominator) <= 1e-12:
        return default

    return numerator / denominator


def ensure_utc(
    value: datetime | None,
) -> datetime | None:
    if value is None:
        return None

    if value.tzinfo is None:
        return value.replace(
            tzinfo=timezone.utc
        )

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

class DepreciationMethod(str, Enum):
    STRAIGHT_LINE = "straight_line"
    DECLINING_BALANCE = "declining_balance"
    DOUBLE_DECLINING = "double_declining"
    UNITS_OF_PRODUCTION = "units_of_production"
    REMAINING_LIFE = "remaining_life"


class AssetValueCondition(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class EconomicRiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class EconomicDecision(str, Enum):
    RETAIN = "retain"
    MONITOR = "monitor"
    MAINTAIN = "maintain"
    REPAIR = "repair"
    OVERHAUL = "overhaul"
    REPLACE = "replace"
    HUMAN_REVIEW = "human_review"


class ValuationConfidence(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


# =========================================================
# Asset financial inputs
# =========================================================

@dataclass(slots=True)
class AssetFinancialProfile:
    asset_id: str

    asset_name: str | None = None
    asset_type: str | None = None

    site_id: str | None = None
    component: str | None = None

    purchase_price: float = 0.0

    replacement_cost: float = 0.0

    salvage_value: float = 0.0

    installation_cost: float = 0.0

    commissioning_cost: float = 0.0

    acquisition_date: datetime | None = None

    expected_life_years: float | None = None

    expected_operating_hours: float | None = None

    current_operating_hours: float | None = None

    currency: str = "INR"

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.purchase_price = non_negative(
            self.purchase_price
        )

        self.replacement_cost = non_negative(
            self.replacement_cost
        )

        self.salvage_value = non_negative(
            self.salvage_value
        )

        self.installation_cost = non_negative(
            self.installation_cost
        )

        self.commissioning_cost = non_negative(
            self.commissioning_cost
        )

        if self.expected_life_years is not None:
            self.expected_life_years = (
                non_negative(
                    self.expected_life_years
                )
            )

        if (
            self.expected_operating_hours
            is not None
        ):
            self.expected_operating_hours = (
                non_negative(
                    self.expected_operating_hours
                )
            )

        if (
            self.current_operating_hours
            is not None
        ):
            self.current_operating_hours = (
                non_negative(
                    self.current_operating_hours
                )
            )

        self.acquisition_date = ensure_utc(
            self.acquisition_date
        )


# =========================================================
# Condition
# =========================================================

@dataclass(slots=True)
class AssetConditionProfile:
    health_score: float | None = None

    failure_probability: float | None = None

    anomaly_score: float | None = None

    predicted_rul_hours: float | None = None

    degradation_score: float | None = None

    defect_severity: float | None = None

    safety_risk_score: float | None = None

    repeated_failure_count: int = 0

    critical_defect: bool = False

    obsolete: bool = False

    parts_available: bool = True

    model_confidence: float | None = None

    data_quality_score: float | None = None

    model_drift_detected: bool = False

    def __post_init__(self) -> None:
        if self.health_score is not None:
            self.health_score = max(
                0.0,
                min(
                    100.0,
                    finite(self.health_score),
                ),
            )

        for name in (
            "failure_probability",
            "anomaly_score",
            "degradation_score",
            "defect_severity",
            "safety_risk_score",
            "model_confidence",
            "data_quality_score",
        ):
            value = getattr(self, name)

            if value is not None:
                setattr(
                    self,
                    name,
                    clamp(value),
                )

        if self.predicted_rul_hours is not None:
            self.predicted_rul_hours = (
                non_negative(
                    self.predicted_rul_hours
                )
            )

        self.repeated_failure_count = max(
            0,
            int(self.repeated_failure_count),
        )


# =========================================================
# Cost profile
# =========================================================

@dataclass(slots=True)
class AssetCostProfile:
    estimated_repair_cost: float = 0.0

    estimated_overhaul_cost: float = 0.0

    annual_maintenance_cost: float = 0.0

    downtime_cost_per_hour: float = 0.0

    expected_repair_downtime_hours: float = 0.0

    expected_replacement_downtime_hours: float = 0.0

    expected_failure_cost: float = 0.0

    expected_emergency_cost: float = 0.0

    annual_energy_cost: float = 0.0

    replacement_energy_savings_per_year: float = 0.0

    replacement_maintenance_savings_per_year: float = 0.0

    repair_success_probability: float = 1.0

    repair_life_extension_years: float = 0.0

    overhaul_life_extension_years: float = 0.0

    replacement_life_years: float | None = None

    def __post_init__(self) -> None:
        monetary_fields = (
            "estimated_repair_cost",
            "estimated_overhaul_cost",
            "annual_maintenance_cost",
            "downtime_cost_per_hour",
            "expected_failure_cost",
            "expected_emergency_cost",
            "annual_energy_cost",
            "replacement_energy_savings_per_year",
            "replacement_maintenance_savings_per_year",
        )

        for name in monetary_fields:
            setattr(
                self,
                name,
                non_negative(
                    getattr(self, name)
                ),
            )

        self.expected_repair_downtime_hours = (
            non_negative(
                self.expected_repair_downtime_hours
            )
        )

        self.expected_replacement_downtime_hours = (
            non_negative(
                self.expected_replacement_downtime_hours
            )
        )

        self.repair_success_probability = clamp(
            self.repair_success_probability
        )

        self.repair_life_extension_years = (
            non_negative(
                self.repair_life_extension_years
            )
        )

        self.overhaul_life_extension_years = (
            non_negative(
                self.overhaul_life_extension_years
            )
        )

        if self.replacement_life_years is not None:
            self.replacement_life_years = (
                non_negative(
                    self.replacement_life_years
                )
            )


# =========================================================
# Depreciation
# =========================================================

@dataclass(slots=True)
class DepreciationResult:
    method: DepreciationMethod

    original_cost: float

    depreciable_amount: float

    accumulated_depreciation: float

    annual_depreciation: float | None

    book_value: float

    age_years: float

    life_consumed_ratio: float

    remaining_life_ratio: float

    currency: str

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# =========================================================
# Economic exposure
# =========================================================

@dataclass(slots=True)
class EconomicExposure:
    failure_probability: float

    expected_failure_loss: float

    expected_emergency_loss: float

    expected_downtime_loss: float

    total_expected_risk_cost: float

    currency: str

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# =========================================================
# Lifecycle option
# =========================================================

@dataclass(slots=True)
class LifecycleOption:
    name: str

    initial_cost: float

    downtime_cost: float

    residual_failure_cost: float

    annual_operating_cost: float

    present_value_operating_cost: float

    salvage_credit: float

    lifecycle_cost: float

    equivalent_annual_cost: float | None

    horizon_years: float

    currency: str

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# =========================================================
# Valuation result
# =========================================================

@dataclass(slots=True)
class AssetValuationResult:
    valuation_id: str

    asset_id: str

    currency: str

    original_cost: float

    replacement_cost: float

    book_value: float

    condition_adjusted_value: float

    risk_adjusted_value: float

    economic_value_score: float

    condition: AssetValueCondition

    economic_risk: EconomicRiskLevel

    decision: EconomicDecision

    confidence: float

    confidence_level: ValuationConfidence

    age_years: float

    life_consumed_ratio: float

    remaining_life_ratio: float

    failure_probability: float

    expected_failure_loss: float

    repair_cost_ratio: float | None

    replacement_value_gap: float

    repair_option: LifecycleOption | None

    replacement_option: LifecycleOption | None

    preferred_lifecycle_option: str | None

    reasons: list[str]

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
class AssetValuationConfig:
    default_expected_life_years: float = 10.0

    default_discount_rate: float = 0.08

    lifecycle_horizon_years: float = 5.0

    repair_ratio_warning: float = 0.40

    repair_ratio_replace: float = 0.70

    critical_health_score: float = 25.0

    poor_health_score: float = 50.0

    good_health_score: float = 75.0

    excellent_health_score: float = 90.0

    high_failure_probability: float = 0.70

    critical_failure_probability: float = 0.90

    low_rul_hours: float = 168.0

    critical_rul_hours: float = 24.0

    minimum_confidence: float = 0.50

    condition_floor: float = 0.10


# =========================================================
# Asset valuation
# =========================================================

class AssetValuation:

    def __init__(
        self,
        *,
        config: AssetValuationConfig | None = None,
    ) -> None:
        self.config = (
            config or AssetValuationConfig()
        )

    # =====================================================
    # Asset age
    # =====================================================

    def calculate_age_years(
        self,
        profile: AssetFinancialProfile,
        *,
        as_of: datetime | None = None,
    ) -> float:

        if profile.acquisition_date is None:
            return 0.0

        as_of = ensure_utc(as_of) or utc_now()

        seconds = max(
            0.0,
            (
                as_of - profile.acquisition_date
            ).total_seconds(),
        )

        return seconds / (
            365.2425 * 24 * 3600
        )

    # =====================================================
    # Life consumption
    # =====================================================

    def life_consumed_ratio(
        self,
        profile: AssetFinancialProfile,
        *,
        age_years: float | None = None,
    ) -> float:

        age = (
            self.calculate_age_years(profile)
            if age_years is None
            else non_negative(age_years)
        )

        ratios: list[float] = []

        expected_life = (
            profile.expected_life_years
            or self.config
            .default_expected_life_years
        )

        if expected_life > 0:
            ratios.append(
                age / expected_life
            )

        if (
            profile.expected_operating_hours
            is not None
            and profile.expected_operating_hours > 0
            and profile.current_operating_hours
            is not None
        ):
            ratios.append(
                profile.current_operating_hours
                / profile.expected_operating_hours
            )

        if not ratios:
            return 0.0

        # Use the most consumed dimension rather than
        # averaging calendar and usage life.
        return max(ratios)

    # =====================================================
    # Depreciation
    # =====================================================

    def depreciate(
        self,
        profile: AssetFinancialProfile,
        *,
        method: DepreciationMethod = (
            DepreciationMethod.STRAIGHT_LINE
        ),
        declining_rate: float = 0.20,
        as_of: datetime | None = None,
    ) -> DepreciationResult:

        original_cost = (
            profile.purchase_price
            + profile.installation_cost
            + profile.commissioning_cost
        )

        salvage = min(
            profile.salvage_value,
            original_cost,
        )

        depreciable = max(
            0.0,
            original_cost - salvage,
        )

        age = self.calculate_age_years(
            profile,
            as_of=as_of,
        )

        expected_life = (
            profile.expected_life_years
            or self.config
            .default_expected_life_years
        )

        consumed = self.life_consumed_ratio(
            profile,
            age_years=age,
        )

        remaining = clamp(
            1.0 - consumed
        )

        annual_depreciation: float | None

        if method == (
            DepreciationMethod.STRAIGHT_LINE
        ):
            annual_depreciation = (
                depreciable / expected_life
                if expected_life > 0
                else None
            )

            accumulated = min(
                depreciable,
                (
                    annual_depreciation * age
                    if annual_depreciation
                    is not None
                    else depreciable
                ),
            )

            book_value = max(
                salvage,
                original_cost - accumulated,
            )

        elif method == (
            DepreciationMethod
            .DECLINING_BALANCE
        ):
            rate = clamp(
                declining_rate,
                0.0,
                0.99,
            )

            book_value = max(
                salvage,
                original_cost
                * ((1.0 - rate) ** age),
            )

            accumulated = max(
                0.0,
                original_cost - book_value,
            )

            annual_depreciation = (
                book_value * rate
            )

        elif method == (
            DepreciationMethod
            .DOUBLE_DECLINING
        ):
            rate = (
                min(
                    0.99,
                    2.0 / expected_life,
                )
                if expected_life > 0
                else 0.99
            )

            book_value = max(
                salvage,
                original_cost
                * ((1.0 - rate) ** age),
            )

            accumulated = max(
                0.0,
                original_cost - book_value,
            )

            annual_depreciation = (
                book_value * rate
            )

        elif method == (
            DepreciationMethod
            .UNITS_OF_PRODUCTION
        ):
            usage_ratio = 0.0

            if (
                profile.expected_operating_hours
                and profile.current_operating_hours
                is not None
            ):
                usage_ratio = clamp(
                    profile.current_operating_hours
                    / profile
                    .expected_operating_hours
                )

            accumulated = (
                depreciable * usage_ratio
            )

            book_value = (
                original_cost - accumulated
            )

            annual_depreciation = None

        else:
            accumulated = (
                depreciable
                * clamp(consumed)
            )

            book_value = max(
                salvage,
                original_cost - accumulated,
            )

            annual_depreciation = None

        return DepreciationResult(
            method=method,

            original_cost=round(
                original_cost,
                2,
            ),

            depreciable_amount=round(
                depreciable,
                2,
            ),

            accumulated_depreciation=round(
                accumulated,
                2,
            ),

            annual_depreciation=(
                round(
                    annual_depreciation,
                    2,
                )
                if annual_depreciation
                is not None
                else None
            ),

            book_value=round(
                book_value,
                2,
            ),

            age_years=round(age, 4),

            life_consumed_ratio=round(
                consumed,
                4,
            ),

            remaining_life_ratio=round(
                remaining,
                4,
            ),

            currency=profile.currency,
        )

    # =====================================================
    # Condition
    # =====================================================

    def determine_condition(
        self,
        condition: AssetConditionProfile,
    ) -> AssetValueCondition:

        if condition.critical_defect:
            return AssetValueCondition.CRITICAL

        health = condition.health_score

        if health is None:
            if (
                condition.failure_probability
                is None
                and condition.predicted_rul_hours
                is None
                and condition.degradation_score
                is None
            ):
                return AssetValueCondition.UNKNOWN

            score = 100.0

            if (
                condition.failure_probability
                is not None
            ):
                score -= (
                    condition.failure_probability
                    * 50.0
                )

            if (
                condition.degradation_score
                is not None
            ):
                score -= (
                    condition.degradation_score
                    * 30.0
                )

            health = max(0.0, score)

        if (
            health
            <= self.config.critical_health_score
        ):
            return AssetValueCondition.CRITICAL

        if health <= self.config.poor_health_score:
            return AssetValueCondition.POOR

        if health <= self.config.good_health_score:
            return AssetValueCondition.FAIR

        if (
            health
            < self.config.excellent_health_score
        ):
            return AssetValueCondition.GOOD

        return AssetValueCondition.EXCELLENT

    # =====================================================
    # Condition multiplier
    # =====================================================

    def condition_multiplier(
        self,
        condition: AssetConditionProfile,
    ) -> float:

        if condition.health_score is None:
            multiplier = 0.75
        else:
            multiplier = (
                condition.health_score / 100.0
            )

        if (
            condition.degradation_score
            is not None
        ):
            multiplier *= (
                1.0
                - 0.30
                * condition.degradation_score
            )

        if condition.defect_severity is not None:
            multiplier *= (
                1.0
                - 0.35
                * condition.defect_severity
            )

        if condition.critical_defect:
            multiplier *= 0.50

        if condition.obsolete:
            multiplier *= 0.80

        if not condition.parts_available:
            multiplier *= 0.85

        return clamp(
            multiplier,
            self.config.condition_floor,
            1.0,
        )

    # =====================================================
    # Failure probability
    # =====================================================

    def estimate_failure_probability(
        self,
        condition: AssetConditionProfile,
    ) -> float:

        if (
            condition.failure_probability
            is not None
        ):
            return condition.failure_probability

        factors: list[float] = []

        if condition.health_score is not None:
            factors.append(
                1.0
                - condition.health_score / 100.0
            )

        if condition.anomaly_score is not None:
            factors.append(
                condition.anomaly_score
            )

        if (
            condition.degradation_score
            is not None
        ):
            factors.append(
                condition.degradation_score
            )

        if condition.defect_severity is not None:
            factors.append(
                condition.defect_severity
            )

        if (
            condition.predicted_rul_hours
            is not None
        ):
            rul = (
                condition.predicted_rul_hours
            )

            if rul <= 24:
                factors.append(0.95)
            elif rul <= 72:
                factors.append(0.80)
            elif rul <= 168:
                factors.append(0.65)
            elif rul <= 720:
                factors.append(0.35)

        if not factors:
            return 0.0

        # Noisy-OR avoids simply averaging away
        # several independent risk indicators.
        probability_no_failure = 1.0

        for factor in factors:
            probability_no_failure *= (
                1.0 - clamp(factor)
            )

        return clamp(
            1.0 - probability_no_failure
        )

    # =====================================================
    # Expected economic exposure
    # =====================================================

    def economic_exposure(
        self,
        *,
        condition: AssetConditionProfile,
        costs: AssetCostProfile,
    ) -> EconomicExposure:

        probability = (
            self.estimate_failure_probability(
                condition
            )
        )

        expected_failure = (
            probability
            * costs.expected_failure_cost
        )

        emergency_probability = (
            probability
            * (
                condition.safety_risk_score
                if condition.safety_risk_score
                is not None
                else 0.25
            )
        )

        expected_emergency = (
            emergency_probability
            * costs.expected_emergency_cost
        )

        expected_downtime = (
            probability
            * costs.downtime_cost_per_hour
            * max(
                costs
                .expected_repair_downtime_hours,
                0.0,
            )
        )

        total = (
            expected_failure
            + expected_emergency
            + expected_downtime
        )

        return EconomicExposure(
            failure_probability=round(
                probability,
                4,
            ),

            expected_failure_loss=round(
                expected_failure,
                2,
            ),

            expected_emergency_loss=round(
                expected_emergency,
                2,
            ),

            expected_downtime_loss=round(
                expected_downtime,
                2,
            ),

            total_expected_risk_cost=round(
                total,
                2,
            ),

            currency="",
        )

    # =====================================================
    # Present value
    # =====================================================

    @staticmethod
    def present_value_annuity(
        annual_amount: float,
        years: float,
        discount_rate: float,
    ) -> float:

        annual_amount = non_negative(
            annual_amount
        )

        years = non_negative(years)

        rate = max(
            0.0,
            finite(discount_rate),
        )

        if years <= 0:
            return 0.0

        full_years = int(math.floor(years))
        fractional = years - full_years

        pv = 0.0

        for year in range(
            1,
            full_years + 1,
        ):
            pv += (
                annual_amount
                / ((1.0 + rate) ** year)
            )

        if fractional > 0:
            pv += (
                annual_amount
                * fractional
                / (
                    (1.0 + rate)
                    ** (full_years + fractional)
                )
            )

        return pv

    # =====================================================
    # Equivalent annual cost
    # =====================================================

    @staticmethod
    def equivalent_annual_cost(
        present_value: float,
        years: float,
        discount_rate: float,
    ) -> float | None:

        years = non_negative(years)

        if years <= 0:
            return None

        rate = max(
            0.0,
            finite(discount_rate),
        )

        if rate <= 1e-12:
            return present_value / years

        factor = (
            rate
            * ((1.0 + rate) ** years)
            / (
                ((1.0 + rate) ** years)
                - 1.0
            )
        )

        return present_value * factor

    # =====================================================
    # Repair lifecycle option
    # =====================================================

    def repair_option(
        self,
        *,
        profile: AssetFinancialProfile,
        condition: AssetConditionProfile,
        costs: AssetCostProfile,
        horizon_years: float,
        discount_rate: float,
    ) -> LifecycleOption:

        probability = (
            self.estimate_failure_probability(
                condition
            )
        )

        downtime = (
            costs.downtime_cost_per_hour
            * costs.expected_repair_downtime_hours
        )

        residual_failure_probability = (
            probability
            * (
                1.0
                - costs.repair_success_probability
            )
        )

        residual_failure_cost = (
            residual_failure_probability
            * (
                costs.expected_failure_cost
                + costs.expected_emergency_cost
            )
        )

        annual_operating = (
            costs.annual_maintenance_cost
            + costs.annual_energy_cost
        )

        pv_operating = (
            self.present_value_annuity(
                annual_operating,
                horizon_years,
                discount_rate,
            )
        )

        lifecycle_cost = (
            costs.estimated_repair_cost
            + downtime
            + residual_failure_cost
            + pv_operating
        )

        eac = self.equivalent_annual_cost(
            lifecycle_cost,
            horizon_years,
            discount_rate,
        )

        return LifecycleOption(
            name="repair",

            initial_cost=round(
                costs.estimated_repair_cost,
                2,
            ),

            downtime_cost=round(
                downtime,
                2,
            ),

            residual_failure_cost=round(
                residual_failure_cost,
                2,
            ),

            annual_operating_cost=round(
                annual_operating,
                2,
            ),

            present_value_operating_cost=round(
                pv_operating,
                2,
            ),

            salvage_credit=0.0,

            lifecycle_cost=round(
                lifecycle_cost,
                2,
            ),

            equivalent_annual_cost=(
                round(eac, 2)
                if eac is not None
                else None
            ),

            horizon_years=round(
                horizon_years,
                4,
            ),

            currency=profile.currency,

            metadata={
                "repair_success_probability":
                    costs
                    .repair_success_probability,

                "residual_failure_probability":
                    round(
                        residual_failure_probability,
                        4,
                    ),
            },
        )

    # =====================================================
    # Replacement lifecycle option
    # =====================================================

    def replacement_option(
        self,
        *,
        profile: AssetFinancialProfile,
        costs: AssetCostProfile,
        horizon_years: float,
        discount_rate: float,
    ) -> LifecycleOption:

        replacement_capex = (
            profile.replacement_cost
            if profile.replacement_cost > 0
            else profile.purchase_price
        )

        replacement_capex += (
            profile.installation_cost
            + profile.commissioning_cost
        )

        downtime = (
            costs.downtime_cost_per_hour
            * costs
            .expected_replacement_downtime_hours
        )

        annual_operating = max(
            0.0,
            costs.annual_maintenance_cost
            + costs.annual_energy_cost
            - costs
            .replacement_energy_savings_per_year
            - costs
            .replacement_maintenance_savings_per_year
        )

        pv_operating = (
            self.present_value_annuity(
                annual_operating,
                horizon_years,
                discount_rate,
            )
        )

        salvage_credit = min(
            profile.salvage_value,
            replacement_capex,
        )

        lifecycle_cost = (
            replacement_capex
            + downtime
            + pv_operating
            - salvage_credit
        )

        eac = self.equivalent_annual_cost(
            lifecycle_cost,
            horizon_years,
            discount_rate,
        )

        return LifecycleOption(
            name="replace",

            initial_cost=round(
                replacement_capex,
                2,
            ),

            downtime_cost=round(
                downtime,
                2,
            ),

            residual_failure_cost=0.0,

            annual_operating_cost=round(
                annual_operating,
                2,
            ),

            present_value_operating_cost=round(
                pv_operating,
                2,
            ),

            salvage_credit=round(
                salvage_credit,
                2,
            ),

            lifecycle_cost=round(
                lifecycle_cost,
                2,
            ),

            equivalent_annual_cost=(
                round(eac, 2)
                if eac is not None
                else None
            ),

            horizon_years=round(
                horizon_years,
                4,
            ),

            currency=profile.currency,
        )

    # =====================================================
    # Confidence
    # =====================================================

    def valuation_confidence(
        self,
        *,
        profile: AssetFinancialProfile,
        condition: AssetConditionProfile,
    ) -> tuple[
        float,
        ValuationConfidence,
    ]:

        checks = [
            profile.purchase_price > 0,
            profile.replacement_cost > 0,
            profile.expected_life_years
            is not None,

            condition.health_score is not None,

            condition.failure_probability
            is not None,

            condition.predicted_rul_hours
            is not None,
        ]

        coverage = (
            sum(checks) / len(checks)
        )

        model_confidence = (
            condition.model_confidence
            if condition.model_confidence
            is not None
            else 0.60
        )

        quality = (
            condition.data_quality_score
            if condition.data_quality_score
            is not None
            else 0.60
        )

        score = clamp(
            0.50 * coverage
            + 0.25 * model_confidence
            + 0.25 * quality
        )

        if condition.model_drift_detected:
            score *= 0.80

        score = clamp(score)

        if score >= 0.90:
            level = (
                ValuationConfidence.VERY_HIGH
            )

        elif score >= 0.75:
            level = ValuationConfidence.HIGH

        elif score >= 0.50:
            level = (
                ValuationConfidence.MODERATE
            )

        else:
            level = ValuationConfidence.LOW

        return round(score, 4), level

    # =====================================================
    # Economic risk
    # =====================================================

    def economic_risk_level(
        self,
        *,
        failure_probability: float,
        risk_cost: float,
        replacement_cost: float,
    ) -> EconomicRiskLevel:

        ratio = safe_divide(
            risk_cost,
            replacement_cost,
            default=0.0,
        )

        score = max(
            failure_probability,
            clamp(ratio),
        )

        if score >= 0.80:
            return EconomicRiskLevel.CRITICAL

        if score >= 0.60:
            return EconomicRiskLevel.HIGH

        if score >= 0.30:
            return EconomicRiskLevel.MODERATE

        return EconomicRiskLevel.LOW

    # =====================================================
    # Main valuation
    # =====================================================

    def value_asset(
        self,
        *,
        profile: AssetFinancialProfile,
        condition: AssetConditionProfile,
        costs: AssetCostProfile | None = None,
        depreciation_method: DepreciationMethod = (
            DepreciationMethod.STRAIGHT_LINE
        ),
        discount_rate: float | None = None,
        lifecycle_horizon_years: float | None = None,
        as_of: datetime | None = None,
    ) -> AssetValuationResult:

        costs = costs or AssetCostProfile()

        rate = (
            self.config.default_discount_rate
            if discount_rate is None
            else max(
                0.0,
                finite(discount_rate),
            )
        )

        horizon = (
            lifecycle_horizon_years
            if lifecycle_horizon_years
            is not None
            else self.config
            .lifecycle_horizon_years
        )

        horizon = max(
            0.25,
            finite(horizon, default=5.0),
        )

        depreciation = self.depreciate(
            profile,
            method=depreciation_method,
            as_of=as_of,
        )

        condition_state = (
            self.determine_condition(
                condition
            )
        )

        condition_factor = (
            self.condition_multiplier(
                condition
            )
        )

        condition_adjusted_value = (
            depreciation.book_value
            * condition_factor
        )

        failure_probability = (
            self.estimate_failure_probability(
                condition
            )
        )

        exposure = self.economic_exposure(
            condition=condition,
            costs=costs,
        )

        exposure.currency = profile.currency

        # Risk adjustment is bounded so expected
        # economic exposure cannot create a negative
        # physical asset value.
        risk_adjustment = min(
            condition_adjusted_value,
            exposure.total_expected_risk_cost,
        )

        risk_adjusted_value = max(
            profile.salvage_value,
            condition_adjusted_value
            - risk_adjustment,
        )

        replacement_cost = (
            profile.replacement_cost
            if profile.replacement_cost > 0
            else depreciation.original_cost
        )

        economic_value_score = clamp(
            safe_divide(
                risk_adjusted_value,
                max(replacement_cost, 1e-9),
            )
        )

        repair_ratio = None

        if replacement_cost > 0:
            repair_ratio = (
                costs.estimated_repair_cost
                / replacement_cost
            )

        repair_option = self.repair_option(
            profile=profile,
            condition=condition,
            costs=costs,
            horizon_years=horizon,
            discount_rate=rate,
        )

        replacement_option = (
            self.replacement_option(
                profile=profile,
                costs=costs,
                horizon_years=horizon,
                discount_rate=rate,
            )
        )

        preferred_option = (
            "repair"
            if repair_option.lifecycle_cost
            <= replacement_option.lifecycle_cost
            else "replace"
        )

        confidence, confidence_level = (
            self.valuation_confidence(
                profile=profile,
                condition=condition,
            )
        )

        economic_risk = (
            self.economic_risk_level(
                failure_probability=(
                    failure_probability
                ),

                risk_cost=(
                    exposure
                    .total_expected_risk_cost
                ),

                replacement_cost=(
                    replacement_cost
                ),
            )
        )

        decision, reasons = (
            self._decision(
                profile=profile,
                condition=condition,
                costs=costs,
                condition_state=condition_state,
                failure_probability=(
                    failure_probability
                ),
                repair_ratio=repair_ratio,
                repair_option=repair_option,
                replacement_option=(
                    replacement_option
                ),
                confidence=confidence,
            )
        )

        warnings: list[str] = []

        if (
            confidence
            < self.config.minimum_confidence
        ):
            warnings.append(
                "Valuation confidence is low; "
                "human financial/engineering review "
                "is recommended."
            )

        if (
            condition.data_quality_score
            is not None
            and condition.data_quality_score < 0.50
        ):
            warnings.append(
                "Asset condition data quality is low."
            )

        if condition.model_drift_detected:
            warnings.append(
                "Model drift was detected; predicted "
                "health/failure values may be less "
                "reliable."
            )

        if profile.replacement_cost <= 0:
            warnings.append(
                "Replacement cost was unavailable; "
                "original asset cost was used as a "
                "replacement-cost proxy."
            )

        if (
            condition.safety_risk_score
            is not None
            and condition.safety_risk_score >= 0.80
        ):
            warnings.append(
                "High safety risk detected. Safety "
                "decisions must take precedence over "
                "economic optimization."
            )

        return AssetValuationResult(
            valuation_id=new_id(),

            asset_id=profile.asset_id,

            currency=profile.currency,

            original_cost=(
                depreciation.original_cost
            ),

            replacement_cost=round(
                replacement_cost,
                2,
            ),

            book_value=(
                depreciation.book_value
            ),

            condition_adjusted_value=round(
                condition_adjusted_value,
                2,
            ),

            risk_adjusted_value=round(
                risk_adjusted_value,
                2,
            ),

            economic_value_score=round(
                economic_value_score,
                4,
            ),

            condition=condition_state,

            economic_risk=economic_risk,

            decision=decision,

            confidence=confidence,

            confidence_level=(
                confidence_level
            ),

            age_years=(
                depreciation.age_years
            ),

            life_consumed_ratio=(
                depreciation
                .life_consumed_ratio
            ),

            remaining_life_ratio=(
                depreciation
                .remaining_life_ratio
            ),

            failure_probability=round(
                failure_probability,
                4,
            ),

            expected_failure_loss=(
                exposure
                .total_expected_risk_cost
            ),

            repair_cost_ratio=(
                round(repair_ratio, 4)
                if repair_ratio is not None
                else None
            ),

            replacement_value_gap=round(
                replacement_cost
                - risk_adjusted_value,
                2,
            ),

            repair_option=repair_option,

            replacement_option=(
                replacement_option
            ),

            preferred_lifecycle_option=(
                preferred_option
            ),

            reasons=reasons,

            warnings=warnings,

            metadata={
                "depreciation_method":
                    depreciation_method.value,

                "discount_rate": rate,

                "lifecycle_horizon_years":
                    horizon,

                "condition_multiplier":
                    round(
                        condition_factor,
                        4,
                    ),

                "economic_exposure":
                    exposure.to_dict(),
            },
        )

    # =====================================================
    # Decision engine
    # =====================================================

    def _decision(
        self,
        *,
        profile: AssetFinancialProfile,
        condition: AssetConditionProfile,
        costs: AssetCostProfile,
        condition_state: AssetValueCondition,
        failure_probability: float,
        repair_ratio: float | None,
        repair_option: LifecycleOption,
        replacement_option: LifecycleOption,
        confidence: float,
    ) -> tuple[
        EconomicDecision,
        list[str],
    ]:

        reasons: list[str] = []

        # Safety is a constraint, not a financial
        # trade-off.
        if (
            condition.safety_risk_score
            is not None
            and condition.safety_risk_score >= 0.90
        ):
            reasons.append(
                "Critical safety risk requires "
                "engineering/safety review before "
                "economic optimization."
            )

            return (
                EconomicDecision.HUMAN_REVIEW,
                reasons,
            )

        if confidence < 0.35:
            reasons.append(
                "Insufficient reliable data for an "
                "automatic economic decision."
            )

            return (
                EconomicDecision.HUMAN_REVIEW,
                reasons,
            )

        if condition.critical_defect:
            reasons.append(
                "A critical defect materially reduces "
                "the asset's usable economic value."
            )

            if (
                replacement_option.lifecycle_cost
                <= repair_option.lifecycle_cost
            ):
                reasons.append(
                    "Replacement has lower lifecycle "
                    "cost than repair."
                )

                return (
                    EconomicDecision.REPLACE,
                    reasons,
                )

            reasons.append(
                "Repair remains economically preferable "
                "but requires engineering approval."
            )

            return (
                EconomicDecision.REPAIR,
                reasons,
            )

        if condition.obsolete:
            reasons.append(
                "Asset is marked obsolete."
            )

            if not condition.parts_available:
                reasons.append(
                    "Replacement parts are unavailable."
                )

                return (
                    EconomicDecision.REPLACE,
                    reasons,
                )

        if (
            failure_probability
            >= self.config
            .critical_failure_probability
        ):
            reasons.append(
                "Predicted failure probability is "
                "critical."
            )

            if (
                replacement_option.lifecycle_cost
                < repair_option.lifecycle_cost
            ):
                return (
                    EconomicDecision.REPLACE,
                    reasons,
                )

            return (
                EconomicDecision.REPAIR,
                reasons,
            )

        if (
            condition.predicted_rul_hours
            is not None
            and condition.predicted_rul_hours
            <= self.config.critical_rul_hours
        ):
            reasons.append(
                "Predicted remaining useful life is "
                "critically low."
            )

            if (
                replacement_option.lifecycle_cost
                <= repair_option.lifecycle_cost
            ):
                return (
                    EconomicDecision.REPLACE,
                    reasons,
                )

            return (
                EconomicDecision.REPAIR,
                reasons,
            )

        if (
            repair_ratio is not None
            and repair_ratio
            >= self.config.repair_ratio_replace
        ):
            reasons.append(
                "Repair cost is a large fraction of "
                "replacement cost."
            )

            if (
                replacement_option.lifecycle_cost
                <= repair_option.lifecycle_cost
            ):
                return (
                    EconomicDecision.REPLACE,
                    reasons,
                )

        if (
            replacement_option.lifecycle_cost
            < repair_option.lifecycle_cost
            * 0.90
        ):
            reasons.append(
                "Replacement has materially lower "
                "discounted lifecycle cost."
            )

            return (
                EconomicDecision.REPLACE,
                reasons,
            )

        if (
            condition_state
            in {
                AssetValueCondition.CRITICAL,
                AssetValueCondition.POOR,
            }
        ):
            reasons.append(
                "Asset condition is poor or critical."
            )

            return (
                EconomicDecision.REPAIR,
                reasons,
            )

        if (
            failure_probability
            >= self.config
            .high_failure_probability
        ):
            reasons.append(
                "Failure probability is high."
            )

            return (
                EconomicDecision.MAINTAIN,
                reasons,
            )

        if (
            repair_ratio is not None
            and repair_ratio
            >= self.config.repair_ratio_warning
        ):
            reasons.append(
                "Repair economics should be monitored "
                "against replacement economics."
            )

            return (
                EconomicDecision.MONITOR,
                reasons,
            )

        reasons.append(
            "Current asset economics support continued "
            "operation with normal condition monitoring."
        )

        return (
            EconomicDecision.RETAIN,
            reasons,
        )


# =========================================================
# Default instance
# =========================================================

asset_valuation = AssetValuation()


# =========================================================
# Agent-facing helpers
# =========================================================

def calculate_asset_value(
    *,
    profile: AssetFinancialProfile,
    condition: AssetConditionProfile,
    costs: AssetCostProfile | None = None,
    depreciation_method: DepreciationMethod = (
        DepreciationMethod.STRAIGHT_LINE
    ),
) -> AssetValuationResult:

    return asset_valuation.value_asset(
        profile=profile,
        condition=condition,
        costs=costs,
        depreciation_method=(
            depreciation_method
        ),
    )


def calculate_book_value(
    *,
    profile: AssetFinancialProfile,
    method: DepreciationMethod = (
        DepreciationMethod.STRAIGHT_LINE
    ),
) -> float:

    return (
        asset_valuation
        .depreciate(
            profile,
            method=method,
        )
        .book_value
    )


def calculate_life_consumed(
    profile: AssetFinancialProfile,
) -> float:

    return (
        asset_valuation
        .life_consumed_ratio(profile)
    )


def compare_repair_replacement(
    *,
    profile: AssetFinancialProfile,
    condition: AssetConditionProfile,
    costs: AssetCostProfile,
    horizon_years: float = 5.0,
    discount_rate: float = 0.08,
) -> dict[str, Any]:

    repair = asset_valuation.repair_option(
        profile=profile,
        condition=condition,
        costs=costs,
        horizon_years=horizon_years,
        discount_rate=discount_rate,
    )

    replacement = (
        asset_valuation.replacement_option(
            profile=profile,
            costs=costs,
            horizon_years=horizon_years,
            discount_rate=discount_rate,
        )
    )

    preferred = (
        "repair"
        if repair.lifecycle_cost
        <= replacement.lifecycle_cost
        else "replace"
    )

    return {
        "repair": repair.to_dict(),

        "replacement":
            replacement.to_dict(),

        "preferred_option":
            preferred,

        "lifecycle_cost_difference":
            round(
                abs(
                    repair.lifecycle_cost
                    - replacement.lifecycle_cost
                ),
                2,
            ),

        "currency": profile.currency,
    }


# =========================================================
# Exports
# =========================================================

__all__ = [
    "DepreciationMethod",
    "AssetValueCondition",
    "EconomicRiskLevel",
    "EconomicDecision",
    "ValuationConfidence",

    "AssetFinancialProfile",
    "AssetConditionProfile",
    "AssetCostProfile",

    "DepreciationResult",
    "EconomicExposure",
    "LifecycleOption",
    "AssetValuationResult",

    "AssetValuationConfig",
    "AssetValuation",

    "asset_valuation",

    "calculate_asset_value",
    "calculate_book_value",
    "calculate_life_consumed",
    "compare_repair_replacement",
]