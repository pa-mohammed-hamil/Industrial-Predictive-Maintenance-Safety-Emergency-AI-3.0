# app/agents/tools/labor_cost.py

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

class LaborType(str, Enum):
    INTERNAL = "internal"
    CONTRACTOR = "contractor"
    SPECIALIST = "specialist"
    OEM_TECHNICIAN = "oem_technician"
    ENGINEER = "engineer"
    ELECTRICIAN = "electrician"
    MECHANIC = "mechanic"
    INSPECTOR = "inspector"
    SAFETY_OFFICER = "safety_officer"
    OPERATOR = "operator"
    SUPERVISOR = "supervisor"
    OTHER = "other"


class LaborActivity(str, Enum):
    DIAGNOSIS = "diagnosis"
    INSPECTION = "inspection"

    PREVENTIVE_MAINTENANCE = (
        "preventive_maintenance"
    )

    PREDICTIVE_MAINTENANCE = (
        "predictive_maintenance"
    )

    CORRECTIVE_MAINTENANCE = (
        "corrective_maintenance"
    )

    REPAIR = "repair"
    OVERHAUL = "overhaul"
    REPLACEMENT = "replacement"

    INSTALLATION = "installation"
    REMOVAL = "removal"

    TESTING = "testing"
    COMMISSIONING = "commissioning"

    EMERGENCY_RESPONSE = "emergency_response"

    SAFETY_INSPECTION = "safety_inspection"

    FAILURE_INVESTIGATION = (
        "failure_investigation"
    )

    CLEANUP = "cleanup"
    RECOVERY = "recovery"

    TRAINING = "training"

    OTHER = "other"


class LaborCostCategory(str, Enum):
    REGULAR = "regular"
    OVERTIME = "overtime"
    SHIFT_PREMIUM = "shift_premium"
    EMERGENCY_PREMIUM = "emergency_premium"

    CONTRACTOR = "contractor"

    TRAVEL = "travel"
    ACCOMMODATION = "accommodation"

    TRAINING = "training"
    MOBILIZATION = "mobilization"

    STANDBY = "standby"

    ADMINISTRATION = "administration"

    OTHER = "other"


class ShiftType(str, Enum):
    DAY = "day"
    EVENING = "evening"
    NIGHT = "night"
    WEEKEND = "weekend"
    HOLIDAY = "holiday"
    EMERGENCY = "emergency"


class LaborCostBasis(str, Enum):
    HOURLY = "hourly"
    FIXED = "fixed"
    MIXED = "mixed"


class LaborCostRisk(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class LaborCostConfidence(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


# =========================================================
# Worker / crew
# =========================================================

@dataclass(slots=True)
class LaborResource:
    role: str

    labor_type: LaborType = LaborType.INTERNAL

    worker_count: int = 1

    regular_hours_per_worker: float = 0.0
    overtime_hours_per_worker: float = 0.0

    hourly_rate: float = 0.0

    overtime_multiplier: float = 1.5

    shift_type: ShiftType = ShiftType.DAY

    shift_premium_rate: float = 0.0

    emergency_premium_rate: float = 0.0

    fixed_cost: float = 0.0

    contractor_callout_cost: float = 0.0

    standby_hours_per_worker: float = 0.0
    standby_rate: float = 0.0

    productivity_factor: float = 1.0

    confidence: float = 1.0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.worker_count = max(
            0,
            int(self.worker_count),
        )

        for name in (
            "regular_hours_per_worker",
            "overtime_hours_per_worker",
            "hourly_rate",
            "shift_premium_rate",
            "emergency_premium_rate",
            "fixed_cost",
            "contractor_callout_cost",
            "standby_hours_per_worker",
            "standby_rate",
        ):
            setattr(
                self,
                name,
                non_negative(getattr(self, name)),
            )

        self.overtime_multiplier = non_negative(
            self.overtime_multiplier,
            1.5,
        )

        self.productivity_factor = clamp(
            self.productivity_factor,
            0.01,
            1.0,
        )

        self.confidence = clamp(
            self.confidence
        )

    @property
    def regular_worker_hours(self) -> float:
        return (
            self.worker_count
            * self.regular_hours_per_worker
        )

    @property
    def overtime_worker_hours(self) -> float:
        return (
            self.worker_count
            * self.overtime_hours_per_worker
        )

    @property
    def standby_worker_hours(self) -> float:
        return (
            self.worker_count
            * self.standby_hours_per_worker
        )

    @property
    def total_worker_hours(self) -> float:
        return (
            self.regular_worker_hours
            + self.overtime_worker_hours
            + self.standby_worker_hours
        )


# =========================================================
# Additional labor costs
# =========================================================

@dataclass(slots=True)
class LaborAdditionalCosts:
    travel_cost: float = 0.0
    accommodation_cost: float = 0.0

    mobilization_cost: float = 0.0
    demobilization_cost: float = 0.0

    training_cost: float = 0.0

    permit_cost: float = 0.0

    supervision_cost: float = 0.0
    administration_cost: float = 0.0

    PPE_cost: float = 0.0

    other_cost: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "travel_cost",
            "accommodation_cost",
            "mobilization_cost",
            "demobilization_cost",
            "training_cost",
            "permit_cost",
            "supervision_cost",
            "administration_cost",
            "PPE_cost",
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
                self.travel_cost,
                self.accommodation_cost,
                self.mobilization_cost,
                self.demobilization_cost,
                self.training_cost,
                self.permit_cost,
                self.supervision_cost,
                self.administration_cost,
                self.PPE_cost,
                self.other_cost,
            )
        )


# =========================================================
# Labor cost input
# =========================================================

@dataclass(slots=True)
class LaborCostInput:
    activity: LaborActivity

    resources: list[LaborResource] = field(
        default_factory=list
    )

    additional_costs: LaborAdditionalCosts = field(
        default_factory=LaborAdditionalCosts
    )

    cost_basis: LaborCostBasis = (
        LaborCostBasis.HOURLY
    )

    fixed_all_in_cost: float | None = None

    burden_rate: float = 0.0

    contingency_rate: float = 0.0

    probability: float = 1.0

    currency: str = "INR"

    asset_id: str | None = None
    component_id: str | None = None
    site_id: str | None = None

    work_order_id: str | None = None
    incident_id: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.fixed_all_in_cost is not None:
            self.fixed_all_in_cost = non_negative(
                self.fixed_all_in_cost
            )

        self.burden_rate = non_negative(
            self.burden_rate
        )

        self.contingency_rate = non_negative(
            self.contingency_rate
        )

        self.probability = clamp(
            self.probability
        )


# =========================================================
# Resource result
# =========================================================

@dataclass(slots=True)
class LaborResourceCost:
    role: str

    labor_type: LaborType

    worker_count: int

    regular_worker_hours: float
    overtime_worker_hours: float
    standby_worker_hours: float

    regular_cost: float
    overtime_cost: float
    shift_premium_cost: float
    emergency_premium_cost: float

    standby_cost: float

    fixed_cost: float
    contractor_callout_cost: float

    gross_resource_cost: float

    productivity_factor: float

    effective_resource_cost: float

    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# =========================================================
# Breakdown
# =========================================================

@dataclass(slots=True)
class LaborCostBreakdown:
    regular_labor: float = 0.0
    overtime_labor: float = 0.0

    shift_premium: float = 0.0
    emergency_premium: float = 0.0

    standby: float = 0.0

    fixed_labor: float = 0.0
    contractor_callout: float = 0.0

    productivity_adjustment: float = 0.0

    burden: float = 0.0

    travel: float = 0.0
    accommodation: float = 0.0

    mobilization: float = 0.0
    demobilization: float = 0.0

    training: float = 0.0
    permit: float = 0.0

    supervision: float = 0.0
    administration: float = 0.0

    ppe: float = 0.0

    other: float = 0.0

    contingency: float = 0.0

    @property
    def total(self) -> float:
        return sum(
            (
                self.regular_labor,
                self.overtime_labor,
                self.shift_premium,
                self.emergency_premium,
                self.standby,
                self.fixed_labor,
                self.contractor_callout,
                self.productivity_adjustment,
                self.burden,
                self.travel,
                self.accommodation,
                self.mobilization,
                self.demobilization,
                self.training,
                self.permit,
                self.supervision,
                self.administration,
                self.ppe,
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
class LaborCostResult:
    calculation_id: str

    activity: LaborActivity

    asset_id: str | None
    component_id: str | None
    site_id: str | None

    work_order_id: str | None
    incident_id: str | None

    resource_count: int
    worker_count: int

    regular_worker_hours: float
    overtime_worker_hours: float
    standby_worker_hours: float

    total_worker_hours: float

    direct_labor_cost: float

    burden_cost: float

    additional_cost: float

    contingency_cost: float

    gross_labor_cost: float

    expected_labor_cost: float

    average_cost_per_worker_hour: float | None

    breakdown: LaborCostBreakdown

    resource_costs: list[LaborResourceCost]

    confidence: float
    confidence_level: LaborCostConfidence

    cost_risk: LaborCostRisk

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
# Labor scenario
# =========================================================

@dataclass(slots=True)
class LaborCostScenario:
    name: str

    labor: LaborCostInput

    scenario_probability: float = 1.0

    scenario_weight: float = 1.0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.scenario_probability = clamp(
            self.scenario_probability
        )

        self.scenario_weight = non_negative(
            self.scenario_weight
        )


@dataclass(slots=True)
class LaborScenarioResult:
    name: str

    probability: float

    labor_cost: float

    expected_cost: float

    weight: float

    weighted_expected_cost: float

    currency: str

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# =========================================================
# Config
# =========================================================

@dataclass(slots=True)
class LaborCostConfig:
    default_overtime_multiplier: float = 1.5

    moderate_cost_threshold: float = 25_000.0
    high_cost_threshold: float = 100_000.0
    critical_cost_threshold: float = 500_000.0

    minimum_confidence: float = 0.50

    default_currency: str = "INR"


# =========================================================
# Calculator
# =========================================================

class LaborCostCalculator:

    def __init__(
        self,
        *,
        config: LaborCostConfig | None = None,
    ) -> None:
        self.config = (
            config
            or LaborCostConfig()
        )

    # =====================================================
    # Resource cost
    # =====================================================

    def calculate_resource(
        self,
        resource: LaborResource,
    ) -> LaborResourceCost:

        regular_hours = (
            resource.regular_worker_hours
        )

        overtime_hours = (
            resource.overtime_worker_hours
        )

        standby_hours = (
            resource.standby_worker_hours
        )

        regular_cost = (
            regular_hours
            * resource.hourly_rate
        )

        overtime_multiplier = (
            resource.overtime_multiplier
            if resource.overtime_multiplier > 0
            else self.config
            .default_overtime_multiplier
        )

        overtime_cost = (
            overtime_hours
            * resource.hourly_rate
            * overtime_multiplier
        )

        productive_hours = (
            regular_hours
            + overtime_hours
        )

        shift_premium_cost = (
            productive_hours
            * resource.shift_premium_rate
        )

        emergency_premium_cost = (
            productive_hours
            * resource.emergency_premium_rate
        )

        standby_cost = (
            standby_hours
            * resource.standby_rate
        )

        gross_before_productivity = sum(
            (
                regular_cost,
                overtime_cost,
                shift_premium_cost,
                emergency_premium_cost,
                standby_cost,
                resource.fixed_cost,
                resource.contractor_callout_cost,
            )
        )

        # If productivity is 80%, the effective
        # cost of accomplishing the same work is
        # gross / 0.80.
        effective_cost = (
            gross_before_productivity
            / resource.productivity_factor
        )

        return LaborResourceCost(
            role=resource.role,

            labor_type=resource.labor_type,

            worker_count=resource.worker_count,

            regular_worker_hours=round(
                regular_hours,
                4,
            ),

            overtime_worker_hours=round(
                overtime_hours,
                4,
            ),

            standby_worker_hours=round(
                standby_hours,
                4,
            ),

            regular_cost=round(
                regular_cost,
                2,
            ),

            overtime_cost=round(
                overtime_cost,
                2,
            ),

            shift_premium_cost=round(
                shift_premium_cost,
                2,
            ),

            emergency_premium_cost=round(
                emergency_premium_cost,
                2,
            ),

            standby_cost=round(
                standby_cost,
                2,
            ),

            fixed_cost=round(
                resource.fixed_cost,
                2,
            ),

            contractor_callout_cost=round(
                resource.contractor_callout_cost,
                2,
            ),

            gross_resource_cost=round(
                gross_before_productivity,
                2,
            ),

            productivity_factor=round(
                resource.productivity_factor,
                4,
            ),

            effective_resource_cost=round(
                effective_cost,
                2,
            ),

            confidence=round(
                resource.confidence,
                4,
            ),
        )

    # =====================================================
    # Confidence
    # =====================================================

    @staticmethod
    def calculate_confidence(
        resources: Sequence[
            LaborResource
        ],
    ) -> tuple[
        float,
        LaborCostConfidence,
    ]:

        if not resources:
            return (
                0.0,
                LaborCostConfidence.LOW,
            )

        weights: list[float] = []
        values: list[float] = []

        for resource in resources:
            hours = max(
                1.0,
                resource.total_worker_hours,
            )

            weights.append(hours)
            values.append(
                resource.confidence
            )

        total_weight = sum(weights)

        score = (
            sum(
                value * weight
                for value, weight
                in zip(values, weights)
            )
            / total_weight
        )

        score = clamp(score)

        if score >= 0.90:
            level = (
                LaborCostConfidence.VERY_HIGH
            )

        elif score >= 0.75:
            level = (
                LaborCostConfidence.HIGH
            )

        elif score >= 0.50:
            level = (
                LaborCostConfidence.MODERATE
            )

        else:
            level = (
                LaborCostConfidence.LOW
            )

        return round(score, 4), level

    # =====================================================
    # Cost risk
    # =====================================================

    def classify_cost_risk(
        self,
        total_cost: float,
    ) -> LaborCostRisk:

        cost = non_negative(
            total_cost
        )

        if (
            cost
            >= self.config
            .critical_cost_threshold
        ):
            return LaborCostRisk.CRITICAL

        if (
            cost
            >= self.config
            .high_cost_threshold
        ):
            return LaborCostRisk.HIGH

        if (
            cost
            >= self.config
            .moderate_cost_threshold
        ):
            return LaborCostRisk.MODERATE

        return LaborCostRisk.LOW

    # =====================================================
    # Main calculation
    # =====================================================

    def calculate(
        self,
        *,
        labor: LaborCostInput,
        metadata: Mapping[str, Any] | None = None,
    ) -> LaborCostResult:

        warnings: list[str] = []

        resource_results = [
            self.calculate_resource(resource)
            for resource in labor.resources
        ]

        # ---------------------------------------------
        # ALL-IN fixed cost
        # ---------------------------------------------

        if (
            labor.cost_basis
            == LaborCostBasis.FIXED
            and labor.fixed_all_in_cost
            is not None
        ):
            direct_labor_cost = (
                labor.fixed_all_in_cost
            )

            regular_cost = 0.0
            overtime_cost = 0.0
            shift_premium = 0.0
            emergency_premium = 0.0
            standby_cost = 0.0
            fixed_resource_cost = 0.0
            contractor_callout = 0.0
            productivity_adjustment = 0.0

        else:
            regular_cost = sum(
                item.regular_cost
                for item in resource_results
            )

            overtime_cost = sum(
                item.overtime_cost
                for item in resource_results
            )

            shift_premium = sum(
                item.shift_premium_cost
                for item in resource_results
            )

            emergency_premium = sum(
                item.emergency_premium_cost
                for item in resource_results
            )

            standby_cost = sum(
                item.standby_cost
                for item in resource_results
            )

            fixed_resource_cost = sum(
                item.fixed_cost
                for item in resource_results
            )

            contractor_callout = sum(
                item.contractor_callout_cost
                for item in resource_results
            )

            raw_resource_cost = sum(
                item.gross_resource_cost
                for item in resource_results
            )

            effective_resource_cost = sum(
                item.effective_resource_cost
                for item in resource_results
            )

            productivity_adjustment = max(
                0.0,
                effective_resource_cost
                - raw_resource_cost,
            )

            direct_labor_cost = (
                effective_resource_cost
            )

            if (
                labor.cost_basis
                == LaborCostBasis.MIXED
                and labor.fixed_all_in_cost
                is not None
            ):
                direct_labor_cost += (
                    labor.fixed_all_in_cost
                )

                fixed_resource_cost += (
                    labor.fixed_all_in_cost
                )

        # ---------------------------------------------
        # Burden
        # ---------------------------------------------

        burden_cost = (
            direct_labor_cost
            * labor.burden_rate
        )

        additional = (
            labor.additional_costs
        )

        additional_total = (
            additional.total
        )

        subtotal = (
            direct_labor_cost
            + burden_cost
            + additional_total
        )

        contingency = (
            subtotal
            * labor.contingency_rate
        )

        gross_cost = (
            subtotal
            + contingency
        )

        expected_cost = (
            gross_cost
            * labor.probability
        )

        # ---------------------------------------------
        # Hours
        # ---------------------------------------------

        regular_hours = sum(
            resource.regular_worker_hours
            for resource in labor.resources
        )

        overtime_hours = sum(
            resource.overtime_worker_hours
            for resource in labor.resources
        )

        standby_hours = sum(
            resource.standby_worker_hours
            for resource in labor.resources
        )

        total_hours = (
            regular_hours
            + overtime_hours
            + standby_hours
        )

        worker_count = sum(
            resource.worker_count
            for resource in labor.resources
        )

        average_hourly_cost = None

        if total_hours > 0:
            average_hourly_cost = (
                gross_cost
                / total_hours
            )

        confidence, confidence_level = (
            self.calculate_confidence(
                labor.resources
            )
        )

        risk = self.classify_cost_risk(
            gross_cost
        )

        breakdown = LaborCostBreakdown(
            regular_labor=regular_cost,

            overtime_labor=overtime_cost,

            shift_premium=shift_premium,

            emergency_premium=(
                emergency_premium
            ),

            standby=standby_cost,

            fixed_labor=fixed_resource_cost,

            contractor_callout=(
                contractor_callout
            ),

            productivity_adjustment=(
                productivity_adjustment
            ),

            burden=burden_cost,

            travel=additional.travel_cost,

            accommodation=(
                additional.accommodation_cost
            ),

            mobilization=(
                additional.mobilization_cost
            ),

            demobilization=(
                additional.demobilization_cost
            ),

            training=additional.training_cost,

            permit=additional.permit_cost,

            supervision=(
                additional.supervision_cost
            ),

            administration=(
                additional.administration_cost
            ),

            ppe=additional.PPE_cost,

            other=additional.other_cost,

            contingency=contingency,
        )

        # ---------------------------------------------
        # Warnings
        # ---------------------------------------------

        if not labor.resources and (
            labor.fixed_all_in_cost is None
        ):
            warnings.append(
                "No labor resources or fixed labor "
                "cost were supplied."
            )

        if (
            confidence
            < self.config.minimum_confidence
        ):
            warnings.append(
                "Labor cost estimate has low "
                "confidence."
            )

        if (
            labor.cost_basis
            == LaborCostBasis.FIXED
            and labor.fixed_all_in_cost is None
        ):
            warnings.append(
                "FIXED cost basis selected without "
                "fixed_all_in_cost."
            )

        if (
            labor.cost_basis
            == LaborCostBasis.FIXED
            and labor.resources
        ):
            warnings.append(
                "Labor resources were supplied with "
                "FIXED cost basis. Resource costs "
                "are not added to the fixed all-in "
                "labor amount."
            )

        if (
            labor.cost_basis
            == LaborCostBasis.MIXED
            and labor.fixed_all_in_cost is not None
        ):
            warnings.append(
                "MIXED basis adds fixed_all_in_cost "
                "to calculated resource labor. "
                "Confirm the fixed amount does not "
                "already contain hourly labor."
            )

        if (
            labor.burden_rate > 0.75
        ):
            warnings.append(
                "Labor burden rate is unusually high; "
                "verify the input uses a decimal "
                "fraction rather than a percentage."
            )

        if (
            labor.contingency_rate > 0.50
        ):
            warnings.append(
                "Labor contingency rate is unusually "
                "high."
            )

        for resource in labor.resources:
            if (
                resource.overtime_hours_per_worker > 0
                and resource.overtime_multiplier < 1.0
            ):
                warnings.append(
                    f"Overtime multiplier for "
                    f"{resource.role!r} is below 1.0."
                )

            if (
                resource.productivity_factor < 0.50
            ):
                warnings.append(
                    f"Low productivity factor for "
                    f"{resource.role!r} materially "
                    f"increases effective labor cost."
                )

        return LaborCostResult(
            calculation_id=new_id(),

            activity=labor.activity,

            asset_id=labor.asset_id,
            component_id=labor.component_id,
            site_id=labor.site_id,

            work_order_id=labor.work_order_id,
            incident_id=labor.incident_id,

            resource_count=len(
                labor.resources
            ),

            worker_count=worker_count,

            regular_worker_hours=round(
                regular_hours,
                4,
            ),

            overtime_worker_hours=round(
                overtime_hours,
                4,
            ),

            standby_worker_hours=round(
                standby_hours,
                4,
            ),

            total_worker_hours=round(
                total_hours,
                4,
            ),

            direct_labor_cost=round(
                direct_labor_cost,
                2,
            ),

            burden_cost=round(
                burden_cost,
                2,
            ),

            additional_cost=round(
                additional_total,
                2,
            ),

            contingency_cost=round(
                contingency,
                2,
            ),

            gross_labor_cost=round(
                gross_cost,
                2,
            ),

            expected_labor_cost=round(
                expected_cost,
                2,
            ),

            average_cost_per_worker_hour=(
                round(
                    average_hourly_cost,
                    2,
                )
                if average_hourly_cost
                is not None
                else None
            ),

            breakdown=breakdown,

            resource_costs=resource_results,

            confidence=confidence,

            confidence_level=(
                confidence_level
            ),

            cost_risk=risk,

            currency=(
                labor.currency
                or self.config.default_currency
            ),

            warnings=warnings,

            metadata={
                **labor.metadata,
                **dict(metadata or {}),
            },
        )

    # =====================================================
    # Simple hourly labor calculation
    # =====================================================

    @staticmethod
    def hourly_labor_cost(
        *,
        workers: int,
        hours: float,
        hourly_rate: float,
    ) -> float:

        return round(
            max(0, int(workers))
            * non_negative(hours)
            * non_negative(hourly_rate),
            2,
        )

    # =====================================================
    # Overtime calculation
    # =====================================================

    @staticmethod
    def overtime_labor_cost(
        *,
        workers: int,
        overtime_hours: float,
        hourly_rate: float,
        overtime_multiplier: float = 1.5,
    ) -> float:

        return round(
            max(0, int(workers))
            * non_negative(overtime_hours)
            * non_negative(hourly_rate)
            * non_negative(
                overtime_multiplier,
                1.5,
            ),
            2,
        )

    # =====================================================
    # Expected labor cost
    # =====================================================

    @staticmethod
    def expected_cost(
        *,
        labor_cost: float,
        probability: float,
    ) -> float:

        return round(
            non_negative(labor_cost)
            * clamp(probability),
            2,
        )

    # =====================================================
    # Escalate future labor cost
    # =====================================================

    @staticmethod
    def future_labor_cost(
        *,
        current_cost: float,
        annual_escalation_rate: float,
        years: float,
    ) -> float:

        current = non_negative(
            current_cost
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
            current * (base ** years),
            2,
        )

    # =====================================================
    # Labor savings
    # =====================================================

    @staticmethod
    def labor_savings(
        *,
        baseline_cost: float,
        improved_cost: float,
    ) -> dict[str, float]:

        baseline = non_negative(
            baseline_cost
        )

        improved = non_negative(
            improved_cost
        )

        savings = (
            baseline
            - improved
        )

        savings_percent = (
            safe_divide(
                savings,
                baseline,
            )
            * 100.0
        )

        return {
            "baseline_cost": round(
                baseline,
                2,
            ),

            "improved_cost": round(
                improved,
                2,
            ),

            "savings": round(
                savings,
                2,
            ),

            "savings_percent": round(
                savings_percent,
                2,
            ),
        }

    # =====================================================
    # Scenario analysis
    # =====================================================

    def scenario_analysis(
        self,
        scenarios: Sequence[
            LaborCostScenario
        ],
    ) -> list[LaborScenarioResult]:

        results: list[
            LaborScenarioResult
        ] = []

        for scenario in scenarios:
            calculation = self.calculate(
                labor=scenario.labor
            )

            expected = (
                calculation.gross_labor_cost
                * scenario.scenario_probability
            )

            weighted = (
                expected
                * scenario.scenario_weight
            )

            results.append(
                LaborScenarioResult(
                    name=scenario.name,

                    probability=round(
                        scenario
                        .scenario_probability,
                        6,
                    ),

                    labor_cost=round(
                        calculation
                        .gross_labor_cost,
                        2,
                    ),

                    expected_cost=round(
                        expected,
                        2,
                    ),

                    weight=round(
                        scenario.scenario_weight,
                        4,
                    ),

                    weighted_expected_cost=round(
                        weighted,
                        2,
                    ),

                    currency=(
                        calculation.currency
                    ),
                )
            )

        return results

    # =====================================================
    # Batch
    # =====================================================

    def calculate_many(
        self,
        inputs: Sequence[
            LaborCostInput
        ],
    ) -> list[LaborCostResult]:

        return [
            self.calculate(labor=item)
            for item in inputs
        ]

    # =====================================================
    # Aggregate
    # =====================================================

    @staticmethod
    def aggregate(
        results: Sequence[
            LaborCostResult
        ],
    ) -> dict[str, Any]:

        if not results:
            return {
                "job_count": 0,
                "worker_hours": 0.0,
                "gross_labor_cost": 0.0,
                "expected_labor_cost": 0.0,
                "currency": None,
            }

        currencies = {
            result.currency
            for result in results
        }

        gross = sum(
            result.gross_labor_cost
            for result in results
        )

        expected = sum(
            result.expected_labor_cost
            for result in results
        )

        hours = sum(
            result.total_worker_hours
            for result in results
        )

        overtime_hours = sum(
            result.overtime_worker_hours
            for result in results
        )

        return {
            "job_count": len(results),

            "worker_hours": round(
                hours,
                2,
            ),

            "overtime_worker_hours": round(
                overtime_hours,
                2,
            ),

            "gross_labor_cost": round(
                gross,
                2,
            ),

            "expected_labor_cost": round(
                expected,
                2,
            ),

            "average_cost_per_job": round(
                gross / len(results),
                2,
            ),

            "average_cost_per_worker_hour": (
                round(
                    gross / hours,
                    2,
                )
                if hours > 0
                else None
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

labor_cost_calculator = (
    LaborCostCalculator()
)


# =========================================================
# Agent-facing helpers
# =========================================================

def calculate_labor_cost(
    *,
    labor: LaborCostInput,
) -> LaborCostResult:

    return labor_cost_calculator.calculate(
        labor=labor
    )


def calculate_hourly_labor_cost(
    *,
    workers: int,
    hours: float,
    hourly_rate: float,
) -> float:

    return (
        labor_cost_calculator
        .hourly_labor_cost(
            workers=workers,
            hours=hours,
            hourly_rate=hourly_rate,
        )
    )


def calculate_overtime_labor_cost(
    *,
    workers: int,
    overtime_hours: float,
    hourly_rate: float,
    overtime_multiplier: float = 1.5,
) -> float:

    return (
        labor_cost_calculator
        .overtime_labor_cost(
            workers=workers,
            overtime_hours=overtime_hours,
            hourly_rate=hourly_rate,
            overtime_multiplier=(
                overtime_multiplier
            ),
        )
    )


def calculate_future_labor_cost(
    *,
    current_cost: float,
    annual_escalation_rate: float,
    years: float,
) -> float:

    return (
        labor_cost_calculator
        .future_labor_cost(
            current_cost=current_cost,
            annual_escalation_rate=(
                annual_escalation_rate
            ),
            years=years,
        )
    )


# =========================================================
# Exports
# =========================================================

__all__ = [
    "LaborType",
    "LaborActivity",
    "LaborCostCategory",
    "ShiftType",
    "LaborCostBasis",
    "LaborCostRisk",
    "LaborCostConfidence",

    "LaborResource",
    "LaborAdditionalCosts",
    "LaborCostInput",

    "LaborResourceCost",
    "LaborCostBreakdown",
    "LaborCostResult",

    "LaborCostScenario",
    "LaborScenarioResult",

    "LaborCostConfig",
    "LaborCostCalculator",

    "labor_cost_calculator",

    "calculate_labor_cost",
    "calculate_hourly_labor_cost",
    "calculate_overtime_labor_cost",
    "calculate_future_labor_cost",
]