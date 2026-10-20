# app/agents/tools/energy_cost.py

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

class EnergySource(str, Enum):
    GRID = "grid"
    SOLAR = "solar"
    WIND = "wind"
    BATTERY = "battery"
    DIESEL_GENERATOR = "diesel_generator"
    GAS_GENERATOR = "gas_generator"
    STEAM = "steam"
    COMPRESSED_AIR = "compressed_air"
    OTHER = "other"


class EnergyCostCategory(str, Enum):
    ACTIVE_ENERGY = "active_energy"
    DEMAND = "demand"
    REACTIVE_ENERGY = "reactive_energy"
    POWER_FACTOR = "power_factor"
    PEAK = "peak"
    IDLE = "idle"
    STANDBY = "standby"
    STARTUP = "startup"
    SHUTDOWN = "shutdown"
    ENERGY_WASTE = "energy_waste"
    CARBON = "carbon"
    TAX = "tax"
    FIXED = "fixed"
    OTHER = "other"


class TariffType(str, Enum):
    FLAT = "flat"
    TIME_OF_USE = "time_of_use"
    TIERED = "tiered"
    DEMAND_BASED = "demand_based"
    HYBRID = "hybrid"


class TimeBand(str, Enum):
    OFF_PEAK = "off_peak"
    NORMAL = "normal"
    PEAK = "peak"
    SUPER_PEAK = "super_peak"


class EnergyEfficiencyStatus(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    MODERATE = "moderate"
    POOR = "poor"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class EnergyCostRisk(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


# =========================================================
# Tariff models
# =========================================================

@dataclass(slots=True)
class EnergyTariff:
    tariff_type: TariffType = TariffType.FLAT

    currency: str = "INR"

    energy_rate_per_kwh: float = 0.0

    off_peak_rate_per_kwh: float | None = None
    normal_rate_per_kwh: float | None = None
    peak_rate_per_kwh: float | None = None
    super_peak_rate_per_kwh: float | None = None

    demand_rate_per_kw: float = 0.0

    reactive_energy_rate_per_kvarh: float = 0.0

    fixed_charge: float = 0.0

    tax_rate: float = 0.0

    power_factor_penalty_rate: float = 0.0

    target_power_factor: float = 0.95

    carbon_price_per_kg: float = 0.0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        for name in (
            "energy_rate_per_kwh",
            "demand_rate_per_kw",
            "reactive_energy_rate_per_kvarh",
            "fixed_charge",
            "power_factor_penalty_rate",
            "carbon_price_per_kg",
        ):
            setattr(
                self,
                name,
                non_negative(getattr(self, name)),
            )

        for name in (
            "off_peak_rate_per_kwh",
            "normal_rate_per_kwh",
            "peak_rate_per_kwh",
            "super_peak_rate_per_kwh",
        ):
            value = getattr(self, name)

            if value is not None:
                setattr(
                    self,
                    name,
                    non_negative(value),
                )

        self.tax_rate = clamp(self.tax_rate)
        self.target_power_factor = clamp(
            self.target_power_factor,
            0.01,
            1.0,
        )

    def rate_for_band(
        self,
        band: TimeBand,
    ) -> float:
        mapping = {
            TimeBand.OFF_PEAK:
                self.off_peak_rate_per_kwh,

            TimeBand.NORMAL:
                self.normal_rate_per_kwh,

            TimeBand.PEAK:
                self.peak_rate_per_kwh,

            TimeBand.SUPER_PEAK:
                self.super_peak_rate_per_kwh,
        }

        value = mapping.get(band)

        if value is None:
            return self.energy_rate_per_kwh

        return value


# =========================================================
# Energy consumption
# =========================================================

@dataclass(slots=True)
class EnergyConsumption:
    total_kwh: float

    off_peak_kwh: float = 0.0
    normal_kwh: float = 0.0
    peak_kwh: float = 0.0
    super_peak_kwh: float = 0.0

    reactive_energy_kvarh: float = 0.0

    peak_demand_kw: float = 0.0

    average_power_kw: float = 0.0

    operating_hours: float = 0.0

    idle_kwh: float = 0.0
    standby_kwh: float = 0.0
    startup_kwh: float = 0.0

    power_factor: float = 1.0

    source: EnergySource = EnergySource.GRID

    carbon_intensity_kg_per_kwh: float = 0.0

    confidence: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "total_kwh",
            "off_peak_kwh",
            "normal_kwh",
            "peak_kwh",
            "super_peak_kwh",
            "reactive_energy_kvarh",
            "peak_demand_kw",
            "average_power_kw",
            "operating_hours",
            "idle_kwh",
            "standby_kwh",
            "startup_kwh",
            "carbon_intensity_kg_per_kwh",
        ):
            setattr(
                self,
                name,
                non_negative(getattr(self, name)),
            )

        self.power_factor = clamp(
            self.power_factor,
            0.0,
            1.0,
        )

        self.confidence = clamp(
            self.confidence
        )


# =========================================================
# Production context
# =========================================================

@dataclass(slots=True)
class EnergyProductionContext:
    production_units: float = 0.0

    good_units: float = 0.0

    rated_power_kw: float | None = None

    baseline_specific_energy_kwh_per_unit: float | None = None

    target_specific_energy_kwh_per_unit: float | None = None

    def __post_init__(self) -> None:
        self.production_units = non_negative(
            self.production_units
        )

        self.good_units = non_negative(
            self.good_units
        )

        if self.rated_power_kw is not None:
            self.rated_power_kw = non_negative(
                self.rated_power_kw
            )

        if (
            self.baseline_specific_energy_kwh_per_unit
            is not None
        ):
            self.baseline_specific_energy_kwh_per_unit = (
                non_negative(
                    self.baseline_specific_energy_kwh_per_unit
                )
            )

        if (
            self.target_specific_energy_kwh_per_unit
            is not None
        ):
            self.target_specific_energy_kwh_per_unit = (
                non_negative(
                    self.target_specific_energy_kwh_per_unit
                )
            )


# =========================================================
# Breakdown
# =========================================================

@dataclass(slots=True)
class EnergyCostBreakdown:
    active_energy: float = 0.0

    demand: float = 0.0

    reactive_energy: float = 0.0

    power_factor_penalty: float = 0.0

    carbon: float = 0.0

    fixed_charge: float = 0.0

    tax: float = 0.0

    @property
    def subtotal(self) -> float:
        return sum(
            (
                self.active_energy,
                self.demand,
                self.reactive_energy,
                self.power_factor_penalty,
                self.carbon,
                self.fixed_charge,
            )
        )

    @property
    def total(self) -> float:
        return self.subtotal + self.tax

    def to_dict(self) -> dict[str, Any]:
        result = json_safe(self)
        result["subtotal"] = round(
            self.subtotal,
            2,
        )
        result["total"] = round(
            self.total,
            2,
        )
        return result


# =========================================================
# Waste
# =========================================================

@dataclass(slots=True)
class EnergyWaste:
    idle_energy_kwh: float = 0.0
    standby_energy_kwh: float = 0.0

    excess_process_energy_kwh: float = 0.0

    power_factor_extra_cost: float = 0.0

    avoidable_peak_cost: float = 0.0

    @property
    def wasted_energy_kwh(self) -> float:
        return sum(
            (
                self.idle_energy_kwh,
                self.standby_energy_kwh,
                self.excess_process_energy_kwh,
            )
        )


# =========================================================
# Result
# =========================================================

@dataclass(slots=True)
class EnergyCostResult:
    calculation_id: str

    asset_id: str | None
    site_id: str | None

    source: EnergySource

    total_kwh: float

    active_energy_cost: float

    total_energy_cost: float

    cost_per_kwh: float

    cost_per_operating_hour: float | None

    cost_per_unit: float | None

    specific_energy_kwh_per_unit: float | None

    peak_demand_kw: float

    power_factor: float

    carbon_emissions_kg: float

    carbon_cost: float

    breakdown: EnergyCostBreakdown

    waste: EnergyWaste

    estimated_waste_cost: float

    efficiency_score: float

    efficiency_status: EnergyEfficiencyStatus

    economic_risk: EnergyCostRisk

    confidence: float

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
# Energy savings
# =========================================================

@dataclass(slots=True)
class EnergySavingsInput:
    baseline_kwh: float

    improved_kwh: float

    energy_rate_per_kwh: float

    intervention_cost: float = 0.0

    annual_operating_periods: float = 1.0

    baseline_peak_kw: float = 0.0
    improved_peak_kw: float = 0.0

    demand_rate_per_kw: float = 0.0

    carbon_intensity_kg_per_kwh: float = 0.0

    carbon_price_per_kg: float = 0.0

    currency: str = "INR"

    def __post_init__(self) -> None:
        for name in (
            "baseline_kwh",
            "improved_kwh",
            "energy_rate_per_kwh",
            "intervention_cost",
            "annual_operating_periods",
            "baseline_peak_kw",
            "improved_peak_kw",
            "demand_rate_per_kw",
            "carbon_intensity_kg_per_kwh",
            "carbon_price_per_kg",
        ):
            setattr(
                self,
                name,
                non_negative(getattr(self, name)),
            )


@dataclass(slots=True)
class EnergySavingsResult:
    baseline_kwh: float

    improved_kwh: float

    energy_saved_kwh: float

    energy_reduction_percent: float

    direct_energy_savings: float

    demand_savings: float

    carbon_reduction_kg: float

    carbon_cost_savings: float

    gross_savings_per_period: float

    annual_savings: float

    intervention_cost: float

    first_year_net_savings: float

    simple_payback_years: float | None

    roi_percent: float | None

    currency: str

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# =========================================================
# Config
# =========================================================

@dataclass(slots=True)
class EnergyCostConfig:
    good_efficiency_score: float = 0.85
    moderate_efficiency_score: float = 0.70
    poor_efficiency_score: float = 0.50

    moderate_waste_ratio: float = 0.10
    high_waste_ratio: float = 0.20
    critical_waste_ratio: float = 0.35

    low_power_factor_threshold: float = 0.90

    minimum_confidence: float = 0.50

    default_currency: str = "INR"


# =========================================================
# Calculator
# =========================================================

class EnergyCostCalculator:

    def __init__(
        self,
        *,
        config: EnergyCostConfig | None = None,
    ) -> None:
        self.config = (
            config or EnergyCostConfig()
        )

    # =====================================================
    # Active energy cost
    # =====================================================

    @staticmethod
    def active_energy_cost(
        *,
        consumption: EnergyConsumption,
        tariff: EnergyTariff,
    ) -> float:

        tou_total = sum(
            (
                consumption.off_peak_kwh,
                consumption.normal_kwh,
                consumption.peak_kwh,
                consumption.super_peak_kwh,
            )
        )

        if (
            tariff.tariff_type
            in {
                TariffType.TIME_OF_USE,
                TariffType.HYBRID,
            }
            and tou_total > 0
        ):
            band_cost = sum(
                (
                    consumption.off_peak_kwh
                    * tariff.rate_for_band(
                        TimeBand.OFF_PEAK
                    ),

                    consumption.normal_kwh
                    * tariff.rate_for_band(
                        TimeBand.NORMAL
                    ),

                    consumption.peak_kwh
                    * tariff.rate_for_band(
                        TimeBand.PEAK
                    ),

                    consumption.super_peak_kwh
                    * tariff.rate_for_band(
                        TimeBand.SUPER_PEAK
                    ),
                )
            )

            # Any unclassified kWh are charged at
            # the base rate instead of silently lost.
            unclassified_kwh = max(
                0.0,
                consumption.total_kwh
                - tou_total,
            )

            return (
                band_cost
                + unclassified_kwh
                * tariff.energy_rate_per_kwh
            )

        return (
            consumption.total_kwh
            * tariff.energy_rate_per_kwh
        )

    # =====================================================
    # Demand cost
    # =====================================================

    @staticmethod
    def demand_cost(
        *,
        consumption: EnergyConsumption,
        tariff: EnergyTariff,
    ) -> float:
        return (
            consumption.peak_demand_kw
            * tariff.demand_rate_per_kw
        )

    # =====================================================
    # Reactive energy cost
    # =====================================================

    @staticmethod
    def reactive_energy_cost(
        *,
        consumption: EnergyConsumption,
        tariff: EnergyTariff,
    ) -> float:
        return (
            consumption.reactive_energy_kvarh
            * tariff.reactive_energy_rate_per_kvarh
        )

    # =====================================================
    # Power factor penalty
    # =====================================================

    @staticmethod
    def power_factor_penalty(
        *,
        consumption: EnergyConsumption,
        tariff: EnergyTariff,
    ) -> float:

        if (
            consumption.power_factor
            >= tariff.target_power_factor
        ):
            return 0.0

        deficit = (
            tariff.target_power_factor
            - consumption.power_factor
        )

        return (
            deficit
            * consumption.total_kwh
            * tariff.power_factor_penalty_rate
        )

    # =====================================================
    # Carbon
    # =====================================================

    @staticmethod
    def carbon_emissions(
        consumption: EnergyConsumption,
    ) -> float:
        return (
            consumption.total_kwh
            * consumption
            .carbon_intensity_kg_per_kwh
        )

    def carbon_cost(
        self,
        *,
        consumption: EnergyConsumption,
        tariff: EnergyTariff,
    ) -> tuple[float, float]:

        emissions = self.carbon_emissions(
            consumption
        )

        cost = (
            emissions
            * tariff.carbon_price_per_kg
        )

        return emissions, cost

    # =====================================================
    # Production metrics
    # =====================================================

    @staticmethod
    def specific_energy(
        *,
        total_kwh: float,
        production_units: float,
    ) -> float | None:

        if production_units <= 0:
            return None

        return total_kwh / production_units

    # =====================================================
    # Excess process energy
    # =====================================================

    def excess_process_energy(
        self,
        *,
        consumption: EnergyConsumption,
        production: EnergyProductionContext | None,
    ) -> float:

        if (
            production is None
            or production.production_units <= 0
        ):
            return 0.0

        target = (
            production
            .target_specific_energy_kwh_per_unit
        )

        if target is None:
            target = (
                production
                .baseline_specific_energy_kwh_per_unit
            )

        if target is None:
            return 0.0

        expected = (
            target
            * production.production_units
        )

        return max(
            0.0,
            consumption.total_kwh - expected,
        )

    # =====================================================
    # Waste
    # =====================================================

    def calculate_waste(
        self,
        *,
        consumption: EnergyConsumption,
        production: EnergyProductionContext | None,
        tariff: EnergyTariff,
        power_factor_penalty: float,
    ) -> EnergyWaste:

        excess = self.excess_process_energy(
            consumption=consumption,
            production=production,
        )

        avoidable_peak_cost = 0.0

        if (
            consumption.peak_kwh > 0
            and tariff.peak_rate_per_kwh
            is not None
        ):
            base_rate = (
                tariff.normal_rate_per_kwh
                if tariff.normal_rate_per_kwh
                is not None
                else tariff.energy_rate_per_kwh
            )

            premium = max(
                0.0,
                tariff.peak_rate_per_kwh
                - base_rate,
            )

            avoidable_peak_cost = (
                consumption.peak_kwh
                * premium
            )

        return EnergyWaste(
            idle_energy_kwh=(
                consumption.idle_kwh
            ),

            standby_energy_kwh=(
                consumption.standby_kwh
            ),

            excess_process_energy_kwh=(
                excess
            ),

            power_factor_extra_cost=(
                power_factor_penalty
            ),

            avoidable_peak_cost=(
                avoidable_peak_cost
            ),
        )

    # =====================================================
    # Efficiency
    # =====================================================

    def efficiency_score(
        self,
        *,
        consumption: EnergyConsumption,
        waste: EnergyWaste,
    ) -> tuple[
        float,
        EnergyEfficiencyStatus,
    ]:

        if consumption.total_kwh <= 0:
            return (
                0.0,
                EnergyEfficiencyStatus.UNKNOWN,
            )

        waste_ratio = clamp(
            waste.wasted_energy_kwh
            / consumption.total_kwh
        )

        score = clamp(
            1.0 - waste_ratio
        )

        if (
            score
            >= self.config.good_efficiency_score
        ):
            status = (
                EnergyEfficiencyStatus.EXCELLENT
                if score >= 0.95
                else EnergyEfficiencyStatus.GOOD
            )

        elif (
            score
            >= self.config
            .moderate_efficiency_score
        ):
            status = (
                EnergyEfficiencyStatus.MODERATE
            )

        elif (
            score
            >= self.config
            .poor_efficiency_score
        ):
            status = (
                EnergyEfficiencyStatus.POOR
            )

        else:
            status = (
                EnergyEfficiencyStatus.CRITICAL
            )

        return round(score, 4), status

    # =====================================================
    # Economic risk
    # =====================================================

    def economic_risk(
        self,
        *,
        consumption: EnergyConsumption,
        waste: EnergyWaste,
    ) -> EnergyCostRisk:

        if consumption.total_kwh <= 0:
            return EnergyCostRisk.LOW

        waste_ratio = (
            waste.wasted_energy_kwh
            / consumption.total_kwh
        )

        if (
            waste_ratio
            >= self.config
            .critical_waste_ratio
        ):
            return EnergyCostRisk.CRITICAL

        if (
            waste_ratio
            >= self.config.high_waste_ratio
        ):
            return EnergyCostRisk.HIGH

        if (
            waste_ratio
            >= self.config
            .moderate_waste_ratio
        ):
            return EnergyCostRisk.MODERATE

        return EnergyCostRisk.LOW

    # =====================================================
    # Main calculation
    # =====================================================

    def calculate(
        self,
        *,
        consumption: EnergyConsumption,
        tariff: EnergyTariff,
        production: EnergyProductionContext | None = None,
        asset_id: str | None = None,
        site_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> EnergyCostResult:

        warnings: list[str] = []

        active_cost = (
            self.active_energy_cost(
                consumption=consumption,
                tariff=tariff,
            )
        )

        demand_cost = self.demand_cost(
            consumption=consumption,
            tariff=tariff,
        )

        reactive_cost = (
            self.reactive_energy_cost(
                consumption=consumption,
                tariff=tariff,
            )
        )

        pf_penalty = (
            self.power_factor_penalty(
                consumption=consumption,
                tariff=tariff,
            )
        )

        emissions, carbon_cost = (
            self.carbon_cost(
                consumption=consumption,
                tariff=tariff,
            )
        )

        subtotal = sum(
            (
                active_cost,
                demand_cost,
                reactive_cost,
                pf_penalty,
                carbon_cost,
                tariff.fixed_charge,
            )
        )

        tax = (
            subtotal * tariff.tax_rate
        )

        breakdown = EnergyCostBreakdown(
            active_energy=active_cost,
            demand=demand_cost,
            reactive_energy=reactive_cost,
            power_factor_penalty=pf_penalty,
            carbon=carbon_cost,
            fixed_charge=tariff.fixed_charge,
            tax=tax,
        )

        total_cost = breakdown.total

        waste = self.calculate_waste(
            consumption=consumption,
            production=production,
            tariff=tariff,
            power_factor_penalty=pf_penalty,
        )

        average_energy_rate = safe_divide(
            active_cost,
            consumption.total_kwh,
            default=tariff.energy_rate_per_kwh,
        )

        waste_energy_cost = (
            waste.wasted_energy_kwh
            * average_energy_rate
        )

        estimated_waste_cost = sum(
            (
                waste_energy_cost,
                waste.power_factor_extra_cost,
                waste.avoidable_peak_cost,
            )
        )

        efficiency_score, efficiency_status = (
            self.efficiency_score(
                consumption=consumption,
                waste=waste,
            )
        )

        risk = self.economic_risk(
            consumption=consumption,
            waste=waste,
        )

        cost_per_kwh = safe_divide(
            total_cost,
            consumption.total_kwh,
        )

        cost_per_hour = None

        if consumption.operating_hours > 0:
            cost_per_hour = (
                total_cost
                / consumption.operating_hours
            )

        specific_energy = None
        cost_per_unit = None

        if production is not None:
            specific_energy = (
                self.specific_energy(
                    total_kwh=(
                        consumption.total_kwh
                    ),
                    production_units=(
                        production.production_units
                    ),
                )
            )

            if production.production_units > 0:
                cost_per_unit = (
                    total_cost
                    / production.production_units
                )

        if consumption.total_kwh <= 0:
            warnings.append(
                "Total energy consumption is zero; "
                "energy efficiency cannot be "
                "meaningfully evaluated."
            )

        if (
            consumption.power_factor
            < self.config
            .low_power_factor_threshold
        ):
            warnings.append(
                "Low power factor detected. Review "
                "motor loading, reactive power and "
                "power-factor correction."
            )

        if (
            consumption.idle_kwh
            + consumption.standby_kwh
            > consumption.total_kwh
        ):
            warnings.append(
                "Idle plus standby energy exceeds "
                "total measured energy. Verify meter "
                "boundaries and telemetry."
            )

        if (
            sum(
                (
                    consumption.off_peak_kwh,
                    consumption.normal_kwh,
                    consumption.peak_kwh,
                    consumption.super_peak_kwh,
                )
            )
            > consumption.total_kwh * 1.01
        ):
            warnings.append(
                "Time-of-use energy exceeds total "
                "energy consumption. Check tariff "
                "band aggregation."
            )

        if consumption.confidence < (
            self.config.minimum_confidence
        ):
            warnings.append(
                "Energy cost estimate has low "
                "telemetry confidence."
            )

        return EnergyCostResult(
            calculation_id=new_id(),

            asset_id=asset_id,
            site_id=site_id,

            source=consumption.source,

            total_kwh=round(
                consumption.total_kwh,
                4,
            ),

            active_energy_cost=round(
                active_cost,
                2,
            ),

            total_energy_cost=round(
                total_cost,
                2,
            ),

            cost_per_kwh=round(
                cost_per_kwh,
                4,
            ),

            cost_per_operating_hour=(
                round(cost_per_hour, 2)
                if cost_per_hour is not None
                else None
            ),

            cost_per_unit=(
                round(cost_per_unit, 4)
                if cost_per_unit is not None
                else None
            ),

            specific_energy_kwh_per_unit=(
                round(specific_energy, 6)
                if specific_energy is not None
                else None
            ),

            peak_demand_kw=round(
                consumption.peak_demand_kw,
                4,
            ),

            power_factor=round(
                consumption.power_factor,
                4,
            ),

            carbon_emissions_kg=round(
                emissions,
                4,
            ),

            carbon_cost=round(
                carbon_cost,
                2,
            ),

            breakdown=breakdown,

            waste=waste,

            estimated_waste_cost=round(
                estimated_waste_cost,
                2,
            ),

            efficiency_score=(
                efficiency_score
            ),

            efficiency_status=(
                efficiency_status
            ),

            economic_risk=risk,

            confidence=round(
                consumption.confidence,
                4,
            ),

            currency=(
                tariff.currency
                or self.config.default_currency
            ),

            warnings=warnings,

            metadata=dict(metadata or {}),
        )

    # =====================================================
    # Energy savings
    # =====================================================

    @staticmethod
    def calculate_savings(
        data: EnergySavingsInput,
    ) -> EnergySavingsResult:

        energy_saved = max(
            0.0,
            data.baseline_kwh
            - data.improved_kwh,
        )

        reduction_percent = (
            safe_divide(
                energy_saved,
                data.baseline_kwh,
            )
            * 100.0
        )

        direct_energy_savings = (
            energy_saved
            * data.energy_rate_per_kwh
        )

        peak_reduction = max(
            0.0,
            data.baseline_peak_kw
            - data.improved_peak_kw,
        )

        demand_savings = (
            peak_reduction
            * data.demand_rate_per_kw
        )

        carbon_reduction = (
            energy_saved
            * data.carbon_intensity_kg_per_kwh
        )

        carbon_cost_savings = (
            carbon_reduction
            * data.carbon_price_per_kg
        )

        gross_per_period = sum(
            (
                direct_energy_savings,
                demand_savings,
                carbon_cost_savings,
            )
        )

        annual_savings = (
            gross_per_period
            * data.annual_operating_periods
        )

        first_year_net = (
            annual_savings
            - data.intervention_cost
        )

        payback = None

        if annual_savings > 0:
            payback = (
                data.intervention_cost
                / annual_savings
            )

        roi = None

        if data.intervention_cost > 0:
            roi = (
                first_year_net
                / data.intervention_cost
                * 100.0
            )

        return EnergySavingsResult(
            baseline_kwh=round(
                data.baseline_kwh,
                4,
            ),

            improved_kwh=round(
                data.improved_kwh,
                4,
            ),

            energy_saved_kwh=round(
                energy_saved,
                4,
            ),

            energy_reduction_percent=round(
                reduction_percent,
                2,
            ),

            direct_energy_savings=round(
                direct_energy_savings,
                2,
            ),

            demand_savings=round(
                demand_savings,
                2,
            ),

            carbon_reduction_kg=round(
                carbon_reduction,
                4,
            ),

            carbon_cost_savings=round(
                carbon_cost_savings,
                2,
            ),

            gross_savings_per_period=round(
                gross_per_period,
                2,
            ),

            annual_savings=round(
                annual_savings,
                2,
            ),

            intervention_cost=round(
                data.intervention_cost,
                2,
            ),

            first_year_net_savings=round(
                first_year_net,
                2,
            ),

            simple_payback_years=(
                round(payback, 3)
                if payback is not None
                else None
            ),

            roi_percent=(
                round(roi, 2)
                if roi is not None
                else None
            ),

            currency=data.currency,
        )

    # =====================================================
    # Energy anomaly cost
    # =====================================================

    @staticmethod
    def anomaly_cost(
        *,
        actual_kwh: float,
        expected_kwh: float,
        energy_rate_per_kwh: float,
    ) -> float:

        excess = max(
            0.0,
            non_negative(actual_kwh)
            - non_negative(expected_kwh),
        )

        return round(
            excess
            * non_negative(
                energy_rate_per_kwh
            ),
            2,
        )

    # =====================================================
    # Idle energy cost
    # =====================================================

    @staticmethod
    def idle_energy_cost(
        *,
        idle_power_kw: float,
        idle_hours: float,
        energy_rate_per_kwh: float,
    ) -> float:

        return round(
            non_negative(idle_power_kw)
            * non_negative(idle_hours)
            * non_negative(
                energy_rate_per_kwh
            ),
            2,
        )

    # =====================================================
    # Annual energy cost
    # =====================================================

    @staticmethod
    def annual_energy_cost(
        *,
        average_kwh_per_period: float,
        periods_per_year: float,
        cost_per_kwh: float,
    ) -> float:

        return round(
            non_negative(
                average_kwh_per_period
            )
            * non_negative(
                periods_per_year
            )
            * non_negative(
                cost_per_kwh
            ),
            2,
        )

    # =====================================================
    # Batch
    # =====================================================

    def calculate_many(
        self,
        *,
        consumptions: Sequence[
            EnergyConsumption
        ],
        tariff: EnergyTariff,
    ) -> list[EnergyCostResult]:

        return [
            self.calculate(
                consumption=item,
                tariff=tariff,
            )
            for item in consumptions
        ]

    # =====================================================
    # Aggregate
    # =====================================================

    @staticmethod
    def aggregate(
        results: Sequence[
            EnergyCostResult
        ],
    ) -> dict[str, Any]:

        if not results:
            return {
                "asset_count": 0,
                "total_kwh": 0.0,
                "total_energy_cost": 0.0,
                "total_carbon_emissions_kg": 0.0,
                "estimated_waste_cost": 0.0,
            }

        total_kwh = sum(
            result.total_kwh
            for result in results
        )

        total_cost = sum(
            result.total_energy_cost
            for result in results
        )

        emissions = sum(
            result.carbon_emissions_kg
            for result in results
        )

        waste_cost = sum(
            result.estimated_waste_cost
            for result in results
        )

        currencies = {
            result.currency
            for result in results
        }

        return {
            "asset_count": len(results),

            "total_kwh": round(
                total_kwh,
                4,
            ),

            "total_energy_cost": round(
                total_cost,
                2,
            ),

            "average_cost_per_kwh": round(
                safe_divide(
                    total_cost,
                    total_kwh,
                ),
                4,
            ),

            "total_carbon_emissions_kg":
                round(emissions, 4),

            "estimated_waste_cost":
                round(waste_cost, 2),

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

energy_cost_calculator = (
    EnergyCostCalculator()
)


# =========================================================
# Agent-facing helpers
# =========================================================

def calculate_energy_cost(
    *,
    consumption: EnergyConsumption,
    tariff: EnergyTariff,
    production: EnergyProductionContext | None = None,
    asset_id: str | None = None,
    site_id: str | None = None,
) -> EnergyCostResult:

    return energy_cost_calculator.calculate(
        consumption=consumption,
        tariff=tariff,
        production=production,
        asset_id=asset_id,
        site_id=site_id,
    )


def calculate_energy_savings(
    data: EnergySavingsInput,
) -> EnergySavingsResult:

    return (
        energy_cost_calculator
        .calculate_savings(data)
    )


def calculate_energy_anomaly_cost(
    *,
    actual_kwh: float,
    expected_kwh: float,
    energy_rate_per_kwh: float,
) -> float:

    return (
        energy_cost_calculator
        .anomaly_cost(
            actual_kwh=actual_kwh,
            expected_kwh=expected_kwh,
            energy_rate_per_kwh=(
                energy_rate_per_kwh
            ),
        )
    )


def calculate_idle_energy_cost(
    *,
    idle_power_kw: float,
    idle_hours: float,
    energy_rate_per_kwh: float,
) -> float:

    return (
        energy_cost_calculator
        .idle_energy_cost(
            idle_power_kw=idle_power_kw,
            idle_hours=idle_hours,
            energy_rate_per_kwh=(
                energy_rate_per_kwh
            ),
        )
    )


# =========================================================
# Exports
# =========================================================

__all__ = [
    "EnergySource",
    "EnergyCostCategory",
    "TariffType",
    "TimeBand",
    "EnergyEfficiencyStatus",
    "EnergyCostRisk",

    "EnergyTariff",
    "EnergyConsumption",
    "EnergyProductionContext",

    "EnergyCostBreakdown",
    "EnergyWaste",
    "EnergyCostResult",

    "EnergySavingsInput",
    "EnergySavingsResult",

    "EnergyCostConfig",
    "EnergyCostCalculator",

    "energy_cost_calculator",

    "calculate_energy_cost",
    "calculate_energy_savings",
    "calculate_energy_anomaly_cost",
    "calculate_idle_energy_cost",
]